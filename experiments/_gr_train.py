# -*- coding: utf-8 -*-
"""GROUND training (``Archive/GROUND/PREREGISTRATION.md``): KR and SM runs.

One loop for both objectives, CV_R3 folds by slice index, CV_R3 recipe
(total-noise labels, p_raw 0.5, L1, LR 1e-3, batch 8, UNet 32/32):

  --mode per   per-repetition objective (``loss_multireal``); ``--target-k`` k-subset LOO
               target (default: all N-1). With defaults (seed 101, dataset 1, no aug,
               fraction 1, ladder divisor 1) this is exactly ``_cv_train.py``; with
               ``--datasets 1 2 3`` exactly ``_mx_train.py --arm A``.
  --mode sub   subset-mean objective (``loss_subset``) with (slice, m) pairs balanced in
               log total noise, as R4b/R4c (``_r4b_train.py``).

  --seed S       torch/numpy seed; the k-subset target generator is seeded 20261027 + (S - 101)
  --train-frac   keep a nested fraction of each fold's training slice indices (fixed per fold)
  --aug          random dihedral transform per slice (exact symmetry of i.i.d. pixel noise)
  --ladder-div   ladder bottom = global smallest sigma_hat / ladder-div (SUB needs 3: sigma_hat/sqrt(9))

Checkpoints every 500 iterations (``l1_<iter>.pkl``), config.json in the format
``_mx_eval.py`` / ``_gr_eval.py`` read. Refuses to overwrite.

    python experiments/_gr_train.py --mode per --fold 0 --target-k 1 --seed 102 --iters 4000 --outdir runs/gr/kr/k1_s102
"""
import sys, os, time, json, argparse, csv
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, DIFF)
os.chdir(DIFF)
sys.path.insert(0, os.path.join(DIFF, "experiments"))

from unet import UNet
from n2n import N2N_DSM
from train import DiffusionTrainer
from _r3_train import pack
from _data import load_datasets, cv_folds

CFG = dict(L=40, SIGMA_MAX=0.4, P_RAW=0.5, CKPT_EVERY=500, BATCH=8, LR=1e-3, GRAD_CLIP=1.0,
           NET_BASE=32, NET_EMB=32, K=5, M_MAX=9, NBINS=12)
LOSS = "l1"


def dihedral(x, k, flip):
    x = torch.rot90(x, k, dims=(1, 2))
    return torch.flip(x, dims=(2,)) if flip else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["per", "sub"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--datasets", type=int, nargs="+", default=[1])
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--target-k", type=int, default=None)
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--ladder-div", type=float, default=1.0)
    args = ap.parse_args()
    assert args.mode == "per" or args.target_k is None, "target_k applies to --mode per"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RUN = os.path.join(args.outdir, f"fold{args.fold}")
    assert not os.path.exists(os.path.join(RUN, "train.csv")), "refusing to overwrite " + RUN
    os.makedirs(os.path.join(RUN, "ckpt"), exist_ok=True)
    log_path = os.path.join(RUN, "train.log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    torch.manual_seed(args.seed)
    rng_np = np.random.default_rng(args.seed)
    x, meta, sig = load_datasets(args.datasets)
    N = x.shape[1]
    folds = cv_folds()
    held = sorted(s for s, k in folds.items() if k == args.fold)
    train_idx = sorted(s for s, k in folds.items() if k != args.fold)
    if args.train_frac < 1.0:                     # nested subsets: a fixed permutation per fold
        perm = np.random.default_rng(20261028 + args.fold).permutation(train_idx)
        train_idx = sorted(int(s) for s in perm[:int(round(args.train_frac * len(train_idx)))])
    tr = [i for i, (d, s) in enumerate(meta) if s in train_idx]
    ho = [i for i, (d, s) in enumerate(meta) if s in held]
    assert not set(meta[i][1] for i in tr) & set(held)
    sigma_min = float(sig.min()) / args.ladder_div
    log(f"GROUND mode={args.mode} fold {args.fold}/{CFG['K']} datasets={args.datasets} seed={args.seed} "
        f"target_k={args.target_k} frac={args.train_frac} aug={args.aug} | train slices={len(tr)} "
        f"(indices {train_idx}) held-out indices={held} device={DEVICE}")
    log(f"ladder {sigma_min:.4f}..{CFG['SIGMA_MAX']:.2f} (global min sigma_hat / {args.ladder_div:g})")

    x_train = pack(x, tr, DEVICE)
    sy_train = sig[tr].to(DEVICE)
    net = UNet(in_ch=2, out_ch=2, base_out=CFG["NET_BASE"], emb_dim=CFG["NET_EMB"]).to(DEVICE)
    n2n = N2N_DSM(L=CFG["L"], sigma_min=sigma_min, sigma_max=CFG["SIGMA_MAX"], backbone=net)
    trainer = DiffusionTrainer(n2n, lr=CFG["LR"], device=DEVICE, grad_clip=CFG["GRAD_CLIP"])
    if args.target_k is not None:                 # seed 101 -> 20261027, the TQ generator
        n2n._tk_gen = torch.Generator(device=x_train.device)
        n2n._tk_gen.manual_seed(20261027 + args.seed - 101)

    if args.mode == "sub":                        # balanced (slice, m) sampler, as _r4b_train.py
        pairs = [(i, m) for i in range(len(tr)) for m in range(1, CFG["M_MAX"] + 1)]
        ls = np.array([np.log(float(sy_train[i]) / np.sqrt(m)) for i, m in pairs])
        edges = np.linspace(ls.min(), ls.max() + 1e-9, CFG["NBINS"] + 1)
        bins = np.digitize(ls, edges) - 1
        w = 1.0 / np.bincount(bins, minlength=CFG["NBINS"])[bins]
        w /= w.sum()
        assert sigma_min <= float(sig.min()) / np.sqrt(CFG["M_MAX"]) + 1e-12, "ladder must reach sigma/3"

    ckpt_cfg = dict(net_type="N2N_DSM", noise_label="total",
                    objective="multireal" if args.mode == "per" else "subset_balanced",
                    N=N, L=CFG["L"], sigma_min=sigma_min, sigma_max=CFG["SIGMA_MAX"],
                    base_out=CFG["NET_BASE"], emb_dim=CFG["NET_EMB"], loss=LOSS, p_raw=CFG["P_RAW"],
                    fold=args.fold, K=CFG["K"], held_out=held, datasets=args.datasets, mode=args.mode,
                    seed=args.seed, target_k=args.target_k, train_frac=args.train_frac, aug=args.aug,
                    ladder_div=args.ladder_div)
    with open(os.path.join(RUN, "config.json"), "w") as f:
        json.dump(dict(cfg=CFG, ckpt_cfg=ckpt_cfg, held_out=held, train_slices=train_idx, iters=args.iters,
                       datasets=args.datasets, train_units=[list(meta[i]) for i in tr],
                       sigma_hat={f"{d}_{s}": float(sig[i]) for i, (d, s) in enumerate(meta)}), f, indent=2)

    xh = x[ho].to(DEVICE).reshape(-1, 2, *x.shape[-2:])
    sh = sig[ho].to(DEVICE).repeat_interleave(N)
    ens_h = x[ho].to(DEVICE)
    Th = ((ens_h.sum(1, keepdim=True) - ens_h) / (N - 1)).reshape_as(xh)

    @torch.no_grad()
    def monitor():                                # held-out L1 at a = 1, logged only
        n2n.net.eval()
        x0 = xh + sh.square().view(-1, 1, 1, 1) * n2n.net(xh, n2n.level_coord(sh))
        n2n.net.train()
        return torch.nn.functional.l1_loss(x0, Th).item()

    csv_path = os.path.join(RUN, "train.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["iter", "train_loss_avg50", "heldout_l1_monitor_only", "sec"])
    recent = []
    t0 = time.time()
    for it in range(1, args.iters + 1):
        if args.mode == "per":
            poss = torch.randint(0, len(tr), (CFG["BATCH"],), device=DEVICE)
            mm = None
        else:
            pick = rng_np.choice(len(pairs), size=CFG["BATCH"], p=w)
            poss = torch.tensor([pairs[j][0] for j in pick], device=DEVICE)
            mm = torch.tensor([pairs[j][1] for j in pick], device=DEVICE)
        xb = x_train[poss]
        if args.aug:
            xb = torch.stack([dihedral(xb[i], int(rng_np.integers(4)), bool(rng_np.integers(2)))
                              for i in range(len(xb))])
        if args.mode == "per":
            l_ = trainer.step_multireal(xb, loss=LOSS, sigma_y=sy_train[poss], p_raw=CFG["P_RAW"],
                                        target_k=args.target_k)
        else:
            l_ = trainer.step_subset(xb, sy_train[poss], loss=LOSS, p_raw=CFG["P_RAW"], m=mm)
        recent = (recent + [l_])[-50:]
        if it % 50 == 0:
            hl = monitor()
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([it, float(np.mean(recent)), hl, time.time() - t0])
        if it % CFG["CKPT_EVERY"] == 0 or it == args.iters:
            log(f"iter {it:>5} train={np.mean(recent):.5f} heldout_l1(monitor)={monitor():.5f}")
            torch.save({"state_dict": n2n.net.state_dict(), "cfg": ckpt_cfg, "train_iters": it},
                       os.path.join(RUN, "ckpt", f"{LOSS}_{it:06d}.pkl"))
    log(f"DONE fold {args.fold}: {args.iters} iterations in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
