# -*- coding: utf-8 -*-
"""MX_CV: pooled training on ASL datasets 1+2+3, folds grouped by slice index.

Implements ``Archive/MX_CV/PREREGISTRATION.md``. Same loop and recipe as
``_cv_train.py`` (total-noise labels, p_raw 0.5, LR 1e-3, batch 8, UNet 32/32,
seed 101) with three differences: the training pool is the training-fold
slices of ALL requested datasets; folds come from CV_R3 by slice index (slice s
of every dataset is held out together); the ladder bottom is the global
smallest sigma_hat over those datasets. Trained to 6000 iterations with a
checkpoint every 500; iteration 2000 is the pre-declared primary checkpoint
(no LR schedule, so it equals a 2000-iteration run).

Arms: A = L1, B = L1 noise-normalized, C = MSE, D = MSE noise-normalized.

    python experiments/_mx_train.py --arm A --fold 0
"""
import sys, os, time, json, argparse, csv
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, DIFF)
os.chdir(DIFF)

from unet import UNet
from n2n import N2N_DSM
from train import DiffusionTrainer
from _r3_train import pack
from _data import load_datasets, cv_folds

ARMS = {"A": ("l1", False), "B": ("l1", True), "C": ("mse", False), "D": ("mse", True)}
CFG = dict(L=40, SIGMA_MAX=0.4, P_RAW=0.5, ITERS=6000, PRIMARY=2000, CKPT_EVERY=500,
           BATCH=8, LR=1e-3, GRAD_CLIP=1.0, NET_BASE=32, NET_EMB=32, SEED=101, K=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--datasets", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--outdir", default="runs/mx_cv")
    ap.add_argument("--iters", type=int, default=None, help="smoke tests only")
    args = ap.parse_args()
    loss, noise_norm = ARMS[args.arm]
    iters = args.iters or CFG["ITERS"]

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RUN = os.path.join(args.outdir, args.arm, f"fold{args.fold}")
    assert not os.path.exists(os.path.join(RUN, "train.csv")), "refusing to overwrite " + RUN
    os.makedirs(os.path.join(RUN, "ckpt"), exist_ok=True)
    log_path = os.path.join(RUN, "train.log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    torch.manual_seed(CFG["SEED"])
    x, meta, sig = load_datasets(args.datasets)
    folds = cv_folds()
    held = sorted(s for s, k in folds.items() if k == args.fold)
    train_idx = sorted(s for s, k in folds.items() if k != args.fold)
    tr = [i for i, (d, s) in enumerate(meta) if s in train_idx]
    ho = [i for i, (d, s) in enumerate(meta) if s in held]
    assert not set(meta[i][1] for i in tr) & set(held)
    sigma_min = float(sig.min())
    sigma_ref = float(torch.exp(torch.log(sig[tr]).mean()))
    log(f"MX arm {args.arm} (loss={loss}, noise_norm={noise_norm}) fold {args.fold}/{CFG['K']} "
        f"datasets={args.datasets} held-out indices={held} | train slices={len(tr)} "
        f"held-out slices={len(ho)} device={DEVICE}")
    log(f"ladder {sigma_min:.4f}..{CFG['SIGMA_MAX']:.2f} (global min sigma_hat); "
        f"sigma_ref {sigma_ref:.4f} (geo-mean of training sigma_hat)")
    qtrain = float(sig[tr].min())
    for i in ho:
        if float(sig[i]) < qtrain:
            log(f"NOTE held-out ds{meta[i][0]} slice {meta[i][1]} sigma {float(sig[i]):.4f} is quieter than "
                f"every training slice of this fold (min {qtrain:.4f}); extrapolated slightly.")

    x_train = pack(x, tr, DEVICE)
    sy_train = sig[tr].to(DEVICE)
    net = UNet(in_ch=2, out_ch=2, base_out=CFG["NET_BASE"], emb_dim=CFG["NET_EMB"]).to(DEVICE)
    n2n = N2N_DSM(L=CFG["L"], sigma_min=sigma_min, sigma_max=CFG["SIGMA_MAX"], backbone=net)
    trainer = DiffusionTrainer(n2n, lr=CFG["LR"], device=DEVICE, grad_clip=CFG["GRAD_CLIP"])
    ckpt_cfg = dict(net_type="N2N_DSM", noise_label="total", objective="multireal", N=x.shape[1],
                    L=CFG["L"], sigma_min=sigma_min, sigma_max=CFG["SIGMA_MAX"],
                    base_out=CFG["NET_BASE"], emb_dim=CFG["NET_EMB"], loss=loss, p_raw=CFG["P_RAW"],
                    fold=args.fold, K=CFG["K"], held_out=held, arm=args.arm, datasets=args.datasets)
    if noise_norm:
        ckpt_cfg.update(noise_norm=True, sigma_ref=sigma_ref)
    with open(os.path.join(RUN, "config.json"), "w") as f:
        json.dump(dict(cfg=CFG, ckpt_cfg=ckpt_cfg, held_out=held, train_slices=train_idx,
                       datasets=args.datasets, train_units=[list(meta[i]) for i in tr],
                       sigma_hat={f"{d}_{s}": float(sig[i]) for i, (d, s) in enumerate(meta)}),
                  f, indent=2)

    xh = x[ho].to(DEVICE).reshape(-1, 2, *x.shape[-2:])
    sh = sig[ho].to(DEVICE).repeat_interleave(x.shape[1])
    ens_h = x[ho].to(DEVICE)
    Th = ((ens_h.sum(1, keepdim=True) - ens_h) / (x.shape[1] - 1)).reshape_as(xh)

    @torch.no_grad()
    def monitor():
        n2n.net.eval()
        x0 = xh + sh.square().view(-1, 1, 1, 1) * n2n.net(xh, n2n.level_coord(sh))
        n2n.net.train()
        return torch.nn.functional.l1_loss(x0, Th).item()

    csv_path = os.path.join(RUN, "train.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["iter", "train_loss_avg50", "heldout_l1_monitor_only", "sec"])
    recent = []
    t0 = time.time()
    for it in range(1, iters + 1):
        poss = torch.randint(0, len(tr), (CFG["BATCH"],), device=DEVICE)
        l_ = trainer.step_multireal(x_train[poss], loss=loss, sigma_y=sy_train[poss], p_raw=CFG["P_RAW"],
                                    noise_norm=noise_norm, sigma_ref=sigma_ref if noise_norm else None)
        recent = (recent + [l_])[-50:]
        if it % 50 == 0:
            hl = monitor()
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([it, float(np.mean(recent)), hl, time.time() - t0])
        if it % CFG["CKPT_EVERY"] == 0 or it == iters:
            log(f"iter {it:>5} train={np.mean(recent):.5f} heldout_l1(monitor)={monitor():.5f}")
            torch.save({"state_dict": n2n.net.state_dict(), "cfg": ckpt_cfg, "train_iters": it},
                       os.path.join(RUN, "ckpt", f"{loss}_{it:06d}.pkl"))
    log(f"DONE arm {args.arm} fold {args.fold}: {iters} iterations in {(time.time() - t0) / 60:.1f} min "
        f"(primary checkpoint: iteration {CFG['PRIMARY']}, fixed in advance)")


if __name__ == "__main__":
    main()
