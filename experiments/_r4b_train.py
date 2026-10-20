# -*- coding: utf-8 -*-
"""R4b/R4c: subset-averaging N2N with BALANCED noise-level coverage.

R4 (``_r4_train.py``) drew the subset size m uniformly per sample, so the
quietest averaged inputs (sigma_hat/3 of the quietest slices) came from only a
handful of training slices and were rarely trained -- exactly where R4 then
under-performed at large a. Here each batch element is a (slice, m) pair drawn
with weights that make the TOTAL input noise sigma_hat_s/sqrt(m) roughly
uniform in log sigma across all (slice, m) pairs (inverse-histogram weights over
NBINS log-sigma bins). All N draws of one slice share its m.

  --aug   also apply a random dihedral transform (rot90 x flip) per slice: the
          noise is i.i.d. per pixel, so these are exact symmetries of the noise
          model and multiply the effective count of the scarce quiet slices.

Validation, checkpoints and selection rule are identical to R4 (geometric-mean
val n_eff over a in {1, 5, 9}, cyclic subsets of the val slices).

    python experiments/_r4b_train.py --tag bal     --iters 3000
    python experiments/_r4b_train.py --tag bal_aug --iters 3000 --aug
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
from _r3_train import load_data, sigma_hat, pack, VAL_SLICES, TEST_SLICES

CFG = dict(
    L=40, SIGMA_MAX=0.4, P_RAW=0.5, M_MAX=9, NBINS=12,
    BATCH=8, LR=1e-3, GRAD_CLIP=1.0,
    NET_BASE=32, NET_EMB=32,
)
VAL_A = (1, 5, 9)


def dihedral(x, k, flip):
    """x: (C,H,W); rotate by k*90 degrees, then optionally flip left-right."""
    x = torch.rot90(x, k, dims=(1, 2))
    return torch.flip(x, dims=(2,)) if flip else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--loss", default="l1", choices=["l1", "mse", "huber", "mag_l1"])
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--outdir", default="runs/r4b_subset")
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RUN = os.path.join(args.outdir, args.tag, args.loss)
    assert not os.path.exists(os.path.join(RUN, "train_val.csv")), "refusing to overwrite " + RUN
    os.makedirs(os.path.join(RUN, "ckpt"), exist_ok=True)
    log_path = os.path.join(RUN, "train.log")
    csv_path = os.path.join(RUN, "train_val.csv")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    torch.manual_seed(args.seed)
    rng_np = np.random.default_rng(args.seed)

    x = load_data()
    Ns, N = x.shape[0], x.shape[1]
    sig = sigma_hat(x)
    train_slices = [s for s in range(Ns) if s not in VAL_SLICES + TEST_SLICES]
    sigma_min = float(sig[train_slices].min()) / np.sqrt(CFG["M_MAX"])
    for s in VAL_SLICES + TEST_SLICES:
        assert sigma_min <= sig[s] / np.sqrt(N) and sig[s] < CFG["SIGMA_MAX"], s

    x_train = pack(x, train_slices, DEVICE)
    sy_train = sig[train_slices].to(DEVICE)

    # ---- balanced (slice, m) sampler ---------------------------------------------
    pairs = [(i, m) for i in range(len(train_slices)) for m in range(1, CFG["M_MAX"] + 1)]
    ls = np.array([np.log(float(sig[train_slices[i]]) / np.sqrt(m)) for i, m in pairs])
    edges = np.linspace(ls.min(), ls.max() + 1e-9, CFG["NBINS"] + 1)
    bins = np.digitize(ls, edges) - 1
    counts = np.bincount(bins, minlength=CFG["NBINS"])
    w = 1.0 / counts[bins]
    w /= w.sum()
    p_unif = np.bincount(bins, minlength=CFG["NBINS"]) / len(pairs)       # R4: uniform m
    p_bal = np.bincount(bins, weights=w, minlength=CFG["NBINS"])

    log(f"R4b tag={args.tag} aug={args.aug} loss={args.loss} device={DEVICE} "
        f"train={len(train_slices)} val={VAL_SLICES} test={TEST_SLICES}")
    log(f"ladder L={CFG['L']} sigma {sigma_min:.4f}..{CFG['SIGMA_MAX']:.4f} p_raw={CFG['P_RAW']}")
    log("log-sigma bin  lower-edge sigma  share(R4 uniform m)  share(balanced)")
    for b in range(CFG["NBINS"]):
        log(f"  bin {b:2d}  {np.exp(edges[b]):.4f}  {p_unif[b]:.3f}  {p_bal[b]:.3f}")

    net = UNet(in_ch=2, out_ch=2, base_out=CFG["NET_BASE"],
               emb_dim=CFG["NET_EMB"]).to(DEVICE)
    n2n = N2N_DSM(L=CFG["L"], sigma_min=sigma_min, sigma_max=CFG["SIGMA_MAX"],
                  backbone=net)
    trainer = DiffusionTrainer(n2n, lr=CFG["LR"], device=DEVICE,
                               grad_clip=CFG["GRAD_CLIP"])

    ckpt_cfg = dict(net_type="N2N_DSM", noise_label="total", objective="subset_balanced",
                    aug=bool(args.aug), N=N, L=CFG["L"], sigma_min=sigma_min,
                    sigma_max=CFG["SIGMA_MAX"], base_out=CFG["NET_BASE"],
                    emb_dim=CFG["NET_EMB"], loss=args.loss, p_raw=CFG["P_RAW"],
                    m_max=CFG["M_MAX"], nbins=CFG["NBINS"])
    with open(os.path.join(RUN, "config.json"), "w") as f:
        json.dump(dict(cfg=CFG, ckpt_cfg=ckpt_cfg, iters=args.iters, seed=args.seed,
                       aug=bool(args.aug), val_slices=VAL_SLICES, test_slices=TEST_SLICES,
                       train_slices=train_slices,
                       sigma_hat={int(s): float(sig[s]) for s in range(Ns)},
                       share_uniform_m=p_unif.tolist(), share_balanced=p_bal.tolist()),
                  f, indent=2)

    # ---- deterministic validation (identical to R4) ----------------------------------
    ens_v = x[VAL_SLICES].to(DEVICE)
    s2v = []
    for k in range(len(VAL_SLICES)):
        e = ens_v[k].double()
        s2v.append(float(((e - e.mean(0)) ** 2).sum(0).mean() / (N - 1)))
    VAL = {}
    for a in VAL_A:
        X, T, S, V = [], [], [], []
        for k in range(len(VAL_SLICES)):
            for c in range(N):
                A = [(c + j) % N for j in range(a)]
                B = [j for j in range(N) if j not in A]
                X.append(ens_v[k, A].mean(0)); T.append(ens_v[k, B].mean(0))
                S.append(float(sig[VAL_SLICES[k]]) / np.sqrt(a)); V.append(s2v[k])
        VAL[a] = (torch.stack(X), torch.stack(T),
                  torch.tensor(S, device=DEVICE, dtype=torch.float32),
                  torch.tensor(V, device=DEVICE, dtype=torch.float64))

    @torch.no_grad()
    def validate():
        n2n.net.eval()
        res = {}
        for a, (X, T, S, V) in VAL.items():
            x0 = X + S.square().view(-1, 1, 1, 1) * n2n.net(X, n2n.level_coord(S))
            res[f"val_l1_a{a}"] = torch.nn.functional.l1_loss(x0, T).item()
            mse_ref = ((x0 - T).double() ** 2).mean(dim=(1, 2, 3))
            mse = (mse_ref - V / (N - a)).mean().item()
            res[f"val_neff_a{a}"] = float(V.mean().item() / mse) if mse > 0 else float("inf")
        n2n.net.train()
        res["val_neff_gmean"] = float(np.exp(np.mean(
            [np.log(res[f"val_neff_a{a}"]) for a in VAL_A])))
        return res

    fields = (["loss", "iter", "train_loss", "train_loss_avg50"] +
              [f"val_l1_a{a}" for a in VAL_A] + [f"val_neff_a{a}" for a in VAL_A] +
              ["val_neff_gmean", "sec"])
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    def save(tag, it, loss):
        p = os.path.join(RUN, "ckpt", f"{args.loss}_{tag}.pkl")
        torch.save({"state_dict": n2n.net.state_dict(), "cfg": ckpt_cfg,
                    "train_iters": it, "loss_final": loss}, p)

    def fmt(r):
        return " ".join(f"neff_a{a}={r[f'val_neff_a{a}']:.2f}" for a in VAL_A) + \
            f" gmean={r['val_neff_gmean']:.2f}"

    r = validate()
    log("init " + fmt(r))
    recent = []
    t0 = time.time()
    for it in range(1, args.iters + 1):
        pick = rng_np.choice(len(pairs), size=CFG["BATCH"], p=w)
        idx = torch.tensor([pairs[j][0] for j in pick], device=DEVICE)
        mm = torch.tensor([pairs[j][1] for j in pick], device=DEVICE)
        xb = x_train[idx]
        if args.aug:
            xb = torch.stack([dihedral(xb[i], int(rng_np.integers(4)), bool(rng_np.integers(2)))
                              for i in range(len(xb))])
        loss = trainer.step_subset(xb, sy_train[idx], loss=args.loss,
                                   p_raw=CFG["P_RAW"], m=mm)
        recent = (recent + [loss])[-50:]
        if it % args.val_every == 0 or it == 1:
            r = validate()
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(dict(
                    loss=args.loss, iter=it, train_loss=loss,
                    train_loss_avg50=float(np.mean(recent)), sec=time.time() - t0, **r))
            if it % (args.val_every * 4) == 0:
                log(f"iter {it:>5} train={np.mean(recent):.5f} " + fmt(r))
        if it % args.ckpt_every == 0:
            save(f"{it:06d}", it, loss)

    save("final", args.iters, loss)
    log(f"DONE {args.tag}: iters={args.iters} " + fmt(r) +
        f" elapsed={(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
