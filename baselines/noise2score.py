# -*- coding: utf-8 -*-
"""Noise2Score baseline (Kim & Ye, NeurIPS 2021), re-implemented with this repository's U-Net.

Noise2Score needs no pairs: it learns the score of the NOISY data distribution, s(y) ~ grad log p(y), with the
amortized residual denoising auto-encoder (AR-DAE) objective

    L = E || u + delta_t * s(y + delta_t * u, sigma_hat) ||^2,   u ~ N(0, I),

and decodes with Tweedie's formula for Gaussian noise, x_hat = y + sigma_hat^2 s(y). Every repetition is treated
as an independent noisy image.

Version 2 (2026-10-30). The perturbed input y + delta u carries TOTAL noise sigma_tot = sqrt(sigma_hat^2 + delta^2)
around the clean image, and the AR-DAE optimum is exactly the score of the noisy data smoothed by delta, i.e. the
score at total noise sigma_tot. The network is therefore conditioned on sigma_tot (``N2N_DSM.level_coord``), with
delta drawn log-uniformly in [DELTA_LO, DELTA_HI] x sigma_hat, so one model learns the whole family and is evaluated
at delta -> 0 (sigma_tot = sigma_hat), where the score of p(y) is what Tweedie needs. Version 1 (delta annealed
toward 0, net conditioned on sigma_hat only) failed its sanity check: the learned score was noise-dominated
(final loss 0.98 vs 1.0 for no learning) and its estimates were ~50x worse than the noisy input. A checkpoint loads
with ``_r3_eval.load`` and decodes with ``N2N_DSM.denoise_to``. Same folds, U-Net, optimizer, batch and iterations
as ours.

    python baselines/noise2score.py --fold 0 --seed 101 --outdir runs/base/n2s/s101
"""
import os, sys, time, json, argparse, csv, math
import numpy as np
import torch
import _common

from unet import UNet                                        # noqa: E402
from n2n import N2N_DSM                                      # noqa: E402
from _r3_train import pack                                   # noqa: E402

CFG = dict(L=40, SIGMA_MAX=0.4, ITERS=2000, BATCH=8, LR=1e-3, GRAD_CLIP=1.0, NET_BASE=32, NET_EMB=32,
           DELTA_HI=2.0, DELTA_LO=0.05, CKPT_EVERY=500)        # delta = sigma_hat * U_log[DELTA_LO, DELTA_HI]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--iters", type=int, default=CFG["ITERS"])
    args = ap.parse_args()
    os.chdir(_common.REPO)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RUN = os.path.join(args.outdir, f"fold{args.fold}")
    assert not os.path.exists(os.path.join(RUN, "train.csv")), "refusing to overwrite " + RUN
    os.makedirs(os.path.join(RUN, "ckpt"), exist_ok=True)
    torch.manual_seed(args.seed)
    from _data import load_datasets, cv_folds
    x, meta, sig = load_datasets([1])
    N = x.shape[1]
    folds = cv_folds()
    held = sorted(s for s, k in folds.items() if k == args.fold)
    train_idx = sorted(s for s, k in folds.items() if k != args.fold)
    tr = [i for i, (d, s) in enumerate(meta) if s in train_idx]
    sigma_min = float(sig.min())
    x_train = x[tr].to(DEVICE)                                    # (S, N, 2, H, W)
    sy = sig[tr].to(DEVICE)
    net = UNet(in_ch=2, out_ch=2, base_out=CFG["NET_BASE"], emb_dim=CFG["NET_EMB"]).to(DEVICE)
    m = N2N_DSM(L=CFG["L"], sigma_min=sigma_min, sigma_max=CFG["SIGMA_MAX"], backbone=net)
    opt = torch.optim.Adam(net.parameters(), lr=CFG["LR"])
    ckpt_cfg = dict(net_type="N2N_DSM", noise_label="total", objective="noise2score_ardae", N=N, L=CFG["L"],
                    sigma_min=sigma_min, sigma_max=CFG["SIGMA_MAX"], base_out=CFG["NET_BASE"], emb_dim=CFG["NET_EMB"],
                    loss="n2s", version=2, fold=args.fold, held_out=held, seed=args.seed, datasets=[1])
    json.dump(dict(cfg=CFG, ckpt_cfg=ckpt_cfg, held_out=held, train_slices=train_idx, seed=args.seed),
              open(os.path.join(RUN, "config.json"), "w"), indent=2)
    log = open(os.path.join(RUN, "train.log"), "a", encoding="utf-8")
    with open(os.path.join(RUN, "train.csv"), "w", newline="") as f:
        csv.writer(f).writerow(["iter", "loss_avg50", "delta_rel", "sec"])
    recent, t0 = [], time.time()
    for it in range(1, args.iters + 1):
        pick = torch.randint(0, len(tr), (CFG["BATCH"],), device=DEVICE)
        y = x_train[pick].reshape(-1, 2, *x.shape[-2:])                       # all N repetitions, independent
        s = sy[pick].repeat_interleave(N)
        rel = torch.exp(math.log(CFG["DELTA_LO"]) + torch.rand_like(s) * math.log(CFG["DELTA_HI"] / CFG["DELTA_LO"]))
        d = rel * s
        u = torch.randn_like(y)
        score = net(y + d.view(-1, 1, 1, 1) * u, m.level_coord(torch.sqrt(s ** 2 + d ** 2)))
        loss = ((u + d.view(-1, 1, 1, 1) * score) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CFG["GRAD_CLIP"]); opt.step()
        recent = (recent + [loss.item()])[-50:]
        if it % 50 == 0:
            with open(os.path.join(RUN, "train.csv"), "a", newline="") as f:
                csv.writer(f).writerow([it, float(np.mean(recent)), float(rel.mean()), time.time() - t0])
        if it % CFG["CKPT_EVERY"] == 0 or it == args.iters:
            line = f"[{time.strftime('%H:%M:%S')}] iter {it} loss={np.mean(recent):.5f}"
            print(line, flush=True); log.write(line + "\n"); log.flush()
            torch.save({"state_dict": net.state_dict(), "cfg": ckpt_cfg, "train_iters": it},
                       os.path.join(RUN, "ckpt", f"n2s_{it:06d}.pkl"))
    log.write(f"DONE fold {args.fold}: {args.iters} iterations in {(time.time() - t0) / 60:.1f} min\n"); log.close()


if __name__ == "__main__":
    main()
