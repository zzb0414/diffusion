# -*- coding: utf-8 -*-
"""Self-supervised single-image DL baselines (Archive/BASE/PREREGISTRATION_DL.md), one trainer.

  ne2ne   Neighbor2Neighbor (Huang et al., CVPR 2021): random neighbour sub-sampler g1, g2 on 2x2 cells,
          L = |f(g1 y) - g2 y|^2 + gamma |f(g1 y) - g2 y - (g1 f(y) - g2 f(y))|^2, gamma = 2, f(y) without gradient.
  r2r     Recorrupted-to-Recorrupted (Pang et al., CVPR 2021): input y + alpha z, target y - z / alpha,
          z ~ N(0, sigma_hat^2 I), alpha = 0.5; inference averages f(y + alpha z) over 8 draws (in evaluate.py).
  n2self  Noise2Self (Batson & Royer, ICML 2019): 4x4 grid masking (one of 16 phases per step), masked pixels
          replaced by the mean of their 3x3 neighbours, MSE on masked pixels only; inference f(y).

Same U-Net (no sigma input: constant level coordinate, direct output), Adam 1e-3, batch 8 slices x 10 repetitions
(each repetition an independent noisy image), grad clip 1, 2000 iterations, CV_R3 folds, dataset-1 training.
Checkpoints load with ``experiments/_r3_eval.load`` (param "direct", blind) and decode with ``denoise_to``.

    python baselines/dl_selfsup.py --method ne2ne --fold 0 --seed 101 --outdir runs/base/ne2ne/s101
"""
import os, time, json, argparse, csv
import numpy as np
import torch
import torch.nn.functional as F
import _common

from unet import UNet                                        # noqa: E402
from n2n import N2N_DSM                                      # noqa: E402

CFG = dict(L=40, SIGMA_MAX=0.4, ITERS=2000, BATCH=8, LR=1e-3, GRAD_CLIP=1.0, NET_BASE=32, NET_EMB=32,
           CKPT_EVERY=500, GAMMA=2.0, ALPHA=0.5, GRID=4)
ADJ = {0: (1, 2), 1: (0, 3), 2: (0, 3), 3: (1, 2)}          # 2x2 cell positions; horizontal / vertical neighbours


def f_net(net, y):
    """direct output, no sigma input; pads to a multiple of 16 (U-Net) and crops back."""
    H, W = y.shape[-2:]
    ph, pw = (-H) % 16, (-W) % 16
    yp = F.pad(y, (0, pw, 0, ph), mode="reflect") if (ph or pw) else y
    out = net(yp, torch.zeros(y.shape[0], device=y.device))
    return out[..., :H, :W]


def subsample(y, i1, i2):
    """Neighbor2Neighbor sub-images: per 2x2 cell, the pixels at positions i1 and i2 (K, H/2, W/2 index maps)."""
    K, C, H, W = y.shape
    cells = y.view(K, C, H // 2, 2, W // 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(K, C, H // 2, W // 2, 4)
    g1 = torch.gather(cells, -1, i1[:, None, :, :, None].expand(K, C, H // 2, W // 2, 1))[..., 0]
    g2 = torch.gather(cells, -1, i2[:, None, :, :, None].expand(K, C, H // 2, W // 2, 1))[..., 0]
    return g1, g2


def loss_fn(method, net, y, sig):
    K = y.shape[0]
    if method == "ne2ne":
        H, W = y.shape[-2:]
        i1 = torch.randint(0, 4, (K, H // 2, W // 2), device=y.device)
        pick = torch.randint(0, 2, i1.shape, device=y.device)
        i2 = torch.where(pick == 0, torch.tensor([ADJ[k][0] for k in range(4)], device=y.device)[i1],
                         torch.tensor([ADJ[k][1] for k in range(4)], device=y.device)[i1])
        g1, g2 = subsample(y, i1, i2)
        with torch.no_grad():
            fy = f_net(net, y)
        f1, f2 = subsample(fy, i1, i2)
        out = f_net(net, g1)
        return ((out - g2) ** 2).mean() + CFG["GAMMA"] * ((out - g2 - (f1 - f2)) ** 2).mean()
    if method == "r2r":
        z = torch.randn_like(y) * sig.view(-1, 1, 1, 1)
        return ((f_net(net, y + CFG["ALPHA"] * z) - (y - z / CFG["ALPHA"])) ** 2).mean()
    if method == "n2self":
        G = CFG["GRID"]
        ph = int(torch.randint(0, G * G, (1,)))
        mask = torch.zeros(y.shape[-2:], device=y.device, dtype=torch.bool)
        mask[ph // G::G, ph % G::G] = True
        kern = torch.ones(1, 1, 3, 3, device=y.device); kern[0, 0, 1, 1] = 0; kern /= 8
        nb = F.conv2d(F.pad(y.reshape(-1, 1, *y.shape[-2:]), (1, 1, 1, 1), mode="reflect"), kern).view_as(y)
        out = f_net(net, torch.where(mask, nb, y))
        return ((out - y) ** 2)[..., mask].mean()
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["ne2ne", "r2r", "n2self"])
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
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    from _data import load_datasets, cv_folds
    x, meta, sig = load_datasets([1])
    N = x.shape[1]
    folds = cv_folds()
    held = sorted(s for s, k in folds.items() if k == args.fold)
    train_idx = sorted(s for s, k in folds.items() if k != args.fold)
    tr = [i for i, (d, s) in enumerate(meta) if s in train_idx]
    x_train, sy = x[tr].to(DEVICE), sig[tr].to(DEVICE)
    net = UNet(in_ch=2, out_ch=2, base_out=CFG["NET_BASE"], emb_dim=CFG["NET_EMB"]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=CFG["LR"])
    ckpt_cfg = dict(net_type="N2N_DSM", noise_label="total", objective=args.method, N=N, L=CFG["L"],
                    sigma_min=float(sig.min()), sigma_max=CFG["SIGMA_MAX"], base_out=CFG["NET_BASE"],
                    emb_dim=CFG["NET_EMB"], loss=args.method, param="direct", blind=True, alpha=CFG["ALPHA"],
                    fold=args.fold, held_out=held, seed=args.seed, datasets=[1])
    json.dump(dict(cfg=CFG, ckpt_cfg=ckpt_cfg, held_out=held, train_slices=train_idx, seed=args.seed),
              open(os.path.join(RUN, "config.json"), "w"), indent=2)
    log = open(os.path.join(RUN, "train.log"), "a", encoding="utf-8")
    with open(os.path.join(RUN, "train.csv"), "w", newline="") as f:
        csv.writer(f).writerow(["iter", "loss_avg50", "sec"])
    recent, t0 = [], time.time()
    for it in range(1, args.iters + 1):
        pick = torch.randint(0, len(tr), (CFG["BATCH"],), device=DEVICE)
        y = x_train[pick].reshape(-1, 2, *x.shape[-2:])
        s = sy[pick].repeat_interleave(N)
        loss = loss_fn(args.method, net, y, s)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CFG["GRAD_CLIP"]); opt.step()
        recent = (recent + [loss.item()])[-50:]
        if it % 50 == 0:
            for attempt in range(5):
                try:
                    with open(os.path.join(RUN, "train.csv"), "a", newline="") as f:
                        csv.writer(f).writerow([it, float(np.mean(recent)), time.time() - t0])
                    break
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(2.0)
        if it % CFG["CKPT_EVERY"] == 0 or it == args.iters:
            line = f"[{time.strftime('%H:%M:%S')}] iter {it} loss={np.mean(recent):.6f}"
            print(line, flush=True); log.write(line + "\n"); log.flush()
            torch.save({"state_dict": net.state_dict(), "cfg": ckpt_cfg, "train_iters": it},
                       os.path.join(RUN, "ckpt", f"{args.method}_{it:06d}.pkl"))
    log.write(f"DONE fold {args.fold}: {args.iters} iterations in {(time.time() - t0) / 60:.1f} min\n"); log.close()


if __name__ == "__main__":
    main()
