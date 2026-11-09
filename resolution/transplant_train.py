# -*- coding: utf-8 -*-
"""GAP B training (Archive/GAP/PREREGISTRATION_B.md): the MEAN-SNR recipe (train.py: U-Net base 32 / emb 32, Adam 1e-3,
grad clip 1, 3000 iterations, MSE, batch 8, 64 x 64 crops, 62 training volumes) at matrix 80 with the copies of
transplant.py. Arms A (PER), SUB, SUBM as models.ARMS; label SIG.

    python resolution/transplant_train.py --arm SUB --noise T --seed 101 --outdir runs/gap/b/SUB_T_s101
"""
import os, time, json, argparse, csv, importlib.util
import numpy as np
import torch
import data as D
import models as Mo
import transplant as TP
from unet import UNet
from n2n import N2N_DSM
from models import fft2c, ifft2c
_sp = importlib.util.spec_from_file_location("res_train", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"))
RT = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(RT)       # resolution/train.py (the repo root has its own)
CFG, stack = RT.CFG, RT.stack

ARMS = ("A", "SUB", "SUBM")


def _noisy(c, split, i, noise, whiten=False):
    y = c[None] + TP.fields(split, i, noise, clean=c)
    return (y / TP.wmap(y)).astype(np.complex64) if whiten else y


def build(arm, dev):
    spec = Mo.ARMS[arm]
    net = UNet(in_ch=2, out_ch=2, base_out=Mo.NET_BASE, emb_dim=Mo.NET_EMB).to(dev)
    m = N2N_DSM(L=Mo.L, sigma_min=TP.SIG / (3.0 if spec.get("sub") else 1.0), sigma_max=Mo.LADDER_TOP * TP.SIG,
                backbone=net, ch=2, param="score", blind=False)
    return net, m


def load(path, dev):
    blob = torch.load(path, map_location=dev, weights_only=False)
    net, m = build(blob["cfg"]["arm"], dev)
    net.load_state_dict(blob["state_dict"]); net.eval()
    return m, blob["cfg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS); ap.add_argument("--noise", required=True, choices=TP.KINDS)
    ap.add_argument("--outdir", required=True); ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--iters", type=int, default=CFG["ITERS"]); ap.add_argument("--n-vol", type=int, default=62)
    ap.add_argument("--crop", type=int, default=CFG["CROP"], help="GAP C1: crop size")
    ap.add_argument("--batch", type=int, default=CFG["BATCH"], help="GAP C1: batch size (crop^2 x batch kept constant)")
    ap.add_argument("--grid", type=int, default=TP.M, choices=(TP.M, 160), help="GAP C2: 160 = matrix-80 content on a 160 grid")
    ap.add_argument("--content", default="F", choices=TP.CONTENTS, help="GAP D/E: D = ASL-SNR-spectrum content, E = D random phase")
    ap.add_argument("--whiten", action="store_true", help="GAP-M: divide each image's copies by their own noise map (TP.wmap)")
    ap.add_argument("--one-slice", action="store_true", help="GAP-N: one slice per volume, position v mod NSL (original indices kept)")
    args = ap.parse_args()
    os.chdir(D.REPO)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert not os.path.exists(os.path.join(args.outdir, "train.csv")), "refusing to overwrite " + args.outdir
    os.makedirs(os.path.join(args.outdir, "ckpt"), exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    spec = Mo.ARMS[args.arm]
    x, vol = D.load_clean("train")
    keep = np.flatnonzero(vol < args.n_vol)                                                       # a prefix: indices = positions
    if args.one_slice:                                                                            # GAP-N
        keep = np.array([i for i in keep if (i - np.flatnonzero(vol == vol[i])[0]) == vol[i] % D.NSL])
    x = x[keep]
    Y = torch.from_numpy(np.stack([_noisy(TP.content(x[j], "train", i, args.content, args.noise), "train", i, args.noise, args.whiten) for j, i in enumerate(keep)])).to(dev)
    net, m = build(args.arm, dev)
    opt = torch.optim.Adam(net.parameters(), lr=CFG["LR"])
    ckpt_cfg = dict(arm=args.arm, noise=args.noise, seed=args.seed, M=TP.M, sig=TP.SIG, sigma_min=float(m.dsm.sigmas[0]),
                    sigma_max=float(m.dsm.sigmas[-1]), p_raw=CFG["P_RAW"], n_vol=args.n_vol)
    B = args.batch
    if args.crop != CFG["CROP"] or B != CFG["BATCH"] or args.grid != TP.M:                       # GAP C; absent keys = the GAP B setting
        ckpt_cfg.update(batch=B, crop=args.crop, grid=args.grid)
    if args.content != "F":                                                                       # GAP D/E
        ckpt_cfg.update(content=args.content)
    if args.whiten:                                                                               # GAP-M
        ckpt_cfg.update(whiten=True)
    if args.one_slice:                                                                            # GAP-N
        ckpt_cfg.update(one_slice=True, train_indices=keep.tolist())
    json.dump(dict(cfg=CFG, ckpt_cfg=ckpt_cfg, n_train_images=len(x)), open(os.path.join(args.outdir, "config.json"), "w"), indent=2)
    log = open(os.path.join(args.outdir, "train.log"), "a", encoding="utf-8")
    with open(os.path.join(args.outdir, "train.csv"), "w", newline="") as f:
        csv.writer(f).writerow(["iter", "loss_avg50", "sec"])
    c, recent, t0, G = args.crop, [], time.time(), args.grid
    if G != TP.M:
        pad = (G - TP.M) // 2

        def copies(i):
            k = torch.zeros((D.N, G, G), dtype=Y.dtype, device=dev); k[:, pad:pad + TP.M, pad:pad + TP.M] = fft2c(Y[i])
            return ifft2c(k) * (G / TP.M)
    else:
        def copies(i):
            return Y[i]
    sy = torch.full((B,), TP.SIG, device=dev)
    for it in range(1, args.iters + 1):
        idx = torch.randint(0, len(x), (B,)).tolist()
        tl = torch.randint(0, G - c + 1, (B, 2)).tolist()
        y = torch.stack([stack(copies(i)[:, t:t + c, l:l + c]) for i, (t, l) in zip(idx, tl)])
        if spec.get("sub"):
            loss = m.loss_subset(y, sigma_y=sy, loss="mse", p_raw=CFG["P_RAW"], m_dist=spec.get("m_dist"))
        else:
            loss = m.loss_multireal(y, loss="mse", sigma_y=sy, p_raw=CFG["P_RAW"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CFG["GRAD_CLIP"]); opt.step()
        recent = (recent + [loss.item()])[-50:]
        if it % 50 == 0:
            with open(os.path.join(args.outdir, "train.csv"), "a", newline="") as f:
                csv.writer(f).writerow([it, float(np.mean(recent)), time.time() - t0])
        if it % CFG["CKPT_EVERY"] == 0 or it == args.iters:
            line = f"[{time.strftime('%H:%M:%S')}] iter {it} loss={np.mean(recent):.6f}"
            print(line, flush=True); log.write(line + "\n"); log.flush()
            torch.save({"state_dict": net.state_dict(), "cfg": ckpt_cfg, "train_iters": it},
                       os.path.join(args.outdir, "ckpt", f"res_{it:06d}.pkl"))
    log.write(f"DONE arm {args.arm} noise {args.noise} seed {args.seed}: {args.iters} iterations in {(time.time() - t0) / 60:.1f} min\n")
    log.close()


if __name__ == "__main__":
    main()
