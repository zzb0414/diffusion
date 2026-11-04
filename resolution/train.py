# -*- coding: utf-8 -*-
"""Multi-resolution study training (Archive/RES/PREREGISTRATION.md).

Same recipe for every arm: 2-channel (Re/Im) U-Net base 32 / emb 32, Adam 1e-3, grad clip 1, 3000 iterations, MSE.
Each step draws one training matrix M (uniform over the arm's set) and 8 training images; the N = 10 fixed copies of
each image at M are made on the GPU from its cached clean k-space plus its own seeded k-space noise (the same noise
field at every M, as data.py); one random 64 x 64 crop is shared by the copies. Arm D works on the 320 grid
(zero-filled k-space). Labels: sigma(M) per channel (known; physical setting).

    python resolution/train.py --arm B --seed 101 --outdir runs/res/B_s101
"""
import os, time, json, argparse, csv
import numpy as np
import torch
import data as D
import models as Mo
from models import fft2c, ifft2c

CFG = dict(ITERS=3000, BATCH=8, CROP=64, LR=1e-3, GRAD_CLIP=1.0, P_RAW=0.5, CKPT_EVERY=1000)


class Copies:
    """GPU generator of the fixed noisy copies of training image i at matrix M (or on the 320 grid for arm D)."""

    def __init__(self, x320, dev):
        self.K = fft2c(torch.from_numpy(x320).to(dev))                       # (n, 320, 320) clean k-space
        self.dev = dev

    def noise(self, i):
        g = torch.Generator(device=self.dev).manual_seed(D.SEED + 7919 * i)
        z = torch.randn((D.N, D.FULL, D.FULL, 2), generator=g, device=self.dev)
        return D.S_K * torch.complex(z[..., 0], z[..., 1])

    def get(self, i, M, grid=None):
        k = self.K[i][None] + self.noise(i)                                   # (N, 320, 320)
        c = (D.FULL - M) // 2
        if grid == D.FULL:                                                     # zero-filled to the 320 grid
            mask = torch.zeros(D.FULL, D.FULL, device=self.dev); mask[c:c + M, c:c + M] = 1
            return ifft2c(k * mask)
        return ifft2c(k[:, c:c + M, c:c + M]) * (M / D.FULL)


def stack(z):
    """(N, h, w) complex -> (2N, h, w) real, interleaved [Re1, Im1, Re2, ...]."""
    return torch.stack([z.real, z.imag], 1).reshape(-1, *z.shape[-2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(Mo.ARMS))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--iters", type=int, default=CFG["ITERS"])
    ap.add_argument("--n-vol", type=int, default=0, help="MEAN study: train on the first n training volumes only (0 = all)")
    args = ap.parse_args()
    os.chdir(D.REPO)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert not os.path.exists(os.path.join(args.outdir, "train.csv")), "refusing to overwrite " + args.outdir
    os.makedirs(os.path.join(args.outdir, "ckpt"), exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    spec = Mo.ARMS[args.arm]
    x, vol = D.load_clean("train")
    if args.n_vol:
        x = x[vol < args.n_vol]
    gen = Copies(x, dev)
    net, m = Mo.build(args.arm, dev)
    opt = torch.optim.Adam(net.parameters(), lr=CFG["LR"])
    ckpt_cfg = dict(arm=args.arm, seed=args.seed, train_res=list(spec["train_res"]), sigma_min=float(m.dsm.sigmas[0]),
                    sigma_max=float(m.dsm.sigmas[-1]), param=m.param, blind=m.blind, p_raw=CFG["P_RAW"],
                    n_vol=args.n_vol)
    json.dump(dict(cfg=CFG, ckpt_cfg=ckpt_cfg, n_train_images=len(x)), open(os.path.join(args.outdir, "config.json"), "w"), indent=2)
    log = open(os.path.join(args.outdir, "train.log"), "a", encoding="utf-8")
    with open(os.path.join(args.outdir, "train.csv"), "w", newline="") as f:
        csv.writer(f).writerow(["iter", "M", "loss_avg50", "sec"])
    c, recent, t0 = CFG["CROP"], [], time.time()
    for it in range(1, args.iters + 1):
        M = spec["train_res"][int(torch.randint(len(spec["train_res"]), (1,)))]
        G = spec.get("grid") or M
        idx = torch.randint(0, len(x), (CFG["BATCH"],)).tolist()
        tl = torch.randint(0, G - c + 1, (CFG["BATCH"], 2)).tolist()
        y = torch.stack([stack(gen.get(i, M, spec.get("grid"))[:, t:t + c, l:l + c]) for i, (t, l) in zip(idx, tl)])
        sy = torch.full((CFG["BATCH"],), D.sigma(M), device=dev)
        if spec.get("vox"):
            net.vox = Mo.vox_coord(M)
        if spec.get("sub"):
            loss = m.loss_subset(y, sigma_y=sy, loss="mse", p_raw=CFG["P_RAW"], m_dist=spec.get("m_dist"),
                                 m_weight=spec.get("m_weight"))
        elif spec.get("classic"):
            loss = m.loss_multireal(y, loss="mse", sigma_y=sy, p_raw=1.0, target_k=1)
        else:
            loss = m.loss_multireal(y, loss="mse", sigma_y=sy, p_raw=CFG["P_RAW"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CFG["GRAD_CLIP"]); opt.step()
        recent = (recent + [loss.item()])[-50:]
        if it % 50 == 0:
            for attempt in range(5):
                try:
                    with open(os.path.join(args.outdir, "train.csv"), "a", newline="") as f:
                        csv.writer(f).writerow([it, M, float(np.mean(recent)), time.time() - t0])
                    break
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(2.0)
        if it % CFG["CKPT_EVERY"] == 0 or it == args.iters:
            line = f"[{time.strftime('%H:%M:%S')}] iter {it} loss={np.mean(recent):.6f}"
            print(line, flush=True); log.write(line + "\n"); log.flush()
            torch.save({"state_dict": net.state_dict(), "cfg": ckpt_cfg, "train_iters": it},
                       os.path.join(args.outdir, "ckpt", f"res_{it:06d}.pkl"))
    log.write(f"DONE arm {args.arm} seed {args.seed}: {args.iters} iterations in {(time.time() - t0) / 60:.1f} min\n"); log.close()


if __name__ == "__main__":
    main()
