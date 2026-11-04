# -*- coding: utf-8 -*-
"""MEAN study evaluation (Archive/MEAN/PREREGISTRATION.md): denoise each copy then average (PER, arm A) vs denoise the
mean of the copies (SUB), fastMRI test images at matrix 160, physical noise sigma(160) per channel.

For every test image and every fixed subset A of a copies (baselines/_common.splits, a in 1, 3, 5, 9):
  PER  est = mean_A(x0(y_k)) + r (mean_A(y) - mean_A(x0(y_k)))     x0 decoded at sigma
  SUB  est = x0(mean_A(y)) + r (mean_A(y) - x0(mean_A(y)))          x0 decoded at sigma / sqrt(a)
Rows per (image, r, a): mse_gt_head (vs the clean image, per channel, head mask), mse_sh_head (split-half vs the mean of
the complementary copies, minus its noise; no ground truth), and the band transfer T1..T5 (Re<FFT est, FFT clean> /
|FFT clean|^2 over radial bands of k / k_max: 0-.1, .1-.25, .25-.5, .5-.75, .75-1; 1 = detail kept, 0 = removed),
averaged over the subsets. Same copies and subsets for both designs (paired).

    python resolution/mean_eval.py --run runs/mean/SUB_v8_s101 --out runs/mean/eval/SUB_v8_s101_it3000.csv
"""
import os, csv, time, argparse, importlib.util
import numpy as np
import torch
import data as D
import models as Mo
from models import fft2c
from _common import splits

_sp = importlib.util.spec_from_file_location("res_evaluate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py"))
E = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(E)

M, A_EVAL, RGRID = 160, (1, 3, 5, 9), E.RGRID
BANDS = ((0, .1), (.1, .25), (.25, .5), (.5, .75), (.75, 1.0001))


@torch.no_grad()
def x0_of(m, z, s, dev):
    """(K, M, M) complex numpy -> one-shot estimates at per-channel noise s (scalar or (K,))."""
    zt = torch.from_numpy(np.ascontiguousarray(z)).to(dev)
    inp = torch.stack([zt.real, zt.imag], 1).float()
    s = torch.as_tensor(s, dtype=torch.float32, device=dev).expand(len(z))
    out = m._x0(inp, s)
    return torch.complex(out[:, 0].double(), out[:, 1].double())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iter", type=int, default=3000)
    ap.add_argument("--max-img", type=int, default=0)
    args = ap.parse_args()
    os.chdir(D.REPO)
    assert not os.path.exists(args.out), "refusing to overwrite " + args.out
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, cfg = Mo.load(os.path.join(args.run, "ckpt", f"res_{args.iter:06d}.pkl"), dev)
    sub = cfg["arm"].startswith("SUB")
    x, vol = D.load_clean("test")
    s = D.sigma(M)
    k = torch.from_numpy(np.hypot(*np.meshgrid(*(np.arange(M) - M // 2,) * 2)) / (M / 2)).to(dev)
    bmask = [((k >= lo) & (k < hi)) for lo, hi in BANDS]
    rows, t0 = [], time.time()
    for i in range(args.max_img or len(x)):
        clean, y = E.copies(x[i], i, M, s)
        head = torch.from_numpy(np.abs(clean) > 0.05 * np.percentile(np.abs(clean), 99)).to(dev)
        c = torch.from_numpy(clean.astype(np.complex128)).to(dev); Y = torch.from_numpy(y.astype(np.complex128)).to(dev)
        N = len(y)
        v = (Y - Y.mean(0)).abs().square().sum(0) / (N - 1) / 2; s2 = float(v[head].mean())
        C = fft2c(c); cpow = [float(C[b].abs().square().sum()) for b in bmask]
        X0 = None if sub else x0_of(m, y, s, dev)
        for a in A_EVAL:
            sub_ = splits(a, N)
            WA = torch.zeros(len(sub_), N, dtype=torch.complex128, device=dev); WB = torch.zeros_like(WA)
            for j, cmb in enumerate(sub_):
                WA[j, list(cmb)] = 1 / a; WB[j, [q for q in range(N) if q not in cmb]] = 1 / (N - a)
            ybar = torch.tensordot(WA, Y, 1); ref = torch.tensordot(WB, Y, 1)
            xh = x0_of(m, ybar.cpu().numpy(), s / np.sqrt(a), dev) if sub else torch.tensordot(WA, X0, 1)
            for r in RGRID:
                est = xh + r * (ybar - xh)
                e_gt = (est - c[None]).abs().square() / 2; e_sh = (est - ref).abs().square() / 2
                F = fft2c(est)
                T = [float((F * C.conj()[None])[:, b].real.sum() / len(sub_)) / p for b, p in zip(bmask, cpow)]
                rows.append([cfg["arm"], cfg["seed"], cfg.get("n_vol", 0), args.iter, int(vol[i]), i, round(r, 4), a, s, s2,
                             float(e_gt[:, head].mean()), float(e_sh[:, head].mean()) - s2 / (N - a)] + T)
        if i % 20 == 0:
            print(f"image {i} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "n_vol", "iter", "vol", "img", "r", "a", "sigma", "s2_head", "mse_gt_head", "mse_sh_head",
                    "T1", "T2", "T3", "T4", "T5"])
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} rows in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
