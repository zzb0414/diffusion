# -*- coding: utf-8 -*-
"""Multi-resolution study evaluation (Archive/RES/PREREGISTRATION.md).

For one checkpoint: every test image (15 volumes x 8 slices) at every matrix M in data.RES, under two noise settings:
  phys     the physical setting (per-channel noise sigma(M) = S_K M / 320; primary)
  matched  per-channel noise 0.5 * S_K (= sigma(160)) at every M (diagnostic: separates image scale from noise level)
Each copy is decoded once (one-shot Tweedie / direct output), blended with r, averaged over the fixed subsets of size a
(baselines/_common.splits). Arm D: copies zero-filled to the 320 grid, decoded there, and brought back to M.
Arm C: the voxel size of M is passed. Rows per (image, r, a): mse_gt (vs the clean image at M) and mse_sh (split-half,
no ground truth), per channel, over the head mask (|clean| > 5% of its p99) and over the whole image; s2 = per-channel
noise variance from the copies (head mask).

    python resolution/evaluate.py --run runs/res/B_s101 --out runs/res/eval/B_s101.csv
"""
import os, csv, time, argparse
import numpy as np
import torch
import data as D
import models as Mo
from models import fft2c, ifft2c
from _common import splits

A_EVAL = (1, 3, 5, 9)
RGRID = (0.0, 1 / 12, 1 / 6, 0.25, 1 / 3, 0.5)
SETTINGS = {"phys": lambda M: D.sigma(M), "matched": lambda M: 0.5 * D.S_K}


def copies(x320, i, M, s):
    """clean (M, M) and N copies at matrix M with per-channel noise s (numpy, deterministic per image)."""
    k = fft2c(torch.from_numpy(x320))
    c = (D.FULL - M) // 2
    noise = torch.from_numpy(D.kspace_noise("test", i)) * (s / D.sigma(M))
    clean = ifft2c(k[c:c + M, c:c + M]) * (M / D.FULL)
    y = ifft2c((k[None] + noise)[:, c:c + M, c:c + M]) * (M / D.FULL)
    return clean.numpy(), y.numpy()


@torch.no_grad()
def decode(m, arm, y, s, M, dev):
    """y (N, M, M) complex -> one-shot estimates (N, M, M) complex."""
    yt = torch.from_numpy(y).to(dev)
    if arm == "D":                                                     # zero-fill to the 320 grid
        c = (D.FULL - M) // 2
        k = torch.zeros((len(y), D.FULL, D.FULL), dtype=yt.dtype, device=dev)
        k[:, c:c + M, c:c + M] = fft2c(yt) * (D.FULL / M)
        yt = ifft2c(k)
    if arm == "C":
        m.net.vox = Mo.vox_coord(M)
    inp = torch.stack([yt.real, yt.imag], 1).float()
    out = m.denoise_to(inp, s, target_ratio=0.0, device=dev)
    z = torch.complex(out[:, 0], out[:, 1])
    if arm == "D":
        c = (D.FULL - M) // 2
        z = ifft2c(fft2c(z)[:, c:c + M, c:c + M]) * (M / D.FULL)
    return z.cpu().numpy()


def score(clean, y, x0, dev="cuda"):
    """rows {(r, a): (mse_gt_head, mse_gt_all, mse_sh_head, mse_sh_all)} per channel; s2 over the head mask.
    Computed on the GPU in complex128 (same arithmetic as the numpy reference ``score_np``)."""
    N = y.shape[0]
    head_np = np.abs(clean) > 0.05 * np.percentile(np.abs(clean), 99)
    c, Y, X0 = (torch.from_numpy(np.ascontiguousarray(z)).to(dev) for z in (clean, y, x0))
    head = torch.from_numpy(head_np).to(dev)
    v = (Y - Y.mean(0)).abs().square().sum(0) / (N - 1) / 2
    s2 = {"head": float(v[head].mean()), "all": float(v.mean())}
    res = {}
    for a in A_EVAL:
        sub = splits(a, N)
        WA = np.zeros((len(sub), N)); WB = np.zeros((len(sub), N))
        for j, cmb in enumerate(sub):
            WA[j, list(cmb)] = 1 / a; WB[j, [q for q in range(N) if q not in cmb]] = 1 / (N - a)
        WA, WB = (torch.from_numpy(w).to(dev, torch.complex128) for w in (WA, WB))
        ref = torch.tensordot(WB, Y, 1)
        for r in RGRID:
            est = torch.tensordot(WA, X0 + r * (Y - X0), 1)
            e_gt = (est - c[None]).abs().square() / 2; e_sh = (est - ref).abs().square() / 2
            res[(round(r, 4), a)] = (float(e_gt[:, head].mean()), float(e_gt.mean()),
                                     float(e_sh[:, head].mean()) - s2["head"] / (N - a), float(e_sh.mean()) - s2["all"] / (N - a))
    return res, s2["head"]


def score_np(clean, y, x0):
    """rows {(r, a): (mse_gt_head, mse_gt_all, mse_sh_head, mse_sh_all)} per channel; s2 over the head mask."""
    N = y.shape[0]
    head = np.abs(clean) > 0.05 * np.percentile(np.abs(clean), 99)
    dev = y - y.mean(0)
    v = (np.abs(dev) ** 2).sum(0) / (N - 1) / 2                        # per-channel variance map
    s2 = {"head": float(v[head].mean()), "all": float(v.mean())}
    res = {}
    for a in A_EVAL:
        sub = splits(a, N)
        WA = np.zeros((len(sub), N)); WB = np.zeros((len(sub), N))
        for j, cmb in enumerate(sub):
            WA[j, list(cmb)] = 1 / a; WB[j, [q for q in range(N) if q not in cmb]] = 1 / (N - a)
        ref = np.tensordot(WB, y, 1)
        for r in RGRID:
            est = np.tensordot(WA, x0 + r * (y - x0), 1)
            e_gt = np.abs(est - clean[None]) ** 2 / 2; e_sh = np.abs(est - ref) ** 2 / 2
            res[(round(r, 4), a)] = (float(e_gt[:, head].mean()), float(e_gt.mean()),
                                     float(e_sh[:, head].mean()) - s2["head"] / (N - a), float(e_sh.mean()) - s2["all"] / (N - a))
    return res, s2["head"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iter", type=int, default=3000)
    ap.add_argument("--max-img", type=int, default=0, help="smoke test: first n test images only")
    args = ap.parse_args()
    os.chdir(D.REPO)
    assert not os.path.exists(args.out), "refusing to overwrite " + args.out
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, cfg = Mo.load(os.path.join(args.run, "ckpt", f"res_{args.iter:06d}.pkl"), dev)
    x, vol = D.load_clean("test")
    rows, t0 = [], time.time()
    for setting, sfun in SETTINGS.items():
        for M in D.RES:
            s = sfun(M)
            for i in range(args.max_img or len(x)):
                clean, y = copies(x[i], i, M, s)
                x0 = decode(m, cfg["arm"], y, s, M, dev)
                res, s2 = score(clean.astype(np.complex128), y.astype(np.complex128), x0.astype(np.complex128))
                rows += [(cfg["arm"], cfg["seed"], setting, M, int(vol[i]), i, r, a, s, s2) + v for (r, a), v in res.items()]
            print(f"{setting} M={M} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "setting", "M", "vol", "img", "r", "a", "sigma", "s2_head",
                    "mse_gt_head", "mse_gt_all", "mse_sh_head", "mse_sh_all"])
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} rows in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
