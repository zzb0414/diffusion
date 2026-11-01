# -*- coding: utf-8 -*-
"""NAT evaluation (Archive/NAT/PREREGISTRATION.md). Long-format CSV per (condition, method):
split (set12 / bsd68), image, r, a -> mse_gt (vs the clean image), psnr (estimate clipped to [0, 1]),
mse_sh (split-half estimate, no ground truth: mean over subsets of |est - ref|^2 minus s2/(N - a), with s2 the
per-image noise variance estimated from the copies), averaged over the fixed subsets of size a (shared by all methods).

Learned methods: each copy decoded once (one-shot Tweedie / direct output; R2R: mean of 8 recorrupted draws), then
blended with r and averaged over the subset. Classical: ``avg`` (subset mean) and ``bm3d`` (BM3D of the subset mean,
sigma_psd = sigma/sqrt(a)), CPU pool. Refuses to overwrite.

    python natural/evaluate.py --cond W05 --methods ours avg9 ...      (learned, GPU)
    python natural/evaluate.py --cond W05 --methods avg bm3d           (classical, CPU pool)
"""
import os, sys, csv, json, time, argparse
import numpy as np
import data as D
from _common import splits                                   # baselines/_common.py: fixed subsets per a

A_EVAL = (1, 3, 5, 9)
RGRID = (0.0, 1 / 12, 1 / 6, 0.25, 1 / 3, 0.5)
LEARNED = ("ours", "avg9", "avgu", "sup", "n2n", "ne2ne", "r2r", "n2s", "sub", "subsup")
MEAN_MODE = ("sub", "subsup")             # NAT-SUB: denoise the a-copy mean once at sigma/sqrt(a)
_G = {}


def metrics(x, y, outs_by_r):
    """x clean (H,W); y copies (N,H,W); outs_by_r {r: (N,H,W)} -> rows {(r, a): (mse_gt, psnr, mse_sh)}."""
    N = y.shape[0]
    s2 = float((((y - y.mean(0)) ** 2).sum(0) / (N - 1)).mean())
    res = {}
    for a in A_EVAL:
        sub = splits(a, N)
        refs = [y[[j for j in range(N) if j not in c]].mean(0) for c in sub]
        for r, outs in outs_by_r.items():
            g, p, h = [], [], []
            for c, ref in zip(sub, refs):
                est = outs[list(c)].mean(0)
                g.append(float(((est - x) ** 2).mean()))
                p.append(float(10 * np.log10(1.0 / ((np.clip(est, 0, 1) - x) ** 2).mean())))
                h.append(float(((est - ref) ** 2).mean()) - s2 / (N - a))
            res[(r, a)] = (float(np.mean(g)), float(np.mean(p)), float(np.mean(h)), s2)
    return res


def _classical(task):
    split, i, method = task
    x, y = _G[split][0][i], _G[split][1][i]
    sigma = D.CONDITIONS[_G["cond"]][0]
    outs = {}
    if method == "avg":
        outs[0.0] = y
        res = metrics(x, y, outs)
        return [(split, i, method, r, a) + v for (r, a), v in res.items()]
    import bm3d
    N = y.shape[0]
    rows = []
    s2 = float((((y - y.mean(0)) ** 2).sum(0) / (N - 1)).mean())
    for a in A_EVAL:
        sub = splits(a, N)
        g, p, h = [], [], []
        for c in sub:
            ref = y[[j for j in range(N) if j not in c]].mean(0)
            est = np.asarray(bm3d.bm3d(y[list(c)].mean(0), sigma_psd=sigma / np.sqrt(a)), np.float32)
            g.append(float(((est - x) ** 2).mean()))
            p.append(float(10 * np.log10(1.0 / ((np.clip(est, 0, 1) - x) ** 2).mean())))
            h.append(float(((est - ref) ** 2).mean()) - s2 / (N - a))
        rows.append((split, i, method, 0.0, a, float(np.mean(g)), float(np.mean(p)), float(np.mean(h)), s2))
    return rows


def mean_mode(model, x, y, sigma, device):
    """rows (r, a, mse_gt, psnr, mse_sh, s2): each subset's mean denoised once at sigma/sqrt(a), blended with r."""
    import torch
    N = y.shape[0]
    s2 = float((((y - y.mean(0)) ** 2).sum(0) / (N - 1)).mean())
    out = []
    for a in A_EVAL:
        sub = splits(a, N)
        means = np.stack([y[list(c)].mean(0) for c in sub])
        refs = [y[[j for j in range(N) if j not in c]].mean(0) for c in sub]
        with torch.no_grad():
            x0 = model.denoise_to(torch.from_numpy(means[:, None]).float(), sigma / np.sqrt(a), target_ratio=0.0,
                                  device=device).cpu()[:, 0].numpy().astype(np.float64)
        for r in RGRID:
            est = x0 + r * (means - x0)
            g = [float(((e - x) ** 2).mean()) for e in est]
            p = [float(10 * np.log10(1.0 / ((np.clip(e, 0, 1) - x) ** 2).mean())) for e in est]
            h = [float(((e - ref) ** 2).mean()) - s2 / (N - a) for e, ref in zip(est, refs)]
            out.append((round(r, 4), a, float(np.mean(g)), float(np.mean(p)), float(np.mean(h)), s2))
    return out


def _init(cond):
    os.chdir(D.REPO)
    _G["cond"] = cond
    for split in ("set12", "bsd68"):
        ims = D.load_split(split)
        _G[split] = (ims, D.copies(ims, cond, split))


def learned(method, cond, run, device):
    import torch
    from _r3_eval import load
    model = load(os.path.join(run, "ckpt", f"{method}_003000.pkl"), device)[0]
    sigma = D.CONDITIONS[cond][0]
    rows = []
    for split in ("set12", "bsd68"):
        ims, ys = _G[split]
        for i, (x, y) in enumerate(zip(ims, ys)):
            if method in MEAN_MODE:
                rows += [(split, i, method) + v for v in mean_mode(model, x.astype(np.float64), y.astype(np.float64),
                                                                    sigma, device)]
                continue
            yt = torch.from_numpy(y[:, None])
            with torch.no_grad():
                if method == "r2r":
                    g = torch.Generator().manual_seed(D.SEED + 7 * i)
                    x0 = torch.stack([model.denoise_to(yt + 0.5 * sigma * torch.randn(yt.shape, generator=g), sigma,
                                                       target_ratio=0.0, device=device).cpu() for _ in range(8)]).mean(0)
                else:
                    x0 = model.denoise_to(yt, sigma, target_ratio=0.0, device=device).cpu()
            x0 = x0[:, 0].numpy().astype(np.float64)
            res = metrics(x.astype(np.float64), y.astype(np.float64), {r: x0 + r * (y - x0) for r in RGRID})
            rows += [(split, i, method, round(r, 4), a) + v for (r, a), v in res.items()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=list(D.CONDITIONS))
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    _init(args.cond)
    out_dir = os.path.join("runs", "nat", "eval", args.cond)
    os.makedirs(out_dir, exist_ok=True)
    for method in args.methods:
        path = os.path.join(out_dir, f"{method}.csv")
        assert not os.path.exists(path), "refusing to overwrite " + path
        t0 = time.time()
        if method in LEARNED:
            rows = learned(method, args.cond, os.path.join("runs", "nat", args.cond, method), args.device)
        else:
            from multiprocessing import Pool
            tasks = [(sp, i, method) for sp in ("set12", "bsd68") for i in range(len(_G[sp][0]))]
            with Pool(args.workers, initializer=_init, initargs=(args.cond,)) as pool:
                rows = [r for rr in pool.imap_unordered(_classical, tasks) for r in rr]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["split", "image", "method", "r", "a", "mse_gt", "psnr", "mse_sh", "s2"]); w.writerows(rows)
        print(f"{args.cond} {method}: {len(rows)} rows in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
