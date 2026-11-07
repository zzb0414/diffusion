# -*- coding: utf-8 -*-
"""GAP K2 (Archive/GAP/PREREGISTRATION_K2.md): leakage-free scoring of the GAP K pre-whitened SUB / SUBM models. Same as
``_gr_eval.score_slice`` (sub mode) except that, for every A/B split, the model's noise map is built from the A-half
repetitions only (a = 1: full map, flagged). ``--full-map`` substitutes the full 10-repetition map everywhere, which must
reproduce the GAP K neff.csv (``_wh_eval.py``). Refuses to overwrite.

    python experiments/_wh_eval_clean.py --run runs/gap/k/SUB_s101 --iter 3000 --datasets 6 --out runs/gap/k2/SUB_s101/eval_it3000
"""
import os, csv, json, argparse, itertools
import _gr_eval as G                                          # first (OpenMP load order); sets sys.path / cwd to the repo
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from _data import load_datasets, cv_folds
from _ratio_core import RATIOS, A_SIZES, cmag
import _wh


def amap(e):
    """e (k, 2, H, W) float64 numpy, k >= 2 -> s (H, W), same formula as _wh.smap."""
    v = (((e - e.mean(0)) ** 2).sum(0) / (e.shape[0] - 1)).mean(0)
    v = gaussian_filter(v, _wh.SMOOTH, mode="nearest")
    return np.sqrt(v / v.mean())


def score_slice_clean(model, ens_t, sg, device, full_map=False):
    N = ens_t.shape[0]
    ens = ens_t.double().cpu().numpy()
    H, W = ens.shape[-2:]
    mean10 = ens.mean(0)
    vpix = ((((ens - mean10) ** 2).sum(0)) / (N - 1)).mean(0)
    smooth = gaussian_filter(cmag(mean10), 3.0)
    masks = {"signal": smooth > np.median(smooth), "all": np.ones((H, W), bool)}
    s2 = {r: float(vpix[v].mean()) for r, v in masks.items()}
    full = _wh.smap(ens_t)

    def dec(y, maps, s):
        out = []
        with torch.no_grad():
            for i in range(0, y.shape[0], G.CHUNK):
                yt = torch.from_numpy(y[i:i + G.CHUNK]).to(ens_t.dtype)
                w = _wh.Whitened(model, maps[i:i + G.CHUNK][:, None])
                out.append(w.denoise_to(yt, s, target_ratio=0.0, device=device).double().cpu().numpy())
        return np.concatenate(out)

    res = {}
    for a in A_SIZES:
        combos = list(itertools.combinations(range(N), a))
        WA = np.zeros((len(combos), N)); WB = np.zeros((len(combos), N))
        for i, c in enumerate(combos):
            WA[i, list(c)] = 1.0 / a
            WB[i, [j for j in range(N) if j not in c]] = 1.0 / (N - a)
        ref = (WB @ ens.reshape(N, -1)).reshape(-1, 2, H, W)
        van = (WA @ ens.reshape(N, -1)).reshape(-1, 2, H, W)
        if full_map or a == 1:
            maps = np.broadcast_to(full, (len(combos), H, W))
        else:
            maps = np.stack([amap(ens[list(c)]) for c in combos])
        x0a = dec(van, maps, sg / np.sqrt(a))
        for rg in masks:
            res[(a, rg)] = {}
        for r in RATIOS:
            e = x0a + r * (van - x0a)
            d2 = ((e - ref) ** 2).mean(1)
            for rg, v in masks.items():
                mse = float((d2[:, v].mean(1) - s2[rg] / (N - a)).mean())
                res[(a, rg)][r] = s2[rg] / mse if mse > 0 else float("inf")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--datasets", type=int, nargs="+", default=[6]); ap.add_argument("--out", required=True)
    ap.add_argument("--full-map", action="store_true"); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--slices", type=int, default=0, help="debug: first n only")
    a_ = ap.parse_args()
    assert os.path.isfile(os.path.join(a_.run, "foldall", "wh.json")), "model not trained by _wh_train.py"
    assert not os.path.exists(os.path.join(a_.out, "neff.csv")), "refusing to overwrite " + a_.out
    cfg = json.load(open(os.path.join(a_.run, "foldall", "config.json"), encoding="utf-8"))
    assert cfg["held_out"] == [] and not set(cfg["datasets"]) & set(a_.datasets), "test data used in training"
    os.makedirs(a_.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() and not a_.cpu else "cpu"
    ck = os.path.join(a_.run, "foldall", "ckpt", "%s_%06d.pkl" % (cfg.get("ckpt_cfg", {}).get("loss", "l1"), a_.iter))
    m = G.load(ck, dev)[0]
    x, meta, sig = load_datasets(a_.datasets)
    folds = cv_folds(); rows = []
    for i, (d, s) in enumerate(meta):
        if a_.slices and i >= a_.slices:
            break
        res = score_slice_clean(m, x[i].to(dev), float(sig[i]), dev, a_.full_map)
        for (a, rg), v in res.items():
            rows.append(dict(dataset=d, slice=s, fold=folds[s], sigma_hat=float(sig[i]), a=a, region=rg,
                             **{f"neff_r{r:.4f}": v[r] for r in RATIOS}))
    with open(os.path.join(a_.out, "neff.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(run=a_.run, iter=a_.iter, mode="sub", datasets=a_.datasets, whitened=True, smooth=_wh.SMOOTH,
                   map="full" if a_.full_map else "A-half (a=1 full)"),
              open(os.path.join(a_.out, "eval_config.json"), "w"), indent=2)
    print("wrote", a_.out, len(rows), "rows")


if __name__ == "__main__":
    main()
