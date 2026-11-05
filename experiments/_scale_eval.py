# -*- coding: utf-8 -*-
"""GAP A1/A2 (Archive/GAP/PREREGISTRATION.md): decode with a re-scaled noise label, evaluation only.

As ``_gr_eval.py --single`` (same estimator, masks, splits, r grid), but the decoder is told c * sigma_hat (PER) or
c * sigma_hat / sqrt(a) (SUB) for every c in SCALES, plus c = "brain": the slice's own in-signal / whole-image noise
std ratio (A2: the scalar label averages the whole image, in-brain noise is larger). Column names: neff_c<c>_r<r>.

    python experiments/_scale_eval.py --run runs/subj/SUB_s101 --iter 3000 --mode sub --datasets 6 --out <dir>
"""
import os, csv, json, argparse, itertools
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import _gr_eval as G                                          # sets cwd / paths
from _ratio_core import RATIOS, A_SIZES, REGIONS, cmag

SCALES = (0.5, 0.65, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0)


def score_slice(model, ens_t, sg, device, mode):
    """{(a, region, c): {r: n_eff}}; c is a float or "brain"."""
    N = ens_t.shape[0]
    ens = ens_t.double().cpu().numpy()
    H, W = ens.shape[-2:]
    mean10 = ens.mean(0)
    vpix = ((((ens - mean10) ** 2).sum(0)) / (N - 1)).mean(0)
    smooth = gaussian_filter(cmag(mean10), 3.0)
    masks = {"signal": smooth > np.median(smooth), "all": np.ones((H, W), bool)}
    s2 = {r: float(vpix[v].mean()) for r, v in masks.items()}
    cb = float(np.sqrt(s2["signal"] / s2["all"]))
    scales = [(c, c) for c in SCALES] + [("brain", cb)]

    def dec(y, s):
        out = []
        with torch.no_grad():
            for i in range(0, y.shape[0], G.CHUNK):
                yt = torch.from_numpy(y[i:i + G.CHUNK]).to(ens_t.dtype)
                out.append(model.denoise_to(yt, s, target_ratio=0.0, device=device).double().cpu().numpy())
        return np.concatenate(out)

    rep = {name: dec(ens, c * sg) for name, c in scales} if mode == "per" else None
    res = {}
    for a in A_SIZES:
        combos = list(itertools.combinations(range(N), a))
        WA = np.zeros((len(combos), N)); WB = np.zeros((len(combos), N))
        for i, cmb in enumerate(combos):
            WA[i, list(cmb)] = 1.0 / a
            WB[i, [j for j in range(N) if j not in cmb]] = 1.0 / (N - a)
        ref = (WB @ ens.reshape(N, -1)).reshape(-1, 2, H, W)
        van = (WA @ ens.reshape(N, -1)).reshape(-1, 2, H, W)
        for name, c in scales:
            if mode == "per":
                x0a = (WA @ rep[name].reshape(N, -1)).reshape(-1, 2, H, W)
            else:
                x0a = dec(van, c * sg / np.sqrt(a))
            for rg in masks:
                res[(a, rg, name)] = {}
            for r in RATIOS:
                e = x0a + r * (van - x0a)
                d2 = ((e - ref) ** 2).mean(1)
                for rg, v in masks.items():
                    mse = float((d2[:, v].mean(1) - s2[rg] / (N - a)).mean())
                    res[(a, rg, name)][r] = s2[rg] / mse if mse > 0 else float("inf")
            for rg in REGIONS:
                assert np.isclose(res[(a, rg, name)][1.0], a, rtol=1e-6), ("r=1 must equal a", a, rg)
    return res, cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--mode", required=True, choices=["per", "sub"])
    ap.add_argument("--datasets", type=int, nargs="+", required=True); ap.add_argument("--out", required=True)
    a_ = ap.parse_args()
    assert not os.path.exists(os.path.join(a_.out, "neff.csv")), "refusing to overwrite " + a_.out
    os.makedirs(a_.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, meta, sig = G.load_datasets(a_.datasets)
    folds = G.cv_folds()
    cfg = json.load(open(os.path.join(a_.run, "foldall", "config.json"), encoding="utf-8"))
    assert cfg["held_out"] == [] and not set(cfg["datasets"]) & set(a_.datasets), "test data used in training"
    ck = os.path.join(a_.run, "foldall", "ckpt", "%s_%06d.pkl" % (cfg["ckpt_cfg"].get("loss", "l1"), a_.iter))
    model = G.load(ck, device)[0]
    rows = []
    for i, (d, s) in enumerate(meta):
        res, cb = score_slice(model, x[i].to(device), float(sig[i]), device, a_.mode)
        for a in A_SIZES:
            for rg in REGIONS:
                row = dict(dataset=d, slice=s, fold=folds[s], sigma_hat=float(sig[i]), c_brain=cb, a=a, region=rg)
                for name in [*SCALES, "brain"]:
                    tag = name if name == "brain" else f"{name:g}"
                    row.update({f"neff_c{tag}_r{r:.4f}": res[(a, rg, name)][r] for r in RATIOS})
                rows.append(row)
    with open(os.path.join(a_.out, "neff.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", len(rows), "rows")


if __name__ == "__main__":
    main()
