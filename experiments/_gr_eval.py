# -*- coding: utf-8 -*-
"""GROUND evaluation: split-half unbiased n_eff for per-repetition and subset-mean models.

Same estimator, masks, splits and ratio grid as ``_ratio_core.evaluate_slice``, without the blur
baselines (fast enough for learning curves):

  --mode per   decode every repetition once at sigma_hat, blend each ratio, average over the split's A half
               ("denoise each, then average").
  --mode sub   for every split, decode the MEAN of the A half once at sigma_hat / sqrt(a), then blend
               ("denoise the mean").

Every slice is decoded by the fold model that held its slice index out. ``--check CSV`` compares the
per-mode n_eff with an ``evaluate_slice`` CSV (must agree to 1e-6 relative). Refuses to overwrite.

    python experiments/_gr_eval.py --run runs/gr/kr/k1_s102 --iter 2000 --mode per --out runs/gr/kr/k1_s102/eval_it2000
"""
import sys, os, csv, json, argparse, itertools
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, DIFF); os.chdir(DIFF)
sys.path.insert(0, os.path.join(DIFF, "experiments"))
from _r3_eval import load                                    # noqa: E402
from _data import load_datasets, cv_folds                    # noqa: E402
from _ratio_core import RATIOS, A_SIZES, REGIONS, cmag       # noqa: E402

CHUNK = 64


def score_slice(model, ens_t, sg, device, mode):
    """{(a, region): {r: n_eff}} for one slice."""
    N = ens_t.shape[0]
    ens = ens_t.double().cpu().numpy()
    H, W = ens.shape[-2:]
    mean10 = ens.mean(0)
    vpix = ((((ens - mean10) ** 2).sum(0)) / (N - 1)).mean(0)
    smooth = gaussian_filter(cmag(mean10), 3.0)
    masks = {"signal": smooth > np.median(smooth), "all": np.ones((H, W), bool)}
    s2 = {r: float(vpix[v].mean()) for r, v in masks.items()}

    def dec(y, s):                                # full Tweedie estimate, chunked
        out = []
        with torch.no_grad():
            for i in range(0, y.shape[0], CHUNK):
                yt = torch.from_numpy(y[i:i + CHUNK]).to(ens_t.dtype)
                out.append(model.denoise_to(yt, s, target_ratio=0.0, device=device).double().cpu().numpy())
        return np.concatenate(out)

    x0_rep = dec(ens, sg) if mode == "per" else None
    res = {}
    for a in A_SIZES:
        combos = list(itertools.combinations(range(N), a))
        WA = np.zeros((len(combos), N)); WB = np.zeros((len(combos), N))
        for i, c in enumerate(combos):
            WA[i, list(c)] = 1.0 / a
            WB[i, [j for j in range(N) if j not in c]] = 1.0 / (N - a)
        ref = (WB @ ens.reshape(N, -1)).reshape(-1, 2, H, W)
        van = (WA @ ens.reshape(N, -1)).reshape(-1, 2, H, W)
        if mode == "per":
            x0a = (WA @ x0_rep.reshape(N, -1)).reshape(-1, 2, H, W)   # mean of decoded A half
        else:
            x0a = dec(van, sg / np.sqrt(a))                            # decode the A-half mean
        for rg, v in masks.items():
            res[(a, rg)] = {}
        for r in RATIOS:
            e = x0a + r * (van - x0a)
            d2 = ((e - ref) ** 2).mean(1)
            for rg, v in masks.items():
                mse = float((d2[:, v].mean(1) - s2[rg] / (N - a)).mean())
                res[(a, rg)][r] = s2[rg] / mse if mse > 0 else float("inf")
        for rg in REGIONS:
            assert np.isclose(res[(a, rg)][1.0], a, rtol=1e-6), ("r=1 must equal a", a, rg)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--mode", required=True, choices=["per", "sub"])
    ap.add_argument("--datasets", type=int, nargs="+", default=[1])
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default=None, help="evaluate_slice CSV to reproduce (per mode)")
    a_ = ap.parse_args()
    assert not os.path.exists(os.path.join(a_.out, "neff.csv")), "refusing to overwrite " + a_.out
    os.makedirs(a_.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, meta, sig = load_datasets(a_.datasets)
    folds = cv_folds()
    rows = []
    for k in range(5):
        cfg = json.load(open(os.path.join(a_.run, f"fold{k}", "config.json"), encoding="utf-8"))
        assert sorted(cfg["held_out"]) == sorted(s for s, f in folds.items() if f == k), "fold mismatch"
        model = load(os.path.join(a_.run, f"fold{k}", "ckpt", "l1_%06d.pkl" % a_.iter), device)[0]
        for i, (d, s) in enumerate(meta):
            if folds[s] != k:
                continue
            assert s not in cfg["train_slices"], "model trained on this slice index"
            res = score_slice(model, x[i].to(device), float(sig[i]), device, a_.mode)
            for (a, rg), v in res.items():
                rows.append(dict(dataset=d, slice=s, fold=k, sigma_hat=float(sig[i]), a=a, region=rg,
                                 **{f"neff_r{r:.4f}": v[r] for r in RATIOS}))
        del model; torch.cuda.empty_cache()
    if a_.check:
        ref = {}
        for r in csv.DictReader(open(a_.check, encoding="utf-8")):
            if r["region"] in REGIONS:
                ref[(int(r.get("dataset", 1)), int(r["slice"]), int(r["a"]), r["region"])] = r
        worst = 0.0
        for r in rows:
            q = ref[(r["dataset"], r["slice"], r["a"], r["region"])]
            for c in (f"neff_r{v:.4f}" for v in RATIOS):
                worst = max(worst, abs(r[c] / float(q[c]) - 1))
        print(f"check vs {a_.check}: worst relative difference {worst:.2e}")
        assert worst < 1e-6, "does not reproduce evaluate_slice"
    with open(os.path.join(a_.out, "neff.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(run=a_.run, iter=a_.iter, mode=a_.mode, datasets=a_.datasets),
              open(os.path.join(a_.out, "eval_config.json"), "w"), indent=2)
    print("wrote", a_.out, len(rows), "rows")


if __name__ == "__main__":
    main()
