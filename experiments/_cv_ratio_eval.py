# -*- coding: utf-8 -*-
"""CV_RATIO evaluation: target_ratio sweep x split size, per loss arm.

Implements ``Archive/CV_RATIO/PREREGISTRATION.md`` (evaluation only). For one
arm (a CV directory of 5 fold models), every slice is decoded by the fold model
that held it out; each of its 10 realizations is decoded ALONE once at r = 0
(x0_hat), and every ratio is the exact blend out = x0_hat + r (y - x0_hat).

Per slice, r in RATIOS, a in A_SIZES, all C(10, a) splits:
  n_eff (signal / all regions) of the average of a outputs vs mean(B), with the
  same unbiased estimator as ``_cv_eval.py`` (CV_R3).
Per slice, per r: achieved residual noise of a single output (unbiased spread of
the 10 outputs / sigma_hat) and the magnitude gain of the average of 10.
Per band (a in BAND_A): n_eff in 4 radial frequency bands.
Baselines: plain average (r = 1), oracle-MSE blur (as CV_R3) and NOISE-MATCHED
blur (width solved so a blurred realization has the arm's residual noise at
r = 1/3).

Self-checks against the existing CV results of the same arm: r = 0 reproduces
the R3 n_eff, r = 1 gives n_eff = a, oracle blur is identical.

Also saves x0_hat of the display slices (``x0_s<NN>.npy``) for the map figures.

    python experiments/_cv_ratio_eval.py --cv runs/cv_r3
"""
import sys, os, csv, json, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, DIFF); os.chdir(DIFF)

from _r3_train import load_data, sigma_hat                   # noqa: E402
from _r3_eval import load                                    # noqa: E402
from _ratio_core import evaluate_slice                      # noqa: E402  (ratios, splits, blur, bands)

DISPLAY = (17, 16, 11, 1, 25, 22, 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True)
    ap.add_argument("--out", default=None, help="output dir (default <cv>/ratio_eval)")
    a_ = ap.parse_args()
    CV = a_.cv
    OUT = a_.out or os.path.join(CV, "ratio_eval")
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = load_data(); sig = sigma_hat(x)
    Ns = x.shape[0]
    fold_of, ckname = {}, {}
    for k in range(5):
        cfg = json.load(open(os.path.join(CV, f"fold{k}", "config.json"), encoding="utf-8"))
        ckname[k] = "%s_002000.pkl" % cfg["ckpt_cfg"]["loss"]
        for s in cfg["held_out"]:
            fold_of[s] = k
    assert sorted(fold_of) == list(range(Ns))
    old = {(int(r["slice"]), int(r["a"]), r["region"]): r
           for r in csv.DictReader(open(os.path.join(CV, "eval", "per_slice.csv"), encoding="utf-8"))}

    rows, brows, model = [], [], {}
    for s in range(Ns):
        k = fold_of[s]
        if k not in model:
            model = {k: load(os.path.join(CV, f"fold{k}", "ckpt", ckname[k]), device)[0]}
        ens_t = x[s].to(device); sg = float(sig[s])

        def check(a, rg, res, blur_best, s=s):
            o = old[(s, a, rg)]
            assert np.isclose(res[0.0], float(o["r3"]), rtol=1e-6), ("r=0 mismatch", s, a, rg)
            assert np.isclose(blur_best, float(o["blur_oracle"]), rtol=1e-6), ("blur", s, a, rg)

        r_rows, b_rows, x0, n_lock, w_star = evaluate_slice(
            model[k], ens_t, sg, device, key=dict(slice=s, fold=k), bkey=dict(slice=s), check=check)
        if s in DISPLAY:
            np.save(os.path.join(OUT, f"x0_s{s:02d}.npy"), x0.astype(np.float32))
        rows += r_rows; brows += b_rows
        print(f"slice {s:2d} fold {k} sigma {sg:.4f}  noise@r=1/3 {n_lock:.3f}  "
              f"w_matched {w_star:.2f}  a=1 signal n_eff r=1/3 "
              f"{next(q for q in r_rows if q['a'] == 1 and q['region'] == 'signal')['neff_r0.3333']:.2f}",
              flush=True)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(OUT, "per_slice_ratio.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "bands_ratio.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(brows[0].keys())); w.writeheader(); w.writerows(brows)
    print("wrote", OUT, len(rows), "rows;", len(brows), "band rows; self-checks passed")


if __name__ == "__main__":
    main()
