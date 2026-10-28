# -*- coding: utf-8 -*-
"""GROUND step 3 (exploratory, read-only): a bias/variance model of n_eff(a, r) and the optimal r*(a).

Each denoised map's error is (1 - r) d_k + r n_k, with d_k = x0_hat_k - x and n_k the raw noise. Given
the clean map, x0_hat_k depends on y_k alone, so d_k = beta + u_k with beta shared (bias) and the u_k
independent across repetitions. In units of the noise variance (the n_eff denominator), averaging a maps:

    m(a, r) = 1 / n_eff = (1 - r)^2 (b + v / a) + r^2 / a + 2 r (1 - r) c / a

with b = bias^2, v = per-map variance of the Tweedie error, c = its covariance with the noise. Linear in
(b, v, c): least squares per slice. Optimal blend r* = (P - R) / (P + Q - 2R), P = b + v/a, Q = 1/a,
R = c/a; r* -> 1 as a -> infinity (bias does not average out, so less shrinkage pays).

NOTE: for the split-half estimator averaged over ALL C(N, a) splits this form is an exact combinatorial
identity (mean over a-subsets of |subset mean|^2 = diagonal/a + cross-products * (1 - 1/a)), so the fit
error and the a<=3 -> a=5..9 "prediction" are ~0 by construction and are NOT evidence for the model. What is
substantive: the whole (a, r) surface of a slice is fixed by three numbers (b, v, c); r*(a) has a closed form;
b, v, c read as bias^2, variance and noise covariance only under the independence argument above.
Reported: fit error (identity check), r* vs the measured best r on the grid, n_eff at r*.

    python experiments/_gr_theory.py
"""
import sys, os, csv, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
os.chdir(ROOT)
OUT = "runs/gr/theory"
RATIOS = (0.0, 1 / 12, 1 / 6, 0.25, 1 / 3, 0.5, 1.0)
A_SIZES = (1, 2, 3, 5, 7, 9)
SOURCES = [("(A) L1, dataset 1", "runs/cv_r3/ratio_eval/per_slice_ratio.csv", 1),
           ("(D) MSE norm., dataset 1", "runs/cv_ablation/mse_norm/ratio_eval/per_slice_ratio.csv", 1),
           ("(A) L1 (dataset-1 model), dataset 2", "runs/gen/A/per_slice_ratio.csv", 2),
           ("(A) L1 (dataset-1 model), dataset 3", "runs/gen/A/per_slice_ratio.csv", 3),
           ("(A) L1 pooled, dataset 1", "runs/mx_cv/A/eval_it2000/per_slice_ratio.csv", 1),
           ("(A) L1 pooled, dataset 2", "runs/mx_cv/A/eval_it2000/per_slice_ratio.csv", 2)]


def design(a, r):
    return np.array([(1 - r) ** 2, (1 - r) ** 2 / a, 2 * r * (1 - r) / a]), r ** 2 / a


def fit(pts):
    X = np.array([design(a, r)[0] for a, r, _ in pts])
    y = np.array([m - design(a, r)[1] for a, r, m in pts])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def model(th, a, r):
    d, off = design(a, r)
    return float(d @ th + off)


def rstar(th, a):
    b, v, c = th
    P, Q, R = b + v / a, 1 / a, c / a
    return float(np.clip((P - R) / (P + Q - 2 * R), 0, 1))


def load(path, ds):
    S = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        if row["region"] != "signal" or int(row.get("dataset", 1)) != ds:
            continue
        S.setdefault(int(row["slice"]), []).extend(
            (int(row["a"]), r, 1 / float(row["neff_r%.4f" % r])) for r in RATIOS if r < 1.0)
    return S


os.makedirs(OUT, exist_ok=True)
summary, L = {}, []
gm = lambda v: float(np.exp(np.mean(np.log(v))))
for name, path, ds in SOURCES:
    S = load(path, ds)
    fit_err, pred_err, rows = [], [], {a: [] for a in A_SIZES}
    params = []
    for s, pts in sorted(S.items()):
        th = fit(pts)
        params.append(th)
        fit_err += [abs(model(th, a, r) / m - 1) for a, r, m in pts]
        th_lo = fit([p for p in pts if p[0] <= 3])
        pred_err += [abs(model(th_lo, a, r) / m - 1) for a, r, m in pts if a >= 5]
        for a in A_SIZES:
            meas = {r: 1 / m for aa, r, m in pts if aa == a}
            rows[a].append((rstar(th, a), rstar(th_lo, a), max(meas, key=meas.get),
                            1 / model(th, a, rstar(th, a)), max(meas.values()), meas))
    P = np.array(params)
    res = {"n_slices": len(S), "fit_mape": float(np.mean(fit_err)), "fit_p95": float(np.percentile(fit_err, 95)),
           "extrap_a5to9_mape": float(np.mean(pred_err)), "extrap_p95": float(np.percentile(pred_err, 95)),
           "params_median": dict(zip("bvc", map(float, np.median(P, 0)))), "by_a": {}}
    L.append(f"\n{name}  ({len(S)} slices)")
    L.append(f"  fit error |model/measured - 1|: mean {res['fit_mape']:.3f}, 95th pct {res['fit_p95']:.3f}")
    L.append(f"  fit on a<=3, predict a=5,7,9:   mean {res['extrap_a5to9_mape']:.3f}, 95th pct {res['extrap_p95']:.3f}")
    L.append("  median b (bias^2), v (variance), c (noise covariance), units of sigma^2: "
             + ", ".join("%s %.4f" % kv for kv in res["params_median"].items()))
    L.append("   a | r* (median, fit all) | r* (from a<=3) | best grid r (median) | n_eff gmean: at r* / best grid / r=1/3")
    for a in A_SIZES:
        R = rows[a]
        grid_best = [g for _, _, g, _, _, _ in R]
        at_lock = [m[1 / 3] for *_, m in R]
        res["by_a"][a] = dict(rstar_median=float(np.median([x[0] for x in R])),
                              rstar_from_a123_median=float(np.median([x[1] for x in R])),
                              grid_best_median=float(np.median(grid_best)),
                              neff_at_rstar=gm([x[3] for x in R]), neff_grid_best=gm([x[4] for x in R]),
                              neff_r13=gm(at_lock))
        b_ = res["by_a"][a]
        L.append("  %2d | %.3f | %.3f | %.3f | %.2f / %.2f / %.2f" % (
            a, b_["rstar_median"], b_["rstar_from_a123_median"], b_["grid_best_median"],
            b_["neff_at_rstar"], b_["neff_grid_best"], b_["neff_r13"]))
    summary[name] = res
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print("\n".join(L))
