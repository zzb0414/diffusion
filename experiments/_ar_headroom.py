# -*- coding: utf-8 -*-
"""Adaptive-r headroom (exploratory, read-only): how much n_eff does making the strength r adaptive buy?

For every slice (decoded by its held-out fold model) and averaging depth a, the split-half error of
x_hat = x0_hat + r (y - x0_hat), averaged over all C(N, a) splits, is an exact quadratic in r at every pixel:

    err_p(r) = (1-r)^2 A_p + r^2 B_p + 2 r (1-r) G_p - v_p/(N-a)

(A, B, G: mean squared Tweedie error, raw error and their cross term against the complementary-half reference;
v: unbiased per-pixel noise variance). Any rule that sets r per slice, per region or per pixel therefore has a
closed-form optimum, and its n_eff is s2 / (region mean of err), exactly as ``_ratio_core.evaluate_slice``
(uniform r reproduces it; checked).

Rules, signal region, n_eff geometric mean over slices:
  fixed_1/3     r = 1/3 (current abstract setting)
  global_cf     one r per a, chosen on the OTHER folds' slices (cross-fitted)          <- baseline
  slice_oracle  one r per slice and a, chosen on the slice itself (upper bound for per-slice adaptivity)
  signal_cf     r per (a, local-signal quintile), cross-fitted; quintiles of the smoothed mean magnitude
  edge_cf       r per (a, edge-strength quintile), cross-fitted; gradient magnitude of the smoothed mean
  block_oracle  r per 8x8 block, chosen in-sample (optimistic upper bound for spatial adaptivity)
Clipped to r in [0, 1]. Features use the mean of all repetitions (as at deployment); smoothing (sigma 2 px)
keeps leakage from the reference half small.

    python experiments/_ar_headroom.py
"""
import sys, os, json, itertools
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, sobel

ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from _r3_eval import load                                    # noqa: E402
from _data import load_datasets, cv_folds                    # noqa: E402
from _ratio_core import cmag                                 # noqa: E402

OUT = "runs/ar/headroom"
A_SIZES = (1, 2, 3, 5, 7, 9)
NB, BLK = 5, 8
MODELS = [("A (L1)", "runs/cv_r3", "l1"), ("D (MSE, noise-norm.)", "runs/cv_ablation/mse_norm", "mse")]
DATASETS = (1, 2, 3)


def rstar(A, B, G):
    den = A + B - 2 * G
    return float(np.clip((A - G) / den, 0, 1)) if den > 0 else 0.0


def quad(A, B, G, r):
    return (1 - r) ** 2 * A + r ** 2 * B + 2 * r * (1 - r) * G


def slice_stats(model, ens_t, sg, device):
    """Per a: pixel maps A, B, G (signal-region use), noise correction, masks and features."""
    N = ens_t.shape[0]
    ens = ens_t.double().cpu().numpy()
    H, W = ens.shape[-2:]
    mean10 = ens.mean(0)
    vpix = (((ens - mean10) ** 2).sum(0) / (N - 1)).mean(0)
    sm = gaussian_filter(cmag(mean10), 3.0)
    sig_mask = sm > np.median(sm)
    with torch.no_grad():
        x0 = model.denoise_to(ens_t, sg, target_ratio=0.0, device=device).double().cpu().numpy()
    f_sig = gaussian_filter(cmag(mean10), 2.0)
    f_edge = np.hypot(sobel(f_sig, 0), sobel(f_sig, 1))
    bins = {}
    for name, f in (("signal", f_sig), ("edge", f_edge)):
        q = np.quantile(f[sig_mask], np.linspace(0, 1, NB + 1)[1:-1])
        bins[name] = np.digitize(f, q)                      # 0..NB-1
    blocks = (np.arange(H)[:, None] // BLK) * ((W + BLK - 1) // BLK) + np.arange(W)[None, :] // BLK
    per_a = {}
    for a in A_SIZES:
        combos = list(itertools.combinations(range(N), a))
        WA = np.zeros((len(combos), N)); WB = np.zeros((len(combos), N))
        for i, c in enumerate(combos):
            WA[i, list(c)] = 1.0 / a
            WB[i, [j for j in range(N) if j not in c]] = 1.0 / (N - a)
        ref = (WB @ ens.reshape(N, -1)).reshape(-1, 2, H, W)
        e1 = (WA @ ens.reshape(N, -1)).reshape(-1, 2, H, W) - ref
        e0 = (WA @ x0.reshape(N, -1)).reshape(-1, 2, H, W) - ref
        per_a[a] = dict(A=(e0 ** 2).mean((0, 1)), B=(e1 ** 2).mean((0, 1)), G=(e0 * e1).mean((0, 1)),
                        corr=float(vpix[sig_mask].mean()) / (N - a))
    return dict(per_a=per_a, mask=sig_mask, s2=float(vpix[sig_mask].mean()), bins=bins, blocks=blocks)


def neff(st, a, rmap):
    p = st["per_a"][a]; m = st["mask"]
    mse = float(quad(p["A"], p["B"], p["G"], rmap)[m].mean() - p["corr"])
    return st["s2"] / mse if mse > 0 else float("inf")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds = cv_folds()
    x, meta, sig = load_datasets(list(DATASETS))
    os.makedirs(OUT, exist_ok=True)
    summary, lines = {}, []
    for mname, run, loss in MODELS:
        ST = {}
        for k in range(5):
            model = load(os.path.join(run, f"fold{k}", "ckpt", f"{loss}_002000.pkl"), device)[0]
            for i, (d, s) in enumerate(meta):
                if folds[s] == k:
                    ST[(d, s)] = slice_stats(model, x[i].to(device), float(sig[i]), device)
            del model; torch.cuda.empty_cache()
        # sanity: uniform r reproduces the logged n_eff (dataset 1, r = 1/3)
        if run == "runs/cv_r3":
            import csv
            ref = {(int(r["slice"]), int(r["a"])): float(r["neff_r0.3333"]) for r in
                   csv.DictReader(open("runs/cv_r3/ratio_eval/per_slice_ratio.csv", encoding="utf-8"))
                   if r["region"] == "signal"}
            worst = max(abs(neff(ST[(1, s)], a, np.full((80, 80), 1 / 3)) / ref[(s, a)] - 1)
                        for s in range(34) for a in A_SIZES)
            assert worst < 1e-9, worst
            lines.append(f"(check: uniform r = 1/3 reproduces the logged dataset-1 n_eff, worst rel. diff {worst:.1e})")

        def agg(keys, a, sel=None):
            """normalized sums of A, B, G over signal pixels (optionally a pixel selector per slice)."""
            t = np.zeros(3)
            for key in keys:
                st = ST[key]; p = st["per_a"][a]
                m = st["mask"] if sel is None else st["mask"] & sel(st)
                t += np.array([p["A"][m].sum(), p["B"][m].sum(), p["G"][m].sum()]) / (st["s2"] * st["mask"].sum())
            return t

        res = {}
        for d in DATASETS:
            keys_d = [(d, s) for s in range(34)]
            for a in A_SIZES:
                R = {n: [] for n in ("fixed_1/3", "global_cf", "slice_oracle", "signal_cf", "edge_cf", "block_oracle")}
                rs_used = {"global_cf": [], "slice_oracle": []}
                for key in keys_d:
                    st = ST[key]; f = folds[key[1]]
                    train = [kk for kk in keys_d if folds[kk[1]] != f]
                    R["fixed_1/3"].append(neff(st, a, np.full((80, 80), 1 / 3)))
                    rg = rstar(*agg(train, a)); rs_used["global_cf"].append(rg)
                    R["global_cf"].append(neff(st, a, np.full((80, 80), rg)))
                    p = st["per_a"][a]; m = st["mask"]
                    ro = rstar(p["A"][m].sum(), p["B"][m].sum(), p["G"][m].sum()); rs_used["slice_oracle"].append(ro)
                    R["slice_oracle"].append(neff(st, a, np.full((80, 80), ro)))
                    for feat in ("signal", "edge"):
                        rmap = np.zeros((80, 80))
                        for b in range(NB):
                            rb = rstar(*agg(train, a, lambda t, b=b, feat=feat: t["bins"][feat] == b))
                            rmap[st["bins"][feat] == b] = rb
                        R[f"{feat}_cf"].append(neff(st, a, rmap))
                    rmap = np.zeros((80, 80))
                    for b in np.unique(st["blocks"]):
                        sel = (st["blocks"] == b) & m
                        if sel.any():
                            rmap[st["blocks"] == b] = rstar(p["A"][sel].sum(), p["B"][sel].sum(), p["G"][sel].sum())
                    R["block_oracle"].append(neff(st, a, rmap))
                base = np.log(R["global_cf"])
                res[(d, a)] = {n: dict(gmean=float(np.exp(np.mean(np.log(v)))),
                                       vs_global=float(np.exp(np.mean(np.log(v) - base))),
                                       wins=int((np.log(v) > base + 1e-12).sum()))
                               for n, v in R.items()}
                res[(d, a)]["r_global_median"] = float(np.median(rs_used["global_cf"]))
                res[(d, a)]["r_slice_iqr"] = [float(q) for q in np.percentile(rs_used["slice_oracle"], (25, 50, 75))]
        summary[mname] = {f"{d}|{a}": v for (d, a), v in res.items()}
        lines.append(f"\n{mname}: n_eff geometric mean (ratio to global_cf), signal region")
        for d in DATASETS:
            lines.append(f"  dataset {d}")
            lines.append("    a | global r | slice r* IQR        | fixed 1/3      | global_cf | slice_oracle   | signal_cf      | edge_cf        | block_oracle")
            for a in A_SIZES:
                v = res[(d, a)]
                cell = lambda n: "%6.2f (%.3f)" % (v[n]["gmean"], v[n]["vs_global"])
                lines.append("    %d | %.3f    | %.3f %.3f %.3f | %s | %6.2f    | %s | %s | %s | %s" % (
                    a, v["r_global_median"], *v["r_slice_iqr"], cell("fixed_1/3"), v["global_cf"]["gmean"],
                    cell("slice_oracle"), cell("signal_cf"), cell("edge_cf"), cell("block_oracle")))
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
