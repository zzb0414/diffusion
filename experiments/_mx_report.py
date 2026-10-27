# -*- coding: utf-8 -*-
"""MX_CV report: pooled (datasets 1+2+3) vs dataset1-only training, per dataset.

Implements ``Archive/MX_CV/PREREGISTRATION.md``. Pooled results come from
``runs/mx_cv/<arm>/eval_it{2000,6000}``; the dataset1-only comparator from
``<arm dir>/ratio_eval`` (dataset 1, CV_RATIO) and ``runs/gen/<arm>``
(datasets 2-3, GEN). Every numeral is computed here, every verdict follows the
pre-registered rule. ``--dry`` substitutes the comparator for the pooled
results (every ratio must read 1.000) and writes to a scratch dir, no log.

    python experiments/_mx_report.py [--dry]
"""
import sys, os, csv, json, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from _data import load_datasets                              # noqa: E402
from _figures_window import WIN                              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dry", action="store_true")
DRY = ap.parse_args().dry
OUT = "runs/_mx_dry" if DRY else "runs/mx_cv/report"
MD = "WEEKEND_ITERATIONS.md"
HEAD = "## RUN 2026-10-27 — MX_CV results"
ARMS = [("A", "L1", "runs/cv_r3"), ("B", "L1, noise-normalized", "runs/cv_ablation/l1_norm"),
        ("C", "MSE", "runs/cv_ablation/mse"), ("D", "MSE, noise-normalized", "runs/cv_ablation/mse_norm")]
COL = {"A": "C3", "B": "C4", "C": "C1", "D": "C2"}
RATIOS = (0.0, 1 / 12, 1 / 6, 0.25, 1 / 3, 0.5, 1.0)
A_SIZES = (1, 2, 3, 5, 7, 9)
DS = (1, 2, 3)
LOCK = 1 / 3
KL = "neff_r%.4f" % LOCK
NBOOT = 10000
S34 = list(range(34))
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "legend.fontsize": 11,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "mathtext.fontset": "cm"})
rng = np.random.default_rng(0)


def read(path, ds_col):
    R = list(csv.DictReader(open(os.path.join(path, "per_slice_ratio.csv"), encoding="utf-8")))
    ne, pr = {}, {}
    for r in R:
        d = int(r["dataset"]) if ds_col else 1
        s = int(r["slice"])
        if r["region"] in ("signal", "all"):
            ne[(d, s, int(r["a"]), r["region"])] = {k: float(v) for k, v in r.items() if v != "" and k != "region"}
        elif r["region"] == "per_r":
            pr[(d, s, round(float(r["ratio"]), 4))] = (float(r["noise"]), float(r["gain10"]))
    B = {}
    for r in csv.DictReader(open(os.path.join(path, "bands_ratio.csv"), encoding="utf-8")):
        d = int(r["dataset"]) if ds_col else 1
        B[(d, int(r["slice"]), int(r["a"]))] = {k: float(v) for k, v in r.items()}
    return ne, pr, B


def merge(*parts):
    out = ({}, {}, {})
    for p in parts:
        for i in range(3):
            out[i].update(p[i])
    return out


# comparator: dataset1-only models (ds1 from CV_RATIO, ds2/3 from GEN)
ONE = {k: merge(read(os.path.join(d, "ratio_eval"), False),
                tuple({kk: vv for kk, vv in t.items() if kk[0] in (2, 3)} for t in read(os.path.join("runs/gen", k), True)))
       for k, _, d in ARMS}
POOL = {it: ({k: ONE[k] for k, _, _ in ARMS} if DRY else
             {k: read(os.path.join("runs/mx_cv", k, f"eval_it{it}"), True) for k, _, _ in ARMS})
        for it in (2000, 6000)}

ne = lambda src, k, d, s, a, c=KL, rg="signal": src[k][0][(d, s, a, rg)][c]
gm = lambda v: float(np.exp(np.mean(np.log(v))))


def stat(vals, alt):
    lr = np.log(np.asarray(vals, float))
    b = rng.choice(lr, size=(NBOOT, len(lr))).mean(1)
    p = float(wilcoxon(lr, alternative=alt).pvalue) if np.any(lr != 0) else 1.0
    return dict(gmean=float(np.exp(lr.mean())), ci_lo=float(np.exp(np.percentile(b, 2.5))),
                ci_hi=float(np.exp(np.percentile(b, 97.5))), wins=int((lr > 0).sum()), n=len(lr), p=p)


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


def verdict(r, q):
    return ("pooled better" if q < 0.05 and r["gmean"] > 1 else
            "pooled worse" if q < 0.05 and r["gmean"] < 1 else "no significant difference")


P2 = POOL[2000]
# ---- primary: arm A, it 2000, r = 1/3, a = 1, signal: pooled / dataset1-only per dataset -----------
prim = {d: stat([ne(P2, "A", d, s, 1) / ne(ONE, "A", d, s, 1) for s in S34], "two-sided") for d in DS}
for d, q in zip(DS, holm([prim[d]["p"] for d in DS])):
    prim[d]["p_holm"] = float(q); prim[d]["verdict"] = verdict(prim[d], q)

# ---- secondary -------------------------------------------------------------------------------------------
vsone = {(k, d, a): stat([ne(P2, k, d, s, a) / ne(ONE, k, d, s, a) for s in S34], "two-sided")
         for k, _, _ in ARMS for d in DS for a in (1, 5, 9)}
base = {(k, d, a, b): stat([ne(P2, k, d, s, a) / (a if b == "plain" else ne(P2, k, d, s, a, c=b)) for s in S34], "greater")
        for k, _, _ in ARMS for d in DS for a in (1, 5, 9) for b in ("plain", "blur_matched")}
loss = {}
for d in DS:
    r = {k: stat([ne(P2, k, d, s, 1) / ne(P2, "A", d, s, 1) for s in S34], "two-sided") for k in "BCD"}
    for k, q in zip("BCD", holm([r[k]["p"] for k in "BCD"])):
        r[k]["p_holm"] = float(q)
    loss[d] = r
late = {(k, d): stat([ne(POOL[6000], k, d, s, 1) / ne(P2, k, d, s, 1) for s in S34], "two-sided")
        for k, _, _ in ARMS for d in DS}
surf = {(k, d, r, a): gm([ne(P2, k, d, s, a, c="neff_r%.4f" % r) for s in S34])
        for k, _, _ in ARMS for d in DS for r in RATIOS for a in A_SIZES}
calib = {(k, d): [P2[k][1][(d, s, round(LOCK, 4))][0] for s in S34] for k, _, _ in ARMS for d in DS}


def bands(k, d, a=1):
    out = []
    for b in range(4):
        x = np.array([P2[k][2][(d, s, a)]["r%.4f_band%d" % (LOCK, b)] for s in S34])
        y = np.array([P2[k][2][(d, s, a)]["blur_matched_band%d" % b] for s in S34])
        fin = np.isfinite(x) & np.isfinite(y)
        out.append((float(np.median(x[fin] / y[fin])), int(((x > y) & ~(np.isinf(x) & np.isinf(y))).sum())))
    return out


bnd = {(k, d): bands(k, d) for k, _, _ in ARMS for d in DS}
if DRY:
    assert all(abs(prim[d]["gmean"] - 1) < 1e-12 for d in DS)
    assert all(abs(v["gmean"] - 1) < 1e-12 for v in vsone.values())
    print("DRY RUN OK: pooled := dataset1-only gives ratio 1.000 everywhere")
    sys.exit(0)

# ---- figures (no titles; not committed) -----------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
for ax, d in zip(axs, DS):
    for k, n, _ in ARMS:
        ax.plot(A_SIZES, [surf[(k, d, LOCK, a)] for a in A_SIZES], "o-", lw=1.2, ms=4, color=COL[k],
                label=f"({k}) {n}, pooled")
        ax.plot(A_SIZES, [gm([ne(ONE, k, d, s, a) for s in S34]) for a in A_SIZES], "o--", lw=0.9, ms=3,
                mfc="none", color=COL[k], label=f"({k}) dataset 1 only")
    ax.plot(A_SIZES, A_SIZES, "k:", lw=1.0, label="Plain averaging")
    ax.set_yscale("log"); ax.set_xticks(A_SIZES); ax.set_xlabel(r"Repetitions averaged, $a$")
    ax.text(0.03, 0.97, f"Dataset {d}", transform=ax.transAxes, va="top", fontsize=14)
    ax.grid(alpha=0.3, which="both")
axs[0].set_ylabel(r"$n_{\mathrm{eff}}$ at $r = 1/3$")
axs[-1].legend(loc="lower right", fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "mx_neff_vs_a.png"), dpi=200); plt.close(fig)

X, meta, S = load_datasets(list(DS))
X = X.double().numpy()
sig = {m: float(S[i]) for i, m in enumerate(meta)}
mag = lambda z: np.sqrt(z[..., 0, :, :] ** 2 + z[..., 1, :, :] ** 2)
S2 = 16
fig = plt.figure(figsize=(17.3, 3.0 * 2 * 3 + 0.6))
gs = GridSpec(8, 7, figure=fig, height_ratios=[1, 1, 0.18, 1, 1, 0.18, 1, 1],
              width_ratios=[1, 1, 0.2, 1, 1, 1, 1], hspace=0.06, wspace=0.04)
heads = ["Noisy", "Mean of 10"] + [f"({k}) {n}".replace(", noise-normalized", ",\nnoise-normalized") for k, n, _ in ARMS]


def off(a):
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)


for i, d in enumerate(DS):
    r0 = 3 * i
    ens = X[meta.index((d, S2))]; m10 = ens.mean(0); vmax = WIN * float(np.percentile(mag(m10), 99.5))
    for j, img in enumerate((mag(ens[0]), mag(m10))):
        a = fig.add_subplot(gs[r0:r0 + 2, j]); a.imshow(img.T, cmap="gray", vmin=0, vmax=vmax); off(a)
        if i == 0:
            a.set_title(heads[j], fontsize=15)
        if j == 0:
            a.set_ylabel(f"Dataset {d}\n" + r"$\hat\sigma = %.3f$" % sig[(d, S2)], fontsize=14)
    for j, (k, n, _) in enumerate(ARMS):
        x0 = np.load(os.path.join("runs/mx_cv", k, "eval_it2000", f"x0_d{d}_s{S2:02d}.npy")).astype(np.float64)
        out = x0 + LOCK * (ens - x0)
        for sub, img in enumerate((mag(out[0]), mag(out.mean(0)))):
            a = fig.add_subplot(gs[r0 + sub, 3 + j]); a.imshow(img.T, cmap="gray", vmin=0, vmax=vmax); off(a)
            if i == 0 and sub == 0:
                a.set_title(heads[2 + j], fontsize=14)
            if j == 0:
                a.set_ylabel("Single" if sub == 0 else "Mean of 10", fontsize=14)
fig.savefig(os.path.join(OUT, "mx_maps_s16.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- summary + log ---------------------------------------------------------------------------------------------
K = lambda t: "|".join(str(x) for x in t)
json.dump(dict(prim=prim, vsone={K(k): v for k, v in vsone.items()}, base={K(k): v for k, v in base.items()},
               loss={str(d): v for d, v in loss.items()}, late={K(k): v for k, v in late.items()},
               surf={"%s|%d|%.4f|%d" % k: v for k, v in surf.items()},
               calib={K(k): [float(np.median(v)), *map(float, np.percentile(v, (25, 75)))] for k, v in calib.items()},
               bands={K(k): v for k, v in bnd.items()}), open(os.path.join(OUT, "summary.json"), "w"), indent=2)
f3 = lambda r: "%.3f× (95%% CI %.3f–%.3f; %d/%d)" % (r["gmean"], r["ci_lo"], r["ci_hi"], r["wins"], r["n"])
L = ["", "---", "", HEAD, "",
     "Pre-registered (`Archive/MX_CV/PREREGISTRATION.md`). 20 pooled models (arms A–D × 5 slice-index folds, "
     "datasets 1+2+3, iteration 2000 primary); every slice decoded by the fold that held its index out. "
     "Comparator: the dataset1-only models of the same arm (CV_RATIO for dataset 1, GEN for datasets 2–3). "
     "All 8 evaluations passed their self-checks; the report's dry run (pooled := dataset1-only) gave 1.000.", "",
     "**PRIMARY (arm A, r = 1/3, a = 1, signal; n_eff pooled / dataset1-only):**", ""]
for d in DS:
    L.append("- Dataset %d: %s, p = %.2g, Holm %.2g → **%s**" % (d, f3(prim[d]), prim[d]["p"], prim[d]["p_holm"], prim[d]["verdict"]))
L += ["", "Pooled / dataset1-only for every arm (r = 1/3; geometric mean, slices better, two-sided p):", "",
      "| loss | dataset | a = 1 | a = 5 | a = 9 |", "|---|---:|---|---|---|"]
for k, n, _ in ARMS:
    for d in DS:
        L.append("| (%s) %s | %d | %s |" % (k, n, d, " | ".join(
            "%.3f× (%d/34, p=%.1g)" % (vsone[(k, d, a)]["gmean"], vsone[(k, d, a)]["wins"], vsone[(k, d, a)]["p"]) for a in (1, 5, 9))))
L += ["", "Pooled models at r = 1/3 against the baselines (×plain, ×noise-matched blur with slices won and one-sided p):", "",
      "| loss | dataset | a = 1 | a = 5 | a = 9 |", "|---|---:|---|---|---|"]
for k, n, _ in ARMS:
    for d in DS:
        L.append("| (%s) %s | %d | %s |" % (k, n, d, " | ".join(
            "×%.2f plain; ×%.3f blur (%d/34, p=%.1g)" % (base[(k, d, a, "plain")]["gmean"], base[(k, d, a, "blur_matched")]["gmean"],
                                                         base[(k, d, a, "blur_matched")]["wins"], base[(k, d, a, "blur_matched")]["p"])
            for a in (1, 5, 9))))
L += ["", "Losses within the pooled models (a = 1, r = 1/3; ratio vs A, Holm over B/C/D per dataset): " + "; ".join(
      "ds%d: " % d + ", ".join("%s %.3f× (Holm %.2g)" % (k, loss[d][k]["gmean"], loss[d][k]["p_holm"]) for k in "BCD") for d in DS) + ".",
      "", "Iteration 6000 / 2000 (a = 1, r = 1/3): " + "; ".join(
      "(%s) " % k + ", ".join("ds%d %.3f× (%d/34)" % (d, late[(k, d)]["gmean"], late[(k, d)]["wins"]) for d in DS) for k, _, _ in ARMS) + ".",
      "", "Calibration σ̂_out/σ̂ at r = 1/3 (median): " + "; ".join(
      "(%s) " % k + ", ".join("ds%d %.3f" % (d, np.median(calib[(k, d)])) for d in DS) for k, _, _ in ARMS) + ".",
      "", "Per band vs noise-matched blur, a = 1 (median ratio; slices won), lowest → finest:", ""]
for k, n, _ in ARMS:
    L.append("- (%s) %s: " % (k, n) + "; ".join("ds%d: %s" % (d, ", ".join("%.2f (%d)" % b for b in bnd[(k, d)])) for d in DS))
L += ["", "**Figures** (`%s/`, not committed): `mx_neff_vs_a.png`, `mx_maps_s16.png`. **Data:** "
      "`runs/mx_cv/<arm>/eval_it{2000,6000}/`, `%s/summary.json`. **Commands:** `python experiments/_mx_train.py --arm X "
      "--fold K`; `python experiments/_mx_eval.py --run runs/mx_cv/X --iter {2000,6000} --datasets 1 2 3`; "
      "`python experiments/_mx_report.py`." % (OUT, OUT), ""]
if HEAD not in open(MD, encoding="utf-8").read():
    open(MD, "a", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
