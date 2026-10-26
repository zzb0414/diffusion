# -*- coding: utf-8 -*-
"""GEN report: dataset1-only models on the unseen datasets 2 and 3.

Implements ``Archive/MX_CV/PREREGISTRATION_GEN.md``. Reads ``runs/gen/<arm>``
(datasets 2/3; arm A also dataset 1) and the dataset-1 CV_RATIO results of each
arm; every numeral is computed here and every verdict follows the
pre-registered rule. Also performs verification 4 (GEN's dataset-1 rows for arm
A must equal the CV_RATIO numbers).

Outputs ``runs/gen/report/`` (summary.json, figures without titles, not
committed) and an appended WEEKEND_ITERATIONS.md section (once).

    python experiments/_gen_report.py
"""
import sys, os, csv, json
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

OUT = "runs/gen/report"
MD = "WEEKEND_ITERATIONS.md"
HEAD = "## RUN 2026-10-26 — GEN results"
ARMS = [("A", "L1", "runs/cv_r3"), ("B", "L1, noise-normalized", "runs/cv_ablation/l1_norm"),
        ("C", "MSE", "runs/cv_ablation/mse"), ("D", "MSE, noise-normalized", "runs/cv_ablation/mse_norm")]
NAME = {k: n for k, n, _ in ARMS}
COL = {"A": "C3", "B": "C4", "C": "C1", "D": "C2"}
RATIOS = (0.0, 1 / 12, 1 / 6, 0.25, 1 / 3, 0.5, 1.0)
A_SIZES = (1, 2, 3, 5, 7, 9)
LOCK = 1 / 3
KL = "neff_r%.4f" % LOCK
NBOOT = 10000
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "legend.fontsize": 12,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "mathtext.fontset": "cm"})
rng = np.random.default_rng(0)


def read(path, ds_col):
    """{(dataset, slice, a, region): row} and {(dataset, slice, r): (noise, gain)} and below-ladder."""
    R = list(csv.DictReader(open(os.path.join(path, "per_slice_ratio.csv"), encoding="utf-8")))
    ne, pr, bl = {}, {}, {}
    for r in R:
        d = int(r["dataset"]) if ds_col else 1
        s = int(r["slice"])
        if r["region"] in ("signal", "all"):
            ne[(d, s, int(r["a"]), r["region"])] = {k: float(v) for k, v in r.items() if v != "" and k != "region"}
        elif r["region"] == "per_r":
            pr[(d, s, round(float(r["ratio"]), 4))] = (float(r["noise"]), float(r["gain10"]))
        if ds_col:
            bl[(d, s)] = int(r["below_ladder"])
    B = {}
    for r in csv.DictReader(open(os.path.join(path, "bands_ratio.csv"), encoding="utf-8")):
        d = int(r["dataset"]) if ds_col else 1
        B[(d, int(r["slice"]), int(r["a"]))] = {k: float(v) for k, v in r.items()}
    return ne, pr, bl, B


G = {k: read(os.path.join("runs/gen", k), True) for k, _, _ in ARMS}         # datasets 2, 3 (+1 for A)
R1 = {k: read(os.path.join(d, "ratio_eval"), False) for k, _, d in ARMS}    # dataset 1 (CV_RATIO)
slices = list(range(34))

# ---- verification 4: GEN's dataset-1 rows for arm A == CV_RATIO's ---------------------------------
n4 = 0
for (d, s, a, rg), row in G["A"][0].items():
    if d != 1:
        continue
    ref = R1["A"][0][(1, s, a, rg)]
    for c in [c for c in ref if c.startswith(("neff_r", "blur_"))]:
        assert np.isclose(row[c], ref[c], rtol=1e-9), (s, a, rg, c); n4 += 1
assert n4 > 0
print("verification 4: GEN dataset-1 rows equal CV_RATIO for arm A (%d values)" % n4)


def ne(k, d, s, a, rg="signal", c=KL):
    src = R1 if d == 1 else G
    return src[k][0][(d, s, a, rg)][c]


def stat(vals, alt):
    lr = np.log(np.asarray(vals))
    b = rng.choice(lr, size=(NBOOT, len(lr))).mean(1)
    p = float(wilcoxon(lr, alternative=alt).pvalue) if np.any(lr != 0) else 1.0
    return dict(gmean=float(np.exp(lr.mean())), ci_lo=float(np.exp(np.percentile(b, 2.5))),
                ci_hi=float(np.exp(np.percentile(b, 97.5))), wins=int((lr > 0).sum()), n=len(lr), p=p)


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


# ---- primary: arm A, r = 1/3, a = 1, signal; ds2 and ds3 ------------------------------------------
prim = {}
for comp in ("plain", "blur_matched"):
    res = {d: stat([ne("A", d, s, 1) / (1.0 if comp == "plain" else ne("A", d, s, 1, c="blur_matched"))
                    for s in slices], "greater") for d in (2, 3)}
    for d, q in zip((2, 3), holm([res[2]["p"], res[3]["p"]])):
        res[d]["p_holm"] = float(q)
        res[d]["verdict"] = "supported" if q < 0.05 and res[d]["gmean"] > 1 else "not supported"
    prim[comp] = res

# ---- calibration, secondary ----------------------------------------------------------------------
def noise(k, d, s, r=LOCK):
    src = R1 if d == 1 else G
    return src[k][1][(d, s, round(r, 4))][0]


calib = {(k, d): [noise(k, d, s) for s in slices] for k, _, _ in ARMS for d in (1, 2, 3)}
below = {d: sorted(s for s in slices if G["A"][2].get((d, s), 0)) for d in (2, 3)}
gm = lambda v: float(np.exp(np.mean(np.log(v))))
surf = {(k, d, r, a): gm([ne(k, d, s, a, c="neff_r%.4f" % r) for s in slices])
        for k, _, _ in ARMS for d in (1, 2, 3) for r in RATIOS for a in A_SIZES}
sec = {(k, d, a, b): stat([ne(k, d, s, a) / (a if b == "plain" else ne(k, d, s, a, c=b)) for s in slices], "greater")
       for k, _, _ in ARMS for d in (1, 2, 3) for a in (1, 5, 9) for b in ("plain", "blur_matched")}


def bands(k, d, a=1):
    src = R1 if d == 1 else G
    out = []
    for b in range(4):
        x = np.array([src[k][3][(d, s, a)]["r%.4f_band%d" % (LOCK, b)] for s in slices])
        y = np.array([src[k][3][(d, s, a)]["blur_matched_band%d" % b] for s in slices])
        fin = np.isfinite(x) & np.isfinite(y)
        out.append((float(np.median(x[fin] / y[fin])), int(((x > y) & ~(np.isinf(x) & np.isinf(y))).sum())))
    return out


bnd = {(k, d): bands(k, d) for k, _, _ in ARMS for d in (1, 2, 3)}
sig = {}
_, meta, S = load_datasets([1, 2, 3])
for i, (d, s) in enumerate(meta):
    sig[(d, s)] = float(S[i])

# ---- figures (no titles; not committed) ---------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
for ax, d in zip(axs, (1, 2, 3)):
    for k, n, _ in ARMS:
        ax.plot(A_SIZES, [surf[(k, d, LOCK, a)] for a in A_SIZES], "o-", lw=1.2, ms=4, color=COL[k], label=f"({k}) {n}")
    ax.plot(A_SIZES, [gm([ne("A", d, s, a, c="blur_matched") for s in slices]) for a in A_SIZES], "d--", lw=1.0,
            color="C0", label="Noise-matched blur (A)")
    ax.plot(A_SIZES, A_SIZES, "k:", lw=1.0, label="Plain averaging")
    ax.set_yscale("log"); ax.set_xticks(A_SIZES); ax.set_xlabel(r"Repetitions averaged, $a$")
    ax.text(0.03, 0.97, f"Dataset {d}" + ("" if d == 1 else " (unseen)"), transform=ax.transAxes, va="top", fontsize=14)
    ax.grid(alpha=0.3, which="both")
axs[0].set_ylabel(r"$n_{\mathrm{eff}}$ at $r = 1/3$"); axs[-1].legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "gen_neff_vs_a.png"), dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 5.4))
pos = 0
for d in (1, 2, 3):
    for k, n, _ in ARMS:
        v = calib[(k, d)]
        ax.boxplot([v], positions=[pos], widths=0.6, patch_artist=True, showfliers=True,
                   boxprops=dict(facecolor=COL[k], alpha=0.4))
        pos += 1
    pos += 1
ax.axhline(LOCK, color="k", ls=":", lw=1.0)
ax.set_xticks([1.5, 6.5, 11.5]); ax.set_xticklabels(["Dataset 1", "Dataset 2\n(unseen)", "Dataset 3\n(unseen)"])
ax.set_ylabel(r"$\hat\sigma_{\mathrm{out}} / \hat\sigma$ at $r = 1/3$")
for k, n, _ in ARMS:
    ax.plot([], [], color=COL[k], lw=8, alpha=0.4, label=f"({k}) {n}")
ax.legend(loc="upper left", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "gen_calibration.png"), dpi=200); plt.close(fig)

X, meta, _ = load_datasets([1, 2, 3])
X = X.double().numpy()
mag = lambda z: np.sqrt(z[..., 0, :, :] ** 2 + z[..., 1, :, :] ** 2)
S2 = 16
fig = plt.figure(figsize=(17.3, 3.0 * 2 * 3 + 0.6))
hr = [1, 1, 0.18, 1, 1, 0.18, 1, 1]
gs = GridSpec(len(hr), 7, figure=fig, height_ratios=hr, width_ratios=[1, 1, 0.2, 1, 1, 1, 1], hspace=0.06, wspace=0.04)
heads = ["Noisy", "Mean of 10"] + [f"({k}) {n}".replace(", noise-normalized", ",\nnoise-normalized") for k, n, _ in ARMS]


def off(a):
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)


for i, d in enumerate((1, 2, 3)):
    r0 = 3 * i
    ens = X[meta.index((d, S2))]; m10 = ens.mean(0); vmax = WIN * float(np.percentile(mag(m10), 99.5))
    for j, img in enumerate((mag(ens[0]), mag(m10))):
        a = fig.add_subplot(gs[r0:r0 + 2, j]); a.imshow(img.T, cmap="gray", vmin=0, vmax=vmax); off(a)
        if i == 0:
            a.set_title(heads[j], fontsize=15)
        if j == 0:
            a.set_ylabel(f"Dataset {d}" + ("" if d == 1 else " (unseen)") + "\n" + r"$\hat\sigma = %.3f$" % sig[(d, S2)], fontsize=14)
    for j, (k, n, dd) in enumerate(ARMS):
        x0p = (os.path.join(dd, "ratio_eval", f"x0_s{S2:02d}.npy") if d == 1 else
               os.path.join("runs/gen", k, f"x0_d{d}_s{S2:02d}.npy"))
        x0 = np.load(x0p).astype(np.float64)
        out = x0 + LOCK * (ens - x0)
        for sub, img in enumerate((mag(out[0]), mag(out.mean(0)))):
            a = fig.add_subplot(gs[r0 + sub, 3 + j]); a.imshow(img.T, cmap="gray", vmin=0, vmax=vmax); off(a)
            if i == 0 and sub == 0:
                a.set_title(heads[2 + j], fontsize=14)
            if j == 0:
                a.set_ylabel("Single" if sub == 0 else "Mean of 10", fontsize=14)
fig.savefig(os.path.join(OUT, "gen_maps_s16.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- summary + log ----------------------------------------------------------------------------------------
json.dump(dict(prim=prim, calib={"%s|%d" % k: [float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))]
                                 for k, v in calib.items()},
               below=below, surf={"%s|%d|%.4f|%d" % k: v for k, v in surf.items()},
               sec={"%s|%d|%d|%s" % k: v for k, v in sec.items()}, bands={"%s|%d" % k: v for k, v in bnd.items()}),
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
f3 = lambda r: "%.3f× (95%% CI %.3f–%.3f; %d/%d; p = %.1g, Holm %.1g)" % (
    r["gmean"], r["ci_lo"], r["ci_hi"], r["wins"], r["n"], r["p"], r["p_holm"])
cal = lambda k, d: "%.3f (IQR %.3f–%.3f)" % (np.median(calib[(k, d)]), *np.percentile(calib[(k, d)], (25, 75)))
L = ["", "---", "", HEAD, "",
     "Pre-registered (`Archive/MX_CV/PREREGISTRATION_GEN.md`). The dataset1-only fold models (iteration 2000) "
     "decode every slice of datasets 2 and 3 — a different receiver coil, never seen in training — each by the "
     "fold that held its slice index out, so data and anatomy are unseen. Verification 4 passed: GEN's "
     "dataset-1 rows for arm A equal the CV_RATIO numbers (%d values)." % n4, "",
     "**PRIMARY (arm A, r = 1/3, a = 1, signal):**", ""]
for comp, lab in (("plain", "vs plain averaging (= n_eff of one repetition)"), ("blur_matched", "vs noise-matched blur")):
    for d in (2, 3):
        L.append("- Dataset %d, %s: %s → **%s**" % (d, lab, f3(prim[comp][d]), prim[comp][d]["verdict"]))
L += ["", "**Calibration** σ̂_out/σ̂ at r = 1/3 (median, IQR):", "",
      "| loss | dataset 1 | dataset 2 (unseen) | dataset 3 (unseen) |", "|---|---|---|---|"]
for k, n, _ in ARMS:
    L.append("| (%s) %s | %s | %s | %s |" % (k, n, cal(k, 1), cal(k, 2), cal(k, 3)))
L += ["", "Slices below the dataset1 models' ladder (σ̂ < 0.0368, extrapolated): " + "; ".join(
      "dataset %d: %s" % (d, ", ".join(map(str, below[d])) or "none") for d in (2, 3)) + ".", "",
      "**n_eff at r = 1/3 per dataset** (geometric mean over 34 slices; ×plain, ×noise-matched blur, slices won):", "",
      "| loss | dataset | a = 1 | a = 5 | a = 9 |", "|---|---:|---|---|---|"]
for k, n, _ in ARMS:
    for d in (1, 2, 3):
        L.append("| (%s) %s | %d | %s |" % (k, n, d, " | ".join(
            "%.2f (×%.2f plain, ×%.3f blur, %d/34)" % (surf[(k, d, LOCK, a)], sec[(k, d, a, "plain")]["gmean"],
                                                       sec[(k, d, a, "blur_matched")]["gmean"],
                                                       sec[(k, d, a, "blur_matched")]["wins"]) for a in (1, 5, 9))))
L += ["", "Per band vs noise-matched blur, a = 1 (median ratio; slices won), lowest → finest:", ""]
for k, n, _ in ARMS:
    L.append("- (%s) %s: " % (k, n) + "; ".join("ds%d: %s" % (d, ", ".join("%.2f (%d)" % b for b in bnd[(k, d)]))
                                               for d in (1, 2, 3)))
L += ["", "**Figures** (`%s/`, not committed): `gen_neff_vs_a.png`, `gen_calibration.png`, `gen_maps_s16.png`. "
      "**Data:** `runs/gen/<arm>/*.csv`, `%s/summary.json`. **Commands:** `python experiments/_mx_eval.py --run "
      "<dataset1-only arm> --datasets 2 3 --out runs/gen/<arm>` (A: `--datasets 1 2 3`); "
      "`python experiments/_gen_report.py`." % (OUT, OUT), ""]
if HEAD not in open(MD, encoding="utf-8").read():
    open(MD, "a", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
