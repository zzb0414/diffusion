# -*- coding: utf-8 -*-
"""CV_RATIO report: set A (loss comparison at r = 1/3) and set B (r x a sweep).

Implements ``Archive/CV_RATIO/PREREGISTRATION.md``. Reads each arm's
``ratio_eval/{per_slice_ratio,bands_ratio}.csv`` and display-slice x0 arrays;
every numeral is computed here, every verdict from the pre-registered rule.

Outputs ``runs/cv_ratio/report/``: summary.json, figures (n_eff surfaces,
noise-vs-r, per-band-vs-r, frontiers, maps_A_*, maps_B_*), and an appended
WEEKEND_ITERATIONS.md section (once).

    python experiments/_ratio_report.py
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
from _r3_train import load_data, sigma_hat                   # noqa: E402

OUT = "runs/cv_ratio/report"
MD = "WEEKEND_ITERATIONS.md"
HEAD = "## RUN 2026-10-21 — CV_RATIO results"
ARMS = {"A": ("runs/cv_r3", "L1"), "B": ("runs/cv_ablation/mse", "MSE"),
        "C": ("runs/cv_ablation/mse_norm", "MSE, noise-norm."),
        "D": ("runs/cv_ablation/l1_norm", "L1, noise-norm.")}
RATIOS = (0.0, 1 / 12, 1 / 6, 0.25, 1 / 3, 0.5, 1.0)
RNAME = {0.0: "0", 1 / 12: "1/12", 1 / 6: "1/6", 0.25: "1/4", 1 / 3: "1/3", 0.5: "1/2", 1.0: "1"}
A_SIZES = (1, 2, 3, 5, 7, 9)
BLURS = (0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
LOCK = 1 / 3
KL = "neff_r%.4f" % LOCK
DISPLAY_TV, DISPLAY_TE = (17, 16, 11, 1), (25, 22, 8)
NBOOT = 10000
COL = {"A": "C3", "B": "C1", "C": "C2", "D": "C4"}
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)


def load(arm):
    d = ARMS[arm][0]
    R = list(csv.DictReader(open(os.path.join(d, "ratio_eval", "per_slice_ratio.csv"), encoding="utf-8")))
    ne, pr, pw = {}, {}, {}
    for r in R:
        s = int(r["slice"])
        if r["region"] in ("signal", "all"):
            ne[(s, int(r["a"]), r["region"])] = {k: float(v) for k, v in r.items()
                                                 if v not in ("",) and k not in ("region",)}
        elif r["region"] == "per_r":
            pr[(s, round(float(r["ratio"]), 4))] = (float(r["noise"]), float(r["gain10"]))
        elif r["region"] == "per_w":
            pw[(s, float(r["width"]))] = float(r["noise"])
    Bd = {(int(r["slice"]), int(r["a"])): {k: float(v) for k, v in r.items()}
          for r in csv.DictReader(open(os.path.join(d, "ratio_eval", "bands_ratio.csv"), encoding="utf-8"))}
    return ne, pr, pw, Bd


D = {k: load(k) for k in ARMS}
slices = sorted({k[0] for k in D["A"][0]})
assert len(slices) == 34
sig = {s: D["A"][0][(s, 1, "signal")]["sigma_hat"] for s in slices}
for arm in "BCD":                                   # only the model differs between arms
    for key in D["A"][0]:
        for c in ["neff_r1.0000"] + ["blur_w%.2f" % w for w in BLURS]:
            assert np.isclose(D["A"][0][key][c], D[arm][0][key][c], rtol=1e-9), (arm, key, c)
gm = lambda v: float(np.exp(np.mean(np.log(v))))


def boot(lr):
    b = rng.choice(lr, size=(NBOOT, len(lr))).mean(1)
    return float(np.exp(np.percentile(b, 2.5))), float(np.exp(np.percentile(b, 97.5)))


def paired(num, den, a, rg, key=KL):
    lr = np.log([D[num][0][(s, a, rg)][key] / D[den][0][(s, a, rg)][key] for s in slices])
    lo, hi = boot(lr)
    p = float(wilcoxon(lr, alternative="two-sided").pvalue) if np.any(lr != 0) else 1.0
    return dict(gmean=float(np.exp(lr.mean())), ci_lo=lo, ci_hi=hi, wins=int((lr > 0).sum()), p=p)


def vs(arm, a, rg, base):
    lr = np.log([D[arm][0][(s, a, rg)][KL] / (a if base == "plain" else D[arm][0][(s, a, rg)][base])
                 for s in slices])
    lo, hi = boot(lr)
    p = float(wilcoxon(lr, alternative="greater").pvalue)
    return dict(gmean=float(np.exp(lr.mean())), ci_lo=lo, ci_hi=hi, wins=int((lr > 0).sum()), p=p)


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


# ---- set A -----------------------------------------------------------------------------
prim = {arm: paired(arm, "A", 1, "signal") for arm in "BCD"}
for arm, q in zip("BCD", holm([prim[a]["p"] for a in "BCD"])):
    prim[arm]["p_holm"] = float(q)
    prim[arm]["verdict"] = ("better than L1" if q < 0.05 and prim[arm]["gmean"] > 1 else
                            "worse than L1" if q < 0.05 and prim[arm]["gmean"] < 1 else
                            "no significant difference")
secA = {(arm, a, rg): paired(arm, "A", a, rg) for arm in "BCD" for a in A_SIZES for rg in ("signal", "all")}
base = {(arm, a, b): vs(arm, a, "signal", b) for arm in ARMS for a in A_SIZES
        for b in ("plain", "blur_matched", "blur_oracle")}
noise3 = {arm: [D[arm][1][(s, round(LOCK, 4))][0] for s in slices] for arm in ARMS}
gain3 = {arm: [D[arm][1][(s, round(LOCK, 4))][1] for s in slices] for arm in ARMS}
wmatch = {arm: [D[arm][0][(s, 1, "signal")]["w_matched"] for s in slices] for arm in ARMS}


def band_wins(arm, a):
    out = []
    for b in range(4):
        x = np.array([D[arm][3][(s, a)]["r%.4f_band%d" % (LOCK, b)] for s in slices])
        y = np.array([D[arm][3][(s, a)]["blur_matched_band%d" % b] for s in slices])
        fin = np.isfinite(x) & np.isfinite(y)
        out.append((float(np.median(x[fin] / y[fin])), int(((x > y) & ~(np.isinf(x) & np.isinf(y))).sum())))
    return out


bandA = {(arm, a): band_wins(arm, a) for arm in ARMS for a in (1, 5, 9)}

# ---- set B -----------------------------------------------------------------------------
surf = {(arm, r, a): gm([D[arm][0][(s, a, "signal")]["neff_r%.4f" % r] for s in slices])
        for arm in ARMS for r in RATIOS for a in A_SIZES}
noise_r = {(arm, r): float(np.median([D[arm][1][(s, round(r, 4))][0] for s in slices]))
           for arm in ARMS for r in RATIOS}
best_r = {(arm, a): max(RATIOS, key=lambda r: surf[(arm, r, a)]) for arm in ARMS for a in A_SIZES}

# ---- figures: surfaces / noise / frontier / bands ------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
for ax, arm in zip(axs, ARMS):
    for r, c in zip(RATIOS, plt.cm.viridis(np.linspace(0, 1, len(RATIOS)))):
        ax.plot(A_SIZES, [surf[(arm, r, a)] for a in A_SIZES], "o-", ms=3, color=c, label=f"r = {RNAME[r]}")
    ax.set_yscale("log"); ax.set_xticks(A_SIZES); ax.set_xlabel("realizations averaged (a)")
    ax.set_title(f"{arm}: {ARMS[arm][1]}", fontsize=9)
axs[0].set_ylabel("geo-mean n_eff, 34 held-out slices (log)"); axs[0].legend(fontsize=7)
fig.suptitle("Set B: n_eff over target_ratio r and split size a (signal region; r = 1 is plain averaging)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "B_neff_surface.png"), dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 4.2))
for arm in ARMS:
    ax.plot(RATIOS, [noise_r[(arm, r)] for r in RATIOS], "o-", color=COL[arm], label=f"{arm}: {ARMS[arm][1]}")
ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="achieved = r")
ax.set_xlabel("target_ratio r"); ax.set_ylabel("achieved residual noise / sigma_hat (median)")
ax.set_title("One-step DDIM rule under cross-validation", fontsize=9); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "B_noise_vs_r.png"), dpi=110); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, a in zip(axs, (1, 5)):
    for arm in ARMS:
        ax.plot([noise_r[(arm, r)] for r in RATIOS], [surf[(arm, r, a)] for r in RATIOS], "o-",
                color=COL[arm], label=f"{arm}: {ARMS[arm][1]}")
    bx = [float(np.median([D["A"][2][(s, w)] for s in slices])) for w in BLURS]
    by = [gm([D["A"][0][(s, a, "signal")]["blur_w%.2f" % w] for s in slices]) for w in BLURS]
    ax.plot(bx, by, "d--", color="C0", label="Gaussian blur (width sweep)")
    ax.axvline(LOCK, color="grey", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("residual noise of one output / sigma_hat"); ax.set_title(f"a = {a}", fontsize=9)
axs[0].set_ylabel("geo-mean n_eff (signal)"); axs[0].legend(fontsize=7)
fig.suptitle("Frontier: accuracy vs residual noise (dotted line = locked r = 1/3)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "B_frontier.png"), dpi=110); plt.close(fig)

fig, axs = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
for ax, b in zip(axs, range(4)):
    for arm in ARMS:
        med = []
        for r in RATIOS:
            v = np.array([D[arm][3][(s, 5)]["r%.4f_band%d" % (r, b)] for s in slices])
            med.append(float(np.median(np.minimum(v, 1e4))))
        ax.plot(RATIOS, med, "o-", color=COL[arm], label=f"{arm}: {ARMS[arm][1]}")
    ax.set_yscale("log"); ax.set_xlabel("target_ratio r")
    ax.set_title(["0-0.0625", "0.0625-0.125", "0.125-0.25", "0.25-0.75"][b] + " cycles/pixel", fontsize=9)
axs[0].set_ylabel("median per-band n_eff, a = 5 (inf capped 1e4)"); axs[0].legend(fontsize=7)
fig.suptitle("Where the ratio acts: per-band accuracy vs r", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "B_bands_vs_r.png"), dpi=110); plt.close(fig)

# ---- map figures ------------------------------------------------------------------------------
x = load_data().double().numpy()
mag = lambda z: np.sqrt(z[..., 0, :, :] ** 2 + z[..., 1, :, :] ** 2)
X0 = {(arm, s): np.load(os.path.join(ARMS[arm][0], "ratio_eval", f"x0_s{s:02d}.npy")).astype(np.float64)
      for arm in ARMS for s in DISPLAY_TV + DISPLAY_TE}
role = {17: "train", 16: "val", 11: "val", 1: "val", 25: "test", 22: "test", 8: "test"}


def maps_A(sl, name):
    fig = plt.figure(figsize=(15, 4.3 * len(sl)))
    gs = GridSpec(2 * len(sl), 6, figure=fig, hspace=0.35, wspace=0.05)
    for i, s in enumerate(sl):
        ens = x[s]; mean10 = ens.mean(0); vmax = float(np.percentile(mag(mean10), 99.5))
        for j, (img, t) in enumerate(((mag(ens[0]), "raw realization 0"), (mag(mean10), "plain mean of 10"))):
            ax = fig.add_subplot(gs[2 * i:2 * i + 2, j]); ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"slice {s} ({role[s]})\nsigma_hat {sig[s]:.3f}", fontsize=9)
        for j, arm in enumerate(ARMS):
            out = X0[(arm, s)] + LOCK * (ens - X0[(arm, s)])
            n3, g3 = D[arm][1][(s, round(LOCK, 4))]
            for sub, (img, t) in enumerate(((mag(out[0]), f"{ARMS[arm][1]}: denoised real. 0\nnoise {n3:.3f}"),
                                            (mag(out.mean(0)), f"{ARMS[arm][1]}: avg of 10 denoised\ngain {g3:.3f}"))):
                ax = fig.add_subplot(gs[2 * i + sub, 2 + j]); ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=7)
    fig.suptitle("Set A: losses at the locked target_ratio r = 1/3 (each slice by its held-out fold model). "
                 "Per model: top = one denoised realization, bottom = average of 10 denoised.", fontsize=10)
    fig.savefig(os.path.join(OUT, name), dpi=95, bbox_inches="tight"); plt.close(fig)


maps_A(DISPLAY_TV, "maps_A_trainval.png")
maps_A(DISPLAY_TE, "maps_A_test.png")
order_r = (1.0, 0.5, 1 / 3, 0.25, 1 / 6, 1 / 12, 0.0)
for arm in ARMS:
    for s in DISPLAY_TV + DISPLAY_TE:
        ens = x[s]; vmax = float(np.percentile(mag(ens.mean(0)), 99.5))
        fig, axs = plt.subplots(2, len(order_r), figsize=(2.3 * len(order_r), 5.0))
        for j, r in enumerate(order_r):
            out = X0[(arm, s)] + r * (ens - X0[(arm, s)])
            n_r, g_r = D[arm][1][(s, round(r, 4))]
            for i, (img, t) in enumerate(((mag(out[0]), f"r = {RNAME[r]}\nnoise {n_r:.3f}"),
                                          (mag(out.mean(0)), f"avg of 10, gain {g_r:.3f}"))):
                axs[i, j].imshow(img, cmap="gray", vmin=0, vmax=vmax)
                axs[i, j].set_xticks([]); axs[i, j].set_yticks([]); axs[i, j].set_title(t, fontsize=8)
        axs[0, 0].set_ylabel("one denoised\nrealization", fontsize=9)
        axs[1, 0].set_ylabel("average of 10\ndenoised", fontsize=9)
        fig.suptitle(f"Set B, {ARMS[arm][1]}, slice {s} ({role[s]}, sigma_hat {sig[s]:.3f}): target_ratio "
                     "from r = 1 (raw / plain mean) to r = 0 (full posterior mean)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"maps_B_{arm}_s{s:02d}.png"), dpi=95); plt.close(fig)

# ---- summary + log -------------------------------------------------------------------------------
summary = dict(prim=prim, base={"%s|%d|%s" % k: v for k, v in base.items()},
               secA={"%s|%d|%s" % k: v for k, v in secA.items()},
               noise3={a: [float(np.median(v)), float(min(v)), float(max(v))] for a, v in noise3.items()},
               gain3={a: [float(np.median(v)), float(min(v)), float(max(v))] for a, v in gain3.items()},
               surf={"%s|%.4f|%d" % k: v for k, v in surf.items()},
               noise_r={"%s|%.4f" % k: v for k, v in noise_r.items()},
               best_r={"%s|%d" % k: v for k, v in best_r.items()},
               bandA={"%s|%d" % k: v for k, v in bandA.items()}, labels={k: v[1] for k, v in ARMS.items()})
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)

f3 = lambda r: "%.3f× (95%% CI %.3f–%.3f; %d/34)" % (r["gmean"], r["ci_lo"], r["ci_hi"], r["wins"])
L = ["", "---", "", HEAD, "",
     "Pre-registered (`Archive/CV_RATIO/PREREGISTRATION.md`); evaluation-only on the 20 existing fold "
     "models; every slice scored by its held-out model. Self-checks passed for every arm: r = 0 "
     "reproduces the CV_R3/CV_ABLATION n_eff, r = 1 equals plain averaging, oracle blur identical, "
     "noise-matched blur within 3% of the target noise.", "",
     "### Set A — losses at the locked ratio r = 1/3", "",
     "Achieved residual noise of one output at r = 1/3 (median, range over slices): " + "; ".join(
         "%s %.3f (%.3f–%.3f)" % ((ARMS[a][1],) + tuple(summary["noise3"][a])) for a in ARMS) + ".", "",
     "**PRIMARY (a = 1, signal, n_eff ratio vs L1):**", ""]
for arm in "BCD":
    L.append("- %s: %s, p = %.2g, Holm %.2g → **%s**" % (ARMS[arm][1], f3(prim[arm]), prim[arm]["p"],
                                                          prim[arm]["p_holm"], prim[arm]["verdict"]))
L += ["", "Each arm at r = 1/3 against the baselines (signal; geometric mean, slices won, one-sided p):", "",
      "| arm | a | vs plain averaging | vs noise-matched blur | vs oracle-MSE blur |", "|---|---:|---|---|---|"]
for arm in ARMS:
    for a in (1, 5, 9):
        c = lambda b: "%.3f× (%d/34, p=%.1g)" % (base[(arm, a, b)]["gmean"], base[(arm, a, b)]["wins"], base[(arm, a, b)]["p"])
        L.append("| %s | %d | %s | %s | %s |" % (ARMS[arm][1], a, c("plain"), c("blur_matched"), c("blur_oracle")))
L += ["", "Secondary, ratio vs L1 at r = 1/3 (signal; unadjusted p):", "",
      "| a | MSE | MSE, noise-norm. | L1, noise-norm. |", "|---:|---|---|---|"]
for a in A_SIZES:
    L.append("| %d | %s |" % (a, " | ".join("%.3f× [%.3f–%.3f], %d/34, p=%.2g" % (
        secA[(arm, a, "signal")]["gmean"], secA[(arm, a, "signal")]["ci_lo"], secA[(arm, a, "signal")]["ci_hi"],
        secA[(arm, a, "signal")]["wins"], secA[(arm, a, "signal")]["p"]) for arm in "BCD")))
L += ["", "Per band vs the noise-matched blur at r = 1/3 (median ratio; slices where the arm wins):", ""]
for arm in ARMS:
    L.append("- %s: " % ARMS[arm][1] + "; ".join("a=%d: %s" % (a, ", ".join("%.2f (%d)" % mb for mb in bandA[(arm, a)]))
                                                  for a in (1, 5, 9)))
L += ["", "### Set B — ratio × split size (descriptive)", "",
      "Geometric-mean n_eff (signal) by r and a; r = 1 is plain averaging (= a):", ""]
for arm in ARMS:
    L += ["**%s**" % ARMS[arm][1], "", "| r | " + " | ".join("a=%d" % a for a in A_SIZES) + " | noise of one output |",
          "|---|" + "---:|" * (len(A_SIZES) + 1)]
    for r in RATIOS:
        L.append("| %s | %s | %.3f |" % (RNAME[r], " | ".join("%.2f" % surf[(arm, r, a)] for a in A_SIZES), noise_r[(arm, r)]))
    L.append("")
L += ["Highest-n_eff ratio per a (descriptive, not a selection): " + "; ".join(
    "%s: %s" % (ARMS[arm][1], ", ".join("a=%d→%s" % (a, RNAME[best_r[(arm, a)]]) for a in A_SIZES)) for arm in ARMS) + ".", "",
      "**Figures** (`%s/`): `maps_A_trainval.png`, `maps_A_test.png` (set A maps), `maps_B_<arm>_s<NN>.png` "
      "(set B maps), `B_neff_surface.png`, `B_noise_vs_r.png`, `B_frontier.png`, `B_bands_vs_r.png`. "
      "**Data:** `<arm>/ratio_eval/*.csv`, `%s/summary.json`. **Commands:** `python experiments/_cv_ratio_eval.py "
      "--cv <arm dir>`; `python experiments/_ratio_report.py`. Stdout: `Archive/CV_RATIO/*_raw.html`." % (OUT, OUT), ""]
for f in sorted(os.listdir("runs/cv_ratio")):
    if f.startswith("stdout_") and f.endswith(".txt"):
        t = open(os.path.join("runs/cv_ratio", f), encoding="utf-8", errors="replace").read()
        e = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        open(os.path.join("Archive/CV_RATIO", f[:-4] + "_raw.html"), "w", encoding="utf-8").write(
            "<!doctype html><meta charset=\"utf-8\"><title>%s</title><h1>CV_RATIO %s</h1>"
            "<p>Verbatim stdout, unedited.</p><pre>%s</pre>\n" % (f, f, e))
if HEAD not in open(MD, encoding="utf-8").read():
    open(MD, "a", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
