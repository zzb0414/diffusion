# -*- coding: utf-8 -*-
"""Candidate figures for the abstract (no titles; captions live in the draft).

Arm order for every figure: (A) L1, (B) L1 noise-normalized, (C) MSE,
(D) MSE noise-normalized. Notation: r = target ratio (output keeps r*sigma_hat
of noise), a = repetitions averaged, N = 10, sigma_hat = acquisition noise,
sigma_hat_out = residual noise of a denoised output, n_eff = sigma^2 / MSE.

Reads each arm's ``ratio_eval`` CSVs and display arrays (from
``_cv_ratio_eval.py``); writes ``runs/cv_ratio/figures/fig{1..4}_*.png``.

    python experiments/_figures.py
"""
import sys, os, csv
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from _r3_train import load_data                              # noqa: E402

OUT = "runs/cv_ratio/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 15,
                     "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "mathtext.fontset": "cm"})

ARMS = [("A", "runs/cv_r3", "L1"),
        ("B", "runs/cv_ablation/l1_norm", "L1, noise-normalized"),
        ("C", "runs/cv_ablation/mse", "MSE"),
        ("D", "runs/cv_ablation/mse_norm", "MSE, noise-normalized")]
COL = {"A": "C3", "B": "C4", "C": "C1", "D": "C2"}
RATIOS = (0.0, 1 / 12, 1 / 6, 0.25, 1 / 3, 0.5, 1.0)
RTEX = {0.0: "0", 1 / 12: r"\frac{1}{12}", 1 / 6: r"\frac{1}{6}", 0.25: r"\frac{1}{4}",
        1 / 3: r"\frac{1}{3}", 0.5: r"\frac{1}{2}", 1.0: "1"}
A_SIZES = (1, 2, 3, 5, 7, 9)
LOCK = 1 / 3


def load(d):
    R = list(csv.DictReader(open(os.path.join(d, "ratio_eval", "per_slice_ratio.csv"), encoding="utf-8")))
    ne = {(int(r["slice"]), int(r["a"])): r for r in R if r["region"] == "signal"}
    pr = {(int(r["slice"]), round(float(r["ratio"]), 4)): float(r["noise"]) for r in R if r["region"] == "per_r"}
    return ne, pr


DATA = {k: load(d) for k, d, _ in ARMS}
slices = sorted({s for s, _ in DATA["A"][0]})
sig = {s: float(DATA["A"][0][(s, 1)]["sigma_hat"]) for s in slices}
x = load_data().double().numpy()
mag = lambda z: np.sqrt(z[..., 0, :, :] ** 2 + z[..., 1, :, :] ** 2)
x0 = lambda d, s: np.load(os.path.join(d, "ratio_eval", f"x0_s{s:02d}.npy")).astype(np.float64)
gm = lambda v: float(np.exp(np.mean(np.log(v))))


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ---- Fig 1: maps at r = 1/3 for slices 17 / 16 / 8 ------------------------------------------
SL = [(17, "train"), (16, "val"), (8, "test")]
nrow = 3 * len(SL) - 1                                # two sub-rows per slice + spacer rows
hr = []
for i in range(len(SL)):
    hr += [1, 1] + ([0.18] if i < len(SL) - 1 else [])
fig = plt.figure(figsize=(16.5, 3.0 * 2 * len(SL) + 0.6))
gs = GridSpec(nrow, 6, figure=fig, height_ratios=hr, hspace=0.06, wspace=0.04)
heads = ["Noisy", "Mean of 10"] + [f"({k}) {n}" for k, _, n in ARMS]
for i, (s, role) in enumerate(SL):
    r0 = 3 * i
    ens = x[s]; mean10 = ens.mean(0); vmax = float(np.percentile(mag(mean10), 99.5))
    for j, img in enumerate((mag(ens[0]), mag(mean10))):
        ax = fig.add_subplot(gs[r0:r0 + 2, j]); ax.imshow(img, cmap="gray", vmin=0, vmax=vmax); off(ax)
        if i == 0:
            ax.set_title(heads[j], fontsize=15)
        if j == 0:
            ax.set_ylabel(f"Slice {s} ({role})\n" + r"$\hat\sigma = %.3f$" % sig[s], fontsize=15)
    for j, (k, d, _) in enumerate(ARMS):
        out = x0(d, s) + LOCK * (ens - x0(d, s))
        for sub, img in enumerate((mag(out[0]), mag(out.mean(0)))):
            ax = fig.add_subplot(gs[r0 + sub, 2 + j]); ax.imshow(img, cmap="gray", vmin=0, vmax=vmax); off(ax)
            if i == 0 and sub == 0:
                ax.set_title(heads[2 + j].replace(", noise-normalized", ",\nnoise-normalized"), fontsize=14)
            if j == 0:
                ax.set_ylabel("Single" if sub == 0 else "Mean of 10", fontsize=14)
fig.savefig(os.path.join(OUT, "fig1_maps_matched_noise.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- Fig 2: r sweep for slice 16, all arms stacked --------------------------------------------
S2 = 16
order_r = (1.0, 0.5, 1 / 3, 0.25, 1 / 6, 1 / 12, 0.0)
ens = x[S2]; vmax = float(np.percentile(mag(ens.mean(0)), 99.5))
hr = []
for i in range(len(ARMS)):
    hr += [1, 1] + ([0.32] if i < len(ARMS) - 1 else [])
fig = plt.figure(figsize=(17, 2.55 * 2 * len(ARMS) + 1.0))
gs = GridSpec(len(hr), len(order_r), figure=fig, height_ratios=hr, hspace=0.06, wspace=0.04)
for i, (k, d, name) in enumerate(ARMS):
    r0 = 3 * i
    X0 = x0(d, S2)
    for j, r in enumerate(order_r):
        out = X0 + r * (ens - X0)
        for sub, img in enumerate((mag(out[0]), mag(out.mean(0)))):
            ax = fig.add_subplot(gs[r0 + sub, j]); ax.imshow(img, cmap="gray", vmin=0, vmax=vmax); off(ax)
            if i == 0 and sub == 0:
                ax.set_title(r"$r = %s$" % RTEX[r], fontsize=17, pad=8)
            if j == 0:
                ax.set_ylabel((f"({k}) {name}\n".replace(", noise-normalized", ",\nnoise-norm.") if sub == 0 else "")
                              + ("Single" if sub == 0 else "Mean of 10"), fontsize=13)
fig.savefig(os.path.join(OUT, "fig2_ratio_sweep_s16.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- Fig 3: n_eff surface over (r, a), one panel per arm ---------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(19, 5.0), sharey=True)
cmap = plt.cm.viridis(np.linspace(0, 0.92, len(RATIOS)))
for ax, (k, d, name) in zip(axs, ARMS):
    ne = DATA[k][0]
    for r, c in zip(RATIOS, cmap):
        y = [gm([float(ne[(s, a)]["neff_r%.4f" % r]) for s in slices]) for a in A_SIZES]
        ax.plot(A_SIZES, y, "o-", ms=5, lw=2, color=c, label=r"$r = %s$" % RTEX[r])
    ax.set_yscale("log"); ax.set_xticks(A_SIZES)
    ax.set_xlabel(r"Repetitions averaged, $a$")
    ax.text(0.03, 0.97, f"({k}) {name}", transform=ax.transAxes, va="top", fontsize=14)
    ax.grid(alpha=0.3, which="both")
axs[0].set_ylabel(r"$n_{\mathrm{eff}}$")
axs[-1].legend(loc="lower right", ncol=1, frameon=True)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_neff_r_a.png"), dpi=200); plt.close(fig)

# ---- Fig 4: achieved residual noise vs r ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 6.0))
ax.plot([0, 1], [0, 1], "k:", lw=1.2, label=r"$\hat\sigma_{\mathrm{out}} = r\,\hat\sigma$")
for k, d, name in ARMS:
    pr = DATA[k][1]
    v = np.array([[pr[(s, round(r, 4))] for s in slices] for r in RATIOS])
    ax.plot(RATIOS, np.median(v, 1), "o-", lw=2, ms=5, color=COL[k], label=f"({k}) {name}")
    ax.fill_between(RATIOS, np.percentile(v, 25, 1), np.percentile(v, 75, 1), color=COL[k], alpha=0.15)
ax.set_xlabel(r"Target ratio, $r$")
ax.set_ylabel(r"Residual noise, $\hat\sigma_{\mathrm{out}} / \hat\sigma$")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3)
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_noise_vs_r.png"), dpi=200); plt.close(fig)

print("wrote", sorted(os.listdir(OUT)))
