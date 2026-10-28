# -*- coding: utf-8 -*-
"""TQ figures (not committed): reads runs/tq/report/summary.json and the saved decoded maps.

  tq_e1_neff_ratio.png    n_eff (target from k repetitions) / n_eff (k = 9), per dataset and arm
  tq_e1_maps_s16.png      slice 16, datasets 1 and 2: decoded with targets from k = 1, 3, 9 (arm A)
  tq_e2_edge_vs_sigma.png edge over the noise-matched blur vs added-noise factor, by noise colour
  tq_e2_maps_s16.png      dataset-1 slice 16 degraded to 2.0 x sigma, by colour, decoded (arm A)

Maps: transposed, shared grey window per slice (WIN x 99.5th pct of the plain mean of 10), r = 1/3.
Writes to runs/tq/report/figures/ and refuses to overwrite.

    python experiments/_tq_figures.py
"""
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from _data import load_datasets                              # noqa: E402
from _degrade import degrade                                 # noqa: E402
from _figures_window import WIN                              # noqa: E402

OUT = "runs/tq/report/figures"
assert not os.path.exists(OUT), "refusing to overwrite " + OUT
os.makedirs(OUT)
plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "legend.fontsize": 12,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "mathtext.fontset": "cm"})
J = json.load(open("runs/tq/report/summary.json", encoding="utf-8"))
LOCK, S, A3 = 1 / 3, 16, (1, 5, 9)
COLK = {1: "C3", 3: "C0"}
COLC = {"self": "C2", "white": "C7", "ds2": "C1"}
NAMEC = {"self": "Own noise colour", "white": "White", "ds2": "Dataset-2 colour"}
MODELS = [("one_A", "(A) L1, dataset 1 only"), ("one_D", "(D) MSE norm., dataset 1 only"),
          ("pool_A", "(A) L1, pooled"), ("pool_D", "(D) MSE norm., pooled")]
mag = lambda z: np.sqrt(z[..., 0, :, :] ** 2 + z[..., 1, :, :] ** 2)


def off(a):
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)


# ---- E1: n_eff ratio vs k = 9 -------------------------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
for i, (arm, an) in enumerate((("A", "(A) L1"), ("D", "(D) MSE, noise-normalized"))):
    for j, d in enumerate((1, 2, 3)):
        ax = axs[i, j]
        ax.axhline(1, color="k", lw=0.9, ls=":")
        for k, dx in ((1, -0.12), (3, 0.12)):
            R = [J["e1"]["%s|%d|%d|%d|neff_r0.3333" % (arm, k, d, a)] for a in A3]
            x = np.array(A3) + dx
            ax.errorbar(x, [r["gmean"] for r in R], yerr=[[r["gmean"] - r["ci_lo"] for r in R], [r["ci_hi"] - r["gmean"] for r in R]],
                        fmt="o-", color=COLK[k], ms=7, lw=1.5, capsize=3, label=f"Target from $k = {k}$")
        ax.text(0.03, 0.97, f"{an}\nDataset {d}", transform=ax.transAxes, va="top", fontsize=13)
        ax.grid(alpha=0.3); ax.set_xticks(A3)
        if i == 1:
            ax.set_xlabel(r"Repetitions averaged, $a$")
        if j == 0:
            ax.set_ylabel(r"$n_{\mathrm{eff}}(k)\,/\,n_{\mathrm{eff}}(k=9)$")
axs[0, -1].legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "tq_e1_neff_ratio.png"), dpi=200); plt.close(fig)

# ---- E1: maps --------------------------------------------------------------------------------------------
X, meta, _ = load_datasets([1, 2])
X = X.double().numpy()
K9 = {1: "runs/cv_r3/ratio_eval/x0_s16.npy", 2: "runs/gen/A/x0_d2_s16.npy"}
fig, axs = plt.subplots(4, 5, figsize=(16, 13.4), gridspec_kw=dict(hspace=0.04, wspace=0.03))
for i, d in enumerate((1, 2)):
    ens = X[meta.index((d, S))]; m10 = ens.mean(0); vmax = WIN * float(np.percentile(mag(m10), 99.5))
    cols = [("Noisy", mag(ens[0]), None), ("Mean of 10", mag(m10), None)]
    for k in (1, 3, 9):
        x0 = np.load(K9[d] if k == 9 else f"runs/tq/A_k{k}/eval_it2000/x0_d{d}_s{S:02d}.npy").astype(np.float64)
        out = x0 + LOCK * (ens - x0)
        cols.append((f"Target from $k = {k}$", mag(out[0]), mag(out.mean(0))))
    for j, (h, top, bot) in enumerate(cols):
        for sub, img in enumerate((top, bot)):
            a = axs[2 * i + sub, j]; off(a)
            if img is not None:
                a.imshow(img.T, cmap="gray", vmin=0, vmax=vmax)
            if i == 0 and sub == 0:
                a.set_title(h, fontsize=15)
        if j == 0:
            axs[2 * i, 0].set_ylabel(f"Dataset {d}\nSingle", fontsize=14)
    axs[2 * i + 1, 2].set_ylabel("Mean of 10", fontsize=14)
fig.savefig(os.path.join(OUT, "tq_e1_maps_s16.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- E2: edge vs noise factor ----------------------------------------------------------------------------
F = (1.0, 1.4, 2.0)
fig, axs = plt.subplots(2, 4, figsize=(19, 8.4), sharex=True)
for i, a in enumerate((1, 9)):
    for j, (m, mn) in enumerate(MODELS):
        ax = axs[i, j]
        ax.axhline(1, color="k", lw=0.9, ls=":")
        for d, ls in ((2, "--"), (3, "-.")):
            ax.axhline(J["real"]["%s|%d|%d" % (m, d, a)], color="0.45", lw=1.1, ls=ls, label=f"Real dataset {d}")
        und = J["edge_e2"]["%s|None|2.0|%d" % (m, a)]
        for c in ("self", "white", "ds2"):
            ax.plot(F, [und] + [J["edge_e2"]["%s|%s|%s|%d" % (m, c, f, a)] for f in ("1.4", "2.0")], "o-",
                    color=COLC[c], ms=8, lw=2, label=NAMEC[c])
        ax.text(0.97, 0.97, f"{mn}\n$a = {a}$", transform=ax.transAxes, ha="right", va="top", fontsize=12)
        ax.grid(alpha=0.3); ax.set_xticks(F)
        if i == 1:
            ax.set_xlabel(r"Noise factor $\hat\sigma_{\mathrm{deg}}/\hat\sigma$ (dataset 1)")
        if j == 0:
            ax.set_ylabel(r"$n_{\mathrm{eff}}\,/\,n_{\mathrm{eff}}^{\mathrm{blur}}$ at $r = 1/3$")
    lo = min(min(l.get_ydata()) for ax in axs[i] for l in ax.get_lines())
    hi = max(max(l.get_ydata()) for ax in axs[i] for l in ax.get_lines())
    for ax in axs[i]:
        ax.set_ylim(lo - 0.02, hi + 0.06)
axs[1, 0].legend(loc="lower left", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "tq_e2_edge_vs_sigma.png"), dpi=200); plt.close(fig)

# ---- E2: maps --------------------------------------------------------------------------------------------
ens0 = X[meta.index((1, S))]
vmax = WIN * float(np.percentile(mag(ens0.mean(0)), 99.5))
cols = [("Undegraded", ens0, "runs/cv_r3/ratio_eval/x0_s16.npy")]
for c in ("self", "white", "ds2"):
    e = degrade(ens0.copy(), 2.0, c, seed=20261027 + 1000 * 1 + S)[0]
    cols.append((NAMEC[c] + r", $2\hat\sigma$", e, f"runs/degrade/one_A_{c}_f2.0/x0_d1_s{S:02d}.npy"))
rows = ("Noisy", "Mean of 10", "Denoised single", "Denoised mean of 10")
fig, axs = plt.subplots(4, 4, figsize=(13, 13.4), gridspec_kw=dict(hspace=0.04, wspace=0.03))
for j, (h, e, p) in enumerate(cols):
    x0 = np.load(p).astype(np.float64); out = x0 + LOCK * (e - x0)
    for i, img in enumerate((mag(e[0]), mag(e.mean(0)), mag(out[0]), mag(out.mean(0)))):
        a = axs[i, j]; off(a); a.imshow(img.T, cmap="gray", vmin=0, vmax=vmax)
        if i == 0:
            a.set_title(h, fontsize=14)
        if j == 0:
            a.set_ylabel(rows[i], fontsize=14)
fig.savefig(os.path.join(OUT, "tq_e2_maps_s16.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
for f in sorted(os.listdir(OUT)):
    print("wrote", os.path.join(OUT, f))
