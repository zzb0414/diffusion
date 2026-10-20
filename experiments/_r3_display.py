# -*- coding: utf-8 -*-
"""R3 displays: single output vs average of 10 outputs, swept over target_ratio.

For each slice, every one of the N realizations is decoded SEPARATELY with
``denoise_to(y_k, sigma_hat, target_ratio)``; then
  single    = output of realization 0 alone, compared to the LOO mean of the
              other 9 (the fair reference for one map);
  ensemble  = average of the N separate outputs, compared to the full mean of
              the N raw realizations (the quantity it estimates).
Rows sweep target_ratio from 1 (identity: no denoising, so single = raw
realization 0 and ensemble = the raw mean) down to 0 (full Tweedie).

Caveat for the ensemble frame: the ensemble average is built from the SAME
realizations as the full mean, so identity scores relE 0 against it. Ensemble
relE therefore measures how far denoising moves the mean, not its quality;
``ensemble - full mean`` shows what was removed (noise-like = good, structure =
signal loss). Quality numbers live in the single-map frame.

    python experiments/_r3_display.py --run runs/r3_total/l1 --iter 2000
"""
import sys, os, csv, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, DIFF); os.chdir(DIFF)

from _r1_eval import mag, hf                                 # noqa: E402
from _r2_displays import cstats                              # noqa: E402
from _headtohead import Scorer                               # noqa: E402
from _r3_train import load_data, sigma_hat                   # noqa: E402
from _r3_eval import load, SAMPLES                           # noqa: E402

RATIOS = (1.0, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 6.0, 1.0 / 12.0, 0.0)
NAMES = {1.0: "1 (identity)", 1.0 / 3.0: "1/3", 1.0 / 4.0: "1/4",
         1.0 / 6.0: "1/6", 1.0 / 12.0: "1/12", 0.0: "0 (full Tweedie)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/r3_total/l1")
    ap.add_argument("--iter", type=int, default=2000)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss = os.path.basename(a.run)
    out = os.path.join(a.run, "eval", "sweep"); os.makedirs(out, exist_ok=True)
    m, it = load(os.path.join(a.run, "ckpt", "%s_%06d.pkl" % (loss, a.iter)), device)
    x = load_data(); sig = sigma_hat(x)

    rows = []
    ov, ax_ov = plt.subplots(len(SAMPLES), 5, figsize=(15, 3.0 * len(SAMPLES)))
    ov_cols = ["reference\n|mean of 10 raw|", "average of 10 outputs\nratio 1/3",
               "average of 10 outputs\nratio 0", "single output (real. 0)\nratio 1/3",
               "single output (real. 0)\nratio 0"]
    print(f"{'split':6s}{'sl':>3s}{'ratio':>7s} | single:{'noise':>7s}{'relE':>7s}{'pha':>6s}"
          f" | ensemble:{'gain':>7s}{'sHF%':>7s}{'relE_vs_mean':>13s}")
    for row_i, (split, s) in enumerate(SAMPLES):
        ens = x[s].to(device); N = ens.shape[0]; sg = float(sig[s])
        full = ens.mean(0)
        loo0 = (ens.sum(0) - ens[0]) / (N - 1)
        refm = mag(full).cpu().numpy()
        loom = mag(loo0).cpu().numpy()
        sc = Scorer(ens, sg, loo0, refm)
        vmax = float(np.percentile(refm, 99.5))
        dlim = vmax / 3
        fig, ax = plt.subplots(len(RATIOS), 4, figsize=(12.5, 3.0 * len(RATIOS)))
        outs_at = {}
        for i, r in enumerate(RATIOS):
            with torch.no_grad():
                outs = m.denoise_to(ens, sg, target_ratio=r, device=device)  # each map alone
            outs_at[r] = outs
            met = sc(outs.detach())
            avg = outs.mean(0)
            ce = cstats(avg.detach().cpu(), full.detach().cpu())
            rows.append(dict(iter=it, split=split, slice=s, sigma_hat=sg, ratio=r,
                             single_noise=met["noise"], single_relE=met["relE"],
                             single_pha=met["pha"], ens_gain=met["gain"],
                             ens_s_bias=met["sb"], ens_shf=met["shf"],
                             ens_relE_vs_mean=ce["rel_err"], ens_pha_vs_mean=ce["pha_med_deg"]))
            print(f"{split:6s}{s:>3d}{r:>7.3f} |       {met['noise']:>7.3f}{met['relE']:>7.3f}"
                  f"{met['pha']:>6.1f} |          {met['gain']:>7.3f}{met['shf']:>7.1f}"
                  f"{ce['rel_err']:>13.3f}", flush=True)
            sm = mag(outs[0]).cpu().numpy()
            am = mag(avg).cpu().numpy()
            panels = [
                (sm, "gray", 0, vmax,
                 f"single (real. 0), ratio {NAMES[r]}\nnoise {met['noise']:.3f}  relE {met['relE']:.3f}"),
                (sm - loom, "RdBu_r", -dlim, dlim, "single minus LOO mean of other 9"),
                (am, "gray", 0, vmax,
                 f"average of {N} outputs, ratio {NAMES[r]}\ngain {met['gain']:.3f}  sHF {met['shf']:.1f}%"),
                (am - refm, "RdBu_r", -dlim, dlim,
                 "average minus raw mean of 10\n(what denoising removed)"),
            ]
            for j, (img, cm, lo, hi, title) in enumerate(panels):
                ax[i, j].imshow(img, cmap=cm, vmin=lo, vmax=hi)
                ax[i, j].set_title(title, fontsize=8); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        fig.suptitle(f"R3 {loss} iter {it} — {split} slice {s}, sigma_hat {sg:.4f}. Each of the {N} "
                     "realizations decoded separately with denoise_to(target_ratio). Row 1 = identity "
                     "(raw).", fontsize=9)
        fig.tight_layout()
        p = os.path.join(out, f"sweep_{split}_{s}.png"); fig.savefig(p, dpi=100); plt.close(fig)
        print("wrote", p)

        imgs = [refm, mag(outs_at[1.0 / 3.0].mean(0)).cpu().numpy(),
                mag(outs_at[0.0].mean(0)).cpu().numpy(),
                mag(outs_at[1.0 / 3.0][0]).cpu().numpy(), mag(outs_at[0.0][0]).cpu().numpy()]
        for j, img in enumerate(imgs):
            ax_ov[row_i, j].imshow(img, cmap="gray", vmin=0, vmax=vmax)
            ax_ov[row_i, j].set_xticks([]); ax_ov[row_i, j].set_yticks([])
            if row_i == 0:
                ax_ov[0, j].set_title(ov_cols[j], fontsize=9)
        ax_ov[row_i, 0].set_ylabel(f"{split} s{s}\nsigma {sg:.3f}", fontsize=9)

    ov.suptitle(f"R3 {loss} iter {it}: average of 10 separate outputs vs single output, "
                "shared grey scale per row", fontsize=11)
    ov.tight_layout()
    p = os.path.join(out, "sweep_overview.png"); ov.savefig(p, dpi=90); plt.close(ov)
    print("wrote", p)
    p = os.path.join(out, "sweep_metrics.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", p, len(rows), "rows")


if __name__ == "__main__":
    main()
