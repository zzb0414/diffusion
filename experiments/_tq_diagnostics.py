# -*- coding: utf-8 -*-
"""EXPLORATORY diagnostics behind the TQ plan (reference quality vs SNR vs noise colour).

Recomputes from the data and existing evaluations: per-repetition SNR per slice,
how the edge over the noise-matched blur relates to SNR and sigma_hat, the
dataset residual after an SNR fit, and each dataset's noise colour
(finest/coarsest band noise power) and spatial noise variation. Writes
``runs/tq/diagnostics.json`` (refuses to overwrite) and appends one log section.

    python experiments/_tq_diagnostics.py
"""
import sys, os, csv, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from _data import load_datasets                              # noqa: E402

OUT = "runs/tq/diagnostics.json"
MD = "WEEKEND_ITERATIONS.md"
HEAD = "### TQ exploratory diagnostics — reference quality vs SNR vs noise colour"
BANDS = ((0, 0.0625), (0.0625, 0.125), (0.125, 0.25), (0.25, 0.75))
assert not os.path.exists(OUT), "refusing to overwrite " + OUT
os.makedirs(os.path.dirname(OUT), exist_ok=True)

X, meta, S = load_datasets([1, 2, 3]); X = X.double().numpy()
H = X.shape[-1]; fr = np.hypot(np.fft.fftfreq(H)[:, None], np.fft.fftfreq(H)[None, :])
snr, colour, spat = {}, {}, {}
for i, (d, s) in enumerate(meta):
    e = X[i]; N = e.shape[0]; m = e.mean(0); dev = e - m
    v = (dev ** 2).sum(0) / (N - 1)
    mg = np.sqrt((m ** 2).sum(0)); sm = gaussian_filter(mg, 3); msk = sm > np.median(sm)
    snr[(d, s)] = float(((m ** 2).sum(0) - v.sum(0) / N)[msk].mean() / (2 * v.mean(0)[msk].mean()))
    pn = ((np.abs(np.fft.fft2(dev)) ** 2).sum(0) / (N - 1)).mean(0)
    bp = [pn[(fr >= lo) & (fr < hi)].mean() for lo, hi in BANDS]
    colour[(d, s)] = float(bp[3] / bp[0])
    vs = gaussian_filter(v.mean(0), 4)
    spat[(d, s)] = float(np.percentile(vs, 95) / np.percentile(vs, 5))


def rows(path, ds_col):
    out = {}
    for r in csv.DictReader(open(os.path.join(path, "per_slice_ratio.csv"), encoding="utf-8")):
        if r["region"] == "signal" and r["a"] in ("1", "9"):
            out[(int(r["dataset"]) if ds_col else 1, int(r["slice"]), int(r["a"]))] = \
                float(r["neff_r0.3333"]) / float(r["blur_matched"])
    return out


SRC = {"dataset1-only (A)": {**rows("runs/cv_r3/ratio_eval", False),
                             **{k: v for k, v in rows("runs/gen/A", True).items() if k[0] > 1}},
       "pooled (A) @2000": rows("runs/mx_cv/A/eval_it2000", True),
       "pooled (B) @2000": rows("runs/mx_cv/B/eval_it2000", True)}
keys = [(d, s) for d in (1, 2, 3) for s in range(34)]
lsnr = np.log([snr[k] for k in keys]); sg = [float(S[meta.index(k)]) for k in keys]
res = {}
for name, R in SRC.items():
    for a in (1, 9):
        y = np.log([R[k + (a,)] for k in keys])
        A = np.vstack([lsnr, np.ones_like(lsnr)]).T
        resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        res[f"{name}|{a}"] = dict(rho_snr=float(spearmanr(lsnr, y).correlation),
                                  rho_sigma=float(spearmanr(sg, y).correlation),
                                  resid_ds=[float(np.mean(resid[i * 34:(i + 1) * 34])) for i in range(3)])
med = lambda dd, d: [float(np.median([dd[(d, s)] for s in range(34)])), float(min(dd[(d, s)] for s in range(34))),
                     float(max(dd[(d, s)] for s in range(34)))]
summary = dict(snr={d: med(snr, d) for d in (1, 2, 3)}, colour={d: med(colour, d) for d in (1, 2, 3)},
               spatial={d: med(spat, d) for d in (1, 2, 3)}, edge_vs_blur=res)
json.dump(summary, open(OUT, "w"), indent=2)
assert summary["colour"][2][0] > 1 > max(summary["colour"][1][0], summary["colour"][3][0])

L = ["", HEAD, "",
     "*Exploratory, read-only (no training).* Per-repetition SNR in the signal region (median [range]): " + "; ".join(
         "ds%d %.2f [%.2f–%.2f]" % ((d,) + tuple(summary["snr"][d])) for d in (1, 2, 3)) + ".",
     "Noise colour, finest/coarsest band noise power (white = 1): " + "; ".join(
         "ds%d %.2f" % (d, summary["colour"][d][0]) for d in (1, 2, 3)) + " — dataset 2's noise is fine-heavy, "
     "datasets 1 and 3 coarse-heavy. Spatial noise variation p95/p5: " + "; ".join(
         "ds%d %.2f" % (d, summary["spatial"][d][0]) for d in (1, 2, 3)) + ".", "",
     "Edge over the noise-matched blur (r = 1/3) across all 102 slices — Spearman ρ with log SNR and with σ̂; "
     "mean log-residual per dataset after a linear fit on log SNR:", ""]
for k, v in res.items():
    L.append("- %s, a = %s: ρ(SNR) %+.2f, ρ(σ̂) %+.2f, residual ds1/2/3 %+.3f/%+.3f/%+.3f" % (
        tuple(k.split("|")) + (v["rho_snr"], v["rho_sigma"]) + tuple(v["resid_ds"])))
L += ["", "Reading (to be tested by TQ E1/E2): the edge tracks noise strength, and dataset 2 stays below the "
      "SNR trend even for pooled models trained on it, consistent with its fine-heavy noise colour (cheap for a "
      "blur to remove) rather than with reference quality alone.", ""]
if HEAD not in open(MD, encoding="utf-8").read():
    open(MD, "a", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
