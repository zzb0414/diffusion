# -*- coding: utf-8 -*-
"""DEC part A, exploratory: bias / variance / noise-covariance decomposition per arm.

1/n_eff(a, r) = (1-r)^2 (b + v/a) + r^2/a + 2r(1-r) c/a holds exactly for the all-splits estimator
(`_gr_theory.py`), so (b, v, c) per slice follow by least squares from the DEC n_eff grid. b = bias^2,
v = per-map variance of the one-shot estimate, c = its covariance with the noise (units of sigma^2).

    python experiments/_dec_bvc.py
"""
import os, csv, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
os.chdir(r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion")


def fit(pts):
    X = np.array([[(1 - r) ** 2, (1 - r) ** 2 / a, 2 * r * (1 - r) / a] for a, r, m in pts])
    y = np.array([m - r * r / a for a, r, m in pts])
    return np.linalg.lstsq(X, y, rcond=None)[0]


print("median over slices x seeds 101-103; b = bias^2, v = per-map variance, c = noise covariance (units of sigma^2)")
for d in (1, 2, 3):
    for arm in ("dsm", "dirl", "dirn", "blind"):
        P = []
        for sd in (101, 102, 103):
            S = {}
            for r in csv.DictReader(open(f"runs/dec/A/eval/{arm}_s{sd}/neff.csv", encoding="utf-8")):
                if int(r["dataset"]) == d:
                    S.setdefault(int(r["slice"]), []).append((int(r["a"]), float(r["r"]), 1 / float(r["neff_signal"])))
            P += [fit(p) for p in S.values()]
        b, v, c = np.median(P, 0)
        print("ds%d %-5s b %.4f  v %.4f  c %.4f   single map 1/(b+v) = %.1f   a -> inf limit 1/b = %.1f" % (
            d, arm, b, v, c, 1 / (b + v), 1 / b))
