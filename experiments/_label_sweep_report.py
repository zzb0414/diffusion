# -*- coding: utf-8 -*-
"""LABEL SWEEP report (exploratory, 2026-11-03): does feeding c * sigma_hat to the dataset-1 DSM models recover
datasets 2-3? Reads runs/label_sweep/dsm_s<seed>/c<c>/neff.csv (written by _dec_eval.py --sigma-scale c).

Per dataset and a: geometric-mean n_eff (signal region; r cross-fitted over the other folds' slices; geometric mean
over seeds per slice, then over slices) for every c, and the ratio to c = 1 with a slice bootstrap 90% CI.
"c cross-fitted": c chosen per fold from the other folds' slices (an honest, deployable choice).
c = 1 must reproduce runs/dec/A/eval/dsm_s<seed> (checked).

    python experiments/_label_sweep_report.py
"""
import os, csv
import numpy as np

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
os.chdir(DIFF)
CS = (0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.5)
SEEDS, S34, RG = (101, 102, 103), list(range(34)), (0.0, 0.0833, 0.1667, 0.25, 0.3333, 0.5)
rng = np.random.default_rng(0)


def read(path):
    T = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["decoder"] == "tw":
            T[(int(r["dataset"]), int(r["slice"]), round(float(r["r"]), 4), int(r["a"]))] = (float(r["neff_signal"]), int(r["fold"]))
    return T


def inv(v):
    return 1 / v if v > 0 and np.isfinite(v) else 0.0


def r_cf(T, d, s, a):
    f = T[(d, s, 0.0, a)][1]
    return min(RG, key=lambda r: sum(inv(T[(d, q, r, a)][0]) for q in S34 if T[(d, q, 0.0, a)][1] != f))


def per_slice(Ts, d, a):
    """log n_eff per slice (cross-fitted r, geometric mean over seeds)."""
    return np.array([np.mean([np.log(T[(d, s, r_cf(T, d, s, a), a)][0]) for T in Ts]) for s in S34])


def boot(v):
    b = rng.choice(v, size=(10000, len(v))).mean(1)
    return np.exp(v.mean()), np.exp(np.percentile(b, 5)), np.exp(np.percentile(b, 95))


T = {c: [read(f"runs/label_sweep/dsm_s{sd}/c{c}/neff.csv") for sd in SEEDS] for c in CS}
# c = 1 reproduces the DEC evaluation
worst = 0.0
for sd, Tc in zip(SEEDS, T[1.0]):
    ref = read(f"runs/dec/A/eval/dsm_s{sd}/neff.csv")
    worst = max(worst, max(abs(Tc[k][0] / ref[k][0] - 1) for k in ref if np.isfinite(ref[k][0]) and ref[k][0] > 0))
print(f"check c = 1 vs runs/dec/A/eval: worst relative difference {worst:.2e}")

fold = {s: T[1.0][0][(1, s, 0.0, 1)][1] for s in S34}
print("\nGeometric-mean n_eff (signal), r cross-fitted; ratio to c = 1 (90% CI) for the best c and the cross-fitted c")
for d in (1, 2, 3):
    for a in (1, 9):
        L = {c: per_slice(T[c], d, a) for c in CS}
        line = " ".join("c%.2f:%6.2f" % (c, np.exp(L[c].mean())) for c in CS)
        cb = max(CS, key=lambda c: L[c].mean())
        g, lo, hi = boot(L[cb] - L[1.0])
        # cross-fitted c: per fold, the c maximizing the mean log n_eff over the other folds' slices
        lcf = np.array([L[max(CS, key=lambda c: np.mean([L[c][q] for q in S34 if fold[q] != fold[s]]))][s] for s in S34])
        g2, lo2, hi2 = boot(lcf - L[1.0])
        print(f"ds{d} a={a}: {line} | best c {cb:.2f}: {g:.3f} ({lo:.3f}-{hi:.3f}) | c cross-fitted: {g2:.3f} ({lo2:.3f}-{hi2:.3f})")
