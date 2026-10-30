# -*- coding: utf-8 -*-
"""BASE report (Archive/BASE/PREREGISTRATION.md): the method vs every baseline.

Reads runs/base/{classical,ours,n2s_eval}/neff.csv. For every method, its parameter (r for ours and Noise2Score,
the strength for each baseline) is chosen per (dataset, a, seed) as the grid value minimizing the summed normalized
error (1/n_eff) over the OTHER folds' slices. Seeded methods: per-slice geometric mean over seeds. ``--dry``
replaces ours by each baseline (every ratio must read 1.000) and writes nothing.

    python baselines/report.py [--dry]
"""
import os, sys, csv, json, argparse
import numpy as np
from scipy.stats import wilcoxon
import _common

os.chdir(_common.REPO)
ap = argparse.ArgumentParser()
ap.add_argument("--dry", action="store_true")
DRY = ap.parse_args().dry
OUT, MD = "runs/base/report", "WEEKEND_ITERATIONS.md"
HEAD = "## RUN 2026-10-30 — BASE: the method vs conventional and self-supervised baselines"
A_EVAL, DS, S34, NBOOT = _common.A_EVAL, (1, 2, 3), list(range(34)), 10000
NAMES = {"ours": "Ours (DSM + dial)", "avg": "Plain averaging", "blur": "Gaussian blur", "nlm": "Non-local means",
         "bm3d": "BM3D", "mppca": "MP-PCA", "lpca": "LPCA", "p2s": "Patch2Self", "n2s": "Noise2Score (r = 0)",
         "n2s_dial": "Noise2Score + dial"}
BASELINES = ("avg", "blur", "nlm", "bm3d", "mppca", "lpca", "p2s", "n2s")
rng = np.random.default_rng(0)

T = {}                                                   # (method, seed, d, s, a, param) -> (n_eff, band3, fold)
for tag in ("classical", "ours", "n2s2_eval"):        # n2s2 = Noise2Score version 2 (v1 failed its check)
    if not os.path.exists(f"runs/base/{tag}/neff.csv"):
        print(f"(missing runs/base/{tag}: its methods are skipped)")
        continue
    for r in csv.DictReader(open(f"runs/base/{tag}/neff.csv", encoding="utf-8")):
        T[(r["method"], int(r["seed"]), int(r["dataset"]), int(r["slice"]), int(r["a"]), float(r["param"]))] = (
            float(r["neff_signal"]), float(r["band3"]), int(r["fold"]))
params = {}
for (m, sd, d, s, a, p) in T:
    params.setdefault((m, sd), set()).add(p)
seeds = {m: sorted({sd for (mm, sd) in params if mm == m}) for m in {k[0] for k in params}}


def value(method, d, s, a, col=0):
    """per-slice value at the cross-fitted parameter; geometric mean over seeds. method 'n2s' = r 0, 'n2s_dial' = cf r."""
    base = "n2s" if method.startswith("n2s") else method
    logs = []
    for sd in seeds[base]:
        f = T[(base, sd, d, s, a, min(params[(base, sd)]))][2]
        if method == "n2s":
            p = 0.0
        else:
            others = [q for q in S34 if T[(base, sd, d, q, a, min(params[(base, sd)]))][2] != f]
            def cost(p):
                v = [T[(base, sd, d, q, a, p)][0] for q in others]
                return sum(1 / x if np.isfinite(x) and x > 0 else 0.0 for x in v)
            p = min(sorted(params[(base, sd)]), key=cost)
        v = T[(base, sd, d, s, a, p)][col]
        logs.append(np.log(v) if np.isfinite(v) and v > 0 else np.nan)
    return float(np.exp(np.nanmean(logs))) if np.isfinite(logs).any() else np.nan


def stat(vals):
    v = np.asarray(vals, float); ok = np.isfinite(v) & (v > 0); lr = np.log(v[ok])
    b = rng.choice(lr, size=(NBOOT, len(lr))).mean(1)
    p = float(wilcoxon(lr).pvalue) if np.any(lr != 0) else 1.0
    return dict(gmean=float(np.exp(lr.mean())), ci_lo=float(np.exp(np.percentile(b, 2.5))),
                ci_hi=float(np.exp(np.percentile(b, 97.5))), wins=int((lr > 0).sum()), n=len(lr), p=p,
                excluded=int((~ok).sum()))


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


BASELINES = tuple(b for b in BASELINES if b in seeds)
methods = ["ours"] + list(BASELINES) + (["n2s_dial"] if "n2s" in seeds else [])
V = {(m, d, s, a, c): value(m, d, s, a, c) for m in methods for d in DS for s in S34 for a in A_EVAL for c in (0, 1)}
ours = lambda b: b if DRY else "ours"
cmp = {(b, d, a, c): stat([V[(ours(b), d, s, a, c)] / V[(b, d, s, a, c)] for s in S34])
       for b in methods[1:] for d in DS for a in A_EVAL for c in (0, 1)}
prim = {b: cmp[(b, 1, 9, 0)] for b in BASELINES}
for b, q in zip(BASELINES, holm([prim[b]["p"] for b in BASELINES])):
    prim[b]["p_holm"] = float(q)
if DRY:
    assert all(abs(v["gmean"] - 1) < 1e-12 for v in cmp.values())
    print("DRY RUN OK: ours := each baseline gives 1.000")
    sys.exit(0)
gm_tab = {(m, d, a): float(np.exp(np.nanmean(np.log([V[(m, d, s, a, 0)] for s in S34])))) for m in methods for d in DS for a in A_EVAL}

os.makedirs(OUT, exist_ok=True)
K = lambda t: "|".join(map(str, t))
json.dump(dict(primary=prim, compare={K(k): v for k, v in cmp.items()}, gmean={K(k): v for k, v in gm_tab.items()}),
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
verdict = lambda r: ("ours better" if r["p_holm"] < 0.05 and r["gmean"] > 1 else
                     "baseline better" if r["p_holm"] < 0.05 else "no significant difference")
L = ["", "---", "", HEAD, "",
     "Pre-registered (`Archive/BASE/PREREGISTRATION.md`). Ours = DSM, MSE noise-normalized, dataset-1 training, seeds "
     "101–103, one-shot Tweedie + dial. Every method sees only the A half of each split; all methods are scored on the "
     "SAME fixed subset of 10 splits per a (unbiased, paired); every parameter is cross-fitted over the other folds' "
     "slices. Code: `baselines/`. The dry run gave 1.000.", "",
     "**Deviation from the pre-registration (Noise2Score).** The first implementation (annealed perturbation, network "
     "conditioned on σ̂ only) failed its sanity check: n_eff 0.03, i.e. far worse than the noisy input, with a final loss "
     "of 0.98 against 1.0 for no learning (runs kept in `runs/base/n2s_v1*`). Version 2 conditions the network on the "
     "total noise √(σ̂² + δ²) of the perturbed input with δ log-uniform in [0.05, 2]·σ̂; the AR-DAE objective is unchanged. "
     "It was validated on synthetic data with known ground truth (`baselines/_n2s_validate.py`, "
     "`runs/base/report/n2s_validation.txt`): it denoises white noise at SNR 1 and 0.1, also from a fixed set of 10 "
     "realizations, but with noise coloured like dataset 1 it is WORSE than the noisy input (error 1.25× the noise "
     "variance), while a Noise2Noise model on the same data is unaffected (0.002×). Noise2Score's Tweedie step assumes "
     "white noise of known σ; real ASL noise is coloured. The Noise2Score rows below are version 2.", "",
     "**PRIMARY (dataset 1, a = 9; n_eff ours / baseline, Holm over %d):**" % len(BASELINES), ""]
for b in BASELINES:
    r = prim[b]
    L.append("- vs %s: %.3f× (95%% CI %.3f–%.3f; ours better on %d/%d), p = %.2g, Holm %.2g → **%s**" % (
        NAMES[b], r["gmean"], r["ci_lo"], r["ci_hi"], r["wins"], r["n"], r["p"], r["p_holm"], verdict(r)))
L += ["", "Geometric-mean n_eff (signal region) by method:", "",
      "| method | " + " | ".join(f"ds{d} a={a}" for d in DS for a in A_EVAL) + " |",
      "|---|" + "---:|" * (len(DS) * len(A_EVAL))]
for m in methods:
    L.append("| %s | %s |" % (NAMES[m], " | ".join("%.2f" % gm_tab[(m, d, a)] for d in DS for a in A_EVAL)))
L += ["", "Ours / baseline, overall and finest band (geometric mean, slices where ours is better):", "",
      "| baseline | dataset | a = 1 | a = 3 | a = 5 | a = 9 | finest band a = 1 / 9 |", "|---|---:|---|---|---|---|---|"]
for b in methods[1:]:
    for d in DS:
        L.append("| %s | %d | %s | %.3f / %.3f |" % (NAMES[b], d, " | ".join(
            "%.3f (%d/%d)" % (cmp[(b, d, a, 0)]["gmean"], cmp[(b, d, a, 0)]["wins"], cmp[(b, d, a, 0)]["n"]) for a in A_EVAL),
            cmp[(b, d, 1, 1)]["gmean"], cmp[(b, d, 9, 1)]["gmean"]))
L += ["", "**Data:** `runs/base/{classical,ours,n2s_eval}/neff.csv`, `%s/summary.json`. **Commands:** "
      "`python baselines/evaluate.py --methods ...`; `bash runs/base/n2s_queue.sh`; `python baselines/report.py`." % OUT, ""]
if "n2s" in seeds and HEAD not in open(MD, encoding="utf-8").read():     # log only the complete report
    open(MD, "a", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
