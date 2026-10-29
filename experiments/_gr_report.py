# -*- coding: utf-8 -*-
"""GROUND report: KR (is target size k marginal?) and SM (denoise the mean vs denoise each).

Implements ``Archive/GROUND/PREREGISTRATION.md`` from ``runs/gr/kr_eval`` and ``runs/gr/sm_eval``
(``_gr_eval.py`` outputs). Every numeral is computed here; verdicts follow the pre-registered rules.
``--dry`` replaces every k = 1 / SUB table by its comparator (all ratios must read 1.000) and writes nothing.

    python experiments/_gr_report.py [--dry]
"""
import sys, os, csv, json, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from scipy.stats import wilcoxon

ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--dry", action="store_true")
DRY = ap.parse_args().dry
OUT, MD = "runs/gr/report", "WEEKEND_ITERATIONS.md"
HEAD = "## RUN 2026-10-29 — GROUND results (KR: is k marginal? SM: denoise the mean vs denoise each)"
RATIOS = (0.0, 1 / 12, 1 / 6, 0.25, 1 / 3, 0.5)
COL = lambda r: "neff_r%.4f" % r
S34, NBOOT = list(range(34)), 10000
rng = np.random.default_rng(0)


def read(path):
    T = {}
    for r in csv.DictReader(open(os.path.join(path, "neff.csv"), encoding="utf-8")):
        if r["region"] == "signal":
            T[(int(r["dataset"]), int(r["slice"]), int(r["a"]))] = {c: float(r[c]) for c in map(COL, RATIOS)}
    return T


def stat(vals, ci=95):
    """Ratios with a non-finite n_eff (estimated error <= 0: below measurable) are excluded and counted."""
    v = np.asarray(vals, float)
    ok = np.isfinite(v) & (v > 0)
    lr = np.log(v[ok])
    b = rng.choice(lr, size=(NBOOT, len(lr))).mean(1)
    lo = (100 - ci) / 2
    p = float(wilcoxon(lr).pvalue) if np.any(lr != 0) else 1.0
    return dict(gmean=float(np.exp(lr.mean())), ci_lo=float(np.exp(np.percentile(b, lo))),
                ci_hi=float(np.exp(np.percentile(b, 100 - lo))), wins=int((lr > 0).sum()), n=len(lr), p=p,
                excluded=int((~ok).sum()))


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


gm = lambda v: float(np.exp(np.mean(np.log(v))))
f3 = lambda r, c=95: "%.3f× (%d%% CI %.3f–%.3f; %d/%d better)" % (r["gmean"], c, r["ci_lo"], r["ci_hi"], r["wins"], r["n"])

# ---- KR ----------------------------------------------------------------------------------------------
E = "runs/gr/kr_eval"
IT_SEED = {101: (500, 1000, 1500, 2000), 102: (500, 1000, 1500, 2000, 3000, 4000), 103: (500, 1000, 1500, 2000, 3000, 4000)}
KR = {(k, s, it): read(f"{E}/k{k}_s{s}/it{it}") for k in (1, 9) for s in IT_SEED for it in IT_SEED[s]}
FR = {(k, f, it): read(f"{E}/k{k}_f{f}/it{it}") for k in (1, 9) for f in ("0.5", "0.25") for it in (1000, 2000)}
if DRY:
    for (k, s, it) in list(KR):
        KR[(1, s, it)] = KR[(9, s, it)]
    for (k, f, it) in list(FR):
        FR[(1, f, it)] = FR[(9, f, it)]

ratio = lambda A, B, a, c: [A[(1, s, a)][c] / B[(1, s, a)][c] for s in S34]
seeds_at = lambda it: [s for s in IT_SEED if it in IT_SEED[s]]


def kr_ratio(it, a, c, seeds=None):
    """per slice: geometric mean over seeds of n_eff(k=1)/n_eff(k=9)."""
    seeds = seeds or seeds_at(it)
    R = np.array([ratio(KR[(1, s, it)], KR[(9, s, it)], a, c) for s in seeds])
    return list(np.exp(np.log(R).mean(0)))


def best_r(T, a, d=1):
    return max(RATIOS, key=lambda r: gm([T[(d, s, a)][COL(r)] for s in S34]))


P1 = stat(kr_ratio(2000, 1, COL(1 / 3)), ci=90)
v1 = ("k marginal (equivalent within ±5%)" if P1["ci_lo"] >= 0.95 and P1["ci_hi"] <= 1.05 else
      "k = 1 worse" if P1["ci_hi"] < 0.95 else "k = 1 better" if P1["ci_lo"] > 1.05 else "inconclusive")
kr_tab = {}
for it in (500, 1000, 1500, 2000, 3000, 4000):
    for a in (1, 5, 9):
        kr_tab[(it, a, "r13")] = stat(kr_ratio(it, a, COL(1 / 3)), ci=90)
        R = []
        for s in seeds_at(it):
            r1, r9 = best_r(KR[(1, s, it)], a), best_r(KR[(9, s, it)], a)
            R.append([KR[(1, s, it)][(1, q, a)][COL(r1)] / KR[(9, s, it)][(1, q, a)][COL(r9)] for q in S34])
        kr_tab[(it, a, "best")] = stat(list(np.exp(np.log(np.array(R)).mean(0))), ci=90)
per_seed = {s: stat(ratio(KR[(1, s, 2000)], KR[(9, s, 2000)], 1, COL(1 / 3)), ci=90) for s in IT_SEED}
floor = {s: stat(ratio(KR[(9, s, 2000)], KR[(9, 101, 2000)], 1, COL(1 / 3)), ci=90) for s in (102, 103)}
floor1 = {s: stat(ratio(KR[(1, s, 2000)], KR[(1, 101, 2000)], 1, COL(1 / 3)), ci=90) for s in (102, 103)}
fr_tab = {(f, it, a): stat(ratio(FR[(1, f, it)], FR[(9, f, it)], a, COL(1 / 3)), ci=90)
          for f in ("0.5", "0.25") for it in (1000, 2000) for a in (1, 5, 9)}
abs_ne = {(k, it, a): gm([KR[(k, 101 if it <= 2000 else 102, it)][(1, s, a)][COL(1 / 3)] for s in S34])
          for k in (1, 9) for it in (500, 1000, 2000, 4000) for a in (1, 9)}
fr_abs = {(k, f, a): gm([FR[(k, f, 2000)][(1, s, a)][COL(1 / 3)] for s in S34]) for k in (1, 9) for f in ("0.5", "0.25") for a in (1, 9)}

# ---- SM ----------------------------------------------------------------------------------------------
SE = "runs/gr/sm_eval"
SM = {(arm, tr, it): read(f"{SE}/{arm}_{tr}/it{it}") for arm in ("per", "sub") for tr in ("ds1", "pool") for it in (2000, 3000)}
if DRY:
    for tr in ("ds1", "pool"):
        for it in (2000, 3000):
            SM[("sub", tr, it)] = SM[("per", tr, it)]


def sm_ratio(tr, it, a, d=1, rsel="best"):
    S, P = SM[("sub", tr, it)], SM[("per", tr, it)]
    rs = best_r(S, a, d) if rsel == "best" else rsel
    rp = best_r(P, a, d) if rsel == "best" else rsel
    return [S[(d, s, a)][COL(rs)] / P[(d, s, a)][COL(rp)] for s in S34], rs, rp


v2a, rs2a, rp2a = sm_ratio("pool", 3000, 9)
P2a = stat(v2a)
v2b = list(np.exp(np.log(v2a) - np.log(sm_ratio("ds1", 3000, 9)[0])))
P2b = stat(v2b)
for r, q in zip((P2a, P2b), holm([P2a["p"], P2b["p"]])):
    r["p_holm"] = float(q)
vd = lambda r, pos, neg: pos if r["p_holm"] < 0.05 and r["gmean"] > 1 else neg if r["p_holm"] < 0.05 else "no significant difference"
v2a_t = vd(P2a, "SUB better", "PER better")
v2b_t = vd(P2b, "pooling helps SUB more (data-limited)", "pooling helps PER more")
sm_tab = {}
for tr, ds_list in (("ds1", (1,)), ("pool", (1, 2, 3))):
    for d in ds_list:
        for it in (2000, 3000):
            for a in (1, 2, 3, 5, 7, 9):
                for rsel in ("best", 0.0, 1 / 3):
                    v, rs, rp = sm_ratio(tr, it, a, d, rsel)
                    sm_tab[(tr, d, it, a, rsel)] = dict(stat(v), r_sub=rs, r_per=rp)
sm_abs = {(arm, tr, d, a): gm([SM[(arm, tr, 3000)][(d, s, a)][COL(best_r(SM[(arm, tr, 3000)], a, d))] for s in S34])
          for arm in ("per", "sub") for tr, dl in (("ds1", (1,)), ("pool", (1, 2, 3))) for d in dl for a in (1, 5, 9)}
# augmentation (+ ladder/3) effect at iteration 2000: PER-aug vs the existing no-aug recipe
noaug_pool = read(f"{SE}/per_pool_noaug/it2000")
aug = {}
for name, A, B, dl in (("dataset-1 training", SM[("per", "ds1", 2000)], KR[(9, 101, 2000)], (1,)),
                       ("pooled training", SM[("per", "pool", 2000)], noaug_pool, (1, 2, 3))):
    for d in dl:
        for a in (1, 9):
            ra, rb = best_r(A, a, d), best_r(B, a, d)
            aug[(name, d, a)] = stat([A[(d, s, a)][COL(ra)] / B[(d, s, a)][COL(rb)] for s in S34])

if DRY:
    assert abs(P1["gmean"] - 1) < 1e-12 and all(abs(v["gmean"] - 1) < 1e-12 for v in fr_tab.values())
    assert abs(P2a["gmean"] - 1) < 1e-12 and abs(P2b["gmean"] - 1) < 1e-12
    assert all(abs(v["gmean"] - 1) < 1e-12 for k, v in sm_tab.items())
    print("DRY RUN OK: k=1 := k=9 and SUB := PER give 1.000 everywhere")
    sys.exit(0)

os.makedirs(OUT, exist_ok=True)
K = lambda t: "|".join(str(x) for x in t)
json.dump(dict(P1=P1, v1=v1, kr={K(k): v for k, v in kr_tab.items()}, per_seed={str(k): v for k, v in per_seed.items()},
               floor_k9={str(k): v for k, v in floor.items()}, floor_k1={str(k): v for k, v in floor1.items()},
               frac={K(k): v for k, v in fr_tab.items()}, kr_abs={K(k): v for k, v in abs_ne.items()},
               frac_abs={K(k): v for k, v in fr_abs.items()}, P2a=dict(P2a, r_sub=rs2a, r_per=rp2a), P2b=P2b,
               sm={K(k): v for k, v in sm_tab.items()}, sm_abs={K(k): v for k, v in sm_abs.items()},
               aug={K(k): v for k, v in aug.items()}), open(os.path.join(OUT, "summary.json"), "w"), indent=2)

L = ["", "---", "", HEAD, "",
     "Pre-registered (`Archive/GROUND/PREREGISTRATION.md`, written before any run). 60 new models: KR-seed (k ∈ {1, 9} × "
     "seeds 102, 103 × 5 folds, 4000 iterations), KR-frac (k ∈ {1, 9} × training fraction 0.5, 0.25 × 5 folds, 2000 iterations), "
     "SM (PER and SUB, dihedral augmentation, ladder down to σ̂_min/3, dataset 1 and pooled datasets 1–3, 3000 iterations); "
     "seed 101 comes from the existing `runs/cv_r3` (k = 9) and `runs/tq/A_k1` (k = 1). All evaluated with `_gr_eval.py` "
     "(reproduces `evaluate_slice` n_eff to 6e-15); `_gr_train.py` reproduces `_cv_train.py` and `_mx_train.py` exactly on "
     "deterministic CPU. The report's dry run gave 1.000. Note: GPU training is not bit-reproducible run to run (also for the "
     "original scripts), so every seed is an independent draw.", "",
     "### KR — is the target size k marginal?", "",
     "**PRIMARY (dataset 1, a = 1, r = 1/3, iteration 2000; per slice the geometric mean over seeds 101–103 of "
     "n_eff(k=1)/n_eff(k=9)):** %s → **%s**." % (f3(P1, 90), v1), "",
     "Per seed (same read-out): " + "; ".join("seed %d %s" % (s, f3(per_seed[s], 90)) for s in IT_SEED) + ".",
     "Seed-to-seed noise floor (same k, seed s / seed 101): k = 9: " + ", ".join("%d: %.3f×" % (s, floor[s]["gmean"]) for s in floor)
     + "; k = 1: " + ", ".join("%d: %.3f×" % (s, floor1[s]["gmean"]) for s in floor1) + ".", "",
     "Learning curve, n_eff(k=1)/n_eff(k=9) (geometric mean, 90% CI; seeds 101–103 up to 2000, 102–103 beyond; "
     "at r = 1/3 and at each k's best grid r):", "",
     "| iteration | a = 1, r = 1/3 | a = 9, r = 1/3 | a = 1, best r | a = 9, best r |", "|---:|---|---|---|---|"]
for it in (500, 1000, 1500, 2000, 3000, 4000):
    L.append("| %d | %s |" % (it, " | ".join("%.3f (%.3f–%.3f)" % (kr_tab[(it, a, c)]["gmean"], kr_tab[(it, a, c)]["ci_lo"], kr_tab[(it, a, c)]["ci_hi"])
                                            for c, a in (("r13", 1), ("r13", 9), ("best", 1), ("best", 9)))))
L += ["", "Absolute n_eff (geometric mean, r = 1/3; seed 101 ≤ 2000, seed 102 at 4000), k = 1 / k = 9: " + "; ".join(
      "it %d a=%d: %.2f / %.2f" % (it, a, abs_ne[(1, it, a)], abs_ne[(9, it, a)]) for it in (500, 1000, 2000, 4000) for a in (1, 9)) + ".", "",
      "Less training data (seed 101), n_eff(k=1)/n_eff(k=9) at r = 1/3 (90% CI):", "",
      "| training fraction | iteration | a = 1 | a = 5 | a = 9 |", "|---:|---:|---|---|---|"]
for f in ("0.5", "0.25"):
    for it in (1000, 2000):
        L.append("| %s | %d | %s |" % (f, it, " | ".join("%.3f (%.3f–%.3f)" % (fr_tab[(f, it, a)]["gmean"], fr_tab[(f, it, a)]["ci_lo"],
                                                                            fr_tab[(f, it, a)]["ci_hi"]) for a in (1, 5, 9))))
L += ["", "Absolute n_eff at 2000 (r = 1/3), k = 1 / k = 9: " + "; ".join(
      "frac %s a=%d: %.2f / %.2f" % (f, a, fr_abs[(1, f, a)], fr_abs[(9, f, a)]) for f in ("0.5", "0.25") for a in (1, 9))
      + "; full data (seed 101) a=1: %.2f / %.2f, a=9: %.2f / %.2f." % (abs_ne[(1, 2000, 1)], abs_ne[(9, 2000, 1)], abs_ne[(1, 2000, 9)], abs_ne[(9, 2000, 9)]), "",
      "### SM — denoise the mean (SUB) vs denoise each, then average (PER)", "",
      "**PRIMARY 2a (pooled training, dataset 1, a = 9, iteration 3000, each arm at its best grid r: SUB r = %.3f, PER r = %.3f):** "
      "%s, p = %.2g, Holm %.2g → **%s**." % (rs2a, rp2a, f3(P2a), P2a["p"], P2a["p_holm"], v2a_t),
      "**PRIMARY 2b (data limitation: SUB/PER pooled ÷ SUB/PER dataset-1-only, dataset 1, a = 9):** %s, p = %.2g, Holm %.2g → **%s**."
      % (f3(P2b), P2b["p"], P2b["p_holm"], v2b_t), "",
      "SUB / PER n_eff at each arm's best r (iteration 3000; geometric mean, slices better, two-sided p):", "",
      "| training | dataset | a = 1 | a = 2 | a = 3 | a = 5 | a = 7 | a = 9 |", "|---|---:|---|---|---|---|---|---|"]
for tr, dl in (("ds1", (1,)), ("pool", (1, 2, 3))):
    for d in dl:
        L.append("| %s | %d | %s |" % ("dataset 1" if tr == "ds1" else "pooled", d, " | ".join(
            "%.3f× (%d/34, p=%.1g)" % (sm_tab[(tr, d, 3000, a, "best")]["gmean"], sm_tab[(tr, d, 3000, a, "best")]["wins"],
                                       sm_tab[(tr, d, 3000, a, "best")]["p"]) for a in (1, 2, 3, 5, 7, 9))))
L += ["", "Same at fixed r (iteration 3000), a = 1 / 5 / 9: " + "; ".join(
      "%s ds%d r=%s: %s" % (tr, d, "0" if rs == 0 else "1/3", " / ".join("%.3f" % sm_tab[(tr, d, 3000, a, rs)]["gmean"] for a in (1, 5, 9)))
      for tr, dl in (("ds1", (1,)), ("pool", (1, 2, 3))) for d in dl for rs in (0.0, 1 / 3)) + ".",
      "Iteration 2000, best r, a = 1 / 5 / 9: " + "; ".join(
      "%s ds%d: %s" % (tr, d, " / ".join("%.3f" % sm_tab[(tr, d, 2000, a, "best")]["gmean"] for a in (1, 5, 9)))
      for tr, dl in (("ds1", (1,)), ("pool", (1, 2, 3))) for d in dl) + ".",
      "Absolute n_eff at best r (iteration 3000), PER / SUB: " + "; ".join(
      "%s ds%d a=%d: %.2f / %.2f" % (tr, d, a, sm_abs[("per", tr, d, a)], sm_abs[("sub", tr, d, a)])
      for tr, dl in (("ds1", (1,)), ("pool", (1, 2, 3))) for d in dl for a in (1, 9)) + ".",
      "Augmentation + ladder to σ̂/3 vs the existing PER recipe (iteration 2000, best r): " + "; ".join(
      "%s ds%d a=%d %.3f× (%d/%d)" % (n, d, a, v["gmean"], v["wins"], v["n"]) for (n, d, a), v in aug.items()) + ".", "",
      "**Data:** `runs/gr/{kr,sm}/`, `runs/gr/{kr_eval,sm_eval}/`, `%s/summary.json`. **Commands:** `bash runs/gr/queue_a.sh`; "
      "`bash runs/gr/queue_b.sh`; `bash runs/gr/queue_eval.sh`; `python experiments/_gr_report.py`." % OUT, ""]
if HEAD not in open(MD, encoding="utf-8").read():
    open(MD, "a", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
