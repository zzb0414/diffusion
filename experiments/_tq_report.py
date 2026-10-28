# -*- coding: utf-8 -*-
"""TQ report: reference quality (E1) and input SNR vs noise colour (E2).

Implements ``Archive/TQ/PREREGISTRATION.md``. E1 compares models trained with a
k-subset LOO target (``runs/tq/<arm>_k<k>/eval_it2000``) against the k = 9
models of the same arm (dataset 1: ``<arm dir>/ratio_eval``; datasets 2-3:
``runs/gen/<arm>``). E2 reads ``runs/degrade/<model>_<colour>_f<f>`` and
compares each slice's edge over the noise-matched blur between colours. Every
numeral is computed here; verdicts follow the pre-registered rules. ``--dry``
substitutes the comparator for the k-models (every ratio must read 1.000) and
writes nothing.

    python experiments/_tq_report.py [--dry]
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
OUT = "runs/tq/report"
MD = "WEEKEND_ITERATIONS.md"
HEAD = "## RUN 2026-10-27 — TQ results (reference quality vs input SNR vs noise colour)"
ARMS = {"A": ("L1", "runs/cv_r3"), "D": ("MSE, noise-normalized", "runs/cv_ablation/mse_norm")}
POOLDIR = {"A": "runs/mx_cv/A/eval_it2000", "D": "runs/mx_cv/D/eval_it2000"}
DS, S34, NBOOT = (1, 2, 3), list(range(34)), 10000
COLOURS, FACS = ("self", "white", "ds2"), ("2.0", "1.4")
LOCK, R0 = "neff_r0.3333", "neff_r0.0000"
rng = np.random.default_rng(0)


def read(path, ds_col):
    out = {}
    for r in csv.DictReader(open(os.path.join(path, "per_slice_ratio.csv"), encoding="utf-8")):
        if r["region"] == "signal":
            d = int(r["dataset"]) if ds_col else 1
            out[(d, int(r["slice"]), int(r["a"]))] = {k: float(v) for k, v in r.items() if v not in ("", "signal")}
    return out


def stat(vals, alt="two-sided"):
    lr = np.log(np.asarray(vals, float))
    b = rng.choice(lr, size=(NBOOT, len(lr))).mean(1)
    p = float(wilcoxon(lr, alternative=alt).pvalue) if np.any(lr != 0) else 1.0
    return dict(gmean=float(np.exp(lr.mean())), ci_lo=float(np.exp(np.percentile(b, 2.5))),
                ci_hi=float(np.exp(np.percentile(b, 97.5))), wins=int((lr > 0).sum()), n=len(lr), p=p)


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


gm = lambda v: float(np.exp(np.mean(np.log(v))))
f3 = lambda r: "%.3f× (95%% CI %.3f–%.3f; %d/%d better)" % (r["gmean"], r["ci_lo"], r["ci_hi"], r["wins"], r["n"])

# ---- E1 -------------------------------------------------------------------------------------------------
K9 = {k: {**read(os.path.join(d, "ratio_eval"), False),
          **{kk: v for kk, v in read(os.path.join("runs/gen", k), True).items() if kk[0] in (2, 3)}}
      for k, (_, d) in ARMS.items()}
KM = {(k, n): (K9[k] if DRY else read(f"runs/tq/{k}_k{n}/eval_it2000", True)) for k in ARMS for n in (1, 3)}
e1 = {(k, n, d, a, c): stat([KM[(k, n)][(d, s, a)][c] / K9[k][(d, s, a)][c] for s in S34])
      for k in ARMS for n in (1, 3) for d in DS for a in (1, 5, 9) for c in (LOCK, R0)}
abs_ne = {(k, n, d, a): gm([(K9[k] if n == 9 else KM[(k, n)])[(d, s, a)][LOCK] for s in S34])
          for k in ARMS for n in (1, 3, 9) for d in DS for a in (1, 5, 9)}
edge1 = {(k, n, d, a): gm([(lambda R: R[LOCK] / R["blur_matched"])((K9[k] if n == 9 else KM[(k, n)])[(d, s, a)]) for s in S34])
         for k in ARMS for n in (1, 3, 9) for d in DS for a in (1, 9)}
P = e1[("A", 1, 1, 1, LOCK)]
if DRY:
    assert all(abs(v["gmean"] - 1) < 1e-12 for v in e1.values())
    print("DRY RUN OK (E1): k-models := k=9 models gives 1.000 everywhere")

# ---- E2 -------------------------------------------------------------------------------------------------
BASE = {"one_A": K9["A"], "one_D": K9["D"],
        "pool_A": read(POOLDIR["A"], True), "pool_D": read(POOLDIR["D"], True)}
DEG = {(m, c, f): (BASE[m] if DRY else read(f"runs/degrade/{m}_{c}_f{f}", True))
       for m in BASE for c in COLOURS for f in FACS}
edge = lambda R, s, a: R[(1, s, a)][LOCK] / R[(1, s, a)]["blur_matched"]
sig_ok = {(m, c, f): float(np.median([DEG[(m, c, f)][(1, s, 1)]["sigma_hat"] / BASE[m][(1, s, 1)]["sigma_hat"]
                                       for s in S34])) for m in BASE for c in COLOURS for f in FACS}
e2 = {(m, f, a, c): stat([edge(DEG[(m, c, f)], s, a) / edge(DEG[(m, "self", f)], s, a) for s in S34])
      for m in BASE for f in FACS for a in (1, 9) for c in ("white", "ds2")}
for m in BASE:
    for f in FACS:
        for a in (1, 9):
            for c, q in zip(("white", "ds2"), holm([e2[(m, f, a, c)]["p"] for c in ("white", "ds2")])):
                e2[(m, f, a, c)]["p_holm"] = float(q)
gedge = {(m, c, f, a): gm([edge(DEG[(m, c, f)] if c else BASE[m], s, a) for s in S34])
         for m in BASE for c in (None,) + COLOURS for f in FACS for a in (1, 9)}
drop = {(m, c, f, a): stat([edge(DEG[(m, c, f)], s, a) / edge(BASE[m], s, a) for s in S34])
        for m in BASE for c in COLOURS for f in FACS for a in (1, 9)}
real = {(m, d, a): gm([(lambda R: R[LOCK] / R["blur_matched"])(
    (K9[m[-1]] if m.startswith("one") else BASE[m])[(d, s, a)]) for s in S34]) for m in BASE for d in (2, 3) for a in (1, 9)}
if DRY:
    assert all(abs(v["gmean"] - 1) < 1e-12 for v in e2.values())
    print("DRY RUN OK (E2): degraded := undegraded gives 1.000 everywhere")
    sys.exit(0)

# ---- verdicts -------------------------------------------------------------------------------------------
v1 = ("k = 1 worse" if P["p"] < 0.05 and P["gmean"] < 1 else "k = 1 better" if P["p"] < 0.05 else "no significant difference")
pw, pd2 = e2[("one_A", "2.0", 1, "white")], e2[("one_A", "2.0", 1, "ds2")]
v2 = {c: ("lower edge than self" if r["p_holm"] < 0.05 and r["gmean"] < 1 else
          "higher edge than self" if r["p_holm"] < 0.05 else "no significant difference") for c, r in (("white", pw), ("ds2", pd2))}

os.makedirs(OUT, exist_ok=True)
K = lambda t: "|".join(str(x) for x in t)
json.dump(dict(e1={K(k): v for k, v in e1.items()}, abs_neff={K(k): v for k, v in abs_ne.items()},
               edge_e1={K(k): v for k, v in edge1.items()}, e2={K(k): v for k, v in e2.items()},
               edge_e2={K(k): v for k, v in gedge.items()}, drop={K(k): v for k, v in drop.items()},
               sigma_factor={K(k): v for k, v in sig_ok.items()}, real={K(k): v for k, v in real.items()}),
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)

L = ["", "---", "", HEAD, "",
     "Pre-registered (`Archive/TQ/PREREGISTRATION.md`). E1: 20 models (arms A, D × k ∈ {1, 3} × 5 folds, 2000 iterations) "
     "trained with the LOO target averaged over a random k of the other 9 repetitions; compared with the k = 9 models of the "
     "same arm, all slices decoded by their held-out fold. E2: 24 evaluations (4 models × 3 noise colours × f ∈ {2.0, 1.4}) of "
     "dataset 1 with added independent noise. All evaluations passed their self-checks; the report's dry run gave 1.000.", "",
     "Achieved noise factor σ̂_degraded/σ̂ (median over slices): " + "; ".join(
         "f%s: " % f + ", ".join("%s %.3f" % (c, sig_ok[("one_A", c, f)]) for c in COLOURS) for f in FACS) + ".", "",
     "### E1 — does the quality of the training reference matter?", "",
     "**PRIMARY (arm A, dataset 1, r = 1/3, a = 1, signal; n_eff k=1 / k=9):** %s, Wilcoxon p = %.2g → **%s**." % (f3(P), P["p"], v1), "",
     "n_eff ratio vs k = 9 (r = 1/3; geometric mean, slices better, two-sided p):", "",
     "| arm | k | dataset | a = 1 | a = 5 | a = 9 | a = 1, r = 0 |", "|---|---:|---:|---|---|---|---|"]
for k in ARMS:
    for n in (1, 3):
        for d in DS:
            L.append("| (%s) | %d | %d | %s | %s |" % (k, n, d, " | ".join(
                "%.3f× (%d/34, p=%.1g)" % (e1[(k, n, d, a, LOCK)]["gmean"], e1[(k, n, d, a, LOCK)]["wins"], e1[(k, n, d, a, LOCK)]["p"])
                for a in (1, 5, 9)), "%.3f× (%d/34)" % (e1[(k, n, d, 1, R0)]["gmean"], e1[(k, n, d, 1, R0)]["wins"])))
L += ["", "Edge over the noise-matched blur (geometric mean of n_eff / n_eff_blur, r = 1/3) by target size k = 1 / 3 / 9:", ""]
for k in ARMS:
    L.append("- (%s): " % k + "; ".join("ds%d a=%d: %s" % (d, a, " / ".join("%.3f" % edge1[(k, n, d, a)] for n in (1, 3, 9)))
                                        for d in DS for a in (1, 9)))
L += ["", "### E2 — input SNR or noise colour?", "",
      "**PRIMARY (dataset1-only A, f = 2.0, a = 1, r = 1/3, signal; edge ratio colour / self, Holm over 2):**", "",
      "- white / self: %s, p = %.2g, Holm %.2g → **%s**" % (f3(pw), pw["p"], pw["p_holm"], v2["white"]),
      "- ds2 / self: %s, p = %.2g, Holm %.2g → **%s**" % (f3(pd2), pd2["p"], pd2["p_holm"], v2["ds2"]), "",
      "Edge over the noise-matched blur (geometric mean), undegraded → degraded by colour, with the real datasets 2/3 for reference:", "",
      "| model | a | undegraded | self f1.4 | white f1.4 | ds2 f1.4 | self f2.0 | white f2.0 | ds2 f2.0 | real ds2 | real ds3 |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for m in BASE:
    for a in (1, 9):
        L.append("| %s | %d | %.3f | %s | %.3f | %.3f |" % (m, a, gedge[(m, None, "2.0", a)], " | ".join(
            "%.3f" % gedge[(m, c, f, a)] for f in ("1.4", "2.0") for c in COLOURS), real[(m, 2, a)], real[(m, 3, a)]))
L += ["", "Edge ratio colour / self for every model (geometric mean, slices better, Holm over the 2 colours):", "",
      "| model | f | a | white / self | ds2 / self |", "|---|---:|---:|---|---|"]
for m in BASE:
    for f in FACS:
        for a in (1, 9):
            L.append("| %s | %s | %d | %s |" % (m, f, a, " | ".join(
                "%.3f× (%d/34, Holm %.1g)" % (e2[(m, f, a, c)]["gmean"], e2[(m, f, a, c)]["wins"], e2[(m, f, a, c)]["p_holm"])
                for c in ("white", "ds2"))))
L += ["", "**Data:** `runs/tq/<arm>_k<k>/eval_it2000/`, `runs/degrade/`, `%s/summary.json`. **Commands:** "
      "`bash runs/tq_queue.sh`; `python experiments/_tq_report.py`." % OUT, ""]
if HEAD not in open(MD, encoding="utf-8").read():
    open(MD, "a", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
