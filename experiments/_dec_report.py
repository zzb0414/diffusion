# -*- coding: utf-8 -*-
"""DEC report: regressor vs diffusion (A), generative sampling vs Tweedie (B).

Implements ``Archive/DEC/PREREGISTRATION.md`` from ``runs/dec/A/eval/<arm>_s<seed>`` and ``runs/dec/B/s<seed>``
(``_dec_eval.py`` outputs). Every numeral is computed here. ``--part B`` reports part B alone (available first).
``--dry`` replaces each compared arm / decoder by its comparator (all ratios must read 1.000) and writes nothing.

    python experiments/_dec_report.py [--part A|B|all] [--dry]
"""
import sys, os, csv, json, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from scipy.stats import wilcoxon

ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--part", default="all", choices=["A", "B", "all"])
ap.add_argument("--dry", action="store_true")
args = ap.parse_args()
OUT, MD = "runs/dec/report", "WEEKEND_ITERATIONS.md"
HEAD = {"B": "## RUN 2026-10-29 — DEC part B: generative sampling vs one-shot Tweedie",
        "A": "## RUN 2026-10-30 — DEC part A: regressor vs diffusion"}
SEEDS, S34, DS, NBOOT = (101, 102, 103), list(range(34)), (1, 2, 3), 10000
RGRID = (0.0, 0.0833, 0.1667, 0.25, 0.3333, 0.5)
rng = np.random.default_rng(0)


def read(path):
    T = {}
    for r in csv.DictReader(open(os.path.join(path, "neff.csv"), encoding="utf-8")):
        k = (r["decoder"], int(r["dataset"]), int(r["slice"]), round(float(r["r"]), 4), int(r["a"]))
        T[k] = {c: (float(r[c]) if r[c] != "" else np.nan) for c in ("neff_signal", "band3", "walk", "sigma_hat")}
        T[k]["fold"] = int(r["fold"])
    return T


def stat(vals, ci=95):
    v = np.asarray(vals, float); ok = np.isfinite(v) & (v > 0)
    lr = np.log(v[ok]); lo = (100 - ci) / 2
    b = rng.choice(lr, size=(NBOOT, len(lr))).mean(1)
    p = float(wilcoxon(lr).pvalue) if np.any(lr != 0) else 1.0
    return dict(gmean=float(np.exp(lr.mean())), ci_lo=float(np.exp(np.percentile(b, lo))),
                ci_hi=float(np.exp(np.percentile(b, 100 - lo))), wins=int((lr > 0).sum()), n=len(lr), p=p,
                excluded=int((~ok).sum()))


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


def r_cf(T, dec, d, s, a):
    """grid r minimizing the summed normalized error (1/n_eff) over the OTHER folds' slices."""
    f = T[(dec, d, s, 0.0, a)]["fold"]
    others = [q for q in S34 if T[(dec, d, q, 0.0, a)]["fold"] != f]
    return min(RGRID, key=lambda r: sum(1 / T[(dec, d, q, r, a)]["neff_signal"]
                                        if np.isfinite(T[(dec, d, q, r, a)]["neff_signal"]) and T[(dec, d, q, r, a)]["neff_signal"] > 0
                                        else 0.0 for q in others))


def ratio(TA, decA, TB, decB, d, a, col="neff_signal", r=None):
    """per slice, geometric mean over seeds of col(A)/col(B); r=None -> each at its cross-fitted r(a)."""
    out = []
    for s in S34:
        lr = []
        for sd in SEEDS:
            ra = r_cf(TA[sd], decA, d, s, a) if r is None else r
            rb = r_cf(TB[sd], decB, d, s, a) if r is None else r
            x, y = TA[sd][(decA, d, s, ra, a)][col], TB[sd][(decB, d, s, rb, a)][col]
            lr.append(np.log(x / y) if np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0 else np.nan)
        out.append(float(np.exp(np.nanmean(lr))) if np.isfinite(lr).any() else np.nan)
    return out


f3 = lambda r, c=95: "%.3f× (%d%% CI %.3f–%.3f; %d/%d better%s)" % (
    r["gmean"], c, r["ci_lo"], r["ci_hi"], r["wins"], r["n"], "; %d excluded" % r["excluded"] if r["excluded"] else "")
os.makedirs(OUT, exist_ok=True)
summary, L = {}, []

# ---- part B ------------------------------------------------------------------------------------------
if args.part in ("B", "all"):
    TB = {sd: read(f"runs/dec/B/s{sd}") for sd in SEEDS}
    T1 = {sd: dict(TB[sd]) for sd in SEEDS}
    if args.dry:
        for sd in SEEDS:
            for k, v in list(TB[sd].items()):
                if k[0] == "tw":
                    for dec in ("ode", "anc", "anc8", "ancx"):
                        T1[sd][(dec,) + k[1:]] = v
    P = {c: stat(ratio(T1, "anc", TB, "tw", 1, 1, col=c, r=0.0)) for c in ("neff_signal", "band3")}
    for c, q in zip(P, holm([P[c]["p"] for c in P])):
        P[c]["p_holm"] = float(q)
    sec = {}
    for dec in ("anc", "ode", "anc8", "ancx"):
        for d in DS:
            for a in (1, 5, 9):
                for rs in ("0", "cf"):
                    for c in ("neff_signal", "band3"):
                        sec[(dec, d, a, rs, c)] = stat(ratio(T1, dec, TB, "tw", d, a, col=c, r=0.0 if rs == "0" else None))
    walk = np.array([TB[101][("tw", 1, s, 0.0, 1)]["walk"] for s in S34])
    lo_w = walk <= np.median(walk)
    strat = {}
    for dec in ("anc", "ode"):
        v = np.array(ratio(T1, dec, TB, "tw", 1, 1, r=0.0))
        strat[dec] = (stat(v[lo_w]), stat(v[~lo_w]))
    if args.dry:
        assert all(abs(v["gmean"] - 1) < 1e-12 for v in list(P.values()) + list(sec.values()))
        print("DRY RUN OK (B): every decoder := tw gives 1.000")
    else:
        verdict = lambda r: ("sampling worse" if r["p_holm"] < 0.05 and r["gmean"] < 1 else
                             "sampling better" if r["p_holm"] < 0.05 else "no significant difference")
        summary["B"] = dict(P=P, sec={"|".join(map(str, k)): v for k, v in sec.items()},
                            strat={k: v for k, v in strat.items()}, walk_median=float(np.median(walk)))
        L += ["", "---", "", HEAD["B"], "",
              "Pre-registered (`Archive/DEC/PREREGISTRATION.md`). DSM models (MSE, noise-normalized; seeds 101–103, "
              "iteration 2000), every held-out slice of datasets 1–3 decoded per repetition by: one-shot Tweedie (TW); a "
              "20-step deterministic reverse walk (ODE); a 20-step ancestral walk = one posterior sample (ANC); the mean of "
              "8 samples (ANC8); and ANC extrapolated below the trained noise floor to σ̂/10 (ANCX, off-distribution). The "
              "stochastic walk stops at the ladder bottom σ_floor = 0.0368: N2N never trains below the quietest "
              "acquisition. Per slice, ratios are geometric means over the 3 seeds. The dry run gave 1.000.", "",
              "**PRIMARY (dataset 1, a = 1, r = 0; n_eff ANC / TW, Holm over 2):**", "",
              "- overall (signal region): %s, p = %.2g, Holm %.2g → **%s**" % (f3(P["neff_signal"]), P["neff_signal"]["p"],
                                                                             P["neff_signal"]["p_holm"], verdict(P["neff_signal"])),
              "- finest band (0.25–0.75 cycles/pixel): %s, p = %.2g, Holm %.2g → **%s**" % (
                  f3(P["band3"]), P["band3"]["p"], P["band3"]["p_holm"], verdict(P["band3"])), "",
              "Decoder / TW, overall n_eff (geometric mean, slices better); r = 0 and each decoder at its cross-fitted r(a):", "",
              "| decoder | dataset | a=1, r=0 | a=5, r=0 | a=9, r=0 | a=1, cf r | a=5, cf r | a=9, cf r |",
              "|---|---:|---|---|---|---|---|---|"]
        for dec in ("anc", "anc8", "ode", "ancx"):
            for d in DS:
                L.append("| %s | %d | %s |" % (dec.upper(), d, " | ".join(
                    "%.3f (%d/%d)" % (sec[(dec, d, a, rs, "neff_signal")]["gmean"], sec[(dec, d, a, rs, "neff_signal")]["wins"],
                                      sec[(dec, d, a, rs, "neff_signal")]["n"]) for rs in ("0", "cf") for a in (1, 5, 9))))
        L += ["", "Finest band, decoder / TW at r = 0, a = 1 / 5 / 9: " + "; ".join(
            "%s ds%d: %s" % (dec.upper(), d, " / ".join("%.3f" % sec[(dec, d, a, "0", "band3")]["gmean"] for a in (1, 5, 9)))
            for dec in ("anc", "anc8", "ode") for d in DS) + ".",
              "Walk length (dataset 1, a = 1, r = 0): median σ̂/σ_floor %.2f; ANC/TW on short vs long walks: %.3f vs %.3f; "
              "ODE/TW: %.3f vs %.3f." % (np.median(walk), strat["anc"][0]["gmean"], strat["anc"][1]["gmean"],
                                         strat["ode"][0]["gmean"], strat["ode"][1]["gmean"]), "",
              "**Data:** `runs/dec/B/s{101,102,103}/neff.csv`, `%s/summary.json`. **Commands:** `bash runs/dec/queue.sh`; "
              "`python experiments/_dec_report.py --part B`." % OUT, ""]

# ---- part A ------------------------------------------------------------------------------------------
if args.part in ("A", "all"):
    ARMS = ("dsm", "dirl", "dirn", "blind")
    TA = {arm: {sd: read(f"runs/dec/A/eval/{arm}_s{sd}") for sd in SEEDS} for arm in ARMS}
    if args.dry:
        TA = {arm: TA["dsm"] for arm in ARMS}
    PA = stat(ratio(TA["dirn"], "tw", TA["dsm"], "tw", 1, 9), ci=90)
    vA = ("equivalent (σ-conditioned N2N regressor + dial)" if PA["ci_lo"] >= 0.95 and PA["ci_hi"] <= 1.05 else
          "regressor worse (diffusion training contributes)" if PA["ci_hi"] < 0.95 else
          "regressor better" if PA["ci_lo"] > 1.05 else "inconclusive")
    comps = (("dirn", "dsm"), ("dirl", "dsm"), ("dirn", "dirl"), ("blind", "dirn"))
    secA = {(x, y, d, a, c): stat(ratio(TA[x], "tw", TA[y], "tw", d, a, col=c), ci=90)
            for x, y in comps for d in DS for a in (1, 5, 9) for c in ("neff_signal", "band3")}
    absA = {(arm, d, a): float(np.exp(np.nanmean([np.log(TA[arm][sd][("tw", d, s, r_cf(TA[arm][sd], "tw", d, s, a), a)]["neff_signal"])
                                                  for sd in SEEDS for s in S34]))) for arm in ARMS for d in DS for a in (1, 9)}
    if args.dry:
        assert abs(PA["gmean"] - 1) < 1e-12 and all(abs(v["gmean"] - 1) < 1e-12 for v in secA.values())
        print("DRY RUN OK (A): every arm := dsm gives 1.000")
    else:
        summary["A"] = dict(P=PA, verdict=vA, sec={"|".join(map(str, k)): v for k, v in secA.items()},
                            abs={"|".join(map(str, k)): v for k, v in absA.items()})
        L += ["", "---", "", HEAD["A"], "",
              "Pre-registered (`Archive/DEC/PREREGISTRATION.md`). Dataset-1 training, MSE noise-normalized, 2000 iterations, "
              "seeds 101–103, four arms: DSM (x̂₀ = x̃ + σ² net, noise ladder), DIR_L (x̂₀ = net, same ladder), DIR_N (direct, "
              "native noise only: a classic σ-conditioned N2N regressor), BLIND (DIR_N without σ input). All decoded one-shot "
              "and scored at each arm's cross-fitted r(a). Per slice, geometric means over the 3 seeds. The dry run gave 1.000.", "",
              "**PRIMARY (DIR_N / DSM, dataset 1, a = 9):** %s → **%s**." % (f3(PA, 90), vA), "",
              "Ratios (geometric mean, 90% CI; overall n_eff at cross-fitted r):", "",
              "| comparison | dataset | a = 1 | a = 5 | a = 9 |", "|---|---:|---|---|---|"]
        name = {"dirn": "DIR_N", "dsm": "DSM", "dirl": "DIR_L", "blind": "BLIND"}
        for x, y in comps:
            for d in DS:
                L.append("| %s / %s | %d | %s |" % (name[x], name[y], d, " | ".join(
                    "%.3f (%.3f–%.3f)" % (secA[(x, y, d, a, "neff_signal")]["gmean"], secA[(x, y, d, a, "neff_signal")]["ci_lo"],
                                          secA[(x, y, d, a, "neff_signal")]["ci_hi"]) for a in (1, 5, 9))))
        L += ["", "Finest band, a = 1 / 9: " + "; ".join(
            "%s/%s ds%d: %.3f / %.3f" % (name[x], name[y], d, secA[(x, y, d, 1, "band3")]["gmean"], secA[(x, y, d, 9, "band3")]["gmean"])
            for x, y in comps for d in DS) + ".",
              "Absolute n_eff (geometric mean over slices and seeds), a = 1 / 9: " + "; ".join(
            "%s ds%d %.2f / %.2f" % (name[arm], d, absA[(arm, d, 1)], absA[(arm, d, 9)]) for arm in ARMS for d in DS) + ".", "",
              "**Data:** `runs/dec/A/<arm>_s<seed>/`, `runs/dec/A/eval/`, `%s/summary.json`. **Commands:** `bash runs/dec/queue.sh`; "
              "`python experiments/_dec_report.py --part A`." % OUT, ""]

if not args.dry and L:
    path = os.path.join(OUT, "summary.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(summary)
    json.dump(old, open(path, "w"), indent=2)
    md = open(MD, encoding="utf-8").read()
    parts = [p for p in ("B", "A") if p in summary and HEAD[p] not in md]
    if parts:
        txt = "\n".join(L)
        open(MD, "a", encoding="utf-8").write(txt)
    print("\n".join(L))
