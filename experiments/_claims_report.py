# -*- coding: utf-8 -*-
"""CLAIMS report (Archive/CLAIMS/PREREGISTRATION.md): LC (descriptive), C3, C2, C7, plus the exploratory C2-LC.

Natural images: r chosen WITHOUT ground truth on Set12 (summed split-half n_eff), scores on BSD68 (paired), as NAT.
ASL (C2): DEC-A rules (cross-fitted grid r per slice fold, per-slice geometric mean over seeds, 90% CI equivalence).
Each part is reported when its data exist; a part's log section is written once, when complete.
``--dry`` replaces every compared arm by its comparator (all differences 0 dB / ratios 1) and writes nothing.

    python experiments/_claims_report.py [--dry]
"""
import os, sys, csv, json, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from scipy.stats import wilcoxon

ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); DRY = ap.parse_args().dry
MD, OUT = "WEEKEND_ITERATIONS.md", "runs/claims/report"
rng = np.random.default_rng(0)
summary, sections = {}, {}


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


def dstat(d):
    b = rng.choice(d, size=(10000, len(d))).mean(1)
    return dict(mean=float(d.mean()), ci_lo=float(np.percentile(b, 2.5)), ci_hi=float(np.percentile(b, 97.5)),
                wins=int((d > 0).sum()), n=len(d), p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)


fdb = lambda d: "%+.2f dB (95%% CI %+.2f to %+.2f; %d/%d images)" % (d["mean"], d["ci_lo"], d["ci_hi"], d["wins"], d["n"])


# ---------------- natural-image helpers ----------------
def nat_read(path):
    T = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        T[(r["split"], int(r["image"]), round(float(r["r"]), 4), int(r["a"]))] = (float(r["psnr"]), float(r["mse_sh"]), float(r["s2"]))
    return T


def nat_psnr(T, a):
    """BSD68 PSNR per image at the r chosen without ground truth on Set12."""
    rs = sorted({k[2] for k in T})
    n12 = len({k[1] for k in T if k[0] == "set12"}); n68 = len({k[1] for k in T if k[0] == "bsd68"})
    r = max(rs, key=lambda q: sum(s2 / h if h > 0 else 0 for _, h, s2 in (T[("set12", i, q, a)] for i in range(n12))))
    return np.array([T[("bsd68", i, r, a)][0] for i in range(n68)]), r


# ---------------- LC ----------------
lc_paths = {(m, it): (f"runs/nat/lc/it{it}/W05/{m}.csv" if it < 3000 else f"runs/nat/eval/W05/{m}.csv")
            for m in ("ours", "sup", "sub", "subsup") for it in (1000, 2000, 3000)}
if all(os.path.exists(p) for p in lc_paths.values()):
    lc = {(m, it, a): float(nat_psnr(nat_read(p), a)[0].mean()) for (m, it), p in lc_paths.items() for a in (1, 9)}
    summary["LC"] = {"|".join(map(str, k)): v for k, v in lc.items()}
    sections["LC"] = ["## RUN 2026-11-02 — CLAIMS LC: were the natural-image models still improving at 3000 iterations?", "",
                      "Descriptive (`Archive/CLAIMS/PREREGISTRATION.md`). W05, BSD68 mean PSNR (dB) at the r chosen without ground "
                      "truth, checkpoints 1000 / 2000 / 3000:", "", "| model | a = 1 | a = 9 |", "|---|---|---|"] + [
        "| %s | %s | %s |" % (m, " / ".join("%.2f" % lc[(m, it, 1)] for it in (1000, 2000, 3000)),
                             " / ".join("%.2f" % lc[(m, it, 9)] for it in (1000, 2000, 3000))) for m in ("ours", "sup", "sub", "subsup")]

# ---------------- C3 ----------------
c3p = {(t, arm): f"runs/nat/eval_c3/{t}/{arm}/{m}.csv" for t in ("W05", "W035", "W07", "C05")
       for arm, m in (("ladder", "ours"), ("noladder", "ours"), ("n2n", "n2n"))}
c3p = {k: (v.replace("/n2n/n2n.csv", "/ladder/n2n.csv")) for k, v in c3p.items()}
if all(os.path.exists(p) for p in c3p.values()):
    P = {k: {a: nat_psnr(nat_read(p), a) for a in (1, 9)} for k, p in c3p.items()}
    prim = {}
    for t in ("W07", "C05"):
        for a in (1, 9):
            lad = P[(t, "ladder")][a][0]; nol = P[(t, "noladder")][a][0]
            prim[(t, a)] = dstat(np.zeros_like(lad) if DRY else lad - nol)
    for k, q in zip(prim, holm([v["p"] for v in prim.values()])):
        prim[k]["p_holm"] = float(q)
    summary["C3"] = {"|".join(map(str, k)): v for k, v in prim.items()}
    sec = ["## RUN 2026-11-02 — CLAIMS C3: does training across noise levels give robustness to shift?", "",
           "Pre-registered (`Archive/CLAIMS/PREREGISTRATION.md`). Trained at W05 (σ 0.5 white); OURS = with the noise ladder, "
           "NOLADDER = native noise only, N2N = classic Noise2Noise. Tested in distribution (W05) and shifted (W035 σ 0.35, "
           "W07 σ 0.7, C05 ASL-coloured).", "", "**PRIMARY (PSNR(OURS) − PSNR(NOLADDER), Holm over 4):**", ""]
    for (t, a), v in prim.items():
        sec.append("- %s, a = %d: %s, p = %.2g, Holm %.2g" % (t, a, fdb(v), v["p"], v["p_holm"]))
    sec += ["", "Mean BSD68 PSNR (dB), a = 1 / a = 9:", "", "| test set | OURS (ladder) | NOLADDER | N2N (classic) |", "|---|---|---|---|"]
    for t in ("W05", "W035", "W07", "C05"):
        sec.append("| %s | %s |" % (t, " | ".join("%.2f / %.2f" % (P[(t, arm)][1][0].mean(), P[(t, arm)][9][0].mean())
                                                for arm in ("ladder", "noladder", "n2n"))))
    sections["C3"] = sec

# ---------------- C2 (ASL) ----------------
S34, SEEDS, RG = list(range(34)), (101, 102, 103), (0.0, 0.0833, 0.1667, 0.25, 0.3333, 0.5)


def asl_read(path):
    T = {}
    for r in csv.DictReader(open(os.path.join(path, "neff.csv"), encoding="utf-8")):
        if r["decoder"] == "tw":
            T[(int(r["dataset"]), int(r["slice"]), round(float(r["r"]), 4), int(r["a"]))] = (float(r["neff_signal"]), int(r["fold"]))
    return T


def r_cf(T, d, s, a):
    f = T[(d, s, 0.0, a)][1]
    return min(RG, key=lambda r: sum(1 / T[(d, q, r, a)][0] if T[(d, q, r, a)][0] > 0 and np.isfinite(T[(d, q, r, a)][0]) else 0
                                     for q in S34 if T[(d, q, 0.0, a)][1] != f))


def asl_ratio(TA, TB, d, a):
    out = []
    for s in S34:
        lr = [np.log(TA[sd][(d, s, r_cf(TA[sd], d, s, a), a)][0] / TB[sd][(d, s, r_cf(TB[sd], d, s, a), a)][0]) for sd in SEEDS]
        out.append(float(np.exp(np.mean(lr))))
    v = np.log(out); b = rng.choice(v, size=(10000, len(v))).mean(1)
    return dict(gmean=float(np.exp(v.mean())), ci_lo=float(np.exp(np.percentile(b, 5))), ci_hi=float(np.exp(np.percentile(b, 95))),
                wins=int((v > 0).sum()), n=len(v))


c2p = {arm: [f"runs/dec/A/eval/{arm}_s{s}" for s in SEEDS] for arm in ("resid", "dsm", "dirl")}
if all(os.path.exists(os.path.join(p, "neff.csv")) for ps in c2p.values() for p in ps):
    TT = {arm: {sd: asl_read(p) for sd, p in zip(SEEDS, ps)} for arm, ps in c2p.items()}
    if DRY:
        TT["resid"] = TT["dsm"]
    res = {(x, y, d, a): asl_ratio(TT[x], TT[y], d, a) for x, y in (("resid", "dsm"), ("resid", "dirl")) for d in (1, 2, 3) for a in (1, 9)}
    P0 = res[("resid", "dsm", 1, 9)]
    verdict = ("equivalent (residual learning is the mechanism; σ² scaling adds nothing)" if P0["ci_lo"] >= 0.95 and P0["ci_hi"] <= 1.05
               else "σ² scaling contributes (residual worse)" if P0["ci_hi"] < 0.95 else
               "residual better" if P0["ci_lo"] > 1.05 else "inconclusive")
    if DRY:
        assert all(abs(v["gmean"] - 1) < 1e-12 for k, v in res.items() if k[1] == "dsm")
    summary["C2"] = dict(verdict=verdict, ratios={"|".join(map(str, k)): v for k, v in res.items()})
    sections["C2"] = ["## RUN 2026-11-02 — CLAIMS C2: residual parametrization (x̂₀ = x̃ + net) vs score (x̃ + σ² net)", "",
                      "Pre-registered (`Archive/CLAIMS/PREREGISTRATION.md`). ASL, dataset-1 training, MSE noise-normalized, noise ladder, "
                      "seeds 101–103, 2000 iterations; DSM and DIR_L from DEC part A.", "",
                      "**PRIMARY (RESID / DSM, dataset 1, a = 9):** %.3f× (90%% CI %.3f–%.3f; %d/34 better) → **%s**." % (
                          P0["gmean"], P0["ci_lo"], P0["ci_hi"], P0["wins"], verdict), "",
                      "| comparison | dataset | a = 1 | a = 9 |", "|---|---:|---|---|"] + [
        "| %s / %s | %d | %s |" % (x.upper(), y.upper(), d, " | ".join("%.3f (%.3f–%.3f)" % (res[(x, y, d, a)]["gmean"], res[(x, y, d, a)]["ci_lo"],
                                                                                         res[(x, y, d, a)]["ci_hi"]) for a in (1, 9)))
        for x, y in (("resid", "dsm"), ("resid", "dirl")) for d in (1, 2, 3)]

# ---------------- C2-LC (exploratory) ----------------
def bvc(T, d):
    P = []
    for s in S34:
        pts = [(a, r, 1 / T[(d, s, r, a)][0]) for a in (1, 2, 3, 5, 7, 9) for r in RG if (d, s, r, a) in T and T[(d, s, r, a)][0] > 0]
        X = np.array([[(1 - r) ** 2, (1 - r) ** 2 / a, 2 * r * (1 - r) / a] for a, r, m in pts])
        y = np.array([m - r * r / a for a, r, m in pts])
        P.append(np.linalg.lstsq(X, y, rcond=None)[0])
    return np.median(P, 0)


lcp = {(arm, sd, it): (f"runs/claims/c2_lc/{arm}_s{sd}/it{it}" if it < 2000 else f"runs/dec/A/eval/{arm}_s{sd}")
       for arm in ("dsm", "dirl") for sd in (102, 103) for it in (500, 1000, 1500, 2000)}
if all(os.path.exists(os.path.join(p, "neff.csv")) for p in lcp.values()) and not DRY:
    B = {(arm, it): np.mean([bvc(asl_read(lcp[(arm, sd, it)]), 1) for sd in (102, 103)], 0) for arm in ("dsm", "dirl") for it in (500, 1000, 1500, 2000)}
    summary["C2LC"] = {"|".join(map(str, k)): list(map(float, v)) for k, v in B.items()}
    sections["C2LC"] = ["## EXPLORATORY 2026-11-02 — CLAIMS C2-LC: does the score-vs-direct bias gap close with training?", "",
                        "Not pre-registered. DEC networks (dataset 1, seeds 102–103; DSM vs direct-with-ladder) at 500–2000 iterations; median "
                        "bias² b and per-map variance v (units of σ²) from the exact decomposition. The linear toy predicts a transient gap.", "",
                        "| iteration | DSM b / v | DIR_L b / v |", "|---:|---|---|"] + [
        "| %d | %.4f / %.4f | %.4f / %.4f |" % (it, B[("dsm", it)][0], B[("dsm", it)][1], B[("dirl", it)][0], B[("dirl", it)][1])
        for it in (500, 1000, 1500, 2000)]

# ---------------- C7 ----------------
c7p = {(c, n, m): f"runs/nat/eval_c7/{c}/n{n}/{m}.csv" for c in ("W05", "W20") for n in (25, 100, 400) for m in ("ours", "sub")}
if all(os.path.exists(p) for p in c7p.values()):
    P = {k: nat_psnr(nat_read(p), 9)[0] for k, p in c7p.items()}
    P1 = {k: nat_psnr(nat_read(p), 1)[0] for k, p in c7p.items()}
    prim = {(c, n): dstat(np.zeros(68) if DRY else P[(c, n, "sub")] - P[(c, n, "ours")]) for c in ("W05", "W20") for n in (25, 100, 400)}
    for k, q in zip(prim, holm([v["p"] for v in prim.values()])):
        prim[k]["p_holm"] = float(q)
    summary["C7"] = {"|".join(map(str, k)): v for k, v in prim.items()}
    sections["C7"] = ["## RUN 2026-11-02 — CLAIMS C7: when does denoising the copy mean win? (training-set size × noise)", "",
                      "Pre-registered (`Archive/CLAIMS/PREREGISTRATION.md`). OURS (per copy + dial) vs SUB (denoise the copy mean), trained on "
                      "the first n Train400 images; BSD68, r chosen without ground truth.", "",
                      "**PRIMARY (PSNR(SUB) − PSNR(OURS) at a = 9, Holm over 6):**", ""] + [
        "- %s, n = %d: %s, p = %.2g, Holm %.2g" % (c, n, fdb(v), v["p"], v["p_holm"]) for (c, n), v in prim.items()] + [
        "", "Mean PSNR (dB), OURS / SUB, a = 1 and a = 9:", "", "| condition | n | a = 1 | a = 9 |", "|---|---:|---|---|"] + [
        "| %s | %d | %.2f / %.2f | %.2f / %.2f |" % (c, n, P1[(c, n, "ours")].mean(), P1[(c, n, "sub")].mean(), P[(c, n, "ours")].mean(),
                                                   P[(c, n, "sub")].mean()) for c in ("W05", "W20") for n in (25, 100, 400)]

if DRY:
    print("DRY RUN OK: parts present:", sorted(sections)); sys.exit(0)
os.makedirs(OUT, exist_ok=True)
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=float)
md = open(MD, encoding="utf-8").read()
for k, sec in sections.items():
    print("\n".join(sec) + "\n")
    if sec[0] not in md:
        open(MD, "a", encoding="utf-8").write("\n---\n\n" + "\n".join(sec) + "\n")
