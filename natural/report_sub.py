# -*- coding: utf-8 -*-
"""NAT-SUB report (Archive/NAT/PREREGISTRATION_SUB.md): denoising the copy mean (SUB, SUBSUP) vs per-copy (ours, sup)
and BM3D. Same rules as ``natural/report.py``: learned methods at the r chosen WITHOUT ground truth on Set12 (summed
split-half n_eff), scores on BSD68 (paired). ``--dry`` sets SUB := ours (0 dB) and writes nothing.

    python natural/report_sub.py [--dry]
"""
import os, sys, csv, json, argparse
import numpy as np
from scipy.stats import wilcoxon
import data as D

os.chdir(D.REPO)
ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); DRY = ap.parse_args().dry
HEAD = "## RUN 2026-11-01 — NAT-SUB (exploratory): does denoising the copy mean close the gap to BM3D?"
CONDS, METHODS, A_EVAL = ("W05", "C05"), ("ours", "sup", "sub", "subsup", "bm3d", "avg"), (1, 3, 5, 9)
rng = np.random.default_rng(0)
T = {}
for c in CONDS:
    for m in METHODS:
        for r in csv.DictReader(open(f"runs/nat/eval/{c}/{m}.csv", encoding="utf-8")):
            T[(c, m, r["split"], int(r["image"]), round(float(r["r"]), 4), int(r["a"]))] = (float(r["psnr"]), float(r["mse_sh"]), float(r["s2"]))
n12 = len({k[3] for k in T if k[2] == "set12"}); n68 = len({k[3] for k in T if k[2] == "bsd68"})


def r_free(c, m, a):
    rs = sorted({k[4] for k in T if k[0] == c and k[1] == m})
    return max(rs, key=lambda r: sum(s2 / h if h > 0 else 0 for _, h, s2 in (T[(c, m, "set12", i, r, a)] for i in range(n12))))


def ps(c, m, a):
    if DRY and m == "sub":
        m = "ours"
    r = r_free(c, m, a)
    return np.array([T[(c, m, "bsd68", i, r, a)][0] for i in range(n68)]), r


def stat(d):
    b = rng.choice(d, size=(10000, len(d))).mean(1)
    return dict(mean=float(d.mean()), ci_lo=float(np.percentile(b, 2.5)), ci_hi=float(np.percentile(b, 97.5)),
                wins=int((d > 0).sum()), n=len(d), p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)


prim = stat(ps("W05", "sub", 9)[0] - ps("W05", "ours", 9)[0])
if DRY:
    assert abs(prim["mean"]) < 1e-12
    print("DRY RUN OK: SUB := ours gives 0 dB"); sys.exit(0)
tab = {(c, m, a): (float(ps(c, m, a)[0].mean()), ps(c, m, a)[1]) for c in CONDS for m in METHODS for a in A_EVAL}
sec = {(c, x, y, a): stat(ps(c, x, a)[0] - ps(c, y, a)[0]) for c in CONDS for a in A_EVAL
       for x, y in (("sub", "ours"), ("sub", "bm3d"), ("subsup", "sub"), ("subsup", "sup"))}
os.makedirs("runs/nat/report", exist_ok=True)
json.dump(dict(primary=prim, table={"|".join(map(str, k)): v for k, v in tab.items()},
               secondary={"|".join(map(str, k)): v for k, v in sec.items()}),
          open("runs/nat/report/summary_sub.json", "w"), indent=2)
f = lambda d: "%+.2f dB (95%% CI %+.2f to %+.2f; %d/%d images)" % (d["mean"], d["ci_lo"], d["ci_hi"], d["wins"], d["n"])
L = ["", "---", "", HEAD, "",
     "Pre-registered as exploratory (`Archive/NAT/PREREGISTRATION_SUB.md`). SUB denoises the mean of m ~ U{1..9} copies "
     "(conditioned on σ/√m, N2N target = mean of the other copies); SUBSUP is the same with the clean target. Same "
     "U-Net, data and budget as NAT; r chosen without ground truth on Set12; scores on BSD68. The dry run gave 0 dB.", "",
     "**PRIMARY (W05, a = 9): PSNR(SUB) − PSNR(ours) = %s, p = %.2g.**" % (f(prim), prim["p"]), "",
     "Mean PSNR on BSD68 (dB; chosen r in brackets):", "",
     "| method | " + " | ".join(f"{c} a={a}" for c in CONDS for a in A_EVAL) + " |", "|---|" + "---:|" * (len(CONDS) * len(A_EVAL))]
names = dict(ours="Ours (per copy + dial)", sup="Supervised, per copy", sub="SUB: denoise the mean (N2N)",
             subsup="SUBSUP: denoise the mean (supervised)", bm3d="BM3D on the mean", avg="Plain averaging")
for m in METHODS:
    L.append("| %s | %s |" % (names[m], " | ".join("%.2f (%.2f)" % tab[(c, m, a)] if m not in ("bm3d", "avg") else "%.2f" % tab[(c, m, a)][0]
                                                    for c in CONDS for a in A_EVAL)))
L += ["", "Differences (paired over 68 images): " + "; ".join(
    "%s %s−%s a=%d: %+.2f dB (%d/68)" % (c, x.upper(), y.upper(), a, sec[(c, x, y, a)]["mean"], sec[(c, x, y, a)]["wins"])
    for c in CONDS for x, y in (("sub", "ours"), ("sub", "bm3d"), ("subsup", "sup")) for a in (1, 9)) + ".", "",
      "**Data:** `runs/nat/eval/{W05,C05}/{sub,subsup}.csv`, `runs/nat/report/summary_sub.json`. **Commands:** "
      "`bash runs/nat/queue_sub.sh`; `python natural/report_sub.py`.", ""]
if HEAD not in open("WEEKEND_ITERATIONS.md", encoding="utf-8").read():
    open("WEEKEND_ITERATIONS.md", "a", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
