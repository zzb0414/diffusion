# -*- coding: utf-8 -*-
"""NAT report (Archive/NAT/PREREGISTRATION.md): P1 estimator bias, P2 ground-truth-free dial, P3 methods, P4 loss.

Reads runs/nat/eval/<cond>/<method>.csv. r for each learned method and a is chosen WITHOUT ground truth: the grid r
maximizing the summed split-half n_eff (s2 / mse_sh) over Set12; the oracle r maximizes mean PSNR on BSD68. All
comparisons are on BSD68 (68 images, paired). Missing methods/conditions are skipped; the log section is written
only when everything pre-registered is present. ``--dry`` replaces each compared method by its comparator (all
differences must be 0) and writes nothing.

    python natural/report.py [--dry]
"""
import os, sys, csv, json, argparse
import numpy as np
from scipy.stats import wilcoxon
import data as D

os.chdir(D.REPO)
ap = argparse.ArgumentParser()
ap.add_argument("--dry", action="store_true")
DRY = ap.parse_args().dry
OUT, MD = "runs/nat/report", "WEEKEND_ITERATIONS.md"
HEAD = "## RUN 2026-11-01 — NAT: step 3, ground-truth validation on natural images"
CONDS = ("W05", "C05", "W20", "W50")
LEARNED = ("ours", "avg9", "avgu", "sup", "n2n", "ne2ne", "r2r", "n2s")
CLASSICAL = ("avg", "bm3d")
A_EVAL = (1, 3, 5, 9)
NAMES = dict(ours="Ours (DSM + dial)", avg9="Averaging-aware loss, a = 9", avgu="Averaging-aware loss, a ~ U{1..9}",
             sup="Supervised (upper bound)", n2n="Noise2Noise (classic)", ne2ne="Neighbor2Neighbor",
             r2r="Recorrupted-to-Recorrupted", n2s="Noise2Score", avg="Plain averaging", bm3d="BM3D")
rng = np.random.default_rng(0)

T = {}                                          # (cond, method, split, image, r, a) -> (mse_gt, psnr, mse_sh, s2)
for cond in CONDS:
    for m in LEARNED + CLASSICAL:
        p = f"runs/nat/eval/{cond}/{m}.csv"
        if os.path.exists(p):
            for r in csv.DictReader(open(p, encoding="utf-8")):
                T[(cond, m, r["split"], int(r["image"]), round(float(r["r"]), 4), int(r["a"]))] = tuple(
                    float(r[c]) for c in ("mse_gt", "psnr", "mse_sh", "s2"))
have = {(k[0], k[1]) for k in T}
RG = sorted({k[4] for k in T if k[1] in LEARNED}) or [0.0]
n_img = {sp: len({k[3] for k in T if k[2] == sp}) for sp in ("set12", "bsd68")}


def r_free(cond, m, a):
    """ground-truth-free r: maximize summed split-half n_eff over Set12."""
    if m in CLASSICAL:
        return 0.0
    def score(r):
        v = [T[(cond, m, "set12", i, r, a)] for i in range(n_img["set12"])]
        return sum(s2 / h if h > 0 else 0.0 for _, _, h, s2 in v)
    return max(RG, key=score)


def r_oracle(cond, m, a):
    if m in CLASSICAL:
        return 0.0
    return max(RG, key=lambda r: np.mean([T[(cond, m, "bsd68", i, r, a)][1] for i in range(n_img["bsd68"])]))


def psnr(cond, m, a, r):
    return np.array([T[(cond, m, "bsd68", i, r, a)][1] for i in range(n_img["bsd68"])])


def diff_stat(d):
    b = rng.choice(d, size=(10000, len(d))).mean(1)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return dict(mean=float(d.mean()), ci_lo=float(np.percentile(b, 2.5)), ci_hi=float(np.percentile(b, 97.5)),
                wins=int((d > 0).sum()), n=len(d), p=p)


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


S = {}
# P1: relative bias of the split-half MSE estimate, OURS at r = 0, a in {1, 9}
P1 = {}
for cond in CONDS:
    if (cond, "ours") not in have:
        continue
    for a in (1, 9):
        rel = np.array([(T[(cond, "ours", "bsd68", i, 0.0, a)][2] - T[(cond, "ours", "bsd68", i, 0.0, a)][0]) /
                        T[(cond, "ours", "bsd68", i, 0.0, a)][0] for i in range(n_img["bsd68"])])
        if DRY:
            rel = np.zeros_like(rel)
        b = rng.choice(rel, size=(10000, len(rel))).mean(1)
        P1[(cond, a)] = dict(mean=float(rel.mean()), ci_lo=float(np.percentile(b, 2.5)), ci_hi=float(np.percentile(b, 97.5)),
                             verdict="unbiased in practice" if np.percentile(b, 2.5) >= -0.05 and np.percentile(b, 97.5) <= 0.05
                             else "biased beyond ±5%")
# P2: PSNR loss of the ground-truth-free r vs the oracle r, OURS, a = 9
P2 = {}
for cond in CONDS:
    if (cond, "ours") in have:
        rf, ro = r_free(cond, "ours", 9), r_oracle(cond, "ours", 9)
        loss = float(psnr(cond, "ours", 9, ro).mean() - psnr(cond, "ours", 9, rf).mean())
        P2[cond] = dict(r_free=rf, r_oracle=ro, loss_db=0.0 if DRY else loss,
                        verdict="ground-truth-free selection suffices" if (0.0 if DRY else loss) <= 0.1 else "loss above 0.1 dB")
# P3 / P4 and the full table
cmp = {}
for cond in CONDS:
    for m in LEARNED + CLASSICAL:
        if (cond, m) not in have or (cond, "ours") not in have:
            continue
        for a in A_EVAL:
            mine = psnr(cond, "ours", a, r_free(cond, "ours", a))
            theirs = psnr(cond, m, a, r_free(cond, m, a))
            cmp[(cond, m, a)] = diff_stat((theirs - theirs) if DRY else (mine - theirs))
P3b = ("n2n", "ne2ne", "r2r", "n2s", "avg", "bm3d")
P3 = {m: cmp[("W05", m, 9)] for m in P3b if ("W05", m, 9) in cmp}
for m, q in zip(P3, holm([P3[m]["p"] for m in P3]) if P3 else []):
    P3[m]["p_holm"] = float(q)
P4 = {}
for m, a in (("avg9", 9), ("avgu", 1), ("avgu", 9)):
    if ("W05", m, a) in cmp:
        d = cmp[("W05", m, a)]
        P4[(m, a)] = dict(d, mean=-d["mean"], ci_lo=-d["ci_hi"], ci_hi=-d["ci_lo"], wins=d["n"] - d["wins"])  # m minus ours
for k, q in zip(P4, holm([P4[k]["p"] for k in P4]) if P4 else []):
    P4[k]["p_holm"] = float(q)
if DRY:
    assert all(abs(v["mean"]) < 1e-12 for v in cmp.values())
    print("DRY RUN OK: comparator := method gives 0 dB everywhere; P1/P2 forced to 0 for the dry run")
    sys.exit(0)

table = {(cond, m, a): (float(psnr(cond, m, a, r_free(cond, m, a)).mean()), r_free(cond, m, a))
         for (cond, m) in have for a in A_EVAL}
os.makedirs(OUT, exist_ok=True)
K = lambda t: "|".join(map(str, t))
json.dump(dict(P1={K(k): v for k, v in P1.items()}, P2=P2, P3=P3, P4={K(k): v for k, v in P4.items()},
               compare={K(k): v for k, v in cmp.items()}, psnr_table={K(k): v for k, v in table.items()}),
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
fmt = lambda d: "%+.2f dB (95%% CI %+.2f to %+.2f; %d/%d images)" % (d["mean"], d["ci_lo"], d["ci_hi"], d["wins"], d["n"])
L = ["", "---", "", HEAD, "",
     "Pre-registered (`Archive/NAT/PREREGISTRATION.md`). Train400 → BSD68 (320×480 crops), 10 fixed noisy copies per "
     "image; conditions W50 (σ 50/255), W05 (σ 0.5), W20 (σ 2.0), C05 (σ 0.5, ASL-coloured). Same 1-channel U-Net and "
     "budget for every learned method (3000 iterations, seed 101; code `natural/`). r chosen WITHOUT ground truth on "
     "Set12 (split-half n_eff); all scores on BSD68 with exact ground truth. The dry run gave 0 dB everywhere.", "",
     "**P1 — is the split-half estimator unbiased?** Relative bias of the estimated MSE (ours, r = 0):", ""]
for (cond, a), v in P1.items():
    L.append("- %s, a = %d: %+.2f%% (95%% CI %+.2f to %+.2f%%) → **%s**" % (cond, a, 100 * v["mean"], 100 * v["ci_lo"], 100 * v["ci_hi"], v["verdict"]))
L += ["", "**P2 — choosing r without ground truth** (ours, a = 9; PSNR loss vs the BSD68 oracle r):", ""]
for cond, v in P2.items():
    L.append("- %s: r chosen %.3f vs oracle %.3f, loss %.3f dB → **%s**" % (cond, v["r_free"], v["r_oracle"], v["loss_db"], v["verdict"]))
L += ["", "**P3 — ours vs baselines (W05, a = 9; PSNR(ours) − PSNR(baseline), Holm over %d):**" % len(P3), ""]
for m, v in P3.items():
    L.append("- vs %s: %s, p = %.2g, Holm %.2g" % (NAMES[m], fmt(v), v["p"], v["p_holm"]))
L += ["", "**P4 — averaging-aware loss (W05; PSNR(method) − PSNR(ours), Holm over %d):**" % len(P4), ""]
for (m, a), v in P4.items():
    L.append("- %s at a = %d: %s, p = %.2g, Holm %.2g" % (NAMES[m], a, fmt(v), v["p"], v["p_holm"]))
L += ["", "Mean PSNR on BSD68 (dB; learned methods at their ground-truth-free r):", "",
      "| method | " + " | ".join(f"{c} a={a}" for c in CONDS for a in (1, 9)) + " |", "|---|" + "---:|" * (2 * len(CONDS))]
for m in LEARNED + CLASSICAL:
    L.append("| %s | %s |" % (NAMES[m], " | ".join("%.2f" % table[(c, m, a)][0] if (c, m, a) in table else "–"
                                                    for c in CONDS for a in (1, 9))))
L += ["", "**Data:** `runs/nat/eval/<cond>/<method>.csv`, `%s/summary.json`. **Commands:** `bash runs/nat/queue_gpu.sh`; "
      "`bash runs/nat/queue_cpu.sh`; `python natural/report.py`." % OUT, ""]
complete = all((c, m) in have for c in CONDS for m in LEARNED + CLASSICAL)
if complete and HEAD not in open(MD, encoding="utf-8").read():
    open(MD, "a", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
