# -*- coding: utf-8 -*-
"""GAP B report (Archive/GAP/PREREGISTRATION_B.md). Reads runs/gap/b/eval/<arm>_<noise>_s<seed>_it<iter>.csv
(transplant_eval.py). Selection without ground truth exactly as mean_report (per test volume, checkpoint and r from the
other volumes' split-half MSE; seeds averaged in log n_eff). n_eff = SIG^2 / mse_gt_head. ``--dry`` sets every model :=
A_W (ratios 1). Appends the log section to WEEKEND_ITERATIONS.md once unless --dry.

    python resolution/transplant_report.py [--dry]
"""
import os, csv, json, argparse
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
import mean_report as R

REPO = R.REPO
EVAL = os.path.join(REPO, "runs", "gap", "b", "eval")
ARMS, NOISE, SEEDS, A, ITERS = ("A", "SUB", "SUBM"), ("W", "G", "T"), (101, 102), (1, 3, 5, 9), (1000, 2000, 3000)
NAME = {"W": "white", "G": "ASL-coloured Gaussian", "T": "real ASL fields"}
HEAD = "## RUN 2026-11-05 — GAP B: does ASL's noise, transplanted onto fastMRI, shrink the denoise-the-mean gain?"
ASL_GAIN = 1.062                                   # SUB / PER on 1104c at a = 9, c = 1 (GAP A1/A2)
rng = np.random.default_rng(0)


def read(arm, noise, seed):
    T = defaultdict(dict)
    for it in ITERS:
        for r in csv.DictReader(open(os.path.join(EVAL, f"{arm}_{noise}_s{seed}_it{it}.csv"), encoding="utf-8")):
            T[(int(r["a"]), it, float(r["r"]))][int(r["img"])] = (int(r["vol"]), float(r["sigma"]), float(r["mse_gt_head"]),
                                                                  float(r["mse_sh_head"]), [float(r[b]) for b in R.TB])
    return T


def design(TT, arm, noise, a):
    per = [R.per_volume(TT[(arm, noise, s)], a) for s in SEEDS]
    vols = sorted(per[0])
    return np.array([np.mean([p[v][0] for p in per]) for v in vols]), np.array([np.mean([p[v][1] for p in per], 0) for v in vols])


def stat(d):
    g, lo, hi = R.boot(d)
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
    return dict(estimate=g, ci_lo=lo, ci_hi=hi, p=p, wins=int((d > 0).sum()), n=len(d))


def fmt(v):
    return f"{v['estimate']:.3f} ({v['ci_lo']:.2f}–{v['ci_hi']:.2f}; {v['wins']}/{v['n']})"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    TT = {(m, n, s): read("A" if args.dry else m, "W" if args.dry else n, s) for m in ARMS for n in NOISE for s in SEEDS}
    V = {(m, n, a): design(TT, m, n, a) for m in ARMS for n in NOISE for a in A}
    lg = {(m, n, a): V[(m, n, a)][0] - V[("A", n, a)][0] for m in ARMS[1:] for n in NOISE for a in A}   # log (arm / PER)
    prim = {"P1 (SUB/PER under T) / (SUB/PER under W), a = 9": stat(lg[("SUB", "T", 9)] - lg[("SUB", "W", 9)]),
            "P2 (SUB/PER under G) / (SUB/PER under W), a = 9": stat(lg[("SUB", "G", 9)] - lg[("SUB", "W", 9)])}
    ps = sorted(prim, key=lambda k: prim[k]["p"])
    for j, k in enumerate(ps):
        prim[k]["p_holm"] = min(1.0, max(prim[q]["p"] * (len(ps) - i) for i, q in enumerate(ps[:j + 1])))
    if args.dry:
        assert all(abs(v["estimate"] - 1) < 1e-12 for v in prim.values()), "dry run"
        print("DRY RUN OK"); return
    gW = float(np.exp(lg[("SUB", "W", 9)].mean()))
    share = {n: float(np.log(gW / np.exp(lg[("SUB", n, 9)].mean())) / np.log(gW / ASL_GAIN)) for n in ("G", "T")}
    L = [HEAD, "", "Pre-registered (`Archive/GAP/PREREGISTRATION_B.md`). fastMRI AXT2 at the ASL matrix (80), 62 training volumes, "
         "N = 10 copies, label σ = 1 per channel (white noise then has fastMRI S_K = 2's per-band SNR). Noise: W white; G "
         "Gaussian with the ASL repetition-deviation power spectrum; T the real ASL deviation fields (coil1; training from "
         "ds1 + 1104a, test from 1104c; distinct slice and repetition per copy). PER (arm A), SUB, SUBM; seeds 101–102; "
         "checkpoint and r chosen without ground truth (leave-one-volume-out split-half); n_eff vs ground truth on 15 test volumes.",
         "", "**PRIMARY (Holm over 2):**", ""]
    for k, v in prim.items():
        L.append(f"- {k}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; higher on {v['wins']}/{v['n']} volumes), "
                 f"p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", f"Share of the gap (W's SUB/PER {gW:.3f} → ASL's {ASL_GAIN}) reproduced, in log ratio: G {share['G']:.2f}, T {share['T']:.2f}.",
          "", "SUB / PER and SUBM / PER (90% CI; volumes higher):", "",
          "| a | " + " | ".join(f"SUB/PER {n}" for n in NOISE) + " | " + " | ".join(f"SUBM/PER {n}" for n in NOISE) + " |",
          "|---:|" + "---|" * 6]
    for a in A:
        L.append(f"| {a} | " + " | ".join(fmt(stat(lg[(m, n, a)])) for m in ARMS[1:] for n in NOISE) + " |")
    L += ["", "n_eff (geometric mean over volumes; σ = 1 label):", "", "| a | " + " | ".join(f"{m} {n}" for n in NOISE for m in ARMS) + " |",
          "|---:|" + "---:|" * 9]
    for a in A:
        L.append(f"| {a} | " + " | ".join(f"{np.exp(V[(m, n, a)][0].mean()):.1f}" for n in NOISE for m in ARMS) + " |")
    L += ["", "Band transfer at a = 9 (share of the true detail kept per band of k / k_max; plain averaging = 1):", "",
          "| design | noise | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for n in NOISE:
        for m in ARMS:
            L.append(f"| {m} | {n} | " + " | ".join(f"{t:.2f}" for t in V[(m, n, 9)][1].mean(0)) + " |")
    print("\n".join(L))
    os.makedirs(os.path.join(REPO, "runs", "gap", "report"), exist_ok=True)
    json.dump({"primary": prim, "gain_W": gW, "share": share}, open(os.path.join(REPO, "runs", "gap", "report", "summary_b.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
