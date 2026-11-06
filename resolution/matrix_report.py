# -*- coding: utf-8 -*-
"""GAP C report (Archive/GAP/PREREGISTRATION_C.md): why does the matrix shrink the learned denoise-the-mean gain?

Conditions (white noise, label 1 per channel, PER = arm A and SUB, seeds 101-102, 62 training volumes; pixels per
training step = 8 x 64^2 throughout):
  160      matrix 160, crop 64 (MEAN-SNR S_K = 2; runs/mean_sk2/eval, read only)
  160c128  matrix 160, crop 128, batch 2
  80       matrix 80, crop 64 (GAP B, W; runs/gap/b/eval, read only)
  80c32    matrix 80, crop 32, batch 32
  U80      matrix-80 content on a 160 grid (k-space zero-padding), crop 64
Selection without ground truth as mean_report. ``--dry`` sets every condition := 80 (ratios 1). Appends once unless --dry.

    python resolution/matrix_report.py [--dry]
"""
import os, csv, json, argparse
from collections import defaultdict
import numpy as np
import mean_report as R
from transplant_report import stat, fmt

REPO = R.REPO
PATH = {"160": "runs/mean_sk2/eval/{arm}_v62_s{s}_it{it}.csv", "160c128": "runs/gap/c/eval/{arm}_160c128_s{s}_it{it}.csv",
        "80": "runs/gap/b/eval/{arm}_W_s{s}_it{it}.csv", "80c32": "runs/gap/c/eval/{arm}_80c32_s{s}_it{it}.csv",
        "U80": "runs/gap/c/eval/{arm}_U80_s{s}_it{it}.csv"}
WIENER = {"160": 1.423, "160c128": 1.423, "80": 1.273, "80c32": 1.273, "U80": 1.273}   # a = 9, resolution/wiener.py (PREREG C0)
SEEDS, A, ITERS = (101, 102), (1, 3, 5, 9), (1000, 2000, 3000)
HEAD = "## RUN 2026-11-06 — GAP C: why does a coarser matrix shrink the learned denoise-the-mean gain?"


def read(cond, arm, s):
    T = defaultdict(dict)
    for it in ITERS:
        for r in csv.DictReader(open(os.path.join(REPO, PATH[cond].format(arm=arm, s=s, it=it)), encoding="utf-8")):
            T[(int(r["a"]), it, float(r["r"]))][int(r["img"])] = (int(r["vol"]), float(r["sigma"]), float(r["mse_gt_head"]),
                                                                  float(r["mse_sh_head"]), [float(r[b]) for b in R.TB])
    return T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    TT = {(c, m, s): read("80" if args.dry else c, m, s) for c in PATH for m in ("A", "SUB") for s in SEEDS}
    V = {}
    for c in PATH:
        for m in ("A", "SUB"):
            for a in A:
                per = [R.per_volume(TT[(c, m, s)], a) for s in SEEDS]
                vols = sorted(per[0])
                V[(c, m, a)] = (np.array([np.mean([p[v][0] for p in per]) for v in vols]),
                                np.array([np.mean([p[v][1] for p in per], 0) for v in vols]))
    lg = {(c, a): V[(c, "SUB", a)][0] - V[(c, "A", a)][0] for c in PATH for a in A}
    prim = {"P1 crop at matrix 80: (SUB/PER, 80c32) / (SUB/PER, 80), a = 9": stat(lg[("80c32", 9)] - lg[("80", 9)]),
            "P2 pixel scale: (SUB/PER, U80) / (SUB/PER, 80), a = 9": stat(lg[("U80", 9)] - lg[("80", 9)]),
            "P3 crop at matrix 160: (SUB/PER, 160c128) / (SUB/PER, 160), a = 9": stat(lg[("160c128", 9)] - lg[("160", 9)])}
    ps = sorted(prim, key=lambda k: prim[k]["p"])
    for j, k in enumerate(ps):
        prim[k]["p_holm"] = min(1.0, max(prim[q]["p"] * (len(ps) - i) for i, q in enumerate(ps[:j + 1])))
    if args.dry:
        assert all(abs(v["estimate"] - 1) < 1e-12 for v in prim.values()), "dry run"
        print("DRY RUN OK"); return
    L = [HEAD, "", "Pre-registered (`Archive/GAP/PREREGISTRATION_C.md`). fastMRI AXT2, white noise σ = 1 per channel, 62 training "
         "volumes, PER (arm A) and SUB, seeds 101–102, pixels per training step 8 × 64² in every condition; checkpoint and r "
         "chosen without ground truth (leave-one-volume-out split-half); n_eff vs ground truth on 15 test volumes. "
         "Conditions: 160 (crop 64), 160c128 (crop 128, batch 2), 80 (crop 64), 80c32 (crop 32, batch 32), U80 (matrix-80 "
         "content on a 160 grid, crop 64).", "", "**PRIMARY (Holm over 3):**", ""]
    for k, v in prim.items():
        L.append(f"- {k}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; higher on {v['wins']}/{v['n']} volumes), "
                 f"p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", "SUB / PER (90% CI; volumes higher), and the learned / linear factor at a = 9 (SUB/PER over the Wiener prediction):", "",
          "| condition | " + " | ".join(f"a = {a}" for a in A) + " | Wiener a = 9 | learned / Wiener |", "|---|" + "---|" * 4 + "---:|---:|"]
    for c in PATH:
        g9 = float(np.exp(lg[(c, 9)].mean()))
        L.append(f"| {c} | " + " | ".join(fmt(stat(lg[(c, a)])) for a in A) + f" | {WIENER[c]:.3f} | {g9 / WIENER[c]:.3f} |")
    L += ["", "n_eff (geometric mean over volumes), PER / SUB:", "", "| a | " + " | ".join(PATH) + " |", "|---:|" + "---|" * len(PATH)]
    for a in A:
        L.append(f"| {a} | " + " | ".join(f"{np.exp(V[(c, 'A', a)][0].mean()):.1f} / {np.exp(V[(c, 'SUB', a)][0].mean()):.1f}" for c in PATH) + " |")
    L += ["", "Band transfer at a = 9 (bands of k / k_max of the content; plain averaging = 1):", "",
          "| condition | design | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for c in PATH:
        for m in ("A", "SUB"):
            L.append(f"| {c} | {m} | " + " | ".join(f"{t:.2f}" for t in V[(c, m, 9)][1].mean(0)) + " |")
    print("\n".join(L))
    os.makedirs(os.path.join(REPO, "runs", "gap", "report"), exist_ok=True)
    json.dump({"primary": prim}, open(os.path.join(REPO, "runs", "gap", "report", "summary_c.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
