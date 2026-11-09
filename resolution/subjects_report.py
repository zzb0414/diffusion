# -*- coding: utf-8 -*-
"""GAP-N report (Archive/GAP/PREREGISTRATION_N.md): subject diversity at a fixed training-image count on fastMRI content D,
noise T. Volumes x slices: 1x8 / 4x8 (GAP-F, read only), 62x8 (GAP-D DT, read only), 8x1 / 32x1 / 62x1 (one slice per
volume, GAP-N). PER (A) / SUB / SUBM, seeds 101-102; selection without ground truth (mean_report.per_volume); n_eff vs ground
truth. ``--dry`` reads 62x8 for every condition (ratios 1). Appends once unless --dry.

    python resolution/subjects_report.py [--dry]
"""
import os, json, argparse
import numpy as np
import mean_report as R
import matrix_report as MR
from transplant_report import stat, fmt

REPO = R.REPO
MR.PATH = PATH = {"1x8": "runs/gap/f/eval/{arm}_DTn1_s{s}_it{it}.csv", "8x1": "runs/gap/n/eval/{arm}_DTo8_s{s}_it{it}.csv",
                  "4x8": "runs/gap/f/eval/{arm}_DTn4_s{s}_it{it}.csv", "32x1": "runs/gap/n/eval/{arm}_DTo32_s{s}_it{it}.csv",
                  "62x1": "runs/gap/n/eval/{arm}_DTo62_s{s}_it{it}.csv", "62x8": "runs/gap/d/eval/{arm}_DT_s{s}_it{it}.csv"}
ARMS, SEEDS, A = ("A", "SUB", "SUBM"), (101, 102), (1, 3, 5, 9)
HEAD = "## RUN 2026-11-09 — GAP-N: at a fixed number of training images, does the number of subjects decide the denoise-the-mean gain?"
PAIRS = (("32x1", "4x8"), ("8x1", "1x8"), ("62x1", "62x8"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    TT = {(c, m, s): MR.read("62x8" if args.dry else c, m, s) for c in PATH for m in ARMS for s in SEEDS}
    V = {}
    for c in PATH:
        for m in ARMS:
            for a in A:
                per = [R.per_volume(TT[(c, m, s)], a) for s in SEEDS]
                vols = sorted(per[0])
                V[(c, m, a)] = np.array([np.mean([p[v][0] for p in per]) for v in vols])
    lg = {(c, m, a): V[(c, m, a)] - V[(c, "A", a)] for c in PATH for m in ARMS[1:] for a in A}
    prim = {"N1 diversity at 32 images: (SUB/PER, 32x1) / (SUB/PER, 4x8), a = 9": stat(lg[("32x1", "SUB", 9)] - lg[("4x8", "SUB", 9)]),
            "N2 diversity at 8 images: (SUB/PER, 8x1) / (SUB/PER, 1x8), a = 9": stat(lg[("8x1", "SUB", 9)] - lg[("1x8", "SUB", 9)])}
    ps = sorted(prim, key=lambda k: prim[k]["p"])
    for j, k in enumerate(ps):
        prim[k]["p_holm"] = min(1.0, max(prim[q]["p"] * (len(ps) - i) for i, q in enumerate(ps[:j + 1])))
    if args.dry:
        assert all(abs(v["estimate"] - 1) < 1e-12 for v in prim.values()), "dry run"
        print("DRY RUN OK"); return
    L = [HEAD, "", "Pre-registered (`Archive/GAP/PREREGISTRATION_N.md`). Content D, noise T (real ASL fields), matrix 80, PER (arm A), "
         "SUB, SUBM, seeds 101–102; checkpoint and r chosen without ground truth; n_eff vs ground truth on 15 test volumes. "
         "Conditions are volumes × slices per volume; one-slice designs take slice position v mod 8, so slice-position coverage "
         "matches the 8-slice designs. 1x8, 4x8 (GAP-F) and 62x8 (GAP-D DT) are read only.", "", "**PRIMARY (Holm over 2):**", ""]
    for k, v in prim.items():
        L.append(f"- {k}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; higher on {v['wins']}/{v['n']} volumes), "
                 f"p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", "Secondary (90% CI; volumes higher), a = 9:", ""]
    for c1, c2 in PAIRS:
        L.append(f"- {c1} vs {c2}: SUB/PER {fmt(stat(lg[(c1, 'SUB', 9)] - lg[(c2, 'SUB', 9)]))}; "
                 f"SUBM/PER {fmt(stat(lg[(c1, 'SUBM', 9)] - lg[(c2, 'SUBM', 9)]))}; absolute n_eff "
                 + "; ".join(f"{m} {fmt(stat(V[(c1, m, 9)] - V[(c2, m, 9)]))}" for m in ARMS))
    L += ["", "| condition | " + " | ".join(f"SUB/PER a = {a}" for a in A) + " | SUBM/PER a = 9 | n_eff a = 9 PER / SUB / SUBM |",
          "|---|" + "---|" * 6]
    for c in PATH:
        L.append(f"| {c} | " + " | ".join(fmt(stat(lg[(c, 'SUB', a)])) for a in A) + f" | {fmt(stat(lg[(c, 'SUBM', 9)]))} | "
                 + " / ".join(f"{np.exp(V[(c, m, 9)].mean()):.1f}" for m in ARMS) + " |")
    print("\n".join(L))
    os.makedirs(os.path.join(REPO, "runs", "gap", "report"), exist_ok=True)
    json.dump({"primary": prim}, open(os.path.join(REPO, "runs", "gap", "report", "summary_n.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
