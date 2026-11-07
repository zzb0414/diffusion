# -*- coding: utf-8 -*-
"""GAP F report (Archive/GAP/PREREGISTRATION_F.md).

F1 (exploratory): fastMRI-trained GAP B (T) and GAP D (DT) models scored on real ASL 1104c (runs/gap/f/asl/<arm>_<c>_s<s>/
eval_it<it>/neff.csv, experiments/_tr_eval.py) against the ASL-trained references; selection as _subj_report.
F2 (PRIMARY, Holm over 2): DT trained on n_vol 1 and 4 (runs/gap/f/eval) vs n_vol 62 (GAP D); selection as mean_report.
``--dry`` sets the n_vol models := n_vol 62 (ratios 1). Appends once unless --dry.

    python resolution/gap_f_report.py [--dry]
"""
import os, sys, json, argparse
import numpy as np
import mean_report as R
import matrix_report as MR
from transplant_report import stat, fmt
sys.path.insert(0, os.path.join(R.REPO, "experiments"))
import _subj_report as S                                                   # noqa: E402
import _crop_report as C                                                   # noqa: E402

REPO = R.REPO
MR.PATH = PATH = {"62": "runs/gap/d/eval/{arm}_DT_s{s}_it{it}.csv", "4": "runs/gap/f/eval/{arm}_DTn4_s{s}_it{it}.csv",
                  "1": "runs/gap/f/eval/{arm}_DTn1_s{s}_it{it}.csv"}
ARMS, SEEDS, A = ("A", "SUB", "SUBM"), (101, 102), (1, 3, 5, 9)
ASLREF = {"A": "runs/subj/PER_s{s}", "SUB": "runs/subj/SUB_s{s}", "SUBM": "runs/coil/subj/SUBM_s{s}"}
HEAD = "## RUN 2026-11-07 — GAP F: one training subject (learnability), and fastMRI-trained models on real ASL"


def f1():
    S.ARMS = {}
    TT = {}
    for m in ARMS:
        for src, path in (("asl", ASLREF[m]), ("T", "runs/gap/f/asl/" + m + "_T_s{s}"), ("DT", "runs/gap/f/asl/" + m + "_DT_s{s}")):
            S.ARMS[f"{m}_{src}"] = SEEDS
            for s in SEEDS:
                TT[(f"{m}_{src}", s)] = C.read(path.format(s=s))
    L = ["**F1 (exploratory): fastMRI-trained models on real ASL 1104c** (split-half n_eff, signal region; cross-fitted selection).", "",
         "| a | " + " | ".join(f"n_eff {m} {src}" for m in ARMS for src in ("asl", "T", "DT")) + " | SUB/PER asl | SUB/PER T | SUB/PER DT | SUBM/PER DT |",
         "|---:|" + "---:|" * 9 + "---|---|---|---|"]
    for a in S.A:
        ne = [np.exp(S.design(TT, f"{m}_{src}", 6, a)[1].mean()) for m in ARMS for src in ("asl", "T", "DT")]
        L.append(f"| {a} | " + " | ".join(f"{v:.1f}" for v in ne) + " | " + " | ".join(
            fmt(S.ratio(TT, x, y, 6, a)) for x, y in (("SUB_asl", "A_asl"), ("SUB_T", "A_T"), ("SUB_DT", "A_DT"), ("SUBM_DT", "A_DT"))) + " |")
    L += ["", "fastMRI-trained / ASL-trained, absolute n_eff at a = 9 (90% CI; slices higher):", ""]
    for m in ARMS:
        L.append(f"- {m}: T {fmt(S.ratio(TT, m + '_T', m + '_asl', 6, 9))}; DT {fmt(S.ratio(TT, m + '_DT', m + '_asl', 6, 9))}")
    return L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    TT = {(c, m, s): MR.read("62" if args.dry else c, m, s) for c in PATH for m in ARMS for s in SEEDS}
    V = {}
    for c in PATH:
        for m in ARMS:
            for a in A:
                per = [R.per_volume(TT[(c, m, s)], a) for s in SEEDS]
                vols = sorted(per[0])
                V[(c, m, a)] = np.array([np.mean([p[v][0] for p in per]) for v in vols])
    lg = {(c, m, a): V[(c, m, a)] - V[(c, "A", a)] for c in PATH for m in ARMS[1:] for a in A}
    prim = {"Q1 (SUB/PER, DT n_vol 1) / (SUB/PER, DT n_vol 62), a = 9": stat(lg[("1", "SUB", 9)] - lg[("62", "SUB", 9)]),
            "Q2 (SUB/PER, DT n_vol 4) / (SUB/PER, DT n_vol 62), a = 9": stat(lg[("4", "SUB", 9)] - lg[("62", "SUB", 9)])}
    ps = sorted(prim, key=lambda k: prim[k]["p"])
    for j, k in enumerate(ps):
        prim[k]["p_holm"] = min(1.0, max(prim[q]["p"] * (len(ps) - i) for i, q in enumerate(ps[:j + 1])))
    if args.dry:
        assert all(abs(v["estimate"] - 1) < 1e-12 for v in prim.values()), "dry run"
        print("DRY RUN OK"); return
    L = [HEAD, "", "Pre-registered (`Archive/GAP/PREREGISTRATION_F.md`).", "",
         "**F2 PRIMARY (Holm over 2)** — fastMRI DT (content with coil1 ASL's per-frequency SNR, real ASL noise), training "
         "volumes 1 / 4 / 62, PER (A), SUB, SUBM, seeds 101–102; n_eff vs ground truth on 15 test volumes:", ""]
    for k, v in prim.items():
        L.append(f"- {k}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; higher on {v['wins']}/{v['n']} volumes), "
                 f"p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", "| n_vol | " + " | ".join(f"SUB/PER a = {a}" for a in A) + " | SUBM/PER a = 9 | n_eff a = 9 PER / SUB / SUBM |", "|---|" + "---|" * 6]
    for c in PATH:
        L.append(f"| {c} | " + " | ".join(fmt(stat(lg[(c, 'SUB', a)])) for a in A) + f" | {fmt(stat(lg[(c, 'SUBM', 9)]))} | "
                 + " / ".join(f"{np.exp(V[(c, m, 9)].mean()):.1f}" for m in ARMS) + " |")
    L += [""] + f1()
    print("\n".join(L))
    os.makedirs(os.path.join(REPO, "runs", "gap", "report"), exist_ok=True)
    json.dump({"primary": prim}, open(os.path.join(REPO, "runs", "gap", "report", "summary_f.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
