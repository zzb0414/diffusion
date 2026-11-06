# -*- coding: utf-8 -*-
"""ASL-CROP report (Archive/CROP/PREREGISTRATION.md): does training on 32-pixel crops (batch 32) instead of whole
80 x 80 slices (batch 8) help denoise-the-mean on the unseen subject 1104c?

Crop models: runs/crop/<arm>_s<seed>; whole-slice references (read only): runs/subj/{PER,SUB}_s<seed>,
runs/coil/subj/SUBM_s<seed>. All trained on dataset 1 + 1104a, seeds 101-102, evaluated by ``_gr_eval.py --single``
on 1104c. Selection without ground truth exactly as ``_subj_report`` (checkpoint and r cross-fitted over slice folds).
``--dry`` sets every crop model := its reference (ratios 1). Appends once unless --dry.

    python experiments/_crop_report.py [--dry]
"""
import os, csv, json, argparse
from collections import defaultdict
import _subj_report as S

REPO = S.REPO
REF = {"PER": "runs/subj/PER_s{s}", "SUB": "runs/subj/SUB_s{s}", "SUBM": "runs/coil/subj/SUBM_s{s}"}
CROP = {k: "runs/crop/" + k + "_s{s}" for k in REF}
HEAD = "## RUN 2026-11-06 — ASL-CROP: does training on crops help denoise-the-mean on ASL?"


def read(path):
    T = defaultdict(dict)
    for it in S.ITERS:
        for r in csv.DictReader(open(os.path.join(REPO, path, f"eval_it{it}", "neff.csv"), encoding="utf-8")):
            if r["region"] != "signal":
                continue
            T[(it, int(r["dataset"]), int(r["a"]))][int(r["slice"])] = (
                int(r["fold"]), {float(k[6:]): float(v) for k, v in r.items() if k.startswith("neff_r")})
    return T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    S.ARMS = {f"{k}{t}": (101, 102) for k in REF for t in ("", "c")}
    TT = {}
    for k in REF:
        for s in (101, 102):
            TT[(k, s)] = read(REF[k].format(s=s))
            TT[(k + "c", s)] = TT[(k, s)] if args.dry else read(CROP[k].format(s=s))
    prim = {f"P{i} {k} crop / {k} whole slice, a = 9": S.ratio(TT, k + "c", k, 6, 9) for i, k in enumerate(("SUBM", "SUB", "PER"), 1)}
    ps = sorted(prim, key=lambda q: prim[q]["p"])
    for j, q in enumerate(ps):
        prim[q]["p_holm"] = min(1.0, max(prim[x]["p"] * (len(ps) - i) for i, x in enumerate(ps[:j + 1])))
    if args.dry:
        assert all(abs(v["estimate"] - 1) < 1e-12 for v in prim.values()), "dry run"
        print("DRY RUN OK"); return
    f = lambda v: f"{v['estimate']:.3f} ({v['ci_lo']:.2f}–{v['ci_hi']:.2f}; {v['wins']}/{v['n']})"
    L = [HEAD, "", "Pre-registered (`Archive/CROP/PREREGISTRATION.md`). Trained on dataset 1 + 1104a (seeds 101–102, 3000 "
         "iterations, MSE), tested on the unseen subject 1104c; crop models use random 32 × 32 crops, batch 32 (same pixels "
         "per step as whole 80 × 80 slices at batch 8); references are ASL-SUBJ (PER, SUB) and ASL-COIL (SUBM). "
         "Checkpoint and r cross-fitted over slice folds, no ground truth.", "", "**PRIMARY (Holm over 3):**", ""]
    for q, v in prim.items():
        L.append(f"- {q}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; higher on {v['wins']}/{v['n']} "
                 f"slices), p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", "Ratios (90% CI; slices higher):", "", "| a | SUBM c / SUBM | SUB c / SUB | PER c / PER | SUB/PER whole | SUB/PER crop | "
          "SUBM/PER whole | SUBM/PER crop | SUBM crop / PER whole |", "|---:|---|---|---|---|---|---|---|---|"]
    for a in S.A:
        L.append(f"| {a} | " + " | ".join(f(S.ratio(TT, x, y, 6, a)) for x, y in (("SUBMc", "SUBM"), ("SUBc", "SUB"), ("PERc", "PER"),
                 ("SUB", "PER"), ("SUBc", "PERc"), ("SUBM", "PER"), ("SUBMc", "PERc"), ("SUBMc", "PER"))) + " |")
    print("\n".join(L))
    os.makedirs(os.path.join(REPO, "runs", "crop", "report"), exist_ok=True)
    json.dump({"primary": prim}, open(os.path.join(REPO, "runs", "crop", "report", "summary.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as fh:
            fh.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
