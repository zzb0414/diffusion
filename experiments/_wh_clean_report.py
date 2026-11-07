# -*- coding: utf-8 -*-
"""GAP K2 report (Archive/GAP/PREREGISTRATION_K2.md): GAP K whitened SUB / SUBM scored with a leakage-free (A-half) map
(runs/gap/k2) vs the original models and vs the GAP K full-map scores, 1104c. ``--dry`` sets clean := real. Appends once.

    python experiments/_wh_clean_report.py [--dry]
"""
import os, json, argparse
import _subj_report as S
import _crop_report as C
from _wh_report import stat, REPO

HEAD = "## RUN 2026-11-07 — GAP K2: leakage control — whitening gain with a map built from the model's own input half"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    S.ARMS = {f"{k}{t}": (101, 102) for k in ("SUB", "SUBM") for t in ("", "w", "c")}
    TT = {}
    for k in ("SUB", "SUBM"):
        for s in (101, 102):
            TT[(k, s)] = C.read(C.REF[k].format(s=s))
            TT[(k + "w", s)] = C.read(f"runs/gap/k/{k}_s{s}")
            TT[(k + "c", s)] = TT[(k, s)] if args.dry else C.read(f"runs/gap/k2/{k}_s{s}")
    L_ = lambda arm, a: S.design(TT, arm, 6, a)[1]
    ab = lambda x, y, a: stat(L_(x, a) - L_(y, a))
    prim = {"K2a n_eff(SUBM, whitened, clean map) / n_eff(SUBM, real), a = 9": ab("SUBMc", "SUBM", 9),
            "K2b n_eff(SUB, whitened, clean map) / n_eff(SUB, real), a = 9": ab("SUBc", "SUB", 9)}
    ps = sorted(prim, key=lambda q: prim[q]["p"])
    for j, q in enumerate(ps):
        prim[q]["p_holm"] = min(1.0, max(prim[x]["p"] * (len(ps) - i) for i, x in enumerate(ps[:j + 1])))
    if args.dry:
        assert all(abs(v["estimate"] - 1) < 1e-12 for v in prim.values()), "dry run"
        print("DRY RUN OK"); return
    f = lambda v: f"{v['estimate']:.3f} ({v['ci_lo']:.3f}–{v['ci_hi']:.3f}; {v['wins']}/{v['n']})"
    L = [HEAD, "", "Pre-registered (`Archive/GAP/PREREGISTRATION_K2.md`). Same GAP K models and scoring; for every A/B split the "
         "model's noise map is built from the A-half repetitions only. Signal region, checkpoint and r cross-fitted over slice folds.",
         "", "**PRIMARY (Holm over 2):**", ""]
    for q, v in prim.items():
        L.append(f"- {q}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; higher on {v['wins']}/{v['n']} slices), "
                 f"p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", "| model | a | clean / real | leaky (GAP K) / real | clean / leaky |", "|---|---:|---|---|---|"]
    for k in ("SUB", "SUBM"):
        for a in (5, 7, 9):
            L.append(f"| {k} | {a} | {f(ab(k + 'c', k, a))} | {f(ab(k + 'w', k, a))} | {f(ab(k + 'c', k + 'w', a))} |")
    print("\n".join(L))
    os.makedirs(os.path.join(REPO, "runs", "gap", "report"), exist_ok=True)
    json.dump({"primary": prim}, open(os.path.join(REPO, "runs", "gap", "report", "summary_k2.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as fh:
            fh.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
