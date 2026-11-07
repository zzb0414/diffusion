# -*- coding: utf-8 -*-
"""GAP L report (Archive/GAP/PREREGISTRATION_L.md): SC vs reference recipe (runs/gap/l), SUBM / SUB, seeds 103-104, 1104c,
checkpoint and r cross-fitted over slice folds on the 6-checkpoint grid. ``--dry`` sets SC := ref. Appends once.

    python experiments/_l_report.py [--dry]
"""
import os, json, argparse
import numpy as np
import _subj_report as S
import _crop_report as C
from _wh_report import stat, REPO

HEAD = "## RUN 2026-11-07 — GAP L: SC vs reference recipe, both early-stopped (fresh seeds)"
G6, G3 = (500, 1000, 1500, 2000, 2500, 3000), (1000, 2000, 3000)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    S.ITERS = G6
    S.ARMS = {f"{r}{k}": (103, 104) for k in ("SUBM", "SUB") for r in ("SC", "ref")}
    TT = {}
    for k in ("SUBM", "SUB"):
        for s in (103, 104):
            TT[("ref" + k, s)] = C.read(f"runs/gap/l/ref_{k}_s{s}")
            TT[("SC" + k, s)] = TT[("ref" + k, s)] if args.dry else C.read(f"runs/gap/l/SC_{k}_s{s}")

    def L_(arm, a, grid):
        S.ITERS = grid
        return S.design(TT, arm, 6, a)[1]

    ab = lambda x, y, a, gx=G6, gy=G6: stat(L_(x, a, gx) - L_(y, a, gy))
    prim = {"L1 n_eff(SC SUBM) / n_eff(ref SUBM), a = 9": ab("SCSUBM", "refSUBM", 9),
            "L2 n_eff(SC SUB) / n_eff(ref SUB), a = 9": ab("SCSUB", "refSUB", 9)}
    ps = sorted(prim, key=lambda q: prim[q]["p"])
    for j, q in enumerate(ps):
        prim[q]["p_holm"] = min(1.0, max(prim[x]["p"] * (len(ps) - i) for i, x in enumerate(ps[:j + 1])))
    if args.dry:
        assert all(abs(v["estimate"] - 1) < 1e-12 for v in prim.values()), "dry run"
        print("DRY RUN OK"); return
    f = lambda v: f"{v['estimate']:.3f} ({v['ci_lo']:.3f}–{v['ci_hi']:.3f}; {v['wins']}/{v['n']})"
    L = [HEAD, "", "Pre-registered (`Archive/GAP/PREREGISTRATION_L.md`). SC (`_ft_train.py`, no init) vs reference (`_gr_train.py`), "
         "datasets 1 + 1104a, seeds 103–104, 3000 iterations; 1104c, signal region; checkpoint and r cross-fitted over slice folds.",
         "", "**PRIMARY (Holm over 2, 6-checkpoint grid):**", ""]
    for q, v in prim.items():
        L.append(f"- {q}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; higher on {v['wins']}/{v['n']} slices), "
                 f"p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", "| arm | a | SC / ref (6-grid) | SC / ref (3-grid) | SC / ref (it 3000 only) | SC 6/3-grid | ref 6/3-grid | n_eff SC / ref (6-grid) |",
          "|---|---:|---|---|---|---|---|---|"]
    for k in ("SUBM", "SUB"):
        for a in (1, 9):
            ne = " / ".join(f"{np.exp(L_(r + k, a, G6).mean()):.1f}" for r in ("SC", "ref"))
            L.append(f"| {k} | {a} | {f(ab('SC' + k, 'ref' + k, a))} | {f(ab('SC' + k, 'ref' + k, a, G3, G3))} | "
                     f"{f(ab('SC' + k, 'ref' + k, a, (3000,), (3000,)))} | {f(ab('SC' + k, 'SC' + k, a, G6, G3))} | "
                     f"{f(ab('ref' + k, 'ref' + k, a, G6, G3))} | {ne} |")
    S.ITERS = G6
    L += ["", "Cross-fitted checkpoint chosen (fold 0 shown per seed), a = 9: " + "; ".join(
        f"{r}{k} s{s} it {min(((it, rr) for it in G6 for rr in next(iter(TT[(r + k, s)][(it, 6, 9)].values()))[1]), key=lambda c: np.mean([1 / v[1][c[1]] for v in TT[(r + k, s)][(c[0], 6, 9)].values() if v[0] != 0]))[0]}"
        for k in ("SUBM", "SUB") for r in ("SC", "ref") for s in (103, 104))]
    print("\n".join(L))
    os.makedirs(os.path.join(REPO, "runs", "gap", "report"), exist_ok=True)
    json.dump({"primary": prim}, open(os.path.join(REPO, "runs", "gap", "report", "summary_l.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as fh:
            fh.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
