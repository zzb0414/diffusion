# -*- coding: utf-8 -*-
"""GAP J2 (EXPLORATORY): checkpoint trajectory of the GAP J FT / SC models and the original-units references on 1104c.
Per checkpoint: geometric-mean n_eff over the 34 slices (signal region) at a = 1 and a = 9, at r = 0 and at the
best single r (oracle, so optimistic; no cross-fitting). Iterations 1000/2000/3000 from the GAP J / reference runs,
500/1500 from runs/gap/j2. Also the GAP F zero-shot fastMRI DT models (runs/gap/f/asl) for comparison.

    python experiments/_j2_traj.py
"""
import os, csv
import numpy as np
import _subj_report as S

REPO = S.REPO
ITS = (500, 1000, 1500, 2000, 3000)


def path(kind, arm, s, it):
    if it in (500, 1500):
        return f"runs/gap/j2/{kind}_{arm}_s{s}/eval_it{it}/neff.csv"
    if kind == "ref":
        base = f"runs/coil/subj/SUBM_s{s}" if arm == "SUBM" else f"runs/subj/{arm}_s{s}"
    else:
        base = f"runs/gap/j/{kind}_{arm}_s{s}"
    return f"{base}/eval_it{it}/neff.csv"


def gm(p, a):
    R = [r for r in csv.DictReader(open(os.path.join(REPO, p), encoding="utf-8")) if r["region"] == "signal" and int(r["a"]) == a]
    ks = [k for k in R[0] if k.startswith("neff_r")]
    g = np.exp(np.log(np.array([[float(r[k]) for k in ks] for r in R])).mean(0))
    return g[0], g.max()


def main():
    L = ["## RUN 2026-11-07 — GAP J2 (EXPLORATORY): checkpoint trajectory of fastMRI-initialized vs random-init ASL models", "",
         "Not pre-registered; written after the GAP J reading. Geometric-mean n_eff over 34 slices of 1104c (signal region); "
         "'best' = best single r (oracle, optimistic). Seed-mean of 101 and 102.", "",
         "| arm | model | a | " + " | ".join(f"it {it} r0 / best" for it in ITS) + " |", "|---|---|---:|" + "---|" * len(ITS)]
    for arm in ("PER", "SUB", "SUBM"):
        for kind in ("ref", "SC", "FT"):
            for a in (1, 9):
                cells = []
                for it in ITS:
                    v = np.array([gm(path(kind, arm, s, it), a) for s in (101, 102)]).mean(0)
                    cells.append(f"{v[0]:.1f} / {v[1]:.1f}")
                L.append(f"| {arm} | {kind} | {a} | " + " | ".join(cells) + " |")
    zs = {arm: np.array([[gm(f"runs/gap/f/asl/{'A' if arm == 'PER' else arm}_DT_s{s}/eval_it{it}/neff.csv", 9)[1]
                          for it in (1000, 2000, 3000)] for s in (101, 102)]).mean(0) for arm in ("PER", "SUB", "SUBM")}
    L += ["", "Zero-shot fastMRI DT models (GAP F1, no ASL training), a = 9, best r, fastMRI it 1000 / 2000 / 3000: "
          + "; ".join(f"{arm} " + " / ".join(f"{x:.1f}" for x in v) for arm, v in zs.items()) + "."]
    print("\n".join(L))
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if L[0] not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as fh:
            fh.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
