# -*- coding: utf-8 -*-
"""MEAN-TAX report (Archive/MEAN/PREREGISTRATION_TAX.md): can the target-noise tax of denoise-the-mean be removed?

Arms (fastMRI, matrix 160, S_K from RES_SK, 62 volumes, seeds 101-102): SUBW (sample weight = target precision N - m),
SUBM (m drawn proportional to m), compared with the MEAN-SNR runs of SUB and A at the same S_K (same eval CSVs, same
selection without ground truth as ``mean_report``). ``--dry`` sets SUBW := SUBM := SUB (ratios 1).
Appends its log section to WEEKEND_ITERATIONS.md once per S_K unless --dry.

    RES_SK=6 python resolution/tax_report.py [--dry]
"""
import os, json, argparse
import numpy as np
from scipy.stats import wilcoxon
import mean_report as R

ARMS, N_VOL = ("A", "SUB", "SUBW", "SUBM"), 62
HEAD = f"## RUN 2026-11-04 — MEAN-TAX: removing the target-noise tax of denoise-the-mean (fastMRI, S_K = {R.SK:g})"


def ratio(V, x, y, a):
    d = V[(x, a)][0] - V[(y, a)][0]
    g, lo, hi = R.boot(d, True)
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
    return dict(estimate=g, ci_lo=lo, ci_hi=hi, p=p, wins=int((d > 0).sum()), n=len(d))


def oracle_mse(T, a):
    """best ground-truth MSE (geometric mean over images) over checkpoints and r -- exploratory, not a read-out."""
    return min(float(np.exp(np.mean([np.log(e[2]) for e in v.values()]))) for k, v in T.items() if k[0] == a)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    TT = {(arm, N_VOL, s): R.read(arm, N_VOL, s) for arm in (ARMS[:2] if args.dry else ARMS) for s in R.SEEDS}
    if args.dry:
        for s in R.SEEDS:
            TT[("SUBW", N_VOL, s)] = TT[("SUBM", N_VOL, s)] = TT[("SUB", N_VOL, s)]
    V = {(arm, a): R.design(TT, arm, N_VOL, a) for arm in ARMS for a in R.A}
    prim = {"P1 SUBW / SUB, a = 9": ratio(V, "SUBW", "SUB", 9), "P2 SUBM / SUB, a = 9": ratio(V, "SUBM", "SUB", 9)}
    ps = sorted(prim, key=lambda k: prim[k]["p"])
    for j, k in enumerate(ps):
        prim[k]["p_holm"] = min(1.0, max(prim[q]["p"] * (len(ps) - i) for i, q in enumerate(ps[:j + 1])))
    if args.dry:
        assert all(abs(v["estimate"] - 1) < 1e-12 for v in prim.values()), "dry run"
        print("DRY RUN OK"); return
    L = [HEAD, "", f"Pre-registered (`Archive/MEAN/PREREGISTRATION_TAX.md`). fastMRI AXT2 at matrix 160, σ = {0.5 * R.SK:g} "
         "per channel, N = 10, 62 training volumes, seeds 101–102. SUBW: denoise the mean with each sample weighted by its "
         "target's precision (N − m); SUBM: denoise the mean with m drawn ∝ m. SUB and PER (arm A) are the MEAN-SNR runs at "
         "the same S_K. Checkpoint and r chosen without ground truth (leave-one-volume-out split-half); n_eff vs ground truth "
         "on 15 test volumes.", "", "**PRIMARY (Holm over 2):**", ""]
    for k, v in prim.items():
        L.append(f"- {k}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; higher on {v['wins']}/{v['n']} "
                 f"volumes), p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", "n_eff (geometric mean over volumes) and ratios to PER (90% CI):", "",
          "| a | PER | SUB | SUBW | SUBM | SUB / PER | SUBW / PER | SUBM / PER |", "|---:|---:|---:|---:|---:|---|---|---|"]
    for a in R.A:
        q = {x: ratio(V, x, "A", a) for x in ARMS[1:]}
        L.append(f"| {a} | " + " | ".join(f"{np.exp(V[(x, a)][0].mean()):.1f}" for x in ARMS) + " | "
                 + " | ".join(f"{q[x]['estimate']:.3f} ({q[x]['ci_lo']:.2f}–{q[x]['ci_hi']:.2f})" for x in ARMS[1:]) + " |")
    L += ["", "Band transfer at a = 9 (same choice; plain averaging = 1):", "",
          "| design | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–1 |", "|---|---:|---:|---:|---:|---:|"]
    for x in ARMS:
        L.append(f"| {'PER' if x == 'A' else x} | " + " | ".join(f"{t:.2f}" for t in V[(x, 9)][1].mean(0)) + " |")
    L += ["", "Exploratory — oracle (best checkpoint and r by ground truth) MSE, seed 101, by decode noise "
          "(SUB arms decode at σ/√a; a monotone tax-free model gets lower MSE as a grows):", "",
          "| a | decode σ (SUB arms) | PER | SUB | SUBW | SUBM |", "|---:|---:|---:|---:|---:|---:|"]
    for a in R.A:
        L.append(f"| {a} | {0.5 * R.SK / np.sqrt(a):.2f} | " +
                 " | ".join(f"{oracle_mse(TT[(x, N_VOL, 101)], a):.4f}" for x in ARMS) + " |")
    print("\n".join(L))
    os.makedirs(os.path.join(R.REPO, "runs", R.MDIR, "report_tax"), exist_ok=True)
    json.dump({"primary": prim}, open(os.path.join(R.REPO, "runs", R.MDIR, "report_tax", "summary.json"), "w"), indent=2)
    log = os.path.join(R.REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
