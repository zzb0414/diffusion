# -*- coding: utf-8 -*-
"""GAP-M report (Archive/GAP/PREREGISTRATION_M.md): uneven noise co-located with the signal (DL), the same envelope
shifted (DS), and DL with a noise-map remedy (DLw), against white noise (DW, GAP-D, read only). Content D, matrix 80,
PER (A) / SUB / SUBM, seeds 101-102; selection without ground truth (mean_report.per_volume); n_eff vs ground truth.
``--dry`` sets every condition := DW (ratios 1). Appends once unless --dry.

    python resolution/coloc_report.py [--dry]
"""
import os, json, argparse
import numpy as np
import mean_report as R
import matrix_report as MR
from transplant_report import stat, fmt

REPO = R.REPO
MR.PATH = PATH = {"DW": "runs/gap/d/eval/{arm}_DW_s{s}_it{it}.csv", "DL": "runs/gap/m/eval/{arm}_DL_s{s}_it{it}.csv",
                  "DS": "runs/gap/m/eval/{arm}_DS_s{s}_it{it}.csv", "DLw": "runs/gap/m/eval/{arm}_DLw_s{s}_it{it}.csv"}
ARMS, SEEDS, A = ("A", "SUB", "SUBM"), (101, 102), (1, 3, 5, 9)
ASL = {"SUB": 1.062, "SUBM": 1.103}
HEAD = "## RUN 2026-11-09 — GAP-M: does uneven noise cost denoise-the-mean only when its strength follows the signal?"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    TT = {(c, m, s): MR.read("DW" if args.dry else c, m, s) for c in PATH for m in ARMS for s in SEEDS}
    V = {}
    for c in PATH:
        for m in ARMS:
            for a in A:
                per = [R.per_volume(TT[(c, m, s)], a) for s in SEEDS]
                vols = sorted(per[0])
                V[(c, m, a)] = (np.array([np.mean([p[v][0] for p in per]) for v in vols]),
                                np.array([np.mean([p[v][1] for p in per], 0) for v in vols]))
    lg = {(c, m, a): V[(c, m, a)][0] - V[(c, "A", a)][0] for c in PATH for m in ARMS[1:] for a in A}
    prim = {"M1 co-located unevenness: (SUB/PER, DL) / (SUB/PER, DW), a = 9": stat(lg[("DL", "SUB", 9)] - lg[("DW", "SUB", 9)]),
            "M2 co-location specifically: (SUB/PER, DL) / (SUB/PER, DS), a = 9": stat(lg[("DL", "SUB", 9)] - lg[("DS", "SUB", 9)])}
    ps = sorted(prim, key=lambda k: prim[k]["p"])
    for j, k in enumerate(ps):
        prim[k]["p_holm"] = min(1.0, max(prim[q]["p"] * (len(ps) - i) for i, q in enumerate(ps[:j + 1])))
    if args.dry:
        assert all(abs(v["estimate"] - 1) < 1e-12 for v in prim.values()), "dry run"
        print("DRY RUN OK"); return
    ab = lambda c1, c2, m, a: stat(V[(c1, m, a)][0] - V[(c2, m, a)][0])
    g = {(c, m): float(np.exp(lg[(c, m, 9)].mean())) for c in PATH for m in ARMS[1:]}
    share = {m: float(np.log(g[("DW", m)] / g[("DL", m)]) / np.log(g[("DW", m)] / ASL[m])) for m in ARMS[1:]}
    L = [HEAD, "", "Pre-registered (`Archive/GAP/PREREGISTRATION_M.md`). Content D (fastMRI at ASL's per-frequency SNR), matrix 80, "
         "σ = 1, 62 training volumes, PER (arm A), SUB, SUBM, seeds 101–102; checkpoint and r chosen without ground truth; n_eff vs "
         "ground truth on 15 test volumes (relative to σ = 1). Noise: DW white (GAP-D); DL white × an envelope following the image's "
         "own signal region (in/out variance 1.37, as ASL); DS the same envelope shifted half the FOV; DLw = DL with a per-image "
         "noise map (input / map, output × map).", "", "**PRIMARY (Holm over 2):**", ""]
    for k, v in prim.items():
        L.append(f"- {k}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; higher on {v['wins']}/{v['n']} volumes), "
                 f"p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", "Secondary (90% CI; volumes higher):", "",
          f"- SUBM: (SUBM/PER, DL)/(DW) {fmt(stat(lg[('DL', 'SUBM', 9)] - lg[('DW', 'SUBM', 9)]))}; (DL)/(DS) {fmt(stat(lg[('DL', 'SUBM', 9)] - lg[('DS', 'SUBM', 9)]))}",
          f"- Remedy, SUB/PER: (DLw)/(DL) {fmt(stat(lg[('DLw', 'SUB', 9)] - lg[('DL', 'SUB', 9)]))}; SUBM/PER {fmt(stat(lg[('DLw', 'SUBM', 9)] - lg[('DL', 'SUBM', 9)]))}",
          "- Remedy, absolute n_eff DLw / DL at a = 9: " + "; ".join(f"{m} {fmt(ab('DLw', 'DL', m, 9))}" for m in ARMS),
          "- Absolute n_eff at a = 9, DL / DW: " + "; ".join(f"{m} {fmt(ab('DL', 'DW', m, 9))}" for m in ARMS)
          + " — DS / DW: " + "; ".join(f"{m} {fmt(ab('DS', 'DW', m, 9))}" for m in ARMS) + " (DL has more noise in the head, so it is lower by construction)",
          f"- Log share of the DW → ASL gap reproduced by DL: SUB {share['SUB']:.2f} (DW {g[('DW', 'SUB')]:.3f} → DL {g[('DL', 'SUB')]:.3f}; ASL 1.062), "
          f"SUBM {share['SUBM']:.2f} (DW {g[('DW', 'SUBM')]:.3f} → DL {g[('DL', 'SUBM')]:.3f}; ASL 1.103)",
          "", "| condition | " + " | ".join(f"SUB/PER a = {a}" for a in A) + " | SUBM/PER a = 9 | n_eff a = 9 PER / SUB / SUBM |", "|---|" + "---|" * 6]
    for c in PATH:
        L.append(f"| {c} | " + " | ".join(fmt(stat(lg[(c, 'SUB', a)])) for a in A) + f" | {fmt(stat(lg[(c, 'SUBM', 9)]))} | "
                 + " / ".join(f"{np.exp(V[(c, m, 9)][0].mean()):.1f}" for m in ARMS) + " |")
    L += ["", "Band transfer at a = 9 (bands of k / k_max):", "", "| condition | design | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–1 |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for c in PATH:
        for m in ARMS:
            L.append(f"| {c} | {m} | " + " | ".join(f"{t:.2f}" for t in V[(c, m, 9)][1].mean(0)) + " |")
    print("\n".join(L))
    os.makedirs(os.path.join(REPO, "runs", "gap", "report"), exist_ok=True)
    json.dump({"primary": prim, "share": share}, open(os.path.join(REPO, "runs", "gap", "report", "summary_m.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
