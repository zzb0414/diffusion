# -*- coding: utf-8 -*-
"""GAP D/E report (Archive/GAP/PREREGISTRATION_D.md): does fastMRI content with ASL's per-frequency SNR show ASL's small
denoise-the-mean gain, and does phase structure matter at a fixed spectrum?

Conditions (matrix 80, sigma 1, PER = arm A, SUB, SUBM, seeds 101-102): T, W (GAP B, read only); DT, DW, EW
(runs/gap/d/eval). Selection without ground truth as mean_report. Wiener predictions from the exact test-split content
and noise spectra (resolution/wiener.gain). ``--dry`` sets every condition := W (ratios 1). Appends once unless --dry.

    python resolution/spectral_report.py [--dry]
"""
import os, json, argparse
import numpy as np
import data as D
import transplant as TP
import wiener as WI
import mean_report as R
import matrix_report as MR
from transplant_report import stat, fmt

REPO = R.REPO
MR.PATH = PATH = {"T": "runs/gap/b/eval/{arm}_T_s{s}_it{it}.csv", "W": "runs/gap/b/eval/{arm}_W_s{s}_it{it}.csv",
                  "DT": "runs/gap/d/eval/{arm}_DT_s{s}_it{it}.csv", "DW": "runs/gap/d/eval/{arm}_DW_s{s}_it{it}.csv",
                  "EW": "runs/gap/d/eval/{arm}_EW_s{s}_it{it}.csv"}
ARMS, SEEDS, A = ("A", "SUB", "SUBM"), (101, 102), (1, 3, 5, 9)
ASL = {"SUB": 1.062, "SUBM": 1.103}                    # SUB/PER, SUBM/PER on 1104c at a = 9 (GAP A1/A2, ASL-CROP refs)
HEAD = "## RUN 2026-11-07 — GAP D/E: is ASL's small denoise-the-mean gain set by its signal-to-noise spectrum?"


def wiener():
    x, _ = D.load_clean("test"); out = {}
    for c in PATH:
        kind, nz = ("F", c) if len(c) == 1 else (c[0], c[1])
        P = np.mean([np.abs(np.fft.fft2(TP.content(z, "test", i, kind, nz), norm="ortho")) ** 2 for i, z in enumerate(x)], 0)
        Nn = TP.ring(TP.pool("test")[1]) if nz == "T" else np.full(P.shape, 2 * TP.SIG ** 2)
        out[c] = WI.gain(P, Nn, False)[-1]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    TT = {(c, m, s): MR.read("W" if args.dry else c, m, s) for c in PATH for m in ARMS for s in SEEDS}
    V = {}
    for c in PATH:
        for m in ARMS:
            for a in A:
                per = [R.per_volume(TT[(c, m, s)], a) for s in SEEDS]
                vols = sorted(per[0])
                V[(c, m, a)] = (np.array([np.mean([p[v][0] for p in per]) for v in vols]),
                                np.array([np.mean([p[v][1] for p in per], 0) for v in vols]))
    lg = {(c, m, a): V[(c, m, a)][0] - V[(c, "A", a)][0] for c in PATH for m in ARMS[1:] for a in A}
    prim = {"P1 spectrum, ASL noise: (SUB/PER, DT) / (SUB/PER, T), a = 9": stat(lg[("DT", "SUB", 9)] - lg[("T", "SUB", 9)]),
            "P2 spectrum, white noise: (SUB/PER, DW) / (SUB/PER, W), a = 9": stat(lg[("DW", "SUB", 9)] - lg[("W", "SUB", 9)]),
            "P3 phase structure: (SUB/PER, EW) / (SUB/PER, DW), a = 9": stat(lg[("EW", "SUB", 9)] - lg[("DW", "SUB", 9)])}
    ps = sorted(prim, key=lambda k: prim[k]["p"])
    for j, k in enumerate(ps):
        prim[k]["p_holm"] = min(1.0, max(prim[q]["p"] * (len(ps) - i) for i, q in enumerate(ps[:j + 1])))
    if args.dry:
        assert all(abs(v["estimate"] - 1) < 1e-12 for v in prim.values()), "dry run"
        print("DRY RUN OK"); return
    g = {(c, m): float(np.exp(lg[(c, m, 9)].mean())) for c in PATH for m in ARMS[1:]}
    share = {m: float(np.log(g[("T", m)] / g[("DT", m)]) / np.log(g[("T", m)] / ASL[m])) for m in ARMS[1:]}
    WN = wiener()
    L = [HEAD, "", "Pre-registered (`Archive/GAP/PREREGISTRATION_D.md`). fastMRI AXT2 at matrix 80, σ = 1 per channel, 62 training "
         "volumes, PER (arm A), SUB, SUBM, seeds 101–102; checkpoint and r chosen without ground truth; n_eff vs ground truth on "
         "15 test volumes. Content F = fastMRI (GAP B), D = filtered to coil1 ASL's per-frequency SNR for its noise, E = D with "
         "random Fourier phase. Noise T = real ASL fields, W = white.", "", "**PRIMARY (Holm over 3):**", ""]
    for k, v in prim.items():
        L.append(f"- {k}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; higher on {v['wins']}/{v['n']} volumes), "
                 f"p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", f"Share of the T → ASL gap closed by DT (log): SUB {share['SUB']:.2f} (T {g[('T', 'SUB')]:.3f} → DT {g[('DT', 'SUB')]:.3f}; "
          f"ASL {ASL['SUB']}), SUBM {share['SUBM']:.2f} (T {g[('T', 'SUBM')]:.3f} → DT {g[('DT', 'SUBM')]:.3f}; ASL {ASL['SUBM']}).",
          "", "SUB / PER and SUBM / PER (90% CI; volumes higher), Wiener and learned / Wiener at a = 9:", "",
          "| condition | " + " | ".join(f"SUB/PER a = {a}" for a in A) + " | SUBM/PER a = 9 | Wiener a = 9 | SUB learned/W | SUBM learned/W |",
          "|---|" + "---|" * 5 + "---:|---:|---:|"]
    for c in PATH:
        L.append(f"| {c} | " + " | ".join(fmt(stat(lg[(c, 'SUB', a)])) for a in A) + f" | {fmt(stat(lg[(c, 'SUBM', 9)]))} | {WN[c]:.3f} | "
                 f"{g[(c, 'SUB')] / WN[c]:.3f} | {g[(c, 'SUBM')] / WN[c]:.3f} |")
    L += ["", "n_eff (geometric mean over volumes), PER / SUB / SUBM:", "", "| a | " + " | ".join(PATH) + " |", "|---:|" + "---|" * len(PATH)]
    for a in A:
        L.append(f"| {a} | " + " | ".join(" / ".join(f"{np.exp(V[(c, m, a)][0].mean()):.1f}" for m in ARMS) for c in PATH) + " |")
    L += ["", "Band transfer at a = 9 (bands of k / k_max; plain averaging = 1):", "",
          "| condition | design | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for c in PATH:
        for m in ARMS:
            L.append(f"| {c} | {m} | " + " | ".join(f"{t:.2f}" for t in V[(c, m, 9)][1].mean(0)) + " |")
    print("\n".join(L))
    os.makedirs(os.path.join(REPO, "runs", "gap", "report"), exist_ok=True)
    json.dump({"primary": prim, "share": share, "wiener": WN}, open(os.path.join(REPO, "runs", "gap", "report", "summary_d.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
