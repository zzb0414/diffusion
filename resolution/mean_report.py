# -*- coding: utf-8 -*-
"""MEAN study report (Archive/MEAN/PREREGISTRATION.md). Reads runs/mean/eval/<arm>_v<n>_s<seed>_it<iter>.csv.

n_eff = sigma^2 / mse_gt_head per test image. Per (design, n_vol, seed, a) and per test volume, the checkpoint and the dial
r are chosen jointly from the OTHER volumes' split-half MSE (no ground truth). Per volume: geometric mean over its 8
images; seeds: mean of log n_eff per volume. Band transfer T1..T5 at the same choice (arithmetic means). ``--dry`` sets
SUB := PER (ratio 1, difference 0). Appends the log section to WEEKEND_ITERATIONS.md once unless --dry.

    python resolution/mean_report.py [--dry]
"""
import os, csv, json, argparse
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SK = float(os.environ.get("RES_SK", "1.0"))          # noise scale; 1 = the MEAN study, else MEAN-SNR (Archive/MEAN/PREREGISTRATION_SNR.md)
MDIR = "mean" if SK == 1.0 else f"mean_sk{SK:g}"
NVOL, SEEDS, A, ITERS = ((2, 8, 62) if SK == 1.0 else (62,)), (101, 102), (1, 3, 5, 9), (1000, 2000, 3000)
TB = ("T1", "T2", "T3", "T4", "T5")
HEAD = ("## RUN 2026-11-03 — MEAN: denoise the mean vs denoise each, then average (fastMRI, training-data scaling)" if SK == 1.0
        else f"## RUN 2026-11-04 — MEAN-SNR: denoise the mean vs denoise each at ASL-like noise (fastMRI, S_K = {SK:g})")
rng = np.random.default_rng(0)


def read(arm, n, seed):
    """{(a, iter, r): {img: (vol, sigma, mse_gt, mse_sh, T(5))}}"""
    T = defaultdict(dict)
    for it in ITERS:
        for r in csv.DictReader(open(os.path.join(REPO, "runs", MDIR, "eval", f"{arm}_v{n}_s{seed}_it{it}.csv"), encoding="utf-8")):
            T[(int(r["a"]), it, float(r["r"]))][int(r["img"])] = (int(r["vol"]), float(r["sigma"]), float(r["mse_gt_head"]),
                                                                  float(r["mse_sh_head"]), [float(r[b]) for b in TB])
    return T


def per_volume(T, a):
    """{vol: (log n_eff, T(5))} at the (iter, r) chosen from the other volumes' split-half MSE."""
    cand = sorted(k[1:] for k in T if k[0] == a)
    vols = sorted({e[0] for e in T[(a,) + cand[0]].values()})
    out = {}
    for v in vols:
        c = min(cand, key=lambda c: sum(e[3] for e in T[(a,) + c].values() if e[0] != v))
        E = [e for e in T[(a,) + c].values() if e[0] == v]
        out[v] = (float(np.mean([np.log(e[1] ** 2 / e[2]) for e in E])), np.mean([e[4] for e in E], 0))
    return out


def design(TT, arm, n, a):
    per = [per_volume(TT[(arm, n, s)], a) for s in SEEDS]
    vols = sorted(per[0])
    return (np.array([np.mean([p[v][0] for p in per]) for v in vols]),
            np.array([np.mean([p[v][1] for p in per], 0) for v in vols]))


def boot(d, log=True):
    b = rng.choice(d, size=(10000, len(d))).mean(1)
    f = np.exp if log else (lambda z: z)
    return float(f(d.mean())), float(f(np.percentile(b, 5))), float(f(np.percentile(b, 95)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    TT = {(arm, n, s): read(arm, n, s) for arm in ("A", "SUB") for n in NVOL for s in SEEDS}
    if args.dry:
        for n in NVOL:
            for s in SEEDS:
                TT[("SUB", n, s)] = TT[("A", n, s)]
    V = {(arm, n, a): design(TT, arm, n, a) for arm in ("A", "SUB") for n in NVOL for a in A}
    d1 = V[("SUB", 62, 9)][0] - V[("A", 62, 9)][0]
    d2 = V[("SUB", 62, 9)][1][:, 3] - V[("A", 62, 9)][1][:, 3]
    prim = {}
    for key, d, log in (("P1 n_eff SUB/PER, 62 volumes, a = 9", d1, True), ("P2 band T4 SUB - PER, 62 volumes, a = 9", d2, False)):
        g, lo, hi = boot(d, log)
        p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
        prim[key] = dict(estimate=g, ci_lo=lo, ci_hi=hi, p=p, wins=int((d > 0).sum()), n=len(d))
    ps = sorted(prim, key=lambda k: prim[k]["p"])
    for j, k in enumerate(ps):
        prim[k]["p_holm"] = min(1.0, max(prim[q]["p"] * (len(ps) - i) for i, q in enumerate(ps[:j + 1])))
    if args.dry:
        assert all(abs(v["estimate"] - (1 if k.startswith("P1") else 0)) < 1e-12 for k, v in prim.items()), "dry run"
        print("DRY RUN OK"); return
    gm = lambda z: float(np.exp(z.mean()))                                                   # noqa: E731
    L = [HEAD, "", f"Pre-registered (`Archive/MEAN/PREREGISTRATION{'' if SK == 1.0 else '_SNR'}.md`). fastMRI AXT2 at matrix 160, physical noise σ = {0.5 * SK:g} "
         "per channel, N = 10 copies; PER = arm A (denoise each copy, average, dial), SUB = denoise the mean of the a copies "
         "(σ/√a, dial). Training on the first " + " / ".join(map(str, NVOL)) + " training volumes (" + " / ".join(str(8 * n) for n in NVOL) + " images), seeds 101–102; checkpoint "
         "and r chosen without ground truth (leave-one-volume-out split-half); n_eff vs ground truth on 15 test volumes.", "",
         "**PRIMARY (Holm over 2):**", ""]
    for k, v in prim.items():
        L.append(f"- {k}: {v['estimate']:.3f}{'×' if k.startswith('P1') else ''} (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; "
                 f"SUB higher on {v['wins']}/{v['n']} volumes), p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", "n_eff (geometric mean over volumes) and SUB / PER (90% CI):", "",
          "| training volumes | a | PER | SUB | SUB / PER |", "|---:|---:|---:|---:|---|"]
    for n in NVOL:
        for a in A:
            d = V[("SUB", n, a)][0] - V[("A", n, a)][0]; g, lo, hi = boot(d)
            L.append(f"| {n} | {a} | {gm(V[('A', n, a)][0]):.2f} | {gm(V[('SUB', n, a)][0]):.2f} | {g:.3f} ({lo:.2f}–{hi:.2f}) |")
    L += ["", "Band transfer at a = 9 (share of the true detail kept per band of k / k_max; plain averaging = 1 in every band):", "",
          "| design | training volumes | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, nm in (("A", "PER"), ("SUB", "SUB")):
        for n in NVOL:
            L.append(f"| {nm} | {n} | " + " | ".join("%.2f" % t for t in V[(arm, n, 9)][1].mean(0)) + " |")
    print("\n".join(L))
    os.makedirs(os.path.join(REPO, "runs", MDIR, "report"), exist_ok=True)
    json.dump({"primary": prim}, open(os.path.join(REPO, "runs", MDIR, "report", "summary.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
