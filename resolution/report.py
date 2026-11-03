# -*- coding: utf-8 -*-
"""Multi-resolution study report (Archive/RES/PREREGISTRATION.md). Reads runs/res/eval/<arm>_s<seed>.csv.

n_eff = sigma^2 / mse_gt_head per test image; the dial r is chosen per (arm, seed, setting, M, a) and per test volume
from the OTHER volumes' split-half MSE (no ground truth). Per volume: geometric mean over its images; seeds: geometric
mean per volume; ratios are per volume (15). ``--dry`` sets C := B (every C/B ratio must be 1). Appends the log section
to WEEKEND_ITERATIONS.md once (guarded by its heading) unless --dry.

    python resolution/report.py [--dry]
"""
import os, sys, csv, json, argparse
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as Dd                                                        # noqa: E402
from scipy.stats import wilcoxon

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARMS, SEEDS, RES, A = ("A", "B", "C", "D", "E"), (101, 102, 103), (320, 240, 160, 128, 80), (1, 3, 5, 9)
HEAD = "## RUN 2026-11-03 — RES: one denoiser across resolutions (fastMRI, physical noise)" + (
    "" if Dd.S_K == 1.0 else f" — low-noise repeat (S_K = {Dd.S_K:g})")
rng = np.random.default_rng(0)


def read(arm, seed):
    """{(setting, M, a, r): {img: (vol, sigma, mse_gt, mse_sh)}}"""
    T = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(REPO, "runs", Dd.RUN_DIR, "eval", f"{arm}_s{seed}.csv"), encoding="utf-8")):
        T[(r["setting"], int(r["M"]), int(r["a"]), float(r["r"]))][int(r["img"])] = (
            int(r["vol"]), float(r["sigma"]), float(r["mse_gt_head"]), float(r["mse_sh_head"]))
    return T


def per_volume(T, setting, M, a, key="gt"):
    """{vol: log n_eff} at the cross-fitted r (geometric mean over the volume's images)."""
    rs = sorted({k[3] for k in T if k[:3] == (setting, M, a)})
    imgs = T[(setting, M, a, rs[0])]
    vols = sorted({v[0] for v in imgs.values()})
    out = {}
    for v in vols:
        r = min(rs, key=lambda r: sum(e[3] for e in T[(setting, M, a, r)].values() if e[0] != v))
        L = [np.log(e[1] ** 2 / (e[2] if key == "gt" else e[3])) for e in T[(setting, M, a, r)].values() if e[0] == v]
        out[v] = float(np.mean(L))
    return out


def arm_vol(TT, arm, setting, M, a, key="gt"):
    per = [per_volume(TT[(arm, s)], setting, M, a, key) for s in SEEDS]
    return np.array([np.mean([p[v] for p in per]) for v in sorted(per[0])])


def boot(d, lo=5, hi=95):
    b = rng.choice(d, size=(10000, len(d))).mean(1)
    return float(np.exp(d.mean())), float(np.exp(np.percentile(b, lo))), float(np.exp(np.percentile(b, hi)))


def holm(p):
    o = np.argsort(p); q = np.empty(len(p)); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (len(p) - k) * p[i])); q[i] = run
    return q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    TT = {(arm, s): read(arm, s) for arm in ARMS for s in SEEDS}
    if args.dry:
        for s in SEEDS:
            TT[("C", s)] = TT[("B", s)]
    V = {(arm, st, M, a): arm_vol(TT, arm, st, M, a) for arm in ARMS for st in ("phys", "matched") for M in RES for a in A}
    prim = {}
    for M in (128, 240):
        for a in (1, 9):
            d = V[("C", "phys", M, a)] - V[("B", "phys", M, a)]
            g, lo, hi = boot(d)
            p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
            prim[(M, a)] = dict(gmean=g, ci_lo=lo, ci_hi=hi, p=p, wins=int((d > 0).sum()), n=len(d))
    for k, q in zip(prim, holm([v["p"] for v in prim.values()])):
        v = prim[k]; v["p_holm"] = float(q)
        v["verdict"] = ("equivalent (±5%)" if v["ci_lo"] >= 0.95 and v["ci_hi"] <= 1.05 else
                        "C better" if q < 0.05 and v["gmean"] > 1 else "B better" if q < 0.05 else "inconclusive")
    if args.dry:
        assert all(abs(v["gmean"] - 1) < 1e-12 for v in prim.values()), "dry run: C/B must be 1"
        print("DRY RUN OK"); return
    gm = lambda x: float(np.exp(x.mean()))                                               # noqa: E731
    # estimator check: split-half n_eff vs ground-truth n_eff (arm B, phys)
    est = {(M, a): float(np.exp(np.mean([np.mean(arm_vol(TT, "B", "phys", M, a, "sh") - V[("B", "phys", M, a)])])))
           for M in RES for a in (1, 9)}
    L = [HEAD, "", "Pre-registered (`Archive/RES/PREREGISTRATION.md`). fastMRI AXT2, 15 test volumes × 8 slices; "
         "physical noise σ(M) = " + ("" if Dd.S_K == 1.0 else f"{Dd.S_K:g} × ") + "M/320 per channel; n_eff = σ²/MSE vs ground truth (head mask), r chosen without ground "
         "truth (leave-one-volume-out); seeds 101–103. Training matrices: A {160}; B–E {80, 160, 320}.", "",
         "**PRIMARY (n_eff C / B, physical noise, held-out matrices; Holm over 4):**", ""]
    for (M, a), v in prim.items():
        L.append(f"- M = {M}, a = {a}: {v['gmean']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; C better on "
                 f"{v['wins']}/{v['n']} volumes), p = {v['p']:.2g}, Holm {v['p_holm']:.2g} → **{v['verdict']}**")
    for st in ("phys", "matched"):
        for a in (1, 9):
            L += ["", f"n_eff ({'physical' if st == 'phys' else f'matched σ = {0.5 * Dd.S_K:g}'} noise, a = {a}; geometric mean over volumes):", "",
                  "| arm | " + " | ".join(f"M = {M}" for M in RES) + " |", "|---|" + "---:|" * len(RES)]
            for arm in ARMS:
                L.append(f"| {arm} | " + " | ".join("%.2f" % gm(V[(arm, st, M, a)]) for M in RES) + " |")
            L += ["", "| ratio | " + " | ".join(f"M = {M}" for M in RES) + " |", "|---|" + "---|" * len(RES)]
            for arm in ("A", "C", "D", "E"):
                cells = []
                for M in RES:
                    g, lo, hi = boot(V[(arm, st, M, a)] - V[("B", st, M, a)])
                    cells.append(f"{g:.3f} ({lo:.2f}–{hi:.2f})")
                L.append(f"| {arm} / B | " + " | ".join(cells) + " |")
    L += ["", "Estimator check (arm B, physical noise): split-half n_eff / ground-truth n_eff, geometric mean over volumes: " +
          ", ".join(f"M {M} a {a}: {v:.3f}" for (M, a), v in est.items()), ""]
    print("\n".join(L))
    os.makedirs(os.path.join(REPO, "runs", Dd.RUN_DIR, "report"), exist_ok=True)
    json.dump({"primary": {f"{M}|{a}": v for (M, a), v in prim.items()}, "estimator": {f"{M}|{a}": v for (M, a), v in est.items()}},
              open(os.path.join(REPO, "runs", Dd.RUN_DIR, "report", "summary.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
