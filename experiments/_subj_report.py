# -*- coding: utf-8 -*-
"""ASL-SUBJ report (Archive/SUBJ/PREREGISTRATION.md): denoise the mean (SUB) vs denoise each, then average (PER, PER-D)
on an UNSEEN SUBJECT. Trained on every slice of dataset 1 + scan 1104a (data id 5; same subject), tested on every slice
of scan 1104c (data id 6; another subject).

Reads runs/subj/<arm>_s<seed>/eval_it<iter>/neff.csv (``_gr_eval.py --single``; split-half n_eff, signal region). Per
(arm, seed, a) and per slice fold k (CV_R3 slice-index folds, used only for cross-fitting the choice), the checkpoint and
the dial r are chosen jointly on the test subject's OTHER folds (lowest mean 1/n_eff, i.e. split-half MSE; no ground
truth), then applied to fold k. Per slice: mean of
log n_eff over the arm's seeds. Ratios: geometric mean over slices of the per-slice ratio, Wilcoxon over slices, 90%
bootstrap CI. ``--dry`` sets SUB := PER (P1 = 1). Appends its log section to WEEKEND_ITERATIONS.md once unless --dry.

    python experiments/_subj_report.py [--dry]
"""
import os, csv, json, argparse
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(REPO, "runs", "subj")
ARMS = {"PER": (101, 102), "SUB": (101, 102), "PERD": (101,)}
ITERS, A, DS = (1000, 2000, 3000), (1, 2, 3, 5, 7, 9), (6,)
HEAD = "## RUN 2026-11-04 — ASL-SUBJ: denoise the mean vs denoise each, then average, on an unseen subject"
rng = np.random.default_rng(0)


def read(arm, seed):
    """{(it, d, a): {slice: (fold, {r: neff})}} (signal region)."""
    T = defaultdict(dict)
    for it in ITERS:
        for r in csv.DictReader(open(os.path.join(RUNS, f"{arm}_s{seed}", f"eval_it{it}", "neff.csv"), encoding="utf-8")):
            if r["region"] != "signal":
                continue
            T[(it, int(r["dataset"]), int(r["a"]))][int(r["slice"])] = (
                int(r["fold"]), {float(k[6:]): float(v) for k, v in r.items() if k.startswith("neff_r")})
    return T


def chosen(T, d, a):
    """{slice: (log neff, r)} at the (iter, r) chosen on the other folds."""
    cands = [(it, r) for it in ITERS for r in next(iter(T[(it, d, a)].values()))[1]]
    out = {}
    for k in range(5):
        def cost(c):
            return np.mean([1 / v[1][c[1]] for v in T[(c[0], d, a)].values() if v[0] != k])
        it, r = min(cands, key=cost)
        for s, (f, nv) in T[(it, d, a)].items():
            if f == k:
                out[s] = (float(np.log(nv[r])), r)
    return out


def design(TT, arm, d, a):
    per = [chosen(TT[(arm, s)], d, a) for s in ARMS[arm]]
    sl = sorted(per[0])
    return sl, np.array([np.mean([p[s][0] for p in per]) for s in sl]), [p[sl[0]][1] for p in per]


def ratio(TT, x, y, d, a):
    sx, lx, _ = design(TT, x, d, a); sy, ly, _ = design(TT, y, d, a)
    assert sx == sy
    dl = lx - ly
    b = rng.choice(dl, size=(10000, len(dl))).mean(1)
    p = 1.0 if np.allclose(dl, 0) else float(wilcoxon(dl).pvalue)
    return dict(estimate=float(np.exp(dl.mean())), ci_lo=float(np.exp(np.percentile(b, 5))),
                ci_hi=float(np.exp(np.percentile(b, 95))), p=p, wins=int((dl > 0).sum()), n=len(dl))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    TT = {(arm, s): read(arm, s) for arm, seeds in ARMS.items() for s in seeds}
    if args.dry:
        for s in ARMS["SUB"]:
            TT[("SUB", s)] = TT[("PER", s)]
    prim = {"P1 SUB / PER, unseen subject (1104c), a = 9": ratio(TT, "SUB", "PER", 6, 9),
            "P2 SUB / PER-D, unseen subject (1104c), a = 9": ratio(TT, "SUB", "PERD", 6, 9)}
    ps = sorted(prim, key=lambda k: prim[k]["p"])
    for j, k in enumerate(ps):
        prim[k]["p_holm"] = min(1.0, max(prim[q]["p"] * (len(ps) - i) for i, q in enumerate(ps[:j + 1])))
    if args.dry:
        assert abs(prim["P1 SUB / PER, unseen subject (1104c), a = 9"]["estimate"] - 1) < 1e-12, "dry run"
        print("DRY RUN OK"); return
    L = [HEAD, "", "Pre-registered (`Archive/SUBJ/PREREGISTRATION.md`). Trained on every slice of dataset 1 + scan 1104a "
         "(same subject, 2026-09-24 and 2026-11-04, good coil); tested on every slice of scan 1104c (another subject, same "
         "protocol and coil). The ASL-SM2 recipe (MSE, p_raw 0.5, no augmentation, 3000 iterations; SUB ladder floor "
         "σ̂_min/3, m ~ U{1..9}; PER-D noise-normalized, seed 101 only). Split-half n_eff on the test subject (signal "
         "region, no ground truth); checkpoint and r cross-fitted over the test subject's slice folds.", "",
         "**PRIMARY (Holm over 2):**", ""]
    for k, v in prim.items():
        L.append(f"- {k}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; SUB better on {v['wins']}/{v['n']} "
                 f"slices), p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", "SUB / PER and SUB / PER-D (90% CI):", "", "| dataset | a | SUB / PER | SUB / PER-D |", "|---:|---:|---|---|"]
    for d in DS:
        for a in A:
            q1, q2 = ratio(TT, "SUB", "PER", d, a), ratio(TT, "SUB", "PERD", d, a)
            L.append(f"| {d} | {a} | {q1['estimate']:.3f} ({q1['ci_lo']:.2f}–{q1['ci_hi']:.2f}; {q1['wins']}/{q1['n']}) | "
                     f"{q2['estimate']:.3f} ({q2['ci_lo']:.2f}–{q2['ci_hi']:.2f}; {q2['wins']}/{q2['n']}) |")
    L += ["", "Absolute n_eff (geometric mean over slices) and chosen r (per seed, fold 0's choice):", "",
          "| dataset | a | PER | SUB | PER-D |", "|---:|---:|---|---|---|"]
    for d in DS:
        for a in A:
            cells = []
            for arm in ("PER", "SUB", "PERD"):
                _, lv, rs = design(TT, arm, d, a)
                cells.append(f"{np.exp(lv.mean()):.2f} (r {', '.join('%.3g' % r for r in rs)})")
            L.append(f"| {d} | {a} | " + " | ".join(cells) + " |")
    print("\n".join(L))
    os.makedirs(os.path.join(RUNS, "report"), exist_ok=True)
    json.dump({"primary": prim}, open(os.path.join(RUNS, "report", "summary.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
