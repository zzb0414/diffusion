# -*- coding: utf-8 -*-
"""ASL-COIL report (Archive/COIL/PREREGISTRATION.md).

Part A: dataset-1-only training (coil1), 5-fold CV by slice index, seed 101: PER, SUB, SUBM (m drawn proportional to
m), scored on dataset 1; compared with ASL-SM2's pooled (coil1 + coil2) seed-101 models on the same slices and folds.
Part B: SUBM trained like ASL-SUBJ (all slices of dataset 1 + 1104a, seeds 101-102), scored on 1104c (data id 6) and
compared with ASL-SUBJ's PER and SUB.

Selection exactly as ``_sm2_report`` / ``_subj_report``: per (model, dataset, a) and held-out slice fold k, checkpoint
and r chosen on the other folds by lowest mean split-half 1/n_eff (no ground truth). Per slice: mean log n_eff over
seeds. Ratios: geometric mean over slices, Wilcoxon over slices, 90% bootstrap CI. ``--dry`` sets SUB := PER (both
trainings) and SUBM := SUB, so every primary is 1. Appends its log section to WEEKEND_ITERATIONS.md once unless --dry.

    python experiments/_coil_report.py [--dry]
"""
import os, csv, json, argparse
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = lambda *p: os.path.join(REPO, "runs", *p)
ITERS, A = (1000, 2000, 3000), (1, 2, 3, 5, 7, 9)
# model key -> (dataset scored, run dirs of its seeds)
MODELS = {
    "PER_d1": (1, [R("coil", "cv", "PER_s101")]),
    "SUB_d1": (1, [R("coil", "cv", "SUB_s101")]),
    "SUBM_d1": (1, [R("coil", "cv", "SUBM_s101")]),
    "PER_pool": (1, [R("sm2", "PER_s101")]),
    "SUB_pool": (1, [R("sm2", "SUB_s101")]),
    "PER_subj": (6, [R("subj", "PER_s101"), R("subj", "PER_s102")]),
    "SUB_subj": (6, [R("subj", "SUB_s101"), R("subj", "SUB_s102")]),
    "SUBM_subj": (6, [R("coil", "subj", "SUBM_s101"), R("coil", "subj", "SUBM_s102")]),
}
HEAD = "## RUN 2026-11-05 — ASL-COIL: does pooling coils hurt denoise-the-mean, and does the MEAN-TAX fix carry to ASL?"
rng = np.random.default_rng(0)


def read(run):
    """{(it, d, a): {slice: (fold, {r: neff})}} (signal region)."""
    T = defaultdict(dict)
    for it in ITERS:
        for r in csv.DictReader(open(os.path.join(run, f"eval_it{it}", "neff.csv"), encoding="utf-8")):
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


def design(TT, m, a):
    d = MODELS[m][0]
    per = [chosen(T, d, a) for T in TT[m]]
    sl = sorted(per[0])
    return sl, np.array([np.mean([p[s][0] for p in per]) for s in sl]), [p[sl[0]][1] for p in per]


def summarize(dl):
    b = rng.choice(dl, size=(10000, len(dl))).mean(1)
    p = 1.0 if np.allclose(dl, 0) else float(wilcoxon(dl).pvalue)
    return dict(estimate=float(np.exp(dl.mean())), ci_lo=float(np.exp(np.percentile(b, 5))),
                ci_hi=float(np.exp(np.percentile(b, 95))), p=p, wins=int((dl > 0).sum()), n=len(dl))


def diff(TT, x, y, a):
    sx, lx, _ = design(TT, x, a); sy, ly, _ = design(TT, y, a)
    assert sx == sy
    return lx - ly


def ratio(TT, x, y, a):
    return summarize(diff(TT, x, y, a))


def interaction(TT, a):
    """(SUB / PER trained on dataset 1 only) / (SUB / PER trained pooled), per slice of dataset 1."""
    return summarize(diff(TT, "SUB_d1", "PER_d1", a) - diff(TT, "SUB_pool", "PER_pool", a))


def fmt(v):
    return f"{v['estimate']:.3f} ({v['ci_lo']:.2f}–{v['ci_hi']:.2f}; {v['wins']}/{v['n']})"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    skip = {"SUB_d1", "SUBM_d1", "SUBM_subj"} if args.dry else set()
    TT = {m: [read(r) for r in runs] for m, (_, runs) in MODELS.items() if m not in skip}
    if args.dry:
        TT["SUB_pool"] = TT["PER_pool"]
        TT["SUB_d1"] = TT["SUBM_d1"] = TT["PER_d1"]
        TT["SUBM_subj"] = TT["SUB_subj"]
    prim = {"P1 pooling interaction (SUB/PER on dataset 1 only) / (SUB/PER pooled), dataset 1, a = 9": interaction(TT, 9),
            "P2 SUBM / SUB, unseen subject (1104c), a = 9": ratio(TT, "SUBM_subj", "SUB_subj", 9),
            "P3 SUBM / SUB, dataset-1-only CV, a = 9": ratio(TT, "SUBM_d1", "SUB_d1", 9)}
    ps = sorted(prim, key=lambda k: prim[k]["p"])
    for j, k in enumerate(ps):
        prim[k]["p_holm"] = min(1.0, max(prim[q]["p"] * (len(ps) - i) for i, q in enumerate(ps[:j + 1])))
    if args.dry:
        assert all(abs(v["estimate"] - 1) < 1e-12 for v in prim.values()), "dry run"
        print("DRY RUN OK"); return
    L = [HEAD, "", "Pre-registered (`Archive/COIL/PREREGISTRATION.md`). Part A: PER, SUB and SUBM (denoise the mean with "
         "m drawn ∝ m) trained on dataset 1 only (coil1), 5-fold CV by slice index, seed 101, compared on the same slices "
         "and folds with ASL-SM2's pooled (datasets 1–3, coil1 + coil2) seed-101 models. Part B: SUBM trained like "
         "ASL-SUBJ (dataset 1 + 1104a, seeds 101–102), scored on the unseen subject 1104c against ASL-SUBJ's PER and SUB. "
         "ASL-SM2 recipe otherwise (MSE, p_raw 0.5, no augmentation, 3000 iterations, SUB/SUBM ladder floor σ̂_min/3). "
         "Split-half n_eff (signal region, no ground truth); checkpoint and r cross-fitted over slice folds.", "",
         "**PRIMARY (Holm over 3):**", ""]
    for k, v in prim.items():
        L.append(f"- {k}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; higher on {v['wins']}/{v['n']} "
                 f"slices), p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", "Dataset 1 (Part A). Ratios (90% CI; slices higher):", "",
          "| a | SUB / PER (d1 only) | SUBM / PER (d1 only) | SUB / PER (pooled) | interaction | PER d1 / pooled | SUB d1 / pooled |",
          "|---:|---|---|---|---|---|---|"]
    for a in A:
        L.append(f"| {a} | " + " | ".join(fmt(v) for v in (
            ratio(TT, "SUB_d1", "PER_d1", a), ratio(TT, "SUBM_d1", "PER_d1", a), ratio(TT, "SUB_pool", "PER_pool", a),
            interaction(TT, a), ratio(TT, "PER_d1", "PER_pool", a), ratio(TT, "SUB_d1", "SUB_pool", a))) + " |")
    L += ["", "Unseen subject 1104c (Part B). Ratios (90% CI; slices higher):", "",
          "| a | SUBM / SUB | SUBM / PER | SUB / PER |", "|---:|---|---|---|"]
    for a in A:
        L.append(f"| {a} | " + " | ".join(fmt(ratio(TT, x, y, a)) for x, y in (
            ("SUBM_subj", "SUB_subj"), ("SUBM_subj", "PER_subj"), ("SUB_subj", "PER_subj"))) + " |")
    L += ["", "Absolute n_eff (geometric mean over slices) and chosen r (per seed, fold 0's choice):", "",
          "| a | " + " | ".join(MODELS) + " |", "|---:|" + "---|" * len(MODELS)]
    for a in A:
        cells = []
        for m in MODELS:
            _, lv, rs = design(TT, m, a)
            cells.append(f"{np.exp(lv.mean()):.2f} (r {', '.join('%.3g' % r for r in rs)})")
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    print("\n".join(L))
    os.makedirs(R("coil", "report"), exist_ok=True)
    json.dump({"primary": prim}, open(R("coil", "report", "summary.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
