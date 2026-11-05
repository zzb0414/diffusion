# -*- coding: utf-8 -*-
"""GAP A1/A2 report (Archive/GAP/PREREGISTRATION.md): does a re-scaled noise label close the ASL gap?

Reads runs/gap/scale/<model>/eval_it<iter>/neff.csv (``_scale_eval.py``; unseen subject 1104c). Selection without ground
truth as ``_subj_report``: per (model seed, a) and slice fold k, the (iter, c, r) with the lowest mean 1/n_eff over the
other folds. Three label policies: "c = 1" (the current decoder; reproduces ASL-SUBJ / ASL-COIL), "tuned c" (c from
SCALES chosen with iter and r), "brain" (c = the slice's in-signal / whole-image noise std ratio, no tuning). Text only;
appends to WEEKEND_ITERATIONS.md once unless --dry (--dry: every model := PER s101, every ratio 1).

    python experiments/_gap_report.py [--dry]
"""
import os, csv, json, argparse
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(REPO, "runs", "gap", "scale")
MODELS = {"PER": ("PER_s101", "PER_s102"), "SUB": ("SUB_s101", "SUB_s102"), "SUBM": ("SUBM_s101", "SUBM_s102")}
ITERS, A, D = (1000, 2000, 3000), (1, 2, 3, 5, 7, 9), 6
SCALES = ("0.5", "0.65", "0.8", "1", "1.2", "1.4", "1.7", "2")
HEAD = "## RUN 2026-11-05 — GAP A1/A2: does a re-scaled noise label close ASL's small denoise-the-mean gain?"
rng = np.random.default_rng(0)


def read(name):
    """{(it, a): {slice: (fold, {(c, r): neff})}} (signal region)."""
    T = defaultdict(dict)
    for it in ITERS:
        for r in csv.DictReader(open(os.path.join(RUNS, name, f"eval_it{it}", "neff.csv"), encoding="utf-8")):
            if r["region"] != "signal" or int(r["dataset"]) != D:
                continue
            nv = {}
            for k, v in r.items():
                if k.startswith("neff_c"):
                    c, rr = k[6:].split("_r")
                    nv[(c, float(rr))] = float(v)
            T[(it, int(r["a"]))][int(r["slice"])] = (int(r["fold"]), nv)
    return T


def chosen(T, a, cs):
    """{slice: (log neff, c)} at the (iter, c, r) chosen on the other folds, c restricted to cs."""
    cands = [(it, c, r) for it in ITERS for (c, r) in next(iter(T[(it, a)].values()))[1] if c in cs]
    out = {}
    for k in range(5):
        def cost(q):
            return np.mean([1 / v[1][(q[1], q[2])] for v in T[(q[0], a)].values() if v[0] != k])
        it, c, r = min(cands, key=cost)
        for s, (f, nv) in T[(it, a)].items():
            if f == k:
                out[s] = (float(np.log(nv[(c, r)])), c)
    return out


POLICY = {"c = 1": ("1",), "tuned c": SCALES, "brain": ("brain",)}


def design(TT, m, a, pol):
    per = [chosen(TT[n], a, POLICY[pol]) for n in MODELS[m]]
    sl = sorted(per[0])
    return sl, np.array([np.mean([p[s][0] for p in per]) for s in sl]), [p[sl[0]][1] for p in per]


def ratio(TT, x, px, y, py, a):
    sx, lx, _ = design(TT, x, a, px); sy, ly, _ = design(TT, y, a, py)
    assert sx == sy
    dl = lx - ly
    b = rng.choice(dl, size=(10000, len(dl))).mean(1)
    p = 1.0 if np.allclose(dl, 0) else float(wilcoxon(dl).pvalue)
    return dict(estimate=float(np.exp(dl.mean())), ci_lo=float(np.exp(np.percentile(b, 5))),
                ci_hi=float(np.exp(np.percentile(b, 95))), p=p, wins=int((dl > 0).sum()), n=len(dl))


def fmt(v):
    return f"{v['estimate']:.3f} ({v['ci_lo']:.2f}–{v['ci_hi']:.2f}; {v['wins']}/{v['n']})"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    names = {n for v in MODELS.values() for n in v}
    TT = {n: read("PER_s101" if args.dry else n) for n in names}
    prim = {"P1 SUB tuned c / SUB c = 1, a = 9": ratio(TT, "SUB", "tuned c", "SUB", "c = 1", 9),
            "P2 (SUB / PER) both tuned c, a = 9": ratio(TT, "SUB", "tuned c", "PER", "tuned c", 9)}
    ps = sorted(prim, key=lambda k: prim[k]["p"])
    for j, k in enumerate(ps):
        prim[k]["p_holm"] = min(1.0, max(prim[q]["p"] * (len(ps) - i) for i, q in enumerate(ps[:j + 1])))
    if args.dry:
        assert abs(prim["P2 (SUB / PER) both tuned c, a = 9"]["estimate"] - 1) < 1e-12, "dry run"
        print("DRY RUN OK"); return
    L = [HEAD, "", "Pre-registered (`Archive/GAP/PREREGISTRATION.md`). Evaluation only: the ASL-SUBJ / ASL-COIL models "
         "(trained on dataset 1 + 1104a; PER, SUB, SUBM, seeds 101–102) decode the unseen subject 1104c with the noise "
         "label scaled by c (PER: c·σ̂; SUB/SUBM: c·σ̂/√a), c ∈ {0.5, 0.65, 0.8, 1, 1.2, 1.4, 1.7, 2} chosen without ground truth "
         "with checkpoint and r (cross-fitted over slice folds), or c = the slice's in-signal / whole-image noise std "
         "ratio (\"brain\", untuned).", "", "**PRIMARY (Holm over 2):**", ""]
    for k, v in prim.items():
        L.append(f"- {k}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; higher on {v['wins']}/{v['n']} "
                 f"slices), p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", "Ratios (90% CI; slices higher):", "",
          "| a | SUB/PER, c = 1 | SUB/PER, tuned | SUB/PER, brain | SUBM/PER, tuned | SUB tuned / c = 1 | PER tuned / c = 1 | SUB brain / c = 1 | PER brain / c = 1 |",
          "|---:|---|---|---|---|---|---|---|---|"]
    for a in A:
        L.append(f"| {a} | " + " | ".join(fmt(ratio(TT, *q, a)) for q in (
            ("SUB", "c = 1", "PER", "c = 1"), ("SUB", "tuned c", "PER", "tuned c"), ("SUB", "brain", "PER", "brain"),
            ("SUBM", "tuned c", "PER", "tuned c"), ("SUB", "tuned c", "SUB", "c = 1"), ("PER", "tuned c", "PER", "c = 1"),
            ("SUB", "brain", "SUB", "c = 1"), ("PER", "brain", "PER", "c = 1"))) + " |")
    L += ["", "Absolute n_eff (geometric mean) and chosen c (per seed, fold 0's choice):", "",
          "| a | " + " | ".join(f"{m} {p}" for m in MODELS for p in POLICY) + " |", "|---:|" + "---|" * 9]
    for a in A:
        cells = []
        for m in MODELS:
            for p in POLICY:
                _, lv, cs = design(TT, m, a, p)
                cells.append(f"{np.exp(lv.mean()):.1f} ({', '.join(cs)})")
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    cb = [float(r["c_brain"]) for r in csv.DictReader(open(os.path.join(RUNS, "PER_s101", "eval_it3000", "neff.csv"),
                                                              encoding="utf-8")) if r["a"] == "1" and r["region"] == "signal"]
    L += ["", f"Untuned in-signal / whole-image noise std ratio on 1104c: median {np.median(cb):.3f} "
          f"(range {min(cb):.3f}–{max(cb):.3f})."]
    print("\n".join(L))
    os.makedirs(os.path.join(REPO, "runs", "gap", "report"), exist_ok=True)
    json.dump({"primary": prim}, open(os.path.join(REPO, "runs", "gap", "report", "summary_scale.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
