# -*- coding: utf-8 -*-
"""GAP J report (Archive/GAP/PREREGISTRATION_J.md): fastMRI-initialized (FT) vs random-init (SC) ASL models in unit-noise
units (runs/gap/j/{FT,SC}_<arm>_s<s>), and both vs the original ASL models (runs/subj, runs/coil/subj), 1104c.
``--dry`` sets FT := SC := reference (ratios 1). Appends once unless --dry.

    python experiments/_ft_report.py [--dry]
"""
import os, json, argparse
import numpy as np
from scipy.stats import wilcoxon
import _subj_report as S
import _crop_report as C

REPO = S.REPO
REF = C.REF
HEAD = "## RUN 2026-11-07 — GAP J: does a fastMRI prior, fine-tuned on ASL, raise ASL performance?"
rng = np.random.default_rng(0)


def stat(dl):
    b = rng.choice(dl, size=(10000, len(dl))).mean(1)
    p = 1.0 if np.allclose(dl, 0) else float(wilcoxon(dl).pvalue)
    return dict(estimate=float(np.exp(dl.mean())), ci_lo=float(np.exp(np.percentile(b, 5))), ci_hi=float(np.exp(np.percentile(b, 95))),
                p=p, wins=int((dl > 0).sum()), n=len(dl))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    S.ARMS = {f"{k}{t}": (101, 102) for k in REF for t in ("", "f", "c")}
    TT = {}
    for k in REF:
        for s in (101, 102):
            TT[(k, s)] = C.read(REF[k].format(s=s))
            for t, lab in (("f", "FT"), ("c", "SC")):
                TT[(k + t, s)] = TT[(k, s)] if args.dry else C.read(f"runs/gap/j/{lab}_{k}_s{s}")
    L_ = lambda arm, a: S.design(TT, arm, 6, a)[1]
    ab = lambda x, y, a: stat(L_(x, a) - L_(y, a))
    prim = {"J1 n_eff(SUBM, FT) / n_eff(SUBM, SC), a = 9": ab("SUBMf", "SUBMc", 9),
            "J2 n_eff(SUB, FT) / n_eff(SUB, SC), a = 9": ab("SUBf", "SUBc", 9)}
    ps = sorted(prim, key=lambda q: prim[q]["p"])
    for j, q in enumerate(ps):
        prim[q]["p_holm"] = min(1.0, max(prim[x]["p"] * (len(ps) - i) for i, x in enumerate(ps[:j + 1])))
    if args.dry:
        assert all(abs(v["estimate"] - 1) < 1e-12 for v in prim.values()), "dry run"
        print("DRY RUN OK"); return
    f = lambda v: f"{v['estimate']:.3f} ({v['ci_lo']:.2f}–{v['ci_hi']:.2f}; {v['wins']}/{v['n']})"
    L = [HEAD, "", "Pre-registered (`Archive/GAP/PREREGISTRATION_J.md`). ASL in unit-noise units (each slice / σ̂; ladder = GAP D's), "
         "ASL-SUBJ / ASL-COIL recipe (dataset 1 + 1104a, seeds 101–102, 3000 iterations, MSE, lr 1e-3). FT = initialized from the "
         "GAP D DT checkpoint (it 3000, same arm and seed); SC = random init. Tested on 1104c; split-half n_eff, signal region; "
         "checkpoint and r cross-fitted over slice folds. ref = the original-units ASL models.", "", "**PRIMARY (Holm over 2):**", ""]
    for q, v in prim.items():
        L.append(f"- {q}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; higher on {v['wins']}/{v['n']} slices), "
                 f"p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", "| a | init | SUB/PER | SUBM/PER | n_eff PER / SUB / SUBM |", "|---:|---|---|---|---|"]
    for a in S.A:
        for t, lab in (("", "ref"), ("c", "SC"), ("f", "FT")):
            ne = " / ".join(f"{np.exp(L_(m + t, a).mean()):.1f}" for m in ("PER", "SUB", "SUBM"))
            L.append(f"| {a} | {lab} | {f(S.ratio(TT, 'SUB' + t, 'PER' + t, 6, a))} | {f(S.ratio(TT, 'SUBM' + t, 'PER' + t, 6, a))} | {ne} |")
    L += ["", "Absolute n_eff at a = 9 (90% CI; slices higher):", ""]
    for m in ("PER", "SUB", "SUBM"):
        L.append(f"- {m}: FT/SC {f(ab(m + 'f', m + 'c', 9))}; FT/ref {f(ab(m + 'f', m, 9))}; SC/ref {f(ab(m + 'c', m, 9))}")
    print("\n".join(L))
    os.makedirs(os.path.join(REPO, "runs", "gap", "report"), exist_ok=True)
    json.dump({"primary": prim}, open(os.path.join(REPO, "runs", "gap", "report", "summary_j.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as fh:
            fh.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
