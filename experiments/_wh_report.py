# -*- coding: utf-8 -*-
"""GAP K report (Archive/GAP/PREREGISTRATION_K.md): spatially pre-whitened ASL models (runs/gap/k/<arm>_s<s>), scored in
original units, vs the original ASL models (runs/subj, runs/coil/subj), 1104c. Absolute n_eff primaries.
``--dry`` sets the whitened models := reference (ratios 1). Appends once unless --dry.

    python experiments/_wh_report.py [--dry]
"""
import os, json, argparse
import numpy as np
from scipy.stats import wilcoxon
import _subj_report as S
import _crop_report as C

REPO = S.REPO
REF = C.REF
HEAD = "## RUN 2026-11-07 — GAP K: does spatial pre-whitening (a noise map) raise ASL performance?"
WIENER = 1.149
rng = np.random.default_rng(0)


def stat(dl):
    b = rng.choice(dl, size=(10000, len(dl))).mean(1)
    p = 1.0 if np.allclose(dl, 0) else float(wilcoxon(dl).pvalue)
    return dict(estimate=float(np.exp(dl.mean())), ci_lo=float(np.exp(np.percentile(b, 5))), ci_hi=float(np.exp(np.percentile(b, 95))),
                p=p, wins=int((dl > 0).sum()), n=len(dl))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    S.ARMS = {f"{k}{t}": (101, 102) for k in REF for t in ("", "w")}
    TT = {}
    for k in REF:
        for s in (101, 102):
            TT[(k, s)] = C.read(REF[k].format(s=s))
            TT[(k + "w", s)] = TT[(k, s)] if args.dry else C.read(f"runs/gap/k/{k}_s{s}")
    L_ = lambda arm, a: S.design(TT, arm, 6, a)[1]
    ab = lambda x, y, a: stat(L_(x, a) - L_(y, a))
    prim = {"K1 n_eff(SUBM, whitened) / n_eff(SUBM, real), a = 9": ab("SUBMw", "SUBM", 9),
            "K2 n_eff(SUB, whitened) / n_eff(SUB, real), a = 9": ab("SUBw", "SUB", 9)}
    ps = sorted(prim, key=lambda q: prim[q]["p"])
    for j, q in enumerate(ps):
        prim[q]["p_holm"] = min(1.0, max(prim[x]["p"] * (len(ps) - i) for i, x in enumerate(ps[:j + 1])))
    if args.dry:
        assert all(abs(v["estimate"] - 1) < 1e-12 for v in prim.values()), "dry run"
        print("DRY RUN OK"); return
    f = lambda v: f"{v['estimate']:.3f} ({v['ci_lo']:.2f}–{v['ci_hi']:.2f}; {v['wins']}/{v['n']})"
    L = [HEAD, "", "Pre-registered (`Archive/GAP/PREREGISTRATION_K.md`). PER, SUB, SUBM trained exactly as ASL-SUBJ / ASL-COIL "
         "(dataset 1 + 1104a, seeds 101–102, 3000 iterations, MSE) on spatially pre-whitened ASL (each slice / its own smooth noise-std "
         "map, Gaussian 4 px, mean-square 1); scored on ORIGINAL-units 1104c (input / map, output × map). Split-half n_eff, signal "
         "region; checkpoint and r cross-fitted over slice folds.", "", "**PRIMARY (Holm over 2):**", ""]
    for q, v in prim.items():
        L.append(f"- {q}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; higher on {v['wins']}/{v['n']} slices), "
                 f"p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", "| a | model | SUB/PER | SUBM/PER | n_eff PER / SUB / SUBM |", "|---:|---|---|---|---|"]
    for a in S.A:
        for t, lab in (("", "real"), ("w", "whitened")):
            ne = " / ".join(f"{np.exp(L_(m + t, a).mean()):.1f}" for m in ("PER", "SUB", "SUBM"))
            L.append(f"| {a} | {lab} | {f(S.ratio(TT, 'SUB' + t, 'PER' + t, 6, a))} | {f(S.ratio(TT, 'SUBM' + t, 'PER' + t, 6, a))} | {ne} |")
    L += ["", "Absolute n_eff, whitened / real (90% CI; slices higher):", ""]
    for m in ("PER", "SUB", "SUBM"):
        L.append(f"- {m}: a = 1 {f(ab(m + 'w', m, 1))}; a = 9 {f(ab(m + 'w', m, 9))}")
    g = {m: S.ratio(TT, m + "w", "PERw", 6, 9)["estimate"] for m in ("SUB", "SUBM")}
    L += ["", f"Learned / Wiener at a = 9 (Wiener {WIENER}, real-data spectra): whitened SUB {g['SUB'] / WIENER:.3f}, SUBM {g['SUBM'] / WIENER:.3f}."]
    print("\n".join(L))
    os.makedirs(os.path.join(REPO, "runs", "gap", "report"), exist_ok=True)
    json.dump({"primary": prim}, open(os.path.join(REPO, "runs", "gap", "report", "summary_k.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as fh:
            fh.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
