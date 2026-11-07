# -*- coding: utf-8 -*-
"""GAP H report (Archive/GAP/PREREGISTRATION_H.md): SUB/PER and SUBM/PER of ASL models trained on --train-frac 0.25 / 0.5
(runs/gap/h/<arm>_f<fr>_s<s>) vs frac 1 (runs/subj/{PER,SUB}_s<s>, runs/coil/subj/SUBM_s<s>), 1104c, selection as _subj_report.
``--dry`` sets the subsampled models := frac 1 (ratios 1). Appends once unless --dry.

    python experiments/_h_report.py [--dry]
"""
import os, json, argparse
import numpy as np
from scipy.stats import wilcoxon
import _subj_report as S
import _crop_report as C

REPO = S.REPO
REF = C.REF
FR = {"q": "0.25", "h": "0.5"}
HEAD = "## RUN 2026-11-07 — GAP H: is the ASL denoise-the-mean gain data-limited? (training-set dose-response)"
rng = np.random.default_rng(0)


def stat(dl):
    b = rng.choice(dl, size=(10000, len(dl))).mean(1)
    p = 1.0 if np.allclose(dl, 0) else float(wilcoxon(dl).pvalue)
    return dict(estimate=float(np.exp(dl.mean())), ci_lo=float(np.exp(np.percentile(b, 5))), ci_hi=float(np.exp(np.percentile(b, 95))),
                p=p, wins=int((dl > 0).sum()), n=len(dl))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    S.ARMS = {f"{k}{t}": (101, 102) for k in REF for t in ("", *FR)}
    TT = {}
    for k in REF:
        for s in (101, 102):
            TT[(k, s)] = C.read(REF[k].format(s=s))
            for t, fr in FR.items():
                TT[(k + t, s)] = TT[(k, s)] if args.dry else C.read(f"runs/gap/h/{k}_f{fr}_s{s}")
    L_ = lambda arm, a: S.design(TT, arm, 6, a)[1]
    rr = lambda m, t, a: stat((L_(m + t, a) - L_("PER" + t, a)) - (L_(m, a) - L_("PER", a)))
    ab = lambda m, t, a: stat(L_(m + t, a) - L_(m, a))
    prim = {"H1 (SUB/PER, frac 0.25) / (SUB/PER, frac 1), a = 9": rr("SUB", "q", 9),
            "H2 (SUBM/PER, frac 0.25) / (SUBM/PER, frac 1), a = 9": rr("SUBM", "q", 9)}
    ps = sorted(prim, key=lambda q: prim[q]["p"])
    for j, q in enumerate(ps):
        prim[q]["p_holm"] = min(1.0, max(prim[x]["p"] * (len(ps) - i) for i, x in enumerate(ps[:j + 1])))
    if args.dry:
        assert all(abs(v["estimate"] - 1) < 1e-12 for v in prim.values()), "dry run"
        print("DRY RUN OK"); return
    f = lambda v: f"{v['estimate']:.3f} ({v['ci_lo']:.2f}–{v['ci_hi']:.2f}; {v['wins']}/{v['n']})"
    L = [HEAD, "", "Pre-registered (`Archive/GAP/PREREGISTRATION_H.md`). PER, SUB, SUBM trained exactly as ASL-SUBJ / ASL-COIL "
         "(dataset 1 + 1104a, seeds 101–102, 3000 iterations, MSE) on a nested fixed subset of slice indices: frac 0.25 = 8 indices "
         "(16 images), frac 0.5 = 17 (34 images), frac 1 = 34 (68 images, the references). Tested on 1104c; split-half n_eff, signal "
         "region; checkpoint and r cross-fitted over slice folds.", "", "**PRIMARY (Holm over 2):**", ""]
    for q, v in prim.items():
        L.append(f"- {q}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; higher on {v['wins']}/{v['n']} slices), "
                 f"p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", "| a | frac | SUB/PER | SUBM/PER | n_eff PER / SUB / SUBM |", "|---:|---|---|---|---|"]
    for a in S.A:
        for t, lab in (("", "1"), ("h", "0.5"), ("q", "0.25")):
            ne = " / ".join(f"{np.exp(L_(m + t, a).mean()):.1f}" for m in ("PER", "SUB", "SUBM"))
            L.append(f"| {a} | {lab} | {f(S.ratio(TT, 'SUB' + t, 'PER' + t, 6, a))} | {f(S.ratio(TT, 'SUBM' + t, 'PER' + t, 6, a))} | {ne} |")
    L += ["", "Absolute n_eff at a = 9, subsampled / frac 1 (90% CI; slices higher):", ""]
    for m in ("PER", "SUB", "SUBM"):
        L.append(f"- {m}: frac 0.5 {f(ab(m, 'h', 9))}; frac 0.25 {f(ab(m, 'q', 9))}")
    L += ["", f"Secondary ratios at frac 0.5, a = 9: SUB {f(rr('SUB', 'h', 9))}; SUBM {f(rr('SUBM', 'h', 9))}."]
    print("\n".join(L))
    os.makedirs(os.path.join(REPO, "runs", "gap", "report"), exist_ok=True)
    json.dump({"primary": prim}, open(os.path.join(REPO, "runs", "gap", "report", "summary_h.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as fh:
            fh.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
