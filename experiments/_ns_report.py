# -*- coding: utf-8 -*-
"""GAP I report (Archive/GAP/PREREGISTRATION_I.md): SUB/PER and SUBM/PER on noise-stationarized ASL (runs/gap/i/<arm>_s<s>)
vs real ASL (runs/subj/{PER,SUB}_s<s>, runs/coil/subj/SUBM_s<s>), 1104c, selection as _subj_report.
``--dry`` sets the noise-stationarized models := real (ratios 1). Appends once unless --dry.

    python experiments/_pr_report.py [--dry]
"""
import os, json, argparse
import numpy as np
from scipy.stats import wilcoxon
import _subj_report as S
import _crop_report as C

REPO = S.REPO
REF = C.REF
PR = {k: "runs/gap/i/" + k + "_s{s}" for k in REF}
HEAD = "## RUN 2026-11-07 — GAP I: noise-stationarized ASL — does the noise non-stationarity cost denoise-the-mean?"
WIENER = 1.149                                   # ASL 1104c, a = 9 (runs/gap/report/wiener.txt); unchanged by the transform
rng = np.random.default_rng(0)


def stat(dl):
    b = rng.choice(dl, size=(10000, len(dl))).mean(1)
    p = 1.0 if np.allclose(dl, 0) else float(wilcoxon(dl).pvalue)
    return dict(estimate=float(np.exp(dl.mean())), ci_lo=float(np.exp(np.percentile(b, 5))), ci_hi=float(np.exp(np.percentile(b, 95))),
                p=p, wins=int((dl > 0).sum()), n=len(dl))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); args = ap.parse_args()
    S.ARMS = {f"{k}{t}": (101, 102) for k in REF for t in ("", "p")}
    TT = {}
    for k in REF:
        for s in (101, 102):
            TT[(k, s)] = C.read(REF[k].format(s=s))
            TT[(k + "p", s)] = TT[(k, s)] if args.dry else C.read(PR[k].format(s=s))
    L_ = lambda arm, a: S.design(TT, arm, 6, a)[1]
    rr = lambda m, a: stat((L_(m + "p", a) - L_("PERp", a)) - (L_(m, a) - L_("PER", a)))
    prim = {"R1 (SUB/PER, noise-stationarized) / (SUB/PER, real), a = 9": rr("SUB", 9),
            "R2 (SUBM/PER, noise-stationarized) / (SUBM/PER, real), a = 9": rr("SUBM", 9)}
    ps = sorted(prim, key=lambda q: prim[q]["p"])
    for j, q in enumerate(ps):
        prim[q]["p_holm"] = min(1.0, max(prim[x]["p"] * (len(ps) - i) for i, x in enumerate(ps[:j + 1])))
    if args.dry:
        assert all(abs(v["estimate"] - 1) < 1e-12 for v in prim.values()), "dry run"
        print("DRY RUN OK"); return
    f = lambda v: f"{v['estimate']:.3f} ({v['ci_lo']:.2f}–{v['ci_hi']:.2f}; {v['wins']}/{v['n']})"
    L = [HEAD, "", "Pre-registered (`Archive/GAP/PREREGISTRATION_I.md`). PER, SUB, SUBM trained exactly as ASL-SUBJ / ASL-COIL "
         "(dataset 1 + 1104a, seeds 101–102, 3000 iterations, MSE) on noise-stationarized ASL (real repetition mean kept; deviations multiplied by one random k-space phase field per "
         "slice: signal structure, noise spectrum and σ̂ kept, noise non-stationarity destroyed); tested on noise-stationarized "
         "1104c. Split-half n_eff, signal region; checkpoint and r cross-fitted over slice folds.", "", "**PRIMARY (Holm over 2):**", ""]
    for q, v in prim.items():
        L.append(f"- {q}: {v['estimate']:.3f}× (90% CI {v['ci_lo']:.3f}–{v['ci_hi']:.3f}; higher on {v['wins']}/{v['n']} slices), "
                 f"p = {v['p']:.2g}, Holm {v['p_holm']:.2g}")
    L += ["", "| a | SUB/PER real | SUB/PER NS | SUBM/PER real | SUBM/PER NS | R SUB | R SUBM | n_eff real PER / SUB / SUBM | n_eff NS PER / SUB / SUBM |",
          "|---:|---|---|---|---|---|---|---|---|"]
    for a in S.A:
        ne = lambda t: " / ".join(f"{np.exp(L_(m + t, a).mean()):.1f}" for m in ("PER", "SUB", "SUBM"))
        L.append(f"| {a} | " + " | ".join(f(S.ratio(TT, x, y, 6, a)) for x, y in (("SUB", "PER"), ("SUBp", "PERp"), ("SUBM", "PER"), ("SUBMp", "PERp")))
                 + f" | {f(rr('SUB', a))} | {f(rr('SUBM', a))} | {ne('')} | {ne('p')} |")
    g = {m: S.ratio(TT, m + "p", "PERp", 6, 9)["estimate"] for m in ("SUB", "SUBM")}
    L += ["", f"Learned / Wiener at a = 9 (Wiener {WIENER}): noise-stationarized SUB {g['SUB'] / WIENER:.3f}, SUBM {g['SUBM'] / WIENER:.3f}; "
          f"real SUB {S.ratio(TT, 'SUB', 'PER', 6, 9)['estimate'] / WIENER:.3f}, SUBM {S.ratio(TT, 'SUBM', 'PER', 6, 9)['estimate'] / WIENER:.3f}."]
    print("\n".join(L))
    os.makedirs(os.path.join(REPO, "runs", "gap", "report"), exist_ok=True)
    json.dump({"primary": prim}, open(os.path.join(REPO, "runs", "gap", "report", "summary_i.json"), "w"), indent=2)
    log = os.path.join(REPO, "WEEKEND_ITERATIONS.md")
    if HEAD not in open(log, encoding="utf-8").read():
        with open(log, "a", encoding="utf-8") as fh:
            fh.write("\n---\n\n" + "\n".join(L) + "\n")
        print("logged")


if __name__ == "__main__":
    main()
