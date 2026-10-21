# -*- coding: utf-8 -*-
"""CV_R3 report: log section + argument document, generated from the CSVs.

Every numeral comes from ``runs/cv_r3/eval/{endpoints,per_slice,bands}.csv``
and the fold configs; every qualitative claim is an assert. Also archives the
five fold checkpoints (copy + sha256) and the raw stdout.

Writes:
  WEEKEND_ITERATIONS.md               appended section (once)
  docs/ARGUMENT_N2N_vs_averaging.md   the argument, for readers outside the project

    python experiments/_cv_report.py
"""
import sys, os, csv, json, shutil, hashlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
os.chdir(ROOT)
EV = "runs/cv_r3/eval"
MD = "WEEKEND_ITERATIONS.md"
HEAD = "## RUN 2026-10-20 — CV_R3"
DOC = "docs/ARGUMENT_N2N_vs_averaging.md"
WRITE_LOG = HEAD not in open(MD, encoding="utf-8").read()   # log once; the doc is always rebuilt

E = list(csv.DictReader(open(os.path.join(EV, "endpoints.csv"), encoding="utf-8")))
P = list(csv.DictReader(open(os.path.join(EV, "per_slice.csv"), encoding="utf-8")))
B = list(csv.DictReader(open(os.path.join(EV, "bands.csv"), encoding="utf-8")))
for r in E:
    for k in ("a", "n", "wins"):
        r[k] = int(r[k])
    for k in ("gmean_ratio", "ci_lo", "ci_hi", "min_ratio", "wilcoxon_p"):
        r[k] = float(r[k])
for r in P:
    r["slice"], r["a"] = int(r["slice"]), int(r["a"])
    for k in ("sigma_hat", "plain", "r3", "blur_oracle", "blur_width"):
        r[k] = float(r[k])
pre = open("Archive/CV_R3/PREREGISTRATION.md", encoding="utf-8").read()
pre_h = hashlib.sha256(pre.encode("utf-8")).hexdigest()
A_SIZES = sorted({r["a"] for r in E})


def ep(a, comp, region="signal", excl="-"):
    return next(r for r in E if r["a"] == a and r["comparison"] == comp and r["region"] == region
                and r["excluded"] == excl)


def fmt(r):
    return "%.2f× (95%% CI %.2f–%.2f; %d/%d slices; p = %.1e)" % (
        r["gmean_ratio"], r["ci_lo"], r["ci_hi"], r["wins"], r["n"], r["wilcoxon_p"])


# ---- claims -------------------------------------------------------------------------
prim = ep(5, "r3/plain")
assert prim["ci_lo"] > 1 and prim["wilcoxon_p"] < 0.01 and prim["wins"] == prim["n"] == 34
all_plain = [ep(a, "r3/plain") for a in A_SIZES]
assert all(r["wins"] == r["n"] and r["ci_lo"] > 1 for r in all_plain)
assert all(x["gmean_ratio"] > y["gmean_ratio"] for x, y in zip(all_plain, all_plain[1:]))   # shrinks with a
vs_blur = {a: ep(a, "r3/blur_oracle") for a in A_SIZES}
sig_blur = [a for a in A_SIZES if vs_blur[a]["ci_lo"] > 1 and vs_blur[a]["wilcoxon_p"] < 0.01]
nonsig = [a for a in A_SIZES if a not in sig_blur]
assert sig_blur == [1, 2, 3, 5] and nonsig == [7, 9], (sig_blur, nonsig)
blur_plain5 = ep(5, "blur_oracle/plain")
share = np.log(blur_plain5["gmean_ratio"]) / np.log(prim["gmean_ratio"])
sens = ep(5, "r3/plain", excl="26,29")
assert sens["ci_lo"] > 1 and sens["wins"] == sens["n"]
allpix = ep(5, "r3/plain", region="all")
assert allpix["ci_lo"] > 1 and allpix["wins"] == allpix["n"]

worst = min((r for r in P if r["region"] == "signal" and r["a"] == 1),
            key=lambda r: r["r3"] / r["blur_oracle"])
qmin = min(json.load(open("runs/cv_r3/fold0/config.json", encoding="utf-8"))["sigma_hat"].values())
assert worst["slice"] == 29 and abs(worst["sigma_hat"] - qmin) < 1e-9     # the quietest slice
s29 = {a: next(r for r in P if r["slice"] == 29 and r["region"] == "signal" and r["a"] == a) for a in (1, 5, 9)}
second = sorted((r for r in P if r["region"] == "signal" and r["a"] == 1),
                key=lambda r: r["r3"] / r["blur_oracle"])[1]

band = []
for b in range(4):
    rr = np.array([float(x[f"r3_band{b}"]) / float(x[f"blur_band{b}"]) for x in B])
    band.append(dict(win=int(sum(float(x[f"r3_band{b}"]) >= float(x[f"blur_band{b}"]) for x in B)),
                     med=float(np.median(rr))))
assert band[0]["win"] == 34 and band[0]["med"] > 2
assert band[3]["med"] > 1 and band[2]["win"] < 34 / 2 + 1

# ---- archive ------------------------------------------------------------------------
cd = "checkpoints/CV_R3"
os.makedirs(cd, exist_ok=True)
man = ["# CV_R3 checkpoint manifest", "",
       "Fold models of the cross-validated R3 recipe (iteration 2000, fixed in advance). Each slice is",
       "scored only by the fold that held it out.", "", "| fold | held-out slices | path | sha256 |",
       "|---:|---|---|---|"]
for k in range(5):
    cfg = json.load(open("runs/cv_r3/fold%d/config.json" % k, encoding="utf-8"))
    src = "runs/cv_r3/fold%d/ckpt/l1_002000.pkl" % k
    dst = os.path.join(cd, "fold%d_l1_002000.pkl" % k)
    shutil.copyfile(src, dst)
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    man.append("| %d | %s | `%s` | `%s` |" % (k, ", ".join(map(str, cfg["held_out"])),
                                              dst.replace("\\", "/"), h))
open(os.path.join(cd, "MANIFEST.md"), "w", encoding="utf-8").write("\n".join(man) + "\n")
srcs = [("runs/cv_r3/stdout_fold%d.txt" % k, "train_fold%d" % k) for k in range(5)]
srcs.append(("runs/cv_r3/eval_stdout.txt", "eval"))
for src, slug in srcs:
    t = open(src, encoding="utf-8", errors="replace").read()
    e = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    open("Archive/CV_R3/stdout_%s_raw.html" % slug, "w", encoding="utf-8").write(
        "<!doctype html><meta charset=\"utf-8\"><title>%s</title><h1>RUN 2026-10-20 &mdash; CV_R3 %s</h1>"
        "<p>Verbatim stdout, unedited.</p><pre>%s</pre>\n" % (slug, slug, e))

# ---- shared tables --------------------------------------------------------------------
tab = ["| scans averaged (a) | R3 vs plain averaging | R3 vs best-case blur | best-case blur vs plain |",
       "|---:|---|---|---|"]
for a in A_SIZES:
    tab.append("| %d | %s | %s | %.2f× |" % (a, fmt(ep(a, "r3/plain")), fmt(vs_blur[a]),
                                             ep(a, "blur_oracle/plain")["gmean_ratio"]))
btab = ["| frequency band (cycles/pixel) | slices where R3 ≥ best-case blur | median R3 / blur |",
        "|---|---:|---:|"]
for (lo, hi), d in zip(((0, 0.0625), (0.0625, 0.125), (0.125, 0.25), (0.25, 0.75)), band):
    btab.append("| %g–%g | %d / 34 | %.2f× |" % (lo, hi, d["win"], d["med"]))

# ---- log section ------------------------------------------------------------------------
L = ["", "---", "", HEAD + ": cross-validated proof that averaging N2N-denoised scans beats plain averaging",
     "",
     "Pre-registration `Archive/CV_R3/PREREGISTRATION.md` (sha256 `%s`), written before training. 5 folds "
     "over all 34 slices (stratified by sigma_hat), fixed R3 recipe, iteration 2000, ratio 0; each slice "
     "scored only by the fold that never saw it; blur width chosen on the scored slice itself (oracle). "
     "Signal region; n_eff ratios are geometric means over slices with slice-bootstrap CIs and one-sided "
     "Wilcoxon tests." % pre_h[:16], "",
     "**PRIMARY (a=5, R3 vs plain): %s → SUPPORTED** under the pre-registered rule." % fmt(prim), ""]
L += tab
L += ["",
      "- R3 beats plain averaging on every slice at every a; the margin shrinks as a grows.",
      "- Against the best-case blur R3 is significantly better for a ∈ {%s} and not for a ∈ {%s}."
      % (", ".join(map(str, sig_blur)), ", ".join(map(str, nonsig))),
      "- Best-case blur alone already gives %.2f× at a=5, i.e. %.0f%% of R3's log-gain over plain "
      "averaging is available to any well-tuned spatial smoother; R3's specific edge is the remainder."
      % (blur_plain5["gmean_ratio"], 100 * share),
      "- Sensitivity: excluding slices 26, 29 → %s; all pixels → %s." % (fmt(sens), fmt(allpix)),
      "- Failure case: slice 29, the quietest slice (sigma_hat %.4f), quieter than every training slice of "
      "its fold (flagged when that fold was trained). R3 n_eff %.2f vs blur %.2f at a=1 (%.2f vs %.2f at "
      "a=9). Next-worst R3/blur ratio at a=1: slice %d at %.2f." % (
          s29[1]["sigma_hat"], s29[1]["r3"], s29[1]["blur_oracle"], s29[9]["r3"], s29[9]["blur_oracle"],
          second["slice"], second["r3"] / second["blur_oracle"]), "",
      "**Resolution / mechanism (a=5, all pixels, per frequency band):**", ""]
L += btab
L += ["",
      "R3's edge over blur is concentrated at the coarsest scale (every slice), where a spatial smoother "
      "cannot separate noise from signal; mid frequencies are a tie and the finest band a modest edge.",
      "",
      "**Figures:** `runs/cv_r3/eval/cv_neff_vs_a_{signal,all}.png`, `cv_ratio_vs_sigma.png`, "
      "`cv_bands_a5.png`, `maps/maps_cv_sNN.png` (all 34 slices: one a=5 split with an independent "
      "reference, and the all-10 deployment). **Data:** `per_slice.csv`, `bands.csv`, `endpoints.csv`. "
      "**Commands:** `python experiments/_cv_train.py --fold K` (K = 0..4), `python experiments/_cv_eval.py`, "
      "`python experiments/_cv_report.py`. Checkpoints: `checkpoints/CV_R3/MANIFEST.md`. Stdout: "
      "`Archive/CV_R3/stdout_*_raw.html`. Argument write-up: `%s`." % DOC, ""]
if WRITE_LOG:
    open(MD, "a", encoding="utf-8").write("\n".join(L))

# ---- argument document --------------------------------------------------------------------
os.makedirs("docs", exist_ok=True)
D = ["# Averaging denoised acquisitions beats averaging raw ones",
     "",
     "*Noise2Noise denoising of mandated repeat acquisitions — evidence summary. Generated from the "
     "experiment CSVs by `experiments/_cv_report.py`; every number below is computed, not typed.*",
     "",
     "## The claim",
     "",
     "ASL protocols acquire several noisy repetitions of every slice and average them; the noise of a "
     "plain average falls only as 1/a with the number of repetitions a. Because those repetitions are "
     "mandated anyway, they are free training data: a Noise2Noise network learns to denoise one "
     "repetition using the others as targets, with no clean reference required. Denoising each "
     "repetition and then averaging gives an estimate of the clean map that is far better than the "
     "plain average of the same repetitions.",
     "",
     "## Headline result",
     "",
     "With 5 of the 10 repetitions, the average of denoised repetitions is as accurate as a plain average "
     "of **%.1f×** as many repetitions (95%% CI %.2f–%.2f), on **every one of 34 slices**, each scored by a "
     "model that never saw that slice (p = %.1e). This was the pre-registered primary endpoint."
     % (prim["gmean_ratio"], prim["ci_lo"], prim["ci_hi"], prim["wilcoxon_p"]),
     "",
     "**Read this together with the baseline.** An idealised Gaussian smoother of the plain average, "
     "tuned on each scored slice, already reaches %.2f× at the same point — %.0f%% of the network's gain "
     "on a log scale. The network's own contribution beyond that idealised smoother is %.0f–%.0f%% when "
     "up to %d repetitions are used, and it comes from removing coarse-scale noise that a Gaussian "
     "smoother cannot reach (see below). In practice a smoother cannot be tuned on the answer, so the "
     "real-world gap to Gaussian smoothing is larger than this conservative figure."
     % (blur_plain5["gmean_ratio"], 100 * share,
        100 * (min(vs_blur[a]["gmean_ratio"] for a in sig_blur) - 1),
        100 * (max(vs_blur[a]["gmean_ratio"] for a in sig_blur) - 1), max(sig_blur)),
     "",
     "## How it was measured",
     "",
     "- **No clean ground truth is needed.** The 10 repetitions of a slice are split into a set A used to "
     "build the estimate and a disjoint set B used only as a reference. Because B's noise is independent "
     "of A, the error against mean(B) equals the error against the true clean map plus B's own noise "
     "variance, which is known from the spread of the repetitions. Subtracting it gives an unbiased "
     "error against the clean map. All possible splits are averaged.",
     "- **Metric:** n_eff = σ² / MSE, the number of repetitions a plain average would need for the same "
     "error. A plain average of a repetitions scores exactly a.",
     "- **No leakage:** 5-fold cross-validation over all 34 slices; every slice is scored only by the "
     "model trained without it. The training recipe and every analysis choice were fixed in a "
     "pre-registration before any of these models existed.",
     "- **A deliberately strong baseline:** a Gaussian blur of the plain average whose width is tuned "
     "*on the very slice being scored*. No real smoother can do better than this.",
     "- **Statistics treat slices as the replicates:** geometric-mean ratios with bootstrap-over-slices "
     "confidence intervals and Wilcoxon signed-rank tests.",
     "",
     "## Results",
     ""]
D += tab
D += ["",
      "1. **Against plain averaging the gain is large and universal**: every slice, every number of "
      "repetitions. It shrinks as more repetitions are averaged, because averaging removes the random "
      "part of each denoised repetition's error but not the part they share.",
      "2. **Against the best possible blur the gain is real but smaller**: significant when up to %d "
      "repetitions are used, not significant beyond (a = %s). Much of the improvement over plain "
      "averaging — %.0f%% of it on a log scale at a = 5 — is available to any well-tuned smoother. "
      "The fair statement is that the network adds a further %.0f–%.0f%% for a ≤ 5 on top of an "
      "idealised smoother that could not be tuned this way in practice."
      % (max(sig_blur), ", ".join(map(str, nonsig)), 100 * share,
         100 * (min(vs_blur[a]["gmean_ratio"] for a in sig_blur) - 1),
         100 * (max(vs_blur[a]["gmean_ratio"] for a in sig_blur) - 1)),
      "",
      "## Why it is not just smoothing",
      "",
      "MSE rewards smoothing, so the error was also split by spatial frequency (a = 5):",
      ""]
D += btab
D += ["",
      "The network's advantage is concentrated at the **coarsest** scale, on every slice. A linear "
      "low-pass smoother can only remove fine-scale noise; it cannot tell coarse-scale noise from signal. The "
      "network, having learned what these maps look like, removes noise at all scales. At finer scales "
      "it is on par with the best blur (0.125–0.25 cycles/pixel: %d/34 slices, median %.2f×) or "
      "slightly ahead (finest band: %d/34, median %.2f×) — so its advantage is not bought by erasing "
      "more detail than the best blur does."
      % (band[2]["win"], band[2]["med"], band[3]["win"], band[3]["med"]),
      "",
      "## Limits — stated plainly",
      "",
      "- **Noise levels outside training fail.** The one clear failure is slice 29, the quietest slice, "
      "whose noise was below every training slice of its fold: there the network scores %.2f against the "
      "blur's %.2f (one repetition). Training must cover the noise levels of deployment."
      % (s29[1]["r3"], s29[1]["blur_oracle"]),
      "- **Only Gaussian smoothing as the classical baseline.** Nonlinear classical denoisers (e.g. "
      "BM3D, non-local means, total variation) can also act at coarse scales and have not been compared "
      "yet; that comparison is the natural next step.",
      "- **One dataset.** All 34 slices come from one acquisition; generalisation to other subjects, "
      "scanners or protocols is untested.",
      "- **MSE-based.** n_eff is a squared-error measure. The frequency analysis addresses resolution; "
      "task-specific measures (e.g. perfusion quantification) are not yet evaluated.",
      "- **Robustness checks passed.** Excluding two slices earlier flagged as unreliable: %s. "
      "Scoring all pixels instead of the signal region: %s." % (fmt(sens), fmt(allpix)),
      "",
      "## What was tried and did not beat this",
      "",
      "Denoising the average of the repetitions once (instead of averaging per-repetition outputs) is "
      "optimal in theory, since the average carries all the information. A model trained for it "
      "improved substantially once its training covered low noise levels evenly and used flip/rotation "
      "augmentation, but on this dataset it still trails the per-repetition approach on quiet slices. "
      "It is a candidate for larger training sets.",
      ""]
ABL = "runs/cv_ablation/report/summary.json"
if os.path.exists(ABL):
    ab = json.load(open(ABL, encoding="utf-8"))
    pr, b5 = ab["prim"], ab["base5"]
    D += ["## Ablation — does the training loss matter?",
          "",
          "The recipe above trains with an unweighted L1 loss. A second pre-registered experiment "
          "(`Archive/CV_ABLATION/PREREGISTRATION.md`) retrained the identical cross-validation with three "
          "alternatives — same folds, batches and noise draws, only the loss changed — and compared every "
          "slice against the L1 model (5 repetitions, signal region; Holm-corrected over the three):",
          "",
          "| training loss | vs the L1 recipe | slices better | Holm p | vs plain averaging | vs best-case blur |",
          "|---|---|---:|---:|---:|---:|",
          "| L1, unweighted (the recipe above) | — | — | — | %.2f× | %.3f× |"
          % (b5["A"]["plain"][0], b5["A"]["blur_oracle"][0])]
    for k in "BCD":
        D.append("| %s | %.3f× (95%% CI %.3f–%.3f) | %d/34 | %.2g | %.2f× | %.3f× |" % (
            ab["labels"][k], pr[k]["gmean"], pr[k]["ci_lo"], pr[k]["ci_hi"], pr[k]["wins"],
            pr[k]["p_holm"], b5[k]["plain"][0], b5[k]["blur_oracle"][0]))
    best = max("BCD", key=lambda k: pr[k]["gmean"])
    assert best == "C" and pr["C"]["verdict"] == "better than A"
    pa = list(csv.DictReader(open("runs/cv_ablation/report/paired.csv", encoding="utf-8")))
    cr = {int(r["a"]): float(r["gmean"]) for r in pa if r["num"] == "C" and r["region"] == "signal"}
    br = {int(r["a"]): float(r["gmean"]) for r in pa if r["num"] == "B" and r["region"] == "signal"}
    bp1 = next(float(r["p"]) for r in pa if r["num"] == "B" and r["region"] == "signal" and int(r["a"]) == 1)
    assert br[1] > 1 and br[1] == max(br[1], cr[1])
    aa = sorted(cr)
    assert all(cr[x] < cr[y] for x, y in zip(aa, aa[1:])) and cr[1] < 1 < cr[5]
    cd = ab["c_vs_d"]
    D += ["",
          "- Every loss keeps the headline: all four beat plain averaging on all 34 slices.",
          "- **The norm matters most.** Under the same noise weighting, MSE beats L1 by %.3f× "
          "(%d/34 slices, p = %.1e) — as expected, since MSE learns exactly the posterior mean that "
          "n_eff scores. Noise-normalized L1 is significantly worse than the plain L1 recipe."
          % (cd["gmean"], cd["wins"], cd["p"]),
          "- **Noise-normalized MSE is the only significant improvement** over the recipe above "
          "(Holm p = %.2g). Its edge grows steadily with the number of repetitions averaged — from "
          "%.3f× with one repetition to %.3f× with %d — so for single-map use it is *not* better. "
          "With one repetition unweighted MSE does best (%.3f×; unadjusted p = %.2g, a secondary "
          "endpoint, so suggestive only)."
          % (pr["C"]["p_holm"], cr[aa[0]], cr[aa[-1]], aa[-1], br[1], bp1),
          "- Weighting by noise level changes MSE little on average but shifts its benefit toward the "
          "quieter slices (rank correlation with slice noise %.2f)." % ab["rho_c"][0],
          "- These are the best of four variants scored on the same slices, so the winning margin is "
          "mildly optimistic; the recipe above remains the pre-registered result.",
          ""]
RAT = "runs/cv_ratio/report/summary.json"
if os.path.exists(RAT):
    rt = json.load(open(RAT, encoding="utf-8"))
    lab = rt["labels"]
    bs = lambda arm, a, b: rt["base"]["%s|%d|%s" % (arm, a, b)]
    sf = lambda arm, r, a: rt["surf"]["%s|%.4f|%d" % (arm, r, a)]
    # claims below, asserted
    assert all(bs(k, a, "plain")["wins"] == 34 for k in "ABCD" for a in (1, 5, 9))
    mb = {k: [bs(k, a, "blur_matched") for a in (1, 5, 9)] for k in "ABCD"}
    assert all(r["gmean"] > 1 and r["p"] < 0.01 for k in "ABCD" for r in mb[k])
    assert rt["prim"]["B"]["verdict"] == "worse than L1"
    assert sf("A", 1 / 3, 9) > sf("A", 0.0, 9) and sf("A", 0.0, 1) > sf("A", 1 / 3, 1)
    floor = []
    for k, d in (("A", "runs/cv_r3"), ("C", "runs/cv_ablation/mse_norm")):
        Bn = [r for r in csv.DictReader(open(d + "/ratio_eval/bands_ratio.csv", encoding="utf-8")) if r["a"] == "1"]
        Nz = {r["slice"]: float(r["noise"]) for r in csv.DictReader(open(d + "/ratio_eval/per_slice_ratio.csv",
              encoding="utf-8")) if r["region"] == "per_r" and abs(float(r["ratio"]) - 1 / 3) < 1e-6}
        floor.append(float(np.median([float(r["r0.3333_band3"]) * Nz[r["slice"]] ** 2 for r in Bn])))
    assert all(0.85 < f <= 1.0 for f in floor)
    hb = [rt["bandA"]["%s|1" % k][3][1] for k in "ABCD"]
    lb = [min(rt["bandA"]["%s|1" % k][b][1] for b in range(3)) for k in "ABCD"]
    D += ["## At matched noise — decoupling the loss from the decoding ratio",
          "",
          "The results above decode every repetition with `target_ratio = 0` (the full posterior mean), "
          "whose error advantage is partly blur. A third pre-registered evaluation "
          "(`Archive/CV_RATIO/PREREGISTRATION.md`, no retraining) instead locks each denoised repetition "
          "to the noise of its own training target, the leave-one-out mean of 9: r = 1/3 of the input "
          "noise. The achieved residual noise was %.3f (median over slices; the one-step rule holds), "
          "except on the quietest slices, which lie below most training data of their fold."
          % rt["noise3"]["A"][0],
          "",
          "| training loss | 1 repetition: vs plain | vs noise-matched blur | 9 repetitions: vs plain | vs noise-matched blur |",
          "|---|---:|---:|---:|---:|"]
    for k in "ABCD":
        D.append("| %s | %.2f× | %.3f× (%d/34) | %.2f× | %.3f× (%d/34) |" % (
            lab[k], bs(k, 1, "plain")["gmean"], bs(k, 1, "blur_matched")["gmean"], bs(k, 1, "blur_matched")["wins"],
            bs(k, 9, "plain")["gmean"], bs(k, 9, "blur_matched")["gmean"], bs(k, 9, "blur_matched")["wins"]))
    D += ["",
          "- **At equal residual noise every loss still beats both plain averaging (all 34 slices) and a "
          "Gaussian blur tuned to the same noise** (p < 0.01 at 1, 5 and 9 repetitions).",
          "- **Where the gain sits.** For single maps, against the noise-matched blur, every loss wins the "
          "three lower frequency bands on %s and loses the finest band on %s. The finest-"
          "band loss is the deliberately kept noise, not lost detail: there the network's error equals its "
          "residual noise (n_eff × noise² = %.2f–%.2f, median; 1 = pure retained noise). At equal total "
          "noise the network keeps the reference's fine-grained noise texture, while blur trades it for "
          "blotchy coarse-scale noise."
          % ("every slice" if min(lb) == 34 else "at least %d of 34 slices" % min(lb),
             "every slice" if max(hb) == 0 else "all but %d of 34 slices" % max(hb),
             min(floor), max(floor)),
          "- **The loss ranking depends on the ratio.** At r = 1/3, unweighted MSE is significantly worse "
          "than L1 for single maps (%.3f×, Holm p = %.2g), the reverse of the r = 0 ranking. Both "
          "noise-normalized losses are on par with plain L1 for single maps; noise-normalized MSE pulls "
          "ahead when 5 or more repetitions are averaged (%.3f× at 9; secondary endpoint, p = %.1g "
          "unadjusted). Losses must therefore be compared at a stated ratio."
          % (rt["prim"]["B"]["gmean"], rt["prim"]["B"]["p_holm"],
             rt["secA"]["C|9|signal"]["gmean"], rt["secA"]["C|9|signal"]["p"]),
          "- **Ratio and averaging interact.** The ratio with the lowest error grows with the number of "
          "repetitions averaged; with 9 repetitions, r = 1/3 (%.1f) beats the full posterior mean (%.1f), "
          "which over-smooths when many outputs are averaged — while for a single map r = 0 has the lowest "
          "error (%.1f vs %.1f) at the price of visible blur."
          % (sf("A", 1 / 3, 9), sf("A", 0.0, 9), sf("A", 0.0, 1), sf("A", 1 / 3, 1)),
          ""]
D += ["## Reproduction",
      "",
      "Pre-registration: `Archive/CV_R3/PREREGISTRATION.md` (sha256 `%s`). Training: "
      "`python experiments/_cv_train.py --fold K` for K = 0–4. Evaluation: `python experiments/_cv_eval.py`. "
      "This document: `python experiments/_cv_report.py`. Model weights and hashes: "
      "`checkpoints/CV_R3/MANIFEST.md`. Per-slice numbers: `runs/cv_r3/eval/per_slice.csv`." % pre_h[:16],
      ""]
open(DOC, "w", encoding="utf-8").write("\n".join(D))
print("\n".join(L))
print("\nwrote", DOC)
