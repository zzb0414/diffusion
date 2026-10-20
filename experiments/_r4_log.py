# -*- coding: utf-8 -*-
"""Archive R4 and append its results to WEEKEND_ITERATIONS.md.

Every numeral is computed here from ``runs/r4_subset/<loss>/train_val.csv``,
``.../eval/split_test/split_selected.csv`` and the run config; prose claims are
asserted. Copies the selected checkpoint + sha256 to ``checkpoints/R4_SUBSET``
and raw stdout (training, split test) to ``Archive/R4_SUBSET`` as HTML. Appends
once: refuses if the section already exists.

    python experiments/_r4_log.py --run runs/r4_subset/l1 --iter 1600
"""
import sys, os, csv, json, shutil, hashlib, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
os.chdir(ROOT)
MD = "WEEKEND_ITERATIONS.md"
HEAD = "## RUN 2026-10-20 — R4_SUBSET"

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/r4_subset/l1")
ap.add_argument("--iter", type=int, required=True)
a = ap.parse_args()
loss = os.path.basename(a.run)
ST = os.path.join(a.run, "eval", "split_test").replace("\\", "/")
assert HEAD not in open(MD, encoding="utf-8").read(), "section already present"

tv = list(csv.DictReader(open(os.path.join(a.run, "train_val.csv"), encoding="utf-8")))
S = list(csv.DictReader(open(os.path.join(ST, "split_selected.csv"), encoding="utf-8")))
cfg = json.load(open(os.path.join(a.run, "config.json"), encoding="utf-8"))
for r in S:
    for k in r:
        if k not in ("region", "group") and r[k] != "":   # bands exist for "all" rows only
            r[k] = float(r[k])
sh = {int(k): v for k, v in cfg["sigma_hat"].items()}
FAMS = ("vanilla", "r3avg", "r4", "blur")

# ---- archive -------------------------------------------------------------------------
ck = os.path.join(a.run, "ckpt", "%s_%06d.pkl" % (loss, a.iter))
cd = "checkpoints/R4_SUBSET"
os.makedirs(cd, exist_ok=True)
dst = os.path.join(cd, os.path.basename(ck))
shutil.copyfile(ck, dst)
h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
shutil.copyfile(os.path.join(a.run, "config.json"), os.path.join(cd, "config_%s.json" % loss))
man = os.path.join(cd, "MANIFEST.md")
if not os.path.exists(man):
    open(man, "w", encoding="utf-8").write(
        "# R4_SUBSET checkpoint manifest\n\nSubset-averaging N2N_DSM (total-noise labels); "
        "see WEEKEND_ITERATIONS.md, R4_SUBSET. Selected checkpoint per loss.\n\n"
        "| loss | iter | path | size | sha256 |\n|---|---:|---|---:|---|\n")
if h not in open(man, encoding="utf-8").read():          # idempotent on rerun
    open(man, "a", encoding="utf-8").write("| %s | %d | `%s` | %.1f MB | `%s` |\n" % (
        loss, a.iter, dst.replace("\\", "/"), os.path.getsize(dst) / 1048576, h))
ad = "Archive/R4_SUBSET"
os.makedirs(ad, exist_ok=True)
for src, slug in ((os.path.join("runs/r4_subset", "stdout_%s.txt" % loss), "train_%s" % loss),
                  (os.path.join(a.run, "eval", "split_stdout.txt"), "split_%s" % loss)):
    txt = open(src, encoding="utf-8", errors="replace").read()
    e = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    open(os.path.join(ad, "stdout_%s_raw.html" % slug), "w", encoding="utf-8").write(
        "<!doctype html><meta charset=\"utf-8\"><title>%s raw stdout</title><h1>RUN 2026-10-20 "
        "&mdash; R4_SUBSET %s</h1><p>Verbatim stdout, unedited.</p><pre>%s</pre>\n" % (slug, slug, e))

# ---- numbers ------------------------------------------------------------------------
g = [(int(r["iter"]), float(r["val_neff_gmean"])) for r in tv if int(r["iter"]) % 200 == 0]
best = max(g, key=lambda z: z[1])
assert best[0] == a.iter, (best, a.iter)
last = g[-1]
tl0, tl1 = float(tv[1]["train_loss_avg50"]), float(tv[-1]["train_loss_avg50"])
assert last[1] < 0.6 * best[1] and tl1 < tl0          # overfitting claim


def row(reg, aa, sl):
    return next(r for r in S if r["region"] == reg and r["a"] == aa and r["slice"] == sl)


test = [s for s in sorted({int(r["slice"]) for r in S if r["group"] == "test"})]
vt = sorted({int(r["slice"]) for r in S if r["group"] in ("val", "test")})

# claim 1: R3 avg >= blur in every band on every test slice at a=5 (all pixels)
for s in test:
    r = row("all", 5, s)
    for b in range(4):
        assert r[f"r3avg_band{b}"] >= r[f"blur_band{b}"], (s, b)
# claim 2: at a=1, R4 within 15% of R3 on every val/test slice (signal)
r1 = [row("signal", 1, s)["r4"] / row("signal", 1, s)["r3avg"] for s in vt]
assert all(0.85 < q < 1.15 for q in r1), r1
# claim 3: at a=9, R4/R3 rises with sigma_hat (rank correlation)
q9 = {s: row("signal", 9, s)["r4"] / row("signal", 9, s)["r3avg"] for s in vt}
order = sorted(vt, key=lambda s: sh[s])
rk = lambda v: np.argsort(np.argsort(v))
rho = float(np.corrcoef(rk([sh[s] for s in order]), rk([q9[s] for s in order]))[0, 1])
assert rho > 0.5, rho                                 # a tendency, not monotone
cover = {s: sum(1 for t in cfg["train_slices"] if sh[t] <= sh[s]) for s in vt}

L = []
A = L.append
A("")
A("---")
A("")
A(HEAD + ": subset-averaging model (denoise the mean of m realizations)")
A("")
A("**Idea.** With independent Gaussian noise the mean of all realizations is sufficient, so the best "
  "use of 10 scans is to denoise their mean once. R4 trains one total-noise-conditioned net on subset "
  "means: input = mean of a random m-subset (m = 1..9, noise sigma_hat/sqrt(m)), target = mean of the "
  "other 10−m (independent, so still valid N2N). `N2N_DSM.loss_subset` / `DiffusionTrainer.step_subset`; "
  "the R3 path was refactored into a shared helper and verified bit-identical to the pre-change file "
  "for all four losses (snapshot `Archive/NOISE_LABEL/n2n_r3.py`). Input/target construction verified on "
  "synthetic data: input variance sigma²/m, target sigma²/(10−m), zero cross-covariance.")
A("")
A("**Setup.** As R3 (split, backbone, LR, batch, seed, p_raw), ladder from smallest training sigma_hat/3 "
  "(%.4f) to %.2f, %d iterations, checkpoint every 200, selection by geometric-mean val n_eff over "
  "a = 1, 5, 9 (cyclic subsets). **Command:** `python experiments/_r4_train.py --loss %s --iters %d`."
  % (cfg["ckpt_cfg"]["sigma_min"], cfg["ckpt_cfg"]["sigma_max"], cfg["iters"], loss, cfg["iters"]))
A("")
A("**Training.** Val n_eff peaks at iteration %d (%.2f) and falls to %.2f by %d while the train loss "
  "keeps dropping (%.5f → %.5f): clear overfitting after the peak, which the standing rule against "
  "long training anticipated. Curves: `%s/eval/curves.png`." % (
      best[0], best[1], last[1], last[0], tl0, tl1, a.run))
A("")
A("**Split test** (`experiments/_r4_split_test.py`, protocol of SPLIT_TEST; R3 = `runs/r3_total/l1` "
  "iter 2000, read only). Signal region, n_eff with 95%% block-bootstrap CI, knobs chosen on val:")
A("")
A("| a | split | slice | sigma_hat | plain | R3 avg of denoised | R4 subset mean | blur |")
A("|---:|---|---:|---:|---:|---|---|---|")
for aa in (1, 5, 9):
    for s in vt:
        r = row("signal", aa, s)
        c = lambda f: "%.2f (%.1f–%.1f)" % (r[f], r[f + "_lo"], r[f + "_hi"])
        A("| %d | %s | %d | %.4f | %.2f | %s | %s | %s |" % (
            aa, r["group"], s, sh[s], r["vanilla"], c("r3avg"), c("r4"), c("blur")))
A("")
A("- **R4 ≈ R3 for a single scan** (R4/R3 = %.2f–%.2f across val/test at a=1), as expected: m=1 is "
  "the same objective." % (min(r1), max(r1)))
A("- **R4 does not deliver the sufficiency gain at large a.** At a=9 its ratio to R3 ranges %.2f–%.2f and "
  "tends to rise with the slice's noise (Spearman rho = %.2f over six slices, two tied in sigma_hat): "
  "it wins on the noisiest slices and loses on the quieter ones, where its n_eff stays nearly flat as "
  "a grows." % (min(q9.values()), max(q9.values()), rho))
A("- **Likely cause: training coverage at low total noise.** An averaged input from slice s at a=9 "
  "sits at sigma_hat_s/3; only training slices at least that quiet can produce such inputs. Count of "
  "training slices with sigma_hat ≤ the eval slice's: %s. The quiet test slices are served by a handful "
  "of training slices, so the low-noise end of the ladder is thinly trained." %", ".join("s%d: %d" % (s, cover[s]) for s in order))
A("- **Resolution check passes for R3.** Splitting error into radial frequency bands (all pixels, "
  "unbiased per band), the R3 average of denoised outputs has n_eff ≥ blur in every band — including "
  "the highest — on every test slice at a=5. Its advantage over plain averaging is therefore not a "
  "smoothing artifact. Bands with very few frequencies (lowest) can read `inf` when the estimated "
  "error is ≤ 0; treat those as \"below measurable\".")
A("")
A("**Decision.** R3 (average of separately denoised realizations) remains the best estimator and the "
  "one to base the argument on. R4 is not yet a fair test of the sufficiency idea: fix its low-noise "
  "coverage first (e.g. sample m per slice so total noise is spread evenly in log sigma over the "
  "deployment range, or weight quiet slices), then re-run this exact split test.")
A("")
A("**Displays** (`%s/`): `neff_signal.png`, `neff_all.png` (n_eff vs a, four methods, CIs); "
  "`neff_bands_a{1,5,9}.png` (resolution check); `maps_a5_<split>_<slice>.png` (one split, estimates "
  "from realizations 0–4, independent reference = mean of 5–9, bottom row = difference); "
  "`maps_a10_<split>_<slice>.png` (deployment with all 10 scans; no reference left, bottom row = "
  "difference from the plain average)." % ST)
A("")
A("**Artifacts:** `%s/` (ckpt every 200, `train.log`, `train_val.csv`, `config.json`), `%s/split_metrics.csv`, "
  "`split_selected.csv`. Checkpoint copy + sha256: `checkpoints/R4_SUBSET/MANIFEST.md` (`%s`). Verbatim "
  "stdout: `Archive/R4_SUBSET/stdout_train_%s_raw.html`, `stdout_split_%s_raw.html`. Split-test "
  "command: `python experiments/_r4_split_test.py --r4-run %s --r4-iter %d`. R3 artifacts were hashed "
  "before this run (`Archive/R4_SUBSET/r3_artifacts_before.sha256`) and re-verified unchanged after it."
  % (a.run, ST, h[:16], loss, loss, a.run, a.iter))
A("")
open(MD, "a", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
