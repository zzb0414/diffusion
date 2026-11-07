# GAP H pre-registration — is the ASL denoise-the-mean gain data-limited? (ASL-side dose-response)

Written 2026-11-06, 21:55, after the GAP F reading and BEFORE any H run. GAP G is running; H does not depend on G's outcome.

## Motivation
- **GAP F:** on fastMRI DT content, going from 62 training volumes to 4 lowers SUB/PER at a = 9 from 1.457 to 1.301; going to 1 lowers it to 0.885. Denoise-the-mean is much more data-hungry than PER.
- **Open question:** is ASL, trained on 68 slices of one subject, on the steep part of its own curve? F cannot answer this, because fastMRI volumes and ASL slices are not equivalent units.
- **The direct test:** subsample the ASL training set and see whether the ASL gain shrinks.

## Design (`runs/gap/queue_h.sh`, `experiments/_h_report.py`; no training or evaluation code changed)
- **Recipe:** exactly ASL-SUBJ / ASL-COIL (datasets 1 + 1104a, fold −1, 3000 iterations, MSE; SUB `--ladder-div 3 --sub-uniform`; SUBM also `--m-dist linear`; seeds 101–102). The one change is the existing `--train-frac`:
  - 0.25 keeps 8 of 34 slice indices, so 16 images;
  - 0.5 keeps 17 indices, so 34 images.
  The subsets are nested and fixed (seed 20261028 − 1). The noise ladder is unchanged, because it is set from the global σ̂ minimum.
- **Arms:** PER, SUB and SUBM at frac 0.25 and 0.5, which makes 12 trainings and 36 evaluations.
- **Evaluation:** 1104c, `_gr_eval --datasets 6 --single`, with selection cross-fitted as in ASL-SUBJ.
- **References (frac 1, read only):** `runs/subj/{PER,SUB}_s*` and `runs/coil/subj/SUBM_s*`.
- **Known confound, accepted:** at a fixed 3000 iterations, a smaller set means more passes per image. Checkpoint selection over 1000/2000/3000 partly absorbs this. The same holds in F2.

## PRIMARY (Holm over 2), a = 9, paired by slice over the 34 slices of 1104c
- H1: (SUB/PER, frac 0.25) / (SUB/PER, frac 1).
- H2: (SUBM/PER, frac 0.25) / (SUBM/PER, frac 1).

## Interpretation (fixed in advance)
- **H ≤ 0.95, with the CI excluding 1:** the ASL gain is data-limited at today's set size. It should grow with more subjects (the coil2 data plan) or with pre-training. ASL then behaves like fastMRI DT in F2.
- **H in [0.97, 1.03]:** flat. Image count is not limiting at today's size. What remains is subject diversity (untestable with one coil1 training subject; needs more subjects) or the content (i); GAP G bears on (i).
- **Between the two bands:** weak dependence; report it as such.
- **Secondary:** frac 0.5 ratios, and absolute n_eff per arm. That shows whether PER and SUB lose data-efficiency at different rates, as in F2, where SUB lost 0.38× and PER 0.63× from 62 to 1 volume.
- **My expectation:** weak dependence, H ≈ 0.95–1.0. ASL's gain is already near the Gaussian-content level, so there is little left to lose.

## No-overwrite
Outputs only go under `runs/gap/h/` and `runs/gap/report/summary_h.json`. No code is modified.
