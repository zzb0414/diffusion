# GROUND pre-registration — is target size k marginal (KR), and does denoising the mean win with more data (SM)?

Written 2026-10-28, BEFORE any GROUND model was trained or evaluated. Motivation: `docs/RESEARCH_NOTES.md` notes 3–4
and TQ E1 (k = 1 vs k = 9 differed by ≤ 4%, one seed, one budget).

## Shared
- Folds: CV_R3's slice-index folds (`_data.cv_folds`); every slice is scored only by the fold that held its index out.
- Backbone, LR, batch, p_raw, ladder top, total-noise labels: the CV_R3 recipe. Loss L1 everywhere (arm A).
- Training: `experiments/_gr_train.py`; evaluation: `experiments/_gr_eval.py` (split-half unbiased n_eff over all
  C(10, a) splits, a ∈ {1, 2, 3, 5, 7, 9}, r ∈ {0, 1/12, 1/6, 1/4, 1/3, 1/2, 1}, signal and all regions; the same
  estimator as `_ratio_core.evaluate_slice`, verified to reproduce its n_eff before use). No blur baselines here.
- Statistics: per-slice ratios, geometric mean over the 34 slice indices, slice bootstrap (10 000 resamples, seed 0).

## Step 1 — KR: is k marginal?
Runs (dataset 1 only, `--mode per`):
- KR-seed: k ∈ {1, 9} × seeds {102, 103} × 5 folds, 4000 iterations, checkpoint every 500. Seed 101 already exists
  (k = 9: `runs/cv_r3`; k = 1: `runs/tq/A_k1`, iterations ≤ 2000) and is included as the third seed.
- KR-frac: k ∈ {1, 9} × training fraction {0.5, 0.25} × 5 folds, seed 101, 2000 iterations. Fractions are nested
  subsets of each fold's training slice indices (fixed permutation per fold).

PRIMARY 1 (equivalence): ratio n_eff(k=1)/n_eff(k=9), dataset 1, a = 1, r = 1/3, signal, iteration 2000; per slice
the geometric mean over the 3 seeds; 90% bootstrap CI of the geometric mean over slices.
- CI inside [0.95, 1.05] → **k marginal (equivalent within ±5%)**; CI entirely below 0.95 → k = 1 worse; entirely
  above 1.05 → k = 1 better; otherwise inconclusive.

SECONDARY: same rule per seed; a = 5, 9; each method's best r on the grid; iterations 500, 1000, 4000 (learning
curve; seed 101 only up to 2000); fractions 0.5 and 0.25 (seed 101); seed-to-seed ratio of k = 9 models
(102/101, 103/101) as the noise floor for interpreting effects.
PREDICTION (N2N theory): equivalent at 2000; any k = 1 deficit appears early in training or with less data.

## Step 2 — SM: denoise the mean vs denoise each, then average
Arms (seed 101, dihedral augmentation for both, 3000 iterations, checkpoint every 500):
- PER: per-repetition objective (`loss_multireal`, full LOO target) — "denoise each, then average".
- SUB: R4c recipe (`loss_subset`, input = mean of m ∈ 1..9 repetitions, target = mean of the other 10 − m,
  (slice, m) pairs balanced in log total noise) — "denoise the mean", decoded at σ̂/√a.
Each trained on dataset 1 only and on datasets 1+2+3 pooled (ladder bottom: global smallest σ̂ / 3 for both arms).
Evaluated on dataset 1 (all models) and datasets 2–3 (pooled models).
r selection: for each (arm, training set, dataset, a), the r on the grid maximizing the geometric-mean n_eff over the
evaluated slices — the same symmetric rule for both arms (a mild optimism shared by both; fixed r = 0 and 1/3 also reported).

PRIMARY 2a: pooled training, dataset 1, a = 9, iteration 3000: n_eff SUB / PER at each arm's selected r;
geometric mean, 95% bootstrap CI, two-sided Wilcoxon.
PRIMARY 2b (data limitation): per slice, log(SUB/PER)_pooled − log(SUB/PER)_dataset1-only, dataset 1, a = 9,
iteration 3000; two-sided Wilcoxon. Holm over 2a and 2b.
SECONDARY: a = 1, 3, 5; datasets 2 and 3; fixed r; iteration 2000; augmentation effect (PER-aug vs `runs/cv_r3` and
`runs/mx_cv/A` at iteration 2000).
PREDICTION: sufficiency theory says SUB ≥ PER given enough data; if R4's deficit was data-limited, 2b > 0.

## No-overwrite
New outputs only under `runs/gr/`, `Archive/GROUND/`; scripts refuse to overwrite; sha256 of all existing results in
`Archive/GROUND/protected_before.sha256`, re-verified at the end.

## Step 3 (read-only, not pre-registered as a test)
Fit n_eff(a, r) from existing sweeps to error ≈ bias²(r) + variance(r)/a + correlation term; compare predicted with
measured best r per a. Exploratory.
