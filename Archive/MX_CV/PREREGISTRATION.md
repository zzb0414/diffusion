# MX_CV pre-registration — pooled training on datasets 1+2+3

Written 2026-10-26 09:37:39, BEFORE any pooled model was trained.

## Design
- Folds grouped by slice index, identical to CV_R3 (runs/cv_r3/fold*/config.json held_out): slice s of
  datasets 1, 2 and 3 is always held out together. 34 indices x 3 datasets = 102 held-out slices.
- Arms (A) L1, (B) L1 noise-normalized, (C) MSE, (D) MSE noise-normalized; recipe as CV_R3/CV_ABLATION
  (p_raw 0.5, LR 1e-3, batch 8, UNet 32/32, seed 101), training pool = training-fold slices of all three
  datasets, ladder 0.0339 (global min sigma_hat) .. 0.40.
- Trained to 6000 iterations; iteration 2000 is the PRIMARY checkpoint (no LR schedule), 6000 secondary.
- Comparator: the dataset1-only fold models of the same arm (GEN for ds2/3; CV_RATIO for ds1).

## Endpoints
- PRIMARY: arm A, iteration 2000, r = 1/3, a = 1, signal region: per-slice paired n_eff ratio
  pooled / dataset1-only, separately for ds1, ds2, ds3 (34 slices each); geometric mean, slice bootstrap
  95% CI, two-sided Wilcoxon, Holm over the 3 datasets.
- SECONDARY: arms B-D and other a; pooled vs plain averaging and vs noise-matched blur per dataset;
  loss comparison within pooled (B, C, D vs A, Holm) per dataset; r x a sweep per dataset; 6000 vs 2000;
  per band.

## Caveats stated in advance
- One subject; dataset1 slices only approximately aligned with datasets 2/3 (2 and 3 match exactly).
- Per-slice max normalization is noise-inflated at low SNR; kept for consistency with all earlier runs.
