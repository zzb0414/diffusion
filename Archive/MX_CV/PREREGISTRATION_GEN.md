# GEN pre-registration — generalization of dataset1-only models to datasets 2 and 3

Written 2026-10-26 09:37:39, BEFORE any model was evaluated on dataset2 or dataset3.

## Setup
The existing dataset1-only cross-validated fold models — (A) L1 = runs/cv_r3, (B) L1 noise-normalized =
runs/cv_ablation/l1_norm, (C) MSE = runs/cv_ablation/mse, (D) MSE noise-normalized =
runs/cv_ablation/mse_norm; evaluated weights = iteration 2000 — decode every slice of dataset2 and
dataset3 (same subject, poorer receiver coil, lower SNR; never used in training). Each slice is decoded by
the fold model that held its slice INDEX out, so both the data and the anatomy are unseen.
Decoding, metrics and estimator exactly as CV_RATIO (r sweep {0, 1/12, 1/6, 1/4, 1/3, 1/2, 1}, a in
{1,2,3,5,7,9}, all C(10,a) splits, unbiased n_eff, sigma_out, oracle and noise-matched blur, bands).
sigma_hat per slice from that slice's own 10 repetitions, same estimator.

## Endpoints
- PRIMARY (per dataset, ds2 and ds3 separately): arm A, r = 1/3, a = 1, signal region: n_eff vs plain
  averaging and n_eff vs the noise-matched blur; geometric mean over 34 slices, slice bootstrap 95% CI,
  one-sided Wilcoxon (greater), Holm over the 2 datasets within each comparison.
- Calibration: sigma_out / sigma_hat at r = 1/3 per dataset (median, IQR).
- SECONDARY: arms B-D; all a and r; per band; comparison with the dataset1 numbers already obtained.

## Caveats stated in advance
- dataset3 slices quieter than 0.0368 (the lowest training sigma_hat of the dataset1 models) are
  extrapolated; reported separately.
- Per-slice max normalization is noise-inflated at low SNR, so the signal scale differs between datasets.
