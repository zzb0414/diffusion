# CV_R3 pre-registration

Written 2026-10-20 13:23:26, BEFORE any cross-validated model was trained or evaluated.

## Claim under test
Averaging N2N-denoised realizations (R3 recipe) estimates the clean map with lower error than
plain averaging of the same realizations, and the gain is not explained by spatial smoothing.

## Design
- 5-fold cross-validation over all 34 slices, folds stratified by sigma_hat. Each slice is scored only
  by the model of the fold that held it out.
- Recipe fixed in advance = R3: total-noise labels, L1, p_raw 0.5, LR 1e-3, batch 8, UNet base 32 /
  emb 32, seed 101, 2000 iterations; the iteration-2000 weights are evaluated. No selection on
  held-out data.
- Estimator: each realization denoised alone with denoise_to(target_ratio = 0), then the outputs of a
  subset A are averaged. Ratio 0 is fixed in advance (no tuning).
- Split-half scoring: all C(10, a) subsets A, a in {1, 2, 3, 5, 7, 9}; reference = mean of the
  complement B; unbiased MSE vs clean = MSE vs mean(B) - noise var / |B|; n_eff = sigma^2 / MSE.
- Baselines: plain average (n_eff = a); Gaussian blur of the plain average with its width chosen
  per slice, per a, per region ON THE SCORED SLICE ITSELF (oracle, i.e. deliberately favouring the
  baseline).

## Endpoints
- PRIMARY: signal region, a = 5: geometric-mean ratio n_eff(R3) / n_eff(plain) over the 34 slices,
  95% CI by bootstrap over slices (10,000 resamples), one-sided Wilcoxon signed-rank on log ratios.
  Claim supported if the CI lower bound > 1 and p < 0.01.
- SECONDARY (same statistics): R3 vs oracle blur at a = 5, signal region; all a in {1,2,3,5,7,9};
  all-pixel region.
- RESOLUTION: per-slice, per-band n_eff (4 radial bands, all pixels) at a = 5; count of slices where
  R3 >= oracle blur in the HIGHEST band.
- SENSITIVITY: all endpoints repeated excluding slices 26 and 29 (earlier flagged as unreliable
  references).

## Known limitations (stated in advance)
- One dataset; slices may share subject-level structure, so generalisation beyond this acquisition is
  not tested.
- n_eff is MSE-based; the band analysis is the resolution check.
