# CV_RATIO pre-registration

Written 2026-10-21 10:45:55, BEFORE any ratio evaluation was run. Evaluation-only: the 20 cross-validated fold
models already exist (A = L1 unweighted = CV_R3; B = MSE; C = MSE noise-normalized; D = L1
noise-normalized; runs/cv_r3, runs/cv_ablation/{mse,mse_norm,l1_norm}).

## Decoding
Every realization decoded alone with denoise_to(target_ratio = r): out = x0_hat + r (y - x0_hat).

## Set A — loss comparison at the locked theoretical ratio
r = 1/sqrt(N-1) = 1/3 for every slice and realization (each output keeps the noise of its training
target, the leave-one-out mean of 9).
- PRIMARY: a = 1 (single-map deliverable), signal region: per-slice paired n_eff ratio of B, C, D vs A
  over the 34 slices; geometric mean, 10,000-resample slice-bootstrap 95% CI, two-sided Wilcoxon
  signed-rank, Holm over the 3 comparisons at alpha = 0.05.
- SECONDARY: the same at a in {2,3,5,7,9} and all-pixel region; each arm vs plain averaging and vs a
  NOISE-MATCHED Gaussian blur (width set so a blurred single realization has the same residual noise as
  that arm's output at r = 1/3, per slice); per-band n_eff; achieved residual noise per arm.

## Set B — target_ratio sweep x split size (descriptive)
r in {0, 1/12, 1/6, 1/4, 1/3, 1/2, 1}, a in {1,2,3,5,7,9}, all four arms: geometric-mean n_eff with
slice-bootstrap CIs, achieved residual noise vs r, per-band n_eff vs r, n_eff vs residual-noise
frontiers against the blur frontier. No hypothesis test; no r is selected on these data.

## Estimator (unchanged from CV_R3)
All C(10, a) splits; reference = mean of the complement; unbiased MSE vs clean = MSE vs mean(B) -
region noise variance / |B|; n_eff = sigma^2 / MSE.

## Figures
Spatial maps for slices 17, 16, 11, 1 and 25, 22, 8 (each by its held-out fold model): set A
(raw | plain mean of 10 | per arm: denoised realization 0 over average of 10 denoised, r = 1/3) and
set B (per arm: denoised realization 0 and average of 10 across r).

## Caveats (stated in advance)
- Single subject / acquisition.
- The old SR_SWEEP (noise injection on legacy models) is not used.
