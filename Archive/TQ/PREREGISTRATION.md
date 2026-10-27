# TQ pre-registration — reference quality vs input SNR vs noise colour

Written 2026-10-27 09:44:48, BEFORE any TQ model was trained or evaluated. Exploratory diagnostics that motivated this
are in WEEKEND_ITERATIONS.md ("TQ exploratory diagnostics") and runs/tq/diagnostics.json.

## E1 — reference (target) quality at fixed input
Dataset1-only recipe (CV_R3 folds, seed 101, 2000 iterations, p_raw 0.5), arms A (L1) and D (MSE
noise-normalized), with the leave-one-out target built from a random k-subset of the other 9 repetitions,
k in {1, 3}; k = 9 is the existing models (runs/cv_r3, runs/cv_ablation/mse_norm). The k-subset draw uses
its own generator, so batches and perturbation noise are identical to the k = 9 runs.
- PRIMARY: arm A, dataset 1, r = 1/3, a = 1, signal: paired n_eff ratio k=1 / k=9 over 34 slices;
  geometric mean, slice bootstrap 95% CI, two-sided Wilcoxon.
- SECONDARY: k = 3; arm D; a = 5, 9; r = 0; datasets 2 and 3.
- PREDICTION (Noise2Noise theory): target noise adds gradient variance but does not change the learned
  function, so the k = 1 loss is modest at a fixed budget (not a collapse to the quality of a 1-repetition
  reference).

## E2 — input SNR vs noise colour at fixed models (evaluation only)
Dataset-1 slices degraded by adding independent noise to every repetition (no renormalization, so the signal
scale is unchanged) to sigma_target = f x sigma_hat, f in {1.4, 2.0}, in three colours:
(self) white noise filtered by the slice's own radial noise spectrum; (white); (ds2) filtered by dataset 2's
median radial noise spectrum. Fixed seed. Models: dataset1-only A and D, pooled A and D (iteration 2000).
- PRIMARY: dataset1-only A, f = 2.0, a = 1, r = 1/3, signal: edge over the noise-matched blur
  (n_eff / n_eff_blur) after degradation, paired per slice between colours: (white) vs (self) and (ds2) vs
  (self), two-sided Wilcoxon, Holm over the 2.
- SECONDARY: f = 1.4; other models; a = 9; change of the edge relative to the undegraded slice.
- PREDICTION: if SNR drives the edge, the colours give similar edges at equal sigma; if colour drives it,
  fine-heavy added noise (white, ds2) lowers the edge more than (self).

## Estimator
Unchanged (split-half, unbiased; the added noise is independent across repetitions).

## No-overwrite
New outputs only under runs/tq/, runs/degrade/, Archive/TQ/; scripts refuse to overwrite; sha256 of all
existing results in Archive/TQ/protected_before.sha256, re-verified at the end.
