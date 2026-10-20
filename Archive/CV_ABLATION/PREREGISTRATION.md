# CV_ABLATION pre-registration

Written 2026-10-20 15:25:51, BEFORE any ablation model was trained.

## Question
Does the loss design of the R3 recipe matter? 2x2 over the regression norm {L1, MSE} and per-sample
weighting {none, noise-normalized}.

## Arms (identical folds, seed, batches and noise draws; only the loss differs)
- A  L1, unweighted          = CV_R3 (runs/cv_r3, reused read-only)
- B  MSE, unweighted          runs/cv_ablation/mse
- C  MSE x (sigma_ref/sigma_tot)^2   runs/cv_ablation/mse_norm
- D  L1  x (sigma_ref/sigma_tot)     runs/cv_ablation/l1_norm
sigma_ref = geometric mean of the fold's training sigma_hat (constant; keeps loss magnitude comparable).
Everything else fixed to the CV_R3 recipe: 5 folds (same assignment), 2000 iterations, p_raw 0.5,
LR 1e-3, batch 8, UNet 32/32, seed 101; evaluated weights = iteration 2000; eval ratio 0.

## Endpoints
- PRIMARY: signal region, a = 5: per-slice paired ratio n_eff(arm) / n_eff(A) over the 34 slices;
  geometric mean with 10,000-resample slice-bootstrap 95% CI; two-sided Wilcoxon signed-rank;
  Holm correction over the three comparisons B, C, D vs A at alpha = 0.05.
- SECONDARY: same at every a in {1,2,3,5,7,9} and all-pixel region; each arm vs plain averaging and
  vs oracle blur (as in CV_R3); per-band n_eff at a = 5; behaviour on slice 29.

## Hypotheses (stated in advance)
- H1: MSE >= L1 (MSE learns the posterior mean, which the MSE-based n_eff scores).
- H2: noise normalization helps quiet slices most -> negative Spearman correlation between the
  per-slice ratio (normalized / unweighted, same norm) and sigma_hat.

## Caveats (stated in advance)
- Choosing the best of four arms on the same 34 slices is mildly optimistic; a "winner" is reported as
  the best of the ablation, not as a second confirmed claim.
- CV_R3 (arm A) remains the pre-registered headline result.
