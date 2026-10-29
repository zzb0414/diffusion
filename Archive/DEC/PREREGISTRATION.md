# DEC pre-registration — regressor vs diffusion (A), generative sampling vs Tweedie (B)

Written 2026-10-29, BEFORE any DEC model was trained or evaluated. Motivation: `docs/RESEARCH_NOTES.md` notes 8–9
(what the method actually is; whether generative sampling recovers detail from the learned prior).

## Shared
- Dataset-1 training, CV_R3 slice-index folds, CV_R3 recipe except as stated; loss MSE, noise-normalized (the
  recommended recipe, note 6; MSE makes "posterior mean" exact); 2000 iterations; seeds 101, 102, 103.
- Every slice of datasets 1–3 is decoded by the fold model that held its index out.
- Scoring: split-half unbiased n_eff over all C(10, a) splits, signal region, a ∈ {1, 2, 3, 5, 7, 9}; per-band n_eff
  (four radial bands, finest = 0.25–0.75 cycles/pixel) at a ∈ {1, 5, 9}. Same estimator as `_ratio_core`
  (reproduction checked before use).
- Strength r on the grid {0, 1/12, 1/6, 1/4, 1/3, 1/2}. "Cross-fitted r(a)" = the grid r maximizing the summed
  normalized error over the OTHER folds' slices of the same dataset, chosen separately for every arm / decoder
  (AR_HEADROOM: one r per a is the right granularity).
- Per slice, ratios are geometric means over the 3 seeds; statistics over the 34 slices (geometric mean, slice
  bootstrap 10 000, seed 0; two-sided Wilcoxon).

## Part A — what does the diffusion framing contribute? (training)
Arms (only the stated element changes from the previous arm):
- **DSM:** current model, x̂₀ = x̃ + σ² net(x̃, σ), trained on the noise ladder (extra noise with probability 0.5).
  Seed 101 = existing `runs/cv_ablation/mse_norm`.
- **DIR_L:** direct parametrization x̂₀ = net(x̃, σ), same ladder.
- **DIR_N:** direct, native noise only (no added noise; p_raw = 1): a classic σ-conditioned N2N regressor.
- **BLIND:** DIR_N without σ input (constant level coordinate): classic N2N.
All decoded one-shot: x̂ = x̂₀ + r (y − x̂₀).

PRIMARY A (equivalence): n_eff(DIR_N)/n_eff(DSM), dataset 1, a = 9, each arm at its cross-fitted r(a); 90% CI of the
geometric mean inside [0.95, 1.05] → **equivalent** ("a σ-conditioned N2N regressor + dial is the method"); entirely
below 0.95 → regressor worse (diffusion training contributes); entirely above 1.05 → regressor better; else
inconclusive.
SECONDARY A: a = 1, 5; DIR_L/DSM (parametrization); DIR_N/DIR_L (noise-ladder training); BLIND/DIR_N (σ
conditioning), expected to matter most on datasets 2–3 whose σ̂ differ; datasets 2, 3; finest band.
PREDICTION: DIR_L ≈ DSM (same objective, reparametrized); the ladder and σ conditioning matter more than the
parametrization.

## Part B — generative sampling vs one-shot Tweedie (evaluation only, DSM models, seeds 101–103)
Decoders for one repetition y at σ̂ to end level r·σ̂ (r = 0: to σ = 0):
- **TW:** one-shot Tweedie / one DDIM step (current).
- **ODE:** deterministic DDIM walk (η = 0), 20 geometric steps from σ̂ to σ_stop = max(r·σ̂, σ_floor), then one
  DDIM step to r·σ̂ (r = 0: Tweedie at σ_stop).
- **ANC:** the same walk with ancestral noise (η = 1): one posterior sample per repetition, fixed seed.
- **ANC8:** mean of 8 such samples.
σ_floor = the model's ladder bottom (0.0368, the quietest acquisition noise): N2N cannot learn the score below the
quietest training input, so the stochastic walk stops there. Walk length is reported per slice (σ̂ / σ_floor).

PRIMARY B (two tests, Holm): n_eff(ANC)/n_eff(TW), dataset 1, a = 1, r = 0: (i) overall signal region, (ii) finest
band. Two-sided Wilcoxon.
SECONDARY B: a = 5, 9; ODE/TW; ANC8/TW; each decoder at its cross-fitted r(a); datasets 2, 3; stratification by walk
length (σ̂/σ_floor above vs below the median); an extrapolated variant walking to σ̂/10 below the floor
(off-distribution, labelled as such).
PREDICTION: ANC < TW overall (a posterior sample carries about twice the squared error of the posterior mean) and in
the finest band; ANC8 → TW; ODE ≈ TW; differences grow with walk length. If ANC wins in the finest band, report it
prominently.

## No-overwrite
New outputs only under `runs/dec/`, `Archive/DEC/`; scripts refuse to overwrite; sha256 of all existing results in
`Archive/DEC/protected_before.sha256`, re-verified at the end. Library changes are additive with the default path
verified identical (deterministic CPU) before any run.
