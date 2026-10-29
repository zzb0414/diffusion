# BASE pre-registration — the method vs conventional and self-supervised baselines

Written 2026-10-29, BEFORE any baseline was evaluated. Completes roadmap step 2 (`docs/RESEARCH_NOTES.md`, notes 8–10).

## Method under test ("ours")
DSM, MSE noise-normalized, dataset-1 training, iteration 2000, seeds 101–103 (`runs/cv_ablation/mse_norm`,
`runs/dec/A/dsm_s102`, `runs/dec/A/dsm_s103`); one-shot Tweedie per repetition, blended with r, averaged. Every
slice is decoded by the fold model that held its index out.

## Baselines (code: `baselines/`)
Each baseline receives exactly the a repetitions of the split's A half (never the reference half) and returns one
estimate; its strength parameter is chosen per a by cross-fitting (below), like r for ours.
- **Plain averaging** (no parameter).
- **Gaussian blur** of the A-half mean (width 0.5–3 px).
- **Non-local means** (scikit-image) of the A-half mean, h = k·σ̂/√a, k ∈ {0.4, 0.6, 0.8, 1.0, 1.2}; Re/Im jointly.
- **BM3D** (`bm3d` 4.0.3) of the A-half mean, per channel, σ_psd = k·σ̂/√a, k ∈ {0.5, 0.75, 1.0, 1.5}.
- **MP-PCA** (dipy 1.12 `mppca`, noise level estimated) jointly over the 2a Re/Im volumes of the A half, patch
  radius ∈ {1, 2, 3}; denoised repetitions averaged.
- **LPCA** (dipy `localpca`, known σ̂), same layout and radii.
- **Patch2Self** (dipy `patch2self` version 1, OLS), same layout, patch radius ∈ {0, 1, 2}.
- **Noise2Score** (Kim & Ye 2021) re-implemented with our U-Net: a σ-conditioned score of the NOISY data learned by
  the AR-DAE objective, decoded by Tweedie x̂ = y + σ̂² s(y). No pairs used. Trained like ours (dataset 1, folds,
  2000 iterations, seeds 101–103); evaluated as published (r = 0) and with our dial r (cross-fitted).
Not implemented: DDM2 (a three-stage pipeline for diffusion MRI) and BM4D; listed as limitations.

## Scoring
Split-half unbiased n_eff, signal region, and finest-band n_eff, at a ∈ {1, 3, 5, 9}, on a fixed random subset of
10 splits per a (all 10 when C(10, a) = 10), the SAME splits for every method (paired). Each split's estimate is
unbiased, so the subset is unbiased with more variance. Parameters (r for ours, the strength for each baseline) are
chosen per a and dataset as the grid value minimizing the summed normalized error over the OTHER folds' slices.
Ours: per-slice geometric mean over seeds 101–103 (as is Noise2Score).

## Read-outs
PRIMARY: for each baseline, n_eff(ours)/n_eff(baseline), dataset 1, a = 9 (deployment: average all), geometric mean
over the 34 slices, 95% bootstrap CI, two-sided Wilcoxon; Holm over the 8 baselines.
SECONDARY: a = 1, 3, 5; datasets 2 and 3 (models trained on dataset 1 only); finest band; ranking of all methods by
geometric-mean n_eff per a.
PREDICTION: ours beats every baseline at a = 1 by a wide margin; at a = 9 the margin narrows; BM3D / NLM are the
strongest classical competitors on the mean; MP-PCA / LPCA / Patch2Self gain little over averaging because the
signal is identical across repetitions (rank one), so cross-repetition structure offers only averaging.

## No-overwrite
Outputs only under `runs/base/` and `Archive/BASE/`; scripts refuse to overwrite; protected hashes in
`Archive/BASE/protected_before.sha256`, re-verified at the end. External packages are installed outside the
repository (`../_pydeps`, versions above) and used only by `baselines/`.
