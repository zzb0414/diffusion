# BASE-DL pre-registration — the method vs self-supervised deep-learning baselines

Written 2026-10-30, BEFORE any of these baselines was trained or evaluated. Extends `PREREGISTRATION.md` (same
method under test, same scoring, same fixed splits, same cross-fitting); a separate primary family.

## Baselines (code: `baselines/dl_selfsup.py`; same U-Net, optimizer, batch, folds, 2000 iterations, seeds 101–103,
## every repetition used as an independent noisy image, dataset-1 training)
- **Noise2Noise (classic, Lehtinen et al. 2018):** pairs of different repetitions, no σ input, direct output, MSE.
  Existing runs `runs/dec/A/blind_s*` (DEC arm BLIND), re-scored on the baseline splits.
- **Noise2Noise + σ:** the same with σ conditioning (`runs/dec/A/dirn_s*`, DEC arm DIR_N), re-scored.
- **Neighbor2Neighbor (Huang et al. 2021):** random neighbour sub-sampler on 2×2 cells; loss
  ‖f(g₁(y)) − g₂(y)‖² + γ‖f(g₁(y)) − g₂(y) − (g₁(f(y)) − g₂(f(y)))‖², γ = 2, f(y) without gradient; sub-images
  reflect-padded to a multiple of 16 for the U-Net. Inference f(y).
- **Recorrupted-to-Recorrupted (Pang et al. 2021):** input y + αz, target y − z/α, z ~ N(0, σ̂²I), α = 0.5, MSE;
  inference the mean of f(y + αz) over 8 draws (fixed seed).
- **Noise2Self (Batson & Royer 2019):** grid masking (16 phases of a 4×4 grid), masked pixels replaced by the mean
  of their 3×3 neighbours, MSE on masked pixels only; inference f(y).
All without σ input except N2N + σ, as published. Each is scored as published (r = 0) and with our dial
(cross-fitted r), the latter as the comparison of record (our method also uses its dial).
Not implemented: DDM2 (three-stage pipeline for 4D diffusion MRI, incl. a Patch2Self stage and a diffusion model;
out of scope for 2D ASL maps) — stated as a limitation.

## Read-outs
PRIMARY: n_eff(ours)/n_eff(baseline + dial), dataset 1, a = 9, for each of the 5 baselines; geometric mean over
34 slices, 95% bootstrap CI, two-sided Wilcoxon, Holm over the 5.
SECONDARY: a = 1, 3, 5; r = 0 versions; datasets 2–3; finest band.
PREDICTION: ours beats all at a = 9. Single-image methods that assume pixel-independent noise (Neighbor2Neighbor,
Noise2Self) or white noise (R2R) are hurt by the coloured ASL noise; Noise2Noise (classic) is the closest competitor,
and beats ours for single maps (DEC A: 1.13×) but not when averaged.

## No-overwrite
Outputs only under `runs/base/` (`ne2ne/`, `r2r/`, `n2self/`, `dl_eval/`); scripts refuse to overwrite. Protected
hashes as in `PREREGISTRATION.md` (re-verified at the end).
