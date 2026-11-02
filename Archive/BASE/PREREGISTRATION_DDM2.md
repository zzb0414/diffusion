# BASE-DDM2 pre-registration — a DDM2-like baseline on ASL

Written 2026-11-02, BEFORE this baseline was trained or evaluated. DDM2 (Xiang et al., ICLR 2023) is the closest
published self-supervised diffusion denoiser for MRI and the likely reviewer reference. Same method under test, splits,
scoring and cross-fitting as `PREREGISTRATION.md`.

## DDM2-like pipeline (code `baselines/ddm2.py`; DDM2's three stages)
- **Stage I (first estimate):** a self-supervised CNN denoiser. Here: the classic Noise2Noise model (no σ input, direct
  output; DEC arm BLIND, `runs/dec/A/blind_s<seed>`, same folds and seeds). **Deviation:** DDM2's stage I predicts each
  volume from the OTHER volumes (Patch2Self-like). Under our protocol a method sees only the A half of a split, and at
  a = 1 there are no other copies, so stage I denoises each repetition from itself.
- **Stage II (place the input on the noise ladder):** per repetition, σ_m = standard deviation of the residual
  y − Φ_I(y) (DDM2 fits a Gaussian to this residual and matches it to a chain step).
- **Stage III (diffusion on pseudo-clean data):** a score network (same U-Net, score parametrization, levels
  0.005–0.4) trained by denoising score matching on stage-I outputs of the training slices as clean data (x = Φ_I(y),
  input x + σε, target x; ε-prediction weighting), dataset-1 training, 2000 iterations, seeds 101–103, CV_R3 folds.
  Inference: reverse diffusion from the matched level σ_m with the noisy repetition as the chain state, 20 ancestral
  steps to σ = 0.005, then the posterior-mean step to 0.
- Variants: **DDM2** (one ancestral sample, as published); **DDM2-TW** (stage-III network, one-shot Tweedie at σ_m,
  no sampling: isolates the training signal, pseudo-clean targets vs our Noise2Noise pairs). Both scored as published
  (r = 0) and with our dial (cross-fitted r, blend with the input).

## Read-outs (fixed splits; per-slice geometric mean over seeds)
- PRIMARY: n_eff(ours)/n_eff(DDM2 + dial), dataset 1, a = 1 and a = 9; two-sided Wilcoxon, Holm over 2.
- SECONDARY: DDM2 at r = 0; DDM2-TW (training-signal effect: ours/DDM2-TW); finest band; datasets 2–3.
- PREDICTION: ours > DDM2 at both a (sampling doubles error, P4; stage-I bias propagates). DDM2-TW ≥ DDM2 (sampling
  hurts). Ours vs DDM2-TW: pseudo-clean targets carry the stage-I bias, so ours ≥ DDM2-TW, most of all when
  averaging.

## No-overwrite
Outputs only under `runs/base/ddm2/`, `runs/base/ddm2_eval/`; scripts refuse to overwrite.
