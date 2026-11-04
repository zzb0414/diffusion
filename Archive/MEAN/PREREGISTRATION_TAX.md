# MEAN-TAX pre-registration — can the target-noise tax of denoise-the-mean be removed?

Written 2026-11-04, after MEAN-SNR and BEFORE any model of this study was trained. Code: `n2n.N2N_DSM.loss_subset`
(new optional `m_dist`, `m_weight`; defaults verified bit-identical), `resolution/models.py` (arms SUBW, SUBM),
`resolution/train.py` (passes them), `resolution/mean_eval.py` (any arm starting with SUB decodes the mean),
`resolution/tax_report.py`. Queue: `runs/mean_tax/queue.sh` (starts after `runs/sm2/queue.done`).

## Why
MEAN-SNR: SUB / PER at a = 9 fell from 1.46× (S_K = 2) and 1.23× (S_K = 3) to 0.66× (S_K = 6). With oracle checkpoint
and r, SUB's ground-truth MSE at S_K = 6 rises from a = 5 (decode σ 1.34) to a = 9 (σ 1.0), and at equal decode noise SUB
models are worse than a model trained at that noise (σ = 1: 0.0379 vs 0.0137). Hypothesis (the "target-noise tax"):
in subset N2N the input mean of m copies is paired with the mean of the N − m others, so the clean end (large m) is
learned against the noisiest targets; the learned function stays unbiased, but its estimation variance at fixed data
and iterations grows with target noise. Two remedies, which act differently:
- **SUBW** — weight each sample by its target's precision, w = (N − m) / (N / 2): inverse-variance weighting, so noisy
  target samples stop dominating the gradient of the shared parameters (but the clean end gets less weight).
- **SUBM** — draw m with probability ∝ m: more samples at the clean end, so its noisy targets are averaged over more
  draws (the noisy end gets fewer samples).

## Design
fastMRI AXT2, matrix 160, N = 10 copies, 62 training volumes, seeds 101–102, 3000 iterations, the MEAN recipe (MSE,
p_raw 0.5, ladder floor σ/3, no augmentation). Evaluation and selection exactly as MEAN-SNR (`mean_eval.py`, checkpoints
1000/2000/3000, r chosen leave-one-volume-out on split-half MSE, n_eff vs ground truth on 15 test volumes). SUB and PER
are the existing MEAN-SNR runs (not retrained).
- **S_K = 6** (SNR 0.18): where the tax dominates. Primary.
- **S_K = 2** (SNR 0.55, closest to ASL dataset 1): secondary — a remedy must not cost where SUB already wins.

## Read-outs (`tax_report.py`)
- **PRIMARY (S_K = 6, Holm over 2):** P1 SUBW / SUB, P2 SUBM / SUB, n_eff at a = 9, per-volume geometric mean over seeds,
  Wilcoxon over 15 volumes, 90% bootstrap CI.
- **SECONDARY:** each SUB arm / PER at every a and both S_K; band transfer at a = 9; the same primaries at S_K = 2.
- **Exploratory:** oracle MSE by decode σ (is the arm monotone in a?).

## Predictions
- If the tax hypothesis is right, at least one remedy gives > 1 at S_K = 6, a = 9, and its oracle MSE falls
  monotonically with a. SUBM is the more likely (it adds samples exactly where the tax sits); SUBW may help or hurt.
- Neither remedy is expected to make SUB beat PER at S_K = 6 on n_eff (PER wins there by blurring, band transfer ≈ 0
  above k/k_max = 0.1); a ratio to PER above 0.9 would be a strong result.
- At S_K = 2 the remedies change SUB / PER by less than 5% (the tax is small there).
- **Interpretation fixed in advance:** if neither remedy exceeds 1 (CI) at S_K = 6, the tax is not a sampling/weighting
  effect at this budget (candidates left: model capacity at the clean end, data size), and the S_K = 6 reversal is
  reported as a limitation of denoise-the-mean at very low SNR. If a remedy works at S_K = 6 without costing at
  S_K = 2, it becomes the default recipe for the ASL runs after ASL-SM2 (ASL-SM2 itself is not changed).

## No-overwrite
Outputs only under `runs/mean_sk{6,2}/{SUBW,SUBM}_v62_s*`, their eval CSVs, `runs/mean_sk*/report_tax/` and
`runs/mean_tax/`; training and evaluation refuse to overwrite; MEAN-SNR runs are read only.
