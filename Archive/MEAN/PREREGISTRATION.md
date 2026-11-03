# MEAN pre-registration — denoise the mean vs denoise each, then average (fastMRI, training-data scaling)

Written 2026-11-03, BEFORE any model of this study was trained (arm A at 62 volumes is the RES model, already trained and
evaluated in RES at every matrix; it is re-scored here with the new evaluator). Code: `resolution/` (train.py `--n-vol`,
models.py arm SUB, mean_eval.py, mean_report.py).

## Why
Per-repetition denoising shrinks each frequency band by its own single-copy SNR, and averaging the estimates does not undo
the shrinkage (at matrix 320, S_K = 1, every arm kept < 15% of the outer half of k-space). With i.i.d. noise the mean of
the copies is sufficient (note 5), so denoising the mean at σ/√a should keep the bands whose SNR lies between 1/a and 1
of a single copy. Known so far: NAT-SUB (natural images, ample data) +1.59 dB at a = 9; GROUND SM (ASL, 34 slices)
no gain, read as data-limited. This study measures the gain and the blur on MRI data, and how it scales with the amount
of training data — the open question for ASL.

## Design
- Data: RES fastMRI split, matrix 160, physical noise σ = 0.5 per channel (S_K = 1), N = 10 copies, test = 15 volumes ×
  8 slices (RES copies and seeds).
- Designs: **PER** = RES arm A recipe (per copy, total-noise ladder from σ, one-shot Tweedie + dial, averaged over the
  a copies). **SUB** = `N2N_DSM.loss_subset` (input = mean of a random m-subset, m ~ U{1..9}, label σ/√m; N2N target =
  mean of the other 10 − m; ladder floor σ/3, MSE, p_raw 0.5); deployed on the mean of the a copies at σ/√a, plus dial.
  Same U-Net, batch, crops, 3000 iterations, checkpoints 1000/2000/3000.
- Training data: the first 2, 8 or 62 training volumes (16, 128, 496 images; ASL dataset 1 ≈ 34 slices of 80² ≈ the
  pixel count of 2 volumes here). Seeds 101–102. PER at 62 = RES runs A_s101, A_s102 (read-only).

## Read-outs (`mean_eval.py`, `mean_report.py`)
- n_eff = σ²/MSE vs ground truth (head mask), a ∈ {1, 3, 5, 9}, fixed subsets (`baselines/_common.splits`), paired.
  Checkpoint and r chosen jointly without ground truth (split-half MSE of the other volumes). Per volume geometric mean;
  seeds averaged.
- Band transfer T1..T5 (Re⟨FFT est, FFT clean⟩/|FFT clean|² per radial band of k/k_max: 0–.1, .1–.25, .25–.5,
  .5–.75, .75–1; 1 = kept, 0 = removed).
- **PRIMARY (Holm over 2), 62 volumes, a = 9:** P1 n_eff SUB/PER (Wilcoxon over 15 test volumes, 90% bootstrap CI);
  P2 T4 (band .5–.75) SUB − PER.
- **SECONDARY:** SUB/PER at every (training volumes, a); transfer per band at a = 9 for every size.
- **PREDICTIONS:** P1 > 1 and P2 > 0 (sufficiency; NAT-SUB). SUB/PER at a = 9 grows with training data; at 2 volumes it is
  near 1 or below (GROUND SM). At a = 1 SUB ≤ PER by a few percent (NAT-SUB −0.08 dB). PER's transfer is the same at
  every a (averaging does not undo shrinkage; seen in the smoke test).

## No-overwrite
Outputs only under `runs/mean/`; scripts refuse to overwrite; RES runs are read only.
