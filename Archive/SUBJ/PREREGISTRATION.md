# ASL-SUBJ pre-registration — denoise the mean vs denoise each, then average, on an unseen subject

Written 2026-11-04, BEFORE any model of this study was trained and before ASL-SM2 results exist. Code:
`experiments/_data.py` (data ids 5 = scan 1104a, 6 = scan 1104c, 7 = scan 1104b; ids 1–3 unchanged),
`experiments/_gr_train.py` (`--fold -1`: train on every slice index, run dir `foldall`, no held-out monitor; other folds
unchanged), `experiments/_gr_eval.py` (`--single`: one model scores every slice; refuses test data seen in training;
the fold column is the slice index's CV fold; per-fold mode unchanged), `experiments/_subj_report.py`.
Queue: `runs/subj/queue.sh` (after `runs/sm2/queue.done`; MEAN-TAX moved behind it).

## Why
Every ASL comparison so far trained and tested on one subject (held-out slice indices of the same anatomy). The
2026-11-04 good-coil scans add a second subject (1104c) acquired with the dataset-1 protocol and coil; its noise
(3.05e-3 of M0), signal (2.30e-3), background suppression and noise colour match dataset 1
(`asl_for_diffusion/check_2026-11-04/SUMMARY_20261104.md`). This is the first test on anatomy the model never saw.

## Design
- Training: every slice of dataset 1 (2026-09-24) and scan 1104a (2026-11-04, same subject, close slice prescription),
  68 slices × 10 repetitions. Training diversity is therefore one subject in two sessions — a deliberately strict
  cross-subject test.
- Test: every slice of scan 1104c (34 slices × 10 repetitions; another subject). Nothing of 1104c enters training.
  Its per-slice σ̂ (0.052–0.174, per-slice normalization) lies inside the training ladder.
- Arms: the ASL-SM2 recipe (U-Net 32/32, batch 8, LR 1e-3, grad clip 1, p_raw 0.5, MSE, no augmentation, 3000 iterations):
  - **PER**: per-repetition N2N, ladder floor σ̂_min. Seeds 101, 102.
  - **SUB**: subset-mean N2N, m ~ U{1..9}, ladder floor σ̂_min/3. Seeds 101, 102.
  - **PER-D**: PER, MSE noise-normalized. Seed 101.
- Evaluation: split-half unbiased n_eff on 1104c (signal region; a ∈ {1, 2, 3, 5, 7, 9}; all C(10, a) splits;
  r grid {0, 1/12, 1/6, 1/4, 1/3, 1/2, 1}; checkpoints 1000/2000/3000).
- Selection without ground truth: per (arm, seed, a) and slice fold k (CV_R3 slice-index folds), the checkpoint and r
  with the lowest mean 1/n_eff over the test subject's OTHER folds, applied to fold k (cross-fitted; split-half only).

## Read-outs (`_subj_report.py`)
- **PRIMARY (Holm over 2), 1104c, a = 9:** P1 SUB / PER, P2 SUB / PER-D. Per-slice log n_eff averaged over seeds;
  geometric mean over 34 slices, Wilcoxon, 90% bootstrap CI.
- **SECONDARY:** both ratios at every a; absolute n_eff and chosen r.
- **Exploratory:** absolute n_eff on 1104c next to ASL-SM2's dataset-1 values (different models and training sets:
  descriptive only, as a rough generalization gap).

## Predictions
- P1 > 1 and P2 > 1. 1104c sits at SNR 0.64 on the common scale, between MEAN's S_K = 1 and 2, where SUB / PER is
  1.45–1.54× on fastMRI; the margin may be smaller on ASL (coloured noise, a single training subject).
- The ratio grows with a; at a = 1 SUB is within 5% of PER.
- **Interpretation fixed in advance:**
  - P1 > 1 here and in ASL-SM2: denoise-the-mean holds on real ASL and transfers across subjects.
  - P1 > 1 in ASL-SM2 but ≤ 1 here: the SUB model does not transfer from one training subject as well as PER (its
    clean end is learned from fewer effective samples — the target-noise tax), to be retested with more subjects.
  - P1 ≤ 1 in both: as ASL-SM2's own rule (ASL-specific cause: noise colour or non-independence).
- Scan 1104a fails the repetition-independence gate (slow perfusion dip); it is used for training only, where N2N
  stays unbiased, and never for scoring.

## No-overwrite
Outputs only under `runs/subj/`; training and evaluation refuse to overwrite; earlier runs untouched.
