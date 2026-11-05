# ASL-COIL pre-registration — does pooling coils hurt denoise-the-mean, and does the MEAN-TAX fix carry to ASL?

Written 2026-11-05, BEFORE any model of this study was trained. Code: `experiments/_gr_train.py` (new `--m-dist linear`,
needs `--sub-uniform`; defaults unchanged), `train.py` (`step_subset` passes `m_dist`, default None = the existing
path), `experiments/_coil_report.py`. Queue: `runs/coil/queue.sh`. Coil naming: coil1 = dataset 1 and the 1104 scans;
coil2 = datasets 2–3.

## Why
- ASL-SM2 (pooled coil1 + coil2, held-out slices): SUB / PER 0.867× on dataset 1, a = 9.
- ASL-SUBJ (coil1 only, unseen subject 1104c): SUB / PER 1.062×.
- The two differ in training data AND test set, so they do not say whether pooling hurts SUB. Datasets 1–3 carry three
  different noise colours (finest/coarsest band power 0.66 / 1.48 / 0.53) while the network is conditioned on one scalar σ.
- MEAN-TAX: drawing m ∝ m (SUBM) removes part of the target-noise tax on fastMRI (1.126× over SUB at S_K = 6, +3.5% at
  S_K = 2). Its pre-registered rule makes SUBM the default denoise-the-mean recipe for ASL runs after ASL-SM2; whether
  it helps on ASL is untested.

## Design
- **Part A (pooling):** PER, SUB, SUBM trained on dataset 1 only, 5-fold CV by slice index (CV_R3 folds), seed 101,
  3000 iterations, scored on dataset 1. Compared slice-by-slice with ASL-SM2's seed-101 PER and SUB (pooled on
  datasets 1–3, same folds, same recipe, same evaluation). Single seed: the cost of 5 folds × 3 arms.
- **Part B (fix on an unseen subject):** SUBM trained exactly as ASL-SUBJ's SUB (all slices of dataset 1 + 1104a,
  seeds 101–102), scored on 1104c, compared with ASL-SUBJ's PER and SUB (read only).
- Recipe as ASL-SM2: MSE, p_raw 0.5, no augmentation, ladder floor σ̂_min (PER) / σ̂_min/3 (SUB, SUBM), m ∈ 1..9.
- Evaluation and selection as ASL-SM2 / ASL-SUBJ: split-half n_eff, signal region, a ∈ {1, 2, 3, 5, 7, 9},
  checkpoints 1000 / 2000 / 3000; checkpoint and r chosen on the other slice folds by lowest mean 1/n_eff.

## Read-outs (`_coil_report.py`)
- **PRIMARY (Holm over 3), a = 9:**
  - P1 pooling interaction on dataset 1: (SUB / PER, dataset-1-only) ÷ (SUB / PER, pooled), per slice.
  - P2 SUBM / SUB on 1104c (seeds 101–102).
  - P3 SUBM / SUB on dataset 1, dataset-1-only CV.
- **SECONDARY:** every ratio at every a; SUB / PER and SUBM / PER dataset-1-only; PER and SUB dataset-1-only ÷ pooled
  (does pooling help PER and hurt SUB, or hurt both?); absolute n_eff and chosen r.
- **PREDICTIONS:**
  - P1 > 1 (coil-mixing hypothesis), with dataset-1-only SUB / PER between 1.0 and 1.1 (ASL-SUBJ gave 1.06).
  - P2 and P3 ≥ 1 but small (at most +5%): ASL dataset 1 / 1104c sit at SNR ≈ 0.63, near fastMRI S_K = 2 where SUBM
    added 3.5%. Not reaching significance would not contradict MEAN-TAX.
- **Interpretation fixed in advance:**
  - P1 > 1 (CI) and dataset-1-only SUB / PER ≥ 1: pooling data with different noise colour hurts denoise-the-mean;
    ASL-SM2's deficit is a data-mixing effect. Follow-up: coil-wise training or colour conditioning, before the
    coil1/coil2 blend experiment.
  - P1 CI includes 1, or dataset-1-only SUB / PER < 1: pooling is not the cause. The SM2–SUBJ difference is then the
    test setting (training subject's held-out slices vs an unseen subject) or seed noise, and the per-band SNR analysis
    (below) becomes the lead for the small ASL gain.
  - P2 or P3 > 1 (CI): SUBM is confirmed as the ASL default. Both ≤ 1: the tax is not the bottleneck on ASL at this
    SNR; SUB stays the reference recipe.
- **Exploratory (CPU, no GPU), reported separately:**
  - Per-band SNR (signal power / noise power per radial frequency band) for ASL datasets 1–3 and 1104c vs fastMRI at
    S_K = 1, 2, 3, 6: does ASL sit at a lower SNR in the mid/fine bands than its overall SNR suggests?
  - Why datasets 2 and 3 (same coil, same session) differ in noise colour (1.48 vs 0.53).

## No-overwrite
Outputs only under `runs/coil/`; training and evaluation refuse to overwrite; ASL-SM2 and ASL-SUBJ runs are read only.
