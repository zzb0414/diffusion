# ASL-SM2 pre-registration — denoise the mean vs denoise each, then average, on re-reconstructed ASL

Written 2026-11-04, BEFORE any model of this study was trained and before the MEAN-SNR results exist. Code:
`experiments/_gr_train.py` (new `--sub-uniform`; noise normalization passed through `step_subset`; defaults unchanged),
`experiments/_gr_eval.py` (checkpoint name from the run's loss; l1 runs unchanged), `experiments/_sm2_report.py`.
Queue: `runs/sm2/queue.sh` (starts after `runs/mean_snr/queue.done`).

## Why
GROUND SM (2026-10-29) found no gain from denoising the mean on ASL (pooled, dataset 1, a = 9: 0.967×). Since then:
- MEAN (fastMRI) showed SUB/PER = 1.54× at a = 9, already 1.7× with 16 training images, so data size does not explain
  GROUND SM.
- On a common SNR scale (rms signal / per-channel noise; `Archive/MEAN/PREREGISTRATION_SNR.md`) ASL dataset 1 is at 0.63
  and datasets 2–3 at 0.37–0.40, vs 1.09 in MEAN, so ASL is not a decade noisier.
- GROUND SM differed from MEAN's recipe in ways that each could matter: an L1 loss (estimates a median, not the mean
  that sufficiency is about), dihedral augmentation (cost pooled PER about 10% on its own), a balanced (slice, m)
  sampler, and the old reconstruction (GIRF and CG-SENSE bugs, fixed 2026-11-03).

This study re-runs the comparison with MEAN's recipe on the new reconstruction.

## Design
- Data: ASL datasets 1–3, re-reconstructed 2026-11-03 (`asl_for_diffusion/dataset{1,2,3}.npy`), 34 slices × 10
  repetitions each. Pooled training; 5-fold CV by slice index (CV_R3 folds); every slice scored only by the model that
  held its index out.
- Arms (same U-Net 32/32, batch 8, LR 1e-3, grad clip 1, p_raw 0.5, total-noise ladder up to 0.4, no augmentation,
  3000 iterations):
  - **PER**: per-repetition N2N (LOO target), MSE, ladder floor σ̂_min. Seeds 101, 102.
  - **SUB**: subset-mean N2N (`loss_subset`, slices uniform, m ~ U{1..9}), MSE, ladder floor σ̂_min/3. Seeds 101, 102.
  - **PER-D**: PER with MSE noise-normalized, the best earlier ASL recipe for averaged maps. Seed 101.
  The November scans are excluded (ASL signal 27–41% low; `asl_for_diffusion/check_2026-11-04`).
- Evaluation (`_gr_eval.py`): split-half unbiased n_eff, signal region, a ∈ {1, 2, 3, 5, 7, 9}, all C(10, a) splits,
  r grid {0, 1/12, 1/6, 1/4, 1/3, 1/2, 1}, checkpoints 1000 / 2000 / 3000. PER decodes each repetition at σ̂; SUB decodes
  the mean of the a-half at σ̂/√a.
- Selection without ground truth: per (arm, seed, dataset, a) and held-out fold, the checkpoint and r with the lowest
  mean 1/n_eff over the OTHER folds' slices of that dataset.

## Read-outs (`_sm2_report.py`)
- **PRIMARY (Holm over 2), dataset 1, a = 9:** P1 SUB / PER; P2 SUB / PER-D. Per-slice log n_eff averaged over seeds;
  geometric-mean ratio over 34 slices, Wilcoxon, 90% bootstrap CI.
- **SECONDARY:** both ratios for every dataset and a; absolute n_eff; chosen r.
- **PREDICTIONS:**
  - P1 > 1 and P2 > 1. Sufficiency plus MEAN: ASL dataset 1 lies between MEAN's S_K = 1 and 2 on the SNR scale.
  - The ratio grows with a, and at a = 1 SUB is within 5% of PER.
  - Dataset 3 shows a smaller ratio than dataset 1. Its repetitions fail the independence gate, and correlated
    repetitions break the sufficiency argument.
- **Interpretation fixed in advance:**
  - If P1 ≤ 1 while MEAN-SNR shows SUB winning at SNR 0.36–0.55, the cause is specific to ASL: coloured or
    non-independent noise, or a single subject. It is not noise level or training-data size.
  - If P1 > 1 here but not in GROUND SM, the GROUND result came from its recipe or the old reconstruction.

## No-overwrite
Outputs only under `runs/sm2/`; training and evaluation refuse to overwrite; earlier runs are untouched.
