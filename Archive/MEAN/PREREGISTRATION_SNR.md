# MEAN-SNR pre-registration — denoise the mean vs denoise each, at ASL-like noise (fastMRI)

Written 2026-11-04, BEFORE any model of this study was trained. Code unchanged from MEAN except `mean_report.py`, which
now takes the noise scale from `RES_SK` (outputs `runs/mean_sk<S_K>/`; the S_K = 1 path verified unchanged by its dry
run). Queue: `runs/mean_snr/queue.sh`.

## Why
MEAN (S_K = 1) found SUB/PER = 1.54× at a = 9 with 62 training volumes, and 1.7× with only 2. So training-data size does
not explain why SUB did not win on ASL (GROUND SM). The remaining candidate is the noise level: on natural images SUB
lost at σ 2.0 (CLAIMS C7). One SNR definition for both data types (rms signal in the brain/head mask, |x| > 25% of its
p99 or |M0| > 25% of its p99, divided by the per-channel noise σ):

| data | per-copy SNR |
|---|---|
| fastMRI, matrix 160, S_K = 1 (MEAN) | 1.09 |
| ASL dataset 1 | 0.63 |
| ASL datasets 2 / 3 | 0.37 / 0.40 |

(ASL: `asl_for_diffusion/check_2026-11-04/compare_stats.txt`, brain noise; the "0.04–0.13" in RESEARCH_NOTES is a
different definition, a per-pixel median.) This study places MRI points on the regime map at ASL-matched SNR and beyond.

## Design
As MEAN (`PREREGISTRATION.md`), except:
- Noise scale S_K ∈ {2, 3, 6}: σ(160) = 1.0, 1.5, 3.0 per channel; per-copy SNR ≈ 0.55 (ASL dataset 1), 0.36 (ASL
  datasets 2–3) and 0.18 (beyond ASL, to find the reversal). Ladders scale with σ (PER floor σ, SUB floor σ/3, top 5σ).
- Training data: all 62 training volumes (496 images); both PER (arm A) and SUB trained fresh at each S_K; seeds 101–102.
- Same evaluation (`mean_eval.py`), selection (checkpoint and r by leave-one-volume-out split-half) and read-outs.

## Read-outs
- **PRIMARY (Holm over 2):** n_eff SUB/PER at a = 9 at S_K = 2 and at S_K = 3 (the ASL-matched levels); Wilcoxon over
  15 test volumes, 90% bootstrap CI.
- **SECONDARY:** S_K = 6; every a; band transfer T1..T5 at a = 9; P2-type T4 difference at each S_K.
- **PREDICTIONS:**
  - SUB/PER > 1 at S_K = 2 and 3, and smaller than at S_K = 1 (1.54×). The Gaussian theory's gap peaks at intermediate
    SNR (THEORY P5) and natural images reversed at heavy noise.
  - At S_K = 6, SUB/PER ≤ 1.1, possibly < 1 (the reversal).
  - At a = 1, SUB ≈ PER (within 5%) at every level.
  - If SUB/PER > 1.2 at S_K = 3, the noise level does not explain the ASL result either. The explanation then lies in
    the ASL data or recipe (coloured or non-independent noise, the old L1/no-floor recipe, the old reconstruction), and
    the re-run on re-reconstructed ASL becomes decisive.

## No-overwrite
Outputs only under `runs/mean_sk2/`, `runs/mean_sk3/`, `runs/mean_sk6/`; train and eval refuse to overwrite; MEAN and RES
runs are untouched.

## Correction to PREREGISTRATION.md (2026-11-04)
The MEAN design listed 8 training volumes as 128 images; the trained models used 64 (8 slices per volume), as the run
configs record. Nothing else changes.
