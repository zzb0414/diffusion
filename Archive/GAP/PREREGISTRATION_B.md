# GAP B pre-registration — noise transplant: is ASL's noise what shrinks the denoise-the-mean gain?

Written 2026-11-05, 17:30, BEFORE any B training. A1/A2 found that the scalar noise label is not the cause (P1 1.008×;
`WEEKEND_ITERATIONS.md`, GAP A1/A2). Training-set size is not the cause either: in the MEAN study, fastMRI's gain is
larger with 2 training volumes (1.71×) than with 62 (1.54×). B asks whether ASL's noise, as opposed to ASL's signal or data, is the cause.

## Design (`resolution/transplant.py`, `transplant_train.py`, `transplant_eval.py`, `transplant_report.py`; `runs/gap/queue_b.sh`)
- **Clean images.** fastMRI AXT2 at matrix 80, the ASL matrix. The MEAN-SNR recipe is unchanged: 62 training volumes, 3000 iterations, 64 crops, MSE, and evaluation on 15 test volumes.
- **Label.** σ = 1 per channel. Calibration: white noise at matrix 80 with σ = 1 has a whole-band in-head SNR of 0.18 (fastMRI S_K = 2 at matrix 160: 0.19; ASL ds1: 0.17; window share 0.19 against 0.18).
- **Noise, same σ over the whole image:**
  - **W** — white (the control).
  - **G** — Gaussian with the 2-D power spectrum of the ASL repetition deviations (colour only; finest/coarsest band power 0.52–0.64 against 1 for white).
  - **T** — the real coil1 deviation fields: colour, plus in-brain excess (in-head variance 1.12× the label), physiological structure and heavy tails (kurtosis 3.6–6.8 against 3).
  - Training fields come from ds1 + 1104a; test fields from 1104c. Each copy has a distinct slice and repetition (max copy-to-copy |corr| 0.03; white 0.02).
- **Arms.** PER (arm A), SUB, SUBM; seeds 101–102; checkpoints 1000/2000/3000; selection without ground truth as in MEAN.
- **Cost.** 18 trainings (about 10 min each) and 54 evaluations.

## PRIMARY (Holm over 2), a = 9, paired over the 15 test volumes
- P1: (SUB/PER under T) / (SUB/PER under W).
- P2: (SUB/PER under G) / (SUB/PER under W).

## Control
W at matrix 80 should reproduce a large gain (SUB/PER ≥ 1.3 at a = 9; it is 1.455 at matrix 160). If W < 1.2, the
matrix or resolution itself carries much of the gap. That is reported first and changes the reading below.

## Predictions and interpretation (fixed in advance)
- If ASL's noise is the cause: P1 ≤ 0.85, with T reproducing at least half the gap in log ratio (W's gain → ASL's 1.062).
  - Then G separates the parts. If G also drops (P2 ≤ 0.9), the cause is colour. If only T drops, the cause is the non-Gaussian, non-stationary or physiological structure.
  - Either way, the fix lies inside denoising: a noise model or training on matched noise.
- If P1 and P2 both lie in [0.95, 1.05]: ASL's noise is not the cause. What remains is the ASL signal or data. That is the "beyond denoising" case: low-SNR perfusion content, the subject/session generalization seen in ASL-SUBJ, or repetitions that are not exchangeable.
- My expectation before the run: P1 is about 0.9, with a partial drop driven by the heavy tails. This is a guess, recorded so it can be checked.

## No-overwrite
Outputs only go under `runs/gap/b/` and `runs/gap/report/summary_b.json`; every script refuses to overwrite; the ASL data is read-only.
