# GAP C pre-registration — why does a coarser matrix shrink the denoise-the-mean gain?

Written 2026-11-06, BEFORE any C training. GAP B found that fastMRI's SUB/PER gain at a = 9 falls from 1.455 (matrix 160)
to 1.205 (matrix 80) at the same per-band SNR (white noise), and that ASL's noise does not shrink it further.

## C0 — linear theory (computed before this pre-registration; exploratory, `resolution/wiener.py`, `runs/gap/report/wiener.txt`)
- **Method.** The best *linear* denoiser (per-frequency Wiener from the population power spectrum, dial r as the networks) predicts SUB/PER at a = 9:

  | data | Wiener, a = 9 |
  |---|---|
  | fastMRI 160 | 1.423 |
  | fastMRI 80 | 1.273 |
  | ASL ds1 | 1.169 |
  | ASL 1104a | 1.114 |
  | ASL 1104c | 1.149 |
  | ASL ds3 | 1.245 |
  | ASL ds2 (VSS) | 1.178 |

- **Validation.** The noisy-data estimator used for ASL recovers fastMRI 80's clean-truth value from noisy copies (1.264 against 1.273). A first per-slice clipped version was badly biased (1.03) and is discarded.
- **Reading.** The gain factors into two parts:
  - a **spectral part**, the second-order statistics of the signal against the noise;
  - a **learned part**, learned / Wiener: 1.02 at matrix 160, 0.95 at matrix 80, and 0.94 (SUB, tuned) / 0.98 (SUBM) on ASL 1104c.
- **Split of the matrix effect, in log ratio.** About 60% is spectral and 40% is the learned part.
- **Split of fastMRI 160 → ASL 1104c (1.455 → 1.062).** About 68% is spectral and 32% is learned. So most of ASL's small gain is predicted by ASL's own spectrum and is not "beyond denoising". The learned part, the network's non-linear edge, is lost at 80 × 80 for both fastMRI and ASL.

## C1/C2 — what removes the learned part at matrix 80? (training; `transplant_train.py --crop --batch --grid`, `train.py --crop --batch`)
- **Common setting.** White noise σ = 1, PER (arm A) and SUB, seeds 101–102, 62 training volumes, 3000 iterations, and pixels per training step fixed at 8 × 64² by trading crop against batch.
- **Candidates:**
  - **crop/context** — 64-pixel crops cover most of an 80 image;
  - **pixel scale** — at 80 × 80 a structure spans half as many pixels, so the U-Net's fixed pixel receptive field and down-sampling see coarser anatomy.
- **Conditions:**
  - **80c32** — matrix 80, crop 32, batch 32 (the crop/image ratio of matrix 160).
  - **160c128** — matrix 160, crop 128, batch 2 (the crop/image ratio of matrix 80).
  - **U80** — matrix-80 content zero-padded in k-space onto a 160 grid, crop 64. The information and the Wiener ratio are identical to matrix 80; only the pixel scale changes. The noise stays white within the band and has a per-pixel std of 1.
- **References (read only).** 160 = MEAN-SNR S_K = 2; 80 = GAP B W.

## PRIMARY (Holm over 3), a = 9, paired over the 15 test volumes (ratios of SUB/PER)
- P1: 80c32 / 80.
- P2: U80 / 80.
- P3: 160c128 / 160.

## Predictions and interpretation (fixed in advance)
- **Full restoration of the learned part** at 80 means a ratio of about 1.08 (= 1.02 / 0.95).
- **P2 ≥ 1.04: pixel scale matters.** The network's edge depends on how many pixels a structure spans, which is an architecture lever for ASL: up-sampling the input, or a receptive field and down-sampling matched to the anatomy.
- **P1 ≥ 1.04, or P3 ≤ 0.96: crop and context fraction matter**, a training-recipe lever.
- **All three within [0.97, 1.03]:** neither lever matters; the learned deficit at 80 belongs to the information itself.
- **My guess before the run:** P2 about 1.05, P1 and P3 about 1.

## No-overwrite
Outputs only go under `runs/gap/c/` and `runs/gap/report/summary_c.json`; every script refuses to overwrite; the reference runs are read only.
Defaults of the edited scripts are unchanged: the new keys are written to `ckpt_cfg` only when set.
