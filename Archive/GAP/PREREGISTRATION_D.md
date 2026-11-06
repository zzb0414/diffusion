# GAP D/E pre-registration — is ASL's small denoise-the-mean gain set by its signal-to-noise spectrum?

Written 2026-11-06, BEFORE any D/E training.

## Where we stand
- **The gain splits into two parts on both datasets.** The SUB/PER gain = a spectral part (the best linear, Wiener, filter) × a learned part (the network relative to that filter). With each design at its better recipe, the learned part is about 0.98 on fastMRI at matrices 160 and 80 (GAP C) and on ASL 1104c with SUBM (1.103 against 1.149).
- **Most of the gap is spectral.** The Wiener ratio at a = 9 is 1.273 for fastMRI at matrix 80 and 1.149 for ASL 1104c (`runs/gap/report/wiener.txt`).
- **What is ruled out.** ASL's noise does not shrink the gain (GAP B: T/W 1.029), and neither does pixel scale (C3).
- **Open.** If the spectrum is the cause, fastMRI content given ASL's per-frequency SNR should show ASL's small gain. If the gain stays high, something in ASL content beyond its spectrum matters.

## Design (`resolution/transplant.py` CONTENTS / filt / content; `transplant_train.py --content`; `runs/gap/queue_d.sh`; `resolution/spectral_report.py`)
- **Content D.** The matrix-80 fastMRI image times a radial filter |H(k)|, chosen so that the content power over the noise power, both ring-averaged population spectra, equals coil1 ASL's single-repetition SNR(k).
  - ASL SNR(k) is the signal (|FFT mean|² − Nn/N) over the noise (repetition deviations), pooled over the slices of ds1, 1104a and 1104c, with each scan's noise scaled to σ.
  - The fastMRI spectrum comes from the train split; the noise spectrum is that of the noise kind used.
  - **Check (done before training):** the Wiener SUB/PER at a = 9 is 1.149 for D-T, equal to ASL 1104c's 1.149, and 1.165 for D-W. fastMRI content F gives 1.273.
  - **Unexpected direction:** |H| is about 1.0–1.3 at k < 0.05 and 1.5–3.5 at mid and high k. Relative to fastMRI, ASL's SNR is flatter, not poorer at fine detail. The whole-band SNR was already matched in GAP B.
- **Content E.** D with uniformly random Fourier phase, seeded per image. It has the same power spectrum and the same Wiener prediction as D-W, but none of the image's phase structure: no edges and no anatomy. It is close to the Gaussian setting, where the Wiener filter is the Bayes optimum. It is not exactly Gaussian, because each image keeps its own spectrum.
- **Conditions.** DT (content D, real ASL noise T), DW (D, white noise), EW (E, white noise).
- **Fixed from GAP B:** arms PER (A), SUB and SUBM; seeds 101–102; the recipe; the noise copies (identical seeds, so noise is paired with GAP B); checkpoint and r chosen without ground truth; 15 test volumes.
- **Cost.** 18 trainings and 54 evaluations, about 5 h.
- **References (read only):** GAP B conditions T and W, `runs/gap/b/eval`.
- **Caveat.** n_eff uses each condition's own head mask (|clean| > 5% of p99). The filter changes the mask a little; for E the mask covers almost the whole FOV.

## PRIMARY (Holm over 3), a = 9, paired over the 15 test volumes
- **P1, spectrum under ASL noise:** (SUB/PER, DT) / (SUB/PER, T).
- **P2, spectrum under white noise:** (SUB/PER, DW) / (SUB/PER, W).
- **P3, phase structure at a fixed spectrum:** (SUB/PER, EW) / (SUB/PER, DW).

## Predictions and interpretation (fixed in advance)
- **Spectral hypothesis.**
  - DT's SUB/PER falls to ASL's level. I count it as reproduced if the log share of the T → ASL gap that DT closes, log(T/DT) / log(T / 1.062), is at least 0.75.
  - Equivalently, DT's SUB/PER is at most about 1.10, and P1 is about 1.062 / (SUB/PER, T) ≈ 0.86.
  - SUBM/PER under DT should be near ASL's 1.103.
  - P2 behaves like P1, with DW about 1.165 × the learned part.
  - If this holds, the ASL gap is explained: it is a denoising problem with a less favourable SNR spectrum.
- **If DT closes less than 0.4 of the gap:** the spectrum is not enough. ASL content differs in ways the spectrum misses: low-SNR perfusion structure, the non-stationary signal, or mask/background structure. A further test would then be needed.
- **P3 in [0.97, 1.03]:** the networks behave like linear filters at this SNR, because the gain does not depend on image structure. Then architecture changes aimed at the image prior should not move the SUB/PER ratio, but spectrum-aware design (noise and SNR conditioning per band) would.
  - **P3 < 0.97:** the image structure helps SUB more than PER.
  - **P3 > 1.03:** it helps PER more than SUB.
  - Either way, the prior matters, and the learned part (≈ 0.98 of Wiener) hides two opposing effects.
- **My guesses before the run:** the share closed is 0.7–1.0; P3 is about 1.0.

## Secondary (reported, not tested)
- Learned / Wiener for every condition and design.
- Absolute n_eff against the Wiener n_eff per design. Wiener n_eff is whole-image, so the comparison is approximate except for E.
- SUBM / PER per condition.
- Band transfer.

## No-overwrite
Outputs only go under `runs/gap/d/` and `runs/gap/report/summary_d.json`. Every script refuses to overwrite. GAP B code paths are bit-identical: content F = `clean`, verified, and the ckpt key `content` is set only when it is not F. The ASL data are read only.
