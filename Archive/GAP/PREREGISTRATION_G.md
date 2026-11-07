# GAP G pre-registration — phase-randomized ASL: do the ASL networks use any of ASL's spatial structure?

Written 2026-11-06, BEFORE any G training, while GAP F runs. G does not depend on F's outcome.

## Motivation
- **GAP D/E:** with the spectrum held fixed, the learned SUB/PER gain comes from the content's phase structure.
  - fastMRI anatomy: learned / Wiener 1.27–1.35.
  - Random-phase content: 0.98.
  - ASL: 0.92 (SUB) and 0.96 (SUBM), the Gaussian-like side.
- **The direct real-data check:** destroy ASL's phase structure while keeping everything else, then retrain.
- **Candidate (ii), signal-like physiological fluctuation, is bounded small.** On 1104c, PER reaches n_eff 16 at a = 1, so at most about 6% of the per-repetition variance in the signal region survives any denoiser. A fluctuation that cannot be separated from signal therefore cannot compress SUB/PER from about 1.4 to 1.06. So G replaces the multiplicative-fluctuation test planned in `PREREGISTRATION_F.md`.

## Design (`experiments/_pr.py`, `_gr_train.py --phase-rand`, `_pr_eval.py`, `_pr_report.py`; `runs/gap/queue_g.sh`)
- **The transform.** Every repetition of a slice is multiplied in k-space by the SAME uniformly random phase field, seeded per (dataset, slice).
  - **What it keeps:** a common signal, independent noise (so the split-half estimator is valid), the signal and noise power spectra (so the Wiener prediction is ASL's, 1.149), and σ̂ exactly (verified).
  - **What it destroys:** the spatial structure of the signal, and also the spatial non-stationarity of the noise. In-brain excess noise gets spread over the FOV.
- **Arms.** PER, SUB and SUBM with exactly the ASL-SUBJ / ASL-COIL recipe (datasets 1 + 1104a, fold −1, 3000 iterations, MSE, seeds 101–102), trained on phase-randomized data.
- **Evaluation.** Phase-randomized 1104c, with `_gr_eval` unchanged and selection cross-fitted as in ASL-SUBJ.
- **References (read only).** The real-ASL models, `runs/subj/{PER,SUB}` and `runs/coil/subj/SUBM`.
- **Cost.** 6 trainings and 18 evaluations, about 2 h. The queue waits for GAP F.

## PRIMARY (Holm over 2), a = 9, paired by slice over the 34 slices of 1104c
- R1: (SUB/PER, phase-randomized) / (SUB/PER, real).
- R2: (SUBM/PER, phase-randomized) / (SUBM/PER, real).

## Predictions and interpretation (fixed in advance)
- **R in [0.97, 1.03]:** randomizing the phase changes nothing. The ASL networks use none of ASL's spatial structure for the SUB/PER gain, and ASL behaves as Gaussian content would. It is then candidate (i), the content, or (iii), learnability (see GAP F).
- **R > 1.03:** real ASL is worse for denoise-the-mean than its own Gaussianized version. Something in its spatial organization hurts SUB relative to PER: noise excess sitting where the signal is, or structure that misleads. The fix would lie in modelling the non-stationary noise (a σ map instead of a scalar).
- **R < 0.97:** real ASL's structure is used and helps SUB. ASL has exploitable structure, but less than fastMRI.
- **My expectation:** R ≈ 1.03–1.08. Random-phase content reached 0.98 of Wiener on fastMRI (EW), against ASL's 0.92–0.96. So I lean towards non-stationarity hurting real ASL slightly.

## Secondary
- Learned / Wiener of the phase-randomized models: the ASL Wiener value, 1.149, still applies by construction.
- Absolute n_eff. It is comparable only loosely, because the signal-region mask moves when the phase is randomized.

## No-overwrite
Outputs only go under `runs/gap/g/` and `runs/gap/report/summary_g.json`. `_gr_train.py` defaults are unchanged; the `phase_rand` key is set only with the flag. `_gr_eval.py` and `_data.py` are unchanged.
