# GAP K pre-registration — deployable spatial pre-whitening: does a noise map raise ASL performance?

Written 2026-11-07, 04:20, after the GAP I reading and BEFORE any K run. GAP J is running; K does not depend on J.

## Motivation
- **GAP I:** making ONLY the noise spatially stationary raises SUB/PER at a = 9 from 1.062 to 1.226, and learned / Wiener from 0.92 to 1.07. ASL's structure is exploitable once the in-brain noise excess no longer misleads a model conditioned on one scalar σ.
- **I is not deployable**, because it needs the noise to be separated from the signal. A rough residual-error comparison also suggests that SUB/SUBM gain only a few percent in absolute terms, while PER loses.
- **K is the deployable version** and is judged on ABSOLUTE n_eff in original units.

## Design (`experiments/_wh.py`, `_wh_train.py`, `_wh_eval.py`, `_wh_report.py`; `runs/gap/queue_k.sh`)
- **The map.** Per slice, take the per-pixel repetition variance (over both channels, unbiased over N), smooth it (Gaussian 4 px, `mode="nearest"`), normalize it to mean 1, and take s = √map.
- **Whitening.** Every repetition is divided by s, and the scalar label σ̂ is kept. Verified on 1104c:
  - the signal-mask / rest noise variance ratio goes from 1.371 to 1.035;
  - the whole-image variance ratio, whitened / real, is 0.989 (median);
  - map range on slice 0: 0.78–1.27.
- **Training.** PER, SUB and SUBM with exactly the ASL-SUBJ / ASL-COIL recipe on whitened datasets 1 + 1104a (seeds 101–102). `_gr_train.py` is unchanged; only the loader is wrapped.
- **Evaluation in ORIGINAL units.** Each 1104c slice's own map is computed from its 10 repetitions, which are available at test time. The model input is divided by it and the output multiplied back (`_wh.Whitened`), with `_gr_eval.score_slice` unchanged and cross-fitted selection.
  - Disclosed caveat: the map uses all 10 repetitions, including the half the split-half estimator holds out. It is a heavily smoothed second-order statistic, so any coupling is expected to be negligible, but it is not zero.
- **References (read only):** `runs/subj/{PER,SUB}` and `runs/coil/subj/SUBM`.

## PRIMARY (Holm over 2), a = 9, absolute n_eff paired by slice over 34 slices
- K1: n_eff(SUBM, whitened) / n_eff(SUBM, real).
- K2: n_eff(SUB, whitened) / n_eff(SUB, real).

## Interpretation (fixed in advance)
- **K ≥ 1.03, with the CI excluding 1:** a noise map is a deployable improvement. It becomes the recommended input normalization, and a learned σ-map conditioning channel is the architectural follow-up.
- **K in [0.98, 1.02]:** whitening with a smooth map does not capture what made I work. The non-stationarity that matters is then not a smooth variance envelope; it may be the excess noise's spatial correlation or colour, which a scalar gain cannot fix. Follow-up: colour-aware or learned conditioning.
- **K ≤ 0.97:** harmful.
- **Secondary:**
  - PER whitened / real;
  - SUB/PER and SUBM/PER;
  - absolute ratios at a = 1;
  - learned / Wiener, using the real-data value of 1.149, so this is approximate.
- **My expectation:** K1 and K2 ≈ 1.00–1.04. The I-based residual-error estimate was SUB 0.95× and SUBM 0.97× error, which is n_eff about 1.03–1.05×; whitening is a cruder fix than I.

## No-overwrite
Outputs only go under `runs/gap/k/` and `runs/gap/report/summary_k.json`.
