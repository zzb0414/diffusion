# GAP I pre-registration — noise-stationarized ASL: is the noise's spatial non-stationarity what costs denoise-the-mean?

Written 2026-11-06, 23:25, after the GAP G reading and BEFORE any I run. GAP H is running; I does not depend on H.

## Motivation
- **GAP G:** phase-randomizing real ASL raises SUB/PER at a = 9 from 1.062 to 1.131 (R1 = 1.065×) and SUBM/PER from 1.103 to 1.141 (R2 = 1.035×). That is, from learned / Wiener 0.92–0.96 to 0.98–0.99.
- **The confound G leaves:** the transform destroys both the signal's structure and the noise's spatial non-stationarity. In-brain physiological excess noise is spread over the FOV.
- **I removes only the second.**

## Design (`experiments/_ns.py`, `_ns_train.py`, `_ns_eval.py`, `_ns_report.py`; `runs/gap/queue_i.sh`)
- **The transform:** x_r' = m + U(x_r − m), per slice.
  - m is the real repetition mean.
  - U is one random k-space phase field per (dataset, slice), the same for every repetition (seed 20261108, distinct from G's).
- **Verified on 1104c:**
  - m is kept (relative difference 1e-7), and so are the deviation power (ratio 1.000000) and σ̂ (computed from the raw data, unchanged).
  - The noise variance ratio, bright 20% of pixels over the rest, goes from 1.312 to 1.002.
  - The transform is reproducible.
- **Kept:** the signal's structure and spectrum, and the noise spectrum, so Wiener is still 1.149. **Destroyed:** the noise's non-stationarity, and any spatial coupling between noise and signal.
- **Arms and evaluation:** as in G. PER, SUB and SUBM with exactly the ASL-SUBJ / ASL-COIL recipe on transformed datasets 1 + 1104a (seeds 101–102), tested on transformed 1104c.
- **Code:** `_gr_train.py`, `_gr_eval.py` and `_data.py` are unchanged. The wrappers redirect the existing `--phase-rand` hook to `_ns.apply` and write `foldall/noise_stat.json`. `_ns_eval` refuses to score models without that file. Note that `ckpt_cfg.phase_rand = True` also appears in I's checkpoints: it means "a transform was applied", and `noise_stat.json` says which one.
- **References:** `runs/subj/{PER,SUB}` and `runs/coil/subj/SUBM`, read only.

## PRIMARY (Holm over 2), a = 9, paired by slice over 34 slices
- R1: (SUB/PER, NS) / (SUB/PER, real).
- R2: (SUBM/PER, NS) / (SUBM/PER, real).

## Interpretation (fixed in advance), against G (R1_G = 1.065, R2_G = 1.035)
- **R1 ≥ 1.04, with the CI excluding 1:** the noise non-stationarity is (most of) the cost. Fix: condition on a σ map instead of a scalar σ. Next comes a σ-map architecture test.
- **R1 in [0.98, 1.02]:** stationarizing the noise does nothing. The cost in G came from the signal's structure: real ASL structure misleads SUB. The fix lies in the prior and the data (more subjects; see H).
- **Between the two bands:** both contribute; report the share as (R1_I − 1) / (R1_G − 1) and do not over-read it.
- **My expectation:** R1 ≈ 1.02–1.05, the noise half. The in-brain excess was the main noise-colour difference between scans (`asl-noise-colour-source`), and SUB's scalar label is wrong precisely in the brain.

## No-overwrite
Outputs only go under `runs/gap/i/` and `runs/gap/report/summary_i.json`.
