# GAP J pre-registration — does a fastMRI prior, fine-tuned on ASL, raise ASL performance?

Written 2026-11-07, 02:25, after the GAP H reading and BEFORE any J run. GAP I is running; J does not depend on I.

## Motivation
- **F2 and H:** denoise-the-mean is data-hungry, and on ASL its gain is not saturated at one training subject.
- **F1:** fastMRI DT models with NO ASL data reach 0.94 (SUB) and 0.92 (SUBM) of the ASL-trained n_eff on 1104c.
- **The question:** a transferred structural prior is the lever that needs no new data. Does it help once fine-tuned?

## Design (`experiments/_ft.py`, `_ft_train.py`, `_ft_eval.py`, `_ft_report.py`; `runs/gap/queue_j.sh`)
- **Unit-noise ASL.** Each ASL slice is divided by its σ̂, so its noise is 1, the fastMRI models' label SIG. With SIGMA_MAX = 5, the `_gr_train` ladder becomes exactly the GAP D ladder (PER 1..5, SUB 1/3..5, L = 40), so the t-embedding of a fastMRI checkpoint keeps its meaning.
- **Arms**, each × PER, SUB, SUBM × seeds 101–102, so 12 trainings and 36 evaluations:
  - **FT:** initialized from the GAP D DT checkpoint at 3000 iterations, matched by arm and seed (A → PER, SUB → SUB, SUBM → SUBM). It is then trained with the ASL-SUBJ / ASL-COIL recipe (datasets 1 + 1104a, fold −1, 3000 iterations, MSE, Adam 1e-3; SUB `--ladder-div 3 --sub-uniform`, SUBM also `--m-dist linear`).
  - **SC:** the same recipe on the same unit-noise data from random init. This control isolates the initialization from the change of units.
- **Evaluation:** 1104c in unit-noise units via `_gr_eval` (unchanged and scale-free), with cross-fitted selection as in ASL-SUBJ.
- **Code:** `_gr_train.py`, `_gr_eval.py` and `_data.py` are unchanged. The wrappers patch the loader, CFG["SIGMA_MAX"] and the UNet constructor (strict `load_state_dict`). A 2-iteration smoke test showed the ladder 0.3333..5.00, sigma_max 5.0 in the saved config, and a strict load without error.
  - Disclosure: that smoke test ran on the GPU for about 10 s while GAP I was training; `CUDA_VISIBLE_DEVICES=""` was not honoured.
- **References (read only):** the original-units ASL models, `runs/subj/{PER,SUB}` and `runs/coil/subj/SUBM`.

## PRIMARY (Holm over 2), a = 9, absolute n_eff paired by slice over the 34 slices of 1104c
- J1: n_eff(SUBM, FT) / n_eff(SUBM, SC).
- J2: n_eff(SUB, FT) / n_eff(SUB, SC).

## Interpretation (fixed in advance)
- **J ≥ 1.03, with the CI excluding 1:** the fastMRI prior transfers. Pre-train and fine-tune becomes the recommended ASL recipe pending more ASL subjects. Check the secondary FT / reference to confirm the gain is not just a units artefact.
- **J in [0.98, 1.02]:** fine-tuning erases the prior (3000 iterations at lr 1e-3 is a full retraining), or the prior is irrelevant. A lower-lr or shorter fine-tune would be the follow-up, not pre-registered here.
- **J ≤ 0.97:** negative transfer.
- **Secondary:**
  - SUB/PER and SUBM/PER under FT and SC;
  - FT / reference and SC / reference (absolute n_eff);
  - PER FT / SC.
- **My expectation:** J1 ≈ 1.02–1.06. This is weak, because lr 1e-3 for 3000 iterations is long enough to wash much of the initialization out.

## No-overwrite
Outputs only go under `runs/gap/j/` and `runs/gap/report/summary_j.json`.
