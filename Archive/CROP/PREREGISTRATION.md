# ASL-CROP pre-registration — does training on crops help denoise-the-mean on ASL?

Written 2026-11-06, BEFORE any crop training.

## Motivation
GAP C (`WEEKEND_ITERATIONS.md`) found on fastMRI that the share of the field of view covered by a training crop drives the SUB/PER ratio, with the pixel budget per step fixed.
- **Larger crops, fewer patches:** help PER and hurt SUB.
  - matrix 80: PER 78 → 85, SUB 106 → 103;
  - matrix 160: PER 104 → 109, SUB 151 → 140.
- **ASL** trains on whole 80 × 80 slices (crop = FOV, batch 8), the regime that is worst for SUB.

## Design (`experiments/_gr_train.py --crop 32 --batch 32`, `experiments/_crop_report.py`, `runs/crop/queue.sh`)
- **Arms.** PER, SUB and SUBM, each with exactly the ASL-SUBJ / ASL-COIL recipe: datasets 1 + 1104a, fold −1, 3000 iterations, MSE, seeds 101–102. SUB and SUBM add ladder-div 3 and sub-uniform; SUBM adds m-dist linear.
- **The change.** Random 32 × 32 crops with batch 32, the same pixels per step as 8 × 80². A 32 crop covers 0.4 of the FOV, as fastMRI's best SUB recipe does.
- **Evaluation.** `_gr_eval.py --single` on 1104c, with selection as in ASL-SUBJ.
- **References (read only).** `runs/subj/{PER,SUB}`, `runs/coil/subj/SUBM`.

## PRIMARY (Holm over 3), a = 9, absolute n_eff paired over the 34 slices
- P1: SUBM crop / SUBM whole.
- P2: SUB crop / SUB whole.
- P3: PER crop / PER whole.

## Predictions (from fastMRI, fixed in advance)
- SUB and SUBM gain (P1, P2 > 1; fastMRI at matrix 80: +3.7%). ASL's change is larger (FOV share 1.0 → 0.4, against 0.8 → 0.4), so I expect +3 to +8%.
- PER loses (P3 < 1; fastMRI −8%).
- SUBM crop / PER whole, the application comparison with each design at its fastMRI-better recipe, rises above the current 1.13.

## Interpretation
- **P1 > 1.03:** the crop recipe is a real-data lever, and SUBM-crop becomes the ASL candidate default.
- **P1 within [0.97, 1.03]:** the fastMRI crop effect does not carry to ASL, perhaps because of the limited training slices or ASL's spectrum.
- **P3 < 1 with P1 > 1:** confirms the two designs need different training recipes, which bears on any comparison between them.

## No-overwrite
Outputs only go under `runs/crop/`; the references are read only. `_gr_train.py` defaults are unchanged: crop and batch are set only when given.
