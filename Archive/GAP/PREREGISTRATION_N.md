# GAP-N pre-registration — at a fixed number of training images, does the number of subjects decide the denoise-the-mean gain?

Written 2026-11-09, BEFORE any GAP-N training. It is part of S3 (why ASL's denoise-the-mean gain is small); see `docs/ABLATION_MAP.md` §2. It follows GAP-M, where uneven noise was not the cause on fastMRI content.

## Motivation
- **GAP-F (fastMRI, content D, real ASL noise DT)** varied the number of training volumes, each with 8 slices: 62 → 4 → 1 volumes gave SUB/PER at a = 9 of 1.457 → 1.301 → 0.885. That varied image count and subject count *together*.
- **GAP-H (ASL)** found the gain data-limited. ASL trains on about 1–2 subjects, with 68 slices.
- **GAP-F caveat:** in image count, ASL sits between 4 and 62 fastMRI volumes, yet its gain (1.06) is below the 4-volume value (1.30). So either subject diversity, not image count, is what matters, or ASL content differs.
- **Practical question:** when more ASL data come in, should we collect more subjects, or are more scans and slices of the same subjects as good?

## Design (`resolution/transplant_train.py --one-slice`; `resolution/subjects_report.py`; `runs/gap/queue_n.sh`)
- **Shared setup:** content D, noise T (real ASL fields), matrix 80, GAP-D/F recipe unchanged, arms PER (A), SUB and SUBM, seeds 101–102, checkpoints 1000/2000/3000.
  - 15 test volumes; n_eff is measured against ground truth.
  - Checkpoint and r are chosen without ground truth (`mean_report.per_volume`).
- **`--one-slice`:** one slice per training volume, at slice position v mod 8 of the 8 cached central slices.
  - Each image keeps its original index, so its content filter and noise draws are identical to the GAP-D/F runs.
  - Slice-position coverage matches the 8-slice designs exactly: 8×1 has each position once, 32×1 each position 4 times.
- **Conditions** (volumes × slices per volume):

  | Images | Few subjects | Many subjects |
  |---|---|---|
  | 8 | 1×8 (GAP-F DTn1, read only) | **8×1** (new) |
  | 32 | 4×8 (GAP-F DTn4, read only) | **32×1** (new) |
  | 62 vs 496 | — | **62×1** (new) vs 62×8 (GAP-D DT, read only) |

- **Checks done before this pre-registration:**
  - the default path is unchanged: the kept indices equal the old prefix `vol < n_vol` (asserted);
  - the one-slice indices are as intended;
  - a 20-iteration smoke test trains and evaluates.

## PRIMARY (Holm over 2), a = 9, paired by test volume (15)
- **N1** (SUB/PER, 32×1) / (SUB/PER, 4×8): does diversity matter at 32 images?
- **N2** (SUB/PER, 8×1) / (SUB/PER, 1×8): the same at 8 images.

## Interpretation (fixed in advance; read from N1, with N2 as the low-data check)
- **N1 ≥ 1.05, CI excluding 1:** subject diversity drives the gain at a fixed image count. New ASL data should add **subjects**; more scans or slices of existing subjects are worth less. This supports the "ASL is effectively a 1–2-volume regime" reading of GAP-F.
- **N1 within (0.97, 1.03):** image count, not diversity, is what matters. More slices or scans of the same subjects are as useful, and ASL's low gain then needs another explanation (content).
- **N1 ≤ 0.95:** within-subject coherence helps (unexpected); report as is.
- **Other values:** a weak effect; reported as inconclusive.

## Secondary
- Absolute n_eff ratios per design for both pairs (the deployable question: which data give the best model).
- **N3, count at fixed diversity:** (SUB/PER, 62×1) / (SUB/PER, 62×8), and absolute n_eff. At 62 subjects, how much do 7 more slices per subject add?
- The full table of SUB/PER (a = 1, 3, 5, 9), SUBM/PER and n_eff for all six conditions.

## My expectation
N1 ≈ 1.05–1.12; N2 larger (1.1–1.3), since one volume is the harshest case; N3 ≈ 0.90–0.97.

## Limits for transfer to ASL
- **Within-subject redundancy is probably higher here than on ASL.** The 8 cached slices are adjacent central slices, while ASL's 34 slices cover the whole brain.
- **ASL's second scan of a subject (same slices, new noise) is not modelled.**
- So GAP-N bounds the value of diversity against *adjacent-slice* redundancy only.

## No-overwrite
Outputs only go under `runs/gap/n/` and `runs/gap/report/summary_n.json`; the scripts refuse to overwrite. GAP-D/F outputs are read only.
