# GAP-M pre-registration — does uneven noise cost denoise-the-mean only when its strength follows the signal?

Written 2026-11-09, BEFORE any GAP-M training. It is part of S3 (why ASL's denoise-the-mean gain is small); see `docs/ABLATION_MAP.md` §2.

## Motivation
- **GAP-I** made only ASL's noise uniform across the image, and SUB/PER at a = 9 rose from 1.062 to 1.226. So uneven noise looked like ASL's main cost.
- **GAP-B** put real ASL noise fields onto fastMRI images and the gain did *not* shrink (T/W 1.029).
- **Hypothesis (co-location).** The harm comes from noise strength that follows the signal. ASL's physiological excess sits where the perfusion signal is (in-region / rest variance 1.371 on 1104c), so a model given one σ confuses the excess with structure. In GAP-B the ASL fields came from other ASL slices; they were not tied to the fastMRI anatomy, and their in-head excess was only 1.12.
- **The closest fastMRI analogue of ASL** is content D (fastMRI filtered to ASL's per-frequency SNR, matrix 80). It gives SUB/PER = 1.540 with white noise (DW) and 1.457 with real ASL noise (DT), against ASL's 1.062. GAP-M asks how much of that gap co-located unevenness accounts for.

## Design (`resolution/transplant.py` RHO / region / envelope / wmap; `transplant_train.py --noise L|S [--whiten]`; `transplant_eval.py`; `resolution/coloc_report.py`; `runs/gap/queue_m.sh`)
- **Shared setup.** Content D, matrix 80, σ = 1 per channel, 62 training volumes, MEAN-SNR recipe, arms PER (A), SUB and SUBM, seeds 101–102. 15 test volumes; n_eff is measured against ground truth. Checkpoint and r are chosen without ground truth (`mean_report.per_volume`), as in GAP-D.
- **Noise conditions** (all use *the same white draws* as DW, multiplied by an envelope whose mean square is 1, so the σ label stays 1):
  - **DW** (reference, GAP-D, read only): no envelope.
  - **DL (co-located):** variance envelope 1 + (RHO − 1)·G₄[signal region of the image itself]. The region is defined as in the ASL estimator, gaussian(|x|, 3) > median, and G₄ is a 4-px Gaussian. RHO = 1.5 was calibrated on 60 training images to give an in-region / rest variance ratio of 1.369 (10–90%: 1.32–1.38), matching ASL 1104c's 1.371.
  - **DS (not co-located):** the same envelope rolled by half the field of view. Same shape and scale; 42% overlap with the image's own region; in-region / rest ratio 0.94.
  - **DLw (remedy):** DL with each image's copies divided by a smooth noise map estimated from those copies (GAP-K's formula, 4 px). In evaluation, the input is divided by the map and the output multiplied back; scored in original units.
- **Checks done before this pre-registration:**
  - the DW model's evaluation through the edited code reproduces GAP-D's CSV exactly (2 images, max difference 0);
  - content D is identical under noise W and L;
  - 20-iteration smoke tests of DL, DS and DLw train and evaluate.

## PRIMARY (Holm over 2), a = 9, paired by test volume (15)
- **M1** (SUB/PER, DL) / (SUB/PER, DW): does co-located unevenness of ASL's magnitude shrink the gain?
- **M2** (SUB/PER, DL) / (SUB/PER, DS): is it co-location specifically, rather than unevenness as such?

## Interpretation (fixed in advance)
- **M1 ≤ 0.95 and M2 ≤ 0.97, both CIs excluding 1:** co-location confirmed. It resolves the GAP-B / GAP-I contradiction, and the noise map (or a noise channel) is a required part of the ASL recipe.
- **M1 ≤ 0.95 but M2 within (0.97, 1.03):** unevenness costs wherever it sits. Co-location is not special, and GAP-B's null comes from its weak in-head excess (1.12).
- **M1 ≥ 0.98:** unevenness of ASL's magnitude does not cost on fastMRI content. GAP-I's effect is then ASL-specific: temporal or physiological structure of the excess, or GAP-I's lower in-region noise (0.85×). S3 must not rest on uneven noise, and the noise channel is deprioritized.
- **Secondary:**
  - SUBM versions of M1 and M2;
  - absolute n_eff DL / DW and DS / DW per design;
  - **the remedy:** (SUB/PER, DLw) / (SUB/PER, DL), and absolute n_eff DLw / DL per design. "Recovers" means SUB DLw / DL ≥ 1.02 with the CI excluding 1;
  - the log share of the DW → ASL gap (1.540 → 1.062) reproduced by DL;
  - band transfer.
- **My expectation:** M1 ≈ 0.90–0.96, M2 ≈ 0.95–0.98, and the remedy recovers part of M1's loss. If co-location alone explained GAP-I, M1 would be about 1.062 / 1.226 ≈ 0.87; I expect less, because ASL's excess also has coarse spatial correlation, which DL does not model.

## No-overwrite
Outputs only go under `runs/gap/m/` and `runs/gap/report/summary_m.json`; the scripts refuse to overwrite. The W, G, T and content F/D/E code paths are unchanged (verified bit-identical above).
