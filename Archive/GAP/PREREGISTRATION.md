# GAP pre-registration — why is ASL's denoise-the-mean gain small?

Written 2026-11-05, BEFORE any of these evaluations ran. On the unseen subject 1104c, denoise-the-mean beats denoise-each
by 1.06× (SUB) / 1.10× (SUBM) at a = 9, against 1.46× on fastMRI at a matching per-band SNR (S_K = 2;
`runs/coil/band_snr.txt`). The question is whether the gap is a denoising problem we can fix (noise model) or something
beyond denoising (the signal or the noise of ASL itself). Tests in order of cost; each gets its own report section.

## A1 / A2 — the noise label (evaluation only; `experiments/_scale_eval.py`, `_gap_report.py`, `runs/gap/queue_scale.sh`)
- Hypothesis: the scalar label σ̂ (whole-image average) under-states ASL's in-brain noise (1.2–1.5× in variance) and
  its physiological component. Denoise-each can absorb a wrong label through the dial r; denoise-the-mean picks r ≈ 0
  and decodes the mean at σ̂/√a, so it would be hurt more.
- Models (read only): ASL-SUBJ's PER and SUB, ASL-COIL's SUBM, seeds 101–102, trained on dataset 1 + 1104a.
  Test: 1104c. Iterations 1000 / 2000 / 3000.
- Decoder label c·σ̂ (PER) or c·σ̂/√a (SUB, SUBM); c ∈ {0.5, 0.65, 0.8, 1, 1.2, 1.4, 1.7, 2} chosen without ground truth jointly
  with iteration and r (cross-fitted over slice folds, as ASL-SUBJ), and c = "brain" (in-signal / whole-image noise
  std ratio of the slice; untuned).
- **PRIMARY (Holm over 2), a = 9:** P1 SUB tuned c / SUB c = 1; P2 SUB / PER with both tuned.
- **PREDICTIONS (if the label is the cause):** SUB's chosen c ≥ 1.2; P1 > 1.05; P2 above the c = 1 value (1.06);
  PER gains less than SUB from tuning. "brain" moves SUB in the same direction, less.
- **Interpretation fixed in advance:** P1 ≤ 1.02 with c ≈ 1 chosen: the label is not the cause, go to B. P1 > 1.05:
  the noise model is part of the gap (a fix inside denoising: per-slice or spatial noise labels).

## B — noise transplant (training; pre-registered separately before it runs)
fastMRI at S_K = 2 with white noise replaced by real ASL repetition deviations at the same per-band SNR: (1) background
deviations only (recon-shaped thermal noise), (2) in-brain deviations (plus the physiological coarse component).
If the gain falls toward ASL's only in (2), the physiological noise is the cause; if in neither, the cause is the ASL
signal or data (beyond the noise model).

## No-overwrite
Outputs only under `runs/gap/`; evaluation refuses to overwrite; trained models are read only.

## Amendment (2026-11-05, before the run)
A single-slice timing check (1104c slice 15, SUB s101, iteration 3000, r = 0) gave n_eff 22.3 / 20.2 / 15.9 at
c = 0.8 / 1 / 1.2, i.e. better with a SMALLER label. The c grid is extended down to 0.5 and 0.65 before any full run.
The P1 reading becomes two-sided: a chosen c < 1 with P1 > 1.05 means the label over-states the noise of what is decoded
(over-smoothing), which is also a fix inside denoising.
