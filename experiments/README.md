# experiments/

Scripts behind the real-ASL Noise2Noise experiments. Run them from the repo root,
e.g. `python experiments/_r3_train.py`. Each script sets `DIFF` to the repo root
as an absolute path near its top — point it at your clone. The data
(`../asl_for_diffusion/dataset1.npy`, 34 slices × 10 realizations × 80×80
complex) is not part of the repository.

Every run is logged in [`../WEEKEND_ITERATIONS.md`](../WEEKEND_ITERATIONS.md);
log sections are generated from the result files by the `_*_log.py` /
`_*_report.py` scripts, so no number in them is typed by hand.

## Current experiments (total-noise labels, correct channel layout)

| script | what it does |
|---|---|
| `_rung_label_check.py` | Diagnostic that exposed the noise-label flaw: decodes a legacy model at the "added-noise" vs "total-noise" reading of its rung label. |
| `_r3_train.py` | **R3** — `N2N_DSM.loss_multireal(sigma_y=...)`: one realization in, LOO mean of the other 9 as target, conditioned on total noise. |
| `_r3_eval.py`, `_r3_display.py`, `_r3_log.py` | R3 matched-noise evaluation vs Gaussian blur, single-vs-ensemble displays, log section. |
| `_split_test.py` | Split-half test (all C(10,a) subsets; unbiased error vs the clean map from the independent half; n_eff = σ²/MSE). |
| `_r4_train.py`, `_r4_curves.py`, `_r4_split_test.py`, `_r4_log.py` | **R4** — subset-averaging model `N2N_DSM.loss_subset` (denoise the mean of m realizations); 4-way split test with a per-frequency-band resolution check. |
| `_r4b_train.py`, `_r4b_log.py` | R4b/R4c — balanced noise-level coverage, + dihedral augmentation. |
| `_cv_train.py`, `_cv_eval.py`, `_cv_report.py` | **CV_R3** — the pre-registered 5-fold cross-validated proof over all 34 slices; writes [`../docs/ARGUMENT_N2N_vs_averaging.md`](../docs/ARGUMENT_N2N_vs_averaging.md). `_cv_train.py --loss/--noise-norm` also drives the loss ablation. |
| `_cv_ablation_report.py` | CV_ABLATION — {L1, MSE} × {unweighted, noise-normalized}, paired over the 34 held-out slices. |

Shared helpers used by the above: `_r1_eval.py` (data loading, magnitude / HF
helpers), `_r2_displays.py` (checkpoint loading, scoring), `_headtohead.py`
(`Scorer`, `blur_stack`, `interp_at`).

## Superseded

Earlier R0–R2 / SR_SWEEP / FRONTIER scripts are kept locally but not committed:
those runs used the legacy added-noise labels and a training data layout that
paired Re/Re and Im/Im channels (see the log), so their conclusions do not carry
over.
