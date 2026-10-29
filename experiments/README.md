# experiments/

Scripts behind the real-ASL Noise2Noise experiments. Run them from the repo root, e.g.
`python experiments/_cv_train.py --fold 0`. Each script sets `DIFF` / `ROOT` to the repo root as an absolute path
near its top; point it at your clone. The data (`../asl_for_diffusion/dataset{1,2,3}.npy`, 34 slices × 10
repetitions × 80×80 complex each) is not part of the repository.

Every run is logged in [`../WEEKEND_ITERATIONS.md`](../WEEKEND_ITERATIONS.md). Log sections are generated from the
result files by the `_*_log.py` / `_*_report.py` scripts, so no number in them is typed by hand. Pre-registrations
are in `../Archive/<EXPERIMENT>/PREREGISTRATION.md`. What each finding means for the project:
[`../docs/RESEARCH_NOTES.md`](../docs/RESEARCH_NOTES.md) (note numbers below).

All experiments here use total-noise labels and the correct Re/Im channel layout (from R3 on).

| Experiment | Scripts | Finding (research note) |
|---|---|---|
| RUNG_LABEL | `_rung_label_check.py` | Exposed the noise-label flaw that R3 fixed (provenance) |
| R3_TOTAL | `_r3_train.py`, `_r3_eval.py`, `_r3_display.py`, `_r3_log.py`, `_split_test.py` | Retrain with total-noise labels; split-half n_eff estimator introduced |
| R4_SUBSET / R4b / R4c | `_r4_train.py`, `_r4b_train.py`, `_r4_curves.py`, `_r4_split_test.py`, `_r4_log.py`, `_r4b_log.py` | Denoise the mean, single split (note 5, history) |
| CV_R3 | `_cv_train.py`, `_cv_eval.py`, `_cv_report.py` | Pre-registered proof vs plain averaging (note 1); writes `docs/ARGUMENT_N2N_vs_averaging.md` |
| CV_ABLATION | `_cv_train.py --loss --noise-norm`, `_cv_ablation_report.py` | Loss choice (note 6) |
| CV_RATIO | `_ratio_core.py`, `_cv_ratio_eval.py`, `_ratio_report.py`, `_figures.py`, `_figures_window.py`, `_abstract.py` | Strength dial r and noise-matched blur (notes 1, 2); abstract figures and draft |
| GEN / MX_CV | `_data.py`, `_mx_train.py`, `_mx_eval.py`, `_gen_report.py`, `_mx_report.py`, `_mx_explore_6000.py` | Datasets 2–3: generalization and pooled training (notes 1, 6) |
| TQ | `_tq_diagnostics.py`, `_degrade.py`, `_cv_train.py --target-k`, `_tq_report.py`, `_tq_figures.py` | Target quality vs input SNR vs noise colour (note 4) |
| GROUND | `_gr_train.py`, `_gr_eval.py`, `_gr_report.py`, `_gr_theory.py` | k marginal (note 4); denoise the mean (note 5); closed-form r*(a) (note 3) |
| Housekeeping | `_prune_runs.py` | 2026-10-29 checkpoint pruning (rules documented in the script) |

**Shared helpers:**
- `_r1_eval.py`: data loading, magnitude and high-frequency helpers.
- `_r2_displays.py`: checkpoint loading and scoring.
- `_headtohead.py`: `Scorer`, `blur_stack`, `interp_at`.

These come from the early phase but are still imported by the scripts above.

**Queues** that ran the multi-model experiments are kept next to their outputs: `runs/mx_queue.sh`,
`runs/tq_queue.sh`, `runs/gr/queue_{a,b,eval}.sh`.

**Removed 2026-10-29:** the pre-R3 scripts (R0–R2, SR_SWEEP, FRONTIER, SRSTR_01, head-to-head and decode probes).
Their runs used legacy added-noise labels and a Re/Re–Im/Im channel pairing, so their conclusions do not carry over.
A copy is in `../../diffusion_pruned_2026-10-29.tar.gz`, outside the repository.
