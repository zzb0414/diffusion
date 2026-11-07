# GAP L pre-registration — does the fast-learning SC recipe beat the reference recipe once both are early-stopped?

Written 2026-11-07, about 09:30, after the GAP J2 (exploratory) and K2 readings, and BEFORE any L run.

## Motivation
- GAP J2 (exploratory) found that every ASL model overfits within 3000 iterations.
  - The SC recipe (random init; unit-noise units, x / σ̂; σ_max = 5) peaks at iteration 1000 at a = 9: SUBM 31.2, SUB 29.9.
  - The reference recipe peaks at iteration 2000: SUBM 30.1, SUB 28.9.
- Those numbers use the oracle best checkpoint and r on the same test set, and come from the same seeds (101–102) that suggested the hypothesis.
- L is the confirmatory test. It uses fresh seeds and cross-fitted selection over a denser checkpoint grid.

## Design (`runs/gap/queue_l.sh`, `experiments/_l_report.py`; existing training/eval code unchanged)
- **Arms:** SUBM and SUB, each trained with:
  - SC: `_ft_train.py` without `--init`, exactly as GAP J;
  - ref: `_gr_train.py`, exactly as ASL-SUBJ / ASL-COIL, which GAP K also used.
- **Recipe:** datasets 1 + 1104a, 3000 iterations, MSE, **fresh seeds 103 and 104**.
- **Scoring:** on 1104c, at checkpoints 500, 1000, 1500, 2000, 2500 and 3000, with `_ft_eval.py --single` or `_gr_eval.py --single`.
- **Selection:** the checkpoint and r are chosen jointly by cross-fitting over slice folds (`_subj_report.chosen` with ITERS set to the six checkpoints). Split-half n_eff needs no clean reference, so this selection is deployable.
- Outputs go to `runs/gap/l/{SC,ref}_{SUBM,SUB}_s{103,104}`.

## PRIMARY (Holm over 2), a = 9, paired by slice (34)
- L1: n_eff(SC SUBM) / n_eff(ref SUBM)
- L2: n_eff(SC SUB) / n_eff(ref SUB)

## Interpretation (fixed in advance)
- **≥ 1.02, CI excluding 1:** the fast recipe with early stopping is better. It becomes the recommended ASL training recipe, with checkpoint selection by split-half n_eff.
- **(0.98, 1.02):** the recipes are equivalent once early-stopped. The J2 peak was selection optimism, and the reference stays.
- **≤ 0.98:** SC is worse; the J2 observation does not replicate.
- **Secondary:**
  - the same ratios with the 3-checkpoint grid {1000, 2000, 3000} used in all earlier reports;
  - each recipe's 6-grid / 3-grid ratio (what denser early stopping buys);
  - the fixed-iteration-3000 ratio SC / ref (the cost of not stopping);
  - a = 1.
- **My expectation:** L1 and L2 ≈ 1.00–1.04. The J2 oracle ratio was 1.037, and cross-fitting plus fresh seeds should shrink it.

## No-overwrite
Outputs only go under `runs/gap/l/` and `runs/gap/report/summary_l.json`. The training/eval scripts refuse to overwrite.
