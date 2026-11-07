# GAP K2 pre-registration — leakage control for GAP K: is the whitening gain real when the map never sees the held-out repetitions?

Written 2026-11-07, after the GAP K reading and BEFORE any K2 run. Evaluation only; no training. It runs after GAP J2, one GPU job at a time.

## Motivation
- GAP K found that whitening raises absolute n_eff at a = 9 by 1.037× for SUB and 1.030× for SUBM.
- The test-time noise map used all 10 repetitions, including the split-half reference half (B). The reference noise could therefore leak into the model's input scaling.
- PER did not gain, which argues against leakage, but it does not exclude it.

## Design (`experiments/_wh_eval_clean.py`, `experiments/_wh_clean_report.py`; `runs/gap/queue_k2.sh`)
- **Same models, same checkpoints:** the GAP K SUB and SUBM models (runs/gap/k, seeds 101–102, iterations 1000, 2000 and 3000), scored on 1104c.
- **Scoring is identical to `_gr_eval.score_slice`,** including masks, s², references, r grid and cross-fitting, with one change: for every A/B split, the model's map is rebuilt from the **A-half repetitions only**, using the same formula as `_wh.smap`.
  - a = 1 cannot give a variance, so it uses the full 10-repetition map. It is flagged as not leakage-free and is not read out.
- **Check:** with the full map substituted for every split, the new code must reproduce the GAP K neff.csv to within 1e-6. This is run before the queue, on one slice.
- **PER is not re-scored.** A per-split map for PER means a separate decode of every repetition for every split, which is expensive. PER showed no gain to explain away.
- Outputs go to `runs/gap/k2/{SUB,SUBM}_s{s}/eval_it{it}` in the `--single` neff.csv format.

## PRIMARY (Holm over 2), a = 9, paired by slice (34), cross-fitted checkpoint and r
- K2a: n_eff(SUBM, whitened, clean map) / n_eff(SUBM, real)
- K2b: n_eff(SUB, whitened, clean map) / n_eff(SUB, real)

## Interpretation (fixed in advance)
- **Clean ratios ≥ 1.02, CI excluding 1:** the K gain is real, not leakage, and the deployable recommendation stands.
- **Clean ratios in [0.99, 1.02):** most of the K gain was leakage; whitening at most helps marginally.
- **Clean ratios < 0.99:** the gain was leakage, and a map estimated from the model's own input alone is harmful.
- **Secondary:**
  - clean / leaky at a = 9 (how much the full map inflated K);
  - the same ratios at a = 5 and 7, where the map comes from fewer repetitions and is noisier.
- **My expectation:** clean is about 0.995–1.0× leaky. Under the strongest assumption, the B half is 1/10 of the variance estimate, and that estimate is then smoothed over ~200 pixels.

## No-overwrite
Outputs only go under `runs/gap/k2/` and `runs/gap/report/summary_k2.json`.

## Amendment (before any K2 queue result)
- The full-map reproduction check on slice 0 of SUB_s101 at iteration 3000 matched GAP K to 3.3e-6 relative, not 1e-6.
- The check ran on the CPU, while GAP K ran on the GPU, so this is float32 device rounding. I set the 1e-6 tolerance without considering the device.
- Accepted at 1e-5. On the same slice, clean / leaky was about 0.999 at a = 9; this was seen before the queue.
