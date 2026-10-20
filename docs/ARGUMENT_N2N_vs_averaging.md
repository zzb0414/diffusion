# Averaging denoised acquisitions beats averaging raw ones

*Noise2Noise denoising of mandated repeat acquisitions — evidence summary. Generated from the experiment CSVs by `experiments/_cv_report.py`; every number below is computed, not typed.*

## The claim

ASL protocols acquire several noisy repetitions of every slice and average them; the noise of a plain average falls only as 1/a with the number of repetitions a. Because those repetitions are mandated anyway, they are free training data: a Noise2Noise network learns to denoise one repetition using the others as targets, with no clean reference required. Denoising each repetition and then averaging gives an estimate of the clean map that is far better than the plain average of the same repetitions.

## Headline result

With 5 of the 10 repetitions, the average of denoised repetitions is as accurate as a plain average of **6.0×** as many repetitions (95% CI 5.23–6.85), on **every one of 34 slices**, each scored by a model that never saw that slice (p = 5.8e-11). This was the pre-registered primary endpoint.

**Read this together with the baseline.** An idealised Gaussian smoother of the plain average, tuned on each scored slice, already reaches 5.26× at the same point — 93% of the network's gain on a log scale. The network's own contribution beyond that idealised smoother is 13–25% when up to 5 repetitions are used, and it comes from removing coarse-scale noise that a Gaussian smoother cannot reach (see below). In practice a smoother cannot be tuned on the answer, so the real-world gap to Gaussian smoothing is larger than this conservative figure.

## How it was measured

- **No clean ground truth is needed.** The 10 repetitions of a slice are split into a set A used to build the estimate and a disjoint set B used only as a reference. Because B's noise is independent of A, the error against mean(B) equals the error against the true clean map plus B's own noise variance, which is known from the spread of the repetitions. Subtracting it gives an unbiased error against the clean map. All possible splits are averaged.
- **Metric:** n_eff = σ² / MSE, the number of repetitions a plain average would need for the same error. A plain average of a repetitions scores exactly a.
- **No leakage:** 5-fold cross-validation over all 34 slices; every slice is scored only by the model trained without it. The training recipe and every analysis choice were fixed in a pre-registration before any of these models existed.
- **A deliberately strong baseline:** a Gaussian blur of the plain average whose width is tuned *on the very slice being scored*. No real smoother can do better than this.
- **Statistics treat slices as the replicates:** geometric-mean ratios with bootstrap-over-slices confidence intervals and Wilcoxon signed-rank tests.

## Results

| scans averaged (a) | R3 vs plain averaging | R3 vs best-case blur | best-case blur vs plain |
|---:|---|---|---|
| 1 | 22.80× (95% CI 19.29–26.45; 34/34 slices; p = 5.8e-11) | 1.24× (95% CI 1.06–1.38; 30/34 slices; p = 1.2e-06) | 18.38× |
| 2 | 13.26× (95% CI 11.47–15.38; 34/34 slices; p = 5.8e-11) | 1.25× (95% CI 1.10–1.36; 30/34 slices; p = 1.2e-06) | 10.64× |
| 3 | 9.39× (95% CI 8.23–10.84; 34/34 slices; p = 5.8e-11) | 1.21× (95% CI 1.10–1.31; 30/34 slices; p = 5.1e-06) | 7.73× |
| 5 | 5.96× (95% CI 5.23–6.85; 34/34 slices; p = 5.8e-11) | 1.13× (95% CI 1.05–1.21; 22/34 slices; p = 9.6e-04) | 5.26× |
| 7 | 4.37× (95% CI 3.85–5.04; 34/34 slices; p = 5.8e-11) | 1.05× (95% CI 0.98–1.12; 20/34 slices; p = 1.7e-01) | 4.16× |
| 9 | 3.46× (95% CI 3.03–3.99; 34/34 slices; p = 5.8e-11) | 0.98× (95% CI 0.92–1.04; 12/34 slices; p = 8.5e-01) | 3.54× |

1. **Against plain averaging the gain is large and universal**: every slice, every number of repetitions. It shrinks as more repetitions are averaged, because averaging removes the random part of each denoised repetition's error but not the part they share.
2. **Against the best possible blur the gain is real but smaller**: significant when up to 5 repetitions are used, not significant beyond (a = 7, 9). Much of the improvement over plain averaging — 93% of it on a log scale at a = 5 — is available to any well-tuned smoother. The fair statement is that the network adds a further 13–25% for a ≤ 5 on top of an idealised smoother that could not be tuned this way in practice.

## Why it is not just smoothing

MSE rewards smoothing, so the error was also split by spatial frequency (a = 5):

| frequency band (cycles/pixel) | slices where R3 ≥ best-case blur | median R3 / blur |
|---|---:|---:|
| 0–0.0625 | 34 / 34 | 3.42× |
| 0.0625–0.125 | 24 / 34 | 1.30× |
| 0.125–0.25 | 16 / 34 | 0.98× |
| 0.25–0.75 | 24 / 34 | 1.10× |

The network's advantage is concentrated at the **coarsest** scale, on every slice. A linear low-pass smoother can only remove fine-scale noise; it cannot tell coarse-scale noise from signal. The network, having learned what these maps look like, removes noise at all scales. At finer scales it is on par with the best blur (0.125–0.25 cycles/pixel: 16/34 slices, median 0.98×) or slightly ahead (finest band: 24/34, median 1.10×) — so its advantage is not bought by erasing more detail than the best blur does.

## Limits — stated plainly

- **Noise levels outside training fail.** The one clear failure is slice 29, the quietest slice, whose noise was below every training slice of its fold: there the network scores 4.08 against the blur's 24.93 (one repetition). Training must cover the noise levels of deployment.
- **Only Gaussian smoothing as the classical baseline.** Nonlinear classical denoisers (e.g. BM3D, non-local means, total variation) can also act at coarse scales and have not been compared yet; that comparison is the natural next step.
- **One dataset.** All 34 slices come from one acquisition; generalisation to other subjects, scanners or protocols is untested.
- **MSE-based.** n_eff is a squared-error measure. The frequency analysis addresses resolution; task-specific measures (e.g. perfusion quantification) are not yet evaluated.
- **Robustness checks passed.** Excluding two slices earlier flagged as unreliable: 5.94× (95% CI 5.21–6.84; 32/32 slices; p = 2.3e-10). Scoring all pixels instead of the signal region: 9.80× (95% CI 8.52–11.38; 34/34 slices; p = 5.8e-11).

## What was tried and did not beat this

Denoising the average of the repetitions once (instead of averaging per-repetition outputs) is optimal in theory, since the average carries all the information. A model trained for it improved substantially once its training covered low noise levels evenly and used flip/rotation augmentation, but on this dataset it still trails the per-repetition approach on quiet slices. It is a candidate for larger training sets.

## Reproduction

Pre-registration: `Archive/CV_R3/PREREGISTRATION.md` (sha256 `230c76431ae08029`). Training: `python experiments/_cv_train.py --fold K` for K = 0–4. Evaluation: `python experiments/_cv_eval.py`. This document: `python experiments/_cv_report.py`. Model weights and hashes: `checkpoints/CV_R3/MANIFEST.md`. Per-slice numbers: `runs/cv_r3/eval/per_slice.csv`.
