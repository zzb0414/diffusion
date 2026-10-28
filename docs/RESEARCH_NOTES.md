# Research notes: N2N-DSM denoising of ASL with mandated repetitions

Key notes recorded 2026-10-28, toward a full article for an AI venue. The ISMRM abstract is frozen and is not to be
revised with these findings. Numbers come from the logged, pre-registered runs in `WEEKEND_ITERATIONS.md`.

## 1. Scope and current claim

This project develops and evaluates a Noise2Noise denoising score-matching (N2N-DSM) approach for noisy ASL
subtraction maps. It exploits the repetitions that common ASL protocols already mandate. So far, the approach is
more effective and more SNR-efficient than both plain averaging and noise-matched Gaussian filtering.

*Evidence (CV_R3, CV_RATIO, GEN):* with 5 repetitions averaged, the denoised estimate gives 5.96× the effective number
of averages of plain averaging, and is better on all 34 slices. At r = 1/3, a single denoised map reaches
n_eff ≈ 6.8 and beats the noise-matched blur by 1.06×; averaging 9 maps gives 1.19×. The advantage generalizes
to the lower-SNR datasets 2 and 3, except that dataset 2 only ties the noise-matched blur.

## 2. Where the SNR gain comes from: two stages

The gain comes from two stages:

1. **Per-repetition denoising.** A one-shot Tweedie estimate from the learned score, x̂₀ = y + σ̂² s_θ(y, σ̂).
2. **Averaging.** The independently denoised repetitions are averaged.

The model was first trained with a leave-one-out (LOO) mean as the target. A first-class parameter r controls the
denoising strength: x̂ = x̂₀ + r (y − x̂₀). Lower r denoises more strongly but smooths more spatially. Each repetition
carries little spatial information because its noise is so strong (per-repetition SNR ≈ 0.04–0.13), so the full
Tweedie estimate is necessarily blurred.

*Evidence (CV_RATIO):* the best r depends on how many repetitions are averaged: 0 for a single map, about 1/12 for
3–5 maps, and about 1/6 for 9 maps. Averaging removes variance, so less per-map smoothing is needed.

## 3. Hypothesis (untested): denoise the mean instead of each repetition

With enough training data, we expect the per-repetition model to be inferior to a model trained to denoise the
mean of all repetitions directly. The per-repetition model maps one repetition to the noise level of its LOO mean
and then averages the results. The expectation follows from the limited spatial information in each repetition:
each repetition is denoised from too little information, and averaging afterwards cannot restore what each
denoising step discarded.

*Theory:* with i.i.d. Gaussian noise the mean of the repetitions is a sufficient statistic for the signal, so
E[x | mean] is the MMSE estimate. The average of per-repetition estimates E[x | y_i] generally is not, because each
one shrinks toward the prior using only one repetition's information.

*Status: partly tested, inconclusive* (R4_SUBSET, 2026-10-20, single split; see `WEEKEND_ITERATIONS.md`). A
subset-mean model was trained with input = mean of m repetitions and target = mean of the other 10 − m. It matched
"denoise each, then average" at a = 1. At a = 9 it lost on the quiet test slices (ratio 0.70–0.77 even after
balancing and augmentation, R4c) and won on the noisy one (1.15). This was attributed to data: an averaged input at
σ̂/3 is only produced by the few training slices that are quiet enough. The evidence so far is "data-limited", not
"wrong". It needs a cross-validated re-test, and ideally more training data. The TQ finding (note 4) helps: target
quality is not the bottleneck, so even a = 9 with a single held-out repetition as target is a valid training pair.

## 4. New finding: the LOO-mean target is an unnecessary detour

In the TQ experiment E1 (pre-registered), we built the training target from k = 1, 3 or 9 of the other
repetitions. With k = 1, the targets are the classic Noise2Noise pairs. Every model was decoded the same way.

- Results:
  - **k = 1 is not worse than k = 9.** The primary test gives 1.029× (95% CI 1.004–1.056, p = 0.039); all
    cells fall within 0.96–1.04× (arms A and D, datasets 1–3, a = 1, 5, 9).
  - **The whole r curve is unchanged by k.** The best r is 0, 1/12 and 1/6 for a = 1, 5, 9, whatever the
    target size.
- **Conclusion:** the classic N2N pair is sufficient for a well-trained model, with r setting the denoising
  strength. What sets performance is the noise level of the *input*, not how clean the *target* is.
- **Why:** the model's job is to estimate a conditional expectation. With zero-mean target noise independent of
  the input, E[target | input] = E[clean | input], so target noise adds only gradient variance (learning speed)
  and does not change the solution.
- **Consequence:** r is not tied to the target's noise. It is an output knob trading blur against noise, and its
  best value follows the averaging depth. The earlier rationale, r = √(1/(N−1)) "matching the LOO target noise",
  should be replaced.

The noise-colour experiment (TQ E2) supports the same picture from the input side. We degraded dataset 1 to 1.4×
and 2× σ̂ with noise in its own colour, white noise, or dataset 2's colour:

- **SNR drives the edge.** The edge over the noise-matched blur falls with input SNR (1.062 → 1.028 → 1.014 for a
  single map), and at equal σ̂ the colour makes no significant difference (primary test, 1.002× and 1.003×).
- **Colour matters only after averaging.** With 9 maps averaged, fine-heavy noise lowers the edge by about 1.5%.

Together, performance is governed by input SNR. Neither target quality nor noise colour is a primary factor.

*Caveats:*
- E1 used one training budget (2000 iterations); noisier targets may matter for much shorter training or less data.
- The 2× degradation does not reach dataset 2's SNR, so a small dataset-specific remainder cannot be excluded.
- k = 1 gains up to 1.23× at r = 0. This gain was not pre-registered and is unexplained.

## Directions for the article

**Scope (2026-10-28).**
- **Beyond ASL:** the method is not ASL-specific. As the abstract already states, it applies to any noisy MR
  acquisition with repetitions.
- **More data:** three datasets from one subject are not enough for a full article. If growing k proves marginal
  (note 4), collaborators' data can be pooled with plain N2N pairs as targets, without needing many repetitions per
  site.
- **Expected next problem:** heterogeneous data. Feature extractors and decoders, and alignment across resolutions
  and matrix sizes.
- **First, a solid ground:** confirm note 4 (seeds, learning curves), settle note 3, and derive r*(a).

- Test the hypothesis in note 3 (denoise the mean vs denoise each, then average) under cross-validation.
- Derive the optimal strength r*(a) for averaged estimates (averaging removes variance, not bias) and check it
  against the measured sweeps; candidate central contribution.
- Add baselines for an AI venue: MP-PCA, Patch2Self, Noise2Score / Recorrupted-to-Recorrupted, BM3D, DDM2.
- Recast the method around plain N2N pairs, with r (or its a-dependent optimum) as the strength control.
- Extend the E2 degradation to 3× σ̂ to cover dataset 2's SNR (evaluation only).
- Strengthen baselines: a SURE-tuned blur, and correct the oracle-blur wording.
