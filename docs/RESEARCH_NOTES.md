# Research notes: self-supervised denoising of repeated MR acquisitions (N2N-DSM)

Started 2026-10-28; updated 2026-10-29 with the GROUND results and a draft roadmap. These notes work toward a full
article for an AI venue. The ISMRM abstract is frozen and is not revised with these findings. Numbers come from the
logged, pre-registered runs in `WEEKEND_ITERATIONS.md`.

**Status labels:** *established* (pre-registered, cross-validated), *supported* (consistent evidence, not
pre-registered), *hypothesis* (untested or not supported yet).

---

## Part I — What we know

### 1. Scope and current claim (established)

The project develops a Noise2Noise denoising score-matching (N2N-DSM) approach for noisy ASL subtraction maps. It
exploits the repetitions that ASL protocols already mandate. The approach is more SNR-efficient than plain averaging
and than noise-matched Gaussian filtering.

*Evidence (CV_R3, CV_RATIO, GEN):*
- **vs plain averaging:** with 5 repetitions averaged, the denoised estimate gives 5.96× the effective number of
  averages (n_eff) of plain averaging, better on all 34 slices.
- **vs noise-matched blur:** at r = 1/3 a single denoised map reaches n_eff ≈ 6.8 and beats the noise-matched blur by
  1.06×; averaging 9 maps gives 1.19×.
- **Other datasets:** the advantage carries over to the lower-SNR datasets 2 and 3. On dataset 2 it only ties the
  noise-matched blur.

### 2. Where the gain comes from: two stages and a strength dial (established)

1. **Per-repetition denoising.** A one-shot Tweedie estimate from the learned score, x̂₀ = y + σ̂² s_θ(y, σ̂).
2. **Averaging.** The independently denoised repetitions are averaged.

A first-class parameter r sets the denoising strength: x̂ = x̂₀ + r (y − x̂₀).
- **Trade-off:** lower r denoises more strongly but smooths more. Each repetition carries little spatial information
  (per-repetition SNR ≈ 0.04–0.13), so the full Tweedie estimate (r = 0) is necessarily blurred.
- **The best r depends on the averaging depth a**, not on the training target: about 0 for a single map, 1/12 for
  3–5 maps, and 1/6–1/4 for 9 maps. Averaging removes variance but not bias, so less per-map smoothing is needed
  as a grows.

### 3. A closed form for the best strength (established as mathematics; interpretation supported)

For the split-half estimator averaged over all C(N, a) splits, the error of the averaged estimate is exactly

  1/n_eff(a, r) = (1−r)² (b + v/a) + r²/a + 2r(1−r) c/a      (in units of the noise variance)

per slice. The three numbers are b (bias²), v (variance of the per-map error) and c (its covariance with the noise).
The b and v terms read as bias and variance because, given the clean map, each repetition is denoised from its own
noise alone.

- **The optimum** is r*(a) = (P − R)/(P + Q − 2R), with P = b + v/a, Q = 1/a, R = c/a. It rises toward 1 as a
  grows, and it matches the measured best r on every model and dataset (for example 0.02 → 0.23 from a = 1 to 9 on
  dataset 1).
- **Caution:** the decomposition is an identity of the estimator, so a perfect fit is not evidence. The substance
  is the closed form and the interpretation of b, v and c (`experiments/_gr_theory.py`).
- **Ground-truth-free:** all three numbers can be estimated from a slice's own repetitions, which makes an adaptive
  r possible.
- **Adaptive r is subtle; the depth-dependent r is the lever (AR_HEADROOM, exploratory, 2026-10-29).**
  - **Fixed r = 1/3 is far off:** it reaches only 0.19–0.31× the n_eff of one r chosen per averaging depth a for a
    single map, and 0.78–0.92× at a = 9.
  - **Per-slice r adds at most 0.5–4.2%,** even chosen on the scored slice itself.
  - **Free spatial r maps are fitted noise:** an 8×8 block map chosen on one half of the repetitions scores
    0.79–0.98× on the other, independent half.
  - **A structure rule helps a little:** r set by local signal level, chosen on other slices, adds +3–8% at a = 9 and
    survives the cross-half check.
  - **Conclusion:** the best r is essentially a simple function of the averaging depth.
  (`experiments/_ar_headroom.py`, `_ar_crosshalf.py`.)

### 4. Target quality is marginal; input SNR governs performance (established)

- **TQ E1 and GROUND KR (pre-registered equivalence test, 3 seeds):** a target from a single other repetition
  (k = 1, the classic N2N pair) is equivalent to the 9-repetition leave-one-out mean: 1.000× (90% CI 0.984–1.017).
  Seed-to-seed variation alone is up to ±5%. The earlier +2.9% was one seed's draw.
- **Learning dynamics:**
  - k = 1 trails by about 5% early in training (500–1500 iterations), as N2N theory predicts.
  - From 3000 iterations, or with 25–50% of the training data, k = 1 is 14–28% better at a = 9. The k = 9 models
    overfit more.
- **Why:** the network estimates a conditional expectation. With zero-mean target noise independent of the input,
  E[target | input] = E[clean | input]. Target noise adds only gradient variance.
  - Under an MSE loss, the mean over single-repetition targets and the leave-one-out mean give the same expected
    gradient. The LOO mean is the low-variance version of the 1-to-many pair pool, so extra 1-to-many pairs add no
    new information.
- **Input side (TQ E2):**
  - **SNR drives the edge.** The edge over the noise-matched blur falls with input SNR (1.062 → 1.028 → 1.014 for a
    single map).
  - **Colour is secondary.** At equal σ̂ the noise colour makes no difference for single maps. It costs about 1.5%
    after averaging 9 maps.
- **Practical value:** data with only 2 repetitions, and sites with different N, can be pooled without changing the
  recipe.

### 5. Denoising the mean is not better at this scale (hypothesis, not supported)

- **Theory:** with i.i.d. Gaussian noise the mean is a sufficient statistic, so E[x | mean] is the minimum-MSE
  estimate. Averaging per-repetition estimates generally is not.
- **GROUND SM (pre-registered, cross-validated):**
  - **SUB vs PER:** a subset-mean model ("SUB") vs per-repetition denoising then averaging ("PER"), with pooled
    training at a = 9: 0.967× (p = 0.26), not significant.
  - **Effect of more data:** pooling 3× more data did not significantly close the gap (1.033×, p = 0.18).
  - **Across settings:** SUB is never significantly better, and clearly worse on dataset 3 at large a (0.72×).
- **Reading:**
  - **The theory isn't contradicted.** Without the dial (r = 0), SUB beats PER at large a (1.11–1.20× on
    dataset 1), as the theory predicts.
  - **The learned SUB falls short.** PER with a tuned dial beats every setting of SUB, so SUB is far from
    E[x | mean]. It is limited by capacity and data, and an L1 loss estimates a median, not a mean.
  - **The dial acts as a bias correction** for an imperfect denoiser.
- *Conclusion:* statistically optimal in principle; at this data scale, PER with a tuned dial is the better practical
  estimator. Revisit with much larger data.

### 6. Training-recipe findings (supported)

- **Loss:** MSE with noise normalization is best when maps are averaged (CV_ABLATION). Noise-normalized losses gain
  most from pooling (MX_CV).
- **Pooling on its own:** naive pooling (L1, 2000 iterations) did not help dataset 1. Training past about 2000
  iterations helps single maps but hurts averaged estimates (overfitting).
- **Augmentation:** dihedral augmentation together with a ladder extended to σ̂/3 made pooled per-repetition models
  6–18% worse than the plain recipe. The two effects are confounded.
- **Reproducibility:** GPU training is not bit-reproducible, so every seed is an independent draw. Claims need
  several seeds.

### 6b. What the method is: decoder and parametrization ablations (established, DEC)

Pre-registered, seeds 101–103, MSE noise-normalized, dataset-1 training (`Archive/DEC`).
- **Generative sampling hurts.** One posterior sample (20-step ancestral walk) vs one-shot Tweedie, dataset 1, single
  map: 0.73× overall and **0.70× in the finest band**. The finest band is where recovered detail would show.
  - Sampling below the trained noise floor is far worse (0.24–0.36×).
  - Under distribution shift (datasets 2–3), averaged samples or a deterministic multi-step walk beat one-shot Tweedie
    without the dial (up to 1.2×). With the dial tuned, at most about 5% remains.
  - N2N cannot learn the score below the quietest acquisition noise, which limits sampling by construction.
- **Score parametrization vs a plain regressor.** A classic σ-conditioned N2N regressor is worse once outputs are
  averaged (0.875× at a = 9, 90% CI 0.85–0.90) but better for a single map (1.13×).
  - **Mechanism:** the score parametrization (x̂₀ = x̃ + σ² net) halves the bias² (0.031 vs 0.046–0.048 σ²) at about
    three times the per-map variance. Averaging removes variance, not bias. Principle: **when outputs will be averaged,
    prefer the low-bias estimator.**
  - **Noise-ladder training** matters little in-distribution but gives robustness under shift: without it, datasets
    2–3 lose 11–24%.
  - **σ conditioning** helps only together with the ladder: a native-only regressor is better blind (+3% dataset 1,
    +16% dataset 2).
- **Candidate method (untested):** a loss on the error of the averaged estimate, targeting low bias directly.

### 6c. Baselines (established, BASE)

Pre-registered, all methods on the same splits, every parameter cross-fitted (`Archive/BASE`, code `baselines/`).
- **Dataset 1, a = 9, ours vs each baseline:** all eight are significantly worse (Holm).

  | Baseline | Ours / baseline |
  |---|---|
  | BM3D | 1.29× |
  | Gaussian blur | 1.36× |
  | Non-local means | 1.43× |
  | Patch2Self | 1.65× |
  | MP-PCA / LPCA | 2.6× |
  | Plain averaging | 4.3× |
  | Noise2Score | 11.2× |

- **Joint-across-repetitions methods** (MP-PCA, LPCA) hardly improve with more repetitions. The signal is identical
  across repetitions, so they offer little beyond averaging.
- **Noise2Score fails on real data because ASL noise is coloured.** A synthetic check with known ground truth
  (`baselines/_n2s_validate.py`) shows it denoises white noise but is worse than the input with dataset-1-coloured
  noise (1.25× the noise variance), where N2N is unaffected (0.002×). Pair-based training needs no noise model;
  score-from-noisy-data methods do.
- **Soft spot:** on dataset 2 (low-SNR, fine-grained noise, model trained on dataset 1) a tuned Gaussian blur ties
  ours (62.3 vs 60.7 at a = 9). Generalization across coils is the open problem.
- **Not yet compared:** Neighbor2Neighbor / Blind2Unblind, Recorrupted-to-Recorrupted, DDM2, and a supervised upper
  bound (needs ground truth; planned with step 3).

---

## Part II — Draft roadmap (for discussion, 2026-10-29)

### 7. The most effective framework given current evidence

| Component | Choice | Basis |
|---|---|---|
| Model | Small U-Net conditioned on the total noise level, score parametrization x̂₀ = y + σ² s_θ(y, σ) (low bias), trained on the noise ladder (robust to shift); one model for all slices, SNRs and datasets | Notes 1, 4, 6b |
| Decoding | One-shot Tweedie estimate per repetition, blended with r, then averaged ("PER"). One network pass per repetition; no sampling (6b) | Notes 2, 5, 6b |
| Strength | r chosen for the averaging depth a: r*(a) from the closed form, or one r per a chosen on other slices. Optional: r by local signal level (+3–8% at large a) | Note 3 |
| Loss | MSE, noise-normalized | Note 6 |
| Targets | Leave-one-out mean where N ≥ 3 (equivalent and lower variance); single other repetition when N = 2 | Note 4 |
| Training | About 2000 iterations; checkpoint selected by validation n_eff at the intended a, not by loss | Notes 4, 6 |
| Data pool | Pool everything with per-slice σ̂ labels; noise normalization balances SNRs; ladder spans the smallest σ̂ to 0.4. No augmentation until its effect is separated | Notes 4, 6 |
| Heterogeneous data | Only linear operations on the noisy data (note 8). Candidates: k-space-consistent resampling with noise-spectrum conditioning, or native-resolution fully convolutional training, with or without voxel-size conditioning | Hypothesis, see notes 8–9 |
| Evaluation | Split-half unbiased n_eff over all splits; noise-matched blur and plain averaging as reference points | Note 1 |

### 8. Positioning and candidate novelties

**Conventional methods we must compare against**
- **Non-DL:**
  - Plain averaging.
  - Gaussian / noise-matched filtering.
  - BM3D / BM4D.
  - Non-local means.
  - MP-PCA (Marchenko–Pastur PCA): it exploits repetitions jointly, is the standard MRI baseline, and is the
    strongest non-DL competitor here.
- **Self-supervised DL:**
  - Noise2Noise (pairs).
  - Noise2Self / Noise2Void (blind spot).
  - Neighbor2Neighbor.
  - Noise2Score (Tweedie from a learned score; closest to our decoding).
  - Recorrupted-to-Recorrupted.
  - Patch2Self (MRI volumes; closest in data setting).
  - DDM2 (diffusion-based MRI denoising).
- **Supervised DL** needs clean data, which this setting doesn't have.

**Not novel by itself:** a one-shot Tweedie estimate from an N2N-trained score (≈ Noise2Score + N2N), and target
insensitivity (N2N theory).

**Candidate novelties, strongest first:**
1. **Denoising for averaging.** An estimator whose strength is set by how many outputs will be averaged, with a
   closed-form optimum r*(a) whose ingredients (bias, variance, noise covariance) are estimable without ground truth
   from the repetitions themselves. Extension: an adaptive r per input, per slice, or as a spatial map. This is
   general to any repeated acquisition and is the best candidate for the central contribution.
2. **Ground-truth-free evaluation.** The exact, unbiased split-half estimator of n_eff over all splits, with its
   bias/variance decomposition. It enables model selection, r tuning and fair comparison without clean data.
3. **One noise-conditional model across heterogeneous acquisitions.** Total-noise conditioning lets one model serve
   all SNRs. Because target quality is marginal, sites with any N ≥ 2 can be pooled.
   - **Motivation (user, 2026-10-29):** as in other computer-vision problems, a useful DL method should handle a range
     of resolutions. Two generic routes: standardize everything to one resolution, or encode each resolution into
     shared feature maps, denoise there, and decode back to full resolution.
   - **Design principle: only linear operators on the noisy data.** N2N needs zero-mean noise independent of the
     signal in the space where it operates, so that E[target | input] = E[clean | input].
     - A nonlinear encoder (autoencoder, learned latent) breaks this: the encoded noisy map minus the encoded clean map
       is no longer zero-mean independent noise. N2N in a learned latent space is therefore biased, and σ conditioning
       and the split-half estimator lose their footing. Latent diffusion avoids the problem only because it trains on
       clean images, which we never have.
     - At per-repetition SNR 0.04–0.13, an autoencoder trained on noisy data would also mostly encode noise.
     - A linear operator A keeps Gaussian noise Gaussian with known covariance A Σ Aᵀ, so N2N, conditioning and
       evaluation all remain valid.
   - **Candidate designs:**
     - (a) Standardize by k-space-consistent (linear) resampling. The noise becomes coloured in a known way; condition
       on the noise spectrum (TQ E2: colour costs about 1.5% after averaging).
     - (b) Linear per-resolution adapters into a shared denoiser: the feature-map route restricted to linear maps.
     - (c) No standardization: a fully convolutional network handles any matrix size natively. What differs between
       sites is voxel size (anatomy appears at a different pixel scale), so condition on voxel size or train across
       scales.
   - **Status:** to be tested once multi-resolution data exist (ablation in note 9). If the linear-operator principle
     holds up, it is a defensible claim of its own: pooling heterogeneous resolutions without breaking
     self-supervision.

### 9. Ablations that would justify each claim

| Claim | Ablation | Status |
|---|---|---|
| Score + Tweedie decoding is needed | Same network trained as a direct N2N regressor (x̂ = f(y, σ)), decoded with the same dial | Done (DEC A): score parametrization wins when averaging (lower bias), regressor wins for single maps (note 6b) |
| Generative sampling recovers detail from the learned prior | Full reverse diffusion (sampling to σ = 0) vs one-shot Tweedie (posterior mean) vs direct regressor, scored overall and on the finest-frequency n_eff bands. See the note below the table | Done (DEC B): refuted, sampling is worse overall and in the finest band (note 6b) |
| Dial for averaging | r = 0 (Noise2Score-like), fixed r = 1/3, global r*(a), per-slice r, signal-dependent r, spatial r map; for a = 1…N | Headroom done (AR_HEADROOM, note 3); confirm on new data |
| Ground-truth-free estimator | Synthetic data with known ground truth (digital phantom or high-average reference + real noise): estimated n_eff vs true n_eff; predicted vs true r*(a) | To do |
| Target insensitivity enables pooling | k = 1 vs k = 9 (done). Pooling sites with N = 2 vs N = 10 at equal total scans | Partly done |
| Noise conditioning | Blind model (no σ) vs scalar σ vs noise-spectrum conditioning; within- and cross-dataset | Partly done (DEC A): σ conditioning helps only with ladder training; spectrum conditioning to do |
| Pooling heterogeneous data | Single-site vs pooled training; noise-normalized vs plain loss | Partly done (MX_CV) |
| Resolution handling | (a) linear k-space resampling + noise-spectrum conditioning; (c) native-resolution fully convolutional model without resolution conditioning; (c') the same with voxel-size conditioning; nonlinear autoencoder latent as a negative control for the linear-operator principle. Within-site and leave-one-site-out | Needs multi-resolution data |
| Comparison | All baselines in note 8 at matched a, scored with the same n_eff estimator, plus a reader-independent structure metric (per-band n_eff) | Done for classical + Noise2Score (note 6c); further DL baselines to do |
| Generality | A second application with repetitions (e.g., multi-average DWI or low-field MRI) and several subjects | Needs data |

**On generative sampling (user's motivation, 2026-10-29).** Our data are exactly the kind diffusion models are trained
on. The hope is that the model predicts what the noise has buried, from what it has learned on other slices and
subjects.
- **Current decoder doesn't sample.** The one-shot Tweedie step returns the posterior mean E[x | y], the same
  quantity an MSE regressor learns. It already uses the learned prior optimally under squared error.
- **Where diffusion currently helps:**
  - one network gives a denoiser at every noise level;
  - the dial r is one DDIM step to noise level r·σ, i.e. stopping the reverse process partway.
- **Prediction (to be pre-registered):** full sampling lowers n_eff overall. A posterior sample carries about twice
  the squared error of the posterior mean. At per-repetition SNR 0.04–0.13 the posterior is broad, so sampled detail
  is largely invented, which is a risk for quantitative perfusion maps. Averaging many samples only converges back
  to the posterior mean.
- **If wrong:** if sampling wins in the finest bands, that is a striking result and deserves its own section.

### 10. Order of work

1. ~~Adaptive-r headroom~~ (done 2026-10-29): adaptive r is a minor refinement; the depth-dependent r*(a) is the
   centrepiece.
2. ~~Decoder ablations and baselines on current data~~ (done 2026-10-30, notes 6b–6c). Remaining from this step:
   further DL baselines (Neighbor2Neighbor / Blind2Unblind, Recorrupted-to-Recorrupted, DDM2).
3. **Synthetic validation** of the estimator and of r*(a).
4. **Recipe clean-up for pooled data:** separate augmentation from ladder; select checkpoints by validation n_eff.
5. **Multi-subject, multi-site data:** resolution handling (linear operators; with vs without voxel-size
   conditioning), noise-spectrum conditioning, second application.

---

## Log of note changes
- 2026-10-28: notes 1–4 recorded from the user's key notes; scope beyond ASL added.
- 2026-10-29: GROUND results folded in (note 4 upgraded to established; note 5 not supported at this scale); closed
  form for r*(a) (note 3); recipe findings (note 6); draft roadmap (Part II).
- 2026-10-29: resolution handling (linear-operator principle, voxel-size conditioning) and the generative-sampling
  ablation added after discussion with the user.
- 2026-10-29: AR_HEADROOM folded into note 3; adaptive r demoted to a refinement.
- 2026-10-30: step 2 results (DEC, BASE) added as notes 6b–6c; roadmap rows updated.
