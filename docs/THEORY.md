# Theory draft: denoising repeated acquisitions for averaging

Draft 2026-11-02. The theory legs of the claims ledger (`RESEARCH_NOTES.md`, note 10). Propositions with short
proofs; where only a proof of concept exists, this is said explicitly. Notation is per pixel: ‖·‖² means the mean
over pixels.

**Setting.**
- **Data:** a clean image x; N copies y_k = x + n_k. The noise vectors n_k are independent of each other and of x,
  zero-mean, with covariance Σ (not necessarily white); T = tr Σ / P is the per-pixel noise variance.
- **Denoiser:** D is applied to one copy at a time and trained without the copies being scored (cross-validation).
  The dial is D_r(y) = D(y) + r (y − D(y)).
- **Averaged estimate:** for a subset A of a copies, x̂_A = (1/a) Σ_{k∈A} D_r(y_k).

---

## P1 (C8). The split-half error estimate is unbiased, for any noise colour

Let B be the complementary N − a copies, ȳ_B their mean, and s² the per-pixel sample variance across all N copies
(E s² = T). Then

  E[ ‖x̂_A − ȳ_B‖² ] − E[s²] / (N − a) = E[ ‖x̂_A − x‖² ].

*Proof.* x̂_A − ȳ_B = (x̂_A − x) − n̄_B.
- x̂_A depends only on the copies in A, so it is independent of n̄_B given x.
- E n̄_B = 0, so the cross term vanishes.
- E‖n̄_B‖² = T / (N − a).

Linearity of expectation gives the result. No assumption on the noise distribution or colour is used. ∎

*Remarks.*
- Averaging the estimate over subsets keeps it unbiased.
- Its *precision* falls when the true error is small next to the subtracted term T/(N − a). This is what NAT saw at
  σ = 2.0, where the true error was about 0.25% of that term.
- Validated with ground truth: within 1% (NAT P1).

## P2 (C1). Bias/variance decomposition and the depth-dependent optimal strength

Given x, write the one-shot error as D(y_k) − x = β + u_k:
- β = E[D(y) | x] − x is the bias, shared by all copies;
- the u_k are zero-mean and i.i.d. across k, because the copies are i.i.d. given x.

With B = ‖β‖², V = E‖u‖² and C = E⟨u, n⟩,

  E[ ‖x̂_A − x‖² | x ] = (1 − r)² (B + V/a) + r² T/a + 2 r (1 − r) C/a.

*Proof.* x̂_A − x = (1 − r)(β + ū_A) + r n̄_A. Expand. Across copies, cross terms between u_k and n_j (k ≠ j)
vanish by independence; within a copy, E⟨u_k, n_k⟩ = C. ∎

The error is a quadratic in r. With P = B + V/a, Q = T/a and R = C/a, the minimizer is

  r*(a) = (P − R) / (P + Q − 2R), clipped to [0, 1].

As a → ∞, P → B while Q and R → 0, so r*(a) → 1 whenever B > 0.

*Interpretation.* Averaging divides the variance terms by a but leaves the bias untouched. The more copies are
averaged, the less per-copy smoothing pays, so r*(a) increases with a. All three ingredients (B, V, C) can be
estimated without ground truth via P1. For the all-splits estimator the decomposition holds exactly as an identity
of the estimator (`_gr_theory.py`).

*Evidence.* ASL sweeps; NAT P2: r chosen without ground truth equals the oracle in 3 of 4 conditions.

## P3 (C4, C5). Pair-based training needs no noise model; score-from-noisy-data methods do

(a) **Noise2Noise identity.** Let the target be t = x + m, with m zero-mean and independent of the input y. Then
E[t | y] = E[x | y], so the minimizer of E‖f(y) − t‖² is the posterior mean E[x | y]. This holds whatever m's
distribution, level or colour. Targets averaged from k other copies (any k ≥ 1) satisfy this; target noise changes
only the gradient variance (C5: k is marginal, KR).

(b) **Tweedie with coloured noise.** For Gaussian noise of covariance Σ,

  E[x | y] = y + Σ ∇ log p(y).

Noise2Score-type methods estimate ∇ log p(y) from noisy data and assume Σ = σ² I. Their output is then
y + σ² ∇ log p(y), and the error relative to the posterior mean is (Σ − σ² I) ∇ log p(y). This error does not shrink
with more data or capacity, and it vanishes only for white noise. ∎

*Evidence.*
- Noise2Score denoises white noise but is worse than the input with ASL-coloured noise (`baselines/_n2s_validate.py`).
- On natural images with ASL-coloured noise (NAT C05), Neighbor2Neighbor, R2R and Noise2Score collapse, while
  pair-trained models do not.

## P4 (C6). Generative sampling doubles the error of the posterior mean

Let X′ be a posterior sample, X′ ~ p(x | y), independent of the truth X given y. Then

  E‖X′ − X‖² = 2 E tr Cov(X | Y) / P = 2 MSE(E[X | Y]).

The mean of M independent samples has error (1 + 1/M) MSE(E[X | Y]).

*Proof.* Given y, X and X′ are i.i.d. with mean m(y), so E‖X′ − X‖² = 2 E‖X − m(Y)‖². For the M-sample mean, the
deviation from m(y) has covariance Cov/M. ∎

*Evidence.* DEC B: a sample loses 27% of n_eff overall and 30% in the finest band. With a network trained by N2N,
sampling is also limited by the noise floor: the score is never learned below the quietest acquisition (DEC B
extrapolation: 0.24–0.36×).

## P5 (C7). The copy mean is sufficient; denoising each copy is not optimal, by an SNR-dependent margin

For Gaussian noise of covariance Σ, the likelihood of the copies in A factorizes through their mean:

  p(y_A | x) ∝ exp(−(a/2)(ȳ_A − x)ᵀ Σ⁻¹ (ȳ_A − x)) · h(y_A).

So E[x | y_A] = E[x | ȳ_A], and denoising the mean at noise Σ/a is MMSE-optimal. The average of per-copy posterior
means, (1/a) Σ_k E[x | y_k], differs from it in general. ∎

*Size of the gap (proof of concept, infinite data).* In the linear-Gaussian model (natural-image-like spectrum,
`experiments/_theory_c7_gaussian.py`), compare per-copy Wiener plus the best scalar dial against Wiener on the mean:

| per-pixel SNR | a = 3 | a = 9 |
|---|---|---|
| 5 | 1.000 | 1.000 |
| 1 | 1.024 | 1.032 |
| 0.2 | 1.072 | 1.219 |
| 0.05 | 1.034 | 1.138 |

(n_eff ratio, mean-denoising / per-copy; a = 1 is 1.000 throughout.)
- The gap grows with a and peaks at intermediate SNR. At high SNR both are near the identity; at very low SNR both
  rely on the prior.
- On natural images the measured gap is larger (+1.59 dB ≈ 1.44× n_eff at SNR ≈ 1, NAT-SUB). Nonlinear priors widen
  it: one very noisy copy carries less usable structure than the linear model credits.
- **Regime map (CLAIMS C7, natural images, a = 9):** mean-denoising wins by +1.6 to +2.2 dB at σ 0.5 for 25, 100
  and 400 training images, but loses by 0.2–0.9 dB at σ 2.0. Training-set size is not the driver; noise level is.
- **Open:** the reversal at very low SNR is outside this proposition. In the Gaussian model the gap shrinks toward 1
  at low SNR but never favours per-copy denoising. Candidates: the learned mean denoiser sees a single noise level
  and a copies fewer training samples per image; nonlinear priors learned at very low SNR are poor. ASL
  (per-repetition SNR ≈ 0.04–0.13) sits in this regime.

## P6 (C2, proof of concept). The parametrization fixes the side from which the optimum is approached

Linear per-frequency gain w, N2N loss, power spectrum S. The optimum is w* = S/(S + σ²) for every parametrization.
Gradient descent from zero initialization, at rate η, with E = S + σ²:
- **Direct** (w = h): w_t = w* (1 − (1 − ηE)^t). This approaches w* from below: **over-smoothing**, bias above
  optimal and variance below.
- **Residual** (w = 1 + g) and **score** (w = 1 + σ² g): w_t = 1 − (1 − w*)(1 − (1 − η′E)^t). This approaches w*
  from above: **under-smoothing**, bias below optimal and variance above.
- Score and residual differ only in the effective step size (σ⁴ vs 1), so they coincide at a single noise level.
  The σ² scaling matters only when σ varies across samples: it keeps the network's output on one scale across noise
  levels.

Simulation (`experiments/_theory_c2_toy.py`, exact errors): at 10 gradient steps, direct has bias² 0.47 and variance
0.02, while residual/score have bias² 0.004 and variance 0.85. **In the linear model the difference is transient:**
all three reach the Wiener optimum within about 1000 steps.

*Status.*
- The toy proves the *direction*: parametrizations that start at the identity under-smooth, i.e. lower bias and
  higher variance, as DEC A measured (bias² halved, variance ×3).
- It does not explain why the gap *persists* in networks at 2000 iterations. Candidates are non-convexity, finite
  capacity, and implicit regularization of over-parametrized models.
- **Network results contradict the toy's two other predictions:**
  - *Residual ≈ score is false in networks.* The residual parametrization behaves like the direct one (within 5%
    everywhere) and loses to score at a = 9 (0.857×; CLAIMS C2). Starting at the identity is not enough; the σ²
    scaling is what matters.
  - *The gap is not transient.* From 500 to 2000 iterations DSM's bias² stays about one third below the direct
    model's (0.043→0.032 vs 0.061→0.048), while DSM's variance rises after 1000 iterations (CLAIMS C2-LC).
- **Revised status:** the toy explains the direction only. The working hypothesis is that σ² scaling acts as a
  persistent, level-dependent shrinkage of the network's correction (small corrections at low noise, one output
  scale across levels). It needs its own argument; this is the main open theory item.

## Not yet covered

- **C3 (multi-noise training → robustness):** coverage argument only. CLAIMS C3 supports it only upward: the
  ladder adds noise, so σ-conditioned models fail when the test noise is below the training noise (W035, a = 1:
  16.4 dB vs 22.6 dB for blind N2N). The coverage argument predicts exactly this; the claim must be stated as
  coverage, not as general robustness.
- **The C7 reversal at very low SNR** (P5, open).
- **Why σ² scaling regularizes** (P6, open).
