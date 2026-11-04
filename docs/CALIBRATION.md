# Calibration after MEAN (2026-11-03)

Where the evidence stands, what it does to our hypotheses, and a proposed main-text / appendix split. Numbers are from
`WEEKEND_ITERATIONS.md`; claim IDs refer to `RESEARCH_NOTES.md` note 10 and `THEORY.md`. For discussion; nothing in
`RESEARCH_NOTES.md` has been changed yet.

---

## 0. The one-paragraph picture

Mandated repetitions are enough to train a denoiser without clean data or a noise model, and we can measure its
accuracy without ground truth. Two ways to use the repetitions exist. **Denoise each, then average (PER)** is what our
ASL results use; it needs the strength dial r set by the averaging depth. Its blur is structural: every copy shrinks
the fine bands the same way, and averaging does not undo it. **Denoise the mean (SUB)** is optimal in principle
(sufficiency). On fastMRI and natural images at moderate noise it wins by 1.5–1.7× n_eff, with as few as 16 training
images. It also recovers detail that PER loses. On ASL (heavy noise, coloured, one subject) it has not yet won. The
article's central open question is therefore no longer "how much data does SUB need" but **"where on the SNR axis
does SUB stop winning, and which side is ASL on?"**

---

## 1. What MEAN changed

| Pre-registered prediction | Result | Consequence |
|---|---|---|
| P1: SUB beats PER at a = 9, 62 volumes | **1.54×** n_eff (CI 1.51–1.57, 15/15 volumes, Holm 1e-4) | Sufficiency holds on MRI data |
| P2: SUB keeps more fine detail (band .5–.75) | **+0.21** transfer (0.49 vs 0.28; 15/15) | SUB fixes the blur that PER cannot fix |
| The advantage grows with training data; near 1 at 2 volumes | **Refuted.** 1.71× at 2 volumes (16 images), 1.72× at 8, 1.54× at 62. PER gains more from data than SUB does | "SUB is data-limited" is dead, for the second time (C7 already showed this on natural images) |
| SUB slightly below PER at a = 1 | 0.97× at 62 volumes; a tie at 2 and 8 | A single SUB model is nearly as good as PER for single maps too |
| PER's transfer is constant across a | **Holds for the denoiser itself** (r = 0: 0.85 / 0.50 / 0.14 / 0.04 at every a). The apparent rise at a = 9 is the dial blending noisy input back in | The blur mechanism is confirmed exactly |

Two side results matter:
- SUB is chosen with **r = 0** at every a: it needs no dial. The dial is a correction that PER needs, not a property
  of denoising for averaging in general.
- In RES, PER's gain from averaging is far below a: at M = 320, going from a = 1 to 9 only doubles n_eff (both noise
  levels). This is the same loss seen from the other side.

## 2. Effect on our hypotheses

1. **Blur (your hypothesis).** Correct, and now sharper. There are two kinds of blur:
   - **Architectural blur** comes from denoising each copy and then averaging. Each copy is shrunk per band by about
     S/(S+N) at the single-copy SNR, and averaging keeps that shrinkage. Denoising the mean removes this blur
     (MEAN, NAT-SUB).
   - **Information blur** is irreducible. Even the optimal E[x | mean] shrinks bands whose SNR *after averaging* is
     below 1. SUB reaches its supervised bound (NAT-SUB: 26.07 vs 26.07 dB), so what blur it keeps is this kind.

   For the article: we can offer a fix for the first kind (at least at moderate noise), and state the second as a
   limit of any denoiser, which is the "beyond denoising" discussion you proposed.
2. **Note 5 / C7 ("denoising the mean doesn't help on ASL").** The data-size explanation is refuted twice. What's
   left:
   - **The heavy-noise regime.** ASL's per-repetition SNR is about 0.04–0.13. MEAN is at about 0.8. Natural images
     at σ 2.0, where SUB lost, are 4× noisier than at σ 0.5. We have no common SNR scale yet to place them exactly.
   - **ASL-specific causes:**
     - coloured noise;
     - non-independent repetitions (dataset 3's gate);
     - the old GROUND SM recipe (L1 loss, no lower ladder extension, old reconstruction).

   We can't tell these apart yet; section 5 lists the experiments that would.
3. **C1, the dial set by a.** Still true and still exact, but its role shrinks: it is the best way to *average
   per-copy outputs*, and the fallback where SUB fails. Whether it stays a headline claim depends on the regime result.
4. **C3, ladder robustness.** It only extrapolates upward. RES reproduces this independently: A collapses on noise
   below its ladder at both noise levels (0.17× and 0.52× at M = 80), and recovers when the noise is matched. The
   claim is narrowed and consistent.
5. **Resolution (note 8, linear-operator principle).**
   - One fully convolutional network trained across matrices (B) is enough.
   - Voxel-size conditioning (C) adds nothing at normal noise and costs 9% at an in-between matrix at low noise.
   - Resampling to a standard grid (D) fails when noise is held fixed (0.70× at M = 320).
   - A matrix specialist (A) wins at its own size by 1.2× at both noise levels.

   The design principle survives, but the nonlinear negative control was never run. This is a robustness result,
   not a headline.
6. **C9 (noise colour limits transfer).** Still a candidate. TQ E2 found colour secondary (about 1.5% at a = 9); the
   label sweep was not run. Today's coil findings (dataset 2/3 noise/M0 is 1.4–1.6× dataset 1's) point to
   acquisition, not recon, as the source of differences between datasets.

## 3. Proposed main text (four claims + one application)

| # | Claim | Intuition → theory → evidence | Status |
|---|---|---|---|
| **M1** | Mandated repetitions train a denoiser with **no clean data and no noise model**; target quality is marginal | E[target \| input] = E[clean \| input] (THEORY P3) → KR equivalence; Noise2Score, Neighbor2Neighbor and R2R collapse on coloured noise, ours doesn't (BASE, NAT) | Complete (C4, C5) |
| **M2** | **Denoise the mean, not each copy.** Per-copy denoising + averaging causes architectural blur; the copy mean is sufficient | Per-band shrinkage S/(S+N) is not undone by averaging; sufficiency (P5) → MEAN (1.54×, detail +0.21), NAT-SUB (+1.59 dB), C7 regime map | Complete at moderate noise; **heavy-noise boundary open** (§5) |
| **M3** | When per-copy outputs are averaged, set the strength by the averaging depth, and use the low-bias score parametrization | Averaging removes variance, not bias → closed form r*(a) (P2); σ² output scaling halves bias² (DEC A, C2) | Complete; the "why σ² scaling" theory is still missing (could stay empirical) |
| **M4** | **Ground-truth-free evaluation and selection** | Independent halves (P1) → unbiased within 0.1% (RES), 1% (NAT); leave-one-out choice of r/checkpoint matches the oracle | Complete (C8) |
| **App.** | ASL perfusion | CV results, all baselines (BM3D 1.29×, DDM2 1.14×, classic N2N 1.11× at a = 9), with the data caveats | **Needs a rerun on the new reconstruction** |

Narrative: M1 makes training possible, M4 makes it measurable, M2 says what to compute, and M3 says what to do when M2
isn't available (heavy noise, or single-copy outputs). The regime map, SUB/PER against per-copy SNR with the
Gaussian-theory curve, the natural-image points, the fastMRI points and the ASL point, is the central figure of M2 and
possibly of the paper.

## 4. Appendix (supports the main claims, not central)

- **C6:** generative sampling hurts, including in the finest band (DEC B).
- **C3:** the ladder only extrapolates upward (CLAIMS C3; RES arm A).
- **RES:** one model across resolutions, both noise levels; voxel conditioning unnecessary; standardization fails.
- **Adaptive-r headroom:** per-slice r adds ≤4%; spatial r maps overfit.
- **Recipe:**
  - losses (MSE noise-normalized);
  - iterations and checkpoint selection;
  - augmentation (confounded);
  - GPU non-reproducibility (several seeds per claim).
- **Baselines in full:** BASE, BASE-DL, DDM2, including the dataset 2–3 soft spot.
- **Estimator validation tables:** RES, NAT, synthetic.
- **ASL data:**
  - re-reconstruction and GIRF fix;
  - noise colour from spiral DCF recon;
  - coil configuration;
  - the independence gate (with a multiplicity correction).

## 5. Superseded — do not reuse

- "Denoising the mean failed on ASL because of limited data" (note 5, note 6d's "consequence" line, and the frozen
  abstract's future-work sentence). Refuted by C7 and MEAN.
- "Diffusion is better." Not a claim; replaced by M3 (parametrization) and C3 (ladder).
- "Sampling recovers buried detail." Refuted (C6).
- Fixed r = 1/3. 0.19–0.31× of the depth-chosen r for single maps.
- "The ladder gives robustness to shift" (unqualified). Upward only.
- "Voxel-size conditioning is needed across resolutions." Not needed (RES at both noise levels).

## 6. Gaps and decisions for you

**Experiments that decide M2's scope, in order:**
1. **MEAN at ASL-like SNR on fastMRI.**
   - Design: the MEAN design at 2–3 lower SNRs (noise ×4 and ×8 at matrix 160, about 0.2 and 0.1 per copy), 8 and 62
     volumes.
   - What it settles: whether the reversal seen on natural images at σ 2.0 also happens on MRI, and at what SNR.
   - Cost: about 3–4 h per noise level.
   - Before running it, define one SNR measure (head-mask signal power / noise power) and compute it for every point
     on the map, including ASL.
2. **SUB on the re-reconstructed ASL data.**
   - Recipe: the new SUB recipe (MSE, ladder floor σ/3), cross-validated.
   - Retraining is needed anyway, since every ASL result used the old reconstruction.
   - Run together with the PER retraining.
3. **The low-SNR reversal, as theory.** Candidates:
   - SUB is trained at fewer effective noise draws per image;
   - learned nonlinear priors are poor at very low SNR.

   Results 1–2 tell us which one to pursue.

**Decisions:**
- **Framing.** (a) "Denoise the mean" is the method, with PER + dial as the heavy-noise fallback. (b) PER + dial is the
  method, with denoise-the-mean as the principled extension and the blur discussion. I'd **write toward (a) and
  decide after experiments 1–2**. If SUB also wins on re-reconstructed ASL, (a) is clearly stronger. If ASL sits past
  the reversal, (b) with a quantified regime boundary is still a solid, honest paper.
- **Blur framing.** As §2.1: architectural blur is solved (at moderate noise); information blur is a limit of any
  denoiser and belongs to the "beyond denoising" discussion (more repetitions, better acquisition, priors from other
  contrasts).
- **The dataset-4 rescan** with dataset 1's settings would add a second clean subject to the ASL application.
  Retraining waits for it, or proceeds on datasets 1–3 now.
- **RESEARCH_NOTES update.** I'd fold this calibration into notes 5, 8, 10 and 11 once you've adjusted it.
