# Ablation map: which experiments led to the proposed approach (2026-11-09)

For discussion; not yet folded into `RESEARCH_NOTES.md`. Experiment names follow the archive folders. GAP experiments
are written GAP-B, GAP-C, …; claims keep their bare IDs (C1–C9). Selected arguments are S1–S4. Numbers are from
`WEEKEND_ITERATIONS.md`.

**Tiers**
- ★ **Key:** the decision rests on this experiment; it goes in the main text.
- ○ **Supporting:** it confirms or narrows the decision; appendix.
- ⚠ **Caveated:** the result stands, but it was run on the old reconstruction (before 2026-11-03), so its numbers need a rerun before publication.

## 1. The proposed approach and the experiments behind each component

| Component of the approach | Decision | Deciding experiment(s) | Result | Tier |
|---|---|---|---|---|
| Training signal | Noise2Noise pairs; the target can be the leave-one-out mean or a single other repetition | GROUND-KR, TQ-E1 | k = 1 ≡ k = 9: 1.000× (CI 0.984–1.017) | ★ ⚠ |
| Noise model | None needed (pair-based) | BASE (Noise2Score validation), NAT | Score-from-noisy methods collapse on coloured noise; ours doesn't | ★ |
| Output parametrization | Score form, x̂₀ = y + σ²·net | DEC-A, CLAIMS-C2 | Bias² halved vs a regressor (0.875× → wins at a = 9); residual without σ² is 0.857× | ★ ⚠ |
| Decoding | One-shot Tweedie, no sampling | DEC-B | Sampling 0.73× overall, 0.70× in the finest band | ★ ⚠ |
| Noise conditioning | σ-conditioned, trained on the noise ladder | DEC-A, CLAIMS-C3 | No ladder: −11–24% under shift; robust only to more noise than trained on | ○ |
| Loss | MSE, noise-normalized | CV_ABLATION, CV_RATIO, MX_CV | Best when maps are averaged; gains most from pooling | ○ ⚠ |
| Design for averaged estimates | Denoise the mean (SUBM) | NAT-SUB, MEAN, MEAN-SNR, MEAN-TAX, ASL-SUBJ, ASL-COIL | Natural images +1.59 dB; fastMRI 1.54×; reverses below SNR ≈ 0.2; SUBM fixes part of it; ASL unseen subject SUBM/PER 1.10× | ★ |
| Fallback for per-repetition outputs | Dial r set by averaging depth, closed-form r*(a) | CV_RATIO, AR_HEADROOM, NAT | Fixed r = 1/3 gives only 0.19–0.31× of r*(a) for single maps; r chosen without ground truth ≈ oracle | ★ |
| Evaluation and selection | Split-half n_eff, no ground truth; cross-fitted r, checkpoint and early stopping | NAT, RES, GAP-L | Unbiased within 1% (NAT) and 0.1% (RES); early stopping is required | ★ |
| Training patches | Whole slices for SUBM | ASL-CROP | SUBM on crops 0.947× | ○ |
| Input normalization | Per-slice noise map (optional, ASL-only) | GAP-K, GAP-K2, GAP-M | ASL: +3.6% (SUB), +3.0% (SUBM) at a = 9, leakage-free. fastMRI with co-located uneven noise: −5.5% (SUB), so not a generic fix; no mechanism claimed | ○ |

**Negative results that rule out alternatives** (all ○): fastMRI pretraining then fine-tuning (GAP-J, 0.61–0.69×), voxel-size conditioning (RES), dihedral augmentation (confounded, recipe note 6), Noise2Score-style decoding without pairs (BASE).

**Comparisons** (main text, but not ablations): BASE, BASE-DL, BASE-DDM2 ⚠. They need a rerun on the current reconstruction and setup.

**Superseded** (keep for the record only):
- R3/R4 recipe rounds, CV_R3: early recipes.
- GROUND-SM: old recipe and old reconstruction; replaced by ASL-SM2, ASL-SUBJ and ASL-COIL.
- MX_CV and GEN pooling conclusions: datasets later re-organized by coil.

## 2. S3 — why the denoise-the-mean gain is smaller on ASL: experiment status

The quantity is SUB/PER at a = 9. Real ASL (1104c, unseen subject) gives 1.062 (SUBM 1.103).

| Factor | Experiment(s) | What it showed | Tier | Open |
|---|---|---|---|---|
| Noise label, level, kind | GAP-A, GAP-B | Not the cause | ○ | — |
| Matrix / pixel scale | GAP-B, GAP-C | On fastMRI content, matrix 160 → 80 lowers the gain from 1.46 to 1.21 (spectral); pixel scale is not a lever | ★ | Doesn't carry to ASL-spectrum content (next row) |
| Signal spectrum | GAP-D/E | fastMRI content filtered to ASL's per-frequency SNR (D) supports 1.457 (real ASL noise) / 1.540 (white); without phase structure (E) only 1.146 | ★ | **The closest fastMRI analogue (DT, 1.457) is still far from ASL (1.062)** |
| Training data | GAP-F, GAP-H | fastMRI DT: 62 → 4 → 1 volumes gives 1.457 → 1.301 → 0.885; ASL with ¼ of its slices: 1.00 | ★ | Image count vs subject diversity untested |
| Uneven noise | GAP-I, GAP-K, GAP-K2, GAP-M (GAP-G confounded) | Uniform noise on ASL: 1.062 → 1.226; a smooth map recovers about 30% of that. On fastMRI with 62 volumes, ASL-sized co-located unevenness costs only 1% (M1 0.990; 3% of the gap) | ○ (was ★) | Not the cause with abundant data. Open: data × unevenness interaction (GAP-O), GAP-I's 0.85× in-region noise (SNR) |
| Prior transfer, overfitting | GAP-J, GAP-J2, GAP-L | Fine-tuning hurts; early stopping is required | ○ | — |

**fastMRI tests that lock down S3** (pre-registered one at a time, run in order):
1. **GAP-M (co-location), done 2026-11-09:** no. M1 = 0.990, and the noise-map remedy hurts on fastMRI (SUB 0.945×).
2. **GAP-N (diversity):** at a fixed image count, do fewer subjects cost the gain?
3. **GAP-O (composition / interaction):** training data (62 vs few volumes) × noise (DW vs DL). Does uneven noise cost only when data are scarce, and do the factors together reproduce ASL's 1.06 starting from DW's 1.54?

Together these tell us what to do when more ASL data arrive: how much data and how many subjects, and whether a noise-map normalization or channel is needed.
