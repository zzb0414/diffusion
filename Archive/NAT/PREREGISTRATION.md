# NAT pre-registration — step 3: validation with ground truth on natural images

Written 2026-10-30, BEFORE any NAT model was trained or evaluated. Roadmap step 3 (`docs/RESEARCH_NOTES.md`):
validate the ground-truth-free estimator and the depth-dependent strength r*(a) against exact ground truth, compare
methods with exact PSNR, and test the averaging-aware loss.

## Data (`../natural_images`, DnCNN benchmark split, github.com/cszn/DnCNN e93b278)
- Train: Train400 (400 grayscale 180×180), intensities /255. Validation (r selection only): Set12. Test: BSD68,
  centre-cropped to 320×480 (U-Net needs multiples of 16).
- Each clean image x gets N = 10 FIXED noisy copies y_k = x + n_k (seeded), as with real repetitions; training uses
  only those copies (and, for the supervised bound only, x).
- Conditions, each trained separately: **W50** white σ = 50/255; **W05** white σ = 0.5 (per-pixel SNR ≈ 1);
  **W20** white σ = 2.0 (extreme, ASL-like); **C05** σ = 0.5 coloured with dataset-1's median ASL noise spectrum
  (radial profile, unit mean power, as `experiments/_degrade.py`).

## Methods (code `natural/`; same U-Net (1 channel), Adam 1e-3, batch 8 images × 10 copies, 96×96 crops,
## 3000 iterations, seed 101, MSE unless stated)
- **OURS:** DSM score parametrization, total-noise conditioning, noise ladder (p_raw 0.5), leave-one-out N2N target;
  one-shot Tweedie + dial.
- **AVG9 / AVGU (averaging-aware loss, candidate method):** same model; the loss is the N2N error of the AVERAGED
  estimate, ‖mean_{k∈A} x̂₀(y_k) − mean_{j∉A} y_j‖² over random subsets A of size a = 9 (AVG9) or a ~ U{1..9} (AVGU).
- **SUP (upper bound):** same model and ladder, clean target x.
- **N2N:** classic Noise2Noise (pairs, direct output, no σ input, native noise).
- **NE2NE, R2R, N2S:** as in `baselines/` (Neighbor2Neighbor γ = 2; R2R α = 0.5, 8-draw inference; Noise2Score v2).
- **AVG** (plain averaging) and **BM3D** (`bm3d` 4.0.3 on the a-copy mean, σ_psd = σ/√a; white conditions only,
  as BM3D assumes white noise; for C05 it is run with the same σ and reported as such).
Learned methods are reported with the dial (r chosen as below) and at r = 0 (as published).

## Scoring (BSD68; a ∈ {1, 3, 5, 9}; 10 fixed subsets per a, shared by all methods)
- Ground truth: PSNR of the a-copy estimate (clipped to [0, 1]) and true n_eff = σ²/MSE(estimate, x).
- Ground-truth-free: split-half n_eff of the same estimates (reference = mean of the complementary copies).
- r selection: for each learned method and a, the grid r (0, 1/12, 1/6, 1/4, 1/3, 1/2) maximizing the
  GROUND-TRUTH-FREE split-half n_eff summed over Set12 (no ground truth used); compared with the r maximizing true
  PSNR on BSD68 (oracle).

## Read-outs
- **P1 (estimator):** for OURS at r = 0, every condition, a ∈ {1, 9}: relative bias of the split-half MSE estimate,
  mean over images of (MSE_SH − MSE_GT)/MSE_GT, 95% bootstrap CI. Inside ±5% → **unbiased in practice**.
- **P2 (dial without ground truth):** PSNR of OURS at the Set12-split-half r minus PSNR at the BSD68-oracle r, a = 9,
  every condition. Loss ≤ 0.1 dB → **ground-truth-free selection suffices**. Secondary: closed-form r*(a) from
  (b, v, c) estimated without ground truth vs the ground-truth optimum.
- **P3 (methods, primary condition W05, a = 9):** PSNR(OURS) − PSNR(baseline + dial) for N2N, NE2NE, R2R, N2S, and
  vs AVG, BM3D; paired over 68 images, two-sided Wilcoxon, Holm over the 6. Gap to SUP reported (not tested).
  Secondary: a = 1, 3, 5; conditions W50, W20, C05; r = 0 versions.
- **P4 (averaging-aware loss, W05):** PSNR(AVG9) − PSNR(OURS) at a = 9; PSNR(AVGU) − PSNR(OURS) at a = 1 and 9;
  two-sided Wilcoxon, Holm over the 3. Secondary: other conditions; bias/variance decomposition.
- **PREDICTIONS:** P1 unbiased; P2 ≤ 0.1 dB; OURS beats N2N/BM3D/AVG at a = 9; on white noise NE2NE/R2R/N2S are
  competitive for single images, and with coloured noise (C05) they degrade while OURS does not; AVG9 ≥ OURS at
  a = 9 (lower bias), possibly < OURS at a = 1.

## No-overwrite
Outputs only under `runs/nat/` and `Archive/NAT/`; scripts refuse to overwrite; earlier results hashed
(`Archive/NAT/protected_before.sha256`) and re-verified at the end. Library change (N2N_DSM channel count `ch`,
default 2) verified to leave the default path identical before any run.
