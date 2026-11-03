# RES pre-registration — one denoiser across resolutions (fastMRI, physical noise)

Written 2026-11-03, BEFORE any model of this study was trained or evaluated. Agreed with the user the same day
(physical noise as primary, white noise, arms A–E). Code: `resolution/` (data, models, train, evaluate, report).

## Question
Note 8 (RESEARCH_NOTES) proposes linear-only ways for one self-supervised denoiser to serve several voxel sizes. Does a
model trained at one resolution lose performance at others, and which design avoids the loss?

## Data (`resolution/data.py`)
- fastMRI brain multicoil TRAIN folder (fully sampled; read-only), AXT2 at 220 mm FOV: 77 volumes (320² or 384²
  reconstruction; 384² brought to 320² by keeping the central k-space, same FOV). Seeded split by volume: 62 training,
  15 test volumes; central 8 slices each (496 training, 120 test images).
- Clean image: Inati coil combination, complex, scaled so |x| p99 = 1 per volume. Native noise 0.013–0.055 per
  channel vs 1.0 added at 320 (checked), so the fully sampled image serves as ground truth.
- Resolutions M ∈ {320, 240, 160, 128, 80} at the same FOV (voxel 0.69–2.75 mm) by central k-space truncation.
- **Physical noise (primary):** one complex white k-space noise field per copy (std 1 per channel), shared by all M,
  then truncated: per-channel image noise σ(M) = M/320 (1.0 at 320, 0.25 at 80). N = 10 fixed copies per image.
- **Matched noise (diagnostic, evaluation only):** σ = 0.5 per channel at every M.

## Arms (`resolution/models.py`; same recipe: DSM, total-noise ladder p_raw 0.5, one-shot Tweedie + dial; 2-channel
U-Net 32/32, 3000 iterations, batch 8 × 10 copies, 64² crops, one training matrix per step; seeds 101–103)
- **A** trained at 160 only.
- **B** trained on {80, 160, 320}, fully convolutional, no resolution input.
- **C** as B, plus the voxel size as a second conditioning embedding.
- **D** standardize: every image zero-filled in k-space to the 320 grid (linear), trained/denoised there, brought back.
- **E** classic Noise2Noise (direct, blind, native noise, one other copy as target) on {80, 160, 320} (reference).
Held-out matrices (never trained by B–E): 240 and 128 (in between), with 320 and 80 at the trained edges.

## Read-outs (`resolution/evaluate.py`, `resolution/report.py`)
- n_eff = σ² / MSE against ground truth, per channel, over the head mask (|clean| > 5% of its p99), per test image,
  averaged over the fixed subsets of size a ∈ {1, 3, 5, 9}. Per volume: geometric mean over its 8 images; per arm:
  geometric mean over seeds. The dial r is chosen without ground truth (split-half MSE), leave-one-volume-out.
- **PRIMARY:** n_eff(C) / n_eff(B), physical noise, M ∈ {128, 240}, a ∈ {1, 9}: geometric mean over the 15 test
  volumes, two-sided Wilcoxon over volumes, Holm over 4; equivalence if the 90% bootstrap CI lies within ±5%.
- **SECONDARY:** A/B at every M (cost of single-resolution training); D/B; E/B; the same ratios under matched noise;
  agreement of the split-half n_eff with the ground-truth n_eff (estimator check on a new modality).
- **PREDICTIONS:**
  - C ≈ B at the held-out matrices (fully convolutional + σ conditioning already covers the range); voxel conditioning
    adds little when training spans the scales.
  - A ≈ B at 160. Under physical noise A collapses at 128 and 80, whose noise lies below its ladder (the C3 failure);
    under matched noise its loss there is much smaller. The difference is the share due to noise level, the rest is
    image scale.
  - D < B at low resolutions: zero-filling makes the noise band-limited (coloured) by an M-dependent amount while the
    label stays scalar, and most of the 320 grid carries no information at M = 80.
  - E ≥ B for single maps (a = 1), B > E at a = 9 (as on ASL).

## No-overwrite
Outputs only under `runs/res/`; cache under `~/diffusion_cache/res` (outside OneDrive). Scripts refuse to overwrite.
