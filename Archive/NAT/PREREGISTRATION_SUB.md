# NAT-SUB pre-registration (exploratory addendum) — does seeing all copies close the gap to BM3D?

Written 2026-11-01, BEFORE these models were trained or evaluated. Motivation: NAT (2026-11-01) — the per-copy model
reaches the supervised per-copy bound, yet BM3D on the copy mean wins by 1.85 dB at a = 9 (W05).

## Models (natural/train.py; same U-Net, budget and data as NAT; conditions W05 and C05; seed 101)
- **SUB:** denoise the MEAN of a random m-copy subset (m ~ U{1..9}), conditioned on σ/√m (ladder bottom σ/3), target =
  mean of the other 10 − m copies (N2N, `N2N_DSM.loss_subset`), MSE.
- **SUBSUP:** the same with the clean image as target (supervised upper bound for denoising the mean).
Decoding: the a-copy mean is denoised once at σ/√a, then blended with r (r chosen without ground truth on Set12, as NAT).

## Read-outs (BSD68, same subsets as NAT)
- PRIMARY: PSNR(SUB) − PSNR(ours) at a = 9, W05; two-sided Wilcoxon.
- SECONDARY: SUB vs BM3D and vs SUBSUP at a = 1, 3, 5, 9; condition C05; SUBSUP vs the supervised per-copy bound.
- PREDICTION: SUB > ours at a ≥ 3 and approaches or exceeds BM3D; at a = 1 SUB ≈ ours (the same task).
