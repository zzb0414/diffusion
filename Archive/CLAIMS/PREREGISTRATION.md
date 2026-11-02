# CLAIMS pre-registration — closing the evidence gaps of the claims ledger (2026-11-02)

Written 2026-11-02, BEFORE any of these models was trained or evaluated. Claims ledger: `docs/RESEARCH_NOTES.md`
note 10. Each experiment tests one leg of one claim.

## LC (check, not a test): learning curve of the natural-image models
W05: ours, sup, sub, subsup evaluated at iterations 1000 / 2000 / 3000 (saved checkpoints), BSD68, a = 1 and 9, r
chosen without ground truth (as NAT). Question: were the models still improving at 3000? Reported descriptively.

## C2 — the score parametrization lowers bias (ASL, as DEC part A)
New arm **RESID**: x̂₀ = x̃ + net(x̃, σ) (residual, no σ² scaling), noise ladder, MSE noise-normalized, dataset-1
training, seeds 101–103, 2000 iterations, scored like DEC A (`_dec_eval.py`, one-shot + cross-fitted r) against DSM
(x̃ + σ² net) and DIR_L (direct net, same ladder) from DEC.
- PRIMARY: n_eff(RESID)/n_eff(DSM), dataset 1, a = 9; 90% CI inside [0.95, 1.05] → **equivalent** (residual learning
  is the mechanism; the σ² scaling adds nothing); below → σ² scaling contributes; above → residual better.
- SECONDARY: a = 1; RESID/DIR_L; datasets 2–3; bias² / variance decomposition (`_dec_bvc.py` method).
- PREDICTION (linear toy, `docs/THEORY.md`): a residual parametrization starts at the identity like the score form,
  so it under-smooths rather than over-smooths: low bias like DSM → equivalent or close.

## C7 — when does denoising the copy mean win? (natural images)
OURS (per copy + dial) and SUB (denoise the copy mean) trained on the first n ∈ {25, 100, 400} Train400 images, at
W05 (σ 0.5) and W20 (σ 2.0); everything else as NAT (3000 iterations, seed 101). n = 400 reuses NAT runs where they
exist (W05 ours, W05 sub, W20 ours); W20 SUB at n = 400 is new.
- PRIMARY: PSNR(SUB) − PSNR(OURS) at a = 9, BSD68, for each (n, condition); two-sided Wilcoxon, Holm over the 6 cells.
- PREDICTION: SUB's advantage shrinks as n decreases and as noise increases; in the most ASL-like cell (n = 25, W20)
  OURS ties or wins.

## C3 — training across noise levels gives robustness to shift (natural images)
Trained at W05 (σ 0.5, white): OURS (ladder, NAT run) vs OURS-NOLADDER (same, p_raw = 1: native noise only, new) and
classic N2N (NAT run, native, no σ input). Tested on W05 (in distribution) and shifted: W035 (σ 0.35), W07 (σ 0.7),
C05 (σ 0.5, ASL-coloured); conditioned on the test σ where the model takes σ.
- PRIMARY: PSNR(OURS) − PSNR(OURS-NOLADDER) on the shifted sets (W07, C05) at a = 1 and 9; two-sided Wilcoxon, Holm over
  the 4. In distribution (W05) reported for reference.
- PREDICTION: the ladder helps under shift (above all W07, a noise level the no-ladder model never saw) and is neutral
  in distribution.

## No-overwrite
New outputs only under `runs/dec/A/resid_*`, `runs/dec/A/eval/resid_*`, `runs/nat/lc/`, `runs/nat/c7/`, `runs/nat/c3/`,
`runs/nat/eval_*`, `Archive/CLAIMS/`; scripts refuse to overwrite. Library change (`N2N_DSM(param="residual")`) verified
to leave the default path identical (deterministic CPU) before the RESID trainings start.
