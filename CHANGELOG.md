# Diffusion folder — change log

Notes on what CHANGED / why, grouped by the source function each detail came
from. The in-code docstrings now keep only the implemented behavior; the history
lives here. Not a full commit log — just the design rationale and gotchas encoded
in comments that are too long to sit in the code.

## ddpm.py

### `DDPM.__init__` — endpoint knob
`endpoint` is a per-instance A/B knob: controls the fraction of training
iterations forced onto the final timestep t = T-1. Rationale (small toy setup):
with a batch of 1 and a small T, the final reverse step (which assembles the
clean image out of noise) is otherwise drawn only ~1/T of the time. One notebook
uses this to A/B the two regimes on identical data.

### `DDPM.sample` — stable posterior parametrization
Iterates exactly over t (not spaced) so every step stays on the precomputed
schedule; running fewer `steps` just subsamples the walk.

### `DDPM.sample_ddim` — vec docstring derivation
The full derivation of the deterministic (eta=0) DDIM update was long int the
docstring and is summarized here: from the current noisy x at alpha_bar_t,
predict eps_theta, estimate x0_hat via Tweedie, then recombine along the next
subsampled timestep. DDIM trains with the EXACT same objective as DDPM on the
same UNet backbone -- no retraining needed; only the decoder (subsampled
deterministic walk) differs. eta=1 + num_steps=T recovers the DDPM reverse
posterior.

## dsm.py

### `DSM.sample` — sampler / `sample_for_traj` reference
The predictor-corrector sampler does the anneal as an explicit coarse->fine loop
over the L levels rather than a contiguous step budget, to guarantee at least
one corrector block per level (see `dsm.sample_for_traj` for the level index).
`eps_l` in the Langevin step is the SINGLE-step noise magnitude per level,
independent of the contiguous predictor step, so the fine-level run stays pinned
instead of accumulating sqrt(time)*sqrt(alpha) ~ O(sigma_max) random noise.

### `DSMTrainer` — separated from DiffusionTrainer
Bespoke DSM training loop, deliberately NOT reusing `train.DiffusionTrainer`
(leaving the DDPM pipeline untouched). Trains with the multi-level batch regime
("Option A"): each batch element carries its own noise level, so a single
training image swept fine..coarse sigmas each step.

### `DSM.__init__` — device-matching
The sigma schedule is kept on the same device as the net because it is indexed
by i_vec, which the trainer sends to DEVICE -- device-matching avoids a
CPU-index / GPU-tensor gather mismatch.

## train.py

### `DiffusionTrainer.step` — grad clip rationale
Gradient clipping is off by default (grad_clip=0.0). Rationale recorded: it
guards against the occasional huge-gradient timestep that ejects the model from
a sharp low-t basin -- a probabilistic, RNG-triggered event seen ~4500 iterations
into the iteration sweep. Toy value: 1.0.

## n2n.py

### Module header — N2N motivation + pair packing
Framing: general MRI denoising, agnostic to any specific acquisition, only
noisy images available (no clean reference). N2N (Lehtinen 2018) needs two
independent noisy realizations. Pair packing through the shared trainer: both
realizations ride in one (B, 4, H, W) stack so `DiffusionTrainer` (which calls
`model.loss(x0, rng=rng)` on a single tensor) trains either model with no
modification; the notebook builds the stack with `torch.cat([two_ch(y1),
two_ch(y2)], dim=1)`.

### `N2N_DDPM.__init__` / `N2N_DSM.__init__` — task-appropriate schedules
- N2N-DDPM: pure generation is WRONG for N2N because the Tweedie decode divides
  by sqrt(alpha_bar), amplifying the eps residual to image scale at high t and
  ill-conditioning the loss. Hence the small-T/small-beta_end schedule, decoupled
  from ddpm.py's generative defaults.
- N2N-DSM: the sigma schedule must be tied to the real data's noise, not widened:
  sigma_max should sit ABOVE the task sigma so the anneal covers it (no need for a
  huge generative-scale value); diving sigma_min far below the task sigma makes
  conditional denoise 'wander' under the noise floor and the fine levels over-
  stabilize toward a point clean image.
- `task_sigma` pins sigma_min to the task noise floor (backward compatible:
  None keeps the explicit generative-scale defaults verbatim).

### `draw_pair_batch` — fresh-pair regime
Regime-B pair source. Because gt is a known clean phantom, a brand-new pair of
independent noisy realizations (y1, y2) is resynthesized on EVERY call (every
training iteration), mirroring the acquisition of many distinct noisy scans while
sharing one ground truth. Kills the finite-pool overfitting risk vs a frozen
memorizable pair. seed=None by default so every draw is fresh -- opposite of the
fixed-seed pair drawn in the notebook's data cell (which exists only to visualize
one pair).

### `N2N_DDPM.loss` / `N2N_DSM.loss` — anchored target + half-shuffle
These replaced an earlier objective (regress the decode onto the OTHER noisy
realization y2 alone), which was structurally unreachable: x0_hat is decoded only
from one input half and can never match the other independent draw, leaving an
irreducible (n1-n2) floor, so training rewarded suppressing toward nothing rather
than learning gt. The fix: regress onto the symmetric mean tgt = (y1+y2)/2, whose
conditional mean over pairs is exactly gt (finite-pair N2N limit), plus a random
half-shuffle of which realization is the decode input so neither draw is
privileged.

### `N2N_DDPM.t_schedule` — per-t re-weighting
Optional 1/alpha_bar importance weighting that neutralizes the 1/sqrt(alpha_bar)
decode-amplifier bias when t is drawn over a wide range: down-weights the easy
low-t samples (where the decode sits close to y2 already), focusing training
uniformly on all t. Default off (False): with a small T / beta_end the bias is
negligible and no re-weighting is needed.

### `N2N_DDPM.conditional_denoise` — old sampler bug
The previous conditional decoder was WRONG: it called sample_ddim(shape of y),
which seeds from torch.randn -- so the noisy map was DISCARDED and it did
unconditional generation, never using the learned denoiser on the input. Fixed by
diffuse-then-reverse: re-encode y onto the DDPM ladder at a rung whose added
noise sits slightly ABOVE sigma_task, then a deterministic DDIM walk to 0.

### `N2N_DSM.conditional_denoise` — no injected noise
The "predictor-to-Tweedie" anneal deliberately does NOT inject Langevin noise:
unlike unconditional generation (which must build structure from pure noise), y is
already on/inside the data manifold, and injecting sqrt(alpha)*z at every step (a
final application at the finest level landing AFTER the last refinement) injected
unremoved noise that floored RMSE above the task noise. Dropping it turns the
anneal into a convergent deterministic refinement to the Tweedie estimate; the
final deterministic step finishes exactly at x0_hat. `eps` and `rng` args are
vestigial (unused) on this method. Design choice: "Design A" uses the same number
of corrector steps per level (predictor and corrector are the same refinement).

## fid.py

### Full-fidelity math notes
No change-history; retained as-is. Notes kept in code (`_sqrtm_sym`,
`fid_from_stats`): A@B is not symmetric even when A,B are, so use the symmetric
PSD product M = A^{1/2} @ B @ A^{1/2} whose eigenvalues equal those of A@B.

## demo.py / data_loader.py
Template class notes — no removed change history of note. `demo.py` smoke test
trains DDPM and DSM briefly on toy blobs and reports FID.

## unet.py
No change history beyond the design docstring (kept).
## n2n.py — noise labels, real-data objectives (2026-10-20)

### Total-noise conditioning (`N2N_DSM`, `N2N_DDPM`)
The legacy losses perturbed an already-noisy input (σ_y) by σ_add and labelled
the sample with σ_add, so the net at a label only ever saw total noise
√(σ_y² + σ_add²); the samplers read the label as total noise. Consequences
measured on ASL data: the label that matched a raw map was ill-conditioned (σ²
Tweedie factor ≪ σ_y²), and the label actually used over-denoised (contrast
collapse). Passing `sigma_y` switches to total-noise labels: draw σ_tot ≥ σ_y,
add only √(σ_tot² − σ_y²), condition and decode at σ_tot. Fractional levels
(`level_coord`, `t_of_lambda`) avoid rounding; the sinusoidal embedding takes
them unchanged. DDPM works in λ = √(1−ᾱ)/√ᾱ, where lifting to λ and scaling by
√ᾱ lands exactly on the forward marginal. Legacy paths kept bit-identical.

### `denoise_to` — one DDIM step to a target noise
`out = x0_hat + r·(y − x0_hat)` is exactly one deterministic DDIM step from the
map's own noise σ to r·σ. r = 1/3 matches the noise floor of a leave-one-out
mean of 9 repetitions; it also explains the empirically best blend ≈ 2/3 found
before the fix.

### `loss_subset`, `noise_norm`
`loss_subset` trains on means of random m-subsets (input noise σ_y/√m, target =
mean of the rest) so one model can denoise an average; `m=` lets the caller
balance noise-level coverage. `noise_norm` weights each sample by
(σ_ref/σ_tot)^p (p = 1 L1, 2 MSE) — the x0-space form of σ²-weighted score
matching — so the noisiest slices no longer dominate the loss.

### Channel layout gotcha
`loss_multireal` reads `(B, 2N, H, W)` as `[Re1, Im1, Re2, Im2, ...]`. Packing
with `permute(0, 2, 1, ...)` produces `[Re1..ReN, Im1..ImN]`, which silently
trains on Re/Re and Im/Im "realizations". `experiments/_r3_train.pack` uses the
correct `reshape(S, N*2, H, W)`.
