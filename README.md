# diffusion

Diffusion models and Noise2Noise (N2N) denoising for 2-channel (Real/Imag) MRI-like images, in PyTorch.

This folder implements three generative/denoising backbones — DDPM, DDIM, and a score-based model (DSM/SMLD) — sharing one U-Net, and mounts a **Noise2Noise MRI-denoising** objective on top of them. Everything works with 2-channel complex images shaped `(B, 2, H, W)` and is treated channel-agnostically (no magnitude-domain/Rician assumptions).

## Repository layout

| File | Purpose |
|------|---------|
| [`unet.py`](unet.py) | 5-layer time-conditioned U-Net (`UNet`), shared backbone for DDRM & DSM. |
| [`ddpm.py`](ddpm.py) | `DDPM` — Ho et al. 2020: linear beta schedule, epsilon prediction, `perturb`/`loss`/`sample` (reverse diffusion) and `sample_ddim` (deterministic DDIM, Song et al. 2021). |
| [`dsm.py`](dsm.py) | `DSM` — Song & Ermon 2019: geometric sigma schedule, score matching, annealed predictor-corrector Langevin sampling. `DSMTrainer` for the multi-level batch regime. |
| [`train.py`](train.py) | `DiffusionTrainer` — shared `loss(x)`-driven loop that trains either a DDPM or DSM model. |
| [`n2n.py`](n2n.py) | `N2N_DDPM` / `N2N_DSM` — Tweedie Noise2Noise objectives on each backbone (total-noise labels, multi-realization and subset-averaging losses), plus `pack_pair`, `draw_pair_batch`, and denoise samplers (`denoise_to`, conditional/unconditional). |
| [`CHANGELOG.md`](CHANGELOG.md) | Design rationale and change history grouped per source function. |
| [`experiments/`](experiments/) | Scripts for the real-ASL N2N experiments, mapped to the findings they support — see its [README](experiments/README.md). |
| [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) | **Start here.** Findings with status labels (established / supported / hypothesis) and the draft roadmap toward a full article. |
| [`docs/ARGUMENT_N2N_vs_averaging.md`](docs/ARGUMENT_N2N_vs_averaging.md) | Evidence summary: averaging N2N-denoised acquisitions vs plain averaging, generated from the results. |
| [`docs/ABSTRACT_draft.md`](docs/ABSTRACT_draft.md) | Generated draft of the ISMRM abstract (the submitted version is frozen). |
| [`WEEKEND_ITERATIONS.md`](WEEKEND_ITERATIONS.md) | Experiment log (append-only; sections generated from result files). Entries before R3 used flawed training — see *Known limits*. |
| [`legacy/`](legacy/) | The original Shepp-Logan foundation: `demo.py` / `demo.ipynb` (DDPM / DDIM / DSM showcase), `n2n.ipynb`, `fid.py`, `data_loader.py`, sample figures. Weights (`legacy/saved_models/`) stay local. Run from the repo root with `legacy/` on the path. |
| `runs/`, `checkpoints/`, `Archive/` | Per-run configs, logs and stdout; checkpoint manifests with sha256; pre-registrations and verbatim stdout. Weights, CSVs and figures stay local (see `.gitignore`). |

## Quick start

```bash
PYTHONPATH=. python legacy/demo.py   # trains DDPM + DSM briefly on toy blobs and reports FID
```

The real-ASL experiments start from [`experiments/README.md`](experiments/README.md).

## Models

### DDPM / DDIM (`ddpm.py`)
- Forward process `q(x_t | x_0) = N(sqrt(alpha_bar_t)·x_0, (1−alpha_bar_t)·I)`.
- Loss is epsilon prediction: `|| eps − eps_theta(x_t, t) ||²`.
- `sample` is the standard reverse iterative denoiser; `sample_ddim` runs the **same weights** through a deterministic few-step decoder (no retraining). `eta=0` is classic DDIM; `eta=1` + full steps recovers the DDPM posterior.

### DSM / SMLD (`dsm.py`)
- Additive Gaussian perturbation `x_tilde = x + sigma·eps` over a geometric sigma ladder.
- Loss is score matching: `|| sigma·s_theta(x_tilde, sigma) + eps ||²`.
- `sample` annealed predictor-corrector Langevin dynamics.

### Noise2Noise denoising (`n2n.py`)
The motivating setup for this folder: **only noisy images are available** — no clean reference. N2N needs two independent noisy realizations of the same object; a net trained to map one to the other learns the clean image in expectation.

- **Pair packing**: both realizations ride in one `(B, 4, H, W)` stack so the shared `DiffusionTrainer` trains either model unchanged.
- **Tweedie decode**: DDPM `x0_hat = (x_t − sqrt(1−alpha_bar)·eps)/sqrt(alpha_bar)`; DSM `x0_hat = x_tilde + sigma²·score`.
- **Anchored target**: regress the decode onto the symmetric mean `(y1+y2)/2` (whose conditional mean is the clean image), with a random half-shuffle of which realization is the decode input.
- `conditional_denoise(..., sigma_task, ...)` recovers a clean map from one held-out noisy image (diffuse-then-reverse / diffuse-then-anneal). `unconditional(...)` is reverse generation from pure noise.

#### Noise labels: condition on TOTAL noise
A training input is already noisy (σ_y), so perturbing it by σ_add leaves **total** noise √(σ_y² + σ_add²). The original losses labelled the sample by σ_add only, while the samplers read the label as total noise — a train/test mismatch that also made the label mean something different for every noise level. Pass the acquisition noise `sigma_y` (per Re/Im channel, estimated once per slice from the spread of its realizations) and the net is conditioned on, and Tweedie-decoded with, the total level; only √(σ_tot² − σ_y²) is ever added, and a raw map sits exactly at its own level (fractional levels via `N2N_DSM.level_coord` / `N2N_DDPM.t_of_lambda`, no rounding).

- `N2N_DSM.loss_multireal(x, sigma_y=...)` — one realization in, leave-one-out mean of the others as target.
- `N2N_DSM.loss_subset(x, sigma_y, ...)` — denoise the mean of a random m-subset; target = mean of the rest.
- `noise_norm=True, sigma_ref=...` — per-sample weight (σ_ref/σ_tot)^p (p = 1 L1, 2 MSE) so errors at every noise level count equally.
- `N2N_DDPM.loss(pair, sigma_y=...)` — the same fix for the DDPM backbone, in the VE-equivalent coordinate λ = √(1−ᾱ)/√ᾱ.
- `denoise_to(y, sigma_task, target_ratio)` (both backbones) — evaluate at the map's own noise, nothing injected, and take one deterministic DDIM step to `target_ratio × sigma_task`: `out = x0_hat + target_ratio·(y − x0_hat)`. `0` is the full Tweedie estimate. The best ratio depends on how many denoised maps are averaged afterwards, not on the training target (see `docs/RESEARCH_NOTES.md`, notes 2–4).

Without `sigma_y` every loss falls back to the legacy behaviour (kept bit-identical so logged runs reproduce); legacy samplers are marked as such.

## Results on real ASL data
Ten mandated repetitions per slice, 34 slices. Denoising each repetition with the N2N model and then averaging beats plain averaging of the same repetitions on every slice, in a pre-registered 5-fold cross-validation where each slice is scored only by a model that never saw it; most of that gain is also available to an idealised Gaussian smoother, and the network's own edge comes from removing coarse-scale noise. Numbers, method, limits and reproduction: [`docs/ARGUMENT_N2N_vs_averaging.md`](docs/ARGUMENT_N2N_vs_averaging.md).

```bash
python experiments/_cv_train.py --fold K      # K = 0..4, the fixed R3 recipe
python experiments/_cv_eval.py                # pre-registered endpoints, figures, maps
python experiments/_cv_report.py              # log section + argument document
```

## Data
The ASL data (`../asl_for_diffusion/dataset{1,2,3}.npy`) is not included. For other storage formats, `legacy/data_loader.py` is a template: subclass `ChannelImageDataset` and implement `_load` (npz, nifti, h5, folder of `.npy`); `build_dataloader` / `dataloader_from_dir` wrap it.

## Notes
- Complex images are stored as two real channels (Real, Imag) in `[-1, 1]`.
- FID maps `(Re, Im, Re)` to RGB for Inception-v3; if `torchvision` is unavailable, FID degrades to `float('nan')`.
- Run any notebook in this folder as its working directory so the modules import.

## Known limits
- **Superseded runs.** Log entries before R3 (R0–R2, SR_SWEEP, FRONTIER, SRSTR_01) trained with legacy added-noise labels **and** a channel layout that paired Re/Re and Im/Im instead of Re/Im (`pack_joint` / `pack` built `[Re1..ReN, Im1..ImN]` where `loss_multireal` expects `[Re1, Im1, Re2, Im2, ...]`). Their conclusions do not carry over; their scripts and outputs were removed on 2026-10-29 (the log keeps their sections as history). The local ASL notebooks still contain that `pack_joint`.
- **DDPM total-noise path** is implemented and unit-tested (rung mapping, forward-marginal lifting, decoders) but not yet trained or validated on data.
- **Experiment scripts** hardcode the repo root as an absolute Windows path (`DIFF = r"..."` near the top) — edit it for your clone. The ASL data is not included.
- **Not committed:** model weights (sha256 in `checkpoints/*/MANIFEST.md`), per-slice CSVs and figures. CSVs and PNGs written on the originating workstation are encrypted at rest (E-SafeNet) and unreadable elsewhere; every number is in the plain-text logs and `docs/`. Earlier commits contain encrypted R3/R4 figures; they are no longer tracked.
- **Local checkpoints** were pruned on 2026-10-29 to the iterations that reported results use (rules in `experiments/_prune_runs.py`).