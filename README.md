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
| [`n2n.py`](n2n.py) | `N2N_DDPM` / `N2N_DSM` — Tweedie Noise2Noise objectives on each backbone, plus `pack_pair`, `draw_pair_batch`, and conditional/unconditional denoise samplers. |
| [`fid.py`](fid.py) | FID metric via Inception-v3, with a symmetric square-root form for the covariance product. |
| [`data_loader.py`](data_loader.py) | `ChannelImageDataset` template (subclass for your storage format) + DataLoader helpers. |
| [`demo.py`](demo.py) | Standalone smoke test: trains DDPM & DSM on toy blobs, samples, reports FID. `python demo.py`. |
| [`demo.ipynb`](demo.ipynb) | Notebook showcase of DDPM / DDIM / DSM on a Shepp-Logan phantom. |
| [`n2n.ipynb`](n2n.ipynb) | Notebook for the Noise2Noise denoising pipeline. |
| [`CHANGELOG.md`](CHANGELOG.md) | Design rationale and change history grouped per source function. |
| [`saved_models/`](saved_models/) | Pretrained checkpoints (DDPM/DSM, and the N2N variants). |
| `shepp_logan_*.png` | Sample outputs / loss plots from the notebooks. |

## Quick start

```bash
python demo.py        # trains DDPM + DSM briefly on toy blobs and reports FID
```

Then open the notebooks — `demo.ipynb` for the generative showcase, `n2n.ipynb` for Noise2Noise denoising.

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

## Data
`data_loader.py` is a template for 2-channel images. Subclass `ChannelImageDataset` and implement `_load` to wire your own storage (npz, nifti, h5, folder of `.npy`); `build_dataloader` / `dataloader_from_dir` wrap it.

## Notes
- Complex images are stored as two real channels (Real, Imag) in `[-1, 1]`.
- FID maps `(Re, Im, Re)` to RGB for Inception-v3; if `torchvision` is unavailable, FID degrades to `float('nan')`.
- Run any notebook in this folder as its working directory so the modules import.