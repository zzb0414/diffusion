"""
Denoising Diffusion Probabilistic Model (DDPM) - Ho et al. 2020.

Reference equations:
    q(x_t | x_0) = N(x_t ; sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t) I)
    x_t          = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps
    loss         = || eps - eps_theta(x_t, t) ||^2
    sampling     = reverse iterative denoising (epsilon prediction)

Input is a 2D image with 2 channels (Real / Imag), treated channel-agnostically.
The diffusion schedule (T steps + beta range) is adjustable.
"""
import torch
import torch.nn.functional as F

from unet import UNet


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Linear beta schedule from DDPM."""
    return torch.linspace(beta_start, beta_end, T)


class DDPM:
    """DDPM: holds the UNet backbone + a fixed Gaussian noise schedule."""

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02,
                 backbone=UNet(), endpoint=0.0):
        self.net = backbone
        self.T = T
        # Fraction of training iterations forced onto the final timestep t = T-1.
        # 0.0 = standard uniform-t sampling; >0 forces that fraction onto the last
        # step. A per-instance A/B knob.
        self.endpoint = endpoint

        # Move the schedule onto the backbone's device so sampling (which scales x
        # by betas[t]/alphas[t]) never hits a CPU-vs-GPU mismatch. nn.Module has no
        # `.device` attr, so read it from the first parameter (CPU fallback).
        _dev = next(backbone.parameters()).device
        betas = linear_beta_schedule(T, beta_start, beta_end).to(_dev)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar

        # Precomputed quantities used often.
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)               # (T,)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)  # (T,)
        self.sqrt_one_over_alpha_bar = torch.sqrt(1.0 / alpha_bar)   # (T,)
        self.posterior_variance = torch.zeros_like(betas)

    # ---- training helpers ------------------------------------------------
    def perturb(self, x0, t, rng=None):
        """Return (x_t, eps) corresponding to the forward process at timestep t."""
        eps = torch.randn_like(x0) if rng is None else rng.randn_like(x0)
        sa = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        so = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        x_t = x0 * sa + eps * so
        return x_t, eps

    def loss(self, x0, rng=None):
        """Random t in [0, T), perturb, predict epsilon, return MSE.

        Standard uniform t unless `self.endpoint > 0`: then that fraction of
        iterations is pinned to t = T-1.
        """
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device, dtype=torch.long)
        if self.endpoint > 0 and torch.rand(1, device=x0.device).item() < self.endpoint:
            t = torch.full_like(t, self.T - 1)
        x_t, eps = self.perturb(x0, t, rng)
        eps_pred = self.net(x_t, t)
        return F.mse_loss(eps_pred, eps)

    # ---- sampling -------------------------------------------------------
    @torch.no_grad()
    def sample(self, shape, steps=None, device="cpu", rng=None):
        """Reverse process (DDPM Algorithm 2): start from noise, denoise stepwise.

        Epsilon-prediction posterior update iterated exactly over t (not spaced):
            x_{t-1} = sqrt(1/alpha_bar_t) * ( x_t - sqrt(1-alpha_bar_t) * eps_theta )
                      + sigma_t * z
        Running fewer `steps` steps just subsamples the reverse walk.
        """
        steps = steps or self.T
        x = torch.randn(shape, device=device)
        for i in range(min(steps, self.T)):
            t = self.T - 1 - i                    # walk t from T-1 down to 0
            ts = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps_pred = self.net(x, ts)
            b = self.betas[t]
            al = self.alphas[t]                     # alpha_t (NOT alpha_bar)
            so = self.sqrt_one_minus_alpha_bar[t]   # sqrt(1 - alpha_bar_t)
            mean_coef = torch.rsqrt(al + 1e-8)
            x = mean_coef * (x - (b / so) * eps_pred)
            if t > 0:
                x = x + torch.sqrt(self.betas[t]) * torch.randn_like(x)
        return x

    @torch.no_grad()
    def sample_ddim(self, shape, num_steps=None, eta=0.0, device="cpu", rng=None):
        """Reverse process (DDIM, Song et al. 2021): deterministic denoising along a
        SUBSAMPLED trajectory of the T diffusion timesteps.

        Uses the same epsilon-prediction weights as ``sample`` (no retraining); only
        the decoder changes: we pick `num_steps` evenly-spaced timesteps tau across
        the full ladder and run a deterministic epsilon-prediction update in fewer
        passes. eta=0 -> fully deterministic (classic DDIM); eta=1 + num_steps=T
        recovers the DDPM reverse posterior.

        Update (epsilon parametrization): from x at alpha_bar_t, predict eps_theta,
        estimate the clean image x0_hat = (x - sqrt(1-alpha_bar_t)*eps_theta)/
        sqrt(alpha_bar_t), then recombine along the next subsampled timestep tau_next.

        Args:
            shape:    output shape, e.g. (B, 2, H, W).
            num_steps: number of subsampled reverse timesteps (default self.T).
            eta:      stochasticity of the sampler, 0..1.
            device:   device to create x on.
            rng:      optional RandomGenerator.
        """
        num_steps = num_steps or self.T
        x = torch.randn(shape, device=device, generator=rng)

        # Evenly-spaced subset of the T schedule, ending at tau=0 (t=T-1? the top,
        # tau=0 -> the clean image). Flip so we walk from the top of the ladder down.
        if num_steps >= self.T:
            timesteps = torch.arange(self.T, dtype=torch.long, device=device)
        else:
            timesteps = torch.linspace(0, self.T - 1, num_steps, device=device)
            timesteps = timesteps.round().long()
        timesteps = timesteps.flip(0)

        for i in range(num_steps):
            t_cur = timesteps[i].item()
            t_next = timesteps[i + 1].item() if i + 1 < num_steps else 0
            ts = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            eps_pred = self.net(x, ts)                      # shared epsilon-predictor

            ab_t    = self.alpha_bar[t_cur]                 # alpha_bar_t
            ab_t1   = self.alpha_bar[t_next]                # alpha_bar_{t1} (smaller)
            sab_t   = self.sqrt_alpha_bar[t_cur]
            s1ma_t  = self.sqrt_one_minus_alpha_bar[t_cur]
            s1ma_t1 = self.sqrt_one_minus_alpha_bar[t_next]

            # Tweedie / epsilon-form clean estimate, then deterministic recombination.
            x0_hat = (x - s1ma_t * eps_pred) / sab_t
            sigma_t = eta * s1ma_t1 / (s1ma_t + 1e-8) * torch.sqrt(
                torch.clamp(1 - ab_t / (ab_t1 + 1e-8), min=0.0))
            coef = s1ma_t1 ** 2 - sigma_t ** 2
            x = sab_t * x0_hat + torch.sqrt(coef.clamp(min=0.0)) * eps_pred
            if eta > 1e-12:
                # DDPM-style stochastic correction, zeroed out when eta=0.
                x = x + sigma_t * torch.randn_like(x)
        return x