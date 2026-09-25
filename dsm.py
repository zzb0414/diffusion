"""
Denoising Score Matching (DSM) / Score-based generative model — SMLD, Song & Ermon 2019.

Gaussian perturbation with a geometric sigma schedule:
    q_sigma(x_tilde | x) = N(x_tilde ; x, sigma^2 I)
    perturbed sample      x_tilde = x + sigma * eps
    DSM loss (SMLD Eq. 34)  || sigma * s_theta(x_tilde, sigma) + eps ||^2
                              (score network predicts the noise direction)

The network outputs the (denoising) score per channel (2 channels).
Sampling uses annealed overdamped Langevin dynamics (SMLD Algorithm 1).
The sigma schedule (number of levels L + sigma_min/max) is adjustable.
"""
import torch
import torch.nn.functional as F

from unet import UNet


def geometric_sigma_schedule(L, sigma_min=0.01, sigma_max=50.0):
    """L logarithmically-spaced noise levels from sigma_min to sigma_max."""
    return torch.logspace(
        torch.log10(torch.tensor(sigma_min)),
        torch.log10(torch.tensor(sigma_max)),
        L,
    )


class DSM:
    """Score-based model: UNet backbone outputting a score; annealed Langevin sampler."""

    def __init__(self, L=10, sigma_min=0.01, sigma_max=50.0, backbone=UNet()):
        self.net = backbone
        self.L = L
        self.sigmas = geometric_sigma_schedule(L, sigma_min, sigma_max)

    def perturb(self, x, i, rng=None):
        """Perturb x at noise level i (scalar int). Returns (x_tilde, eps, sigma)."""
        sigma = self.sigmas[i].item()
        eps = torch.randn_like(x) if rng is None else rng.randn_like(x)
        x_tilde = x + sigma * eps
        return x_tilde, eps, sigma

    def loss(self, x, rng=None):
        """Random noise level i (one per batch), perturb, predict score, use SMLD objective."""
        i = torch.randint(0, self.L, (1,), device=x.device).item()
        x_tilde, eps, sigma = self.perturb(x, i, rng)
        scores = self.net(x_tilde, torch.full((x.shape[0],), i, device=x.device, dtype=torch.long))
        # score-matching loss: || sigma * s_theta + eps ||^2  (SMLD Eq. 34)
        return F.mse_loss(scores * sigma, -eps)

    @torch.no_grad()
    def sample(self, shape, steps=None, eps=2e-5, T_anneal=100, device="cpu", rng=None):
        """Anneal over all noise levels, each via overdamped Langevin:

            x = x + 0.5 * eps_l * s_theta(x, sigma) + sqrt(eps_l) * z
        where eps_l = 2*(sigma(l)/sigma_max)^2 * eps and eps_l scaled per level.
        """
        steps = steps or T_anneal * self.L
        sigmas = self.sigmas
        x = torch.randn(shape, device=device) * sigmas[-1]  # start from highest noise
        i_max = self.L - 1
        for step in range(steps):
            # anneal: dwell proportionally longer per level (SMLD Algorithm 1)
            i = min(i_max, int(step * self.L / steps))
            s = sigmas[i].item()
            # SMLD: step size alpha_i = eps * (sigma_i / sigma_max)^2
            alpha = eps * (s / sigmas[-1].item()) ** 2
            ts = torch.full((shape[0],), i, device=device, dtype=torch.long)
            grad = self.net(x, ts)  # score field at current level
            z = torch.randn_like(x)
            x = x + 0.5 * alpha * grad + (alpha ** 0.5) * z
        return x

    @torch.no_grad()
    def score(self, x, i):
        """Expose score directly for an external Langevin loop."""
        return self.net(x, torch.full((x.shape[0],), i, device=x.device, dtype=torch.long))