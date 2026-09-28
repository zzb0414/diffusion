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
        # Keep the sigma schedule on the same device as the net (which follows
        # whatever DEVICE the model is built/moved to). It is indexed by i_vec,
        # and i_vec is sent to DEVICE by the trainer, so device-matching here
        # avoids a CPU-index / GPU-tensor (or vice-versa) gather mismatch.
        self.sigmas = geometric_sigma_schedule(L, sigma_min, sigma_max).to(next(backbone.parameters()).device)

    def perturb(self, x, i_vec, rng=None):
        """Perturb x at per-sample noise levels i_vec (shape (B,), long).

        Each batch element gets its own sigma = sigmas[i_vec[k]]. Returns
        (x_tilde, eps, sigma) where sigma is a per-sample vector.
        """
        sigma = self.sigmas[i_vec]                      # (B,) on the same device as sigmas
        eps = torch.randn_like(x) if rng is None else rng.randn_like(x)
        x_tilde = x + sigma.view(-1, 1, 1, 1) * eps
        return x_tilde, eps, sigma

    def loss(self, x, i_vec=None, rng=None):
        """SMLD objective with one (possibly different) noise level per batch element.

        i_vec: shape (B,) of per-sample levels; if None, one random level is drawn
        and shared across the batch (previous behaviour). Net is conditioned on
        i_vec so each element's sigma drives its own score estimate.
        """
        if i_vec is None:
            i_vec = torch.randint(0, self.L, (x.shape[0],), device=x.device)
        x_tilde, eps, sigma = self.perturb(x, i_vec, rng)
        scores = self.net(x_tilde, i_vec)              # per-sample conditioning
        # score-matching loss: || sigma * s_theta + eps ||^2  (SMLD Eq. 34)
        return F.mse_loss(scores * sigma.view(-1, 1, 1, 1), -eps)

    @torch.no_grad()
    def sample(self, shape, steps=None, eps=2e-5, T_anneal=100, device="cpu",
               rng=None, corrector_steps=5):
        """SMLD Algorithm 2 predictor-corrector sampler (Song & Ermon 2019).

        Each noise level first runs a *predictor* step that anneals the walk to
        the next (finer) sigma, then a *corrector* block of `corrector_steps`
        overdamped Langevin iterations at FIXED sigma. The corrector settles the
        walk onto the sigma_i + quantization manifold before the next descent,
        which is exactly what prevents the finest levels from collapsing the
        already-generated image (see dsm.py header / Session notes).

        Level order DESCENDS from coarse (sigma_max) to fine (sigma_min): we
        start at the highest noise to discover global structure, then anneal
        down to low noise for detail refinement. The seed is drawn at sigma_max
        (i = i_max, the physical image scale).

        Overdamped Langevin corrector step (SMLD Alg 2):
            x = x + eps_l * s_theta(x, sigma) + sqrt(2*eps_l) * z
        with eps_l = 2*(sigma/sigma_max)^2 * eps. NOTE the 'eps_l' here is the
        SINGLE-step noise magnitude for this level — it is independent of the
        (contiguous) predictor step, so the fine-level run stays pinned instead
        of accumulating sqrt(time) * sqrt(alpha) ~ O(sigma_max) random noise.
        """

        def langevin_step(x, i, s):
            alpha = eps * (s / sigmas[-1].item()) ** 2
            ts = torch.full((shape[0],), i, device=device, dtype=torch.long)
            grad = self.net(x, ts)          # score field at current level
            z = torch.randn_like(x)
            return x + 0.5 * alpha * grad + (alpha ** 0.5) * z, alpha, ts

        sigmas = self.sigmas
        # To guarantee at least one corrector block per level, do the anneal as
        # an explicit coarse->fine loop over the L levels rather than a
        # contiguous step budget. See dsm.sample_for_traj for the level index.
        steps = steps or T_anneal * self.L
        x = torch.randn(shape, device=device) * sigmas[-1]  # start from highest noise
        i_max = self.L - 1
        for i in range(i_max, -1, -1):
            s = sigmas[i].item()
            # ---- predictor: one anneal step from coarse sigma_(i) to fine sigma_(i-1) ----
            x, alpha, ts = langevin_step(x, i, s)
            # ---- corrector: settle onto the sigma_i manifold ----
            for _ in range(corrector_steps):
                x, alpha, ts = langevin_step(x, i, s)
        return x

    @torch.no_grad()
    def score(self, x, i):
        """Expose score directly for an external Langevin loop."""
        return self.net(x, torch.full((x.shape[0],), i, device=x.device, dtype=torch.long))


class DSMTrainer:
    """Bespoke DSM training loop, separated from DiffusionTrainer.

    Trains with the multi-level batch regime (Option A): each batch element
    carries its own noise level via i_vec (shape (B,)), so a single training
    image pre-augmented into a minibatch sweeps fine..coarse sigmas each step.
    Does NOT reuse DiffusionTrainer, leaving the DDPM pipeline untouched.
    """

    def __init__(self, model, lr=3e-4, device="cpu", grad_clip=1.0):
        self.model = model
        self.device = device
        self.grad_clip = grad_clip
        self.opt = torch.optim.Adam(model.net.parameters(), lr=lr)

    def step(self, x0, i_vec, rng=None):
        """One gradient step at fixed per-sample levels i_vec (shape (B,), long)."""
        x0 = x0.to(self.device)
        i_vec = i_vec.to(self.device)
        self.opt.zero_grad()
        loss = self.model.loss(x0, i_vec=i_vec, rng=rng)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.net.parameters(),
                                           self.grad_clip)
        self.opt.step()
        return loss.item()