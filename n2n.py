"""
Noise2Noise (N2N, Lehtinen et al. 2018) objectives built on the diffusion stack
in this folder (``ddpm.py`` / ``dsm.py`` / ``unet.py`` / ``train.py``).

General MRI denoising task, agnostic to any specific acquisition/application:
the motivating constraint is that ONLY NOISY images are available -- there is no
clean high-SNR reference. N2N needs only two INDEPENDENT noisy realizations of
the same object: a network trained to map realization_1 -> realization_2 learns
the clean image in expectation.

We mount this on the existing DDPM/DSM machinery as a *Tweedie* N2N objective.
Each training sample is an independent noisy PAIR ``(y1, y2)`` of the same complex
phantom. For a batch element (complex, unpacked to 2-channel Real/Imag):
    y1  ---- perturb(t) ---->  x_t = sqrt(alpha_bar_t)*y1 + sqrt(1-alpha_bar_t)*eps
    net(x_t, t) = eps_pred
    x0_hat = Tweedie(eps_pred)          (the model's MAP clean estimate of x_t)
    loss  = || x0_hat - y2 ||^2         (regress the decode onto the OTHER realization)
Because ``y2`` is an independent noisy draw whose noise has zero mean, this
regression learns the clean image in expectation -- the N2N trick. The clean
``gt`` is used ONLY for validation.

PAIR PACKING through a shared ``loss(x0)`` trainer: the 4-channel stack carries
both realizations side-by-side in the channel dim of one ``(B, 4, H, W)``
Real/Imag stack: [b, 0:2] = y1 (the perturbed input), [b, 2:4] = y2 (the target).
``loss`` slices the two halves itself, so the shared ``DiffusionTrainer`` trains
either model with no modification.
"""
import math

import torch
import torch.nn.functional as F

from unet import UNet
from ddpm import DDPM
from dsm import DSM


def _dev(net):
    """Device of an nn.Module: read from its first parameter (CPU fallback)."""
    return next(net.parameters()).device


def pack_pair(y1, y2):
    """Concatenate two 2-channel stacks into a (B, 4, H, W) training pair.

    y1 / y2: (B, 2, H, W) Real/Imag stacks of two INDEPENDENT noisy realizations
    of the same object. Returns (B, 4, H, W) with [:, 0:2] = y1 (input) and
    [:, 2:4] = y2 (target).
    """
    return torch.cat([y1, y2], dim=1)


def draw_pair_batch(gt, snr, batch=4, ref_mag=1.0, device="cpu"):
    """Resynthesize a fresh minibatch of INDEPENDENT N2N pairs from a clean phantom.

    Because ``gt`` is a KNOWN clean complex phantom, we can draw a brand-new pair
    of independent noisy realizations ``(y1, y2)`` from it on EVERY call (and so
    on every training iteration), mirroring many distinct noisy scans while still
    sharing the same ground truth. This removes the finite-pool overfitting risk.

    Noise model matches ``add_complex_noise``: complex additive Gaussian
    ``noise = (n_re + 1j*n_im)/sqrt(2) * sigma`` with ``sigma = ref_mag/snr``.

    gt: (H, W) complex clean phantom.
    snr: target magnitude SNR (sigma = ref_mag/snr).
    Returns (B, 4, H, W) packed Real/Imag stack, [:, 0:2]=y1 input, [:, 2:4]=y2.
    """
    if not torch.is_complex(gt):
        raise TypeError("gt must be a complex tensor (a complex phantom).")
    gt = gt.to(device)
    tgt = gt.reshape(-1)
    b = batch
    sigma = ref_mag / snr
    numel = tgt.numel()
    # n1 (input half) and n2 (target half) are independent per batch element,
    # guaranteed by separate randn draws.
    n1_re = (torch.randn(b, numel, device=device) + 1j * torch.randn(b, numel, device=device)) / (2.0 ** 0.5) * sigma
    n2_re = (torch.randn(b, numel, device=device) + 1j * torch.randn(b, numel, device=device)) / (2.0 ** 0.5) * sigma
    y1 = tgt + n1_re
    y2 = tgt + n2_re
    shape_2d = gt.shape
    y1 = torch.stack([y1.real, y1.imag], dim=1).reshape(b, 2, shape_2d[0], shape_2d[1]).float()
    y2 = torch.stack([y2.real, y2.imag], dim=1).reshape(b, 2, shape_2d[0], shape_2d[1]).float()
    return torch.cat([y1, y2], dim=1)


class N2N_DDPM:
    """N2N objective with a DDPM epsilon-prediction backbone (Tweedie decode).

    N2N is a denoising task, so unlike pure generation the wide generative
    schedule is WRONG: the Tweedie decode divides by ``sqrt(alpha_bar)``, which
    ill-conditions the loss at high t. We therefore want a small T / small
    beta_end where 1/sqrt(alpha_bar) stays ~1 over the trained timesteps.
    ``T``/``beta_*`` here are the N2N-specific schedule knobs (decoupled from
    ddpm.py's generative defaults); the shared ``DDPM`` backend stays frozen and
    is built with these values.
    """

    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02,
                 backbone=UNet(), endpoint=0.0, reweight_t=False):
        self.ddpm = DDPM(T=T, beta_start=beta_start, beta_end=beta_end,
                         backbone=backbone, endpoint=endpoint)
        self.net = self.ddpm.net          # DiffusionTrainer optimizes .net.parameters()
        self.ddpm.net = self.net          # ensure the wrapper and net agree
        # Optional per-t re-weighting to neutralize the 1/sqrt(alpha_bar) bias over
        # a wide t range: weight 1/alpha_bar down-weights the easy low-t samples so
        # training focuses uniformly on all t. Default off -- with a small T /
        # beta_end the bias is already negligible.
        self.reweight_t = reweight_t

    def t_schedule(self, shape):
        """Per-sample importance weights over timesteps (off by default).

        On: p(t) ~ 1/alpha_bar drawn via inverse-CDF thresholding.
        Off: uniform-1 weights.
        """
        if not self.reweight_t:
            ones = torch.ones(shape, device=self.ddpm.betas.device)
            return ones
        tp = self.ddpm
        # p(t) ~ 1/alpha_bar_t, normalized. Draw u~U[0,1) and invert the CDF.
        cdf = 1.0 - torch.cumsum(1.0 / tp.alpha_bar, 0)
        cdf = cdf / cdf[-1]
        u = torch.rand(shape, device=tp.betas.device)
        t = torch.searchsorted(cdf, u).clamp(max=tp.T - 1)
        return t

    def loss(self, x0_pair, rng=None):
        """Tweedie-N2N loss: regress the clean decode onto an anchored target.

        x0_pair: (B, 4, H, W) stack built by ``pack_pair`` -- [:, 0:2] one noisy
        realization, [:, 2:4] the other. Returns a scalar MSE over all batch
        elements (both Re and Im channels).

        ANCHORED TARGET: without an anchor, regressing the y1-derived decode onto
        the independent draw y2 is structurally unreachable -- x0_hat carries
        y1's noise and can never match y2's, leaving an irreducible (n1-n2)
        floor. Instead we regress onto the SIGNAL estimate ``tgt = (y1 + y2)/2``
        whose conditional mean over pairs is exactly gt, giving a real gradient
        toward gt.

        HALF-SHUFFLE: per batch element we randomly choose WHICH realization is
        the perturbed/decoded input (y1 or y2); the target is always the
        symmetric mean. This keeps training symmetric w.r.t. the two draws.
        """
        tp = self.ddpm
        y1 = x0_pair[:, 0:2]
        y2 = x0_pair[:, 2:4]
        tgt = (y1 + y2) / 2.0
        # Per-batch-element shuffle: which half is the perturbed/decode input.
        swap = torch.randint(0, 2, (y1.shape[0],), device=y1.device).bool()
        src = torch.where(swap[:, None, None, None], y2, y1)
        # Per-sample timestep: importance-weighted if reweight_t, else uniform.
        if self.reweight_t:
            t = self.t_schedule((y1.shape[0],))
        else:
            t = torch.randint(0, tp.T, (y1.shape[0],), device=y1.device, dtype=torch.long)
            if tp.endpoint > 0 and torch.rand(1, device=y1.device).item() < tp.endpoint:
                t = torch.full_like(t, tp.T - 1)
        x_t, eps = tp.perturb(src, t, rng)
        eps_pred = self.net(x_t, t)
        # Tweedie / epsilon-form clean estimate of x_t given the model's eps.
        sa = tp.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        so = tp.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        x0_hat = (x_t - so * eps_pred) / sa
        mse = (x0_hat - tgt).square().mean(dim=[1, 2, 3])
        if self.reweight_t:
            w = 1.0 / tp.sqrt_alpha_bar[t].square()
            mse = mse * w
        return mse.mean()

    @torch.no_grad()
    def unconditional(self, shape, steps=None, device="cpu", rng=None):
        """Pure reverse walk (no conditioning): decode noise -> phantom."""
        return self.ddpm.sample(shape, steps=steps, device=device, rng=rng)

    @torch.no_grad()
    def conditional_denoise(self, y, sigma_task, sigma_ratio=1.5, num_steps=None,
                            device="cpu", rng=None):
        """Conditional denoise: recover clean from one held-out noisy image.

        Diffuse-then-reverse: re-encode the observed map into the DDPM ladder's
        noise-coordinate system at a rung ``t_eff`` whose added-noise std
        ``sqrt(1 - alpha_bar_{t_eff})`` sits SLIGHTLY ABOVE the map's own noise
        ``sigma_task``, then run a deterministic DDIM reverse walk from that rung
        down to t=0. Placing y on the forward rung ``x = sqrt_alpha_bar*x0 +
        sqrt(1-alpha_bar)*z`` as ``x = A*y + B*w`` (A = sab forced by signal
        matching, B = sqrt(s1ma^2 - sab^2*sigma_task^2) closing the noise gap) is
        purely algebraic: no x0_hat estimate, no forward pass at t_eff.

        Args:
            y:            (B,2,H,W) or (2,H,W) Real/Imag stack of a NOISY realization.
            sigma_task:   the acquired noise level sigma = ref_mag/SNR of ``y``.
            sigma_ratio:  how far above ``sigma_task`` the diffuse-to rung sits:
                          sigma_eff = sigma_ratio * sigma_task (default 1.5).
            num_steps:    number of DDIM reverse steps from t_eff to 0.
            device/rng:   as sample_ddim.
        """
        dp = self.ddpm
        if y.dim() == 3:
            y = y.unsqueeze(0)
        shape = y.shape
        sig = dp.sqrt_one_minus_alpha_bar        # sqrt(1 - alpha_bar_t), (T,)
        # Rung whose added-noise std is sigma_ratio * sigma_task (clamp to top rung).
        target = sigma_ratio * sigma_task
        t_eff = torch.argmin((sig - target).abs()).item()
        t_eff = min(t_eff, dp.T - 1)
        s_eff = sig[t_eff].item()                # sigma_eff at the starting rung

        # --- diffuse y onto the ladder at rung t_eff (add noise, no x0_hat) ---
        sab  = dp.sqrt_alpha_bar[t_eff].item()
        s1ma = s_eff
        B = math.sqrt(max(s1ma**2 - sab**2 * sigma_task**2, 0.0))
        x = sab * y.to(device) + B * torch.randn_like(y, generator=rng)

        # --- DDIM reverse walk from t_eff down to t=0 (eta=0, deterministic) ---
        if num_steps is None or num_steps >= t_eff + 1:
            timesteps = torch.arange(t_eff, -1, -1, device=device, dtype=torch.long)
        else:
            timesteps = torch.linspace(0, t_eff, num_steps, device=device)
            timesteps = timesteps.round().long().flip(0)
        for i in range(len(timesteps)):
            t_cur = timesteps[i].item()
            t_next = timesteps[i + 1].item() if i + 1 < len(timesteps) else 0
            ts = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            eps_pred = self.net(x, ts)
            ab_t   = dp.alpha_bar[t_cur].item()
            ab_t1  = dp.alpha_bar[t_next].item()
            sab_t  = dp.sqrt_alpha_bar[t_cur].item()
            s1m_t  = dp.sqrt_one_minus_alpha_bar[t_cur].item()
            s1m_t1 = dp.sqrt_one_minus_alpha_bar[t_next].item()
            x0_hat = (x - s1m_t * eps_pred) / sab_t
            # deterministic recombination (eta=0 DDIM): descent along the schedule
            x = dp.sqrt_alpha_bar[t_next].item() * x0_hat + s1m_t1 * eps_pred
        return x


class N2N_DSM:
    """N2N objective with a DSM / score-matching backbone (Tweedie decode).

    N2N is a denoising task, so the sigma schedule must be tied to the REAL
    data's noise, NOT widened like a generative ladder:
      - sigma_max: the largest noise the score must represent -- at least the
        task's diffusion/residual noise.
      - sigma_min: the floor of the anneal; set near the task sigma so conditional
        denoise does not wander under the noise floor.
    Both are per-task knobs; the shared ``DSM`` backend stays frozen.

    ``loss(x0_pair, rng)``: for each batch element, perturb ``y1`` at a random
    noise level ``i``, score-denoise ``x0_hat = x_tilde + sigma^2 * s``, and
    return the MSE of ``x0_hat`` vs ``y2``.
    """

    def __init__(self, L=10, sigma_min=0.01, sigma_max=50.0, backbone=UNet(),
                 task_sigma=None):
        # Tie the schedule to the real data's noise: if `task_sigma` is given (the
        # acquisition sigma = ref_mag/SNR), defaults sigma_min to it (keep the
        # finest detail that is not pure noise) and leaves sigma_max to the caller.
        if task_sigma is not None:
            sigma_min = task_sigma
        self.dsm = DSM(L=L, sigma_min=sigma_min, sigma_max=sigma_max,
                       backbone=backbone)
        self.net = self.dsm.net           # DiffusionTrainer optimizes .net.parameters()
        self.dsm.net = self.net
        self.sigmas = self.dsm.sigmas     # expose the level schedule for the notebook
        self.task_sigma = task_sigma

    def loss(self, x0_pair, rng=None):
        """Tweedie-N2N loss via the DSM score (anchored target + half-shuffle).

        ANCHORED TARGET: regress the clean decode ``x0_hat = x_tilde + sigma^2*s``
        onto ``tgt = (y1+y2)/2``, the symmetric mean of the two realizations. Its
        conditional mean over pairs is exactly gt, giving the net a genuine
        gradient toward gt.

        HALF-SHUFFLE: per batch element we randomly choose WHICH realization is
        the perturbed/decoded input; the target stays the symmetric mean.

        x0_pair: (B, 4, H, W) stack from ``pack_pair``.
        """
        dm = self.dsm
        y1 = x0_pair[:, 0:2]
        y2 = x0_pair[:, 2:4]
        tgt = (y1 + y2) / 2.0
        # Half-shuffle: per-batch-element pick of which half is the decode input.
        swap = torch.randint(0, 2, (y1.shape[0],), device=y1.device).bool()
        src = torch.where(swap[:, None, None, None], y2, y1)
        # Random level per batch element.
        i_vec = torch.randint(0, dm.L, (y1.shape[0],), device=y1.device)
        x_tilde, eps, sigma = dm.perturb(src, i_vec, rng)
        score = self.net(x_tilde, i_vec)
        # Tweedie clean estimate from the score: x0_hat = x_tilde + sigma^2 * s.
        x0_hat = x_tilde + (sigma ** 2).view(-1, 1, 1, 1) * score
        return F.mse_loss(x0_hat, tgt)

    @torch.no_grad()
    def unconditional(self, shape, steps=None, eps=1.0, device="cpu",
                      rng=None, corrector_steps=5, alpha_bound=0.25):
        """Pure annealed predictor-corrector Langevin walk: noise -> phantom.

        eps / corrector_steps match the foundation demo.ipynb. `dsm.sample()`
        runs an explicit coarse->fine level loop doing 1+corrector_steps Langevin
        steps per level, so `steps`/T_anneal are effectively overridden; the real
        lever is `corrector_steps`.
        """
        return self.dsm.sample(shape, steps=steps, eps=eps, device=device,
                               rng=rng, corrector_steps=corrector_steps)

    @torch.no_grad()
    def conditional_denoise(self, y, sigma_task, sigma_ratio=1.5, eps=1.0,
                            corrector_steps=5, corrector_alpha=0.05, device="cpu",
                            rng=None):
        """Conditional denoise: recover clean from one held-out noisy image.

        y:            (B, 2, H, W) (or (2, H, W)) Real/Imag stack of a NOISY realization.
        sigma_task:   the acquired noise level sigma = ref_mag/SNR of ``y``.
        sigma_ratio:  how far ABOVE ``sigma_task`` the map is diffused before the
                      anneal: sigma_eff = sigma_ratio * sigma_task (default 1.5).
        corrector_steps: corrector Langevin steps per level NEXT TO the one
                      "predictor" step (same refinement operation), so each level
                      runs ``corrector_steps + 1`` steps. 0 gives one per level.
        corrector_alpha: per-step corrector step size (units of sigma^2).

        Diffuse-then-anneal: the map is first DIFFUSED up to ``sigma_eff =
        sigma_ratio * sigma_task`` (adding ``sqrt(sigma_ratio^2 - 1)*sigma_task *
        w`` so its zero-mean noise std becomes sigma_eff) to give the score real
        reverse-prediction distance. DSM schedules are ADDITIVE, so no
        signal-scaling term appears; the step is purely algebraic with no score
        call. The walk then seeds at the level closest to sigma_eff and anneals
        down to sigma_min.

        Refinement ("predictor-to-Tweedie" anneal): ``y`` is already on/inside the
        data manifold, so we do NOT inject Langevin noise (that re-crops the map).
        Instead each level moves toward the local Tweedie clean estimate
        ``x0_hat = x + s^2 * s_theta``:
            x <- x + corrector_alpha * s^2 * (x0_hat - x)
        The final deterministic step at the finest level (alpha=1) lands exactly
        on the Tweedie estimate, so the returned map carries no Langevin noise.
        """
        dm = self.dsm
        if y.dim() == 3:
            y = y.unsqueeze(0)

        # Diffuse y up to sigma_eff = sigma_ratio*sigma_task before the anneal.
        add = math.sqrt(max(sigma_ratio**2 - 1.0, 0.0)) * sigma_task
        result = y.clone() + add * torch.randn_like(y)   # the effective seed

        # Seed the walk at the level whose sigma best matches the diffused level.
        target = sigma_ratio * sigma_task
        i_start = torch.argmin((dm.sigmas - target).abs()).item()

        # Uniform PC-Langevin anneal, faithful to dsm.sample(): each level i
        # (top -> 0) does `corrector_steps + 1` IDENTICAL corrector steps (the
        # +1 "predictor" is the same Tweedie refinement operation). Corrector:
        # pure Tweedie contraction with NO injected noise -- conditional denoise
        # needs a deterministic refinement, and Langevin exploration noise
        # (needed for unconditional generation) would ruin the result here.
        #
        # At the finest level the LAST iteration lands exactly on the Tweedie
        # estimate (alpha=1), so the returned map carries no injected noise.
        def corrector_step(i, final=False):
            s = dm.sigmas[i].item()
            ts = torch.full((y.shape[0],), i, device=device, dtype=torch.long)
            score = self.net(result, ts)
            x0_hat = result + s ** 2 * score     # the net's Tweedie clean estimate
            if final:
                return x0_hat                    # final deterministic refinement step
            alpha = corrector_alpha * s ** 2     # fraction of the residual to close
            return result + alpha * (x0_hat - result)

        for i in reversed(range(i_start + 1)):
            steps_i = corrector_steps + 1
            for k in range(steps_i):
                final = (i == 0) and (k == steps_i - 1)
                result = corrector_step(i, final=final)
        return result