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

PAIR PACKING through a shared ``loss(x0)`` trainer: the ``2*N``-channel stack
carries ``N`` independent realizations side-by-side in the channel dim of one
``(B, 2*N, H, W)`` Real/Imag stack. For N=2 (via ``pack_pair``): [b, 0:2] = y1 (the
perturbed input), [b, 2:4] = y2 (the target). ``loss`` splits the realizations
itself and computes the mean as the anchored target, so the shared
``DiffusionTrainer`` trains either model with no modification.
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

    NOISE LABELS. The legacy ``loss`` labels timestep t by the ADDED noise, but
    the input already carries sigma_y, so the net at t only ever saw total
    variance alpha_bar_t*sigma_y^2 + 1-alpha_bar_t -- while the legacy
    ``conditional_denoise`` reads t as TOTAL noise. Same flaw ``N2N_DSM`` had.

    Fix (pass ``sigma_y`` to ``loss``): work in the VE-equivalent coordinate
    lambda_t = sqrt(1-alpha_bar_t)/sqrt(alpha_bar_t), the noise of x_t/sqrt(alpha_bar_t)
    relative to the signal. A map with noise sigma_y sits at lambda = sigma_y; it is
    lifted to a TOTAL level lambda >= sigma_y by adding only sqrt(lambda^2 -
    sigma_y^2) and scaling by sqrt(alpha_bar), which lands it exactly on the
    forward marginal of that rung. ``t_of_lambda`` gives the (fractional) rung, so
    nothing is rounded; the sinusoidal t-embedding takes fractional t as-is.
    Decode such a model with ``denoise_to`` (one DDIM step to a target noise) or
    ``conditional_denoise_total`` (full DDIM walk); never with the legacy method.
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

    def _lam(self):
        """VE-equivalent noise per rung, lambda_t = sqrt((1-alpha_bar_t)/alpha_bar_t), (T,)."""
        return self.ddpm.sqrt_one_minus_alpha_bar / self.ddpm.sqrt_alpha_bar

    def t_of_lambda(self, lam):
        """Fractional timestep whose VE-equivalent noise equals ``lam``.

        Linear in log(lambda) between rungs, so integer rungs map back exactly;
        clamped to [0, T-1].
        """
        tab = torch.log(self._lam())
        l = torch.log(torch.as_tensor(lam, dtype=tab.dtype, device=tab.device))
        l = torch.minimum(torch.maximum(l, tab[0]), tab[-1])
        i = torch.searchsorted(tab, l).clamp(1, len(tab) - 1)
        lo, hi = tab[i - 1], tab[i]
        return (i - 1).to(tab.dtype) + (l - lo) / (hi - lo)

    def alpha_bar_at(self, t):
        """alpha_bar at (fractional) t, interpolated consistently with ``t_of_lambda``."""
        tab = torch.log(self._lam())
        t = torch.as_tensor(t, dtype=tab.dtype, device=tab.device).clamp(0, len(tab) - 1)
        i = t.floor().long().clamp(max=len(tab) - 2)
        lam = torch.exp(tab[i] + (t - i.to(tab.dtype)) * (tab[i + 1] - tab[i]))
        return 1.0 / (1.0 + lam.square())

    def loss(self, x0_pair, rng=None, sigma_y=None, p_raw=0.5, lam_max=None):
        """Tweedie-N2N loss for exactly TWO realizations (anchored mean + half-shuffle).

        ``sigma_y`` (optional, (B,)): acquisition noise of each pair PER Re/Im
        CHANNEL (for ``draw_pair_batch`` that is ``ref_mag/snr/sqrt(2)``). When
        given, the net is conditioned on TOTAL noise (see the class docstring):
        lambda_tot is log-uniform on [sigma_y, lam_max] (default: the top rung), or
        exactly sigma_y -- nothing added -- with probability ``p_raw``. ``None``
        keeps the legacy added-noise labels (kept so logged runs reproduce).

        x0_pair: (B, 4, H, W) Real/Imag stack of two INDEPENDENT noisy
        realizations y1 ([:, 0:2]) and y2 ([:, 2:4]) of the same object, built by
        ``pack_pair``. Returns a scalar MSE over all batch elements.

        ANCHORED TARGET: regress the clean decode onto the SIGNAL estimate
        ``tgt = (y1 + y2) / 2``, whose conditional mean over the two noisy draws
        is exactly gt. This gives a genuine gradient toward gt -- regressing one
        noisy draw's decode directly onto the other would leave an irreducible
        (n1 - n2) noise floor.

        HALF-SHUFFLE: per batch element we randomly choose WHICH of y1 / y2 is
        the perturbed/decoded input; the target stays the symmetric mean.
        """
        tp = self.ddpm
        y1 = x0_pair[:, 0:2]
        y2 = x0_pair[:, 2:4]
        tgt = 0.5 * (y1 + y2)                    # anchored target == (y1+y2)/2
        # Half-shuffle: 50/50 pick of y1 or y2 as the decode input per element.
        pick = torch.rand(x0_pair.shape[0], device=x0_pair.device) < 0.5
        src = torch.where(pick[:, None, None, None], y1, y2)
        if sigma_y is not None:
            assert not self.reweight_t, "reweight_t belongs to the legacy path"
            K = src.shape[0]
            sy = torch.as_tensor(sigma_y, dtype=src.dtype, device=src.device).expand(K)
            lam_hi = self._lam()[-1].to(src) if lam_max is None else \
                torch.as_tensor(lam_max, dtype=src.dtype, device=src.device)
            assert bool((sy < lam_hi).all()), "sigma_y above the top of the ladder"
            u = torch.rand(K, device=src.device, generator=rng)
            u = torch.where(torch.rand(K, device=src.device, generator=rng) < p_raw,
                            torch.zeros_like(u), u)
            lam = torch.exp(torch.log(sy) + u * torch.log(lam_hi / sy))   # TOTAL noise
            t = self.t_of_lambda(lam)
            ab = self.alpha_bar_at(t).to(src)
            sa = ab.sqrt().view(-1, 1, 1, 1)
            so = (1.0 - ab).sqrt().view(-1, 1, 1, 1)
            add = (lam.square() - sy.square()).clamp_min(0.0).sqrt().view(-1, 1, 1, 1)
            eps = torch.randn(src.shape, dtype=src.dtype, device=src.device, generator=rng)
            x_t = sa * (src + add * eps)          # exactly on the rung-t forward marginal
            x0_hat = (x_t - so * self.net(x_t, t)) / sa
            return (x0_hat - tgt).square().mean()
        # Per-sample timestep: importance-weighted if reweight_t, else uniform.
        if self.reweight_t:
            t = self.t_schedule((src.shape[0],))
        else:
            t = torch.randint(0, tp.T, (src.shape[0],), device=src.device, dtype=torch.long)
            if tp.endpoint > 0 and torch.rand(1, device=src.device).item() < tp.endpoint:
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

        LEGACY: reads rung t as TOTAL noise, which only matches models trained
        WITHOUT ``sigma_y`` by coincidence of the injected amount. Models trained
        with total-noise labels decode with ``denoise_to`` /
        ``conditional_denoise_total``.

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

    def _x0_at(self, x, t):
        """Tweedie estimate from x at (fractional, per-sample) rung t, with sqrt(alpha_bar)."""
        ab = self.alpha_bar_at(t).to(x)
        sa = ab.sqrt().view(-1, 1, 1, 1)
        so = (1.0 - ab).sqrt().view(-1, 1, 1, 1)
        eps = self.net(x, t)
        return (x - so * eps) / sa, eps, sa, so

    @torch.no_grad()
    def denoise_to(self, y, sigma_task, target_ratio=1.0 / 3.0, device="cpu"):
        """ONE noisy map -> ONE map with residual noise target_ratio*sigma_task.

        For models trained with ``loss(..., sigma_y=...)`` (TOTAL-noise labels).
        Nothing is injected: ``y`` (noise ``sigma_task`` per Re/Im channel) already
        sits at lambda = sigma_task, i.e. on rung t* = t_of_lambda(sigma_task) once
        scaled by sqrt(alpha_bar_{t*}). One deterministic DDIM step to
        lambda = target_ratio*sigma_task then reads, in signal units,

            out = x0_hat + target_ratio * (y - x0_hat)

        -- the same rule as ``N2N_DSM.denoise_to``; ``target_ratio=0`` returns the
        full Tweedie estimate.
        """
        if y.dim() == 3:
            y = y.unsqueeze(0)
        y = y.to(device)
        st = torch.as_tensor(sigma_task, dtype=y.dtype, device=device)
        st = st.expand(y.shape[0]) if st.dim() == 0 else st
        t = self.t_of_lambda(st)
        sa = self.alpha_bar_at(t).to(y).sqrt().view(-1, 1, 1, 1)
        x0_hat = self._x0_at(sa * y, t)[0]
        return x0_hat + target_ratio * (y - x0_hat)

    @torch.no_grad()
    def conditional_denoise_total(self, y, sigma_task, num_steps=None, device="cpu"):
        """Full deterministic DDIM walk for TOTAL-noise models, nothing injected.

        Starts at the map's own rung t* = t_of_lambda(sigma_task) (fractional) and
        descends through the integer rungs below it to t = 0, returning the final
        Tweedie estimate. Note every step after the first sees a map cleaner than
        any training input of that slice (training data never go below their own
        sigma_y), so ``denoise_to`` is the in-distribution decoder; this walk is
        provided for comparison.
        """
        if y.dim() == 3:
            y = y.unsqueeze(0)
        y = y.to(device)
        st = torch.as_tensor(float(sigma_task), dtype=y.dtype, device=device)
        t0 = self.t_of_lambda(st)
        rungs = [float(r) for r in range(int(torch.floor(t0).item()), -1, -1) if r < t0.item()]
        if num_steps is not None and len(rungs) > num_steps:
            idx = torch.linspace(0, len(rungs) - 1, num_steps).round().long().tolist()
            rungs = [rungs[i] for i in idx]
        ts = [t0.item()] + rungs
        x = self.alpha_bar_at(t0).to(y).sqrt() * y
        x0_hat = None
        for i, tc in enumerate(ts):
            tt = torch.full((y.shape[0],), tc, device=device, dtype=y.dtype)
            x0_hat, eps, _, _ = self._x0_at(x, tt)
            if i + 1 < len(ts):
                ab1 = self.alpha_bar_at(torch.tensor(ts[i + 1], device=device)).to(y)
                x = ab1.sqrt() * x0_hat + (1.0 - ab1).sqrt() * eps
        return x0_hat


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

    def level_coord(self, sigma):
        """Fractional ladder index of TOTAL noise level(s) ``sigma``.

        The ladder is geometric, so the index is linear in log(sigma): rung i sits
        at coordinate i exactly, and any sigma in between gets a fractional
        coordinate. The U-Net's sinusoidal embedding takes it as-is, so a map can
        be conditioned on its own noise level with no rounding to a rung.
        """
        s = self.dsm.sigmas
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=s.device)
        return (len(s) - 1) * torch.log(sigma / s[0]) / torch.log(s[-1] / s[0])

    def loss(self, x0_pair, rng=None):
        """Tweedie-N2N loss for exactly TWO realizations (anchored mean + half-shuffle).

        x0_pair: (B, 4, H, W) Real/Imag stack of two INDEPENDENT noisy
        realizations y1 ([:, 0:2]) and y2 ([:, 2:4]) of the same object, built by
        ``pack_pair``. Returns a scalar MSE over all batch elements.

        ANCHORED TARGET: regress the clean decode ``x0_hat = x_tilde + sigma^2*s``
        onto the SIGNAL estimate ``tgt = (y1 + y2) / 2``, whose conditional mean
        over the two noisy draws is exactly gt -- giving the net a genuine
        gradient toward gt.

        HALF-SHUFFLE: per batch element we randomly choose WHICH of y1 / y2 is
        the perturbed/decoded input; the target stays the symmetric mean.
        """
        dm = self.dsm
        y1 = x0_pair[:, 0:2]
        y2 = x0_pair[:, 2:4]
        tgt = 0.5 * (y1 + y2)                    # anchored target == (y1+y2)/2
        # Half-shuffle: 50/50 pick of y1 or y2 as the decode input per element.
        pick = torch.rand(x0_pair.shape[0], device=x0_pair.device) < 0.5
        src = torch.where(pick[:, None, None, None], y1, y2)
        # Random level per batch element.
        i_vec = torch.randint(0, dm.L, (src.shape[0],), device=src.device)
        x_tilde, eps, sigma = dm.perturb(src, i_vec, rng)
        score = self.net(x_tilde, i_vec)
        # Tweedie clean estimate from the score: x0_hat = x_tilde + sigma^2 * s.
        x0_hat = x_tilde + (sigma ** 2).view(-1, 1, 1, 1) * score
        return F.mse_loss(x0_hat, tgt)

    def loss_multireal(self, x0_stack, rng=None, weights=None, loss="l1",
                       huber_delta=0.05, sigma_y=None, p_raw=0.5,
                       noise_norm=False, sigma_ref=None, target_k=None):
        """Per-realization Tweedie-N2N over a shared 2-channel net.

        ``target_k`` (optional, 1..N-1): build each target from a random k-subset of
        the other N-1 realizations instead of all of them (a noisier, still unbiased
        N2N target) -- used to study how reference quality affects training. The
        subsets come from a dedicated generator so batches and perturbation noise
        match the default run exactly. ``None`` keeps the leave-one-out mean of all
        N-1 (the default, unchanged).

        NOISE LABELS. A realization already carries acquisition noise sigma_y, so
        perturbing it by sigma_add leaves TOTAL noise sqrt(sigma_y^2 + sigma_add^2).
        Pass ``sigma_y`` (B,) -- each slice's acquisition noise per Re/Im channel,
        estimated once outside the loop from the spread of its N realizations --
        and the net is conditioned on, and Tweedie-decoded with, that TOTAL level:
        sigma_tot is drawn log-uniform on [sigma_y, sigma_max] (or equals sigma_y,
        i.e. nothing is added, with probability ``p_raw`` -- the exact deployment
        condition), only sqrt(sigma_tot^2 - sigma_y^2) is added, and
        ``x0_hat = x_tilde + sigma_tot^2 * s(x_tilde, level_coord(sigma_tot))``.
        Decode such a model with ``denoise_to``.

        ``sigma_y=None`` is the LEGACY path: the label is the ADDED noise only, so
        a rung means a different total noise for every slice and inference that
        reads it as total noise is off-distribution. Kept only so earlier logged
        runs reproduce; do not train new models with it.

        x0_stack: (B, 2*N, H, W) Real/Imag stack of N INDEPENDENT noisy
        realizations of the same object (one slice per batch element), laid out
        interleaved ``[Re1,Im1, Re2,Im2, ...]``. Returns a scalar MSE pooled over
        BOTH the B batch elements and the N realizations.

        PER-REALIZATION OBJECTIVE: each realization k is denoised ALONE through the
        shared 2-channel ``self.net`` (input = that single ``(1,2,H,W)`` realization),
        with N2N target = the LEAVE-ONE-OUT MEAN of the OTHER N-1 realizations. The
        (finite-N) self-noise leak is removed entirely: realization k must recover
        what it CANNOT read from its own 2 channels, so the shared net is forced to
        actually denoise rather than collapse onto the all-N average it could copy
        from its input at small sigma.

        This is per-realization N2N: the input is a single realized map and the target
        is the mean of all the OTHER realizations of the same object. Regression is
        onto the clean conditional mean ``E[s | x_k]``, the exact quantity
        ``conditional_denoise`` lands on at test time.

        ``weights`` (optional, length-L non-negative tensor): per-σ-level sampling
        weight. When given, the noise level is drawn as ``i_vec ~ multinomial(weights)``
        instead of uniformly, so training can over-fit the σ band that
        ``conditional_denoise`` actually contracts in (score-scale calibration). The
        objective formula and target are unchanged. ``None`` (default) reproduces the
        uniform (equal-weight-per-rung) behavior exactly.
        """
        B, C, H, W = x0_stack.shape
        N = C // 2
        enss = x0_stack.view(B, N, 2, H, W)               # (B, N, 2, H, W)
        total = enss.sum(dim=1)                           # (B, 2, H, W) over all N
        # Flatten to B*N per-realization samples: each is ONE realization as input.
        X = enss.reshape(B * N, 2, H, W)                  # inputs
        # Target = LEAVE-ONE-OUT mean of the OTHER N-1 realizations (realization k
        # excluded from its own target). Each sample k is regressed onto the mean of
        # the remaining N-1, so the (finite-N) self-noise leak is removed entirely:
        # realization k must recover what it CANNOT read from its own 2 channels,
        # which is what forces the shared net to actually denoise each realization
        # rather than collapse onto the all-N average it could copy from its input.
        tot_loo = total.unsqueeze(1) - enss               # (B, N, 2, H, W)
        T = (tot_loo / (N - 1)).reshape(B * N, 2, H, W)
        if target_k is not None:
            T = self._subset_target(enss, target_k).reshape(B * N, 2, H, W)
        if sigma_y is not None:
            assert weights is None, "level weights belong to the legacy path"
            sy = torch.as_tensor(sigma_y, dtype=X.dtype,
                                 device=X.device).repeat_interleave(N)
            x0_hat, sig = self._tweedie_total(X, sy, rng, p_raw)
            w = self._noise_weight(sig, loss, sigma_ref) if noise_norm else None
            return self._regress(x0_hat, T, loss, huber_delta, w)
        else:
            assert not noise_norm, "noise_norm needs TOTAL-noise labels (sigma_y)"
            # LEGACY: per-sample random noise level -- uniform by default, or
            # weighted toward a specific σ band (see ``weights`` doc above).
            if weights is None:
                i_vec = torch.randint(0, self.dsm.L, (B * N,), device=x0_stack.device)
            else:
                w = weights.to(x0_stack.device)
                i_vec = torch.multinomial(w, B * N, replacement=True)
            x_tilde, eps, sigma = self.dsm.perturb(X, i_vec, rng)
            score = self.net(x_tilde, i_vec)
        # Tweedie clean estimate from the score: x0_hat = x_tilde + sigma^2 * s.
        x0_hat = x_tilde + (sigma ** 2).view(-1, 1, 1, 1) * score
        # L1 (mean absolute error) regression: the L2 optimum is the conditional MEAN,
        # which contracts the decode toward the noisy spread and systematically dims a
        # small-|s| MRI magnitude field (the shrinkage that kept one-shot bias ~0.36 vs
        # the LOO-9 target ~1.04). L1's optimum is the conditional MEDIAN, far less
        # pulled by the noise spread, so the decode keeps more true magnitude/contrast --
        # the anti-shrinkage we want, applied at the objective itself, with no added
        # loss-weight to tune. (grad_clip bounds the L1 gradient kink at the origin, and
        # perturbed inputs rarely land exactly on T, so smoothness is not a practical issue.)
        #
        # ``loss`` selects the regression norm (R2 sweep): "l1" (default, above),
        # "mse", "huber" (SmoothL1, quadratic inside huber_delta then linear), or
        # "mag_l1" (L1 weighted by |T|, pushing capacity toward high-signal structure).
        return self._regress(x0_hat, T, loss, huber_delta)

    def _subset_target(self, enss, k):
        """Mean of a random k-subset of the OTHER realizations, per (slice, realization).

        enss: (B, N, 2, H, W). Draws come from a dedicated generator (seeded once),
        leaving the global RNG stream -- batches, perturbation noise -- untouched.
        """
        B, N = enss.shape[:2]
        assert 1 <= k <= N - 1, "target_k must be in 1..N-1"
        if getattr(self, "_tk_gen", None) is None or self._tk_gen.device != enss.device:
            self._tk_gen = torch.Generator(device=enss.device)
            self._tk_gen.manual_seed(20261027)
        score = torch.rand(B, N, N, device=enss.device, generator=self._tk_gen)
        score = score.masked_fill(torch.eye(N, dtype=torch.bool, device=enss.device), 2.0)  # never pick self
        rank = score.argsort(-1).argsort(-1)                # (B, N target, N source)
        w = (rank < k).to(enss.dtype) / k
        return torch.einsum("bts,bschw->btchw", w, enss)

    @staticmethod
    def _noise_weight(sigma, loss, sigma_ref):
        """Per-sample weight making the regression error scale-free in noise.

        A sample at TOTAL noise sigma has an error that grows like sigma (L1) or
        sigma^2 (MSE), so an unweighted loss is dominated by the noisiest slices
        and draws. Weighting by (sigma_ref/sigma)^p, p = 1 for L1 and 2 for MSE,
        equalises them -- the x0-space form of the classic sigma^2 weighting of
        score matching. ``sigma_ref`` (a fixed constant) only keeps the loss
        magnitude comparable to the unweighted one.
        """
        assert sigma_ref is not None, "noise_norm needs sigma_ref"
        p = {"l1": 1.0, "mse": 2.0}[loss]
        return (sigma_ref / sigma) ** p

    @staticmethod
    def _regress(x0_hat, T, loss, huber_delta, weight=None):
        """Regression norm shared by every N2N_DSM objective.

        ``weight`` (optional, (K,)): per-sample weights on the per-sample mean
        error (L1 / MSE only); ``None`` is the plain mean over everything.
        """
        if weight is not None:
            d = x0_hat - T
            per = (d.abs() if loss == "l1" else d.square()).mean(dim=(1, 2, 3))
            assert loss in ("l1", "mse"), "weighted regression supports l1 / mse"
            return (weight.to(per) * per).mean()
        if loss == "l1":
            return F.l1_loss(x0_hat, T)
        if loss == "mse":
            return F.mse_loss(x0_hat, T)
        if loss == "huber":
            return F.smooth_l1_loss(x0_hat, T, beta=huber_delta)
        if loss == "mag_l1":
            wgt = T.detach().abs().clamp_min(1e-3).mean(dim=1, keepdim=True)
            wgt = wgt / wgt.mean()
            return (wgt * (x0_hat - T).abs()).mean()
        raise ValueError(f"unknown loss {loss!r}")

    def _tweedie_total(self, X, sy, rng, p_raw):
        """Perturb inputs X (acquisition noise sy, (K,)) to a random TOTAL level
        sigma_tot >= sy; return (Tweedie estimate decoded at that level, sigma_tot).

        sigma_tot is log-uniform on [sy, sigma_max], or exactly sy (nothing added)
        with probability ``p_raw``; only sqrt(sigma_tot^2 - sy^2) is added.
        """
        K = X.shape[0]
        s_hi = self.dsm.sigmas[-1].to(X)
        assert bool((sy < s_hi).all()), "sigma_y above the top of the ladder"
        u = torch.rand(K, device=X.device, generator=rng)
        u = torch.where(torch.rand(K, device=X.device, generator=rng) < p_raw,
                        torch.zeros_like(u), u)
        sigma = torch.exp(torch.log(sy) + u * torch.log(s_hi / sy))  # TOTAL noise
        add = (sigma.square() - sy.square()).clamp_min(0.0).sqrt()
        eps = torch.randn(X.shape, dtype=X.dtype, device=X.device, generator=rng)
        x_tilde = X + add.view(-1, 1, 1, 1) * eps
        score = self.net(x_tilde, self.level_coord(sigma))
        return x_tilde + sigma.square().view(-1, 1, 1, 1) * score, sigma

    def loss_subset(self, x0_stack, sigma_y, rng=None, loss="l1", huber_delta=0.05,
                    p_raw=0.5, m_max=None, m=None, noise_norm=False, sigma_ref=None):
        """Subset-averaging N2N: denoise the MEAN of m realizations.

        ``m`` (optional, (B,) ints): fix the subset size per slice instead of
        drawing it uniformly per sample -- lets the caller balance which TOTAL
        noise levels sigma_y/sqrt(m) are trained. ``None`` keeps the default.

        x0_stack: (B, 2*N, H, W) interleaved ``[Re1,Im1,Re2,Im2,...]``; sigma_y: (B,)
        per-slice acquisition noise per Re/Im channel (as in ``loss_multireal``).

        Each slice yields N training samples. For each, a subset size m is drawn
        uniformly from 1..m_max (default N-1) and a random subset A of that size;
        the input is mean(A), whose noise is sigma_y/sqrt(m), and the target is the
        mean of the complementary N-m realizations -- independent of the input, so
        this is still a valid N2N target whose conditional mean is the clean map.
        The net is conditioned on TOTAL noise (``_tweedie_total``), so a single
        model covers every averaging depth and, at deployment, denoises the mean
        of all available scans at sigma_y/sqrt(N) via ``denoise_to``. m=1 is the
        single-realization LOO objective of ``loss_multireal``.
        """
        B, C, H, W = x0_stack.shape
        N = C // 2
        m_max = N - 1 if m_max is None else m_max
        dev = x0_stack.device
        enss = x0_stack.view(B, N, 2, H, W).repeat_interleave(N, dim=0)  # (K,N,2,H,W)
        K = B * N
        if m is None:
            m = torch.randint(1, m_max + 1, (K,), device=dev, generator=rng)
        else:
            m = torch.as_tensor(m, device=dev).long().repeat_interleave(N)
            assert bool(((m >= 1) & (m < N)).all()), "subset size must be in 1..N-1"
        rank = torch.rand(K, N, device=dev, generator=rng).argsort(1).argsort(1)
        inA = rank < m[:, None]
        wA = inA.to(enss.dtype) / m[:, None]
        wB = (~inA).to(enss.dtype) / (N - m)[:, None]
        X = (wA[:, :, None, None, None] * enss).sum(1)
        T = (wB[:, :, None, None, None] * enss).sum(1)
        sy = torch.as_tensor(sigma_y, dtype=X.dtype, device=dev).repeat_interleave(N)
        sy = sy / m.to(X.dtype).sqrt()
        x0_hat, sig = self._tweedie_total(X, sy, rng, p_raw)
        w = self._noise_weight(sig, loss, sigma_ref) if noise_norm else None
        return self._regress(x0_hat, T, loss, huber_delta, w)

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
                            rng=None, target_sigma=None):
        """Conditional denoise: recover clean from one held-out noisy image.

        LEGACY: reads the rung label as TOTAL noise, but checkpoints trained
        without ``sigma_y`` learned it as ADDED noise, so the walk is
        off-distribution for them. New models decode with ``denoise_to``.

        y:            (B, 2, H, W) (or (2, H, W)) Real/Imag stack of a NOISY realization.
        sigma_task:   the acquired noise level sigma = ref_mag/SNR of ``y``.
        sigma_ratio:  how far ABOVE ``sigma_task`` the map is diffused before the
                      anneal: sigma_eff = sigma_ratio * sigma_task (default 1.5).
                      Set to 1.0 to inject NOTHING -- ``y`` enters the walk
                      unchanged, which is what real noisy ASL data wants.
        target_sigma: OPTIONAL, decouples the reverse-walk SEED level from the
                      forward perturbation. When given, the walk starts at the
                      ladder rung closest to ``target_sigma`` instead of
                      ``sigma_ratio * sigma_task``. Use ``sigma_ratio=1.0`` with
                      ``target_sigma=<reference noise floor>`` to decode a map
                      straight from its own noise down toward the reference
                      quality, with no injected noise at all. Default None keeps
                      the historical behaviour (target = sigma_ratio*sigma_task).
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
        if add > 0:
            noise = (torch.randn_like(y) if rng is None
                     else torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=rng))
        else:
            noise = torch.zeros_like(y)
        result = y.clone() + add * noise                 # the effective seed

        # Seed the walk at the level whose sigma best matches the decoding
        # target. Historically that target was the diffused level itself; an
        # explicit ``target_sigma`` lets the caller aim lower (e.g. at the
        # reference noise floor) while injecting nothing.
        target = target_sigma if target_sigma is not None else sigma_ratio * sigma_task
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

    @torch.no_grad()
    def single_map_denoise(self, y, sigma_task, blend=0.570, rung_mult=1.0,
                           device="cpu"):
        """ONE noisy realization -> ONE denoised map (no ensemble averaging).

        LEGACY: for checkpoints trained without ``sigma_y`` (labels = added
        noise). Models trained with TOTAL-noise labels decode with ``denoise_to``.

        This is the deployable single-map path. It does exactly one Tweedie
        evaluation at the rung matched to the map's OWN noise level,

            x0_hat = y + s^2 * score(y, t)         s = sigmas[argmin |sigmas - rung_mult*sigma_task|]

        and then takes a partial step toward it,

            out = y + blend * (x0_hat - y)

        Why not `conditional_denoise`: its diffuse-then-anneal walk closes only
        ``corrector_alpha * s^2`` of each residual (a few 1e-4 at task noise),
        so a single map comes back statistically identical to its input, while
        its terminal call lands on a full Tweedie step -- which alone collapses
        contrast (magnitude ~0.3 of the reference). The interior blend keeps
        roughly the right amount of both: measured against a leave-one-out
        reference over six train/val/test slices, ``blend`` in 0.50-0.63 puts
        single-map magnitude bias between 0.83 and 1.20 with HF/HP structure
        energy near the reference, at ~1.25x the ensemble RMSE.

        Args:
            y:          (B,2,H,W) or (2,H,W) Real/Imag stack of NOISY realization(s).
            sigma_task: acquired noise sigma = ref_mag/SNR of ``y``.
            blend:      fraction of the Tweedie step to keep, in [0,1].
                        ``1.0`` = full Tweedie (dim, contrast-collapsed);
                        ``0.0`` = identity (returns ``y`` unchanged).
                        Default 0.570 is the grid point with the flattest
                        cross-slice bias; it is NOT selected per map.
            rung_mult:  which ladder rung to evaluate at, as a multiple of
                        ``sigma_task`` (default 1.0 = the map's own noise level).

        Returns the same shape/dtype as ``y``. Deterministic: no injected noise.
        """
        dm = self.dsm
        if y.dim() == 3:
            y = y.unsqueeze(0)
        target = rung_mult * sigma_task
        i = int(torch.argmin((dm.sigmas - target).abs()).item())
        s = dm.sigmas[i].item()
        ts = torch.full((y.shape[0],), i, device=device, dtype=torch.long)
        x0_hat = y.to(device) + s ** 2 * self.net(y.to(device), ts)
        # One linear blend of identity and the Tweedie estimate. Both endpoints
        # are already noise-free operations -- nothing here injects noise.
        return y.to(device) + blend * (x0_hat - y.to(device))

    @torch.no_grad()
    def denoise_to(self, y, sigma_task, target_ratio=1.0 / 3.0, device="cpu"):
        """ONE noisy realization -> ONE map with residual noise target_ratio*sigma_task.

        For models trained with ``loss_multireal(..., sigma_y=...)``, whose labels
        are TOTAL noise. Nothing is injected: ``y`` already sits at total noise
        ``sigma_task``, so the net is evaluated there directly,

            x0_hat = y + sigma_task^2 * s(y, level_coord(sigma_task))

        and one deterministic DDIM step goes from sigma_task to
        ``target_ratio * sigma_task``,

            out = x0_hat + target_ratio * (y - x0_hat)

        i.e. the noise estimate ``y - x0_hat`` is kept at the target fraction. The
        default 1/3 is the noise floor of the leave-one-out mean of 9 other
        realizations; ``target_ratio=0`` returns the full Tweedie estimate.

        Args:
            y:            (B,2,H,W) or (2,H,W) Real/Imag stack of NOISY realization(s).
            sigma_task:   acquisition noise of ``y`` per Re/Im channel (scalar or (B,)),
                          estimated the same way as the training ``sigma_y``.
        """
        if y.dim() == 3:
            y = y.unsqueeze(0)
        y = y.to(device)
        st = torch.as_tensor(sigma_task, dtype=y.dtype, device=device)
        st = st.expand(y.shape[0]) if st.dim() == 0 else st
        x0_hat = y + st.square().view(-1, 1, 1, 1) * self.net(y, self.level_coord(st))
        return x0_hat + target_ratio * (y - x0_hat)


class N2N_DSM_Multi:
    """Multi-realization joint N2N for a DSM / score backbone.

    Unlike ``N2N_DSM`` (whose ``loss`` maps ONE noisy realization to the clean
    image), this model consumes ALL N realizations of the same object AT ONCE --
    the whole 2N-channel Real/Imag stack ``(B, 2*N, H, W)`` is the input -- and
    outputs N denoised maps that each PRESERVE per-acquisition identity.

    Why this is the fix for the double-denoising goal: a single-input denoiser
    learns the score/Tweedie ``E[x0 | x_t]`` = conditional mean given ONE
    realization, which at low SNR collapses to the (global) map ``E[x0]`` -- NOT
    per-acquisition detail. So "denoise each acquisition then average again"
    cannot work with a single-input net: the model already averaged. Only by
    feeding the WHOLE ensemble can the net exploit the shared information across
    realizations while keeping each output's own high-SNR detail.

    Design (all user-confirmed):
      - Batch element = ONE slice's N realizations, gathered into ``(B, 2*N, H, W)``.
        N defaults to 10 (all ASL realizations) but is taken from input shape.
      - U-Net ``UNet(in_ch=2*N, out_ch=2*N)``. Its convolutions / pooling mix
        channels, forcing cross-realization information sharing.
      - PERTURB-WHOLE-STACK: the entire ``(B, 2*N, H, W)`` stack is perturbed ONCE
        per level with the SAME added-noise draw + noise level (not an independent
        eps per realization). This couples the outputs' Tweedie decodes.
      - Each intermediate feature is the score ``s_theta(x_tilde, t)`` of the whole
        stack; the Tweedie decode is taken per-realization channel block.
      - LEAVE-ONE-OUT ANCHOR: for each output k, the regression target is the mean
        of the OTHEr N-1 realizations (which the model sees as inputs). The
        conditional mean of that target over the noisy draws is still the clean
        map, so it grants a genuine gradient toward gt, while keeping the
        model-bias ("global mean collapse") from leaking into the anchor.
      - TARGET_CONSENSUS fallback: when the user passes ``anchor='consensus'`` the
        anchor is the mean of ALL N realizations (an optional alternative knob).
    """

    def __init__(self, N=10, L=40, sigma_min=0.01, sigma_max=2.5,
                 backbone=None, sigma_task=None):
        # Build the DSM on a 2N-channel UNet. If no backbone given, construct a
        # fresh UNet with in/out = 2*N (this mixes all realizations across channels).
        if backbone is None:
            backbone = UNet(in_ch=2 * N, out_ch=2 * N)
        self.N = N
        # Tie sigma_min to the real data's per-realization noise if given (keep the
        # finest detail that is not pure noise); sigma_max stays a free knob.
        if sigma_task is not None:
            sigma_min = sigma_task
        self.dsm = DSM(L=L, sigma_min=sigma_min, sigma_max=sigma_max, backbone=backbone)
        self.net = self.dsm.net
        self.dsm.net = self.net
        self.sigmas = self.dsm.sigmas          # expose level schedule for the notebook
        self.task_sigma = sigma_task

    # ------------------------------------------------------------------ loss ---
    def loss(self, x0_stack, rng=None, anchor="loo"):
        """Multi-realization joint DSM-N2N loss.

        x0_stack: (B, 2*N, H, W) Real/Imag stack of N independent noisy
        realizations of the same object (one slice per batch element).
        Returns a scalar MSE over all batch elements (both Re and Im channels).

        PERTURB-WHOLE-STACK ONCE: split ``x0_stack`` into N realizations ``reals``
        (each (B,2,H,W)), perturb the WHOLE stack at one random level → the shared
        added-noise draw couples the outputs. ``x_tilde = stack + sigma * eps``.

        Each realization k is Tweedie-decoded to ``x0_hat_k = x_tilde_k + sigma^2 * s_k``
        (where ``s = s_theta(x_tilde, t)`` is the score of the WHOLE stack), and
        regressed onto its anchor:
            anchor='loo'       → tgt_k = mean over the OTHER N-1 realizations
            anchor='consensus' → tgt_k = mean over ALL N realizations
        """
        dm = self.dsm
        N = x0_stack.shape[1] // 2
        reals = torch.split(x0_stack, 2, dim=1)            # N tensors, each (B,2,H,W)
        stacked = torch.stack(reals, 0)                    # (N,B,2,H,W)

        # Anchored per-output target: leave-one-out mean of the other N-1, or the
        # all-N consensus.
        total = stacked.sum(0)                             # (B,2,H,W) = sum over N
        if anchor == "consensus":
            tgt = total / N
        else:                                              # 'loo' default
            tgt = (total - stacked) / (N - 1)              # (N,B,2,H,W)

        # Perturb the WHOLE stack at ONE shared random level per batch element.
        i_vec = torch.randint(0, dm.L, (x0_stack.shape[0],), device=x0_stack.device)
        x_tilde, eps, sigma = dm.perturb(x0_stack, i_vec, rng)   # sigma (B,)
        score = self.net(x_tilde, i_vec)                   # (B, 2*N, H, W)
        # Tweedie clean estimate of the WHOLE stack.
        x0_hat_stack = x_tilde + (sigma ** 2).view(-1, 1, 1, 1) * score
        x0_hat = torch.stack(torch.split(x0_hat_stack, 2, dim=1), 0)  # (N,B,2,H,W)

        mse = (x0_hat - tgt).square().mean(dim=[1, 2, 3])  # (N,B)
        return mse.mean()

    # -------------------------------------------------------------- sampler ---
    @torch.no_grad()
    def conditional_denoise_multi(self, y_stack, sigma_task, sigma_ratio=1.5,
                                  eps=1.0, corrector_steps=5, corrector_alpha=0.05,
                                  device="cpu", rng=None, target_sigma=None):
        """Joint denoise of the WHOLE N-realization stack at once.

        y_stack: (B, 2*N, H, W) (or (2*N, H, W)) Real/Imag stack of the N noisy
        realizations of one object (e.g. one test slice's 10 acquisitions).
        Returns (B, 2*N, H, W) OR (2*N, H, W) of the N denoised maps, in the SAME
        channel order as the input so each output preserves its acquisition.

        Diffuse-then-anneal over the WHOLE stack: the stack is diffused up to
        ``sigma_eff = sigma_ratio * sigma_task``, then an annealed
        predictor-corrector Tweedie walk (identical machinery to
        ``N2N_DSM.conditional_denoise``, one forward pass over the joint stack)
        recovers N denoised maps. The joint input lets each realizations' reverse
        step borrow info from the other N-1.

        target_sigma: OPTIONAL, decouples the reverse-walk SEED level from the
                      forward perturbation exactly as in
                      ``conditional_denoise``.
        """
        dm = self.dsm
        if y_stack.dim() == 3:
            y_stack = y_stack.unsqueeze(0)
        B = y_stack.shape[0]

        # Diffuse the whole stack up to sigma_eff before the anneal.
        add = math.sqrt(max(sigma_ratio ** 2 - 1.0, 0.0)) * sigma_task
        result = y_stack.clone() + add * torch.randn_like(y_stack)

        target = target_sigma if target_sigma is not None else sigma_ratio * sigma_task
        i_start = torch.argmin((dm.sigmas - target).abs()).item()

        def corrector_step(i, final=False):
            s = dm.sigmas[i].item()
            ts = torch.full((B,), i, device=device, dtype=torch.long)
            score = self.net(result, ts)                 # joint score (B,2*N,H,W)
            x0_hat = result + s ** 2 * score             # joint Tweedie decode
            if final:
                return x0_hat
            alpha = corrector_alpha * s ** 2
            return result + alpha * (x0_hat - result)

        for i in reversed(range(i_start + 1)):
            steps_i = corrector_steps + 1
            for k in range(steps_i):
                final = (i == 0) and (k == steps_i - 1)
                result = corrector_step(i, final=final)

        if result.shape[0] == 1 and y_stack.dim() != 4:
            return result[0]
        return result