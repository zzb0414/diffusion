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

PAIR PACKING through ``DiffusionTrainer``'s single-tensor ``loss(x0, rng=rng)``:
the trainer forwards ONE tensor and never touches ``y2``, so we pack both
realizations side-by-side in the channel dim of one ``(B, 4, H, W)`` Real/Imag
stack: channels [b, 0:2] = y1 (the perturbed input), channels [b, 2:4] = y2
(the regression target). ``loss`` slices the two halves itself, so the shared
``DiffusionTrainer`` (optimizes ``.net.parameters()``, calls ``model.loss(x0,
rng=rng)``) trains either model with NO modification. The notebook builds the
4-channel stack as ``torch.cat([two_ch(y1), two_ch(y2)], dim=1)``.

Both classes are thin ``net``-holding containers. The shared ``UNet`` (in_ch=2,
out_ch=2) is reused unchanged; the network always sees only the 2-channel y1
half, while the 4-channel stack is the *training* representation.
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
    """Concatenate two 2-channel stack papers into a (B, 4, H, W) training pair.

    y1 / y2: (B, 2, H, W) Real/Imag stacks of two INDEPENDENT noisy realizations
    of the same object. Returns (B, 4, H, W) with [:, 0:2] = y1 (input) and
    [:, 2:4] = y2 (target). This is the tensor the notebook feeds to the trainer.
    """
    return torch.cat([y1, y2], dim=1)


def draw_pair_batch(gt, snr, batch=4, ref_mag=1.0, device="cpu"):
    """Resynthesize a fresh minibatch of INDEPENDENT N2N pairs from a clean phantom.

    Regime-B pair source. Because ``gt`` is a KNOWN clean complex phantom, we can
    draw a brand-new pair of independent noisy realizations ``(y1, y2)`` from it
    on EVERY call (and so on every training iteration), mirroring the acquisition
    of many distinct noisy scans while still having the same ground truth. This
    kills the finite-pool overfitting risk: the model sees the full distribution
    of independent noise, not a frozen pair that it could memorize.

    Noise model matches ``add_complex_noise``: complex additive Gaussian
    ``noise = (n_re + 1j*n_im)/sqrt(2) * sigma`` with ``sigma = ref_mag/snr``.
    ``seed=None`` (the default) is used here so every draw is a fresh independent
    realization -- the exact opposite of the fixed-seed pair drawn in the
    notebook's data cell (which exists only to *visualize* one pair). Each of the
    ``batch`` elements gets its own independent ``(y1_i, y2_i)``.

    gt: (H, W) complex (WM-normalized) clean phantom.
    snr: target magnitude SNR of the corruption (sigma = ref_mag/snr).
    Returns (B, 4, H, W) packed Real/Imag stack, [:, 0:2]=y1 input, [:, 2:4]=y2
    target -- the tensor the training loop feeds to ``DiffusionTrainer.step``.
    """
    if not torch.is_complex(gt):
        raise TypeError("gt must be a complex tensor (a complex phantom).")
    gt = gt.to(device)
    tgt = gt.reshape(-1)               # draw all pairs at once for a clean batch
    b = batch
    # Complex Gaussian noise: (n_re + 1j*n_im)/sqrt(2) * sigma, sigma = ref_mag/snr.
    sigma = ref_mag / snr
    numel = tgt.numel()
    # n1 (input half) and n2 (target half) are independent per batch element,
    # guaranteed by separate randn draws.
    n1_re = (torch.randn(b, numel, device=device) + 1j * torch.randn(b, numel, device=device)) / (2.0 ** 0.5) * sigma
    n2_re = (torch.randn(b, numel, device=device) + 1j * torch.randn(b, numel, device=device)) / (2.0 ** 0.5) * sigma
    y1 = tgt + n1_re                 # (b, numel) complex
    y2 = tgt + n2_re
    # Reshape back to (b, 2, H, W) Real/Imag stacks and pack.
    shape_2d = gt.shape
    y1 = torch.stack([y1.real, y1.imag], dim=1).reshape(b, 2, shape_2d[0], shape_2d[1]).float()
    y2 = torch.stack([y2.real, y2.imag], dim=1).reshape(b, 2, shape_2d[0], shape_2d[1]).float()
    return torch.cat([y1, y2], dim=1)


class N2N_DDPM:
    """N2N objective with a DDPM epsilon-prediction backbone (Tweedie decode).

    The N2N task is DIFFERENT from pure generation and needs its own forward
    schedule. Pure generative DDPM has an MSE(eps_pred, eps) loss -- both sides
    unit-scale noise, so 1/sqrt(alpha_bar) never enters the objective. N2N
    regresses an IMAGE: x0_hat = (x_t - sqrt(1-ab)*eps_pred)/sqrt(ab) against the
    real noisy y2. Here 1/sqrt(alpha_bar) is a per-sample loss AMPLIFIER, and at
    small sqrt(alpha_bar) (high t) it multiplies the eps_pred-eps residual to
    image scale, which ill-conditions the decode. So a wide generative schedule is
    WRONG for N2N: we want a small T / small beta_end where 1/sqrt(alpha_bar) stays
    ~1 over the timesteps actually trained. `T`/`beta_*` here are the N2N-specific
    schedule knobs (decoupled from ddpm.py's generative defaults); the shared
    `DDPM` backend stays frozen and is built with these values.
    """

    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02,
                 backbone=UNet(), endpoint=0.0, reweight_t=False):
        self.ddpm = DDPM(T=T, beta_start=beta_start, beta_end=beta_end,
                         backbone=backbone, endpoint=endpoint)
        self.net = self.ddpm.net          # DiffusionTrainer optimizes .net.parameters()
        self.ddpm.net = self.net          # ensure the wrapper and net agree
        # Optional per-t re-weighting to neutralize the 1/sqrt(alpha_bar) bias when
        # drawn over a wide t range: a weight 1/(sqrt_alpha_bar)^2 == 1/alpha_bar
        # down-weights the low-t samples whose decode is easy (absorbs it close to
        # y2 with 1/sqrt(alpha_bar)~1) so training focuses uniformly on all t rather
        # than saturating on the trivial low-t inputs. Default off (False): with a
        # sufficiently small T / beta_end the bias is already negligible and no
        # re-weighting is needed.
        self.reweight_t = reweight_t

    def t_schedule(self, shape):
        """Optional importance weighting over timesteps, returned as per-sample
        multipliers for the MSE. With reweight_t, p(t) ~ 1/alpha_bar drawn via
        inverse-CDF thresholding, so every degree of noise gets comparable focus.
        Off (reweight_t=False): uniform-1 weights (standard, recommended for a
        small-T / tight schedule).
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

        x0_pair: (B, 4, H, W) stack built by ``pack_pair`` -- [:, 0:2] is one
        noisy realization, [:, 2:4] the other. Returns a scalar MSE over all
        batch elements (both Re and Im channels).

        ANCHORED TARGET (fixes the structural N2N defect): regressing the decode
        onto the OTHER noisy realization y2 alone is unfixable, because x0_hat is
        decoded ONLY from the diffused input half and can never match the
        independent draw y2 -- its irreducible floor is the (n1-n2) residual, so
        training rewards suppressing toward nothing rather than learning gt.
        Instead we regress onto the SIGNAL estimate of the pair
            tgt = (y1 + y2) / 2
        whose noise variance is HALF that of either realization (its conditional
        mean over pairs is exactly gt). The net is pushed toward the cleaner
        shared signal, giving it a real gradient toward gt. This is the
        finite-pair N2N limit: E[tgt | gt] = gt.

        HALF-SHUFFLE: per batch element we randomly choose WHICH realization is
        the perturbed/decoded input (y1 or y2) -- the target is always their
        symmetric mean. This makes training symmetric w.r.t. the two draws (no
        realization is privileged as "the" input), which matters because a draw
        seen as the input half is decoded from noise, while as the target half it
        is only an average contributor.
        """
        tp = self.ddpm
        y1 = x0_pair[:, 0:2]
        y2 = x0_pair[:, 2:4]
        # Anchored signal estimate (target for BOTH halves).
        tgt = (y1 + y2) / 2.0
        # Per-batch-element shuffle: which half is the perturbed/decode input.
        swap = torch.randint(0, 2, (y1.shape[0],), device=y1.device).bool()  # True -> decode y2
        src = torch.where(swap[:, None, None, None], y2, y1)
        # Per-sample timestep: importance-weighted if reweight_t, else uniform.
        if self.reweight_t:
            t = self.t_schedule((y1.shape[0],))
        else:
            # Random t per batch element (with the optional endpoint forcing).
            t = torch.randint(0, tp.T, (y1.shape[0],), device=y1.device, dtype=torch.long)
            if tp.endpoint > 0 and torch.rand(1, device=y1.device).item() < tp.endpoint:
                t = torch.full_like(t, tp.T - 1)
        x_t, eps = tp.perturb(src, t, rng)
        eps_pred = self.net(x_t, t)
        # Tweedie / epsilon-form clean estimate of x_t given the model's eps.
        sa = tp.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        so = tp.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        x0_hat = (x_t - so * eps_pred) / sa
        mse = (x0_hat - tgt).square().mean(dim=[1, 2, 3])   # per-batch-element MSE
        if self.reweight_t:
            w = 1.0 / tp.sqrt_alpha_bar[t].square()        # (B,)
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

        The old sampler was WRONG: it called sample_ddim(shape of y) which starts
        from torch.randn, so the noisy map was DISCARDED and we did unconditional
        generation (never used the learned denoiser on the input). This version does
        genuine diffuse-then-reverse: it re-encodes the observed map into the DDPM
        ladder's noise-coordinate system at a rung ``t_eff`` whose added-noise std
        ``sqrt(1 - alpha_bar_{t_eff})`` sits SLIGHTLY ABOVE the map's own noise
        ``sigma_task``, then runs a deterministic DDIM reverse walk from that rung
        down to t=0. The reverse walk carries the map's real structure down the
        ladder and does the actual denoising.

        Placing ``y``:
          We treat the map as the acquisition model ``y = probe + sigma_task * eps_task``
          (no estimate of the clean probe is made). To land on the DDPM forward rung
          ``x = sqrt_alpha_bar*x0 + sqrt(1-alpha_bar)*z`` (z ~ N(0,I)) at t_eff, write
          ``x = A*y + B*w`` with ``w ~ N(0,I)`` a fresh independent draw. Matching the
          signal forces ``A = sab`` (= sqrt(alpha_bar_{t_eff}); E[x] = sab*probe), and
          matching the total noise variance to the rung's ``s1ma^2`` gives
          ``B = sqrt(s1ma^2 - sab^2 * sigma_task^2)`` -- the extra noise required after
          the map's own sigma_task (attenuated by sab inside sab*y) is already present.
          This is purely algebraic: no ``x0_hat`` estimate and no forward pass at t_eff,
          so the diffuse step never depends on a (possibly imperfect) network. The rung
          ``sigma_eff = s1ma_{t_eff}`` sits SLIGHTLY ABOVE ``sigma_task`` (sigma_ratio>1)
          so (a) there is real reverse-walk distance for the denoiser to act on and
          (b) a slightly-too-low sigma_task estimate still lands above, not below,
          the true rung (avoiding over-smoothing).

        Args:
            y:            (B,2,H,W) or (2,H,W) Real/Imag stack of a NOISY realization.
            sigma_task:   the acquired noise level sigma = ref_mag/SNR of ``y``.
            sigma_ratio:  how far above ``sigma_task`` the diffuse-to rung sits:
                          sigma_eff = sigma_ratio * sigma_task (default 1.5).
            num_steps:    number of DDIM reverse steps from t_eff to 0 (default all
                          t in [t_eff, 0], i.e. the full downward ladder).
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
        # y = GT + sigma_task*eps (acquisition model, sigma_task known). Place x on the
        # DDPM rung as x = A*y + B*w, w~N(0,I) fresh:
        #   A = sab (forced by signal matching: E[x] = sab*GT),
        #   B = sqrt(s1ma^2 - sab^2*sigma_task^2) injects exactly the missing noise so
        #       x's total noise variance equals s1ma^2. No x0_hat estimate, no net call.
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

    N2N is a denoising task, so the sigma schedule must be tied to the REAL data's
    noise, NOT widened like a generative ladder:
      - sigma_max: the LARGEST noise the score must represent -- i.e. at least the
        task's residual/diffusion noise. Should sit ABOVE the actual task sigma
        (sigma = ref_mag/SNR) so the anneal covers it; it does NOT need to be a
        huge generative-scale value.
      - sigma_min: the FLOOR of the anneal. Below the task sigma there is nothing
        left to denoise, so diving far below it (e.g. sigma_min=0.01 vs task
        sigma=0.02) makes conditional denoise 'wander' under the noise floor and
        the fine levels over-stabilize toward a point clean image. Set sigma_min
        near the task sigma (or the smallest detail you want to keep).
    Both are therefore per-task knobs; the shared `DSM` backend stays frozen.

    ``loss(x0_pair, rng)``: for each batch element, perturb ``y1`` (the [:, 0:2]
    slice) at a random noise level ``i``, score-denoise ``x0_hat = x_tilde +
    sigma^2 * s``, and return the MSE of ``x0_hat`` vs ``y2`` (the [:, 2:4]
    slice). The DSM sampler is the annealed predictor-corrector Langevin walk
    (sd. ``dsm.DSM.sample``).
    """

    def __init__(self, L=10, sigma_min=0.01, sigma_max=50.0, backbone=UNet(),
                 task_sigma=None):
        # N2N-appropriate sigma schedule: tie the range to the REAL data's noise
        # level. If `task_sigma` is given (the acquisition standard deviation,
        # sigma = ref_mag/SNR), the anneal is pinned around it so the score never
        # has to represent noise far below the task floor (where there is nothing
        # left to denoise and conditional decode 'wanders'). Concretely this
        # DEFAULTS sigma_min to `task_sigma` (keep the finest detail that is not
        # pure noise; roughly the smallest structure worth preserving) and leaves
        # the caller to set `sigma_max` above it to cover the diffusion noise. If
        # `task_sigma` is None, the explicit sigma_min/sigma_max are used verbatim
        # (generative-scale defaults preserved for backward compatibility).
        if task_sigma is not None:
            sigma_min = task_sigma
        self.dsm = DSM(L=L, sigma_min=sigma_min, sigma_max=sigma_max,
                       backbone=backbone)
        self.net = self.dsm.net           # DiffusionTrainer optimizes .net.parameters()
        self.dsm.net = self.net
        self.sigmas = self.dsm.sigmas     # expose the level schedule for the notebook
        # Expose the task noise floor for the notebook's sigma_guide.
        self.task_sigma = task_sigma

    def loss(self, x0_pair, rng=None):
        """Tweedie-N2N loss via the DSM score, WITH the anchored target + shuffle.

        ANCHORED TARGET: regress the clean decode ``x0_hat = x_tilde + sigma^2 * s``
        onto ``tgt = (y1+y2)/2``, the symmetric mean of the two independent noisy
        realizations. Its noise variance is HALF that of either realization and its
        conditional mean over pairs is exactly the clean phantom ``gt``, so the net
        gets a genuine gradient toward gt (this is the finite-pair N2N limit). This
        replaces the old objective (regress the y1-derived decode onto y2), which
        was structurally unreachable -- x0_hat carries y1's noise n1 and can never
        match y2's n2, so the gradient never pointed at gt.

        HALF-SHUFFLE: per batch element we randomly choose WHICH realization is the
        perturbed/decoded input (y1 or y2); the target stays the symmetric mean. This
        keeps the objective symmetric w.r.t. the two draws so the net doesn't learn a
        bias toward one of them.

        x0_pair: (B, 4, H, W) stack from ``pack_pair`` -- [:, 0:2] / [:, 2:4] are the
        two independent noisy realizations.
        """
        dm = self.dsm
        y1 = x0_pair[:, 0:2]              # one realization
        y2 = x0_pair[:, 2:4]              # the other realization
        # Anchored signal estimate: target for both halves.
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
        # N2N: regress the decode onto the ANCHORED target (mean of the pair).
        return F.mse_loss(x0_hat, tgt)

    @torch.no_grad()
    def unconditional(self, shape, steps=None, eps=1.0, device="cpu",
                      rng=None, corrector_steps=5, alpha_bound=0.25):
        """Pure annealed predictor-corrector Langevin walk: noise -> phantom.

        eps / corrector_steps match the foundation demo.ipynb (DSM_EPS=1,
        DSM_STEPS_CORRECTOR=5). Note: dsm.sample() runs an explicit coarse->fine
        level loop doing 1+corrector_steps Langevin steps per level, so `steps`
        / T_anneal are effectively overridden; the real step-count lever here is
        `corrector_steps`. `alpha_bound` caps the per-step injected noise the same
        way `conditional_denoise` does (see that docstring) so a task-sigma scale
        never blows up.
        """
        return self.dsm.sample(shape, steps=steps, eps=eps, device=device,
                               rng=rng, corrector_steps=corrector_steps)

    @torch.no_grad()
    def conditional_denoise(self, y, sigma_task, sigma_ratio=1.5, eps=1.0,
                            corrector_steps=5, corrector_alpha=0.05, device="cpu",
                            rng=None):
        """Conditional denoise: recover clean from one held-out noisy image.

        y: (B, 2, H, W) (or (2, H, W)) Real/Imag stack of a NOISY realization.
        sigma_task: the acquired noise level sigma = ref_mag/SNR of ``y``.
        sigma_ratio: how far ABOVE ``sigma_task`` the map is diffused before the
                  anneal: sigma_eff = sigma_ratio * sigma_task (default 1.5).
        corrector_steps: number of corrector Langevin steps per level NEXT TO the
                  one "predictor" step, so each level runs ``corrector_steps + 1``
                  identical Tweedie-refinement steps (predictor and corrector are
                  the same operation here). Matches the foundation's per-level
                  (1 + corrector_steps) PC-Langevin structure. 0 gives a single
                  refinement step per level.
        corrector_alpha: per-step corrector step size (units of sigma^2).
        The corrector is the REFINEMENT stage here: each step closes a fraction
        ``corrector_alpha*s^2`` of the gap to the net's Tweedie clean estimate, with NO
        injected noise (see the correction-loop comment below for why). At the finest
        level the last step lands exactly on the Tweedie estimate (alpha=1), so the
        returned map carries no Langevin exploration noise at all.

        Diffuse-then-anneal placement of ``y``:
          The acquisition model is ``y = gt + sigma_task * eps``. The map is first
          DIFFUSED up to ``sigma_eff = sigma_ratio * sigma_task`` before seeding the
          annealed walk -- giving the score real reverse-prediction distance instead of
          starting from the map's own (narrow) noise level where the walk barely moves
          (the DSM "not denoising" failure). Adding ``w ~ N(0,I)`` with
          ``y' = y + sqrt(sigma_ratio^2 - 1) * sigma_task * w``:
              Var(noise in y') = sigma_task^2 + (sigma_ratio^2-1)*sigma_task^2
                               = sigma_ratio^2 * sigma_task^2
          so ``y'`` carries zero-mean noise of std ``sigma_eff`` -- exactly on the seed
          manifold (for sigma_ratio=1.5, sqrt(2.25-1)=sqrt(1.25)). DSM schedules are
          ADDITIVE (unlike DDPM's alpha_bar scaling), so no signal-scaling term appears;
          the step is purely algebraic, no score call, no clean estimate. The walk then
          seeds at the level closest to ``sigma_eff`` and anneals down to sigma_min.

        Refinement of the seeded map -- a "predictor-to-Tweedie" anneal:
          Unlike the unconditional dsm.sample() (which starts from PURE NOISE and must
          build structure, so it walks x + eps_l*s_theta along the raw score), ``y`` here
          is already ON/INSIDE the data manifold. We therefore do NOT continue injecting
          large Langevin noise -- that re-crops the map and is why output was worse than
          input. Instead each level moves the map toward the local Tweedie clean estimate
          ``x0_hat = x + s^2 * s_theta``:
              ref = x + s^2 * grad                    # clean-Tweedie residual target
              x = x + corrector_alpha * s^2 * (ref - x)
            = x + corrector_alpha * s^4 * grad        # , i.e. a FRACTION along the residual
          so the score withdraws the level's resolvable noise without injecting any back
          (injected noise sqrt(corrector_alpha)*s is only for Monte-Carlo settling and is
          a controlled fraction of the working scale). The final iterate is closer to the
          clean image than the seed -- which the old noisy re-injection could never do.
        """
        dm = self.dsm
        if y.dim() == 3:
            y = y.unsqueeze(0)

        # Diffuse y up to sigma_eff = sigma_ratio*sigma_task before the anneal.
        # Pure additive noise: Var(noise) -> sigma_ratio^2 * sigma_task^2. Mirrors the
        # DDPM diffuse step; no score call / clean estimate, so it never depends on net.
        add = math.sqrt(max(sigma_ratio**2 - 1.0, 0.0)) * sigma_task
        result = y.clone() + add * torch.randn_like(y)   # the effective seed

        # Seed the walk at the level whose sigma best matches the diffused level.
        target = sigma_ratio * sigma_task
        i_start = torch.argmin((dm.sigmas - target).abs()).item()

        # Uniform PC-Langevin anneal, faithful to dsm.sample(): each level i
        # (top -> 0) does `corrector_steps + 1` IDENTICAL corrector steps. Because
        # the predictor and the corrector are the SAME refinement operation here
        # (both are the factor-corrector_alpha Tweedie move), there is no distinct
        # factor-eps predictor and no separate top-level settle -- the +1 step per
        # level is the "predictor", exactly as the foundation's per-level
        # (1 + corrector_steps) Langevin structure. Descend i_start -> 0.
        #
        # Corrector: pure Tweedie contraction toward the clean estimate.
        #   x0_hat = x + s^2 * grad            # the net's trained Tweedie denoiser
        #   x      = x + alpha * (x0_hat - x), alpha = correc_alpha*s^2
        # NO injected noise. This is CONDITIONAL denoise: y is already on/inside the
        # data manifold and the net IS the (grounded) denoiser (N2N loss regresses
        # x + s^2*grade  onto the clean symmetric mean). Langevin exploration noise
        # sqrt(alpha)*z -- needed for UNCONDITIONAL generation to avoid collapsing to
        # a point -- is exactly what ruined CONDITIONAL denoise here: it is applied at
        # every step and the FINAL application at the finest level lands AFTER the last
        # refinement, injecting std sqrt(corrector_alpha)*s ~ 0.045 (s=0.1) of fresh
        # noise that is never removed, flooring RMSE above the task noise. Dropping it
        # turns the anneal into a convergent deterministic refinement to x0_hat, and
        # the terminal step (below) finishes fully deterministic at the clean estimate.
        #
        # At the finest level the LAST iteration lands exactly on the Tweedie estimate
        # (alpha=1), so the returned map carries no injected noise at all.
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
            steps_i = corrector_steps + 1      # Design A: same per level
            for k in range(steps_i):
                final = (i == 0) and (k == steps_i - 1)
                result = corrector_step(i, final=final)
        return result