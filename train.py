"""
Shared training helper for the DDPM and DSM models in this folder.

Both models expose `.loss(x)` over a mini-batch x of shape (B, 2, H, W),
so a single loop trains either one.
"""
import torch
import torch.nn as nn


class DiffusionTrainer:
    def __init__(self, model, lr=1e-3, device="cpu", grad_clip=0.0):
        self.model = model
        self.device = device
        self.grad_clip = grad_clip
        self.opt = torch.optim.Adam(model.net.parameters(), lr=lr)

    def step(self, x0, rng=None):
        x0 = x0.to(self.device)
        self.opt.zero_grad()
        loss = self.model.loss(x0, rng=rng)
        loss.backward()
        if self.grad_clip > 0:
            # Cap the total gradient norm before the step.
            torch.nn.utils.clip_grad_norm_(self.model.net.parameters(),
                                           self.grad_clip)
        self.opt.step()
        return loss.item()

    def step_subset(self, x0, sigma_y, rng=None, loss="l1", huber_delta=0.05,
                    p_raw=0.5, m_max=None, m=None):
        """Training step for ``N2N_DSM.loss_subset`` (denoise subset means)."""
        x0 = x0.to(self.device)
        self.opt.zero_grad()
        loss_val = self.model.loss_subset(x0, sigma_y, rng=rng, loss=loss,
                                          huber_delta=huber_delta, p_raw=p_raw,
                                          m_max=m_max, m=m)
        loss_val.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.net.parameters(),
                                           self.grad_clip)
        self.opt.step()
        return loss_val.item()

    def step_multireal(self, x0, rng=None, weights=None, loss="l1",
                       huber_delta=0.05, sigma_y=None, p_raw=0.5,
                       noise_norm=False, sigma_ref=None, target_k=None):
        """Training step for the N-realization per-realization objective.

        Same as ``step``, but drives ``N2N_DSM.loss_multireal`` (per-realization
        Tweedie-N2N over all N realizations of a slice) instead of the 2-real
        ``loss``. Keeps the original ``step`` path intact so the 2-real toy-model
        experiments stay reproducible. ``x0`` is a (B, 2*N, H, W) stack of all N
        realizations of each slice.

        ``weights`` (optional): forwarded to ``loss_multireal``'s per-σ-level
        sampling weights for score-scale calibration. ``None`` (default) reproduces
        the uniform (equal-weight-per-rung) behavior exactly. ``loss`` selects the
        regression norm ("l1", "mse", "huber", "mag_l1") and is forwarded too.
        ``sigma_y`` / ``p_raw``: per-slice acquisition noise for TOTAL-noise
        labels (see ``N2N_DSM.loss_multireal``); ``None`` keeps the legacy path.
        """
        x0 = x0.to(self.device)
        self.opt.zero_grad()
        loss_val = self.model.loss_multireal(x0, rng=rng, weights=weights,
                                             loss=loss, huber_delta=huber_delta,
                                             sigma_y=sigma_y, p_raw=p_raw,
                                             noise_norm=noise_norm,
                                             sigma_ref=sigma_ref, target_k=target_k)
        loss_val.backward()
        if self.grad_clip > 0:
            # Cap the total gradient norm before the step.
            torch.nn.utils.clip_grad_norm_(self.model.net.parameters(),
                                           self.grad_clip)
        self.opt.step()
        return loss_val.item()