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