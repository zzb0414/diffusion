# -*- coding: utf-8 -*-
"""GAP K (Archive/GAP/PREREGISTRATION_K.md): deployable spatial pre-whitening of ASL. Per slice, the per-pixel
repetition variance (both channels, unbiased over N) is smoothed (Gaussian, SMOOTH px) and normalized to mean 1 over the
FOV; s = sqrt of that map. Data are divided by s (noise becomes about spatially uniform at the slice's whole-image level,
so the scalar label sigma_hat still applies); a model's output is multiplied back by s, so scoring is in ORIGINAL units.
The map uses only the slice's own repetitions, which are available at test time."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

SMOOTH = 4.0


def smap(ens):
    """ens (N, 2, H, W) tensor -> s (H, W) float64 numpy, mean(s^2) = 1."""
    e = ens.double().cpu().numpy()
    v = (((e - e.mean(0)) ** 2).sum(0) / (e.shape[0] - 1)).mean(0)
    v = gaussian_filter(v, SMOOTH, mode="nearest")
    return np.sqrt(v / v.mean())


def whiten(loader):
    """Wrap load_datasets: every slice divided by its own map (sigma_hat unchanged)."""
    def load(ids):
        x, meta, sig = loader(ids)
        out = torch.empty_like(x)
        for i in range(len(x)):
            out[i] = x[i] / torch.from_numpy(smap(x[i])).to(x.dtype)
        return out, meta, sig
    return load


class Whitened:
    """``denoise_to`` for original-units data: divide by the slice map, denoise, multiply back."""

    def __init__(self, m, s):
        self.m, self.s = m, s

    def denoise_to(self, y, s, target_ratio=0.0, device="cpu"):
        k = torch.as_tensor(self.s, dtype=y.dtype, device=y.device)
        return self.m.denoise_to(y / k, s, target_ratio=target_ratio, device=device) * k.to(device)
