# -*- coding: utf-8 -*-
"""GAP G (Archive/GAP/PREREGISTRATION_G.md): phase-randomized ASL. Every repetition of a slice is multiplied in k-space
by the SAME uniformly random phase field (seeded per dataset and slice), so the repetitions keep a common signal and
independent noise (the split-half estimator stays valid), the signal and noise power spectra are unchanged (and
sigma_hat exactly, by Parseval), but the signal's spatial structure (phase) is destroyed."""
import numpy as np
import torch

SEED = 20261107


def apply(x, meta):
    """x (Ns, N, 2, H, W) Re/Im stack (load_datasets), meta [(dataset, slice)] -> phase-randomized copy."""
    out = torch.empty_like(x)
    for i, (d, s) in enumerate(meta):
        ph = np.exp(2j * np.pi * np.random.default_rng(SEED + 1009 * d + s).random(x.shape[-2:]))
        z = x[i, :, 0].double().numpy() + 1j * x[i, :, 1].double().numpy()
        z = np.fft.ifft2(np.fft.fft2(z, norm="ortho") * ph, norm="ortho")
        out[i, :, 0] = torch.from_numpy(z.real).to(x.dtype); out[i, :, 1] = torch.from_numpy(z.imag).to(x.dtype)
    return out
