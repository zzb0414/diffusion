# -*- coding: utf-8 -*-
"""GAP I (Archive/GAP/PREREGISTRATION_I.md): noise-stationarized ASL. Each slice keeps its real repetition mean m;
only the deviations are phase-randomized, x_r' = m + U (x_r - m), with U a fixed random k-space phase (unitary) per
(dataset, slice), the same for every repetition. Signal structure kept; noise power spectrum and sigma_hat kept;
the spatial non-stationarity of the noise is destroyed (spread over the FOV). Seeds differ from GAP G's."""
import numpy as np
import torch

SEED = 20261108


def apply(x, meta):
    """x (Ns, N, 2, H, W) Re/Im stack (load_datasets), meta [(dataset, slice)] -> noise-stationarized copy."""
    out = torch.empty_like(x)
    for i, (d, s) in enumerate(meta):
        ph = np.exp(2j * np.pi * np.random.default_rng(SEED + 1009 * d + s).random(x.shape[-2:]))
        z = x[i, :, 0].double().numpy() + 1j * x[i, :, 1].double().numpy()
        m = z.mean(0, keepdims=True)
        z = m + np.fft.ifft2(np.fft.fft2(z - m, norm="ortho") * ph, norm="ortho")
        out[i, :, 0] = torch.from_numpy(z.real).to(x.dtype); out[i, :, 1] = torch.from_numpy(z.imag).to(x.dtype)
    return out
