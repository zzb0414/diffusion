# -*- coding: utf-8 -*-
"""GAP B noise transplant (Archive/GAP/PREREGISTRATION_B.md): fastMRI images at the ASL matrix (80) with ASL noise.

Clean image: the fastMRI AXT2 image at matrix 80 (data.to_res; p99 of the 320 image = 1). Ten copies per image, noise
of kind
- "W": white complex Gaussian, SIG per channel (control: the MEAN-SNR setting moved to matrix 80);
- "G": stationary Gaussian with the 2-D noise power spectrum of the ASL repetition deviations (noise colour only);
- "T": the real ASL repetition-deviation fields (colour + spatial non-stationarity + physiological structure + heavy tails).
ASL fields: coil1 scans, deviation of each repetition from its slice's mean, x sqrt(N / (N - 1)); each scan scaled so
that its mean per-channel variance over the whole image is SIG^2 (as sigma_hat is a whole-image quantity). Training
copies draw from ds1 + 1104a, test copies from 1104c (an unseen subject's noise). The ten copies of an image take
fields from ten different slices and ten different repetitions, so they are independent (deviations within one
slice sum to zero; slices of one repetition share a TR).
Every draw is seeded per (split, image); G uses the power spectrum of its split's pool.
"""
import os
import numpy as np
import data as D

SIG = 1.0                       # per-channel noise std; white noise at matrix 80 then has fastMRI S_K = 2's per-band SNR
M = 80
KINDS = ("W", "G", "T")
ASL = os.path.join(D.REPO, "..", "asl_for_diffusion", "coil1")
POOL = {"train": ("ds1_20260924", "1104a_20261104"), "test": ("1104c_20261104",)}   # asl_for_diffusion/manifest.json
SEED = 20261105
CACHE = os.path.join(os.path.expanduser("~"), "diffusion_cache", "gap")
_pool = {}


def pool(split):
    """(n_slices, N_rep, 80, 80) complex64 deviation fields scaled to SIG per channel, and their mean power spectrum."""
    if split not in _pool:
        os.makedirs(CACHE, exist_ok=True)
        path = os.path.join(CACHE, f"pool_{split}.npz")
        if not os.path.exists(path):
            fs = []
            for p in POOL[split]:
                y = np.load(os.path.join(ASL, p, "asl.npy")).astype(np.complex128)      # (80, 80, slices, N)
                n = y.shape[-1]
                dev = (y - y.mean(-1, keepdims=True)) * np.sqrt(n / (n - 1))
                dev = dev * (SIG / np.sqrt((np.abs(dev) ** 2).mean() / 2))
                fs.append(np.transpose(dev, (2, 3, 0, 1)))
            f = np.concatenate(fs).astype(np.complex64)
            psd = (np.abs(np.fft.fft2(f, norm="ortho")) ** 2).reshape(-1, M, M).mean(0)
            np.savez(path, f=f, psd=psd)
        z = np.load(path)
        _pool[split] = (z["f"], z["psd"])
    return _pool[split]


def fields(split, i, kind, n=D.N):
    """(n, 80, 80) complex64 noise for image i of the split."""
    rng = np.random.default_rng(SEED + 1000003 * D.SPLIT_ID[split] + 7 * i + KINDS.index(kind))
    if kind == "T":
        f, _ = pool(split)
        sl = rng.choice(len(f), n, replace=False)
        return f[sl, rng.permutation(f.shape[1])[:n]]           # distinct slice AND repetition (same-TR slices share physiology)
    w = SIG * (rng.standard_normal((n, M, M)) + 1j * rng.standard_normal((n, M, M)))
    if kind == "W":
        return w.astype(np.complex64)
    psd = pool(split)[1]
    return np.fft.ifft2(np.fft.fft2(w, norm="ortho") * np.sqrt(psd / (2 * SIG ** 2)), norm="ortho").astype(np.complex64)


def up(z, G=160):
    """(..., 80, 80) complex -> (..., G, G): the same content on a finer pixel grid (centred k-space zero-padding,
    intensity preserved; white noise keeps its per-pixel std but becomes band-limited). GAP C2."""
    c = (G - M) // 2
    k = np.zeros(z.shape[:-2] + (G, G), np.complex128); k[..., c:c + M, c:c + M] = D.fft2c(z)
    return (D.ifft2c(k) * (G / M)).astype(np.complex64)


def clean(x320):
    return D.to_res(D.fft2c(x320), M).astype(np.complex64)
