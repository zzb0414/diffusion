# -*- coding: utf-8 -*-
"""Controlled noise degradation of a slice's repetitions (TQ experiment E2).

``degrade(ens, factor, mode, seed)`` adds independent noise to every repetition
and Re/Im channel of one slice so its noise level becomes factor x sigma_hat
(same estimator as ``_r3_train.sigma_hat``), without renormalizing (the signal
scale is unchanged). The added noise is white Gaussian filtered by a radial
noise-power profile:

  self   the slice's own measured noise spectrum (colour unchanged)
  white  flat spectrum
  ds2    dataset 2's median noise spectrum (fine-heavy)

Profiles are normalized to unit mean power, so the added variance is exactly
the requested one in expectation. Deterministic given ``seed``.
"""
import numpy as np

NBINS = 32
BANDS = ((0, 0.0625), (0.0625, 0.125), (0.125, 0.25), (0.25, 0.75))
_DS2 = None


def sigma_of(ens):
    """sigma_hat of an (N,2,H,W) stack, as in ``_r3_train.sigma_hat``."""
    N = ens.shape[0]
    d = ens - ens.mean(0)
    return float(np.sqrt(((d ** 2).sum(0) / (N - 1)).mean()))


def _fr(H, W):
    return np.hypot(np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :])


def noise_profile(ens):
    """Radial noise-power profile (bin centres, unit-mean power) from the repetition spread."""
    N, _, H, W = ens.shape
    d = ens - ens.mean(0)
    pn = ((np.abs(np.fft.fft2(d)) ** 2).sum(0) / (N - 1)).mean(0)
    fr = _fr(H, W)
    edges = np.linspace(0, fr.max() + 1e-9, NBINS + 1)
    idx = np.digitize(fr, edges) - 1
    prof = np.array([pn[idx == b].mean() for b in range(NBINS)])
    return 0.5 * (edges[1:] + edges[:-1]), prof / pn.mean()


def ds2_profile():
    """Median unit-mean radial noise profile over dataset 2's slices (cached)."""
    global _DS2
    if _DS2 is None:
        from _data import load_datasets
        X, _, _ = load_datasets([2])
        X = X.double().numpy()
        profs = [noise_profile(X[i]) for i in range(X.shape[0])]
        c = profs[0][0]
        _DS2 = (c, np.median(np.stack([p for _, p in profs]), 0))
    return _DS2


def band_ratio(noise):
    """finest / coarsest band power of an (M,2,H,W) noise sample (white = 1)."""
    H, W = noise.shape[-2:]
    p = (np.abs(np.fft.fft2(noise)) ** 2).mean(axis=(0, 1))
    fr = _fr(H, W)
    bp = [p[(fr >= lo) & (fr < hi)].mean() for lo, hi in BANDS]
    return float(bp[3] / bp[0])


def degrade(ens, factor, mode, seed):
    """Return (degraded stack, added-noise sample) for an (N,2,H,W) float64 stack."""
    assert factor > 1 and mode in ("self", "white", "ds2")
    N, C, H, W = ens.shape
    s0 = sigma_of(ens)
    s_add = s0 * np.sqrt(factor ** 2 - 1)
    fr = _fr(H, W)
    if mode == "white":
        filt = np.ones((H, W))
    else:
        c, prof = noise_profile(ens) if mode == "self" else ds2_profile()
        P = np.interp(fr, c, prof)
        filt = np.sqrt(P / P.mean())
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((N, C, H, W))
    noise = s_add * np.fft.ifft2(np.fft.fft2(w) * filt).real
    return ens + noise, noise
