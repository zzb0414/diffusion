# -*- coding: utf-8 -*-
"""Shared pieces for the baseline comparison (Archive/BASE/PREREGISTRATION.md).

Paths, data, the fixed split subset, and the split-half unbiased n_eff scoring (same estimator as
``experiments/_ratio_core.evaluate_slice``, restricted to a fixed subset of splits shared by every method).
External packages (bm3d, dipy) live outside the repository in ``../_pydeps`` and are added to the path here.
"""
import os, sys, itertools
import numpy as np
from scipy.ndimage import gaussian_filter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYDEPS = os.path.abspath(os.path.join(REPO, "..", "_pydeps"))
for p in (REPO, os.path.join(REPO, "experiments"), os.path.dirname(os.path.abspath(__file__)), PYDEPS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

A_EVAL = (1, 3, 5, 9)
K_SPLITS = 10
SPLIT_SEED = 20261029
N_REP = 10
BANDS = ((0.0, 0.0625), (0.0625, 0.125), (0.125, 0.25), (0.25, 0.75))


def splits(a, N=N_REP):
    """The fixed subset of K_SPLITS splits (A-half index tuples) used by every method."""
    combos = list(itertools.combinations(range(N), a))
    if len(combos) <= K_SPLITS:
        return combos
    idx = np.random.default_rng(SPLIT_SEED + a).choice(len(combos), K_SPLITS, replace=False)
    return [combos[i] for i in sorted(idx)]


class SliceScorer:
    """Precomputes masks, noise and references of one slice; scores (K,2,H,W) estimates for each a."""

    def __init__(self, ens):
        self.ens = np.asarray(ens, np.float64)                    # (N, 2, H, W)
        N, _, H, W = self.ens.shape
        mean10 = self.ens.mean(0)
        dev = self.ens - mean10
        self.vpix = ((dev ** 2).sum(0) / (N - 1)).mean(0)
        self.pn = ((np.abs(np.fft.fft2(dev)) ** 2).sum(0) / (N - 1)).mean(0)
        mag = np.sqrt(mean10[0] ** 2 + mean10[1] ** 2)
        sm = gaussian_filter(mag, 3.0)
        self.mask = sm > np.median(sm)
        self.s2 = float(self.vpix[self.mask].mean())
        self.f_r = np.hypot(np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :])
        self.N = N
        self.ref = {a: np.stack([self.ens[[j for j in range(N) if j not in c]].mean(0) for c in splits(a)])
                    for a in A_EVAL}

    def a_half(self, a):
        """list of (a, 2, H, W) A-half stacks, one per split."""
        return [self.ens[list(c)] for c in splits(a)]

    def score(self, a, est):
        """(n_eff signal region, n_eff finest band) of K split estimates (K, 2, H, W)."""
        ref = self.ref[a]
        d2 = ((est - ref) ** 2).mean(1)
        mse = float(d2[:, self.mask].mean(1).mean() - self.s2 / (self.N - a))
        n_sig = self.s2 / mse if mse > 0 else float("inf")
        sp = (np.abs(np.fft.fft2(est - ref)) ** 2).mean(axis=(0, 1))
        lo, hi = BANDS[3]
        mk = (self.f_r >= lo) & (self.f_r < hi)
        err = sp[mk].mean() - self.pn[mk].mean() / (self.N - a)
        n_band = self.pn[mk].mean() / err if err > 0 else float("inf")
        return n_sig, n_band


def load_data(datasets):
    from _data import load_datasets, cv_folds
    x, meta, sig = load_datasets(list(datasets))
    return x.double().numpy(), meta, sig.double().numpy(), cv_folds()
