# -*- coding: utf-8 -*-
"""C2 proof of concept (claims ledger, research note 10): why the score parametrization lowers bias.

Linear-Gaussian toy. Clean images x are stationary Gaussian fields with power spectrum S(f) ~ 1/(f^2 + f0^2)
(natural-image-like, mean power 1); N = 10 copies y_k = x + n_k, white noise sigma^2. The denoiser is a linear
per-frequency gain w(f) applied to one copy; the averaged estimate of a copies is w * mean(y), whose exact error is

    MSE_a = mean_f [ (1 - w)^2 S + w^2 sigma^2 / a ]    =  bias^2 + variance / a.

Three parametrizations of the SAME function class, trained on the Noise2Noise loss (leave-one-out target), with
optimum w* = S / (S + sigma^2) (Wiener) for all three:
    direct    w = h                    (h starts at 0  -> output 0)
    residual  w = 1 + g                (g starts at 0  -> identity)
    score     w = 1 + sigma^2 g        (g starts at 0  -> identity)
(1) Population gradient descent (exact expected loss), step size at each parametrization's stability limit.
(2) Finite data: M training images, fixed copies, Adam (as the networks), exact error from the known spectrum.
Reported per training budget: bias^2, variance, n_eff = sigma^2 / MSE at a = 1 and a = 9, at r = 0 and at the best
dial r (w' = w + r (1 - w)).

    python experiments/_theory_c2_toy.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

H = 64
f = np.hypot(np.fft.fftfreq(H)[:, None], np.fft.fftfreq(H)[None, :])
S = 1.0 / (f ** 2 + 0.02 ** 2)
S /= S.mean()                                   # signal power 1 per pixel
RG = np.linspace(0, 1, 101)


def score(w, s2):
    out = {}
    for a in (1, 9):
        b2 = float(((1 - w) ** 2 * S).mean()); v = float((w ** 2).mean() * s2)
        n0 = s2 / (b2 + v / a)
        nb = max(s2 / ((((1 - (w + r * (1 - w))) ** 2) * S).mean() + ((w + r * (1 - w)) ** 2).mean() * s2 / a) for r in RG)
        out[a] = (n0, nb)
    return float(((1 - w) ** 2 * S).mean()), float((w ** 2).mean() * s2), out


def population(s2, steps):
    """exact expected-loss GD; per-parametrization step = 1 / (largest curvature)."""
    E = S + s2                                   # E|Y|^2 per frequency; d/dw of the N2N loss = 2 (w E - S)
    res = {}
    for name in ("direct", "residual", "score"):
        k = s2 if name == "score" else 1.0       # dw/dparam
        eta = 1.0 / (2 * (k ** 2) * E.max())
        p = np.zeros_like(S)
        traj = {}
        for t in range(1, max(steps) + 1):
            w = p if name == "direct" else 1 + k * p
            p -= eta * k * 2 * (w * E - S)
            if t in steps:
                traj[t] = (p if name == "direct" else 1 + k * p).copy()
        res[name] = traj
    return res


def finite(s2, M, steps, lr=0.02, seed=0):
    """M images, N fixed copies, Adam on the empirical N2N loss (leave-one-out target)."""
    rng = np.random.default_rng(seed)
    N = 10
    X = np.fft.fft2(rng.standard_normal((M, H, H))) * np.sqrt(S)          # Fourier coefficients of clean fields
    Y = X[:, None] + np.fft.fft2(rng.standard_normal((M, N, H, H))) * np.sqrt(s2)
    T = (Y.sum(1, keepdims=True) - Y) / (N - 1)
    YY = (np.abs(Y) ** 2).mean((0, 1)); YT = (Y.conj() * T).real.mean((0, 1))   # sufficient statistics
    res = {}
    for name in ("direct", "residual", "score"):
        k = s2 if name == "score" else 1.0
        p = np.zeros_like(S); m1 = np.zeros_like(S); m2 = np.zeros_like(S)
        traj = {}
        for t in range(1, max(steps) + 1):
            w = p if name == "direct" else 1 + k * p
            g = k * 2 * (w * YY - YT) / (H * H)
            m1 = 0.9 * m1 + 0.1 * g; m2 = 0.999 * m2 + 0.001 * g ** 2
            p -= lr * (m1 / (1 - 0.9 ** t)) / (np.sqrt(m2 / (1 - 0.999 ** t)) + 1e-12)
            if t in steps:
                traj[t] = (p if name == "direct" else 1 + k * p).copy()
        res[name] = traj
    return res


def show(title, res, s2):
    print(f"\n{title}")
    print("  steps | param    | bias^2  variance | n_eff a=1 (r=0 / best r) | n_eff a=9 (r=0 / best r)")
    for t in sorted(next(iter(res.values()))):
        for name, traj in res.items():
            b2, v, o = score(traj[t], s2)
            print("  %5d | %-8s | %.4f  %.4f   | %7.2f / %7.2f         | %7.2f / %7.2f" % (t, name, b2, v, *o[1], *o[9]))
    w = S / (S + s2)
    b2, v, o = score(w, s2)
    print("  optimum (Wiener)  | %.4f  %.4f   | %7.2f / %7.2f         | %7.2f / %7.2f" % (b2, v, *o[1], *o[9]))


for s2 in (1.0, 25.0):                           # per-pixel SNR 1 (like W05) and 0.2 (like W20)
    show(f"POPULATION GD, sigma^2 = {s2:g} (signal power 1)", population(s2, (10, 100, 1000, 10000)), s2)
    show(f"FINITE DATA (M = 20 images, 10 copies), Adam, sigma^2 = {s2:g}", finite(s2, 20, (50, 200, 1000, 5000)), s2)
