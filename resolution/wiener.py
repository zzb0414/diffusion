# -*- coding: utf-8 -*-
"""GAP C0 (Archive/GAP/PREREGISTRATION_C.md): the best LINEAR denoiser's denoise-the-mean / denoise-each n_eff ratio.

Per-frequency Wiener filter from a population power spectrum P (signal) and noise power Nn: PER = Wiener at Nn applied
to each copy then averaged (= the filter on the mean), SUB = Wiener at Nn / a on the mean; both with the dial r chosen
from the same grid as the networks (best total error). Ratio = error(PER) / error(SUB) of the average image.
- fastMRI: P from the clean images (test split) at matrix 160 and 80, white noise sigma = 1 (Nn = 2 per k, ortho).
- ASL and the estimator check: P = mean over slices of |FFT(repetition mean)|^2 - Nn/N, Nn from the repetition
  deviations; both ring-averaged (40 radial bins). Validated on fastMRI matrix 80 from noisy copies (no clean data).
Text only -> runs/gap/report/wiener.txt.

    python resolution/wiener.py
"""
import os, numpy as np
import data as D
import transplant as TP

R = np.array([0, 1 / 12, 1 / 6, .25, 1 / 3, .5])
ASL = os.path.join(D.REPO, "..", "asl_for_diffusion")
SCANS = ("coil1/ds1_20260924", "coil1/1104a_20261104", "coil1/1104c_20261104", "coil2/ds3_20261005", "coil2_vss/ds2_20261005")


def ring(P, nb=40):
    H, W = P.shape
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(W), np.fft.fftfreq(H)))
    b = np.minimum((k / k.max() * nb).astype(int), nb - 1); out = np.zeros_like(P)
    for i in range(nb):
        out[b == i] = P[b == i].mean()
    return out


def gain(P, Nn, smooth=True):
    if smooth:
        P, Nn = ring(P), ring(Nn)
    P = np.clip(P, 0, None); out = []
    for a in (1, 3, 5, 9):
        err = lambda f: ((1 - f) ** 2 * P + f ** 2 * Nn / a).sum()
        wP, wS = P / (P + Nn), P / (P + Nn / a)
        out.append(min(err(wP + r * (1 - wP)) for r in R) / min(err(wS + r * (1 - wS)) for r in R))
    return out


def from_noisy(F):
    """F (H, W, slices, N) k-space copies -> (signal, noise) population power spectra."""
    N = F.shape[-1]; Fm = F.mean(-1); Nn = (np.abs(F - Fm[..., None]) ** 2).sum(-1) / (N - 1)
    return (np.abs(Fm) ** 2 - Nn / N).mean(-1), Nn.mean(-1)


def main():
    x, _ = D.load_clean("test")
    L = ["Wiener (linear) SUB / PER n_eff ratio, population spectrum, best r; a = 1, 3, 5, 9", ""]
    for M in (160, 80):
        P = np.mean([np.abs(np.fft.fft2(D.to_res(D.fft2c(z), M), norm="ortho")) ** 2 for z in x], 0)
        L.append(f"fastMRI matrix {M}, clean spectrum, white sigma 1: " + " ".join(f"{g:.3f}" for g in gain(P, np.full_like(P, 2.0), False)))
    F = np.transpose(np.stack([np.fft.fft2(TP.clean(z)[None] + TP.fields("test", i, "W"), norm="ortho") for i, z in enumerate(x)]), (2, 3, 0, 1))
    L.append("fastMRI matrix 80, estimated from noisy copies (check): " + " ".join(f"{g:.3f}" for g in gain(*from_noisy(F))))
    for p in SCANS:
        F = np.fft.fft2(np.load(os.path.join(ASL, p, "asl.npy")).astype(np.complex128), axes=(0, 1), norm="ortho")
        L.append(f"ASL {p}: " + " ".join(f"{g:.3f}" for g in gain(*from_noisy(F))))
    txt = "\n".join(L); print(txt)
    os.makedirs(os.path.join(D.REPO, "runs", "gap", "report"), exist_ok=True)
    open(os.path.join(D.REPO, "runs", "gap", "report", "wiener.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
