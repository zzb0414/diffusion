# -*- coding: utf-8 -*-
"""ASL-COIL exploratory (Archive/COIL/PREREGISTRATION.md): per-band SNR of ASL vs fastMRI, and the in-brain noise.

Per-copy SNR per radial frequency band, inside the brain / head mask (|M0| or |x| > 25% of its p99; masked image
spectra renormalized by the mask's fill fraction):
- ASL: noise spectrum from the repetition deviations (x N/(N-1)), signal spectrum = spectrum of the repetition mean
  minus noise/N. Datasets 1-3 and 1104a/c.
- fastMRI (test split, matrix 160, 15 volumes x 8 slices): clean image spectrum; white noise with sigma = 0.5 S_K per
  channel, S_K in {1, 2, 3, 6} (MEAN and MEAN-SNR).
Also the share of signal power at per-copy SNR in [1/9, 1], the frequencies where denoise-the-mean can beat
denoise-each at a = 9 (THEORY P5 / MEAN pre-registration), and the in-brain / background noise variance ratio (sigma_hat
averages the whole image). Text only -> runs/coil/band_snr.txt.

    python experiments/_band_snr.py
"""
import os, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASL = os.path.join(REPO, "..", "asl_for_diffusion")
sys.path.insert(0, os.path.join(REPO, "resolution"))
BANDS = ((0, .1), (.1, .25), (.25, .5), (.5, .75), (.75, 1.0))     # k / k_max, as the band-transfer tables
NBIN = 40
SCANS = {"ds1 (coil1)": "coil1/ds1_20260924", "ds2 (coil2, VSS)": "coil2_vss/ds2_20261005", "ds3 (coil2)": "coil2/ds3_20261005",
         "1104a (coil1)": "coil1/1104a_20261104", "1104c (coil1)": "coil1/1104c_20261104"}   # asl_for_diffusion/manifest.json


def kgrid(H, W):
    return np.hypot(np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :]) / 0.5      # k / k_max


def spec(y, mask):
    """mean |FFT|^2 over trailing axes of mask * y (H, W, ...), per pixel of the mask."""
    z = np.where(mask, y, 0)
    ps = np.abs(np.fft.fft2(z, axes=(0, 1))) ** 2 / (z.shape[0] * z.shape[1])
    return ps.reshape(ps.shape[0], ps.shape[1], -1).mean(-1) / mask.mean()


def summarize(S, Nn, k):
    band = [S[(k >= lo) & (k < hi)].mean() / Nn[(k >= lo) & (k < hi)].mean() for lo, hi in BANDS]
    e = np.linspace(0, 1, NBIN + 1); inside = k < 1
    sb = np.array([S[(k >= e[i]) & (k < e[i + 1])].sum() for i in range(NBIN)])
    nb = np.array([Nn[(k >= e[i]) & (k < e[i + 1])].sum() for i in range(NBIN)])
    snr = sb / nb
    win = sb[(snr >= 1 / 9) & (snr <= 1)].sum() / sb.sum()
    above = sb[snr > 1].sum() / sb.sum()
    return band, win, above, float(S[inside].sum() / Nn[inside].sum())


def asl_rows():
    out = []
    for name, p in SCANS.items():
        x = np.load(os.path.join(ASL, p, "asl.npy")); N = x.shape[-1]
        m0 = np.abs(np.load(os.path.join(ASL, p, "M0.npy")))
        brain = (m0 > 0.25 * np.percentile(m0, 99))[..., None]
        dev = (x - x.mean(-1, keepdims=True)) * np.sqrt(N / (N - 1))
        Nn = spec(dev, brain); S = spec(x.mean(-1), brain[..., 0]) - Nn / N
        r = np.abs(dev) ** 2
        ratio = r[np.broadcast_to(brain, r.shape)].mean() / r[np.broadcast_to(~brain, r.shape)].mean()
        out.append((name, *summarize(S, Nn, kgrid(*x.shape[:2])), ratio))
    return out


def fastmri_rows():
    import data as D
    x320, _ = D.load_clean("test")
    c = np.stack([D.to_res(D.fft2c(x), 160) for x in x320])                    # (n, 160, 160)
    mask = np.abs(c) > 0.25 * np.percentile(np.abs(c), 99, axis=(1, 2), keepdims=True)
    S = np.mean([spec(ci, mi) for ci, mi in zip(c, mask)], 0)
    k = kgrid(160, 160); out = []
    for sk in (1, 2, 3, 6):
        Nn = np.full_like(S, 2 * (0.5 * sk) ** 2)                               # white, complex variance 2 sigma^2
        out.append((f"fastMRI S_K = {sk}", *summarize(S, Nn, k), 1.0))
    return out


def main():
    rows = fastmri_rows() + asl_rows()
    L = ["Per-copy SNR (power) by band, inside the brain/head mask; k / k_max bands. 'window' = share of signal power at "
         "per-copy SNR in [1/9, 1] (where denoise-the-mean can beat denoise-each at a = 9); 'above' = share at SNR > 1.",
         "", "| data | " + " | ".join(f"{lo:g}–{hi:g}" for lo, hi in BANDS) + " | all | window | above | in-brain / background noise variance |",
         "|---|" + "---:|" * (len(BANDS) + 4)]
    for name, band, win, above, tot, ratio in rows:
        L.append(f"| {name} | " + " | ".join(f"{b:.3g}" for b in band) + f" | {tot:.3g} | {win:.2f} | {above:.2f} | {ratio:.2f} |")
    txt = "\n".join(L); print(txt)
    os.makedirs(os.path.join(REPO, "runs", "coil"), exist_ok=True)
    open(os.path.join(REPO, "runs", "coil", "band_snr.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
