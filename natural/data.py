# -*- coding: utf-8 -*-
"""Natural-image data for NAT (Archive/NAT/PREREGISTRATION.md).

Train400 (training), Set12 (r selection, no ground truth used), BSD68 centre-cropped to 320x480 (test), grayscale
in [0, 1], from ``../../natural_images`` (DnCNN benchmark split). Every clean image gets N = 10 FIXED noisy copies,
seeded per (condition, split, image). Conditions: white sigma 50/255, 0.5, 2.0, and sigma 0.5 coloured with
dataset-1's median ASL noise spectrum (unit mean power, radial profile as ``experiments/_degrade.py``).
"""
import os, sys, glob
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG = os.path.abspath(os.path.join(REPO, "..", "natural_images"))
for p in (REPO, os.path.join(REPO, "experiments"), os.path.join(REPO, "baselines"), os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

N = 10
CONDITIONS = {"W50": (50 / 255, False), "W05": (0.5, False), "W20": (2.0, False), "C05": (0.5, True),
              "W035": (0.35, False), "W07": (0.7, False)}   # last two: test-only noise shift (C3); appended so earlier seeds stay
SEED = 20261031
SPLIT_ID = {"train": 0, "set12": 1, "bsd68": 2}
PROFILE_CACHE = os.path.join(REPO, "runs", "_cache", "asl_noise_profile.npz")


def _read(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def load_split(split):
    """list of float32 clean images in [0, 1]."""
    folder = {"train": "train400", "set12": "set12", "bsd68": "bsd68"}[split]
    ims = [_read(f) for f in sorted(glob.glob(os.path.join(IMG, folder, "*.png")))]
    if split == "bsd68":                                   # centre crop to multiples of 16, orientation kept
        out = []
        for im in ims:
            h, w = (320, 480) if im.shape[0] < im.shape[1] else (480, 320)
            t, l = (im.shape[0] - h) // 2, (im.shape[1] - w) // 2
            out.append(np.ascontiguousarray(im[t:t + h, l:l + w]))
        ims = out
    return ims


def asl_profile():
    """(radial frequency centres, unit-mean noise power) from dataset 1's repetitions (cached)."""
    if not os.path.exists(PROFILE_CACHE):
        from _data import load_datasets
        from _degrade import noise_profile
        x, _, _ = load_datasets([1])
        profs = [noise_profile(x[i].double().numpy()) for i in range(x.shape[0])]
        os.makedirs(os.path.dirname(PROFILE_CACHE), exist_ok=True)
        np.savez(PROFILE_CACHE, c=profs[0][0], p=np.median(np.stack([p for _, p in profs]), 0))
    z = np.load(PROFILE_CACHE)
    return z["c"], z["p"]


def noise(shape, sigma, colored, rng):
    """(..., H, W) Gaussian noise of per-pixel std sigma; coloured noise keeps the same total variance."""
    w = rng.standard_normal(shape).astype(np.float32)
    if not colored:
        return sigma * w
    H, W = shape[-2:]
    c, p = asl_profile()
    fr = np.hypot(np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :])
    P = np.interp(fr, c, p)
    filt = np.sqrt(P / P.mean())
    return (sigma * np.fft.ifft2(np.fft.fft2(w) * filt).real).astype(np.float32)


def copies(ims, cond, split):
    """list of (N, H, W) noisy copies, one per image; deterministic per (condition, split, image)."""
    sigma, colored = CONDITIONS[cond]
    cid = list(CONDITIONS).index(cond)
    out = []
    for i, im in enumerate(ims):
        rng = np.random.default_rng(SEED + 100000 * cid + 10000 * SPLIT_ID[split] + i)
        out.append(im[None] + noise((N,) + im.shape, sigma, colored, rng))
    return out
