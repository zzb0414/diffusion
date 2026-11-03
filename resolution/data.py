# -*- coding: utf-8 -*-
"""Multi-resolution study data (Archive/RES/PREREGISTRATION.md).

fastMRI brain multicoil TRAIN folder (fully sampled; the test folder is undersampled), AXT2 volumes with a 320 x 320
or 384 x 384 reconstruction over a 220 mm FOV (384 volumes are brought to 320 by keeping the central 320 x 320 of
k-space, i.e. the same FOV at 0.69 mm). Volumes are split by a seeded permutation into training and test volumes; the
central NSL slices of each volume are used.

Clean image x (complex, 320 x 320): coil images from the k-space (readout oversampling removed), coil-combined with
Inati maps (spiral/coils.py, per volume), scaled so that the 99th percentile of |x| over the volume is 1.

Resolutions: matrix M in RES at the same FOV, by keeping the central M x M of the (orthonormal, centred) 2-D k-space
and rescaling by M/320 (image intensity preserved). Linear, so Gaussian noise stays Gaussian.

Noise (physical): every copy j of image i gets ONE complex white k-space noise field on the full 320 x 320 grid (real
and imaginary parts N(0, S_K^2)), seeded per (split, image, copy); the copy at resolution M is the truncation of
(clean k-space + that noise). Per-channel image noise is therefore sigma(M) = S_K * M / 320 (SNR rises at lower
resolution, as for the same acquisition reconstructed at a coarser matrix). N = 10 copies.

Clean images are cached outside OneDrive (CACHE). fastMRI is read-only.
"""
import os, sys, glob, re
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (REPO, os.path.join(REPO, "experiments"), os.path.join(REPO, "baselines"), os.path.dirname(os.path.abspath(__file__)),
          os.path.abspath(os.path.join(REPO, "..", "spiral"))):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

FASTMRI = "E:/fastMRI/data/multicoil_train"
CACHE = os.path.join(os.path.expanduser("~"), "diffusion_cache", "res")
FULL = 320
RES = (320, 240, 160, 128, 80)            # matrices at FOV 220 mm: voxel 0.69, 0.92, 1.38, 1.72, 2.75 mm
FOV_MM = 220.0
N = 10
S_K = 1.0                                 # k-space noise std per channel -> sigma(320) = 1.0, sigma(80) = 0.25
NSL = 8                                   # central slices per volume
N_TEST_VOL = 15
SEED = 20261103
SPLIT_ID = {"train": 0, "test": 1}


def voxel_mm(M):
    return FOV_MM / M


def sigma(M):
    """per-channel image noise std at matrix M (physical setting)."""
    return S_K * M / FULL


def volumes():
    """sorted AXT2 files with a 320 x 320 or 384 x 384 / 220 mm reconstruction."""
    import h5py
    out = []
    for f in sorted(glob.glob(os.path.join(FASTMRI, "file_brain_AXT2_*.h5"))):
        with h5py.File(f, "r") as h:
            rs = h["ismrmrd_header"][()].decode().split("<reconSpace>")[1].split("</reconSpace>")[0]
            m = re.search(r"<matrixSize>\s*<x>(\d+)</x>\s*<y>(\d+)</y>", rs).groups()
            fov = re.search(r"<fieldOfView_mm>\s*<x>([\d.]+)</x>\s*<y>([\d.]+)</y>", rs).groups()
            if m in (("320", "320"), ("384", "384")) and abs(float(fov[0]) - FOV_MM) < 1e-3 and abs(float(fov[1]) - FOV_MM) < 1e-3:
                out.append(f)
    return out


def split_volumes():
    vols = volumes()
    perm = np.random.default_rng(SEED).permutation(len(vols))
    test = sorted(vols[i] for i in perm[:N_TEST_VOL])
    train = sorted(vols[i] for i in perm[N_TEST_VOL:])
    return {"train": train, "test": test}


def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def ifft2c(k):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def clean_volume(path):
    """(NSL, 320, 320) complex64 clean images of one volume, |x| p99 = 1."""
    import h5py
    from coils import calculate_csm_inati_iter
    with h5py.File(path, "r") as h:
        S = h["kspace"].shape[0]
        sl = list(range(S // 2 - NSL // 2, S // 2 + NSL // 2))
        k = h["kspace"][sl]                                            # (NSL, C, 640, Ny)
    im = ifft2c(k)                                                     # coil images (NSL, C, 2Mr, Ny)
    Mr = im.shape[-2] // 2                                             # reconstruction matrix (320 or 384)
    r0, c0 = (im.shape[-2] - Mr) // 2, (im.shape[-1] - Mr) // 2
    im = im[..., r0:r0 + Mr, c0:c0 + Mr]                               # readout oversampling removed, Ny -> Mr
    if Mr != FULL:                                                     # 384 -> 320: central k-space, same FOV
        im = _crop_k(im, Mr)
    _, x = calculate_csm_inati_iter(np.transpose(im, (1, 0, 2, 3)), smoothing=5, niter=5, thresh=1e-3)
    x = np.asarray(x, np.complex64)                                    # (NSL, 320, 320)
    return (x / np.percentile(np.abs(x), 99)).astype(np.complex64)


def _crop_k(im, Mr):
    """images at matrix Mr -> matrix 320 at the same FOV (central k-space), intensity preserved."""
    c = (Mr - FULL) // 2
    return ifft2c(fft2c(im)[..., c:c + FULL, c:c + FULL]) * (FULL / Mr)


def load_clean(split):
    """(n_images, 320, 320) complex64 and the volume index of each image (cached)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"clean_{split}.npz")
    if not os.path.exists(path):
        vols = split_volumes()[split]
        xs, vid = [], []
        for v, f in enumerate(vols):
            x = clean_volume(f)
            xs.append(x); vid += [v] * len(x)
            print(f"{split} volume {v + 1}/{len(vols)} {os.path.basename(f)}", flush=True)
        np.savez(path, x=np.concatenate(xs), vol=np.array(vid), files=np.array([os.path.basename(f) for f in vols]))
    z = np.load(path)
    return z["x"], z["vol"]


def to_res(k320, M):
    """centred 320 x 320 k-space (..., 320, 320) -> image at matrix M, intensity preserved."""
    c = (FULL - M) // 2
    return ifft2c(k320[..., c:c + M, c:c + M]) * (M / FULL)


def kspace_noise(split, i, n=N):
    rng = np.random.default_rng(SEED + 1000003 * SPLIT_ID[split] + i)
    return (S_K * (rng.standard_normal((n, FULL, FULL)) + 1j * rng.standard_normal((n, FULL, FULL)))).astype(np.complex64)


def image_res(x, split, i, M):
    """clean (M, M) and N noisy copies (N, M, M) of clean image x (complex 320 x 320) at matrix M."""
    k = fft2c(x)
    return to_res(k, M).astype(np.complex64), to_res(k[None] + kspace_noise(split, i), M).astype(np.complex64)
