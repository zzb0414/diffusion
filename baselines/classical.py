# -*- coding: utf-8 -*-
"""Classical baselines. Each takes the A half of a split, ``stack`` (a, 2, H, W) Re/Im, the single-repetition noise
``sigma`` (sigma_hat, per Re/Im channel) and one parameter value, and returns one estimate (2, H, W).

Image denoisers (blur, NLM, BM3D) act on the A-half mean, whose noise is sigma / sqrt(a). Multi-volume methods
(MP-PCA, LPCA, Patch2Self; dipy reference implementations) act jointly on the 2a Re/Im volumes of the A half
(shape (H, W, 1, 2a)); their denoised repetitions are averaged.
"""
import warnings
import numpy as np
from scipy.ndimage import gaussian_filter
import _common  # noqa: F401  (path setup)

GRIDS = {
    "avg": (0,),
    "blur": (0.5, 0.75, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0),
    "nlm": (0.4, 0.6, 0.8, 1.0, 1.2),
    "bm3d": (0.5, 0.75, 1.0, 1.5),
    "mppca": (1, 2, 3),
    "lpca": (1, 2, 3),
    "p2s": (0, 1, 2),
}


def _vol(stack):
    a = stack.shape[0]
    return np.moveaxis(stack.reshape(2 * a, *stack.shape[2:]), 0, -1)[:, :, None, :]      # (H, W, 1, 2a)


def _unvol(v, a):
    return np.moveaxis(v[:, :, 0, :], -1, 0).reshape(a, 2, *v.shape[:2]).mean(0)


def denoise(method, stack, sigma, p):
    a = stack.shape[0]
    m = stack.mean(0)
    s_a = sigma / np.sqrt(a)
    warnings.filterwarnings("ignore")
    if method == "avg":
        return m
    if method == "blur":
        return np.stack([gaussian_filter(c, p) for c in m])
    if method == "nlm":
        from skimage.restoration import denoise_nl_means
        return np.moveaxis(denoise_nl_means(np.moveaxis(m, 0, -1), h=p * s_a, sigma=s_a, patch_size=5,
                                            patch_distance=6, fast_mode=True, channel_axis=-1), -1, 0)
    if method == "bm3d":
        import bm3d
        return np.stack([np.asarray(bm3d.bm3d(c, sigma_psd=p * s_a)) for c in m])
    if method == "mppca":
        from dipy.denoise.localpca import mppca
        return _unvol(mppca(_vol(stack), patch_radius=(p, p, 0), suppress_warning=True), a)
    if method == "lpca":
        from dipy.denoise.localpca import localpca
        v = _vol(stack)
        return _unvol(localpca(v, sigma=np.full(v.shape[:3], sigma), patch_radius=(p, p, 0), suppress_warning=True), a)
    if method == "p2s":
        from dipy.denoise.patch2self import patch2self
        v = _vol(stack)
        return _unvol(patch2self(v, np.full(v.shape[-1], 1000.0), patch_radius=(p, p, 0), model="ols",
                                 b0_denoising=False, version=1, shift_intensity=False, clip_negative_vals=False), a)
    raise ValueError(method)
