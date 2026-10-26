# -*- coding: utf-8 -*-
"""Per-slice ratio evaluation shared by ``_cv_ratio_eval.py`` and ``_mx_eval.py``.

``evaluate_slice`` is the body that ``_cv_ratio_eval.py`` used to run inline:
decode the slice's N realizations once (x0_hat, r = 0), blend every ratio
(out = x0_hat + r (y - x0_hat)), and score all C(N, a) splits against the
complementary mean with the unbiased n_eff estimator; plus sigma_out, gain,
per-band n_eff, oracle-MSE blur and noise-matched blur. Rows are returned in
exactly the order and with exactly the keys the inline version wrote, so the
refactored ``_cv_ratio_eval.py`` reproduces its earlier CSVs byte for byte.
"""
import itertools
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from _headtohead import blur_stack                           # noqa: E402

RATIOS = (0.0, 1 / 12, 1 / 6, 0.25, 1 / 3, 0.5, 1.0)
A_SIZES = (1, 2, 3, 5, 7, 9)
BAND_A = (1, 5, 9)
BLURS = (0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
REGIONS = ("signal", "all")
BANDS = ((0.0, 0.0625), (0.0625, 0.125), (0.125, 0.25), (0.25, 0.75))
LOCK = 1 / 3


def cmag(z):
    return np.sqrt(z[..., 0, :, :] ** 2 + z[..., 1, :, :] ** 2)


def spread(o, s):
    """Unbiased per-channel spread of N independent outputs, relative to sigma_hat."""
    n = o.shape[0]
    return float(np.sqrt((((o - o.mean(0)) ** 2).sum(0) / (n - 1)).mean()) / s)


def evaluate_slice(model, ens_t, sg, device, key, bkey, ratios=RATIOS, check=None):
    """Evaluate one slice; returns (rows, band_rows, x0, noise_at_lock, w_matched).

    key / bkey: dict fields that lead every row / band row (e.g. slice, fold).
    check(a, region, res, bres_best): optional extra assertions per (a, region),
    with res = {r: n_eff} for this (a, region) and bres_best the oracle blur.
    """
    assert 0.0 in ratios and LOCK in ratios and 1.0 in ratios
    N = ens_t.shape[0]
    ens = ens_t.double().cpu().numpy()
    H, W = ens.shape[-2:]
    f_r = np.hypot(np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :])
    mean10 = ens.mean(0)
    dev = ens - mean10
    vpix = ((dev ** 2).sum(0) / (N - 1)).mean(0)
    pn = ((np.abs(np.fft.fft2(dev)) ** 2).sum(0) / (N - 1)).mean(0)
    smooth = gaussian_filter(cmag(mean10), 3.0)
    masks = {"signal": smooth > np.median(smooth), "all": np.ones((H, W), bool)}
    s2 = {r: float(vpix[v].mean()) for r, v in masks.items()}
    with torch.no_grad():
        x0 = model.denoise_to(ens_t, sg, target_ratio=0.0, device=device).double().cpu().numpy()
    outs = {r: x0 + r * (ens - x0) for r in ratios}
    noise = {r: spread(outs[r], sg) for r in ratios}
    gain = {r: float((cmag(outs[r].mean(0)) * cmag(mean10)).sum() / (cmag(mean10) ** 2).sum())
            for r in ratios}
    # noise-matched blur: blurred realizations' spread as a function of width
    bnoise = {w: spread(blur_stack(ens, w), sg) for w in BLURS}
    target = noise[LOCK]
    ws = np.array(BLURS); bn = np.array([bnoise[w] for w in BLURS])
    assert bn[-1] < target < bn[0], ("matched width outside grid", key, target)
    w_star = float(np.exp(np.interp(-np.log(target), -np.log(bn), np.log(ws))))
    matched_noise = spread(blur_stack(ens, w_star), sg)
    assert abs(matched_noise / target - 1) < 0.03, (key, matched_noise, target)

    rows, brows = [], []
    for a in A_SIZES:
        combos = list(itertools.combinations(range(N), a))
        WA = np.zeros((len(combos), N)); WB = np.zeros((len(combos), N))
        for i, c in enumerate(combos):
            WA[i, list(c)] = 1.0 / a
            WB[i, [j for j in range(N) if j not in c]] = 1.0 / (N - a)
        ref = (WB @ ens.reshape(N, -1)).reshape(-1, 2, H, W)
        van = (WA @ ens.reshape(N, -1)).reshape(-1, 2, H, W)

        def neff(e):
            d2 = ((e - ref) ** 2).mean(1)
            o = {}
            for rg, v in masks.items():
                mse = float((d2[:, v].mean(1) - s2[rg] / (N - a)).mean())
                o[rg] = s2[rg] / mse if mse > 0 else float("inf")
            return o

        def band_neff(e):
            sp = (np.abs(np.fft.fft2(e - ref)) ** 2).mean(axis=(0, 1))
            res = []
            for lo, hi in BANDS:
                mk = (f_r >= lo) & (f_r < hi)
                err = sp[mk].mean() - pn[mk].mean() / (N - a)
                res.append(pn[mk].mean() / err if err > 0 else float("inf"))
            return res

        est = {r: (WA @ outs[r].reshape(N, -1)).reshape(-1, 2, H, W) for r in ratios}
        res = {r: neff(est[r]) for r in ratios}
        bres = {w: neff(blur_stack(van, w)) for w in BLURS}
        matched = blur_stack(van, w_star)
        mres = neff(matched)
        for rg in REGIONS:
            assert np.isclose(res[1.0][rg], a, rtol=1e-6), ("r=1 must equal a", key, a, rg)
            wbest = max(BLURS, key=lambda w: bres[w][rg])
            if check is not None:
                check(a, rg, {r: res[r][rg] for r in ratios}, bres[wbest][rg])
            row = dict(**key, sigma_hat=sg, a=a, region=rg,
                       blur_oracle=bres[wbest][rg], blur_matched=mres[rg], w_matched=w_star)
            for r in ratios:
                row[f"neff_r{r:.4f}"] = res[r][rg]
            for w in BLURS:                       # blur frontier: n_eff per width
                row[f"blur_w{w:.2f}"] = bres[w][rg]
            rows.append(row)
        if a in BAND_A:
            b = dict(**bkey, a=a)
            for r in ratios:
                for i, v in enumerate(band_neff(est[r])):
                    b[f"r{r:.4f}_band{i}"] = v
            for i, v in enumerate(band_neff(matched)):
                b[f"blur_matched_band{i}"] = v
            brows.append(b)
    for r in ratios:
        rows.append(dict(**key, sigma_hat=sg, a=0, region="per_r",
                         w_matched=w_star, ratio=r, noise=noise[r], gain10=gain[r]))
    for w in BLURS:                               # blur frontier: noise per width
        rows.append(dict(**key, sigma_hat=sg, a=0, region="per_w", width=w, noise=bnoise[w]))
    return rows, brows, x0, noise[LOCK], w_star
