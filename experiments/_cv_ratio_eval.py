# -*- coding: utf-8 -*-
"""CV_RATIO evaluation: target_ratio sweep x split size, per loss arm.

Implements ``Archive/CV_RATIO/PREREGISTRATION.md`` (evaluation only). For one
arm (a CV directory of 5 fold models), every slice is decoded by the fold model
that held it out; each of its 10 realizations is decoded ALONE once at r = 0
(x0_hat), and every ratio is the exact blend out = x0_hat + r (y - x0_hat).

Per slice, r in RATIOS, a in A_SIZES, all C(10, a) splits:
  n_eff (signal / all regions) of the average of a outputs vs mean(B), with the
  same unbiased estimator as ``_cv_eval.py`` (CV_R3).
Per slice, per r: achieved residual noise of a single output (unbiased spread of
the 10 outputs / sigma_hat) and the magnitude gain of the average of 10.
Per band (a in BAND_A): n_eff in 4 radial frequency bands.
Baselines: plain average (r = 1), oracle-MSE blur (as CV_R3) and NOISE-MATCHED
blur (width solved so a blurred realization has the arm's residual noise at
r = 1/3).

Self-checks against the existing CV results of the same arm: r = 0 reproduces
the R3 n_eff, r = 1 gives n_eff = a, oracle blur is identical.

Also saves x0_hat of the display slices (``x0_s<NN>.npy``) for the map figures.

    python experiments/_cv_ratio_eval.py --cv runs/cv_r3
"""
import sys, os, csv, json, itertools, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, DIFF); os.chdir(DIFF)

from _headtohead import blur_stack                           # noqa: E402
from _r3_train import load_data, sigma_hat                   # noqa: E402
from _r3_eval import load                                    # noqa: E402

RATIOS = (0.0, 1 / 12, 1 / 6, 0.25, 1 / 3, 0.5, 1.0)
A_SIZES = (1, 2, 3, 5, 7, 9)
BAND_A = (1, 5, 9)
BLURS = (0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
REGIONS = ("signal", "all")
BANDS = ((0.0, 0.0625), (0.0625, 0.125), (0.125, 0.25), (0.25, 0.75))
LOCK = 1 / 3
DISPLAY = (17, 16, 11, 1, 25, 22, 8)


def cmag(z):
    return np.sqrt(z[..., 0, :, :] ** 2 + z[..., 1, :, :] ** 2)


def spread(o, s):
    """Unbiased per-channel spread of N independent outputs, relative to sigma_hat."""
    n = o.shape[0]
    return float(np.sqrt((((o - o.mean(0)) ** 2).sum(0) / (n - 1)).mean()) / s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True)
    a_ = ap.parse_args()
    CV = a_.cv
    OUT = os.path.join(CV, "ratio_eval")
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = load_data(); sig = sigma_hat(x)
    Ns = x.shape[0]
    fold_of, ckname = {}, {}
    for k in range(5):
        cfg = json.load(open(os.path.join(CV, f"fold{k}", "config.json"), encoding="utf-8"))
        ckname[k] = "%s_002000.pkl" % cfg["ckpt_cfg"]["loss"]
        for s in cfg["held_out"]:
            fold_of[s] = k
    assert sorted(fold_of) == list(range(Ns))
    old = {(int(r["slice"]), int(r["a"]), r["region"]): r
           for r in csv.DictReader(open(os.path.join(CV, "eval", "per_slice.csv"), encoding="utf-8"))}

    rows, brows, model = [], [], {}
    f_r = None
    for s in range(Ns):
        k = fold_of[s]
        if k not in model:
            model = {k: load(os.path.join(CV, f"fold{k}", "ckpt", ckname[k]), device)[0]}
        ens_t = x[s].to(device); N = ens_t.shape[0]; sg = float(sig[s])
        ens = ens_t.double().cpu().numpy()
        H, W = ens.shape[-2:]
        if f_r is None:
            f_r = np.hypot(np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :])
        mean10 = ens.mean(0)
        dev = ens - mean10
        vpix = ((dev ** 2).sum(0) / (N - 1)).mean(0)
        pn = ((np.abs(np.fft.fft2(dev)) ** 2).sum(0) / (N - 1)).mean(0)
        smooth = gaussian_filter(cmag(mean10), 3.0)
        masks = {"signal": smooth > np.median(smooth), "all": np.ones((H, W), bool)}
        s2 = {r: float(vpix[v].mean()) for r, v in masks.items()}
        with torch.no_grad():
            x0 = model[k].denoise_to(ens_t, sg, target_ratio=0.0, device=device).double().cpu().numpy()
        if s in DISPLAY:
            np.save(os.path.join(OUT, f"x0_s{s:02d}.npy"), x0.astype(np.float32))
        outs = {r: x0 + r * (ens - x0) for r in RATIOS}
        noise = {r: spread(outs[r], sg) for r in RATIOS}
        gain = {r: float((cmag(outs[r].mean(0)) * cmag(mean10)).sum() / (cmag(mean10) ** 2).sum())
                for r in RATIOS}
        # noise-matched blur: blurred realizations' spread as a function of width
        bnoise = {w: spread(blur_stack(ens, w), sg) for w in BLURS}
        target = noise[LOCK]
        ws = np.array(BLURS); bn = np.array([bnoise[w] for w in BLURS])
        assert bn[-1] < target < bn[0], ("matched width outside grid", s, target)
        w_star = float(np.exp(np.interp(-np.log(target), -np.log(bn), np.log(ws))))
        matched_noise = spread(blur_stack(ens, w_star), sg)
        assert abs(matched_noise / target - 1) < 0.03, (s, matched_noise, target)

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

            est = {r: (WA @ outs[r].reshape(N, -1)).reshape(-1, 2, H, W) for r in RATIOS}
            res = {r: neff(est[r]) for r in RATIOS}
            bres = {w: neff(blur_stack(van, w)) for w in BLURS}
            matched = blur_stack(van, w_star)
            mres = neff(matched)
            for rg in REGIONS:
                o = old[(s, a, rg)]
                assert np.isclose(res[0.0][rg], float(o["r3"]), rtol=1e-6), ("r=0 mismatch", s, a, rg)
                assert np.isclose(res[1.0][rg], a, rtol=1e-6), ("r=1 must equal a", s, a, rg)
                wbest = max(BLURS, key=lambda w: bres[w][rg])
                assert np.isclose(bres[wbest][rg], float(o["blur_oracle"]), rtol=1e-6), ("blur", s, a, rg)
                row = dict(slice=s, fold=k, sigma_hat=sg, a=a, region=rg,
                           blur_oracle=bres[wbest][rg], blur_matched=mres[rg], w_matched=w_star)
                for r in RATIOS:
                    row[f"neff_r{r:.4f}"] = res[r][rg]
                for w in BLURS:                       # blur frontier: n_eff per width
                    row[f"blur_w{w:.2f}"] = bres[w][rg]
                rows.append(row)
            if a in BAND_A:
                b = dict(slice=s, a=a)
                for r in RATIOS:
                    for i, v in enumerate(band_neff(est[r])):
                        b[f"r{r:.4f}_band{i}"] = v
                for i, v in enumerate(band_neff(matched)):
                    b[f"blur_matched_band{i}"] = v
                brows.append(b)
        for r in RATIOS:
            rows.append(dict(slice=s, fold=k, sigma_hat=sg, a=0, region="per_r",
                             w_matched=w_star, ratio=r, noise=noise[r], gain10=gain[r]))
        for w in BLURS:                               # blur frontier: noise per width
            rows.append(dict(slice=s, fold=k, sigma_hat=sg, a=0, region="per_w",
                             width=w, noise=bnoise[w]))
        print(f"slice {s:2d} fold {k} sigma {sg:.4f}  noise@r=1/3 {noise[LOCK]:.3f}  "
              f"w_matched {w_star:.2f}  a=1 signal n_eff r=1/3 "
              f"{next(q for q in rows if q['slice'] == s and q['a'] == 1 and q['region'] == 'signal')['neff_r0.3333']:.2f}",
              flush=True)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(OUT, "per_slice_ratio.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "bands_ratio.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(brows[0].keys())); w.writeheader(); w.writerows(brows)
    print("wrote", OUT, len(rows), "rows;", len(brows), "band rows; self-checks passed")


if __name__ == "__main__":
    main()
