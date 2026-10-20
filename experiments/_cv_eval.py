# -*- coding: utf-8 -*-
"""CV_R3 evaluation: every slice scored by the fold model that never saw it.

Implements ``Archive/CV_R3/PREREGISTRATION.md`` exactly; see there for the
claim, endpoints and decision rule. Outputs under ``runs/cv_r3/eval``:
  per_slice.csv     n_eff per slice x a x region x family (+ oracle blur width)
  bands.csv         per-band n_eff per slice at a = 5 (all pixels)
  endpoints.csv     every pre-registered statistic
  cv_neff_vs_a_<region>.png, cv_ratio_vs_sigma.png, cv_bands_a5.png
  maps/maps_cv_s<slice>.png  one a=5 split + the all-10 deployment, per slice

    python experiments/_cv_eval.py
"""
import sys, os, csv, json, itertools
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, DIFF); os.chdir(DIFF)

from _headtohead import blur_stack                           # noqa: E402
from _r3_train import load_data, sigma_hat                   # noqa: E402
from _r3_eval import load                                    # noqa: E402

CV = "runs/cv_r3"                     # overridable: --cv <dir> (ablation arms)
OUT = os.path.join(CV, "eval")
A_SIZES = (1, 2, 3, 5, 7, 9)
BLURS = (0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
REGIONS = ("signal", "all")
BANDS = ((0.0, 0.0625), (0.0625, 0.125), (0.125, 0.25), (0.25, 0.75))
EXCLUDE_SENS = (26, 29)
NBOOT = 10000


def cmag(z):
    return np.sqrt(z[..., 0, :, :] ** 2 + z[..., 1, :, :] ** 2)


def main():
    global CV, OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default=CV)
    ap.add_argument("--out", default=None, help="output dir (default <cv>/eval)")
    a_ = ap.parse_args()
    CV = a_.cv
    OUT = a_.out or os.path.join(CV, "eval")
    os.makedirs(os.path.join(OUT, "maps"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = load_data(); sig = sigma_hat(x)
    Ns = x.shape[0]
    fold_of, model, ckname = {}, {}, {}
    for k in range(5):
        cfg = json.load(open(os.path.join(CV, f"fold{k}", "config.json"), encoding="utf-8"))
        ckname[k] = "%s_002000.pkl" % cfg["ckpt_cfg"]["loss"]
        for s in cfg["held_out"]:
            assert s not in fold_of, "slice held out twice"
            fold_of[s] = k
    assert sorted(fold_of) == list(range(Ns)), "every slice must be held out exactly once"

    per, bands = [], []
    f_r = None
    for s in range(Ns):
        k = fold_of[s]
        if k not in model:
            model = {k: load(os.path.join(CV, f"fold{k}", "ckpt", ckname[k]), device)[0]}
        m = model[k]
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
            r3 = m.denoise_to(ens_t, sg, target_ratio=0.0, device=device).double().cpu().numpy()
        flat, r3f = ens.reshape(N, -1), r3.reshape(N, -1)
        for a in A_SIZES:
            combos = list(itertools.combinations(range(N), a))
            WA = np.zeros((len(combos), N)); WB = np.zeros((len(combos), N))
            for i, c in enumerate(combos):
                WA[i, list(c)] = 1.0 / a
                WB[i, [j for j in range(N) if j not in c]] = 1.0 / (N - a)
            ref = (WB @ flat).reshape(-1, 2, H, W)
            van = (WA @ flat).reshape(-1, 2, H, W)
            est = {"plain": van, "r3": (WA @ r3f).reshape(-1, 2, H, W)}

            def neff(e):
                d2 = ((e - ref) ** 2).mean(1)
                out = {}
                for r, v in masks.items():
                    mse = float((d2[:, v].mean(1) - s2[r] / (N - a)).mean())
                    out[r] = s2[r] / mse if mse > 0 else float("inf")
                return out

            def band_neff(e):
                sp = (np.abs(np.fft.fft2(e - ref)) ** 2).mean(axis=(0, 1))
                o = []
                for lo, hi in BANDS:
                    mk = (f_r >= lo) & (f_r < hi)
                    err = sp[mk].mean() - pn[mk].mean() / (N - a)
                    o.append(pn[mk].mean() / err if err > 0 else float("inf"))
                return o

            res = {fam: neff(e) for fam, e in est.items()}
            blurred = {w: blur_stack(van, w) for w in BLURS}
            bres = {w: neff(b) for w, b in blurred.items()}
            for r in REGIONS:
                wbest = max(BLURS, key=lambda w: bres[w][r])          # ORACLE on this slice
                per.append(dict(slice=s, fold=k, sigma_hat=sg, a=a, region=r,
                                plain=res["plain"][r], r3=res["r3"][r],
                                blur_oracle=bres[wbest][r], blur_width=wbest))
            if a == 5:
                wb = max(BLURS, key=lambda w: bres[w]["all"])
                row = dict(slice=s, sigma_hat=sg, blur_width=wb)
                for fam, e in (("plain", est["plain"]), ("r3", est["r3"]), ("blur", blurred[wb])):
                    for b, v in enumerate(band_neff(e)):
                        row[f"{fam}_band{b}"] = v
                bands.append(row)
                # maps: split A = 0-4 / B = 5-9, plus the all-10 deployment
                wsig = max(BLURS, key=lambda w: bres[w]["signal"])
                w9 = wsig
                vmax = float(np.percentile(cmag(mean10), 99.5))
                top = [("plain average of A (0-4)", cmag(van[0])),
                       ("R3: average of 5 denoised", cmag(est["r3"][0])),
                       (f"oracle-blurred average (w={wsig:.2f})", cmag(blurred[wsig][0])),
                       ("plain average of B (5-9)\nindependent reference", cmag(ref[0]))]
                bot = [("plain average of all 10", cmag(mean10)),
                       ("R3: average of 10 denoised", cmag(r3.mean(0))),
                       (f"blurred average of 10 (w={w9:.2f})", cmag(blur_stack(mean10[None], w9)[0])),
                       ("R3 minus plain (all 10)", cmag(r3.mean(0)) - cmag(mean10))]
                fig, ax = plt.subplots(2, 4, figsize=(12, 6.4))
                for j, (t, im) in enumerate(top + bot):
                    q = ax[j // 4, j % 4]
                    if j == 7:
                        q.imshow(im, cmap="RdBu_r", vmin=-vmax / 4, vmax=vmax / 4)
                    else:
                        q.imshow(im, cmap="gray", vmin=0, vmax=vmax)
                    q.set_title(t, fontsize=8); q.set_xticks([]); q.set_yticks([])
                fig.suptitle(f"slice {s} (fold {k}, held out; sigma_hat {sg:.4f}). Top: one a=5 split. "
                             "Bottom: deployment with all 10 scans.", fontsize=9)
                fig.tight_layout()
                fig.savefig(os.path.join(OUT, "maps", f"maps_cv_s{s:02d}.png"), dpi=90); plt.close(fig)
        r5 = next(p for p in per if p["slice"] == s and p["a"] == 5 and p["region"] == "signal")
        print(f"slice {s:2d} fold {k} sigma {sg:.4f}  a=5 signal: plain {r5['plain']:.2f}  "
              f"R3 {r5['r3']:.2f}  oracle blur {r5['blur_oracle']:.2f} (w {r5['blur_width']:.2f})",
              flush=True)

    for name, rows in (("per_slice.csv", per), ("bands.csv", bands)):
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- pre-registered statistics ---------------------------------------------------
    rng = np.random.default_rng(0)

    def stat(a, region, num, den, excl=()):
        rs = [p for p in per if p["a"] == a and p["region"] == region and p["slice"] not in excl]
        lr = np.log(np.array([p[num] / p[den] for p in rs]))
        boot = rng.choice(lr, size=(NBOOT, len(lr))).mean(1)
        p = wilcoxon(lr, alternative="greater").pvalue
        return dict(a=a, region=region, comparison=f"{num}/{den}",
                    excluded=",".join(map(str, excl)) or "-", n=len(lr),
                    gmean_ratio=float(np.exp(lr.mean())), ci_lo=float(np.exp(np.percentile(boot, 2.5))),
                    ci_hi=float(np.exp(np.percentile(boot, 97.5))), wins=int((lr > 0).sum()),
                    min_ratio=float(np.exp(lr.min())), wilcoxon_p=float(p))

    ends = []
    for excl in ((), EXCLUDE_SENS):
        for region in REGIONS:
            for a in A_SIZES:
                for num, den in (("r3", "plain"), ("r3", "blur_oracle"), ("blur_oracle", "plain")):
                    ends.append(stat(a, region, num, den, excl))
    with open(os.path.join(OUT, "endpoints.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ends[0].keys())); w.writeheader(); w.writerows(ends)

    print("\nPRE-REGISTERED ENDPOINTS (geometric-mean n_eff ratio over slices; 95% slice-bootstrap CI)")
    print(f"{'excl':6s}{'region':8s}{'a':>3s}  {'comparison':18s}{'n':>3s}{'ratio':>8s}{'95% CI':>18s}"
          f"{'wins':>7s}{'min':>7s}{'p (Wilcoxon, 1-sided)':>24s}")
    for e in ends:
        print(f"{e['excluded']:6s}{e['region']:8s}{e['a']:>3d}  {e['comparison']:18s}{e['n']:>3d}"
              f"{e['gmean_ratio']:>8.2f}   [{e['ci_lo']:6.2f},{e['ci_hi']:6.2f}]"
              f"{e['wins']:>4d}/{e['n']:<2d}{e['min_ratio']:>7.2f}{e['wilcoxon_p']:>24.2e}")
    prim = next(e for e in ends if e["excluded"] == "-" and e["region"] == "signal" and e["a"] == 5
                and e["comparison"] == "r3/plain")
    verdict = prim["ci_lo"] > 1 and prim["wilcoxon_p"] < 0.01
    print(f"\nPRIMARY (signal, a=5, R3 vs plain, all 34 slices): ratio {prim['gmean_ratio']:.2f} "
          f"[{prim['ci_lo']:.2f}, {prim['ci_hi']:.2f}], p = {prim['wilcoxon_p']:.2e} -> "
          f"{'SUPPORTED' if verdict else 'NOT SUPPORTED'} under the pre-registered rule")
    hb = sum(1 for b in bands if b["r3_band3"] >= b["blur_band3"])
    print(f"RESOLUTION (a=5, highest band): R3 >= oracle blur on {hb}/{len(bands)} slices")

    # ---- figures ----------------------------------------------------------------------
    for region in REGIONS:
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        for s in range(Ns):
            rs = [p for p in per if p["slice"] == s and p["region"] == region]
            ax.plot(A_SIZES, [p["r3"] for p in rs], color="C3", alpha=0.18, lw=0.8)
            ax.plot(A_SIZES, [p["blur_oracle"] for p in rs], color="C0", alpha=0.18, lw=0.8)
        for fam, c, lab in (("r3", "C3", "R3: average of denoised (held-out model)"),
                            ("blur_oracle", "C0", "oracle-blurred plain average")):
            g = [np.exp(np.mean(np.log([p[fam] for p in per if p["a"] == a and p["region"] == region])))
                 for a in A_SIZES]
            ax.plot(A_SIZES, g, "o-", color=c, lw=2.2, label=lab + " (geo. mean)")
        ax.plot(A_SIZES, A_SIZES, "ks--", label="plain average (n_eff = a)")
        ax.set_yscale("log"); ax.set_xticks(A_SIZES)
        ax.set_xlabel("realizations used (a)"); ax.set_ylabel("n_eff (log)")
        ax.set_title(f"CV over 34 slices, {region} pixels: thin lines = individual slices", fontsize=9)
        ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"cv_neff_vs_a_{region}.png"), dpi=110); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, a in zip(axs, (1, 5, 9)):
        rs = [p for p in per if p["a"] == a and p["region"] == "signal"]
        sx = [p["sigma_hat"] for p in rs]
        ax.scatter(sx, [p["r3"] / p["plain"] for p in rs], color="C3", s=18, label="R3 / plain")
        ax.scatter(sx, [p["r3"] / p["blur_oracle"] for p in rs], color="C0", s=18, marker="^",
                   label="R3 / oracle blur")
        for p in rs:
            if p["slice"] in EXCLUDE_SENS:
                ax.annotate(str(p["slice"]), (p["sigma_hat"], p["r3"] / p["plain"]), fontsize=7)
        ax.axhline(1, color="k", lw=0.8); ax.set_yscale("log")
        ax.set_xlabel("slice noise sigma_hat"); ax.set_title(f"a = {a}, signal region", fontsize=9)
    axs[0].set_ylabel("n_eff ratio (log)"); axs[0].legend(fontsize=7)
    fig.suptitle("Per-slice gain of R3 (held-out models) — above 1 = R3 better", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "cv_ratio_vs_sigma.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    pos = np.arange(len(BANDS))
    for off, fam, c, lab in ((-0.25, "plain", "k", "plain"), (0.0, "r3", "C3", "R3"),
                             (0.25, "blur", "C0", "oracle blur")):
        data = [[min(b[f"{fam}_band{i}"], 1e4) for b in bands] for i in range(len(BANDS))]
        bp = ax.boxplot(data, positions=pos + off, widths=0.2, patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set_facecolor(c); box.set_alpha(0.35)
        ax.plot([], [], color=c, lw=6, alpha=0.35, label=lab)
    ax.set_xticks(pos); ax.set_xticklabels([f"{lo:g}-{hi:g}" for lo, hi in BANDS])
    ax.set_yscale("log"); ax.set_xlabel("spatial frequency band (cycles/pixel)")
    ax.set_ylabel("per-band n_eff (log)")
    ax.set_title("Resolution check, a = 5, all 34 held-out slices (inf capped at 1e4)", fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cv_bands_a5.png"), dpi=110); plt.close(fig)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
