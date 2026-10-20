# -*- coding: utf-8 -*-
"""R4 vs R3 vs plain averaging vs blur on the split-half protocol.

Same protocol as ``_split_test.py`` (which produced the R3-only results and is
left untouched): for every subset A of size a (all C(10, a)), estimates are
built from A only and scored against the mean of the complement B; subtracting
B's noise variance gives an unbiased MSE vs the clean map, reported as
n_eff = sigma^2 / MSE (equivalent number of plainly averaged scans).

Families (each built from A only):
  vanilla  plain mean of A
  r3avg    R3 model: each realization of A denoised ALONE with
           denoise_to(y_k, sigma_hat, r), then averaged          (ratio r swept)
  r4       R4 subset model applied ONCE to mean(A) at its own noise
           sigma_hat/sqrt(a): denoise_to(mean(A), sigma_hat/sqrt(a), r)
  blur     Gaussian blur of the plain mean of A                   (width swept)

Knobs are chosen per (a, region) on VAL slices and reported on TEST; 95% CIs
from a spatial 8x8 block bootstrap (pixel sampling only).

RESOLUTION CHECK (new): MSE pools all spatial frequencies, so heavy smoothing
can score well while erasing detail. The error is therefore also split into
radial frequency bands: per band, error power of (estimate - mean(B)) minus the
noise power of mean(B) in that band (estimated from the realization spread),
giving an unbiased per-band n_eff = noise power / error power. A method that
erases structure shows a collapsing n_eff in the high bands.

Outputs under ``runs/r4_subset/<loss>/eval/split_test``.

    python experiments/_r4_split_test.py --r4-run runs/r4_subset/l1 --r4-iter 2400
"""
import sys, os, csv, itertools, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, DIFF); os.chdir(DIFF)

from _headtohead import blur_stack                           # noqa: E402
from _r3_train import load_data, sigma_hat                   # noqa: E402
from _r3_eval import load, SAMPLES                           # noqa: E402

RATIOS = (0.0, 1.0 / 12.0, 1.0 / 6.0, 0.25, 1.0 / 3.0, 0.5, 0.75)
BLURS = (0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
A_SIZES = (1, 2, 3, 5, 7, 9)
REGIONS = ("all", "signal")
FAMS = ("vanilla", "r3avg", "r4", "blur")
LABEL = {"vanilla": "plain average", "r3avg": "R3: average of separately denoised",
         "r4": "R4: denoise the subset mean", "blur": "blurred plain average"}
COLOR = {"vanilla": "k", "r3avg": "C1", "r4": "C3", "blur": "C0"}
BANDS = ((0.0, 0.0625), (0.0625, 0.125), (0.125, 0.25), (0.25, 0.75))   # cycles / pixel
BLOCK, NBOOT = 8, 1000


def cmag(z):
    return np.sqrt(z[..., 0, :, :] ** 2 + z[..., 1, :, :] ** 2)


def radial(H, W):
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    return np.hypot(fy, fx)


@torch.no_grad()
def r4_decode(m4, y, s, device, chunk=64):
    """Full Tweedie estimate (ratio 0) of a batch of subset means y (n,2,H,W)."""
    out = []
    for i in range(0, len(y), chunk):
        yt = torch.from_numpy(y[i:i + chunk]).float().to(device)
        out.append(m4.denoise_to(yt, s, target_ratio=0.0, device=device).double().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r3-ckpt", default="runs/r3_total/l1/ckpt/l1_002000.pkl")
    ap.add_argument("--r4-run", default="runs/r4_subset/l1")
    ap.add_argument("--r4-iter", type=int, required=True)
    a_ = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss = os.path.basename(a_.r4_run)
    out = os.path.join(a_.r4_run, "eval", "split_test"); os.makedirs(out, exist_ok=True)
    m3, it3 = load(a_.r3_ckpt, device)
    m4, it4 = load(os.path.join(a_.r4_run, "ckpt", "%s_%06d.pkl" % (loss, a_.r4_iter)), device)
    print(f"R3 model {a_.r3_ckpt} (iter {it3});  R4 model {a_.r4_run} iter {it4}")
    x = load_data(); sig = sigma_hat(x)

    rows, emaps, spec, pix, deploy = [], {}, {}, {}, {}
    for split, s in SAMPLES:
        ens_t = x[s].to(device); N = ens_t.shape[0]; sg = float(sig[s])
        ens = ens_t.double().cpu().numpy()
        H, W = ens.shape[-2:]
        mean10 = ens.mean(0)
        dev = ens - mean10
        vpix = ((dev ** 2).sum(0) / (N - 1)).mean(0)
        pnoise = ((np.abs(np.fft.fft2(dev)) ** 2).sum(0) / (N - 1)).mean(0)   # noise power / channel
        smooth = gaussian_filter(cmag(mean10), 3.0)
        sigm = smooth > np.median(smooth)
        masks = {"all": np.ones((H, W), bool), "signal": sigm}
        s2 = {k: float(vpix[v].mean()) for k, v in masks.items()}
        pix[s] = (vpix, masks, pnoise)
        with torch.no_grad():
            r3x0 = m3.denoise_to(ens_t, sg, target_ratio=0.0, device=device).double().cpu().numpy()
        flat, r3flat = ens.reshape(N, -1), r3x0.reshape(N, -1)
        for a in A_SIZES:
            combos = list(itertools.combinations(range(N), a))
            WA = np.zeros((len(combos), N)); WB = np.zeros((len(combos), N))
            for i, c in enumerate(combos):
                WA[i, list(c)] = 1.0 / a
                WB[i, [k for k in range(N) if k not in c]] = 1.0 / (N - a)
            ref = (WB @ flat).reshape(-1, 2, H, W)
            van = (WA @ flat).reshape(-1, 2, H, W)
            r3m = (WA @ r3flat).reshape(-1, 2, H, W)        # mean of separately denoised (ratio 0)
            r4x0 = r4_decode(m4, van, sg / np.sqrt(a), device)

            def score(est, key):
                d = est - ref
                e = (d ** 2).mean(1)
                emaps[(s, a) + key] = e.mean(0)
                spec[(s, a) + key] = (np.abs(np.fft.fft2(d)) ** 2).mean(axis=(0, 1))
                return {k: float((e[:, v].mean(1) - s2[k] / (N - a)).mean() / s2[k])
                        for k, v in masks.items()}

            res = [("vanilla", 0.0, score(van, ("vanilla", 0.0)))]
            for r in RATIOS:
                res.append(("r3avg", r, score(r3m + r * (van - r3m), ("r3avg", r))))
                res.append(("r4", r, score(r4x0 + r * (van - r4x0), ("r4", r))))
            for w in BLURS:
                res.append(("blur", w, score(blur_stack(van, w), ("blur", w))))
            for fam, knob, rm in res:
                for k in REGIONS:
                    rows.append(dict(split=split, slice=s, sigma_hat=sg, a=a, n_splits=len(combos),
                                     family=fam, knob=knob, region=k, rel_mse=rm[k],
                                     n_eff=(1.0 / rm[k]) if rm[k] > 0 else float("inf")))
            if a == 5:
                deploy[("a5", s)] = dict(van=van[0], r3m=r3m[0], r4x0=r4x0[0], ref=ref[0])
        # deployment: all 10 realizations, no reference left (display only)
        r4x10 = r4_decode(m4, mean10[None], sg / np.sqrt(N), device)[0]
        deploy[("a10", s)] = dict(van=mean10, r3m=r3x0.mean(0), r4x0=r4x10, masks=masks)
        v5 = [r["n_eff"] for r in rows if r["slice"] == s and r["a"] == 5 and r["family"] == "vanilla"
              and r["region"] == "all"][0]
        print(f"{split:6s} s{s:<3d} sigma {sg:.4f} done (vanilla n_eff a=5: {v5:.2f})", flush=True)

    p = os.path.join(out, "split_metrics.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", p, len(rows), "rows")

    # ---- selection on VAL, CIs, bands -------------------------------------------------
    def get(sl, a, fam, knob, reg):
        return next(r for r in rows if r["slice"] == sl and r["a"] == a and r["family"] == fam
                    and abs(r["knob"] - knob) < 1e-9 and r["region"] == reg)

    rng = np.random.default_rng(0)

    def boot_ci(sl, a, fam, knob, reg):
        vpix, masks, _ = pix[sl]
        e = emaps[(sl, a, fam, knob)]
        H, W = e.shape
        nb = (H // BLOCK) * (W // BLOCK)
        blk = lambda z: z[:H // BLOCK * BLOCK, :W // BLOCK * BLOCK].reshape(
            H // BLOCK, BLOCK, W // BLOCK, BLOCK).sum((1, 3)).ravel()
        m = masks[reg].astype(float)
        Em, Vm, Mm = blk(e * m), blk(vpix * m), blk(m)
        idx = rng.integers(0, nb, (NBOOT, nb))
        Es, Vs, Ms = Em[idx].sum(1), Vm[idx].sum(1), Mm[idx].sum(1)
        s2b = Vs / Ms
        mse = Es / Ms - s2b / (10 - a)
        ne = np.where(mse > 0, s2b / np.maximum(mse, 1e-30), np.inf)
        return float(np.percentile(ne, 2.5)), float(np.percentile(ne, 97.5))

    def bands(sl, a, fam, knob):
        _, _, pn = pix[sl]
        sp = spec[(sl, a, fam, knob)]
        f = radial(*sp.shape)
        outb = []
        for lo, hi in BANDS:
            mk = (f >= lo) & (f < hi)
            err = sp[mk].mean() - pn[mk].mean() / (10 - a)
            outb.append(float(pn[mk].mean() / err) if err > 0 else float("inf"))
        return outb

    val = [s for sp, s in SAMPLES if sp == "val"]
    sel = []
    for reg in REGIONS:
        for a in A_SIZES:
            pick = {"vanilla": 0.0}
            for fam, knobs in (("r3avg", RATIOS), ("r4", RATIOS), ("blur", BLURS)):
                pick[fam] = max(knobs, key=lambda k: np.mean([get(s, a, fam, k, reg)["n_eff"] for s in val]))
            for sp, s in SAMPLES:
                row = dict(region=reg, a=a, group=sp, slice=s)
                for fam in FAMS:
                    r = get(s, a, fam, pick[fam], reg)
                    lo, hi = boot_ci(s, a, fam, pick[fam], reg)
                    row.update({f"{fam}": r["n_eff"], f"{fam}_knob": pick[fam],
                                f"{fam}_lo": lo, f"{fam}_hi": hi})
                    if reg == "all":
                        for bi, v in enumerate(bands(s, a, fam, pick[fam])):
                            row[f"{fam}_band{bi}"] = v
                sel.append(row)
    keys = list(dict.fromkeys(k for r in sel for k in r))
    p = os.path.join(out, "split_selected.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(sel)
    print("wrote", p, len(sel), "rows")

    print("\nn_eff (knobs chosen on VAL); [95% CI]; knob in ()")
    print(f"{'region':7s}{'a':>3s} {'grp':5s}{'sl':>4s}{'vanilla':>8s}"
          f"{'R3 avg':>24s}{'R4 subset':>24s}{'blur':>24s}")
    for r in sel:
        cell = lambda f: f"{r[f]:7.2f} [{r[f + '_lo']:5.1f},{r[f + '_hi']:6.1f}]({r[f + '_knob']:.2f})"
        print(f"{r['region']:7s}{r['a']:>3d} {r['group']:5s}{r['slice']:>4d}{r['vanilla']:>8.2f} "
              f"{cell('r3avg')} {cell('r4')} {cell('blur')}")
    print("\nper-band n_eff (all pixels, VAL-chosen knobs); bands in cycles/pixel:",
          ", ".join(f"[{lo},{hi})" for lo, hi in BANDS))
    for r in sel:
        if r["region"] != "all" or r["a"] not in (1, 5, 9):
            continue
        print(f"a={r['a']} {r['group']:5s}{r['slice']:>4d}  " + "  ".join(
            f"{f}:" + "/".join(f"{r[f'{f}_band{b}']:.1f}" for b in range(len(BANDS))) for f in FAMS))

    # ---- figures ----------------------------------------------------------------------
    for reg in REGIONS:
        fig, axs = plt.subplots(1, len(SAMPLES), figsize=(3.2 * len(SAMPLES), 3.8), sharey=True)
        for ax, (sp, s) in zip(axs, SAMPLES):
            rs = [r for r in sel if r["region"] == reg and r["slice"] == s]
            A = [r["a"] for r in rs]
            ax.plot(A, A, ":", color="grey", lw=0.8)
            for fam in FAMS:
                y = np.array([r[fam] for r in rs])
                err = np.clip(np.array([[r[fam] - r[fam + "_lo"] for r in rs],
                                        [r[fam + "_hi"] - r[fam] for r in rs]]), 0, 1e4)
                ax.errorbar(A, y, yerr=err, fmt="o-", ms=3, color=COLOR[fam], capsize=2, label=LABEL[fam])
            ax.set_title(f"{sp} slice {s}", fontsize=9); ax.set_xlabel("realizations used (a)")
            ax.set_xticks(A_SIZES)
        axs[0].set_ylabel("n_eff = sigma^2 / MSE vs clean"); axs[0].legend(fontsize=7)
        fig.suptitle(f"Split-half test, {reg} pixels (R3 iter {it3}, R4 iter {it4}; knobs chosen on val)",
                     fontsize=10)
        fig.tight_layout(); p = os.path.join(out, f"neff_{reg}.png"); fig.savefig(p, dpi=100); plt.close(fig)
        print("wrote", p)

    centers = [0.5 * (lo + hi) for lo, hi in BANDS]
    for a in (1, 5, 9):
        fig, axs = plt.subplots(1, len(SAMPLES), figsize=(3.2 * len(SAMPLES), 3.8), sharey=True)
        for ax, (sp, s) in zip(axs, SAMPLES):
            r = next(q for q in sel if q["region"] == "all" and q["a"] == a and q["slice"] == s)
            for fam in FAMS:
                ax.plot(range(len(BANDS)), [r[f"{fam}_band{b}"] for b in range(len(BANDS))],
                        "o-", color=COLOR[fam], label=LABEL[fam])
            ax.set_xticks(range(len(BANDS)))
            ax.set_xticklabels([f"{lo:g}-{hi:g}" for lo, hi in BANDS], fontsize=7)
            ax.set_yscale("log"); ax.set_title(f"{sp} slice {s}", fontsize=9)
            ax.set_xlabel("spatial frequency band (cycles/pixel)")
        axs[0].set_ylabel("per-band n_eff (log)"); axs[0].legend(fontsize=7)
        fig.suptitle(f"Resolution check, a={a}: per-band n_eff. A method that erases structure "
                     "drops in the high bands.", fontsize=10)
        fig.tight_layout(); p = os.path.join(out, f"neff_bands_a{a}.png"); fig.savefig(p, dpi=100)
        plt.close(fig); print("wrote", p)

    # maps: one a=5 split (A = 0-4) and the a=10 deployment (all scans, no reference)
    def knob(fam, a, reg="signal"):
        return next(r for r in sel if r["region"] == reg and r["a"] == a)[f"{fam}_knob"]

    for (kind, s), d in deploy.items():
        sp = next(q for q, t in SAMPLES if t == s)
        a = 5 if kind == "a5" else 9            # a=10 has no val-selected knob: reuse a=9's
        r3k, r4k, bw = knob("r3avg", a), knob("r4", a), knob("blur", a)
        van = d["van"]
        imgs = [("plain average", cmag(van)),
                (f"R3 avg of denoised (r={r3k:.2f})", cmag(d["r3m"] + r3k * (van - d["r3m"]))),
                (f"R4 subset mean (r={r4k:.2f})", cmag(d["r4x0"] + r4k * (van - d["r4x0"]))),
                ("R4 subset mean (r=0)", cmag(d["r4x0"])),
                (f"blurred plain average (w={bw:.2f})", cmag(blur_stack(van[None], bw)[0]))]
        if kind == "a5":
            imgs.append(("plain average of B (5-9)\nindependent reference", cmag(d["ref"])))
        vmax = float(np.percentile(cmag(x[s].double().numpy().mean(0)), 99.5))
        fig, axs = plt.subplots(2, len(imgs), figsize=(3.0 * len(imgs), 6.4))
        for j, (t, im) in enumerate(imgs):
            axs[0, j].imshow(im, cmap="gray", vmin=0, vmax=vmax); axs[0, j].set_title(t, fontsize=8)
            base = imgs[-1][1] if kind == "a5" else imgs[0][1]
            axs[1, j].imshow(im - base, cmap="RdBu_r", vmin=-vmax / 4, vmax=vmax / 4)
            axs[1, j].set_title("minus " + ("independent reference" if kind == "a5" else "plain average"),
                                fontsize=8)
            for q in (0, 1):
                axs[q, j].set_xticks([]); axs[q, j].set_yticks([])
        title = (f"{sp} slice {s}: a=5 split, estimates from realizations 0-4 only; knobs chosen on val (a=5)"
                 if kind == "a5" else
                 f"{sp} slice {s}: DEPLOYMENT, all 10 realizations (no reference left); knobs from a=9 val choice")
        fig.suptitle(title, fontsize=9); fig.tight_layout()
        p = os.path.join(out, f"maps_{kind}_{sp}_{s}.png"); fig.savefig(p, dpi=100); plt.close(fig)
        print("wrote", p)


if __name__ == "__main__":
    main()
