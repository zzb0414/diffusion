# -*- coding: utf-8 -*-
"""Does averaging DENOISED realizations beat plain averaging? (split-half test)

Each slice has N = 10 realizations. For every subset A of size a (all C(10, a)
of them) with B = its complement, three estimates are built from A ONLY:

  vanilla   mean of the a raw realizations                      (n_eff = a in theory)
  net       mean of the a outputs denoise_to(y_k, sigma_hat, r), each decoded
            alone; ratio r = 1 reproduces vanilla exactly
  blur      Gaussian blur of the vanilla mean (conventional smoothing baseline)

and scored against mean(B), which is independent of A. For any estimate e,

  E|e - mean(B)|^2 = E|e - clean|^2 + sigma^2 / |B|

so subtracting the realization-spread noise variance (estimated per region,
which keeps this unbiased even if noise varies across the image) gives an
UNBIASED estimate of the error against the clean map, without a clean map.
Reported as an equivalent number of averaged scans, n_eff = sigma^2 / MSE:
plain averaging should land at n_eff ~ a, which validates the method itself.

Regions: "all" pixels; "signal" = upper half of a heavily smoothed (Gaussian 3
px) |mean of 10| map; "background" = the rest. Heavy smoothing keeps the mask
nearly independent of any one realization's noise.

Selection is NOT oracle: ratio / blur width are chosen per (a, region) to
maximise mean n_eff over the VAL slices, then reported on the TEST slices.
Slice 17 was a training slice and is reported separately.

    python experiments/_split_test.py --run runs/r3_total/l1 --iter 2000
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

RATIOS = (1.0, 0.75, 0.5, 1.0 / 3.0, 0.25, 1.0 / 6.0, 1.0 / 12.0, 0.0)
BLURS = (0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
BLOCK, NBOOT = 8, 1000
A_SIZES = (1, 2, 3, 5, 7, 9)
REGIONS = ("all", "signal", "background")


def cmag(z):
    return np.sqrt(z[..., 0, :, :] ** 2 + z[..., 1, :, :] ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/r3_total/l1")
    ap.add_argument("--iter", type=int, default=2000)
    a_ = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss = os.path.basename(a_.run)
    out = os.path.join(a_.run, "eval", "split_test"); os.makedirs(out, exist_ok=True)
    m, it = load(os.path.join(a_.run, "ckpt", "%s_%06d.pkl" % (loss, a_.iter)), device)
    x = load_data(); sig = sigma_hat(x)

    rows = []          # one per (slice, a, family, knob, region): mean over all splits
    emaps, pix = {}, {}  # split-averaged squared-error maps; per-slice vpix + masks
    examples = {}
    for split, s in SAMPLES:
        ens_t = x[s].to(device); N = ens_t.shape[0]; sg = float(sig[s])
        ens = ens_t.double().cpu().numpy()                       # (N,2,H,W)
        H, W = ens.shape[-2:]
        mean10 = ens.mean(0)
        vpix = (((ens - mean10) ** 2).sum(0) / (N - 1)).mean(0)  # per-pixel noise var / channel
        smooth = gaussian_filter(cmag(mean10), 3.0)
        sigm = smooth > np.median(smooth)
        masks = {"all": np.ones((H, W), bool), "signal": sigm, "background": ~sigm}
        pix[s] = (vpix, masks)
        s2 = {k: float(vpix[v].mean()) for k, v in masks.items()}
        with torch.no_grad():
            outs = {r: m.denoise_to(ens_t, sg, target_ratio=r, device=device)
                    .double().cpu().numpy() for r in RATIOS}
        assert np.allclose(outs[1.0], ens, atol=1e-5), "ratio 1 must be identity"
        flat = ens.reshape(N, -1)
        for a in A_SIZES:
            combos = list(itertools.combinations(range(N), a))
            WA = np.zeros((len(combos), N)); WB = np.zeros((len(combos), N))
            for i, c in enumerate(combos):
                WA[i, list(c)] = 1.0 / a
                WB[i, [k for k in range(N) if k not in c]] = 1.0 / (N - a)
            ref = (WB @ flat).reshape(-1, 2, H, W)
            van = (WA @ flat).reshape(-1, 2, H, W)

            def rel_mse(est, key):
                e = ((est - ref) ** 2).mean(1)                    # (n,H,W), per channel
                emaps[(s, a) + key] = e.mean(0)                   # split-averaged, for bootstrap
                return {k: (e[:, v].mean(1) - s2[k] / (N - a)) / s2[k] for k, v in masks.items()}

            base = rel_mse(van, ("vanilla", 0.0))
            fams = [("vanilla", 0.0, base)]
            for r in RATIOS:
                if r != 1.0:
                    fams.append(("net", r, rel_mse((WA @ outs[r].reshape(N, -1)).reshape(-1, 2, H, W), ("net", r))))
            for w in BLURS:
                fams.append(("blur", w, rel_mse(blur_stack(van, w), ("blur", w))))
            for fam, knob, rm in fams:
                for k in REGIONS:
                    mu = float(rm[k].mean())
                    rows.append(dict(split=split, slice=s, sigma_hat=sg, a=a, n_splits=len(combos),
                                     family=fam, knob=knob, region=k, rel_mse=mu,
                                     n_eff=(1.0 / mu) if mu > 0 else float("inf"),
                                     frac_beats_vanilla=float((rm[k] < base[k]).mean())))
            if a == 5:
                examples[(split, s)] = dict(ens=ens, outs=outs, masks=masks)
        print(f"{split:6s} s{s:<3d} sigma {sg:.4f} done "
              f"(vanilla n_eff @a=5 all {1.0 / [r['rel_mse'] for r in rows if r['slice'] == s and r['a'] == 5 and r['family'] == 'vanilla' and r['region'] == 'all'][0]:.2f})",
              flush=True)

    p = os.path.join(out, "split_metrics.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", p, len(rows), "rows")

    # ---- knob selection on VAL, report on TEST ---------------------------------------
    def get(sl, a, fam, knob, reg):
        return next(r for r in rows if r["slice"] == sl and r["a"] == a and r["family"] == fam
                    and abs(r["knob"] - knob) < 1e-9 and r["region"] == reg)

    rng = np.random.default_rng(0)

    def boot_ci(sl, a, fam, knob, reg):
        """95% CI of n_eff from a spatial block bootstrap (pixel sampling only)."""
        vpix, masks = pix[sl]
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

    val = [s for sp, s in SAMPLES if sp == "val"]
    test = [s for sp, s in SAMPLES if sp == "test"]
    trn = [s for sp, s in SAMPLES if sp == "train"]
    sel = []
    for reg in ("all", "signal"):
        for a in A_SIZES:
            pick = {}
            for fam, knobs in (("net", [r for r in RATIOS if r != 1.0]), ("blur", BLURS)):
                pick[fam] = max(knobs, key=lambda k: np.mean([get(s, a, fam, k, reg)["n_eff"] for s in val]))
            for grp, sls in (("val", val), ("test", test), ("train", trn)):
                for s in sls:
                    v = get(s, a, "vanilla", 0.0, reg)
                    n3 = get(s, a, "net", 1.0 / 3.0, reg)
                    nb = get(s, a, "net", pick["net"], reg)
                    bb = get(s, a, "blur", pick["blur"], reg)
                    nlo, nhi = boot_ci(s, a, "net", pick["net"], reg)
                    blo, bhi = boot_ci(s, a, "blur", pick["blur"], reg)
                    sel.append(dict(region=reg, a=a, group=grp, slice=s,
                                    vanilla=v["n_eff"], net_third=n3["n_eff"],
                                    net_sel=nb["n_eff"], net_sel_ratio=pick["net"],
                                    net_lo=nlo, net_hi=nhi,
                                    blur_sel=bb["n_eff"], blur_sel_width=pick["blur"],
                                    blur_lo=blo, blur_hi=bhi))
    p = os.path.join(out, "split_selected.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sel[0].keys())); w.writeheader(); w.writerows(sel)
    print("wrote", p, len(sel), "rows")

    print("\nn_eff (equivalent number of plainly averaged scans); knobs chosen on VAL")
    print(f"{'region':8s}{'a':>3s} {'grp':5s}{'sl':>4s}{'vanilla':>8s}{'net@1/3':>8s}"
          f"{'net@sel':>8s}{'95% CI':>16s}{'(r)':>7s}{'blur@sel':>9s}{'95% CI':>16s}{'(w)':>6s}")
    for r in sel:
        print(f"{r['region']:8s}{r['a']:>3d} {r['group']:5s}{r['slice']:>4d}{r['vanilla']:>8.2f}"
              f"{r['net_third']:>8.2f}{r['net_sel']:>8.2f}  [{r['net_lo']:>5.1f},{r['net_hi']:>6.1f}]"
              f"{r['net_sel_ratio']:>7.3f}{r['blur_sel']:>9.2f}  [{r['blur_lo']:>5.1f},{r['blur_hi']:>6.1f}]"
              f"{r['blur_sel_width']:>6.2f}")

    # ---- figures ----------------------------------------------------------------------
    for reg in ("all", "signal"):
        fig, axs = plt.subplots(1, len(SAMPLES), figsize=(3.2 * len(SAMPLES), 3.6), sharey=True)
        for ax, (sp, s) in zip(axs, SAMPLES):
            rs = [r for r in sel if r["region"] == reg and r["slice"] == s]
            A = [r["a"] for r in rs]
            ax.plot(A, A, "k:", lw=0.8, label="n_eff = a")
            ax.plot(A, [r["vanilla"] for r in rs], "ko-", label="plain average")
            ax.plot(A, [r["net_third"] for r in rs], "s-", color="C1", label="avg of denoised, r=1/3")
            for key, lo, hi, st, c, lab in (("net_sel", "net_lo", "net_hi", "^-", "C3", "avg of denoised, r chosen on val"),
                                            ("blur_sel", "blur_lo", "blur_hi", "d--", "C0", "blurred average, w chosen on val")):
                y = np.array([r[key] for r in rs])
                err = np.array([[r[key] - r[lo] for r in rs], [r[hi] - r[key] for r in rs]])
                ax.errorbar(A, y, yerr=np.clip(err, 0, 1e4), fmt=st, color=c, capsize=2, label=lab)
            ax.set_title(f"{sp} slice {s}", fontsize=9); ax.set_xlabel("realizations averaged (a)")
            ax.set_xticks(A_SIZES)
        axs[0].set_ylabel("n_eff = sigma^2 / MSE vs clean"); axs[0].legend(fontsize=7)
        fig.suptitle(f"Split-half test, {reg} pixels: equivalent number of plainly averaged scans "
                     f"(R3 {loss} iter {it}; unbiased MSE vs clean from independent half)", fontsize=10)
        fig.tight_layout(); p = os.path.join(out, f"neff_{reg}.png"); fig.savefig(p, dpi=100); plt.close(fig)
        print("wrote", p)

    fig, axs = plt.subplots(1, len(SAMPLES), figsize=(3.2 * len(SAMPLES), 3.6), sharey=True)
    for ax, (sp, s) in zip(axs, SAMPLES):
        for a, c in ((1, "C0"), (5, "C1"), (9, "C2")):
            for reg, ls in (("all", "-"), ("signal", "--")):
                ys = [get(s, a, "vanilla", 0.0, reg)["n_eff"] if r == 1.0 else
                      get(s, a, "net", r, reg)["n_eff"] for r in RATIOS]
                ax.plot(RATIOS, ys, ls, color=c, marker="o", ms=3, label=f"a={a} {reg}")
        ax.set_title(f"{sp} slice {s}", fontsize=9); ax.set_xlabel("target_ratio (1 = plain average)")
    axs[0].set_ylabel("n_eff"); axs[0].legend(fontsize=7)
    fig.suptitle("Averaging denoised outputs: n_eff vs target_ratio", fontsize=10)
    fig.tight_layout(); p = os.path.join(out, "neff_vs_ratio.png"); fig.savefig(p, dpi=100); plt.close(fig)
    print("wrote", p)

    # example maps: A = realizations 0-4, B = 5-9, knobs chosen on val for a=5, signal region
    r_sel = next(r for r in sel if r["a"] == 5 and r["region"] == "signal")["net_sel_ratio"]
    w_sel = next(r for r in sel if r["a"] == 5 and r["region"] == "signal")["blur_sel_width"]
    for (sp, s), d in examples.items():
        ens, outs = d["ens"], d["outs"]
        A, B = ens[:5], ens[5:]
        imgs = [("plain average of A (0-4)", cmag(A.mean(0))),
                (f"average of denoised A, r={r_sel:.3f}", cmag(outs[r_sel][:5].mean(0))),
                ("average of denoised A, r=1/3", cmag(outs[1.0 / 3.0][:5].mean(0))),
                (f"blurred plain average of A, w={w_sel:.2f}", cmag(blur_stack(A.mean(0)[None], w_sel)[0])),
                ("plain average of B (5-9)\nindependent reference", cmag(B.mean(0))),
                ("plain average of all 10", cmag(ens.mean(0)))]
        vmax = float(np.percentile(imgs[-1][1], 99.5))
        fig, axs = plt.subplots(1, len(imgs), figsize=(3.0 * len(imgs), 3.4))
        for ax, (t, im) in zip(axs, imgs):
            ax.imshow(im, cmap="gray", vmin=0, vmax=vmax); ax.set_title(t, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        axs[0].contour(d["masks"]["signal"], levels=[0.5], colors="y", linewidths=0.5)
        fig.suptitle(f"{sp} slice {s}: one split, a=5 (yellow = signal-region mask). Knobs chosen on "
                     "val for a=5, signal region.", fontsize=9)
        fig.tight_layout()
        p = os.path.join(out, f"maps_split_{sp}_{s}.png"); fig.savefig(p, dpi=100); plt.close(fig)
        print("wrote", p)


if __name__ == "__main__":
    main()
