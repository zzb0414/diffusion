# -*- coding: utf-8 -*-
"""Qualitative displays: denoise one train / val / test slice with the four
R0 objectives around a chosen iteration, and lay the results out next to the
noisy inputs and the 10-realization reference.

    python _r2_displays.py --target-iter 2000 --sr 1.5 --cs 5

For every loss the checkpoint closest to ``--target-iter`` is used (l1 lives in
runs/r0_split, the rest in runs/r0_eval). One figure per slice shows each
estimate and its signed difference from the reference magnitude; a second puts
the magnitude estimates side by side; a third shows REAL, IMAGINARY, magnitude
and phase error, because the model output (and the reference) are complex --
a magnitude-only view hides whether the network preserved phase. Reporting
therefore carries both magnitude metrics (bias/RMSE/HF/HP) and complex ones
(relative complex error, per-quadrature RMSE, median |phase error|). PNGs go
to runs/displays/. All decoding is deterministic: one fixed seed, identical
across losses.
"""
import sys, os, glob, json, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, DIFF)
os.chdir(DIFF)

from unet import UNet
from n2n import N2N_DSM
from _r1_eval import load_data, mag, hf, hp, SEED_BASE

RUNROOT = {"l1": "runs/r0_split", "mse": "runs/r0_eval",
           "huber": "runs/r0_eval", "mag_l1": "runs/r0_eval"}
LOSSLABEL = {"l1": "L1", "mse": "MSE", "huber": "Huber", "mag_l1": "|T|-weighted L1"}

# One held-out-per-split sample set. Slice 17 sits in TRAIN for the R0 split; it
# was also the SRSTR_01 probe slice, so it is easy to compare against the older
# 6000-iteration L1 result.
SAMPLES = [("train", 17), ("val", 16), ("test", 25)]


def nearest_ckpt(loss, target):
    run = RUNROOT[loss]
    paths = sorted(glob.glob(os.path.join(run, loss, "ckpt", f"{loss}_*.pkl")))
    numbered = []
    for p in paths:
        b = os.path.basename(p)[:-4].split("_")[-1]
        if b.isdigit():
            numbered.append((int(b), p))
    return min(numbered, key=lambda t: abs(t[0] - target))


def load_model(path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    c = blob["cfg"]
    net = UNet(in_ch=2, out_ch=2, base_out=c.get("base_out", 32),
               emb_dim=c.get("emb_dim", 32)).to(device)
    net.load_state_dict(blob["state_dict"])
    n2n = N2N_DSM(L=c["L"], sigma_min=c["sigma_min"],
                  sigma_max=c["sigma_max"], backbone=net)
    n2n.net.eval()
    return n2n


def stats(est_m, ref_m):
    b = float(np.asarray(est_m).mean() / (np.asarray(ref_m).mean() + 1e-12))
    r = float(np.sqrt(((est_m - ref_m) ** 2).mean()))
    return b, r


def cstats(est, ref):
    """Complex-domain agreement. ``est``/``ref`` are (..., 2, H, W) real/imag.

    A magnitude-only score cannot tell a correct map from one whose phase is
    scrambled but whose |.| happens to match -- and |.| is exactly what an ASL
    map reports. So report both:
        rel_err   ||est-ref|| / ||ref||            (target 0)
        rmse_re   RMSE on the real part, per quadrature
        rmse_im   RMSE on the imaginary part, per quadrature
        pha_med   median |circular phase error| in degrees, over pixels where
                  the reference has enough signal to define a phase
    """
    d = est - ref
    num = float(np.mean(d.numpy() ** 2) ** 0.5)
    den = float(np.mean(ref.numpy() ** 2) ** 0.5) + 1e-12
    re_err = float(np.sqrt(np.mean(d[0].numpy() ** 2)))
    im_err = float(np.sqrt(np.mean(d[1].numpy() ** 2)))
    a1 = np.arctan2(ref[1].numpy(), ref[0].numpy())
    a2 = np.arctan2(est[1].numpy(), est[0].numpy())
    dphi = np.abs(np.angle(np.exp(1j * (a2 - a1))))
    m = np.hypot(ref[0].numpy(), ref[1].numpy())
    sel = m >= np.percentile(m, 50)          # ignore near-zero-signal pixels
    return dict(rel_err=num / den, rmse_re=re_err, rmse_im=im_err,
                pha_med_deg=float(np.degrees(np.median(dphi[sel]))) if sel.any()
                else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-iter", type=int, default=2000)
    ap.add_argument("--sr", type=float, default=1.5)
    ap.add_argument("--cs", type=int, default=5)
    ap.add_argument("--outdir", default="runs/displays")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = load_data()

    models = {}
    for loss in RUNROOT:
        it, path = nearest_ckpt(loss, args.target_iter)
        print(f"[{loss}] target {args.target_iter} -> {os.path.basename(path)} "
              f"(iter {it}, |d|={abs(it - args.target_iter)})", flush=True)
        models[loss] = (it, load_model(path, device))

    losses = list(RUNROOT)
    tags = " | ".join(f"{LOSSLABEL[l]}@{models[l][0]}" for l in losses)

    def denoise(ens, sigma_task, n2n):
        torch.manual_seed(SEED_BASE)
        return n2n.conditional_denoise(
            ens, sigma_task=sigma_task, sigma_ratio=args.sr,
            corrector_steps=args.cs, corrector_alpha=0.05,
            device=device).detach()

    results = {}
    hdr = ("split,slice,estimate,ckpt_iter,bias,rmse,hf_pct,hp_pct,"
           "rel_err,rmse_re,rmse_im,pha_med_deg\n")
    rows_csv = [hdr]
    for split, s in SAMPLES:
        ens = x[s].to(device)
        resid = ens - ens.mean(0, keepdim=True)
        sigma_task = resid.square().mean().sqrt().item()
        ref_c = ens.mean(0).detach()                   # complex 10-realization mean
        ref = mag(ref_c).cpu().numpy()
        noisy1 = mag(ens[0]).cpu().numpy()             # one raw realization
        # Averaging 10 magnitude images (what one would do without a model) --
        # distinct from the reference, which is the magnitude of the averaged
        # COMPLEX signal, so this row is a genuine baseline.
        noisym = torch.stack([mag(ens[k]) for k in range(ens.shape[0])]
                             ).mean(0).cpu().numpy()
        noisy_c = ens[0].detach()
        est, estc = {}, {}
        for loss in losses:
            rec = denoise(ens, sigma_task, models[loss][1])
            estc[loss] = rec.mean(0).detach()           # complex ensemble estimate
            est[loss] = mag(estc[loss]).cpu().numpy()   # its magnitude map
        results[s] = dict(split=split, sigma=sigma_task, ref_c=ref_c.cpu(), ref=ref,
                          noisy1=noisy1, noisym=noisym, noisy_c=noisy_c.cpu(),
                          est=est, estc=estc)
        # plain-text numbers so the figures can be checked without opening them
        ref_hf, ref_hp = hf(ref), hp(ref)

        def emit(name, m, it, cmap=None):
            b, rm = stats(m, ref)
            cs = cstats(cmap.detach().cpu(), ref_c.cpu()) if cmap is not None else None
            tail = (f"{cs['rel_err']:.5f},{cs['rmse_re']:.5f},{cs['rmse_im']:.5f},"
                    f"{cs['pha_med_deg']:.1f}" if cs else ",,,")
            rows_csv.append(f"{split},{s},{name},{it},{b:.4f},{rm:.5f},"
                            f"{100*hf(m)/ref_hf:.1f},{100*hp(m)/ref_hp:.1f},{tail}\n")

        emit("noisy_realization0", noisy1, "", cmap=noisy_c)
        emit("noisy_mean10", noisym, "")   # magnitude-average: not complex-owned
        for l in losses:
            emit(l, est[l], models[l][0], cmap=estc[l])
        del ens
        torch.cuda.empty_cache()
        print(f"  slice {s} ({split}) sigma_task={sigma_task:.4f}", flush=True)

    # ---- figure 1: one page per slice, estimate + signed difference --------
    cmap_diff = "coolwarm"
    for s, r in results.items():
        ref = r["ref"]
        vmax = np.percentile(ref, 99.7)
        rows = [("noisy input (realization 0)", r["noisy1"], None),
                ("noisy mean of 10", r["noisym"], None)]
        rows += [(f"{LOSSLABEL[l]} @ iter {models[l][0]}", r["est"][l], l)
                 for l in losses]
        dmax = max(np.percentile(np.abs(m - ref), 99.5)
                   for _, m, _ in rows)
        fig, axes = plt.subplots(len(rows), 2, figsize=(7.4, 3.4 * len(rows)),
                                 squeeze=False)
        for i, (title, m, loss) in enumerate(rows):
            b, rm = stats(m, ref)
            ax = axes[i][0]
            im = ax.imshow(m, cmap="viridis", vmin=0, vmax=vmax)
            ax.set_title(f"{title}\nbias {b:.2f} | RMSE {rm:.4f}", fontsize=8)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax = axes[i][1]
            im2 = ax.imshow(m - ref, cmap=cmap_diff, vmin=-dmax, vmax=dmax)
            ax.set_title("difference vs reference mean", fontsize=8)
            ax.axis("off")
            fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.02)
        suptitle = (f"slice {s} ({r['split']}) | sigma_task {r['sigma']:.4f} | "
                    f"src {args.sr}, corrector_steps {args.cs}")
        fig.suptitle(suptitle, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        p = os.path.join(args.outdir,
                         f"display_{r['split']}_s{s}_iter{args.target_iter}.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print("wrote", p, flush=True)

    # ---- figure 2: compact cross-slice scan -------------------------------
    fig, axes = plt.subplots(len(SAMPLES), 2 + len(losses), figsize=(2.35 * (2 + len(losses)), 2.6 * len(SAMPLES)),
                             squeeze=False)
    for i, (split, s) in enumerate(SAMPLES):
        r = results[s]
        ref = r["ref"]
        vmax = np.percentile(ref, 99.7)
        panels = [("noisy mean 10", r["noisym"])] + \
                 [(f"{LOSSLABEL[l]}\niter {models[l][0]}", r["est"][l]) for l in losses] + \
                 [("reference\n(mean 10)", ref)]
        for j, (t, m) in enumerate(panels):
            ax = axes[i][j]
            ax.imshow(m, cmap="viridis", vmin=0, vmax=vmax)
            b, rm = stats(m, ref)
            ax.set_title(t, fontsize=7)
            ax.set_xlabel(f"bias {b:.2f} / RMSE {rm:.4f}", fontsize=6)
            ax.set_xticks([]); ax.set_yticks([])
        axes[i][0].set_ylabel(f"{split} s{s}", fontsize=8)
    fig.suptitle(f"checkpoints near iter {args.target_iter} | sr {args.sr}, cs {args.cs}\n",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(args.outdir, f"overview_iter{args.target_iter}.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("wrote", p, flush=True)
    # ---- figure 3: complex view -------------------------------------------
    # Real / imaginary quadratures, magnitude, and circular phase error against
    # the complex reference. A magnitude-only figure cannot distinguish "correct
    # map" from "phase scrambled but |.| happens to match", and |.| is what an
    # ASL map ultimately reports -- so the quadrature and phase panels are where
    # that failure mode would show up.
    def phase_err_deg(z):
        z = np.asarray(z)
        a1 = np.arctan2(r["ref_c"][1].numpy(), r["ref_c"][0].numpy())
        a2 = np.arctan2(z[1], z[0])
        return np.degrees(np.abs(np.angle(np.exp(1j * (a2 - a1)))))

    for s, r in results.items():
        refc = r["ref_c"].numpy()
        mags = {"noisy real 0": r["noisy_c"].numpy()}
        mags.update({l: r["estc"][l].cpu().numpy() for l in losses})
        vm = max(np.percentile(np.abs(refc), 99.7),
                 max(np.percentile(np.abs(v), 99.7) for v in mags.values()))
        vm = float(vm)
        pmax = 180.0
        entries = [("reference\n(complex mean of 10)", refc, None)]
        entries += [(f"{LOSSLABEL[l]} @ iter {models[l][0]}", mags[l],
                     args.sr) for l in losses]
        fig, axes = plt.subplots(len(entries), 4,
                                 figsize=(9.6, 2.5 * len(entries)), squeeze=False)
        for i, (t, z, _) in enumerate(entries):
            z = np.asarray(z)
            panels = [
                ("real", z[0], "coolwarm", -vm, vm),
                ("imag", z[1], "coolwarm", -vm, vm),
                ("magnitude", np.hypot(z[0], z[1]), "viridis", 0, vm),
                ("|phase error| (deg)", phase_err_deg(z), "magma", 0, pmax),
            ]
            for j, (name, img, cm, lo, hi) in enumerate(panels):
                ax = axes[i][j]
                im = ax.imshow(img, cmap=cm, vmin=lo, vmax=hi)
                ax.set_title(name if i == 0 else "", fontsize=8)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cs = cstats(torch.from_numpy(z), torch.from_numpy(refc))
            axes[i][0].set_ylabel(t, fontsize=7)
            axes[i][-1].set_xlabel(
                f"median |dphi| {cs['pha_med_deg']:.1f}deg | "
                f"rel.err {cs['rel_err']:.3f}", fontsize=7)
        fig.suptitle(f"slice {s} ({r['split']}) complex view | sigma_task"
                     f" {r['sigma']:.4f} | sr {args.sr}, cs {args.cs}\n",
                     fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = os.path.join(args.outdir,
                         f"display_complex_{r['split']}_s{s}_iter{args.target_iter}.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        print("wrote", p, flush=True)

    print("checkpoints: " + tags)
    csv_path = os.path.join(args.outdir, f"display_metrics_iter{args.target_iter}.csv")
    open(csv_path, "w", encoding="utf-8").write("".join(rows_csv))
    print("wrote", csv_path, flush=True)


if __name__ == "__main__":
    main()
