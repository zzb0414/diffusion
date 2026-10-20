# -*- coding: utf-8 -*-
"""R3 evaluation: curves, matched-noise scoring per checkpoint, map displays.

For every checkpoint of an R3 run (total-noise labels) and every sample slice:
  net    ``denoise_to(y, sigma_hat, target_ratio)`` on each of the N
         realizations separately (one map in -> one map out), swept over
         target_ratio so its curve can be read at exactly noise = 1/3;
  blur   separable Gaussian on Re/Im, swept over width (the baseline to beat).
Achieved noise = spread between independently processed realizations,
N/(N-1)-weighted, divided by the UNBIASED sigma_hat (so identity reads 1.000;
the older head-to-head divided by the biased std and read 1.054).
relE / phase use realization 0 vs the LOO mean of the other 9 (complex);
gain / s_bias / sHF use the ensemble mean vs the full 10-scan mean (magnitude).

Outputs under ``<run>/eval``: eval.csv (every point), summary.csv (@1/3 per
ckpt x slice), curves.png, and for the display checkpoint (lowest mean val
relE@1/3 unless --display-iter) maps_<split>_<slice>.png + maps_overview.png.

    python experiments/_r3_eval.py --run runs/r3_total/l1
"""
import sys, os, glob, csv, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, DIFF); os.chdir(DIFF)

from unet import UNet                                        # noqa: E402
from n2n import N2N_DSM                                      # noqa: E402
from _r1_eval import mag, hf                                 # noqa: E402
from _headtohead import Scorer, blur_stack, interp_at        # noqa: E402
from _r3_train import load_data, sigma_hat                   # noqa: E402

SAMPLES = [("train", 17), ("val", 16), ("val", 11), ("val", 1),
           ("test", 25), ("test", 22), ("test", 8)]
TARGET = 1.0 / 3.0
RATIOS = (0.0, 0.1, 0.2, 0.3, 1.0 / 3.0, 0.4, 0.5, 0.6)
BLURS = (0.2, 0.35, 0.5, 0.625, 0.76, 0.887, 1.025, 1.187, 1.365, 1.56, 1.78, 2.13)
KEYS = ("noise", "gain", "sb", "shf", "relE", "pha")


def load(path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    c = blob["cfg"]
    assert c.get("noise_label") == "total", "not a total-noise checkpoint"
    net = UNet(in_ch=2, out_ch=2, base_out=c["base_out"], emb_dim=c["emb_dim"]).to(device)
    m = N2N_DSM(L=c["L"], sigma_min=c["sigma_min"], sigma_max=c["sigma_max"], backbone=net)
    m.net.load_state_dict(blob["state_dict"]); m.net.eval()
    return m, blob["train_iters"]


def pick(m):
    return {k: float(m[k]) for k in KEYS}


def curves_png(run, out):
    rows = list(csv.DictReader(open(os.path.join(run, "train_val.csv"), encoding="utf-8")))
    it = [int(r["iter"]) for r in rows]
    f = lambda k: [float(r[k]) for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(it, f("train_loss_avg50"), label="train (avg of last 50)")
    ax[0].plot(it, f("val_l1"), label="val L1 @ deployment")
    ax[0].set_title("loss: full Tweedie vs LOO mean"); ax[0].set_xlabel("iteration")
    ax[0].legend()
    ax[1].plot(it, f("val_noise13"), color="C2")
    ax[1].axhline(TARGET, ls="--", color="k", lw=0.8, label="target 1/3")
    ax[1].set_title("val: residual noise of denoise_to(1/3) / sigma_hat")
    ax[1].set_xlabel("iteration"); ax[1].legend()
    ax[2].plot(it, f("val_gain13"), color="C3")
    ax[2].axhline(1.0, ls="--", color="k", lw=0.8, label="reference gain 1")
    ax[2].set_title("val: magnitude gain of denoise_to(1/3)")
    ax[2].set_xlabel("iteration"); ax[2].legend()
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


@torch.no_grad()
def decode_all(m, ens, sg, ratio, device):
    return m.denoise_to(ens, sg, target_ratio=ratio, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/r3_total/l1")
    ap.add_argument("--display-iter", type=int, default=None)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join(a.run, "eval"); os.makedirs(out, exist_ok=True)
    curves_png(a.run, os.path.join(out, "curves.png"))
    print("wrote", os.path.join(out, "curves.png"))

    x = load_data(); sig = sigma_hat(x)
    paths = sorted(p for p in glob.glob(os.path.join(a.run, "ckpt", "*.pkl"))
                   if not p.endswith("_final.pkl"))
    allrows, summ = [], []
    print(f"{'iter':>5} {'split':6s}{'sl':>3s} {'fam':5s}{'knob':>7s}{'noise':>7s}"
          f"{'gain':>7s}{'s_bias':>7s}{'sHF%':>7s}{'relE':>7s}{'pha':>6s}")
    models = {}
    for p in paths:
        m, it = load(p, device); models[it] = p
        for split, s in SAMPLES:
            ens = x[s].to(device); N = ens.shape[0]; sg = float(sig[s])
            loo0 = (ens.sum(0) - ens[0]) / (N - 1)
            sc = Scorer(ens, sg, loo0, mag(ens.mean(0)).cpu().numpy())
            fams = {"net": [], "blur": []}
            for r in RATIOS:
                fams["net"].append(dict(knob=r, **pick(sc(decode_all(m, ens, sg, r, device)))))
            for b in BLURS:
                o = torch.from_numpy(blur_stack(ens.cpu().numpy(), b)).to(ens.dtype)
                fams["blur"].append(dict(knob=b, **pick(sc(o))))
            idm = pick(sc(ens))
            for fam, cur in fams.items():
                for c in cur:
                    allrows.append(dict(iter=it, split=split, slice=s, family=fam, **c))
                at = interp_at(cur, TARGET)
                if at is None:
                    print(f"{it:>5} {split:6s}{s:>3d} {fam:5s} unreached "
                          f"(noise {min(c['noise'] for c in cur):.3f}..)")
                    continue
                summ.append(dict(iter=it, split=split, slice=s, family=fam,
                                 sigma_hat=sg, **at))
                print(f"{it:>5} {split:6s}{s:>3d} {fam:5s}{at['knob']:>7.3f}{at['noise']:>7.3f}"
                      f"{at['gain']:>7.3f}{at['sb']:>7.3f}{at['shf']:>7.1f}"
                      f"{at['relE']:>7.3f}{at['pha']:>6.1f}", flush=True)
            summ.append(dict(iter=it, split=split, slice=s, family="id", sigma_hat=sg,
                             knob=0.0, **idm))
        del m; torch.cuda.empty_cache()

    for name, rows in (("eval.csv", allrows), ("summary.csv", summ)):
        with open(os.path.join(out, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("wrote", os.path.join(out, name), len(rows), "rows")

    # ---- per-checkpoint means: net vs blur at noise 1/3 ------------------------
    print("\nmean @ noise 1/3 (net vs blur), per checkpoint and split")
    print(f"{'iter':>5} {'split':6s}{'net relE':>9s}{'blur relE':>10s}{'net gain':>9s}"
          f"{'net sHF':>8s}{'blur sHF':>9s}{'wins':>6s}")
    val_rel = {}
    for it in sorted({r["iter"] for r in summ}):
        for sp in ("train", "val", "test"):
            n = [r for r in summ if r["iter"] == it and r["split"] == sp and r["family"] == "net"]
            b = {r["slice"]: r for r in summ if r["iter"] == it and r["split"] == sp
                 and r["family"] == "blur"}
            if not n:
                continue
            wins = sum(r["relE"] < b[r["slice"]]["relE"] for r in n if r["slice"] in b)
            mn = lambda rs, k: float(np.mean([r[k] for r in rs]))
            if sp == "val":
                val_rel[it] = mn(n, "relE")
            print(f"{it:>5} {sp:6s}{mn(n, 'relE'):>9.4f}{mn(list(b.values()), 'relE'):>10.4f}"
                  f"{mn(n, 'gain'):>9.3f}{mn(n, 'shf'):>8.1f}{mn(list(b.values()), 'shf'):>9.1f}"
                  f"{wins:>4d}/{len(n)}")

    # ---- displays ----------------------------------------------------------------
    it_d = a.display_iter or min(val_rel, key=val_rel.get)
    print(f"\ndisplay checkpoint: iter {it_d} "
          f"({'given' if a.display_iter else 'lowest mean val relE@1/3'})")
    m, _ = load(models[it_d], device)
    cols = ["reference\n|mean of 10|", "noisy single\n(realization 0)",
            "net denoise_to(1/3)\n(realization 0)", "net full Tweedie\n(realization 0)",
            "blur @ noise 1/3\n(realization 0)"]
    ov, ax_ov = plt.subplots(len(SAMPLES), len(cols), figsize=(3.0 * len(cols), 3.0 * len(SAMPLES)))
    for row, (split, s) in enumerate(SAMPLES):
        ens = x[s].to(device); sg = float(sig[s])
        bk = next((r["knob"] for r in summ if r["iter"] == it_d and r["slice"] == s
                   and r["family"] == "blur"), 1.0)
        maps = [mag(ens.mean(0)), mag(ens[0]),
                mag(decode_all(m, ens[:1], sg, TARGET, device)[0]),
                mag(decode_all(m, ens[:1], sg, 0.0, device)[0]),
                mag(torch.from_numpy(blur_stack(ens[:1].cpu().numpy(), bk))[0].to(ens.dtype))]
        maps = [mm.detach().cpu().numpy() for mm in maps]
        loo = mag((ens.sum(0) - ens[0]) / (ens.shape[0] - 1)).cpu().numpy()
        vmax = float(np.percentile(maps[0], 99.5))
        net = next(r for r in summ if r["iter"] == it_d and r["slice"] == s and r["family"] == "net")
        fig, ax = plt.subplots(2, len(cols), figsize=(3.0 * len(cols), 6.2))
        for j, mm in enumerate(maps):
            for a_ in (ax[0, j], ax_ov[row, j]):
                a_.imshow(mm, cmap="gray", vmin=0, vmax=vmax); a_.set_xticks([]); a_.set_yticks([])
            ax[0, j].set_title(cols[j], fontsize=9)
            if j == 0:
                ax[1, j].imshow(loo, cmap="gray", vmin=0, vmax=vmax)
                ax[1, j].set_title("LOO mean of other 9\n(single-map reference)", fontsize=9)
            else:
                ax[1, j].imshow(mm - loo, cmap="RdBu_r", vmin=-vmax / 3, vmax=vmax / 3)
                ax[1, j].set_title("minus LOO mean", fontsize=9)
            ax[1, j].set_xticks([]); ax[1, j].set_yticks([])
            if row == 0:
                ax_ov[0, j].set_title(cols[j], fontsize=9)
        ax_ov[row, 0].set_ylabel(f"{split} s{s}\nsigma {sg:.3f}", fontsize=9)
        fig.suptitle(f"R3 total-noise labels, iter {it_d} — {split} slice {s}, sigma_hat {sg:.4f}. "
                     f"Net @1/3: noise {net['noise']:.3f}, gain {net['gain']:.3f}, "
                     f"relE {net['relE']:.3f}", fontsize=10)
        fig.tight_layout()
        p = os.path.join(out, f"maps_{split}_{s}.png"); fig.savefig(p, dpi=110); plt.close(fig)
        print("wrote", p)
    ov.suptitle(f"R3 iter {it_d}: magnitude maps, shared grey scale per row (0 .. 99.5th pct "
                "of reference)", fontsize=11)
    ov.tight_layout()
    p = os.path.join(out, "maps_overview.png"); ov.savefig(p, dpi=90); plt.close(ov)
    print("wrote", p)


if __name__ == "__main__":
    main()
