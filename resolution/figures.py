# -*- coding: utf-8 -*-
"""Visual comparison of the RES arms on one test slice (seed 101, a = 1, single noisy copy).

Rows: matrices M; columns: clean, noisy copy, arms A-E (dial r per (arm, setting, M) = the r with the lowest split-half
MSE over all test images, a = 1, no ground truth). Second figure: |error| x 3 on the same scale. One figure set per
noise setting (phys, matched). Titles carry the image's n_eff (sigma^2 / MSE vs ground truth, head mask).

    python resolution/figures.py [--img 60]
"""
import os, csv, argparse
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data as D
import models as Mo
import importlib.util as _iu
_sp = _iu.spec_from_file_location("res_evaluate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py"))
E = _iu.module_from_spec(_sp); _sp.loader.exec_module(E)

ARMS, ROWS = ("A", "B", "C", "D", "E"), (320, 128, 80)


def best_r(arm, setting, M):
    s = defaultdict(float)
    for r in csv.DictReader(open(os.path.join("runs", "res", "eval", f"{arm}_s101.csv"), encoding="utf-8")):
        if r["setting"] == setting and int(r["M"]) == M and r["a"] == "1":
            s[float(r["r"])] += float(r["mse_sh_head"])
    return min(s, key=s.get)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--img", type=int, default=60); args = ap.parse_args()
    os.chdir(D.REPO)
    dev = "cuda"
    x, vol = D.load_clean("test")
    models = {a: Mo.load(os.path.join("runs", "res", f"{a}_s101", "ckpt", "res_003000.pkl"), dev)[0] for a in ARMS}
    out = os.path.join("runs", "res", "report"); os.makedirs(out, exist_ok=True)
    for setting, sfun in E.SETTINGS.items():
        fig, ax = plt.subplots(len(ROWS), 2 + len(ARMS), figsize=(2.2 * (2 + len(ARMS)), 2.4 * len(ROWS)))
        fe, axe = plt.subplots(len(ROWS), 1 + len(ARMS), figsize=(2.2 * (1 + len(ARMS)), 2.4 * len(ROWS)))
        for i, M in enumerate(ROWS):
            s = sfun(M)
            clean, y = E.copies(x[args.img], args.img, M, s)
            head = np.abs(clean) > 0.05 * np.percentile(np.abs(clean), 99)
            vmax = np.percentile(np.abs(clean), 99.5)
            panels = [("clean", clean, None), (f"noisy (σ={s:.2f})", y[0], s ** 2 / (np.abs(y[0] - clean)[head] ** 2 / 2).mean())]
            for a in ARMS:
                r = best_r(a, setting, M)
                x0 = E.decode(models[a], a, y[:1], s, M, dev)[0]
                est = x0 + r * (y[0] - x0)
                panels.append((f"{a} (r={r:.2f})", est, s ** 2 / (np.abs(est - clean)[head] ** 2 / 2).mean()))
            for j, (t, im, ne) in enumerate(panels):
                ax[i, j].imshow(np.abs(im), cmap="gray", vmin=0, vmax=vmax)
                ax[i, j].set_title(t + ("" if ne is None else f"\nn_eff {ne:.1f}"), fontsize=8)
                if j:
                    axe[i, j - 1].imshow(np.abs(im - clean), cmap="magma", vmin=0, vmax=vmax / 3)
                    axe[i, j - 1].set_title(t + f"\nn_eff {ne:.1f}", fontsize=8)
            ax[i, 0].set_ylabel(f"M = {M}"); axe[i, 0].set_ylabel(f"M = {M}")
        for a_ in list(ax.flat) + list(axe.flat):
            a_.set_xticks([]); a_.set_yticks([])
        lab = "physical noise σ = M/320" if setting == "phys" else "matched noise σ = 0.5"
        fig.suptitle(f"RES, test image {args.img} (vol {vol[args.img]}), {lab}, single copy (a = 1), seed 101")
        fe.suptitle(f"|error| (×3 display gain), same image, {lab}")
        fig.tight_layout(); fe.tight_layout()
        fig.savefig(os.path.join(out, f"res_images_{setting}.png"), dpi=130)
        fe.savefig(os.path.join(out, f"res_errors_{setting}.png"), dpi=130)
        plt.close("all")
        print("wrote", setting)


if __name__ == "__main__":
    main()
