# -*- coding: utf-8 -*-
"""R4 training/validation curves + checkpoint selection.

Plots train loss, val L1 vs mean(B) and val n_eff for a in {1, 5, 9}, and prints
the SAVED checkpoint (multiple of --ckpt-every) with the highest geometric-mean
val n_eff -- the selection rule declared in ``_r4_train.py``.

    python experiments/_r4_curves.py --run runs/r4_subset/l1
"""
import sys, os, csv, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/r4_subset/l1")
ap.add_argument("--ckpt-every", type=int, default=200)
a = ap.parse_args()
rows = list(csv.DictReader(open(os.path.join(a.run, "train_val.csv"), encoding="utf-8")))
it = np.array([int(r["iter"]) for r in rows])
f = lambda k: np.array([float(r[k]) for r in rows])
out = os.path.join(a.run, "eval"); os.makedirs(out, exist_ok=True)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(it, f("train_loss_avg50"), label="train (avg of last 50)")
ax[0].set_title("train loss (subset objective)"); ax[0].set_xlabel("iteration"); ax[0].legend()
for k, c in ((1, "C0"), (5, "C1"), (9, "C2")):
    ax[1].plot(it, f(f"val_l1_a{k}"), color=c, label=f"a={k}")
    ax[2].plot(it, f(f"val_neff_a{k}"), color=c, label=f"a={k}")
ax[2].plot(it, f("val_neff_gmean"), "k--", label="geometric mean")
ax[1].set_title("val L1: full Tweedie of mean(A) vs mean(B)"); ax[1].set_xlabel("iteration"); ax[1].legend()
ax[2].set_title("val n_eff (unbiased, all pixels)"); ax[2].set_xlabel("iteration"); ax[2].legend()
fig.tight_layout(); p = os.path.join(out, "curves.png"); fig.savefig(p, dpi=110); plt.close(fig)
print("wrote", p)

saved = [(i, g) for i, g in zip(it, f("val_neff_gmean")) if i % a.ckpt_every == 0]
best = max(saved, key=lambda z: z[1])
for i, g in saved:
    print(f"iter {i:>5}  val n_eff gmean {g:.2f}{'   <- selected' if i == best[0] else ''}")
print("SELECTED", best[0])
