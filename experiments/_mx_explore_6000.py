# -*- coding: utf-8 -*-
"""EXPLORATORY (not pre-registered): pooled @6000 vs dataset1-only @2000.

At iteration 6000 the pooled models (3x the slices) have drawn each slice about
as often as the dataset1-only models at 2000 -- the equal-passes pairing. The
pre-registered MX_CV primary compares both at 2000. This prints the paired
n_eff ratio (r = 1/3, signal) per arm, dataset and a, and appends a log section
labelled exploratory (once).

    python experiments/_mx_explore_6000.py
"""
import sys, os, csv
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from scipy.stats import wilcoxon

ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
os.chdir(ROOT)
MD = "WEEKEND_ITERATIONS.md"
HEAD = "### MX_CV exploratory — pooled @6000 (equal passes per slice) vs dataset1-only @2000"
ARMS = [("A", "L1", "runs/cv_r3"), ("B", "L1, noise-normalized", "runs/cv_ablation/l1_norm"),
        ("C", "MSE", "runs/cv_ablation/mse"), ("D", "MSE, noise-normalized", "runs/cv_ablation/mse_norm")]
KL = "neff_r0.3333"


def rows(path, ds_col):
    out = {}
    for r in csv.DictReader(open(os.path.join(path, "per_slice_ratio.csv"), encoding="utf-8")):
        if r["region"] == "signal":
            out[(int(r["dataset"]) if ds_col else 1, int(r["slice"]), int(r["a"]))] = float(r[KL])
    return out


res = {}
L = ["", HEAD, "",
     "*Not pre-registered.* Paired n_eff ratio, r = 1/3, signal region, 34 slices per dataset "
     "(geometric mean; slices where pooled is better; two-sided Wilcoxon p):", "",
     "| loss | dataset | a = 1 | a = 5 | a = 9 |", "|---|---:|---|---|---|"]
for k, n, d1 in ARMS:
    one = {**rows(os.path.join(d1, "ratio_eval"), False),
           **{kk: v for kk, v in rows(os.path.join("runs/gen", k), True).items() if kk[0] > 1}}
    p6 = rows(os.path.join("runs/mx_cv", k, "eval_it6000"), True)
    for d in (1, 2, 3):
        cells = []
        for a in (1, 5, 9):
            lr = np.log([p6[(d, s, a)] / one[(d, s, a)] for s in range(34)])
            res[(k, d, a)] = (float(np.exp(lr.mean())), int((lr > 0).sum()), float(wilcoxon(lr).pvalue))
            cells.append("%.3f× (%d/34, p=%.0e)" % res[(k, d, a)])
        L.append("| (%s) %s | %d | %s |" % (k, n, d, " | ".join(cells)))
up1 = [res[("A", d, 1)][0] for d in (1, 2, 3)]
dn9 = [res[(k, d, 9)][0] for k in "ABD" for d in (1, 2, 3)]
assert min(up1) > 1 and max(dn9) < 1
L += ["",
      "- Single maps improve with longer pooled training (L1 a = 1: %.3f–%.3f× on all datasets), but averaged "
      "estimates degrade (a = 9 for L1 and both noise-normalized arms: %.3f–%.3f×): the longer-trained models carry "
      "more error that is shared by every repetition, which averaging cannot remove. The checkpoint that is best for "
      "single maps is not the one to use when outputs are averaged — the stopping point depends on the use."
      % (min(up1), max(up1), min(dn9), max(dn9)), ""]
if HEAD not in open(MD, encoding="utf-8").read():
    open(MD, "a", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
