# -*- coding: utf-8 -*-
"""GAP F (Archive/GAP/PREREGISTRATION_F.md): score a fastMRI-trained GAP B/D model (resolution/transplant_train.py) on
real ASL with the ASL split-half estimator (``_gr_eval.score_slice``, unchanged).

Each ASL slice is scaled by 1 / sigma_hat (the whole-image per-channel noise, the normalization of the GAP B/D noise
fields), decoded with the model at its label (SIG / sqrt(a) for the mean of a), and scaled back; n_eff is scale-free.
Output: <out>/neff.csv in the ``_gr_eval --single`` format (fold column = the slice's CV fold, for cross-fitted
selection by ``_subj_report``). Refuses to overwrite.

    python experiments/_tr_eval.py --run runs/gap/d/SUB_DT_s101 --iter 3000 --datasets 6 --out runs/gap/f/SUB_DT_s101/eval_it3000
"""
import sys, os, csv, json, argparse
import numpy as np
import torch
import _gr_eval as G                                          # sets sys.path / cwd to the repo
from _data import load_datasets, cv_folds
from _ratio_core import RATIOS
sys.path.insert(0, os.path.join(G.DIFF, "resolution"))
import transplant as TP                                       # noqa: E402
from transplant_train import load                             # noqa: E402


class Scaled:
    """``denoise_to`` of a transplant model for data with noise s0 per channel (model label SIG)."""

    def __init__(self, m, s0):
        self.m, self.s0 = m, s0

    def denoise_to(self, y, s, target_ratio=0.0, device="cpu"):
        k = TP.SIG / self.s0
        return self.m.denoise_to(y * k, s * k, target_ratio=target_ratio, device=device) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--datasets", type=int, nargs="+", default=[6]); ap.add_argument("--out", required=True)
    a_ = ap.parse_args()
    assert not os.path.exists(os.path.join(a_.out, "neff.csv")), "refusing to overwrite " + a_.out
    os.makedirs(a_.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, cfg = load(os.path.join(a_.run, "ckpt", f"res_{a_.iter:06d}.pkl"), dev)
    mode = "sub" if cfg["arm"].startswith("SUB") else "per"
    x, meta, sig = load_datasets(a_.datasets)
    folds = cv_folds(); rows = []
    for i, (d, s) in enumerate(meta):
        res = G.score_slice(Scaled(m, float(sig[i])), x[i].to(dev), float(sig[i]), dev, mode)
        for (a, rg), v in res.items():
            rows.append(dict(dataset=d, slice=s, fold=folds[s], sigma_hat=float(sig[i]), a=a, region=rg,
                             **{f"neff_r{r:.4f}": v[r] for r in RATIOS}))
    with open(os.path.join(a_.out, "neff.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(run=a_.run, iter=a_.iter, mode=mode, datasets=a_.datasets, cfg=cfg),
              open(os.path.join(a_.out, "eval_config.json"), "w"), indent=2)
    print("wrote", a_.out, len(rows), "rows")


if __name__ == "__main__":
    main()
