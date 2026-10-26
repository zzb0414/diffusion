# -*- coding: utf-8 -*-
"""Evaluate fold models on any of datasets 1-3 (GEN and MX_CV).

For a run directory with 5 fold models (``fold<K>/config.json`` + ``ckpt``),
every slice of the requested datasets is decoded by the fold model that held
its slice INDEX out (folds grouped by index, CV_R3's assignment), and scored
with the shared per-slice evaluation (``_ratio_core.evaluate_slice``): r sweep,
all C(10, a) splits, unbiased n_eff, sigma_out, oracle and noise-matched blur,
per-band n_eff. Rows are keyed by (dataset, slice); ``below_ladder`` flags
slices quieter than the model's lowest noise level (extrapolation).

    python experiments/_mx_eval.py --run runs/cv_r3 --datasets 2 3 --out runs/gen/A
"""
import sys, os, csv, json, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, DIFF); os.chdir(DIFF)

from _r3_eval import load                                    # noqa: E402
from _data import load_datasets, cv_folds                    # noqa: E402
from _ratio_core import evaluate_slice, RATIOS               # noqa: E402

DISPLAY = (1, 8, 11, 16, 17, 22, 25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir with fold0..4 (config.json, ckpt/)")
    ap.add_argument("--iter", type=int, default=2000)
    ap.add_argument("--datasets", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--ratios", type=float, nargs="+", default=None)
    ap.add_argument("--out", required=True)
    a_ = ap.parse_args()
    ratios = tuple(RATIOS if a_.ratios is None else
                   sorted({r for r in RATIOS if any(abs(r - q) < 1e-6 for q in a_.ratios)}))
    assert len(ratios) == (len(RATIOS) if a_.ratios is None else len(a_.ratios)), "unknown ratio"
    os.makedirs(a_.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, meta, sig = load_datasets(a_.datasets)
    folds = cv_folds()
    by_fold = {k: [i for i, (d, s) in enumerate(meta) if folds[s] == k] for k in range(5)}

    rows, brows = [], []
    for k in range(5):
        cfg = json.load(open(os.path.join(a_.run, f"fold{k}", "config.json"), encoding="utf-8"))
        assert sorted(cfg["held_out"]) == sorted(s for s, f in folds.items() if f == k), "fold mismatch"
        ck = os.path.join(a_.run, f"fold{k}", "ckpt", "%s_%06d.pkl" % (cfg["ckpt_cfg"]["loss"], a_.iter))
        model = load(ck, device)[0]
        smin = float(cfg["ckpt_cfg"]["sigma_min"])
        for i in by_fold[k]:
            d, s = meta[i]
            assert s not in cfg["train_slices"], "model trained on this slice index"
            sg = float(sig[i])
            key = dict(dataset=d, slice=s, fold=k, below_ladder=int(sg < smin))
            r_rows, b_rows, x0, n_lock, w_star = evaluate_slice(
                model, x[i].to(device), sg, device, key=key, bkey=dict(dataset=d, slice=s), ratios=ratios)
            if s in DISPLAY:
                np.save(os.path.join(a_.out, f"x0_d{d}_s{s:02d}.npy"), x0.astype(np.float32))
            rows += r_rows; brows += b_rows
            n1 = next(q for q in r_rows if q["a"] == 1 and q["region"] == "signal")["neff_r0.3333"]
            print(f"ds{d} slice {s:2d} fold {k} sigma {sg:.4f}{' (below ladder)' if sg < smin else ''}  "
                  f"noise@r=1/3 {n_lock:.3f}  w_matched {w_star:.2f}  a=1 signal n_eff r=1/3 {n1:.2f}",
                  flush=True)
        del model; torch.cuda.empty_cache()

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(a_.out, "per_slice_ratio.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a_.out, "bands_ratio.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(brows[0].keys())); w.writeheader(); w.writerows(brows)
    json.dump(dict(run=a_.run, iter=a_.iter, datasets=a_.datasets, ratios=list(ratios)),
              open(os.path.join(a_.out, "eval_config.json"), "w"), indent=2)
    print("wrote", a_.out, len(rows), "rows;", len(brows), "band rows; self-checks passed")


if __name__ == "__main__":
    main()
