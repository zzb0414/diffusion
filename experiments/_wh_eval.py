# -*- coding: utf-8 -*-
"""GAP K: score a pre-whitened ASL model (experiments/_wh_train.py) on ORIGINAL-units ASL with the unchanged split-half
estimator (``_gr_eval.score_slice``): each slice's own map whitens the model input and the output is multiplied back
(``_wh.Whitened``). Output in the ``_gr_eval --single`` neff.csv format. Refuses to overwrite.

    python experiments/_wh_eval.py --run runs/gap/k/SUB_s101 --iter 3000 --mode sub --datasets 6 --out runs/gap/k/SUB_s101/eval_it3000
"""
import os, csv, json, argparse
import torch
import _gr_eval as G                                          # sets sys.path / cwd to the repo
from _data import load_datasets, cv_folds
from _ratio_core import RATIOS
import _wh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--mode", required=True, choices=["per", "sub"])
    ap.add_argument("--datasets", type=int, nargs="+", default=[6]); ap.add_argument("--out", required=True)
    a_ = ap.parse_args()
    assert os.path.isfile(os.path.join(a_.run, "foldall", "wh.json")), "model not trained by _wh_train.py"
    assert not os.path.exists(os.path.join(a_.out, "neff.csv")), "refusing to overwrite " + a_.out
    cfg = json.load(open(os.path.join(a_.run, "foldall", "config.json"), encoding="utf-8"))
    assert cfg["held_out"] == [] and not set(cfg["datasets"]) & set(a_.datasets), "test data used in training"
    os.makedirs(a_.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = os.path.join(a_.run, "foldall", "ckpt", "%s_%06d.pkl" % (cfg.get("ckpt_cfg", {}).get("loss", "l1"), a_.iter))
    m = G.load(ck, dev)[0]
    x, meta, sig = load_datasets(a_.datasets)
    folds = cv_folds(); rows = []
    for i, (d, s) in enumerate(meta):
        res = G.score_slice(_wh.Whitened(m, _wh.smap(x[i])), x[i].to(dev), float(sig[i]), dev, a_.mode)
        for (a, rg), v in res.items():
            rows.append(dict(dataset=d, slice=s, fold=folds[s], sigma_hat=float(sig[i]), a=a, region=rg,
                             **{f"neff_r{r:.4f}": v[r] for r in RATIOS}))
    with open(os.path.join(a_.out, "neff.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(run=a_.run, iter=a_.iter, mode=a_.mode, datasets=a_.datasets, whitened=True, smooth=_wh.SMOOTH),
              open(os.path.join(a_.out, "eval_config.json"), "w"), indent=2)
    print("wrote", a_.out, len(rows), "rows")


if __name__ == "__main__":
    main()
