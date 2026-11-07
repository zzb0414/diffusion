# -*- coding: utf-8 -*-
"""GAP G: ``_gr_eval.py`` unchanged, on phase-randomized ASL (experiments/_pr.py). Same arguments; the model must have
been trained with --phase-rand (checked).

    python experiments/_pr_eval.py --run runs/gap/g/SUB_s101 --iter 3000 --mode sub --single --datasets 6 --out ...
"""
import os, json, sys
import _gr_eval as G
import _pr

_load = G.load_datasets


def load_pr(ids):
    x, meta, sig = _load(ids)
    return _pr.apply(x, meta), meta, sig


if __name__ == "__main__":
    run = sys.argv[sys.argv.index("--run") + 1]
    cfg = json.load(open(os.path.join(run, "foldall", "config.json"), encoding="utf-8"))
    assert cfg["ckpt_cfg"].get("phase_rand"), "model not trained on phase-randomized data"
    G.load_datasets = load_pr
    G.main()
