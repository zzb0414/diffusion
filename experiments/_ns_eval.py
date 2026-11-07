# -*- coding: utf-8 -*-
"""GAP I: ``_gr_eval.py`` unchanged, on noise-stationarized ASL (experiments/_ns.py). Same arguments; the model must have
been trained by _ns_train.py (checked).

    python experiments/_ns_eval.py --run runs/gap/i/SUB_s101 --iter 3000 --mode sub --single --datasets 6 --out ...
"""
import os, sys
import _gr_eval as G
import _ns

_load = G.load_datasets


def load_ns(ids):
    x, meta, sig = _load(ids)
    return _ns.apply(x, meta), meta, sig


if __name__ == "__main__":
    run = sys.argv[sys.argv.index("--run") + 1]
    assert os.path.isfile(os.path.join(run, "foldall", "noise_stat.json")), "model not trained on noise-stationarized data"
    G.load_datasets = load_ns
    G.main()
