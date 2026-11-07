# -*- coding: utf-8 -*-
"""GAP K: ``_gr_train.py`` unchanged, on spatially pre-whitened ASL (experiments/_wh.py). Same arguments; writes
foldall/wh.json.

    python experiments/_wh_train.py --mode sub --seed 101 --outdir runs/gap/k/SUB_s101 --fold -1 ...
"""
import os, sys, json
import _gr_train as T                              # first: its imports fix the OpenMP load order
import _wh

if __name__ == "__main__":
    assert sys.argv[sys.argv.index("--fold") + 1] == "-1"
    T.load_datasets = _wh.whiten(T.load_datasets)
    T.main()
    out = os.path.join(sys.argv[sys.argv.index("--outdir") + 1], "foldall")
    json.dump({"whitened": True, "smooth": _wh.SMOOTH}, open(os.path.join(out, "wh.json"), "w"))
