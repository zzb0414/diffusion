# -*- coding: utf-8 -*-
"""GAP I: ``_gr_train.py`` unchanged, on noise-stationarized ASL (experiments/_ns.py). Same arguments plus --phase-rand,
whose hook (``import _pr; x = _pr.apply(x, meta)``) is redirected to _ns.apply; writes foldall/noise_stat.json.

    python experiments/_ns_train.py --mode sub --seed 101 --outdir runs/gap/i/SUB_s101 --fold -1 ... --phase-rand
"""
import os, sys, json
import _ns
sys.modules["_pr"] = _ns
import _gr_train as T

if __name__ == "__main__":
    assert "--phase-rand" in sys.argv and sys.argv[sys.argv.index("--fold") + 1] == "-1"
    T.main()
    out = os.path.join(sys.argv[sys.argv.index("--outdir") + 1], "foldall")
    json.dump({"noise_stat": True, "seed": _ns.SEED}, open(os.path.join(out, "noise_stat.json"), "w"))
