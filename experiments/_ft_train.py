# -*- coding: utf-8 -*-
"""GAP J: ``_gr_train.py`` unchanged, on ASL in unit-noise (fastMRI-transplant) units (experiments/_ft.py), optionally
initialized from a GAP D fastMRI DT checkpoint. Same arguments as _gr_train plus ``--init PATH`` (omit: random init,
the SC control). Writes foldall/ft.json.

    python experiments/_ft_train.py --init runs/gap/d/SUB_DT_s101/ckpt/res_003000.pkl --mode sub --fold -1 ...
"""
import os, sys, json
import _gr_train as T                              # first: its imports fix the OpenMP load order
import torch
import _ft

if __name__ == "__main__":
    init = None
    if "--init" in sys.argv:
        j = sys.argv.index("--init"); init = sys.argv[j + 1]; del sys.argv[j:j + 2]
    assert sys.argv[sys.argv.index("--fold") + 1] == "-1"
    T.load_datasets = _ft.unit(T.load_datasets)
    T.CFG["SIGMA_MAX"] = _ft.SIGMA_MAX
    if init:
        blob = torch.load(init, map_location="cpu", weights_only=False)
        Base = T.UNet

        class Init(Base):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.load_state_dict(blob["state_dict"])        # strict: identical architecture required
        T.UNet = Init
    T.main()
    out = os.path.join(sys.argv[sys.argv.index("--outdir") + 1], "foldall")
    json.dump({"unit_noise": True, "sigma_max": _ft.SIGMA_MAX, "init": init}, open(os.path.join(out, "ft.json"), "w"))
