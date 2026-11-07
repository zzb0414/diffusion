# -*- coding: utf-8 -*-
"""GAP J: ``_gr_eval.py`` unchanged, on ASL in unit-noise units (experiments/_ft.py; n_eff is scale-free). The model must
have been trained by _ft_train.py (checked).

    python experiments/_ft_eval.py --run runs/gap/j/FT_SUB_s101 --iter 3000 --mode sub --single --datasets 6 --out ...
"""
import os, sys
import _gr_eval as G
import _ft

if __name__ == "__main__":
    run = sys.argv[sys.argv.index("--run") + 1]
    assert os.path.isfile(os.path.join(run, "foldall", "ft.json")), "model not trained by _ft_train.py"
    G.load_datasets = _ft.unit(G.load_datasets)
    G.main()
