# -*- coding: utf-8 -*-
"""GAP J (Archive/GAP/PREREGISTRATION_J.md): ASL in fastMRI-transplant units. Each slice is divided by its sigma_hat,
so its per-channel noise is TP.SIG = 1 (the label of the GAP B/D models) and the label vector becomes ones; with
SIGMA_MAX = 5 (models.LADDER_TOP * SIG) the _gr_train ladder (1/ladder_div .. 5, L = 40) is the GAP D ladder exactly,
so a fastMRI checkpoint's t-embedding keeps its meaning when it initializes an ASL model."""
import torch

SIGMA_MAX = 5.0
INIT = {"per": "A", "sub": "SUB", "subm": "SUBM"}       # fastMRI DT arm that initializes each ASL arm
INIT_PATH = "runs/gap/d/{arm}_DT_s{s}/ckpt/res_003000.pkl"


def unit(loader):
    def load(ids):
        x, meta, sig = loader(ids)
        return x / sig.view(-1, *([1] * (x.dim() - 1))).to(x.dtype), meta, torch.ones_like(sig)
    return load
