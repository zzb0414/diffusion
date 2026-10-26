# -*- coding: utf-8 -*-
"""Multi-dataset loading and the slice-index-grouped CV folds.

``load_datasets(ids)`` stacks datasets 1..3 of ``../asl_for_diffusion`` with
the per-slice normalization of ``_r3_train.load_data`` (dataset 1 alone is
reproduced exactly) and returns ``meta`` = [(dataset, slice)] and the per-slice
acquisition noise ``sigma_hat``. ``cv_folds()`` returns CV_R3's fold of every
slice index; slice s of every dataset belongs to that fold (same subject: the
datasets share anatomy, so folds are grouped by slice index).
"""
import os, json
import numpy as np
import torch

from _r3_train import sigma_hat                              # noqa: E402

DATA_DIR = "../asl_for_diffusion"
CV_SRC = "runs/cv_r3"


def load_datasets(ids=(1,)):
    xs, meta = [], []
    for k in ids:
        img = np.load(os.path.join(DATA_DIR, f"dataset{k}.npy"))          # (Nx,Ny,Ns,Navg)
        t = torch.from_numpy(img).permute(2, 3, 0, 1).contiguous()          # (Ns,Navg,Nx,Ny)
        t = t / t.abs().amax(dim=(1, 2, 3), keepdim=True)
        xs.append(torch.stack([t.real, t.imag], dim=2).contiguous())        # (Ns,Navg,2,Nx,Ny)
        meta += [(k, s) for s in range(t.shape[0])]
    x = torch.cat(xs)
    return x, meta, sigma_hat(x)


def cv_folds(src=CV_SRC, K=5):
    """{slice index: fold} from CV_R3's configs (each index held out exactly once)."""
    fold_of = {}
    for k in range(K):
        cfg = json.load(open(os.path.join(src, f"fold{k}", "config.json"), encoding="utf-8"))
        for s in cfg["held_out"]:
            assert s not in fold_of, "slice index held out twice"
            fold_of[s] = k
    assert sorted(fold_of) == list(range(len(fold_of)))
    return fold_of
