# -*- coding: utf-8 -*-
"""Multi-dataset loading and the slice-index-grouped CV folds.

``load_datasets(ids)`` stacks scans (code ids, ``PATHS``) of ``../asl_for_diffusion`` with
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
# Scans by code id. The data folder is organized by coil and role (asl_for_diffusion/manifest.json, 2026-11-05):
# coil1 = good coil, coil2 = poor coil, coil2_vss = poor coil with VSS (off-distribution test), aside = not used.
# Ids are stable across the reorganization: 1 = ds1, 2 = ds2 (VSS), 3 = ds3, 5 = 1104a, 6 = 1104c, 7 = 1104b (aside);
# id 4 (the old 20261103 candidate) is excluded.
PATHS = {1: "coil1/ds1_20260924", 2: "coil2_vss/ds2_20261005", 3: "coil2/ds3_20261005", 5: "coil1/1104a_20261104",
         6: "coil1/1104c_20261104", 7: "aside/1104b_20261104"}


def scan_file(k, part="asl"):
    """path of a scan's array (part: asl, M0, ctrl, label) relative to the working directory."""
    return os.path.join(DATA_DIR, PATHS[k], part + ".npy")


def load_datasets(ids=(1,)):
    xs, meta = [], []
    for k in ids:
        img = np.load(scan_file(k))                                         # (Nx,Ny,Ns,Navg)
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
