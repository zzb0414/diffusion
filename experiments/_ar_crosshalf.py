# -*- coding: utf-8 -*-
"""Adaptive-r headroom, honesty check (exploratory): is the block-oracle gain real or fitted noise?

Each slice's 10 repetitions are split into two disjoint halves of 5 (independent noise). The r map is chosen on
one half's split statistics and scored on the other half's (cross-half), compared with choosing and scoring on the
same half (in-sample). Rules: global r (per a, cross-fitted over the other folds, as in _ar_headroom.py), 8x8
blocks, 16x16 blocks, local-signal quintiles (cross-fitted). a in 1..4 (halves have N = 5). Both directions
averaged. Model A (runs/cv_r3), held-out folds.

    python experiments/_ar_crosshalf.py
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import _ar_headroom as H
from _r3_eval import load
from _data import load_datasets, cv_folds

H.A_SIZES = (1, 2, 3, 4)
HALVES = ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])


def block_map(st_choose, a, blk):
    p = st_choose["per_a"][a]; m = st_choose["mask"]
    rmap = np.zeros((80, 80))
    ids = (np.arange(80)[:, None] // blk) * (80 // blk) + np.arange(80)[None, :] // blk
    for b in np.unique(ids):
        sel = (ids == b) & m
        if sel.any():
            rmap[ids == b] = H.rstar(p["A"][sel].sum(), p["B"][sel].sum(), p["G"][sel].sum())
    return rmap


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds = cv_folds()
    x, meta, sig = load_datasets([1, 2, 3])
    ST = {}
    for k in range(5):
        model = load(f"runs/cv_r3/fold{k}/ckpt/l1_002000.pkl", device)[0]
        for i, (d, s) in enumerate(meta):
            if folds[s] == k:
                for h, idx in enumerate(HALVES):
                    ST[(d, s, h)] = H.slice_stats(model, x[i][idx].to(device), float(sig[i]), device)
        del model
    print("n_eff ratio to global r (geometric mean over slices and both directions), model A, signal region")
    print("  ds a | 8x8 in-sample | 8x8 cross-half | 16x16 in-sample | 16x16 cross-half | signal quintiles cf")
    for d in (1, 2, 3):
        keys = [(d, s) for s in range(34)]
        for a in H.A_SIZES:
            out = {n: [] for n in ("b8_in", "b8_x", "b16_in", "b16_x", "sig")}
            for (dd, s) in keys:
                train = [(dd, q) for q in range(34) if folds[q] != folds[s]]
                for h in (0, 1):
                    ev, ch = ST[(d, s, h)], ST[(d, s, 1 - h)]
                    tr = [(dd, q, h) for _, q in train]

                    def agg(sel=None):
                        t = np.zeros(3)
                        for kk in tr:
                            st = ST[kk]; p = st["per_a"][a]
                            mm = st["mask"] if sel is None else st["mask"] & sel(st)
                            t += np.array([p["A"][mm].sum(), p["B"][mm].sum(), p["G"][mm].sum()]) / (st["s2"] * st["mask"].sum())
                        return t
                    g = H.neff(ev, a, np.full((80, 80), H.rstar(*agg())))
                    out["b8_in"].append(H.neff(ev, a, block_map(ev, a, 8)) / g)
                    out["b8_x"].append(H.neff(ev, a, block_map(ch, a, 8)) / g)
                    out["b16_in"].append(H.neff(ev, a, block_map(ev, a, 16)) / g)
                    out["b16_x"].append(H.neff(ev, a, block_map(ch, a, 16)) / g)
                    rmap = np.zeros((80, 80))
                    for b in range(H.NB):
                        rmap[ev["bins"]["signal"] == b] = H.rstar(*agg(lambda t, b=b: t["bins"]["signal"] == b))
                    out["sig"].append(H.neff(ev, a, rmap) / g)
            gm = {n: float(np.exp(np.mean(np.log(v)))) for n, v in out.items()}
            print("  %d  %d | %.3f | %.3f | %.3f | %.3f | %.3f" % (d, a, gm["b8_in"], gm["b8_x"], gm["b16_in"], gm["b16_x"], gm["sig"]))


if __name__ == "__main__":
    main()
