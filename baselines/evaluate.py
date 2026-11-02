# -*- coding: utf-8 -*-
"""Score methods on the fixed split subset (Archive/BASE/PREREGISTRATION.md). Long-format CSV, one row per
(slice, method, seed, parameter, a). Classical methods run in a process pool; network methods decode per
repetition (one-shot Tweedie, blended with r) with the held-out fold model. Refuses to overwrite.

    python baselines/evaluate.py --methods avg blur nlm bm3d mppca lpca p2s --out runs/base/classical
    python baselines/evaluate.py --methods ours --out runs/base/ours
    python baselines/evaluate.py --methods n2s --out runs/base/n2s_eval
    python baselines/evaluate.py --methods ours --check            # all splits: must reproduce evaluate_slice
"""
import os, sys, csv, json, time, argparse
import numpy as np
import _common
from _common import A_EVAL, SliceScorer

RGRID = (0.0, 1 / 12, 1 / 6, 0.25, 1 / 3, 0.5)
NETS = {"ours": {101: ("runs/cv_ablation/mse_norm", "mse"), 102: ("runs/dec/A/dsm_s102", "mse"),
                 103: ("runs/dec/A/dsm_s103", "mse")},
        "n2s": {101: ("runs/base/n2s2/s101", "n2s"), 102: ("runs/base/n2s2/s102", "n2s"),
                103: ("runs/base/n2s2/s103", "n2s")},
        "n2n": {sd: (f"runs/dec/A/blind_s{sd}", "mse") for sd in (101, 102, 103)},        # classic N2N (DEC BLIND)
        "n2n_sigma": {sd: (f"runs/dec/A/dirn_s{sd}", "mse") for sd in (101, 102, 103)},   # N2N + sigma (DEC DIR_N)
        "ne2ne": {sd: (f"runs/base/ne2ne/s{sd}", "ne2ne") for sd in (101, 102, 103)},
        "r2r": {sd: (f"runs/base/r2r/s{sd}", "r2r") for sd in (101, 102, 103)},
        "n2self": {sd: (f"runs/base/n2self/s{sd}", "n2self") for sd in (101, 102, 103)},
        "ddm2": {sd: (f"runs/base/ddm2/s{sd}", "ddm2") for sd in (101, 102, 103)},       # DDM2-like, one sample
        "ddm2_tw": {sd: (f"runs/base/ddm2/s{sd}", "ddm2") for sd in (101, 102, 103)}}    # same net, one-shot Tweedie
R2R_DRAWS, R2R_ALPHA = 8, 0.5
ITER = 2000                                                  # evaluated iteration (fixed in advance)
_G = {}


CACHE = "runs/_cache/base_data.npz"


def _init(datasets, cache=None):
    """Main process: load via torch. Pool workers: read the numpy cache only (no torch import, low memory)."""
    os.chdir(_common.REPO)
    if cache:
        z = np.load(cache)
        _G["x"], _G["sig"] = z["x"], z["sig"]
        _G["meta"] = [tuple(int(v) for v in m) for m in z["meta"]]
        _G["folds"] = {int(s): int(f) for s, f in z["folds"]}
    else:
        _G["x"], _G["meta"], _G["sig"], _G["folds"] = _common.load_data(datasets)


def _classical(task):
    import classical
    i, method = task
    ens, sg = _G["x"][i], float(_G["sig"][i])
    d, s = _G["meta"][i]
    sc = SliceScorer(ens)
    rows = []
    for a in A_EVAL:
        halves = sc.a_half(a)
        for p in classical.GRIDS[method]:
            est = np.stack([classical.denoise(method, h, sg, p) for h in halves])
            n_sig, n_band = sc.score(a, est)
            rows.append(dict(dataset=d, slice=s, fold=_G["folds"][s], sigma_hat=sg, method=method, seed=0,
                             param=p, a=a, neff_signal=n_sig, band3=n_band))
    return rows


def _networks(method, datasets, device):
    import torch
    from _r3_eval import load
    rows = []
    x, meta, sig, folds = _G["x"], _G["meta"], _G["sig"], _G["folds"]
    for seed, (run, loss) in NETS[method].items():
        for k in range(5):
            cfg = json.load(open(os.path.join(run, f"fold{k}", "config.json"), encoding="utf-8"))
            model = load(os.path.join(run, f"fold{k}", "ckpt", "%s_%06d.pkl" % (loss, ITER)), device)[0]
            if method.startswith("ddm2"):
                import ddm2
                m1 = ddm2.stage1(seed, k, device)
            for i, (d, s) in enumerate(meta):
                if folds[s] != k:
                    continue
                assert s not in cfg["train_slices"]
                ens = x[i]; sg = float(sig[i])
                with torch.no_grad():
                    yt = torch.from_numpy(ens).float()
                    if method.startswith("ddm2"):          # stage I -> II (sigma_m) -> III, per repetition
                        phi, sig_m = ddm2.stage12(m1, yt, device)
                        g = torch.Generator(device=device).manual_seed(20261102 + 1000 * d + s)
                        outs = []
                        for j in range(yt.shape[0]):
                            if method == "ddm2":
                                o = model.walk_to(yt[j], float(sig_m[j]), 0.0, steps=ddm2.CFG["WALK_STEPS"], eta=1.0,
                                                  sigma_floor=ddm2.CFG["SIG_LO"], generator=g, device=device)
                            else:
                                o = model.denoise_to(yt[j], float(sig_m[j]), target_ratio=0.0, device=device)
                            outs.append(o.cpu())
                        x0 = torch.cat(outs).double().numpy()
                    elif method == "r2r":                  # published inference: mean of f(y + alpha z) over draws
                        g = torch.Generator().manual_seed(20261030 + 1000 * d + s)
                        x0 = torch.stack([model.denoise_to(yt + R2R_ALPHA * sg * torch.randn(yt.shape, generator=g), sg,
                                                           target_ratio=0.0, device=device).cpu()
                                          for _ in range(R2R_DRAWS)]).mean(0).double().numpy()
                    else:
                        x0 = model.denoise_to(yt, sg, target_ratio=0.0, device=device).double().cpu().numpy()
                sc = SliceScorer(ens)
                for a in A_EVAL:
                    for r in RGRID:
                        out = x0 + r * (ens - x0)
                        est = np.stack([out[list(c)].mean(0) for c in _common.splits(a)])
                        n_sig, n_band = sc.score(a, est)
                        rows.append(dict(dataset=d, slice=s, fold=k, sigma_hat=sg, method=method, seed=seed,
                                         param=round(r, 4), a=a, neff_signal=n_sig, band3=n_band))
            print(f"{method} seed {seed} fold {k} done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--datasets", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    os.chdir(_common.REPO)
    if args.check:                       # all splits, dataset 1, ours seed 101 vs the logged evaluate_slice
        _common.K_SPLITS = 10 ** 6
        NETS["ours"] = {101: NETS["ours"][101]}
        _init([1])
        rows = _networks("ours", [1], args.device)
        ref = {(int(r["slice"]), int(r["a"])): r for r in csv.DictReader(open(
            "runs/cv_ablation/mse_norm/ratio_eval/per_slice_ratio.csv", encoding="utf-8")) if r["region"] == "signal"}
        worst = max(abs(r["neff_signal"] / float(ref[(r["slice"], r["a"])]["neff_r%.4f" % r["param"]]) - 1) for r in rows)
        print(f"check (all splits) vs evaluate_slice: worst relative difference {worst:.2e}")
        assert worst < 1e-5     # float32 decode on CPU vs the logged GPU decode: ~1e-6
        return
    assert not os.path.exists(os.path.join(args.out, "neff.csv")), "refusing to overwrite " + args.out
    os.makedirs(args.out, exist_ok=True)
    _init(args.datasets)
    t0, rows = time.time(), []
    for method in [m for m in args.methods if m in NETS]:
        rows += _networks(method, args.datasets, args.device)
    cls = [m for m in args.methods if m not in NETS]
    if cls:
        from multiprocessing import Pool
        tasks = [(i, m) for m in cls for i in range(len(_G["meta"]))]
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez(CACHE, x=_G["x"], sig=_G["sig"], meta=np.array(_G["meta"]),
                 folds=np.array(sorted(_G["folds"].items())))
        with Pool(args.workers, initializer=_init, initargs=(args.datasets, CACHE)) as pool:
            for n, rr in enumerate(pool.imap_unordered(_classical, tasks, chunksize=1), 1):
                rows += rr
                if n % 25 == 0 or n == len(tasks):
                    print(f"{n}/{len(tasks)} slice-method tasks, {(time.time() - t0) / 60:.1f} min", flush=True)
    with open(os.path.join(args.out, "neff.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(vars(args), k_splits=_common.K_SPLITS, a_eval=list(A_EVAL), rgrid=list(RGRID)),
              open(os.path.join(args.out, "eval_config.json"), "w"), indent=2)
    print("wrote", args.out, len(rows), "rows", f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
