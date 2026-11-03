# -*- coding: utf-8 -*-
"""DEC evaluation (``Archive/DEC/PREREGISTRATION.md``): score any decoder with the split-half n_eff estimator.

For every held-out slice and every decoder x r on the grid, the 10 repetitions are decoded independently and
scored over all C(10, a) splits against the complementary-half mean (unbiased), signal and all regions,
a in {1, 2, 3, 5, 7, 9}; per-band n_eff (four radial bands, as ``_ratio_core``) at a in {1, 5, 9}.

Decoders (``N2N_DSM.walk_to``; sigma_floor = ladder bottom unless stated):
  tw     one-shot Tweedie / one DDIM step to r * sigma_hat
  ode    20-step deterministic DDIM walk (eta = 0)
  anc    20-step ancestral walk (eta = 1): one posterior sample per repetition
  anc8   mean of 8 ancestral samples per repetition
  ancx   ancestral walk extrapolated below the trained floor to sigma_hat / 10 (off-distribution)

``--check CSV`` (with ``--decoders tw``): the tw n_eff must reproduce an ``evaluate_slice`` CSV of the same run.
Refuses to overwrite.

    python experiments/_dec_eval.py --run runs/cv_ablation/mse_norm --iter 2000 --loss mse --decoders tw ode anc anc8 ancx --out runs/dec/B/s101
"""
import sys, os, csv, json, argparse, itertools
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, DIFF); os.chdir(DIFF)
sys.path.insert(0, os.path.join(DIFF, "experiments"))
from _r3_eval import load                                    # noqa: E402
from _data import load_datasets, cv_folds                    # noqa: E402
from _ratio_core import A_SIZES, BAND_A, BANDS, cmag         # noqa: E402

RGRID = (0.0, 1 / 12, 1 / 6, 0.25, 1 / 3, 0.5)
STEPS = 20


def decode(model, ens_t, sg, dec, r, device, seed):
    if dec == "tw":
        return model.denoise_to(ens_t, sg, target_ratio=r, device=device)
    g = torch.Generator(device=device); g.manual_seed(seed)
    if dec == "ode":
        return model.walk_to(ens_t, sg, r, steps=STEPS, eta=0.0, device=device)
    if dec == "anc":
        return model.walk_to(ens_t, sg, r, steps=STEPS, eta=1.0, generator=g, device=device)
    if dec == "ancx":
        return model.walk_to(ens_t, sg, r, steps=STEPS, eta=1.0, sigma_floor=sg / 10, generator=g, device=device)
    if dec == "anc8":
        N = ens_t.shape[0]
        out = model.walk_to(ens_t.repeat(8, 1, 1, 1), sg, r, steps=STEPS, eta=1.0, generator=g, device=device)
        return out.view(8, N, *out.shape[1:]).mean(0)
    raise ValueError(dec)


def score(ens, outs, masks, s2, pn, f_r):
    """{(a, key): n_eff} for one decoded stack; key in regions or band0..3 (bands at BAND_A)."""
    N, _, H, W = ens.shape
    res = {}
    for a in A_SIZES:
        combos = list(itertools.combinations(range(N), a))
        WA = np.zeros((len(combos), N)); WB = np.zeros((len(combos), N))
        for i, c in enumerate(combos):
            WA[i, list(c)] = 1.0 / a
            WB[i, [j for j in range(N) if j not in c]] = 1.0 / (N - a)
        ref = (WB @ ens.reshape(N, -1)).reshape(-1, 2, H, W)
        est = (WA @ outs.reshape(N, -1)).reshape(-1, 2, H, W)
        d2 = ((est - ref) ** 2).mean(1)
        for rg, v in masks.items():
            mse = float((d2[:, v].mean(1) - s2[rg] / (N - a)).mean())
            res[(a, rg)] = s2[rg] / mse if mse > 0 else float("inf")
        if a in BAND_A:
            sp = (np.abs(np.fft.fft2(est - ref)) ** 2).mean(axis=(0, 1))
            for b, (lo, hi) in enumerate(BANDS):
                mk = (f_r >= lo) & (f_r < hi)
                err = sp[mk].mean() - pn[mk].mean() / (N - a)
                res[(a, f"band{b}")] = pn[mk].mean() / err if err > 0 else float("inf")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--iter", type=int, default=2000)
    ap.add_argument("--loss", default="mse")
    ap.add_argument("--datasets", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--decoders", nargs="+", default=["tw"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default=None)
    ap.add_argument("--sigma-scale", type=float, default=1.0, help="feed c * sigma_hat as the noise label (label sweep)")
    a_ = ap.parse_args()
    assert not os.path.exists(os.path.join(a_.out, "neff.csv")), "refusing to overwrite " + a_.out
    os.makedirs(a_.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, meta, sig = load_datasets(a_.datasets)
    folds = cv_folds()
    rows = []
    for k in range(5):
        cfg = json.load(open(os.path.join(a_.run, f"fold{k}", "config.json"), encoding="utf-8"))
        assert sorted(cfg["held_out"]) == sorted(s for s, f in folds.items() if f == k), "fold mismatch"
        model = load(os.path.join(a_.run, f"fold{k}", "ckpt", "%s_%06d.pkl" % (a_.loss, a_.iter)), device)[0]
        floor = float(model.dsm.sigmas[0])
        for i, (d, s) in enumerate(meta):
            if folds[s] != k:
                continue
            assert s not in cfg["train_slices"], "model trained on this slice index"
            ens_t = x[i].to(device); sg = float(sig[i]) * a_.sigma_scale
            ens = ens_t.double().cpu().numpy()
            N, _, H, W = ens.shape
            mean10 = ens.mean(0); dev_ = ens - mean10
            vpix = ((dev_ ** 2).sum(0) / (N - 1)).mean(0)
            pn = ((np.abs(np.fft.fft2(dev_)) ** 2).sum(0) / (N - 1)).mean(0)
            sm = gaussian_filter(cmag(mean10), 3.0)
            masks = {"signal": sm > np.median(sm), "all": np.ones((H, W), bool)}
            s2 = {rg: float(vpix[v].mean()) for rg, v in masks.items()}
            f_r = np.hypot(np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :])
            for dec in a_.decoders:
                for r in RGRID:
                    with torch.no_grad():
                        outs = decode(model, ens_t, sg, dec, r, device,
                                      seed=20261029 + 100000 * d + 1000 * s + int(round(r * 120))).double().cpu().numpy()
                    res = score(ens, outs, masks, s2, pn, f_r)
                    for a in A_SIZES:
                        row = dict(dataset=d, slice=s, fold=k, sigma_hat=float(sig[i]), sigma_scale=a_.sigma_scale, walk=sg / floor, decoder=dec,
                                   r=round(r, 4), a=a, neff_signal=res[(a, "signal")], neff_all=res[(a, "all")])
                        for b in range(4):
                            row[f"band{b}"] = res.get((a, f"band{b}"), "")
                        rows.append(row)
            print(f"ds{d} slice {s:2d} fold {k} sigma {sg:.4f} walk {sg / floor:.2f}", flush=True)
        del model; torch.cuda.empty_cache()
    if a_.check:
        ref = {}
        for r in csv.DictReader(open(a_.check, encoding="utf-8")):
            if r["region"] in ("signal", "all"):
                ref[(int(r.get("dataset", 1)), int(r["slice"]), int(r["a"]), r["region"])] = r
        worst = 0.0
        for row in rows:
            if row["decoder"] != "tw" or row["dataset"] != 1:
                continue
            for rg in ("signal", "all"):
                q = ref[(1, row["slice"], row["a"], rg)]
                worst = max(worst, abs(row[f"neff_{rg}"] / float(q["neff_r%.4f" % row["r"]]) - 1))
        print(f"check vs {a_.check}: worst relative difference {worst:.2e}")
        assert worst < 1e-6, "tw does not reproduce evaluate_slice"
    with open(os.path.join(a_.out, "neff.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(vars(a_), steps=STEPS, rgrid=list(RGRID)), open(os.path.join(a_.out, "eval_config.json"), "w"), indent=2)
    print("wrote", a_.out, len(rows), "rows")


if __name__ == "__main__":
    main()
