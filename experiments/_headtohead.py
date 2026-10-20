# -*- coding: utf-8 -*-
"""Head-to-head at MATCHED residual noise: learned decode vs dumb baselines.

One number decides whether the network is earning its keep. For every estimator
family we trace its curve (achieved residual noise -> agreement with the
reference) by sweeping its single knob, then read off:

    @1/3     the scored metrics at noise == sigma_ref/sigma_task = 1/3, i.e.
             the noise level the deployment target requires
    best<=1/3  the lowest single-map complex relative error achievable WITHOUT
             exceeding the reference noise floor (lower is better)
    minnoise   the floor of the family (how quiet it can become)

Families (all applied per realization; noise measured as the spread between
independently processed realizations, calibrating against identity):
    id       return the input untouched
    blur     separable Gaussian on Re/Im
    wiener   oracle stationary Wiener filter (optimistic ceiling)
    tweedie  the net's one-shot Tweedie decode, out = y + b*(x0_hat - y),
             swept over rungs -- this is what ``single_map_denoise`` deploys

Reported metrics are post-global-gain (``s_bias``/``s_HF%``) because a global
contrast fix is trivially available downstream; the unscaled pair is kept for
transparency. ``relE`` always uses the RAW output against the LOO mean.

    python _headtohead.py
"""
import sys, os, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, DIFF); os.chdir(DIFF)

from _r1_eval import load_data, mag, hf                      # noqa: E402
from _r2_displays import nearest_ckpt, load_model, stats, cstats  # noqa: E402

SAMPLES = [("train", 17), ("val", 16), ("test", 25)]
TARGET = 1.0 / 3.0


def fftc(a):
    return np.fft.fftshift(np.fft.fft2(a))


def ifftc(a):
    return np.fft.ifft2(np.fft.ifftshift(a)).real


def radial_smooth(P):
    H, W = P.shape
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.round(np.hypot(yy - H // 2, xx - W // 2)).astype(int)
    out = np.empty_like(P)
    for r in range(rr.max() + 1):
        m = rr == r
        out[m] = P[m].mean() if m.any() else 0.0
    out[~np.isfinite(out)] = 0.0
    return out


def blur_stack(a, sig):
    """Gaussian blur of (N,2,H,W)."""
    rad = max(1, int(np.ceil(3 * sig)))
    ax = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (ax / sig) ** 2)
    k /= k.sum()
    out = np.pad(a, ((0, 0), (0, 0), (rad, rad), (0, 0)), mode="reflect")
    out = sum(k[i] * out[..., i:i + a.shape[-2], :] for i in range(2 * rad + 1))
    out = np.pad(out, ((0, 0), (0, 0), (0, 0), (rad, rad)), mode="reflect")
    out = sum(k[i] * out[..., :, i:i + a.shape[-1]] for i in range(2 * rad + 1))
    return out


@torch.no_grad()
def decode(n2n, ens, st, mult, bs, device):
    """Cache the full Tweedie estimate once, emit partial blends for all bs."""
    N = ens.shape[0]
    X0 = torch.stack([n2n.single_map_denoise(
        ens[k], sigma_task=st, blend=1.0, rung_mult=mult, device=device)[0]
        for k in range(N)])
    D = X0 - ens
    return {b: ens + b * D for b in bs}


class Scorer:
    def __init__(self, ens, st, loo0, refm):
        self.ens, self.st, self.loo0, self.refm = ens, st, loo0, refm
        self.N = ens.shape[0]

    def __call__(self, outs):
        N, st = self.N, self.st
        om = outs.mean(0)
        sd = outs - om
        noise = float((sd.square().mean() * N / (N - 1)).sqrt()) / st
        em = mag(om).cpu().numpy()
        b, rm = stats(em, self.refm)
        g = float(np.sum(em * self.refm) /
                  (np.sum(self.refm * self.refm) + 1e-12))
        es = em / (g + 1e-12)
        sb, srm = stats(es, self.refm)
        cs = cstats(outs[0].detach().cpu(), self.loo0.detach().cpu())
        return dict(noise=noise, bias=b, shf=100 * hf(es) / hf(self.refm),
                    hf=100 * hf(em) / hf(self.refm),
                    gain=g, sb=sb, relE=cs["rel_err"],
                    pha=cs["pha_med_deg"])


def interp_at(curve, target):
    """Linear interpolation of a metric dict along noise; None if not bracketed."""
    pts = sorted((c["noise"], c) for c in curve)
    for (n0, c0), (n1, c1) in zip(pts, pts[1:]):
        if n0 <= target <= n1 and n1 > n0:
            t = (target - n0) / (n1 - n0)
            return {k: c0[k] + t * (c1[k] - c0[k]) if isinstance(c0[k], float)
                    else c0[k] for k in c0}
    return None          # family cannot BE this quiet (or this noisy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--losses", nargs="+", default=["l1"])
    ap.add_argument("--targets", type=int, nargs="+", default=[1860])
    ap.add_argument("--out", default="runs/displays/target_sweep/headtohead.csv")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = load_data()

    curves_csv = ["split,slice,target_iter,loss,ckpt_iter,family,param,rung,"
                  "noise,bias,hf_pct,shf_pct,gain,s_bias,relE,pha_deg\n"]
    summary = []
    print(f"{'split':6s}{'sl':>4s} {'family':8s}{'par':>7s}"
          f"{'noise@1/3':>10s}{'shf':>7s}{'sbias':>7s}{'relE':>7s}"
          f"{'pha':>6s}{'minNoise':>9s}")
    for tgt in a.targets:
        for loss in a.losses:
            it, path = nearest_ckpt(loss, tgt)
            n2n = load_model(path, device)
            print(f"\n===== {loss}@{it} =====", flush=True)
            for split, s in SAMPLES:
                ens = x[s].to(device)
                N = ens.shape[0]
                mc = ens.mean(0)
                st = float((ens - mc).square().mean().sqrt())
                refm = mag(mc).cpu().numpy()
                loo0 = ((ens.sum(0) - ens[0]) / (N - 1))
                sc = Scorer(ens, st, loo0, refm)

                fams = {}
                fams["id"] = [dict(param=0.0, **sc(ens))]
                fams["blur"] = [
                    dict(param=sg, **sc(torch.from_numpy(blur_stack(
                        ens.cpu().numpy(), sg)).to(ens.dtype)))
                    for sg in (0.2, 0.35, 0.5, 0.625, 0.76, 0.887, 1.025,
                               1.187, 1.365, 1.56, 1.78, 2.13)]
                # --- oracle wiener -------------------------------------------------
                en = ens.cpu().numpy()
                mcn = mc.cpu().numpy()
                dev = en - mcn
                Pn = np.mean([np.mean([np.abs(fftc(dev[k, ch])) ** 2
                                       for k in range(N)], axis=0)
                              for ch in (0, 1)], axis=0) * N / (N - 1)
                Ps = np.mean([np.abs(fftc(mcn[ch])) ** 2
                              for ch in (0, 1)], axis=0)
                Pnr, Psr = radial_smooth(Pn), radial_smooth(Ps)
                Pc = np.maximum(Psr - Pnr / N, 0.0)
                wcurve = []
                for fac in (0.010, 0.018, 0.032, 0.055, 0.090, 0.140, 0.220,
                            0.35, 0.55, 0.85, 1.30, 2.00, 3.20):
                    G = Pc / (Pc + fac * Pnr + 1e-30)
                    o = np.stack([
                        np.stack([ifftc(fftc(en[k, ch]) * G) for ch in (0, 1)])
                        for k in range(N)])
                    wcurve.append(dict(param=fac,
                                       **sc(torch.from_numpy(o).float())))
                fams["wiener"] = wcurve
                # --- net single Tweedie -------------------------------------------
                tcurve = []
                mults = (1.000, 1.323, 1.700, 2.020, 2.400, 2.690)
                for mult in mults:
                    dct = decode(n2n, ens, st, mult,
                                 [0.227, 0.364, 0.500, 0.636, 0.773, 0.909],
                                 device)
                    for b, outs in dct.items():
                        tcurve.append(dict(param=b, rung=mult,
                                           **sc(outs.detach())))
                    del dct
                fams["tweedie"] = tcurve

                for fam, curve in fams.items():
                    at = interp_at(curve, TARGET)
                    ok = [c for c in curve if c["noise"] <= TARGET]
                    best = min(ok, key=lambda c: c["relE"]) if ok else \
                        min(curve, key=lambda c: c["relE"])
                    minn = min(c["noise"] for c in curve)
                    for c in curve:
                        curves_csv.append(
                            f"{split},{s},{tgt},{loss},{it},{fam},"
                            f"{c['param']:.3f},{c.get('rung', 0.0):.3f},"
                            f"{c['noise']:.4f},{c['bias']:.4f},"
                            f"{c['hf']:.1f},{c['shf']:.1f},{c['gain']:.4f},"
                            f"{c['sb']:.4f},{c['relE']:.4f},{c['pha']:.1f}\n")
                    if at is None:
                        summary.append((split, s, fam, "OUT-OF-RANGE",
                                        minn, best))
                        print(f"{split:6s}{s:>4d} {fam:8s}   OUT-OF-RANGE "
                              f"(minnoise {minn:.3f})")
                        continue
                    summary.append((split, s, fam, at, minn, best))
                    print(f"{split:6s}{s:>4d} {fam:8s}{at['param']:>7.2f}"
                          f"{at['noise']:>10.3f}{at['shf']:>7.1f}"
                          f"{at['sb']:>7.3f}{at['relE']:>7.3f}"
                          f"{at['pha']:>6.1f}{minn:>9.3f}", flush=True)
                del ens, loo0
                torch.cuda.empty_cache()
            del n2n
            torch.cuda.empty_cache()

    open(a.out, "w", encoding="utf-8").write("".join(curves_csv))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
