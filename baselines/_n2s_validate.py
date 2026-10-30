# -*- coding: utf-8 -*-
"""Implementation check for the Noise2Score baseline on synthetic data with known ground truth.

Clean images: dataset-1 slices, mean of 10 repetitions, Gaussian-smoothed (sigma 1.5 px) -- structured, nearly clean.
Noisy images: clean + fresh Gaussian noise at a fixed SNR (signal-region RMS / sigma). Noise2Score (version 2 of
``noise2score.py``: AR-DAE, conditioned on total noise) is trained on fold-0 training slices with fresh noise per
batch and evaluated on the held-out slices: MSE of the Tweedie estimate vs the clean image, as a fraction of the
noise variance (< 1 means it denoises). A correct implementation must denoise clearly at SNR ~ 1 (the published
regime). For reference, the same network trained with Noise2Noise pairs (two independent noisy copies) under the
same budget is scored too.

    python baselines/_n2s_validate.py --snr 1.0
"""
import os, sys, math, argparse
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import _common

from unet import UNet                                        # noqa: E402
from n2n import N2N_DSM                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--fixed-reps", type=int, default=0, help="train on a FIXED set of this many noisy copies per slice (as real data)")
    ap.add_argument("--colored", action="store_true", help="noise filtered by dataset 1's median radial spectrum (real colour)")
    args = ap.parse_args()
    os.chdir(_common.REPO)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    x, meta, sig, folds = _common.load_data([1])
    clean = np.stack([np.stack([gaussian_filter(c, 1.5) for c in e.mean(0)]) for e in x])     # (34, 2, H, W)
    mag = np.sqrt((clean ** 2).sum(1))
    sm = np.stack([gaussian_filter(m, 3.0) for m in mag])
    rms = float(np.sqrt(np.mean([(clean[i] ** 2).sum(0)[sm[i] > np.median(sm[i])].mean() / 2 for i in range(34)])))
    sigma = rms / args.snr
    tr = [s for s in range(34) if folds[s] != 0]; te = [s for s in range(34) if folds[s] == 0]
    C = torch.from_numpy(clean).float().to(dev)
    filt = None
    if args.colored:                       # unit-mean-power radial filter from dataset 1's measured noise spectrum
        from _degrade import noise_profile, _fr
        profs = [noise_profile(x[i]) for i in range(34)]
        cen, prof = profs[0][0], np.median(np.stack([p for _, p in profs]), 0)
        P = np.interp(_fr(80, 80), cen, prof)
        filt = torch.from_numpy(np.sqrt(P / P.mean())).float().to(dev)

    def noise(shape, gen=None):
        w = torch.randn(shape, device=dev, generator=gen)
        return w if filt is None else torch.fft.ifft2(torch.fft.fft2(w) * filt).real
    FIX = None
    if args.fixed_reps:                    # the same noisy copies every epoch, like the real 10 repetitions
        gfix = torch.Generator(device=dev); gfix.manual_seed(7)
        FIX = C.unsqueeze(1) + sigma * noise((34, args.fixed_reps, 2, 80, 80), gfix)
    out = {}
    for mode in ("n2s", "n2n"):
        net = UNet(in_ch=2, out_ch=2, base_out=32, emb_dim=32).to(dev)
        m = N2N_DSM(L=40, sigma_min=sigma / 2, sigma_max=sigma * 4, backbone=net)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for it in range(args.iters):
            idx = torch.tensor(np.random.choice(tr, 32), device=dev)
            if FIX is None:
                y = C[idx] + sigma * noise(C[idx].shape)
            else:
                ks = torch.randint(0, args.fixed_reps, (2, 32), device=dev)
                y = FIX[idx, ks[0]]
            s = torch.full((len(idx),), sigma, device=dev)
            if mode == "n2s":
                d = sigma * torch.exp(math.log(0.05) + torch.rand_like(s) * math.log(2.0 / 0.05))
                u = torch.randn_like(y)
                score = net(y + d.view(-1, 1, 1, 1) * u, m.level_coord(torch.sqrt(s ** 2 + d ** 2)))
                loss = ((u + d.view(-1, 1, 1, 1) * score) ** 2).mean()
            else:                                                   # Noise2Noise pair, same parametrization
                y2 = C[idx] + sigma * noise(C[idx].shape) if FIX is None else FIX[idx, torch.where(ks[1] == ks[0], (ks[1] + 1) % args.fixed_reps, ks[1])]
                loss = ((m._x0(y, s) - y2) ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        g = torch.Generator(device=dev); g.manual_seed(1)
        errs = []
        with torch.no_grad():
            for s_ in te:
                c = C[s_:s_ + 1].repeat(16, 1, 1, 1)
                y = c + sigma * noise(c.shape, g)
                x0 = m._x0(y, torch.full((16,), sigma, device=dev))
                errs.append(float(((x0 - c) ** 2).mean() / sigma ** 2))
        out[mode] = float(np.mean(errs))
    print(f"SNR {args.snr} fixed_reps={args.fixed_reps} colored={args.colored}: sigma {sigma:.4f}; held-out MSE / sigma^2 (noisy input = 1.0): "
          f"Noise2Score {out['n2s']:.3f}, Noise2Noise {out['n2n']:.3f}")


if __name__ == "__main__":
    main()
