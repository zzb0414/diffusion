# -*- coding: utf-8 -*-
"""Does the rung label mean ADDED noise or TOTAL noise?

Training (``N2N_DSM.loss_multireal``) perturbs an already-noisy realization y
(noise sigma_task) by sigma_i and labels the sample with rung i, so the net at
rung i has only ever seen inputs with TOTAL noise sqrt(sigma_task^2 + sigma_i^2).
Inference (``single_map_denoise`` rung_mult=1, ``conditional_denoise``) feeds a
map whose total noise is sigma_task at the rung sigma_i ~ sigma_task, i.e. it
reads the label as TOTAL noise.

If the label really means ADDED noise, a raw y belongs at the LOWEST rung, and
the full Tweedie estimate there should keep contrast (gain near 1) where the
rung_mult=1 estimate collapses it. A one-step DDIM from sigma_task down to
sigma_task/3 is out = y + (2/3)(x0_hat - y); that is scored too.

    python experiments/_rung_label_check.py --target 1800
"""
import sys, os, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, DIFF); os.chdir(DIFF)

from _r1_eval import load_data, mag                          # noqa: E402
from _r2_displays import nearest_ckpt, load_model            # noqa: E402
from _headtohead import Scorer                               # noqa: E402

SAMPLES = [("train", 17), ("val", 16), ("test", 25)]


@torch.no_grad()
def tweedie_at(n2n, ens, i, device):
    s = n2n.sigmas[i].item()
    ts = torch.full((ens.shape[0],), i, device=device, dtype=torch.long)
    return ens + s ** 2 * n2n.net(ens, ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="l1")
    ap.add_argument("--target", type=int, default=1800)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = load_data()
    it, path = nearest_ckpt(a.loss, a.target)
    n2n = load_model(path, device)
    sg = n2n.sigmas
    print(f"model {a.loss}@{it}  ladder L={len(sg)} "
          f"sigma {sg.min().item():.4f}..{sg.max().item():.4f}")
    print(f"{'split':6s}{'sl':>4s}{'sig_task':>9s}{'rung':>6s}{'sig_i':>8s}"
          f"{'blend':>6s}{'noise':>7s}{'gain':>7s}{'sHF%':>7s}{'relE':>7s}")
    for split, s in SAMPLES:
        ens = x[s].to(device)
        N = ens.shape[0]
        mc = ens.mean(0)
        st = float((ens - mc).square().mean().sqrt())
        loo0 = (ens.sum(0) - ens[0]) / (N - 1)
        sc = Scorer(ens, st, loo0, mag(mc).cpu().numpy())
        i_task = int(torch.argmin((sg - st).abs()).item())
        for tag, i in (("low", 0), ("task", i_task)):
            X0 = tweedie_at(n2n, ens, i, device)
            for b in (2.0 / 3.0, 1.0):
                m = sc((ens + b * (X0 - ens)).detach())
                print(f"{split:6s}{s:>4d}{st:>9.4f}{tag:>6s}{sg[i].item():>8.4f}"
                      f"{b:>6.3f}{m['noise']:>7.3f}{m['gain']:>7.3f}"
                      f"{m['shf']:>7.1f}{m['relE']:>7.3f}", flush=True)
        # matched-noise baseline: identity at the same scorer
        m = sc(ens)
        print(f"{split:6s}{s:>4d}{st:>9.4f}{'id':>6s}{'-':>8s}{0:>6.3f}"
              f"{m['noise']:>7.3f}{m['gain']:>7.3f}{m['shf']:>7.1f}{m['relE']:>7.3f}")


if __name__ == "__main__":
    main()
