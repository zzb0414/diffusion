# -*- coding: utf-8 -*-
"""DDM2-like baseline (Archive/BASE/PREREGISTRATION_DDM2.md): stage-III training and the shared stage I / II helpers.

DDM2 (Xiang et al., ICLR 2023) = (I) self-supervised CNN first estimate, (II) match the input's noise to a step of
the diffusion chain, (III) diffusion model trained on stage-I outputs as clean data, reverse-sampled from the matched
step. Here:
  stage I   the classic Noise2Noise model of the same fold and seed (runs/dec/A/blind_s<seed>; no sigma input)
  stage II  sigma_m = std of the residual y - Phi_I(y), per repetition
  stage III score network (score parametrization, levels SIG_LO..0.4) trained by denoising score matching on
            Phi_I(y) of the training slices: input x + sigma eps, target x, weight (sigma_ref/sigma)^2 (eps-prediction)

    python baselines/ddm2.py --fold 0 --seed 101 --outdir runs/base/ddm2/s101
"""
import os, time, json, math, argparse, csv
import numpy as np
import torch
import _common

from unet import UNet                                        # noqa: E402
from n2n import N2N_DSM                                      # noqa: E402

CFG = dict(L=40, SIG_LO=0.005, SIGMA_MAX=0.4, ITERS=2000, BATCH=8, LR=1e-3, GRAD_CLIP=1.0, NET_BASE=32, NET_EMB=32,
           CKPT_EVERY=500, WALK_STEPS=20)


def stage1(seed, fold, device):
    from _r3_eval import load
    return load(os.path.join(_common.REPO, "runs", "dec", "A", f"blind_s{seed}", f"fold{fold}", "ckpt", "mse_002000.pkl"), device)[0]


@torch.no_grad()
def stage12(m1, y, device):
    """y: (N, 2, H, W) repetitions -> (Phi_I(y), sigma_m per repetition)."""
    phi = m1.denoise_to(y, 1.0, target_ratio=0.0, device=device)            # blind model: sigma argument unused
    sig_m = (y.to(device) - phi).flatten(1).std(1)
    return phi, sig_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--iters", type=int, default=CFG["ITERS"])
    args = ap.parse_args()
    os.chdir(_common.REPO)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RUN = os.path.join(args.outdir, f"fold{args.fold}")
    assert not os.path.exists(os.path.join(RUN, "train.csv")), "refusing to overwrite " + RUN
    os.makedirs(os.path.join(RUN, "ckpt"), exist_ok=True)
    torch.manual_seed(args.seed)
    from _data import load_datasets, cv_folds
    x, meta, sig = load_datasets([1])
    folds = cv_folds()
    held = sorted(s for s, k in folds.items() if k == args.fold)
    train_idx = sorted(s for s, k in folds.items() if k != args.fold)
    tr = [i for i, (d, s) in enumerate(meta) if s in train_idx]
    m1 = stage1(args.seed, args.fold, dev)
    pseudo = torch.stack([stage12(m1, x[i], dev)[0] for i in tr])            # (S, N, 2, H, W) pseudo-clean (stage I)
    del m1
    N = x.shape[1]
    net = UNet(in_ch=2, out_ch=2, base_out=CFG["NET_BASE"], emb_dim=CFG["NET_EMB"]).to(dev)
    m = N2N_DSM(L=CFG["L"], sigma_min=CFG["SIG_LO"], sigma_max=CFG["SIGMA_MAX"], backbone=net)
    opt = torch.optim.Adam(net.parameters(), lr=CFG["LR"])
    sigma_ref = float(torch.exp(torch.log(sig[tr]).mean()))
    ckpt_cfg = dict(net_type="N2N_DSM", noise_label="total", objective="ddm2_stage3", N=N, L=CFG["L"],
                    sigma_min=CFG["SIG_LO"], sigma_max=CFG["SIGMA_MAX"], base_out=CFG["NET_BASE"], emb_dim=CFG["NET_EMB"],
                    loss="ddm2", fold=args.fold, held_out=held, seed=args.seed, stage1=f"runs/dec/A/blind_s{args.seed}")
    json.dump(dict(cfg=CFG, ckpt_cfg=ckpt_cfg, held_out=held, train_slices=train_idx, seed=args.seed),
              open(os.path.join(RUN, "config.json"), "w"), indent=2)
    log = open(os.path.join(RUN, "train.log"), "a", encoding="utf-8")
    with open(os.path.join(RUN, "train.csv"), "w", newline="") as f:
        csv.writer(f).writerow(["iter", "loss_avg50", "sec"])
    lo, hi = math.log(CFG["SIG_LO"]), math.log(CFG["SIGMA_MAX"])
    recent, t0 = [], time.time()
    for it in range(1, args.iters + 1):
        pick = torch.randint(0, len(tr), (CFG["BATCH"],), device=dev)
        xc = pseudo[pick].reshape(-1, 2, *x.shape[-2:])
        s = torch.exp(lo + torch.rand(xc.shape[0], device=dev) * (hi - lo))
        xn = xc + s.view(-1, 1, 1, 1) * torch.randn_like(xc)
        x0 = m._x0(xn, s)
        w = (sigma_ref / s) ** 2
        loss = (w * ((x0 - xc) ** 2).mean(dim=(1, 2, 3))).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CFG["GRAD_CLIP"]); opt.step()
        recent = (recent + [loss.item()])[-50:]
        if it % 50 == 0:
            for attempt in range(5):
                try:
                    with open(os.path.join(RUN, "train.csv"), "a", newline="") as f:
                        csv.writer(f).writerow([it, float(np.mean(recent)), time.time() - t0])
                    break
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(2.0)
        if it % CFG["CKPT_EVERY"] == 0 or it == args.iters:
            line = f"[{time.strftime('%H:%M:%S')}] iter {it} loss={np.mean(recent):.6f}"
            print(line, flush=True); log.write(line + "\n"); log.flush()
            torch.save({"state_dict": net.state_dict(), "cfg": ckpt_cfg, "train_iters": it},
                       os.path.join(RUN, "ckpt", f"ddm2_{it:06d}.pkl"))
    log.write(f"DONE fold {args.fold}: {args.iters} iterations in {(time.time() - t0) / 60:.1f} min\n"); log.close()


if __name__ == "__main__":
    main()
