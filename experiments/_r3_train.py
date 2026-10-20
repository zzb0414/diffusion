# -*- coding: utf-8 -*-
"""R3: retrain N2N_DSM with TOTAL-noise labels (the noise-label fix).

Same split / backbone / LOO objective as R0, with three changes:
  * each slice's acquisition noise sigma_hat (per Re/Im channel, unbiased over
    its N realizations) is estimated ONCE here and passed to the loss as sigma_y;
  * the ladder starts at the smallest TRAINING sigma_hat (no rung is cleaner
    than any data the net can see) and keeps sigma_max = 0.40;
  * no level weighting -- that band weighting calibrated the old labels.

Validation runs at the DEPLOYMENT condition: raw val realizations, net
evaluated at their own sigma_hat, nothing injected. Reported: L1/MSE of the full
Tweedie estimate vs the LOO mean (the training objective at deployment), and
the achieved residual noise / gain of ``denoise_to(target_ratio=1/3)``.

    python experiments/_r3_train.py --loss l1 --iters 2000
"""
import sys, os, time, json, argparse, csv
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, DIFF)
os.chdir(DIFF)

from unet import UNet
from n2n import N2N_DSM
from train import DiffusionTrainer

VAL_SLICES = [16, 11, 1]
TEST_SLICES = [25, 22, 8]

CFG = dict(
    L=40, SIGMA_MAX=0.4, P_RAW=0.5,
    BATCH=8, LR=1e-3, GRAD_CLIP=1.0,
    NET_BASE=32, NET_EMB=32,
)


def load_data():
    img = np.load("../asl_for_diffusion/dataset1.npy")     # (Nx,Ny,Ns,Navg)
    t = torch.from_numpy(img).permute(2, 3, 0, 1).contiguous()   # (Ns,Navg,Nx,Ny)
    t = t / t.abs().amax(dim=(1, 2, 3), keepdim=True)
    return torch.stack([t.real, t.imag], dim=2).contiguous()     # (Ns,Navg,2,Nx,Ny)


def sigma_hat(x):
    """(Ns,) acquisition noise per Re/Im channel, unbiased over realizations."""
    N = x.shape[1]
    r = x.double() - x.double().mean(1, keepdim=True)
    return (r.square().sum(1) / (N - 1)).mean(dim=(1, 2, 3)).sqrt().float()


def pack(x, slices, device):
    sel = x[slices].to(device)                       # (S,Navg,2,Nx,Ny)
    s, n, c, nx, ny = sel.shape
    return sel.reshape(s, n * c, nx, ny)             # [Re1,Im1,Re2,Im2,...]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="l1", choices=["l1", "mse", "huber", "mag_l1"])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--outdir", default="runs/r3_total")
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RUN = os.path.join(args.outdir, args.loss)
    os.makedirs(os.path.join(RUN, "ckpt"), exist_ok=True)
    log_path = os.path.join(RUN, "train.log")
    csv_path = os.path.join(RUN, "train_val.csv")
    open(log_path, "w").close()

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(0)

    x = load_data()
    Ns, N = x.shape[0], x.shape[1]
    sig = sigma_hat(x)
    train_slices = [s for s in range(Ns) if s not in VAL_SLICES + TEST_SLICES]
    sigma_min = float(sig[train_slices].min())
    for s in VAL_SLICES + TEST_SLICES:
        assert sigma_min <= sig[s] < CFG["SIGMA_MAX"], (s, float(sig[s]))

    x_train = pack(x, train_slices, DEVICE)
    sy_train = sig[train_slices].to(DEVICE)
    Nx = x.shape[-1]
    log(f"loss={args.loss} device={DEVICE} train={len(train_slices)} "
        f"val={VAL_SLICES} test={TEST_SLICES}")
    log(f"ladder L={CFG['L']} sigma {sigma_min:.4f}..{CFG['SIGMA_MAX']:.4f} "
        f"(min = smallest train sigma_hat) p_raw={CFG['P_RAW']}")

    net = UNet(in_ch=2, out_ch=2, base_out=CFG["NET_BASE"],
               emb_dim=CFG["NET_EMB"]).to(DEVICE)
    n2n = N2N_DSM(L=CFG["L"], sigma_min=sigma_min, sigma_max=CFG["SIGMA_MAX"],
                  backbone=net)
    trainer = DiffusionTrainer(n2n, lr=CFG["LR"], device=DEVICE,
                               grad_clip=CFG["GRAD_CLIP"])

    ckpt_cfg = dict(net_type="N2N_DSM", noise_label="total", N=N, L=CFG["L"],
                    sigma_min=sigma_min, sigma_max=CFG["SIGMA_MAX"],
                    base_out=CFG["NET_BASE"], emb_dim=CFG["NET_EMB"],
                    loss=args.loss, p_raw=CFG["P_RAW"])
    with open(os.path.join(RUN, "config.json"), "w") as f:
        json.dump(dict(cfg=CFG, ckpt_cfg=ckpt_cfg, iters=args.iters, seed=args.seed,
                       val_slices=VAL_SLICES, test_slices=TEST_SLICES,
                       train_slices=train_slices,
                       sigma_hat={int(s): float(sig[s]) for s in range(Ns)}),
                  f, indent=2)

    # ---- deployment-condition validation (deterministic: nothing injected) ----
    ens_v = x[VAL_SLICES].to(DEVICE)                         # (V,N,2,H,W)
    sv = sig[VAL_SLICES].to(DEVICE).repeat_interleave(N)     # (V*N,)
    xv = ens_v.reshape(-1, 2, Nx, Nx)
    Tv = ((ens_v.sum(1, keepdim=True) - ens_v) / (N - 1)).reshape(-1, 2, Nx, Nx)

    @torch.no_grad()
    def validate():
        n2n.net.eval()
        x0 = xv + sv.square().view(-1, 1, 1, 1) * n2n.net(xv, n2n.level_coord(sv))
        out = (x0 + (xv - x0) / 3.0).view_as(ens_v)
        l1 = torch.nn.functional.l1_loss(x0, Tv).item()
        mse = torch.nn.functional.mse_loss(x0, Tv).item()
        # achieved residual noise of the 1/3 decode, relative to sigma_hat
        sd = out - out.mean(1, keepdim=True)
        noise = ((sd.square().sum(1) / (N - 1)).mean(dim=(1, 2, 3)).sqrt()
                 / sig[VAL_SLICES].to(DEVICE)).mean().item()
        mo = torch.sqrt(out.mean(1).square().sum(1))
        mr = torch.sqrt(ens_v.mean(1).square().sum(1))
        gain = ((mo * mr).sum(dim=(1, 2)) / mr.square().sum(dim=(1, 2))).mean().item()
        n2n.net.train()
        return l1, mse, noise, gain

    fields = ["loss", "iter", "train_loss", "train_loss_avg50",
              "val_l1", "val_mse", "val_noise13", "val_gain13", "sec"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    def save(tag, it, loss):
        p = os.path.join(RUN, "ckpt", f"{args.loss}_{tag}.pkl")
        torch.save({"state_dict": n2n.net.state_dict(), "cfg": ckpt_cfg,
                    "train_iters": it, "loss_final": loss}, p)

    vl, vm, vn, vg = validate()
    log(f"init val_l1={vl:.5f} val_mse={vm:.6f} noise13={vn:.3f} gain13={vg:.3f}")

    recent = []
    t0 = time.time()
    for it in range(1, args.iters + 1):
        poss = torch.randint(0, len(train_slices), (CFG["BATCH"],), device=DEVICE)
        loss = trainer.step_multireal(x_train[poss], loss=args.loss,
                                      sigma_y=sy_train[poss], p_raw=CFG["P_RAW"])
        recent = (recent + [loss])[-50:]
        if it % args.val_every == 0 or it == 1:
            vl, vm, vn, vg = validate()
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(dict(
                    loss=args.loss, iter=it, train_loss=loss,
                    train_loss_avg50=float(np.mean(recent)), val_l1=vl, val_mse=vm,
                    val_noise13=vn, val_gain13=vg, sec=time.time() - t0))
            if it % (args.val_every * 4) == 0:
                log(f"iter {it:>5} train={np.mean(recent):.5f} val_l1={vl:.5f} "
                    f"val_mse={vm:.6f} noise13={vn:.3f} gain13={vg:.3f}")
        if it % args.ckpt_every == 0:
            save(f"{it:06d}", it, loss)

    save("final", args.iters, loss)
    log(f"DONE {args.loss}: iters={args.iters} val_l1={vl:.5f} val_mse={vm:.6f} "
        f"elapsed={(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
