# -*- coding: utf-8 -*-
"""R3 recipe under K-fold cross-validation over slices (the proof run).

Every one of the 34 slices is held out by exactly one fold, so every slice can
be scored by a model that never saw any of its realizations. Folds are
stratified by sigma_hat (slices sorted by noise, dealt round-robin), so each
fold spans quiet to noisy slices.

The recipe is FIXED in advance and identical to R3 (``_r3_train.py``): total-
noise labels, ``loss_multireal(sigma_y=...)``, p_raw 0.5, L1, LR 1e-3, batch 8,
UNet base 32 / emb 32, seed 101, 2000 iterations. Nothing is selected on the
held-out slices: the final (iteration 2000) weights are the ones evaluated. The
ladder is the same for every fold, 0.0368 (global smallest sigma_hat) .. 0.40,
so every held-out slice is inside it; for the fold that holds out the quietest
slice, its noise level sits slightly below that fold's quietest training slice
(flagged in the log).

The held-out L1 at the deployment condition is LOGGED for transparency only.

    python experiments/_cv_train.py --fold 0      (folds 0..K-1)
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
from _r3_train import load_data, sigma_hat, pack

K = 5
CFG = dict(
    L=40, SIGMA_MAX=0.4, P_RAW=0.5, ITERS=2000,
    BATCH=8, LR=1e-3, GRAD_CLIP=1.0,
    NET_BASE=32, NET_EMB=32, SEED=101,
)


def folds(sig):
    order = np.argsort(sig.numpy())
    return [sorted(int(s) for s in order[k::K]) for k in range(K)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--outdir", default="runs/cv_r3")
    ap.add_argument("--loss", default="l1", choices=["l1", "mse"],
                    help="regression norm (ablation; default = the CV_R3 recipe)")
    ap.add_argument("--noise-norm", action="store_true",
                    help="weight each sample by (sigma_ref/sigma_tot)^p, p=1 L1 / 2 MSE")
    ap.add_argument("--iters", type=int, default=None, help="smoke tests only; default = fixed recipe")
    ap.add_argument("--target-k", type=int, default=None,
                    help="TQ: LOO target from a random k-subset of the other repetitions (default: all N-1)")
    args = ap.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RUN = os.path.join(args.outdir, f"fold{args.fold}")
    assert not os.path.exists(os.path.join(RUN, "train.csv")), "refusing to overwrite " + RUN
    os.makedirs(os.path.join(RUN, "ckpt"), exist_ok=True)
    log_path = os.path.join(RUN, "train.log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    torch.manual_seed(CFG["SEED"])
    x = load_data()
    Ns, N = x.shape[0], x.shape[1]
    sig = sigma_hat(x)
    F = folds(sig)
    held = F[args.fold]
    train_slices = [s for s in range(Ns) if s not in held]
    sigma_min = float(sig.min())
    log(f"fold {args.fold}/{K} held-out={held} (sigma {', '.join(f'{float(sig[s]):.4f}' for s in held)}) "
        f"train={len(train_slices)} device={DEVICE}")
    qtrain = float(sig[train_slices].min())
    for s in held:
        if float(sig[s]) < qtrain:
            log(f"NOTE held-out slice {s} sigma {float(sig[s]):.4f} is quieter than every training "
                f"slice of this fold (min {qtrain:.4f}); its level is extrapolated slightly.")

    x_train = pack(x, train_slices, DEVICE)
    sy_train = sig[train_slices].to(DEVICE)
    # fixed per-fold constant for --noise-norm: keeps the loss magnitude comparable
    sigma_ref = float(torch.exp(torch.log(sig[train_slices]).mean()))
    log(f"loss={args.loss} noise_norm={args.noise_norm}"
        + (f" sigma_ref={sigma_ref:.4f} (geometric mean of training sigma_hat)" if args.noise_norm else ""))
    net = UNet(in_ch=2, out_ch=2, base_out=CFG["NET_BASE"], emb_dim=CFG["NET_EMB"]).to(DEVICE)
    n2n = N2N_DSM(L=CFG["L"], sigma_min=sigma_min, sigma_max=CFG["SIGMA_MAX"], backbone=net)
    trainer = DiffusionTrainer(n2n, lr=CFG["LR"], device=DEVICE, grad_clip=CFG["GRAD_CLIP"])
    ckpt_cfg = dict(net_type="N2N_DSM", noise_label="total", objective="multireal",
                    N=N, L=CFG["L"], sigma_min=sigma_min, sigma_max=CFG["SIGMA_MAX"],
                    base_out=CFG["NET_BASE"], emb_dim=CFG["NET_EMB"], loss=args.loss,
                    p_raw=CFG["P_RAW"], fold=args.fold, K=K, held_out=held)
    if args.noise_norm:                      # recorded only when used: default cfg unchanged
        ckpt_cfg.update(noise_norm=True, sigma_ref=sigma_ref)
    if args.target_k is not None:                # recorded only when used: default cfg unchanged
        ckpt_cfg.update(target_k=args.target_k)
    with open(os.path.join(RUN, "config.json"), "w") as f:
        json.dump(dict(cfg=CFG, ckpt_cfg=ckpt_cfg, folds=F, held_out=held,
                       train_slices=train_slices,
                       sigma_hat={int(s): float(sig[s]) for s in range(Ns)}), f, indent=2)

    ens_h = x[held].to(DEVICE)
    sh = sig[held].to(DEVICE).repeat_interleave(N)
    xh = ens_h.reshape(-1, 2, *ens_h.shape[-2:])
    Th = ((ens_h.sum(1, keepdim=True) - ens_h) / (N - 1)).reshape_as(xh)

    @torch.no_grad()
    def monitor():
        n2n.net.eval()
        x0 = xh + sh.square().view(-1, 1, 1, 1) * n2n.net(xh, n2n.level_coord(sh))
        n2n.net.train()
        return torch.nn.functional.l1_loss(x0, Th).item()

    csv_path = os.path.join(RUN, "train.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["iter", "train_loss_avg50", "heldout_l1_monitor_only", "sec"])
    recent = []
    t0 = time.time()
    for it in range(1, (args.iters or CFG["ITERS"]) + 1):
        poss = torch.randint(0, len(train_slices), (CFG["BATCH"],), device=DEVICE)
        loss = trainer.step_multireal(x_train[poss], loss=args.loss, sigma_y=sy_train[poss],
                                      p_raw=CFG["P_RAW"], noise_norm=args.noise_norm,
                                      sigma_ref=sigma_ref if args.noise_norm else None,
                                      target_k=args.target_k)
        recent = (recent + [loss])[-50:]
        if it % 50 == 0:
            hl = monitor()
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([it, float(np.mean(recent)), hl, time.time() - t0])
            if it % 500 == 0:
                log(f"iter {it:>5} train={np.mean(recent):.5f} heldout_l1(monitor)={hl:.5f}")
                torch.save({"state_dict": n2n.net.state_dict(), "cfg": ckpt_cfg,
                            "train_iters": it}, os.path.join(RUN, "ckpt", f"{args.loss}_{it:06d}.pkl"))
    log(f"DONE fold {args.fold}: elapsed {(time.time() - t0) / 60:.1f} min "
        f"(evaluated weights: iteration {CFG['ITERS']}, fixed in advance)")


if __name__ == "__main__":
    main()
