# -*- coding: utf-8 -*-
"""NAT training (Archive/NAT/PREREGISTRATION.md): one trainer for every learned method on natural images.

Same 1-channel U-Net (base 32, emb 32), Adam 1e-3, grad clip 1, batch 8 images x 10 fixed noisy copies, random 96x96
crops (the same crop for all copies of an image), 3000 iterations, MSE unless the method defines its own loss.

  ours    DSM score parametrization, total-noise labels, noise ladder (p_raw 0.5), leave-one-out N2N target
  avg9    same model; loss = N2N error of the AVERAGED estimate over a random 9-copy subset (target: mean of the rest)
  avgu    same, subset size a ~ U{1..9} per image
  sup     same model and ladder; clean target (upper bound)
  n2n     classic Noise2Noise: direct output, no sigma input, native noise, single other copy as target
  ne2ne / r2r   as baselines/dl_selfsup.py (no sigma input, direct output)
  n2s     Noise2Score v2 (AR-DAE, conditioned on total noise), as baselines/noise2score.py

    python natural/train.py --method ours --cond W05 --outdir runs/nat/W05/ours
"""
import os, time, json, math, argparse, csv
import numpy as np
import torch
import data as D

from unet import UNet                                        # noqa: E402
from n2n import N2N_DSM                                      # noqa: E402

CFG = dict(L=40, ITERS=3000, BATCH=8, CROP=96, LR=1e-3, GRAD_CLIP=1.0, NET_BASE=32, NET_EMB=32, P_RAW=0.5,
           LADDER_TOP=5.0, CKPT_EVERY=1000)
METHODS = ("ours", "avg9", "avgu", "sup", "n2n", "ne2ne", "r2r", "n2s", "sub", "subsup")   # sub*: NAT-SUB addendum


def build(method, sigma, dev):
    direct = method in ("n2n", "ne2ne", "r2r")
    net = UNet(in_ch=1, out_ch=1, base_out=CFG["NET_BASE"], emb_dim=CFG["NET_EMB"]).to(dev)
    bottom = sigma / 3 if method in ("sub", "subsup") else sigma          # mean of up to 9 copies: sigma/3
    m = N2N_DSM(L=CFG["L"], sigma_min=bottom, sigma_max=CFG["LADDER_TOP"] * sigma, backbone=net, ch=1,
                param="direct" if direct else "score", blind=direct)
    return net, m


def step_loss(method, m, net, y, clean, sigma):
    """y: (B, N, h, w) noisy copies; clean: (B, h, w)."""
    B, N, h, w = y.shape
    dev = y.device
    sy = torch.full((B,), sigma, device=dev)
    if method == "ours":
        return m.loss_multireal(y, loss="mse", sigma_y=sy, p_raw=CFG["P_RAW"])
    if method == "n2n":
        return m.loss_multireal(y, loss="mse", sigma_y=sy, p_raw=1.0, target_k=1)
    if method == "sub":                   # denoise the mean of m ~ U{1..9} copies; target: mean of the rest
        return m.loss_subset(y, sy, loss="mse", p_raw=CFG["P_RAW"])
    if method == "subsup":                # same input, clean target
        K = B * N
        mm = torch.randint(1, N, (K,), device=dev)
        rank = torch.rand(K, N, device=dev).argsort(1).argsort(1)
        wA = (rank < mm[:, None]).float() / mm[:, None]
        X = (wA[:, :, None, None] * y.repeat_interleave(N, dim=0)).sum(1, keepdim=True)
        x0, _ = m._tweedie_total(X, sigma / mm.float().sqrt(), None, CFG["P_RAW"])
        return ((x0[:, 0] - clean.repeat_interleave(N, dim=0)) ** 2).mean()
    X = y.reshape(B * N, 1, h, w)
    syN = sy.repeat_interleave(N)
    if method in ("avg9", "avgu", "sup"):
        x0, _ = m._tweedie_total(X, syN, None, CFG["P_RAW"])
        x0 = x0.view(B, N, h, w)
        if method == "sup":
            return ((x0 - clean[:, None]) ** 2).mean()
        a = torch.full((B,), 9, device=dev) if method == "avg9" else torch.randint(1, N, (B,), device=dev)
        rank = torch.rand(B, N, device=dev).argsort(1).argsort(1)
        inA = (rank < a[:, None]).float()
        est = (inA[:, :, None, None] * x0).sum(1) / a.view(B, 1, 1).float()
        tgt = ((1 - inA)[:, :, None, None] * y).sum(1) / (N - a).view(B, 1, 1).float()
        return ((est - tgt) ** 2).mean()
    if method in ("ne2ne", "r2r"):
        from dl_selfsup import loss_fn
        return loss_fn(method, net, X, syN)
    if method == "n2s":
        rel = torch.exp(math.log(0.05) + torch.rand(B * N, device=dev) * math.log(2.0 / 0.05))
        d = rel * syN
        u = torch.randn_like(X)
        score = net(X + d.view(-1, 1, 1, 1) * u, m.level_coord(torch.sqrt(syN ** 2 + d ** 2)))
        return ((u + d.view(-1, 1, 1, 1) * score) ** 2).mean()
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--cond", required=True, choices=list(D.CONDITIONS))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--iters", type=int, default=CFG["ITERS"])
    ap.add_argument("--n-train", type=int, default=0, help="C7: use only the first n training images (0 = all 400)")
    ap.add_argument("--p-raw", type=float, default=None, help="C3: 1.0 = no noise ladder (native noise only)")
    args = ap.parse_args()
    if args.p_raw is not None:
        CFG["P_RAW"] = args.p_raw
    os.chdir(D.REPO)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert not os.path.exists(os.path.join(args.outdir, "train.csv")), "refusing to overwrite " + args.outdir
    os.makedirs(os.path.join(args.outdir, "ckpt"), exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    sigma, colored = D.CONDITIONS[args.cond]
    ims = D.load_split("train")
    if args.n_train:
        ims = ims[:args.n_train]
    Y = torch.from_numpy(np.stack(D.copies(ims, args.cond, "train"))).to(dev)        # (400, N, 180, 180)
    Xc = torch.from_numpy(np.stack(ims)).to(dev)                                      # clean (sup only)
    net, m = build(args.method, sigma, dev)
    opt = torch.optim.Adam(net.parameters(), lr=CFG["LR"])
    ckpt_cfg = dict(net_type="N2N_DSM", noise_label="total", objective=args.method, N=D.N, L=CFG["L"], ch=1,
                    sigma_min=float(m.dsm.sigmas[0]), sigma_max=CFG["LADDER_TOP"] * sigma, base_out=CFG["NET_BASE"],
                    emb_dim=CFG["NET_EMB"], loss=args.method, param=m.param, blind=m.blind, cond=args.cond,
                    seed=args.seed, alpha=0.5, n_train=len(ims), p_raw=CFG["P_RAW"])
    json.dump(dict(cfg=CFG, ckpt_cfg=ckpt_cfg, cond=args.cond, sigma=sigma, colored=colored, seed=args.seed,
                   iters=args.iters), open(os.path.join(args.outdir, "config.json"), "w"), indent=2)
    log = open(os.path.join(args.outdir, "train.log"), "a", encoding="utf-8")
    with open(os.path.join(args.outdir, "train.csv"), "w", newline="") as f:
        csv.writer(f).writerow(["iter", "loss_avg50", "sec"])
    H, c = Y.shape[-1], CFG["CROP"]
    recent, t0 = [], time.time()
    for it in range(1, args.iters + 1):
        idx = torch.randint(0, Y.shape[0], (CFG["BATCH"],), device=dev)
        tl = torch.randint(0, H - c + 1, (CFG["BATCH"], 2))
        y = torch.stack([Y[i, :, t:t + c, l:l + c] for i, (t, l) in zip(idx.tolist(), tl.tolist())])
        x = torch.stack([Xc[i, t:t + c, l:l + c] for i, (t, l) in zip(idx.tolist(), tl.tolist())])
        loss = step_loss(args.method, m, net, y, x, sigma)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CFG["GRAD_CLIP"]); opt.step()
        recent = (recent + [loss.item()])[-50:]
        if it % 50 == 0:
            for attempt in range(5):
                try:
                    with open(os.path.join(args.outdir, "train.csv"), "a", newline="") as f:
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
                       os.path.join(args.outdir, "ckpt", f"{args.method}_{it:06d}.pkl"))
    log.write(f"DONE {args.method} {args.cond}: {args.iters} iterations in {(time.time() - t0) / 60:.1f} min\n"); log.close()


if __name__ == "__main__":
    main()
