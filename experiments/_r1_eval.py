# -*- coding: utf-8 -*-
"""R1: decode-only evaluation of saved R0 checkpoints.

For every checkpoint of a loss variant, decode the held-out val (default) /
test slices through ``conditional_denoise`` and report structure metrics
against the 10-realization mean, per realization and after ensemble averaging.

Writes one long-form CSV per loss + a compact markdown summary. No plots unless
--plot is given (matplotlib on Agg, saved next to the CSVs).

usage:
    python _r1_eval.py --loss l1 --split val --sr 1.5 2.0 --cs 0
"""
import sys, os, glob, json, argparse, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from scipy.ndimage import gaussian_filter as gf

DIFF = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
sys.path.insert(0, DIFF)
os.chdir(DIFF)

from unet import UNet
from n2n import N2N_DSM

VAL_SLICES = [16, 11, 1]
TEST_SLICES = [25, 22, 8]
SEED_BASE = 12345        # fixed diffusion-noise seed for every eval run


def load_data():
    img = np.load("../asl_for_diffusion/dataset1.npy")
    t = torch.from_numpy(img).permute(2, 3, 0, 1).contiguous()
    scale = t.abs().amax(dim=(1, 2, 3), keepdim=True)
    t = t / scale
    return torch.stack([t.real, t.imag], dim=2).contiguous()


def mag(x):
    return torch.sqrt(x[..., 0, :, :] ** 2 + x[..., 1, :, :] ** 2)


def hf(m):
    gy, gx = np.gradient(m)
    return float(np.mean(np.abs(gy) + np.abs(gx)))


def hp(m):
    return float(np.mean(np.abs(m - gf(m, 1.5))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", required=True)
    ap.add_argument("--runroot", default="runs/r0_split")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--sr", type=float, nargs="+", default=[1.5, 2.0])
    ap.add_argument("--cs", type=int, nargs="+", default=[0])
    ap.add_argument("--max-ckpt", type=int, default=0)
    ap.add_argument("--ckpt-iters", type=str, nargs="+", default=None,
                    help="only these checkpoint iterations ('final' allowed)")
    ap.add_argument("--append", action="store_true",
                    help="keep existing rows, skip already-evaluated ones")
    args = ap.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    run = os.path.join(args.runroot, args.loss)
    cfg = json.load(open(os.path.join(run, "config.json")))
    slices = VAL_SLICES if args.split == "val" else TEST_SLICES

    ckpts = sorted(glob.glob(os.path.join(run, "ckpt", f"{args.loss}_*.pkl")))
    finals = [c for c in ckpts if c.endswith("_final.pkl")]
    numbered = [c for c in ckpts if not c.endswith("_final.pkl")]

    def it_of(p):
        b = os.path.basename(p)[:-4].split("_")[-1]
        return 10**9 if b == "final" else int(b)

    ckpts = sorted(numbered + finals, key=it_of)
    if args.max_ckpt:
        step = max(1, len(ckpts) // args.max_ckpt)
        ckpts = ckpts[::step]
    if args.ckpt_iters:
        want = set()
        for s in args.ckpt_iters:
            if s.isdigit():
                want.add(str(int(s)))
            else:
                want.add(s)
        keep = [c for c in ckpts
                if ("final" if it_of(c) == 10**9 else str(it_of(c))) in want]
        missing = want - {("final" if it_of(c) == 10**9 else str(it_of(c)))
                          for c in keep}
        if missing:
            print(f"WARNING: no checkpoint for {sorted(missing)}", flush=True)
        ckpts = keep

    x = load_data()
    N = x.shape[1]
    Nx = x.shape[-1]
    row_tpl = ("loss,split,ckpt_iter,sr,cs,slice,bias1,bias10,"
               "rmse1,rmse10,nmse1,nmse10,hf_pct1,hf_pct10,hp_pct1,hp_pct10\n")

    out_csv = os.path.join(run, f"eval_{args.split}.csv")
    done = set()
    if args.append and os.path.exists(out_csv):
        import pandas as pd
        old = pd.read_csv(out_csv)
        if len(old):
            done = {(int(r.ckpt_iter), float(r.sr), int(r.cs))
                    for r in old.itertuples()}
        tf = open(out_csv, "a", encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] appending to {out_csv} "
              f"({len(done)} configs already done)", flush=True)
    else:
        tf = open(out_csv, "w", encoding="utf-8")
        tf.write(row_tpl)
    ckpts = [c for c in ckpts
             if any((it_of(c), float(sr), int(cs)) not in done
                    for sr in args.sr for cs in args.cs)]
    print(f"[{time.strftime('%H:%M:%S')}] {len(ckpts)} ckpts -> {out_csv}", flush=True)

    net_template = dict(cfg["cfg"], loss=cfg["loss"])
    for ci, path in enumerate(ckpts):
        blob = torch.load(path, map_location=DEVICE, weights_only=False)
        c = blob["cfg"]
        net = UNet(in_ch=2, out_ch=2, base_out=c.get("base_out", 32),
                   emb_dim=c.get("emb_dim", 32)).to(DEVICE)
        net.load_state_dict(blob["state_dict"])
        n2n = N2N_DSM(L=c["L"], sigma_min=c["sigma_min"],
                      sigma_max=c["sigma_max"], backbone=net)
        n2n.net.eval()

        for s in slices:
            ens = x[s].to(DEVICE)                        # (N,2,Nx,Ny)
            resid = ens - ens.mean(0, keepdim=True)
            sigma_task = resid.square().mean().sqrt().item()
            ref_m = mag(ens.mean(0))                     # 10-real mean magnitude
            ref_hf, ref_hp = hf(ref_m.cpu().numpy()), hp(ref_m.cpu().numpy())

            for sr in args.sr:
                for cs in args.cs:
                    # All N realizations are decoded as ONE batch: the reverse walk
                    # is per-sample, so this is mathematically the same as looping
                    # but ~N times cheaper. One fixed seed -> deterministic and
                    # identical across every checkpoint.
                    torch.manual_seed(SEED_BASE)
                    rec = n2n.conditional_denoise(
                        ens, sigma_task=sigma_task, sigma_ratio=sr,
                        corrector_steps=cs, corrector_alpha=0.05,
                        device=DEVICE).detach()          # (N,2,Nx,Ny)

                    def stats(est_map, ref_map):
                        rm = mag(ref_map)
                        b = (est_map.mean() / (rm.mean() + 1e-12)).item()
                        r = ((est_map - rm) ** 2).mean().sqrt().item()
                        nm = (((est_map - rm) ** 2).mean()
                              / (rm ** 2).mean()).item()
                        return b, r, nm

                    m1, mm = mag(rec), mag(rec.mean(0))
                    b1 = np.mean([stats(m1[k], ens.mean(0))[0] for k in range(N)])
                    r1 = np.mean([stats(m1[k], ens.mean(0))[1] for k in range(N)])
                    n1 = np.mean([stats(m1[k], ens.mean(0))[2] for k in range(N)])
                    b10, r10, n10 = stats(mm, ens.mean(0))
                    hf1 = np.mean([hf(m1[k].cpu().numpy()) for k in range(N)])
                    hp1 = np.mean([hp(m1[k].cpu().numpy()) for k in range(N)])
                    hf10 = hf(mm.cpu().numpy())
                    hp10 = hp(mm.cpu().numpy())

                    tf.write(f"{args.loss},{args.split},{it_of(path)},{sr},{cs},{s},"
                             f"{b1:.4f},{b10:.4f},{r1:.5f},{r10:.5f},"
                             f"{n1:.5f},{n10:.5f},"
                             f"{100*hf1/ref_hf:.1f},{100*hf10/ref_hf:.1f},"
                             f"{100*hp1/ref_hp:.1f},{100*hp10/ref_hp:.1f}\n")
                    tf.flush()
                    print(f"  ckpt {it_of(path):>7} slice {s:>2} sr={sr} cs={cs} "
                          f"bias10={b10:.3f} rmse10={r10:.5f} "
                          f"HF={100*hf10/ref_hf:.0f}% HP={100*hp10/ref_hp:.0f}%",
                          flush=True)
        del net, n2n
        torch.cuda.empty_cache()
        print(f"[{time.strftime('%H:%M:%S')}] done {ci+1}/{len(ckpts)} "
              f"({os.path.basename(path)})", flush=True)

    tf.close()
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
