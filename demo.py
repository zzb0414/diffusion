"""
Smoke test for the diffussion folder: train both DDPM and DSM briefly on a toy
set of 2-channel (Real/Imag) blobs, run each sampler, and report FID against the
real data.
Run:  python demo.py
"""
import torch

from unet import UNet
from ddpm import DDPM
from dsm import DSM
from train import DiffusionTrainer
from fid import FID

DEVICE = "cpu"
SEED = 0
torch.manual_seed(SEED)


def make_blobs(n=32, res=32, ch=2):
    """Synthetic 2-channel (B,2,H,W) dataset: one Gaussian blob per channel."""
    y, x = torch.meshgrid(torch.linspace(-1, 1, res), torch.linspace(-1, 1, res))
    r = torch.sqrt(x ** 2 + y ** 2)
    data = []
    for _ in range(n):
        img = torch.zeros(ch, res, res)
        for c in range(ch):
            cx, cy = (torch.rand(1).item() * 1.6) - 0.8, (torch.rand(1).item() * 1.6) - 0.8
            rad = 0.2 + 0.2 * torch.rand(1).item()
            img[c] = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * rad ** 2))
        data.append(img)
    return torch.stack(data)


def main():
    x = make_blobs()

    print("=== DDPM ===")
    ddpm = DDPM(T=200, beta_start=1e-4, beta_end=0.02, backbone=UNet(base_out=16, emb_dim=64))
    ddpm.net.train()
    tr = DiffusionTrainer(ddpm, lr=1e-3, device=DEVICE)
    for it in range(60):
        loss = tr.step(x[:8])
        if it % 20 == 0 or it == 59:
            print(f"  iter {it}: loss={loss:.4f}")
    ddpm.net.eval()
    ddpm_samples = ddpm.sample((8, 2, 32, 32), steps=250, device=DEVICE)
    print(f"  sampled: {tuple(ddpm_samples.shape)}  min={ddpm_samples.min():.3f} max={ddpm_samples.max():.3f}")
    fid_model = FID(device=DEVICE, batch_size=8)
    print(f"  FID real-vs-DDPM: {fid_model.get_fid(x[:16], ddpm_samples):.3f}")

    print("=== DSM ===")
    dsm = DSM(L=10, sigma_min=0.01, sigma_max=10.0, backbone=UNet(base_out=16, emb_dim=64))
    dsm.net.train()
    tr2 = DiffusionTrainer(dsm, lr=1e-3, device=DEVICE)
    for it in range(60):
        loss = tr2.step(x[:8])
        if it in [0, 20, 40, 59]:
            print(f"  iter {it}: loss={loss:.4f}")
    dsm.net.eval()
    dsm_samples = dsm.sample((8, 2, 32, 32), steps=500, eps=2e-5, device=DEVICE)
    print(f"  sampled: {tuple(dsm_samples.shape)}  min={dsm_samples.min():.3f} max={dsm_samples.max():.3f}")
    fid_model2 = FID(device=DEVICE, batch_size=8)
    print(f"  FID real-vs-DSM: {fid_model2.get_fid(x[:16], dsm_samples):.3f}")

    print("\nOK: both models trained, sampled, and FID reported.")


if __name__ == "__main__":
    main()