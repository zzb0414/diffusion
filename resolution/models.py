# -*- coding: utf-8 -*-
"""Arms of the multi-resolution study (Archive/RES/PREREGISTRATION.md) and checkpoint loading.

  A   ours (DSM, total-noise ladder, one-shot Tweedie + dial), trained at matrix 160 only
  B   ours, trained on matrices {80, 160, 320}, fully convolutional, no resolution input
  C   as B, plus the voxel size as a second conditioning (its own sinusoidal embedding added to the noise embedding)
  D   standardize: every image brought to the 320 grid by zero-filling k-space (linear), denoised there, brought back
  E   classic Noise2Noise (direct output, no sigma input, native noise, one other copy as target) on {80, 160, 320}
"""
import torch
import data as D
from unet import UNet, TimeEmbed                              # noqa: E402
from n2n import N2N_DSM                                      # noqa: E402

ARMS = {"A": dict(train_res=(160,)), "B": dict(train_res=(80, 160, 320)), "C": dict(train_res=(80, 160, 320), vox=True),
        "D": dict(train_res=(80, 160, 320), grid=D.FULL), "E": dict(train_res=(80, 160, 320), classic=True)}
NET_BASE, NET_EMB, L, LADDER_TOP = 32, 32, 40, 5.0


def fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c(k):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def vox_coord(M):
    """voxel size on an embedding-friendly scale: 0 at 0.69 mm (320), 20 at 2.75 mm (80)."""
    return 10.0 * torch.log2(torch.as_tensor(D.FULL / M, dtype=torch.float32))


class VoxUNet(UNet):
    """UNet whose time embedding also carries the voxel size; set ``self.vox`` (scalar coordinate) before calling."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.vox_embed = TimeEmbed(kw["emb_dim"])
        self.vox = None

    def forward(self, x, t):
        v = torch.as_tensor(self.vox, dtype=torch.float32, device=x.device).expand(x.shape[0])
        temb = self.time_embed(t) + self.vox_embed(v)
        s1 = self.enc1(x, temb); s2 = self.enc2(torch.nn.functional.avg_pool2d(s1, 2), temb)
        s3 = self.enc3(torch.nn.functional.avg_pool2d(s2, 2), temb); s4 = self.enc4(torch.nn.functional.avg_pool2d(s3, 2), temb)
        b = self.bottleneck(torch.nn.functional.avg_pool2d(s4, 2), temb)
        d4 = self.dec4(torch.cat([self.up(b), s4], 1), temb); d3 = self.dec3(torch.cat([self.up(d4), s3], 1), temb)
        d2 = self.dec2(torch.cat([self.up(d3), s2], 1), temb); d1 = self.dec1(torch.cat([self.up(d2), s1], 1), temb)
        return self.final(d1)


def build(arm, dev):
    spec = ARMS[arm]
    sig = [D.sigma(M) for M in spec["train_res"]]
    Net = VoxUNet if spec.get("vox") else UNet
    net = Net(in_ch=2, out_ch=2, base_out=NET_BASE, emb_dim=NET_EMB).to(dev)
    classic = spec.get("classic", False)
    m = N2N_DSM(L=L, sigma_min=min(sig), sigma_max=LADDER_TOP * max(sig), backbone=net, ch=2,
                param="direct" if classic else "score", blind=classic)
    return net, m


def load(path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    net, m = build(blob["cfg"]["arm"], device)
    net.load_state_dict(blob["state_dict"]); net.eval()
    return m, blob["cfg"]
