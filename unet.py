"""
5-layer U-Net backbone shared by the DDPM and DSM diffusion models in this folder.

Design:
  - Assumes input is a 2D image with 2 channels (Real / Imag).
  - Encoder/decoder are double (conv+GroupNorm+SiLU) blocks, down/up by 2.
       enc1: H     enc2: H/2   enc3: H/4   enc4: H/8   bottleneck: H/16
  - Each level carries a Fourier time embedding so the network is
    conditioned on the diffusion time step t (used by both DDPM and DSM);
    it is added channel-wise to the block output.
  - Base_out = width of the shallowest level, doubling at each level.
"""
import math

import torch
import torch.nn as nn


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal time embedding used in DDPM / score models."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimeEmbed(nn.Module):
    """MLP mapping a sinusoidal t-embedding onto a hidden feature vector."""

    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

    def forward(self, t):
        return self.net(timestep_embedding(t, self.net[0].in_features))


class Block(nn.Module):
    """Conv -> GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU (+ time conditioning)."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = torch.relu(h)
        h = h + self.time_proj(temb)[:, :, None, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = torch.relu(h)
        return h + self.shortcut(x)


class UNet(nn.Module):
    """
    5-layer U-Net. `base_out` is the number of channels of the first encoder level;
    each subsequent level (towards the bottleneck) doubles it:
      [base_out, 2*base_out, 4*base_out, 4*base_out, 8*base_out]
    """

    def __init__(self, in_ch=2, out_ch=2, base_out=64, emb_dim=128):
        super().__init__()
        self.time_embed = TimeEmbed(emb_dim)

        dl = [base_out, 2 * base_out, 4 * base_out, 8 * base_out, 8 * base_out]
        up = list(reversed(dl))

        # Encoder
        self.enc1 = Block(in_ch, dl[0], emb_dim)
        self.enc2 = Block(dl[0], dl[1], emb_dim)
        self.enc3 = Block(dl[1], dl[2], emb_dim)
        self.enc4 = Block(dl[2], dl[3], emb_dim)
        self.bottleneck = Block(dl[3], dl[4], emb_dim)

        # Decoder (skip connections concatenate)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = Block(dl[4] + dl[3], up[0], emb_dim)     # 8b+8b -> 8b
        self.dec3 = Block(up[0] + dl[2], up[1], emb_dim)     # 8b+4b -> 4b
        self.dec2 = Block(up[1] + dl[1], up[2], emb_dim)     # 4b+2b -> 2b
        self.dec1 = Block(up[2] + dl[0], up[3], emb_dim)     # 2b+1b -> 1b

        self.final = nn.Conv2d(up[3], out_ch, 3, padding=1)

    def forward(self, x, t):
        temb = self.time_embed(t)

        s1 = self.enc1(x, temb)
        p1 = nn.functional.avg_pool2d(s1, 2)
        s2 = self.enc2(p1, temb)
        p2 = nn.functional.avg_pool2d(s2, 2)
        s3 = self.enc3(p2, temb)
        p3 = nn.functional.avg_pool2d(s3, 2)
        s4 = self.enc4(p3, temb)
        p4 = nn.functional.avg_pool2d(s4, 2)

        b = self.bottleneck(p4, temb)

        u4 = self.up(b)
        d4 = self.dec4(torch.cat([u4, s4], dim=1), temb)
        u3 = self.up(d4)
        d3 = self.dec3(torch.cat([u3, s3], dim=1), temb)
        u2 = self.up(d3)
        d2 = self.dec2(torch.cat([u2, s2], dim=1), temb)
        u1 = self.up(d2)
        d1 = self.dec1(torch.cat([u1, s1], dim=1), temb)

        return self.final(d1)