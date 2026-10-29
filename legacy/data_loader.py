"""
Data-loader template for the diffussion folder.

Every dataset should yield 2-channel (Real / Imag) images of a fixed resolution,
shaped (2, H, W). Subclass `ChannelImageDataset` and implement `__getitem__` /
`__len__` (see the two examples below: one synthetic, one reading numpy arrays).

NOTE: This is a template. Wire `__getitem__` to YOUR storage format (npz, nifti,
h5, folder of .npy, etc.). The two classes below are placeholders meant to be
adapted, not necessarily runnable against your real data yet.
"""
import os

import torch
from torch.utils.data import Dataset, DataLoader


class ChannelImageDataset(Dataset):
    """Base template: 2-channel (2, H, W) tensors in [-1, 1]."""

    def __init__(self, file_paths=None, normalize=True):
        self.file_paths = file_paths or []
        self.normalize = normalize

    def __getitem__(self, idx):
        # ---- Replace me with your real load / parse logic ----
        path = self.file_paths[idx]
        tensor = self._load(path)        # (2, H, W), float32
        if self.normalize:
            tensor = self._normalize(tensor)
        return tensor

    def _load(self, path):
        raise NotImplementedError("Implement per-data-format loading.")

    @staticmethod
    def _normalize(x):
        """Per-sample min/max to [-1, 1] so both channels share a scale."""
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-8:
            return x * 0.0
        return (x - lo) / (hi - lo) * 2.0 - 1.0

    def __len__(self):
        return len(self.file_paths)


class NpyStackDataset(ChannelImageDataset):
    """Loads .npy stacks where each file is (W, C, H, W) or (C, H, W)."""

    def _load(self, path):
        a = torch.from_numpy(__import__("numpy").load(path)).float()
        a = a.squeeze()
        if a.ndim == 3:
            if a.shape[0] == 2:      # (2, H, W)
                return a
            else:                    # (N, H, W) -> put two of them as channels
                return a[:2]
        if a.ndim == 4:
            return a[0].squeeze()    # take first sample: (2, H, W) or (H, W)...
        raise ValueError(f"Unexpected shape {tuple(a.shape)}")


class PairedComplexDataset(ChannelImageDataset):
    """Complex (H, W) 64 -> split into real/imag channels (2, H, W)."""

    def _load(self, path):
        import numpy as np
        arr = np.load(path)          # complex64 array (H, W) or (N, H, W)
        arr = np.atleast_3d(arr)
        arr = arr[0]                 # (H, W) complex
        return torch.from_numpy(np.stack([arr.real, arr.imag], axis=0)).float()


def build_dataloader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False):
    """Thin wrapper returning a torch DataLoader for the ChannelImageDataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)


def dataloader_from_dir(dir_path, batch_size=16, shuffle=True):
    """Point at a directory of stack-like .npy/.npz files."""
    exts = (".npy", ".npz")
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(exts)]
    return build_dataloader(NpyStackDataset(files), batch_size=batch_size, shuffle=shuffle)