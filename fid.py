"""
Fréchet Inception Distance (FID) for the diffussion folder.

Setup:
  - We work with 2-channel (B, 2, H, W) Real/Imag images in [-1, 1].
  - Inception-v3 needs RGB, so `to_rgb` maps (Re, Im, Re) -> (B, 3, H, W)
    and rescales to [0, 255].
  - FID between two Gaussian statistics on feature activations.

If torchvision (Inception-v3) is missing, the FID object degrades gracefully:
    available == False and compute() returns float('nan').
"""
import torch
import torch.nn.functional as F

try:
    from torchvision.models import inception_v3, Inception_V3_Weights
    _HAS_TV = True
except Exception:  # torchvision may not be installed
    _HAS_TV = False


def to_rgb(x):
    """(B,2,H,W) in [-1,1] -> (B,3,H,W) in [0,255] using (Re, Im, Re)."""
    x = (x + 1.0) / 2.0 * 255.0
    re, im = x[:, 0], x[:, 1]
    return torch.stack([re, im, re], dim=1).clamp(0, 255)


class InceptionEmbedder:
    """Pre-classifier (penultimate) features from Inception-v3."""

    def __init__(self, device="cpu"):
        if not _HAS_TV:
            raise RuntimeError("torchvision is required for the Inception embedder.")
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()      # drop the 1000-way classifier head
        model.eval()
        self.model = model.to(device)

    @torch.no_grad()
    def embed(self, rgb, batch_size=32):
        feats = []
        for i in range(0, rgb.shape[0], batch_size):
            b = rgb[i:i + batch_size]
            b = F.interpolate(b, size=299, mode="bilinear", align_corners=True)
            feats.append(self.model(b).float())
        return torch.cat(feats, dim=0)


def _sqrtm_sym(A, B):
    """Trace of sqrt(A @ B) for symmetric PSD A and B, via the symmetric form.

    A @ B is generally NOT symmetric even when A,B are, so an eigendecomposition
    of the product is invalid. Use the equivalent symmetric PSD product
        M = A^{1/2} @ B @ A^{1/2}
    whose eigenvalues equal those of A@B; Tr(sqrt(M)) is then well-defined.
    """
    A = (A + A.T) / 2  # guard against float asymmetry
    evals, evecs = torch.linalg.eigh(A)
    A_sqrt = evecs @ torch.diag(evals.clamp_min(0.0).sqrt()) @ evecs.T
    M = A_sqrt @ B @ A_sqrt
    M = (M + M.T) / 2
    tr = torch.linalg.eigvalsh(M).clamp_min(0.0).sqrt().sum()
    return tr


def fid_from_stats(mu_r, cov_r, mu_f, cov_f):
    """FID between two fitted Gaussians (float64, CPU)."""
    d = (mu_r - mu_f).double()
    cov_r = cov_r.double()
    cov_f = cov_f.double()
    tr_covprod = _sqrtm_sym(cov_r, cov_f).real
    # FID is a squared Mahalanobis-like distance and must be >= ~0.
    val = float(d @ d + torch.trace(cov_r) + torch.trace(cov_f) - 2 * tr_covprod)
    return val


class FID:
    """
    Online FID accumulator. Feed 2-channel Real/Imag batches to
    `.add_real(...)` / `.add_fake(...)`, then call `.compute()`.

    Also exposes standalone `.compute_fid(real_images, fake_images)`.
    """

    def __init__(self, device="cpu", batch_size=32):
        self.device = device
        self.batch_size = batch_size
        if not _HAS_TV:
            print("[FID] torchvision unavailable -> FID disabled.")
            self.available = False
            self.embedder = None
        else:
            self.available = True
            self.embedder = InceptionEmbedder(device)

    # ---- accumulation ---------------------------------------------
    def _new_stats(self):
        return {"n": 0, "mu": None, "sum": None}

    def __init_features(self):
        if not hasattr(self, "_real"):
            self._real = self._new_stats()
            self._fake = self._new_stats()

    def add_real(self, images): self.__init_features(); self._accum(images, self._real)
    def add_fake(self, images): self.__init_features(); self._accum(images, self._fake)

    def _accum(self, images, stats):
        if not self.available:
            return
        rgb = to_rgb(images).to(self.device)
        f = self.embedder.embed(rgb, self.batch_size).float()
        if stats["n"] == 0:
            stats["mu"] = torch.zeros(f.shape[1], device=self.device)
            stats["sum"] = torch.zeros((f.shape[1], f.shape[1]), device=self.device)
        stats["n"] += f.shape[0]
        stats["mu"] += f.sum(0)
        stats["sum"] += f.T @ f

    def _finalize(self, s):
        mu = s["mu"] / s["n"]
        cov = s["sum"] / (s["n"] - 1) - mu[:, None] * mu[None, :]
        return mu.double(), cov.double()

    @torch.no_grad()
    def compute(self, real_feats=None, fake_feats=None):
        """Return FID over accumulated add_real/add_fake calls."""
        if not self.available or getattr(self, "_real", None) is None \
                or self._real["n"] == 0 or self._fake["n"] == 0:
            return float("nan")
        mu_r, cov_r = self._finalize(self._real)
        mu_f, cov_f = self._finalize(self._fake)
        return fid_from_stats(mu_r, cov_r, mu_f, cov_f)

    # -- one-shot helper -------------------------------------------
    @torch.no_grad()
    def get_fid(self, real_images, fake_images):
        """Direct FID between two explicit sets (B1,2,H,W) and (B2,2,H,W)."""
        if not self.available:
            return float("nan")
        mu_r, cov_r = self._embed_stats(real_images)
        mu_f, cov_f = self._embed_stats(fake_images)
        return fid_from_stats(mu_r, cov_r, mu_f, cov_f)

    def _embed_stats(self, images):
        rgb = to_rgb(images).to(self.device)
        f = self.embedder.embed(rgb, self.batch_size).double()
        mu = f.mean(0)
        cov = torch.cov(f.T)
        return mu, cov