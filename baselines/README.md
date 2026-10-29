# baselines/

Standalone baseline comparison for roadmap step 2 (pre-registration: `../Archive/BASE/PREREGISTRATION.md`).

| File | Content |
|---|---|
| `_common.py` | Paths, data, the fixed subset of 10 splits per a (shared by every method), split-half unbiased n_eff scoring (same estimator as `experiments/_ratio_core.py`) |
| `classical.py` | Plain averaging, Gaussian blur, non-local means (scikit-image), BM3D (`bm3d` 4.0.3), MP-PCA, LPCA and Patch2Self (dipy 1.12 reference implementations); one strength parameter each |
| `noise2score.py` | Noise2Score (Kim & Ye 2021) re-implemented with this repository's U-Net: AR-DAE score of the noisy data, σ-conditioned, Tweedie decoding |
| `evaluate.py` | Scores methods per held-out slice; `--check` reproduces the logged n_eff of our model with all splits |

**Protocol.** Every method sees only the A half of each split (never the reference half). Image denoisers act on
the A-half mean (noise σ̂/√a); multi-volume methods act jointly on the 2a Re/Im volumes. Each method's strength
parameter (r for ours) is chosen per a by cross-fitting over the other folds' slices.

**Dependencies.** `bm3d`, `bm4d`, `dipy`, `nibabel`, `trx-python` are installed outside the repository in
`../../_pydeps` (`pip install --no-deps --target _pydeps bm3d bm4d dipy nibabel trx-python` from the Jupyter folder),
so the base environment is untouched; `_common.py` adds that folder to the path.

**Notes.**
- Patch2Self is run as dipy's version 1 (the published algorithm with spatial patches); version 3 does not accept a
  patch radius and fails with 2 volumes. Its intensity shift is disabled (signed Re/Im data). On a toy phantom it
  shows a mean offset with few volumes, a property of the regression, not of the shift.
- Not implemented: DDM2 (three-stage diffusion-MRI pipeline) and BM4D (volumetric).
- Run from the repo root: `python baselines/evaluate.py --methods ... --out runs/base/<tag>`.
