# GAP F pre-registration — is ASL's small gain a learnability limit (one training subject), and how do fastMRI-trained models do on real ASL?

Written 2026-11-06, 19:40, after GAP D/E and BEFORE any F run. Two exploratory quick-looks were made before writing this
(disclosed): SUB/PER at oracle r of the seed-101 DT models on 1104c, and a band-kurtosis comparison (inconclusive).

## Where we stand (GAP D/E, `WEEKEND_ITERATIONS.md`)
- **Giving fastMRI content ASL's per-frequency SNR does not shrink the gain. It raises it:** SUB/PER is 1.457 (DT), against a Wiener prediction of 1.149.
- **The same spectrum with random phase (EW) gives learned / Wiener 0.98:** the networks are linear-optimal on Gaussian-like content.
- **ASL's learned / Wiener is 0.92–0.96.** So ASL behaves like Gaussian-like content: either ASL has no exploitable structure at this SNR, or it has structure the networks cannot learn or exploit.
- **Candidates for why:**
  - (i) ASL content itself, perfusion maps with little sparse structure;
  - (ii) physiological fluctuation that resembles the signal, which no denoiser can separate;
  - (iii) learnability: ASL models train on ONE subject (dataset 1 + 1104a = subject A, 68 slices), while fastMRI trains on 62 subjects.

## F1 — fastMRI-trained models on real ASL (evaluation only; exploratory, no primary)
- **Models:** the GAP D DT models and the GAP B T models (A, SUB, SUBM; seeds 101–102; checkpoints 1000/2000/3000).
- **Evaluation:** 1104c scored with the unchanged ASL split-half estimator (`experiments/_tr_eval.py` → `_gr_eval.score_slice`). Each slice is scaled by 1/σ̂ to the model's label. Selection is cross-fitted over slice folds as in ASL-SUBJ.
- **Report:** SUB/PER, SUBM/PER and absolute n_eff, against the ASL-trained references (`runs/subj`, `runs/coil/subj`).
- **Reading rule:** fastMRI-trained SUB or SUBM at least 1.03× the ASL-trained one in absolute n_eff, with a = 9 and the CI excluding 1, would mean a fastMRI anatomical prior is worth more on ASL than ASL's own training data. That points to (iii).

## F2 — training-set size at the DT setting (training; PRIMARY)
- **Design:** GAP D's DT condition (content D, noise T, recipe unchanged) trained on n_vol = 1 (8 images, one subject) and n_vol = 4 (32 images), with arms A, SUB, SUBM and seeds 101–102. Test on the same 15 volumes and with the same selection as GAP D.
- **Size:** n_vol = 1 is harsher than ASL, which has 34 distinct slices of one subject.
- **PRIMARY (Holm over 2), a = 9, paired over the 15 test volumes:**
  - Q1: (SUB/PER, DT n_vol 1) / (SUB/PER, DT n_vol 62).
  - Q2: (SUB/PER, DT n_vol 4) / (SUB/PER, DT n_vol 62).
- **Prediction, from MEAN, where 2 volumes kept a 1.71× gain at matrix 160:** Q1 and Q2 ≥ 0.9, and SUB/PER at n_vol 1 stays ≥ 1.3. Then one training subject is not why ASL's gain is small, and (iii) is out, unless F1 says otherwise.
- **If Q1 ≤ 0.8 and SUB/PER at n_vol 1 ≤ 1.15:** single-subject training can by itself reproduce ASL's small gain. More ASL subjects then becomes the priority, which bears on the coil2 data plan.

## Next, fixed now
If (iii) is out, test (ii): fastMRI DT with a signal-like physiological component, a multiplicative smooth fluctuation of the content, calibrated to ASL's in-brain coarse excess, replacing part of the independent noise. That needs its own pre-registration.

## No-overwrite
Outputs only go under `runs/gap/f/` and `runs/gap/report/summary_f.json`. The scripts refuse to overwrite, and training and evaluation code is unchanged apart from the new `_tr_eval.py`.
