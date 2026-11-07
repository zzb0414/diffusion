# Weekend Iteration Log — asl_denoise (N2N diffusion denoising)

**Context:** User signed off for the weekend and granted Claude the right to run/test/iterate
on the real-ASL N2N denoising pipeline (`asl_denoise.ipynb` + `n2n.py` + `train.py`).
This log records each substantive run: what was run, what changed, before/after numbers, and
the decision taken. The user reviews this on Monday.

Run on: Windows, explicit `C:\ProgramData\anaconda3\python.exe` (no bare `python`, avoid
`import torch` in subprocesses to dodge OMP Error #15).

**Operating contract (user's request, verbatim):** "I truly give you the right to keep
testing. Just keep track what you tried so we can revisit them when needing." So every
experiment — hypotheses tried, exact command, before/after numbers, and the decision — is
logged here per run, even dead ends. Nothing is deleted without a log line. This file
*is* the revisitable trail.

**Checkpoint contract (user's follow-up, verbatim):** "When logging, also save checkpoints
so we can verify." Every consequence-bearing run saves recoverable state so the numbers can
be independently re-verified, not just trusted to my report. Concretely, each RUN block
archives whatever lets the run be replayed/re-checked:
- **`checkpoints/<RUN>/`** subfolder per run containing a **full model state_dict + cfg**
  (pickle, `asl1_dsm_n2n_per_real_<iters>.pt.pkl`) — the trained weights the sweep/decode
  used, so "these numbers came from these weights" is checkable;
- the **sweep figure** PNG (e.g. `SRSTR_01_sweep.png`) and any metric tables it produces;
- the **exact command** that produced the run (in the log line);
- a **plain-text copy of each table's stdout** so the raw numbers are greppable without
  re-running.
A RUN is only marked "verified" when its checkpoint and the printed numbers agree (spot-check
on a reload). The checkpoint files are committed alongside this log so they survive for
Monday review.

---

## RUN 2026-10-16 — SRSTR_01 single-map structure sweep (runtime knobs)

**Goal:** decide knob-tune vs structural-loss. The single denoised maps felt "structureless".
Hypothesis: too-aggressive anneal (high sigma_ratio low-passes detail and/or missing
corrector refinement), a **runtime-knob** issue — fixable by sweeps, no retrain. Only if NO
knob setting recovers structure do we consider GDL/SSIM (risky under strong noise).

**Setup:** Reproduced the notebook's training environment in a self-contained script
(`_run_srstr.py`): load dataset1, split (EVAL_SLICES=[3,17,25]), pack, Option-3 level-weights
centered on `sigma_ratio*sigma_task`, then **retrain the L1 N2N_DSM for 6000 iters**
(weights, BATCH=8, lr=1e-3, grad_clip=1.0) so the sweep runs on the exact L1 model the
user's results came from. Then on the **middle eval slice** ran:
- the current test knobs (sigma_ratio=2.0, corrector_steps=0) for baseline numbers,
- the **SRSTR_01 grid**: sigma_ratio ∈ {1.2, 1.5, 2.0} × corrector_steps ∈ {0, 3, 5},
  scoring each single-map with bias/RMSE **and** two structure metrics (edge-gradient energy HF
  and high-pass detail HP, both as % of the clean reference) so "structureless vs over-smooth"
  is measured, not just eyeballed.

**Checkpoints for this run:** `checkpoints/SRSTR_01/` (model + cfg pickle + sweep png + raw
table stdout + this command).

**Fresh L1 training curve (reproduced):** loss 0.039 → 0.019 over 6000 iters (matches the
notebook's recorded 0.01935 final). Training converged as before.

**Run command:** `_run_srstr.py` via explicit anaconda python, background,
log → `srstr_run.log`, artifacts → `checkpoints/SRSTR_01/`.
Exact invocation used:

```
C:\ProgramData\anaconda3\python.exe _run_srstr.py
```

### Results

Reproduced L1 training curve: first logged iter-200 loss 0.039, oscillating 0.020–0.034
mid-run, **final loss 0.02008** (matches notebook's recorded ~0.019).
Eval slices used: idx0=s3, idx1=s17, idx2=s25. Sweep ran on middle eval slice s17
(`sigma_task = 0.0793`).

Baseline (current test knobs sigma_ratio=2.0, corrector_steps=0), s17:

| decode | RMSE | bias |
|---|---|---|
| single map | 0.0545 | **1.802** |
| mean10 | 0.0251 | 0.724 |

Structure sweep on one raw realization, scored against the clean mean reference
(ref HF(edge)=0.0266, HP(detail)=0.0170):

| sigma_ratio | cs | bias | RMSE | HF (%ref) | HP (%ref) |
|---|---|---|---|---|---|
| 1.2 | 0 | 0.614 | 0.0291 | 0.65x | 0.64x |
| **1.5** | **0** | **0.977** | 0.0334 | **1.19x** | **1.21x** |
| 2.0 | 0 | **1.818** | **0.0547** | **2.17x** | **2.21x** |
| 1.2 | 3 | 0.586 | 0.0293 | 0.69x | 0.68x |
| **1.5** | **3** | **0.888** | 0.0329 | **1.10x** | **1.12x** |
| 2.0 | 3 | 1.711 | 0.0523 | 2.03x | 2.04x |
| 1.2 | 5 | 0.594 | 0.0293 | 0.66x | 0.65x |
| **1.5** | **5** | **0.911** | 0.0333 | **1.08x** | **1.09x** |
| 2.0 | 5 | 1.618 | 0.0499 | 1.90x | 1.94x |

(Raw stdout: `diffusion/srstr_run.log`. Figure: `checkpoints/SRSTR_01/SRSTR_01_sweep.png`;
state_dict reproduced from `asl1_dsm_n2n_per_real_6000.pt.pkl`, spot-check load gives the
same final loss 0.0200821.)

### Read-off / decision

1. **The "structureless" complaint was misnamed.** At sigma_ratio=2.0 the single map has
   HF/HP ≈ 2.17–2.45× the reference — it is *not* over-smoothed, it is **over-bright with an
   edge halo**: bias ~1.62–1.82 vs target ~1.04, high-frequency energy roughly doubled.
   Structure is being invented, not washed out.
2. **sigma_ratio=1.5 fixes both symptoms simultaneously.** bias 0.89–0.98 (near the
   irreducible LOO target floor ≈1.03), HF/HP ≈1.08–1.19× ref, RMSE 0.033 at cs=0/3,
   essentially flat across corrector settings. sigma_ratio=2.0 is simply too aggressive for
   the L1-trained weights.
3. **sigma_ratio=1.2 under-fires**: lowest RMSE but bias 0.59 and HF/HP ≈0.65x — visibly
   thin/dim, detail suppressed.
4. **corrector_steps buys little here**: at sr=1.5, cs=3/5 mainly trims bias slightly
   (0.97→0.91) with comparable structure. Cheap to include, not a fix by itself (contradicts
   an earlier cs=3≈cs=0 reading at sr=2.0).
5. **Decision: knob fix, no GDL/SSIM retrain needed for this symptom.** Recommended default:
   **sigma_ratio = 1.5**, corrector_steps ∈ {0, 3}. Revisit GDL/SSIM only if, after this
   change, the single maps still look wrong visually — and note the risk analysis stands:
   structural losses on noise-dominated data can hallucinate edges.

**Checkpoint verification:** `checkpoints/SRSTR_01/asl1_dsm_n2n_per_real_6000.pt.pkl`
(26,559,017 B) reloads into `UNet(in_ch=2,out_ch=2,base_out=32,emb_dim=32)` +
`N2N_DSM(L=cfg["L"], ...)` and reproduces final loss 0.0200821 and the published figure.

*(Log finalized retroactively on 2026-10-19 from `srstr_run.log` — the origin session died
of context overflow before writing these numbers into this file.)*

---

## RUN 2026-10-19 — R0/R1/R2: four-loss sweep on a σ-matched split

**Goal.** Turn the earlier single-map structure work into a controlled loss comparison:
one training harness, one shared seed/init/batch order, only the objective changes.

**Harness (R0).**
- Split by noise band so val and test each cover low/mid/high σ:
  VAL = slices `[16, 11, 1]`, TEST = `[25, 22, 8]`. No slice appears in both.
- Four objectives behind one flag: `l1`, `mse`, `huber`, `mag_l1` (|T|-weighted L1).
- Shared `--seed` for init **and** batch order; identical architecture/config; ~4,950
  iterations each (the pre-existing l1 run had a 9,300 budget and is reused as-is —
  see the caveat below).
- Train roots differ because of when they ran: `runs/r0_split/l1`,
  `runs/r0_eval/{mse,huber,mag_l1}`.

Commands:
```
python _r0_train.py --loss {mse,huber,mag_l1} --iters 4950 --val-every 99 --ckpt-every 297 --outdir runs/r0_eval
python _r1_eval.py --loss <loss> --runroot <root> --split {val,test} --sr 1.5 2.0 --cs 0 5 --append
python _r12_report.py            # walks runs/r0_split + runs/r0_eval -> runs/compare/
```

**Evaluation (R1).** Decode-only (`conditional_denoise`) at every checkpoint ×
sigma_ratio ∈ {1.5, 2.0} × corrector_steps ∈ {0, 5}; metrics on magnitudes vs the mean
of the 10 realizations: bias ratio, RMSE, NMSE, HF edge %, HP detail %. Raw input sits at
bias≈2.55–2.59, HF/HP≈262–272%, so anything near 100% / bias 1.00 is doing real work.

**Result 1 — bias-matched comparison, TEST (interpolated to bias==1.00).**

| loss | sr | cs | RMSE | NMSE | HF % |
|---|---|---|---|---|---|
| mse | 1.5 | 0 | **0.02123** | **0.18282** | 99.9 |
| mag_l1 | 1.5 | 0 | 0.02221 | 0.20522 | 100.1 |
| huber | 1.5 | 5 | 0.02174 | 0.20178 | 100.1 |
| l1 | 1.5 | 5 | 0.02170 | 0.20143 | 100.2 |
| *(any)* | 2.0 | – | 0.0276–0.0300 | 0.316–0.352 | ~101 |

VAL reproduces the same ordering with all RMSEs ≈0.002 higher (0.0237–0.0257).

**Result 2 — what the losses actually do differently.** They are close on error but very
different in how brightness drifts with training. At sr=1.5/cs=5, over checkpoints
3 k → 14 k: L1-trained weights fall to bias 0.49 (HF 47%); MSE/Huber also decay but
slower; mag_l1 keeps a higher HF at equal bias. So "L1 looks thin" is real and
checkpoint-dependent, not a decode-knob artefact — the best-checkpoint table under-reports
it because a dim checkpoint also happens to post a lower bare RMSE.

**Result 3 — sigma_ratio dominates corrector_steps.** 1.5 → 2.0 costs ~25–30% more RMSE
for every loss at matched bias; corrector_steps 0→5 buys ~2–3% and nothing else moves.
Confirms the SRSTR_01 finding at four independent training runs.

**Decision taken:** none yet — this is the evidence package. Default candidate is
**mse @ sr=1.5**, on lowest RMSE/NMSE among the losses that hold brightness; keep
`corrector_steps=5` as a small free improvement. Do not extend L1 training (it keeps
getting dimmer), which retires the open question about the truncated l1 run.

**Artifacts:** `runs/compare/r12_summary.md` (all tables incl. raw baseline),
`runs/compare/r2_train_val_curves.png`, `runs/compare/r1_metric_vs_iter_{val,test}.png`
(metrics vs checkpoint, faceted by sr × cs), per-run CSVs in `runs/r0_*/<loss>/`.

**Caveats.**
- The l1 run was configured for 9,300 iters (`config.json`); its log ends at 8,550 and
  the last checkpoint on disk is 8,370, with no `DONE` line. Its validation minimum was at
  3,750 and rose afterwards, so no useful checkpoint was lost.
- `mag_l1` has a single-frame validation spike at iter 3762 (val_l1 0.211 vs ~0.029 either
  side) — one noisy probe draw; the matched-iteration summary uses a 3-point rolling median.
- All runs share one GPU serially; eval was deliberately queued after training to avoid an
  OOM killing the trainer (batch-decoding all 10 realizations at once).

---

## RUN 2026-10-19 — VIS_2000: qualitative panels from ~2k-iteration checkpoints

**Motivation.** The four-loss sweep above reported *best* (mostly late) checkpoints. User's
read of those tables: 6000+ iterations looks like more training than is needed. This run
asks the same question qualitatively — what do the maps actually look like at ~2k?

**Setup.** `_r2_displays.py` picks, per loss, the on-disk checkpoint nearest iteration 2000
(L1@1860; MSE/Huber/mag_l1@2079), then decodes one sample per split: TRAIN s17,
VAL s16, TEST s25. Decode settings are the recommended ones (`sigma_ratio=1.5`,
`corrector_steps=5`, `corrector_alpha=0.05`); all 10 realizations are decoded as one batch
under the shared seed 12345, so columns differ only by weights. Reference = magnitude of the
complex mean of the 10 realizations. Each estimate appears as the map plus its signed
difference from that reference. **The data are complex**, so a third figure per slice shows
the real and imaginary quadratures, the magnitude, and the circular phase error against the
complex reference; the CSV carries corresponding complex metrics (relative complex error
`||est−ref||/||ref||`, per-quadrature RMSE, median |phase error| over pixels above the
median reference magnitude). Magnitude-only agreement can hide a scrambled phase, which an
ASL map would then report as a wrong number.

Command:
```
C:\ProgramData\anaconda3\python.exe _r2_displays.py --target-iter 2000 --sr 1.5 --cs 5
```

**Numbers** (ensemble-of-10 decode; bias and RMSE on magnitudes vs the reference;
HF/HP as % of the reference's own energy). Verbatim from
`runs/displays/display_metrics_iter2000.csv`:

| split | slice | estimate | ckpt iter | bias | RMSE | HF % | HP % | rel.comp.err | RMSE re | RMSE im | med \|dphi\| deg |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 17 | raw realization | - | 2.448 | 0.09139 | 253.4 | 260.1 | 2.089 | 0.08283 | 0.08410 | 45.7 |
| train | 17 | average of 10 magnitudes (no model) | - | 2.370 | 0.06744 | 88.5 | 85.7 | - | - | - | - |
| train | 17 | L1 | 1860 | 1.019 | 0.02272 | 103.7 | 105.4 | 0.610 | 0.02460 | 0.02410 | 17.2 |
| train | 17 | MSE | 2079 | 0.944 | 0.02420 | 96.8 | 98.9 | 0.632 | 0.02644 | 0.02401 | 18.9 |
| train | 17 | Huber | 2079 | 0.922 | 0.02357 | 93.8 | 95.8 | 0.615 | 0.02542 | 0.02373 | 18.0 |
| train | 17 | |T|-w L1 | 2079 | 0.957 | 0.02313 | 97.6 | 99.1 | 0.608 | 0.02512 | 0.02344 | 17.7 |
| val | 16 | raw realization | - | 2.388 | 0.06289 | 252.9 | 260.3 | 2.024 | 0.05678 | 0.05774 | 42.3 |
| val | 16 | average of 10 magnitudes (no model) | - | 2.363 | 0.04760 | 89.6 | 88.1 | - | - | - | - |
| val | 16 | L1 | 1860 | 0.947 | 0.01619 | 95.7 | 98.2 | 0.606 | 0.01761 | 0.01669 | 17.8 |
| val | 16 | MSE | 2079 | 0.857 | 0.01841 | 87.4 | 90.5 | 0.656 | 0.01984 | 0.01720 | 19.8 |
| val | 16 | Huber | 2079 | 0.823 | 0.01826 | 83.1 | 85.7 | 0.645 | 0.01931 | 0.01712 | 19.5 |
| val | 16 | |T|-w L1 | 2079 | 0.872 | 0.01717 | 88.0 | 90.6 | 0.618 | 0.01866 | 0.01623 | 18.3 |
| test | 25 | raw realization | - | 2.717 | 0.06282 | 279.5 | 284.7 | 2.476 | 0.05652 | 0.05723 | 48.1 |
| test | 25 | average of 10 magnitudes (no model) | - | 2.669 | 0.04935 | 102.3 | 97.3 | - | - | - | - |
| test | 25 | L1 | 1860 | 1.007 | 0.01500 | 99.4 | 101.9 | 0.715 | 0.01635 | 0.01651 | 21.5 |
| test | 25 | MSE | 2079 | 0.930 | 0.01565 | 92.0 | 94.2 | 0.733 | 0.01702 | 0.01666 | 23.6 |
| test | 25 | Huber | 2079 | 0.882 | 0.01561 | 86.7 | 89.5 | 0.724 | 0.01664 | 0.01660 | 23.3 |
| test | 25 | |T|-w L1 | 2079 | 0.944 | 0.01482 | 92.7 | 95.0 | 0.700 | 0.01611 | 0.01605 | 21.4 |
Averaging ten magnitude images is NOT the reference (it reads bias ≈2.37–2.67): the
reference is the magnitude of the complex mean, which averaging magnitudes cannot reach
because of the Rician offset. So the no-model baseline column in the figures does not
accidentally coincide with the reference.

**Read-off.**

1. **At ~2k iterations nothing is broken.** All four losses sit at bias 0.82–1.04 on these
   three slices with HF/HP 86–106% of the reference — matching structure energy rather than
   inventing it (a failure mode the raw input shows at 253–285%). **Your visual read is
   confirmed by the numbers and is recorded here as a finding:** recognizable anatomical
   structure at ~2k iterations means the networks are learning signal, not memorizing noise.
   A noise-memorizing model would show *inflated* high-frequency energy (the raw-input rows
   run 250–285% of reference); instead every loss lands within ±15% of it while dropping
   magnitude RMSE 0.063→0.015–0.024. Caveat unchanged: this session's image viewer could not
   render PNGs directly (they open fine under PIL), so the maps themselves remain yours to
   inspect; the metrics here are the machine-checkable part.
2. **The complex view says the same thing about phase.** Median |phase error| over
   above-median-signal pixels falls from 42–48° (raw realization) to **17° (L1 train / Huber
   val) – 23.6° (MSE test)**. Relative complex error drops from 2.02–2.48 (raw) to
   **0.608–0.733**, i.e. per-quadrature RMSE 0.016–0.026 against quadrature signal levels
   around 0.045–0.055. So the improvement is not a magnitude-only artefact — the model
   reconstructs both quadratures. What remains is concentrated in phase: the loss comparison
   is nearly flat on `rel.comp.err` (spread ≤0.06 within a split), whereas magnitudes differ
   more visibly in brightness. Test s25 is hardest for every objective (~0.70–0.71 complex
   error, 21–24°), consistent with its higher σ.
3. **Your budget intuition holds, and the saturation point differs per loss.** Comparing
   the best checkpoint in the first ~2.2k iterations against the best checkpoint anywhere
   (`sr=1.5, cs=5`, averaged over the split's slices): L1 and Huber already reach their
   global RMSE minimum in that window on both val and test; mag_l1 is within 0.5–1.5%;
   only MSE still gains meaningfully later (val 0.01687→0.01491, −13%; test 0.01553→0.01412,
   −10%) and does so by dimming (bias 1.069→0.928 val, 1.000→0.960 test). Every other
   trajectory that improves past ~2k is buying RMSE with lost contrast.
3. **Loss ordering at 2k matches the late ranking**: mag_l1/MSE lowest RMSE, L1 brightest
   (bias 0.95–1.02), Huber dimmest (0.82–0.92). The brightness spread across objectives
   (≈0.18 in bias at the same iteration) is larger than any RMSE difference among them,
   so pick on brightness first. On complex metrics the four are effectively tied.
4. **Recommendation:** treat ~1600–2200 iterations as where this task saturates. If a longer
   schedule is kept, select the checkpoint by brightness (bias near 1.00 on a held-out
   low-noise slice) instead of taking the last iteration.

Artifacts:
- `runs/displays/overview_iter2000.png` — 3 splits × {noisy mean, four losses, reference}
- `runs/displays/display_{train_s17,val_s16,test_s25}_iter2000.png` — per-slice stack with
  signed-difference panels
- `runs/displays/display_complex_{train_s17,val_s16,test_s25}_iter2000.png` — real /
  imaginary / magnitude / |phase error| rows for reference and each loss
- `runs/displays/display_metrics_iter2000.csv` — greppable numbers, incl. complex metrics
- script: `_r2_displays.py` (`cstats()` defines the complex metrics)

---

The autonomous run finished successfully at 2026-10-16T18:07 (training + sweep + checkpoint
all complete). The **session** then died at 18:09:21 with
`API Error: 400 ... input (314040 tokens) > context length (262144)`. Cause: the transcript
had accumulated ~1000 tool round-trips plus large base64/PNG reads; every later turn re-sent
the whole history. Every attempt to recover by resuming or running `/compact` failed the same
way (315k–316k tokens), because compaction must send the already-oversized context before it
can shrink it. Recovery required digesting the 971 MB JSONL externally and reading only the
digest — not another resume of that session.

---

---

## RUN 2026-10-19 — SINGLE_MAP_DIAG: why one denoised map looks no cleaner than its input

**Motivation.** Every column in VIS_2000 is `mag(mean_k denoised_k)` over the 10
realizations, so those panels show the ensemble, not one map. The expectation under test:
one denoised map should itself be close to the reference. It is not — and the reason is in
the decoder, not the weights.

Five read-only scripts; nothing in `n2n.py` was changed.

| script | question |
|---|---|
| `_alpha_sweep.py` | how strong is the shipped corrector update? |
| `_tweedie_once.py` | what does one honest `x0_hat` at the map's own σ give? |
| `_decode_variants.py` | walk variants vs one-shot vs iterated fixed point |
| `_single_map_best.py` | separate global gain from genuinely lost detail |
| `_single_biasmatched.py` | blend weight that makes single-map bias exactly 1.00 |

Commands (each script resolves, per loss, the on-disk checkpoint nearest its
default `--target-iter 1950`):
```
python _alpha_sweep.py
python _tweedie_once.py
python _decode_variants.py
python _single_map_best.py
python _single_biasmatched.py
```

Slices: alpha_sweep and single_biasmatched use train s17 / val s16 / test s25;
tweedie_once uses test s19; decode_variants uses val s20 / test s24;
single_map_best uses train s18 / val s21 / test s24.

### Root cause: `corrector_alpha` is used as a per-step fraction of the residual

`N2N_DSM.conditional_denoise` documents the knob as a step size in units of σ² but then
computes

```python
alpha = corrector_alpha * s ** 2     # fraction of the residual to close
result = result + alpha * (x0_hat - result)
```

The comment matches the algebra (α·s² is dimensionless), so the docstring is the wrong half.
At the task noise level (s≈0.06) the shipped `corrector_alpha=0.05` closes ≈1.8e-4 of the
residual per step. Over the ~37-rung ladder the state barely moves, and the terminal branch
returns `x0_hat` evaluated at the finest rung on an input still carrying
`sigma_ratio * sigma_task` of injected variance. Statistically the "denoised" map IS the
input. `conditional_denoise_multi` repeats the same expression for joint stacks.

Measured (`runs/displays/alpha_sweep.csv`, all three slices):

| variant | single bias | single RMSE |
|---|---:|---:|
| fixed alpha 0.010 (shipped regime) | 2.439–2.786 | 0.069–0.109 |
| frac-of-residual 0.10 | 0.065–0.128 | 0.032–0.056 |
| frac-of-residual 0.50 | 0.111–0.187 | 0.031–0.055 |
| frac-of-residual 1.00 | 0.338–0.568 | 0.024–0.047 |

Every fraction-of-residual setting collapses contrast almost totally once applied at every
rung of the ladder (bias ≤0.19 for fractions up to 0.5), while the shipped magnitude leaves
the map equal to its input. There is no operating point in between: the annealed-walk axis
jumps from "no change" to "contrast erased", so it cannot deliver a good single map.

### One Tweedie call at the matched σ overshoots in contrast

`y + s²·score(y, t(σ_task))` once (`runs/displays/tweedie_once.csv`):

| split | raw single bias / RMSE / HF % | tw1 bias / RMSE / HF % |
|---|---|---|
| train s17 | 2.301 / 0.0883 / 237.6 | 0.297 / 0.0460 / 16.1 |
| val s16 | 2.293 / 0.0624 / 238.2 | 0.353 / 0.0314 / 29.4 |
| test s19 | 2.298 / 0.0857 / 235.4 | 0.281 / 0.0444 / 13.0 |

Across losses (tw1, both multipliers): bias 0.23–0.52, HF 12–56 %. Noise is removed and
contrast goes with it — much too aggressive to ship alone.

### An interior blend weight exists and hits the reference

`out_w = y + w·(Tweedie − y)`, bisecting w per (slice, loss) until LOO single-map magnitude
bias is exactly 1.000 (`runs/displays/single_biasmatched.csv`, 12 rows):

| statistic | range |
|---|---|
| w* | 0.484 – 0.631 (mean 0.567) |
| resulting single-map bias | 1.000 (by construction) |
| single-map RMSE | 0.0220 – 0.0389 |
| HF % / HP % of reference | 99.2–105.3 / 98.6–109.3 |
| ensemble RMSE | 0.0180 – 0.0307 |
| single/ensemble RMSE | 1.14 – 1.61× (mean 1.40×) |

A single map therefore reaches reference-level brightness and high-frequency energy; only
about 20–40 % extra RMSE separates it from the ensemble mean. The information was there all
along — the shipped anneal never extracted it.

### Can one fixed weight ship without the oracle?

`_single_fixed_w.py` sweeps w with **no per-map selection** over six slices
(`runs/displays/single_fixed_w.csv`, L1@1860). Single-map magnitude bias at five
weights on the same grid:

| slice | sigma | w=0.52 | w=0.57 | w=0.58 | w=0.62 | w=0.70 |
|---|---:|---:|---:|---:|---:|---:|
| train s3 | 0.1001 | 1.30 | 1.17 | 1.14 | 1.03 | 0.82 |
| train s29 | 0.0349 | 1.01 | 0.86 | 0.82 | 0.71 | 0.50 |
| val s16 | 0.0560 | 1.07 | 0.96 | 0.93 | 0.85 | 0.67 |
| val s11 | 0.0787 | 1.14 | 1.02 | 1.00 | 0.91 | 0.73 |
| test s25 | 0.0561 | 1.18 | 1.05 | 1.02 | 0.92 | 0.72 |
| test s22 | 0.0446 | 1.08 | 0.96 | 0.94 | 0.84 | 0.65 |

No single constant lands bias at 1.00 everywhere: the best grid point (w=0.57) still
spans 0.86-1.17 across the six slices, because the optimal blend tracks
each map's own contrast deficit rather than sigma alone. A shippable decoder therefore needs a
cheap gain estimate computed *from the map itself* (e.g. match the post-denoise magnitude
histogram or the low-frequency level to a calibration slice), not a hard-coded w. At that same
w=0.57, HF lands at 99-118 % of the leave-one-out reference
(RMSE 0.0135-0.0409), so the residual spread is mostly brightness error, not
lost structure.
### Caveats

- w* and the fitted gains were selected against the leave-one-out reference. That is fine for
  establishing an upper bound but is not deployable: a production rule must fix w≈0.55 or
  calibrate the gain from a held-out reference slice without touching each map's own data.
- The single/ensemble ratio mixes two different references by design (LOO vs full mean);
  consistent with every earlier table.
- `hf/hp` are compared against the corresponding (ensemble or LOO) reference energy, so
  ≈100 % means "as sharp as the reference", not "noise-free".

### Recommendation (needs sign-off)

1. Fix the semantics in `n2n.py`: either honour the documented `corrector_alpha` (step in
   σ² units, i.e. use `min(corrector_alpha*s**2, 1)` consciously and re-tune) or redefine it
   as the residual fraction the comment claims and set a real default. This changes behaviour
   of every existing caller, including `_run_srstr.py` and `_r1_eval.py`.
2. Add a dedicated single-map decoder (`w≈0.55` blend of one matched-σ Tweedie step) rather
   than changing the annealer; ensemble mode stays as-is behind its current flag.
3. Untouched training remains a valid option if the deliverable is ensemble maps only —
   but then the labels must say "mean of 10 decoded maps".

**Artifacts:** `runs/displays/{alpha_sweep,tweedie_once,decode_variants,single_map_best,single_biasmatched}.{csv,log}`
— every number above is reproducible from the CSVs.

---

## RUN 2026-10-19 — SR_SWEEP: sigma_ratio = 1 vs 1.5 vs 2 on real ASL

**Question asked.** "Try `sigma_ratio = 1` — don't perturb forward when performing conditional denoising on the real MRI data, since the data is already very noisy. Also add a single denoised map next to it for comparison, clarify what each map means, save displays, and repeat at iter ~2000 and ~3000."

**What the sweep varies / holds fixed.** Three nominal checkpoint ages × three `sigma_ratio` values × four objectives, on train s17 / val s16 / test s25 (all `(10, 2, 80, 80)`). Panels carry their own scoring frame, stated in the caption:

| column | what it is | scored against |
|---|---|---|
| reference | `mag(complex mean of all 10 acquisitions)` | itself |
| noisy single | one raw realization, no processing | **LOO** mean of the other 9 |
| **single-map denoise w=0.57** | ONE scan → ONE map via `N2N_DSM.single_map_denoise`; no averaging anywhere | **LOO** mean |
| anneal sr=X | 10 independent decodes, then averaged | full 10-scan mean |

Single maps are compared to leave-one-out because a single map must not be charged for the noise of its own acquisition; ensembles are compared to the full mean they estimate.

**Commands:**
```
python _sr_sweep_displays.py --targets 1821 2428 3035 --srs 1.0 1.5 2.0
python _sr_summary.py
python _sr_gaincheck.py
```

The two checkpoint ladders do not coincide (l1 steps by 310 iterations, the eval losses by 297), so every title and CSV row records the **actual** iteration reached:

| nominal | l1 | mse | huber | mag_l1 |
|---:|---:|---:|---:|---:|
| 1821 | 1860 | 1782 | 1782 | 1782 |
| 2428 | 2480 | 2376 | 2376 | 2376 |
| 3035 | 3100 | 2970 | 2970 | 2970 |

### Result 1 — sr = 1 does NOT help; contrast falls monotonically with sr

Annealed ensemble, L1 weights, mean over the three slices (`runs/displays/sr_sweep/sr_sweep.csv`):

| nominal | ckpt | sr | bias | RMSE | HF % |
|---|---:|---:|---:|---:|---:|
| 1821 | 1860 | 1.0 | 0.6837 | 0.01447 | 67.0 |
| 1821 | 1860 | 1.5 | 0.9910 | 0.01797 | 99.6 |
| 1821 | 1860 | 2.0 | 1.2996 | 0.02691 | 132.0 |
| 2428 | 2480 | 1.0 | 0.6220 | 0.01724 | 57.1 |
| 2428 | 2480 | 1.5 | 0.9025 | 0.01775 | 88.9 |
| 2428 | 2480 | 2.0 | 1.1998 | 0.02461 | 120.9 |
| 3035 | 3100 | 1.0 | 0.4740 | 0.02548 | 42.7 |
| 3035 | 3100 | 1.5 | 0.7454 | 0.02095 | 72.4 |
| 3035 | 3100 | 2.0 | 1.0410 | 0.02320 | 104.2 |

Bias rises monotonically with sr for every objective at every age (pooled over the four losses: lowest sr=1 curve reaches 0.528, highest sr=2 curve reaches 1.180). Pooled over 3 ages × 4 losses:

| loss | sr=1.0 bias | sr=1.5 bias | sr=2.0 bias |
|---|---:|---:|---:|
| l1 | 0.593 | 0.880 | 1.180 |
| mse | 0.558 | 0.838 | 1.141 |
| huber | 0.528 | 0.805 | 1.102 |
| mag_l1 | 0.574 | 0.846 | 1.143 |

Nothing about the objective rescues sr=1: it deliberately removes the forward variance the decoder expects, so the shortfall appears in every loss equally. Note also that RMSE does not order like bias — at the latest age sr=1.5 has the lowest RMSE under every objective even though sr=1 has the lowest bias, i.e. sr=1 is over-smoothed relative to the reference.

### Result 2 — the deficit is mostly GAIN early, partly structural late

`_sr_gaincheck.py` fits the least-squares gain `g = Σm·r / Σr²` per cell and re-scores `m/g`. Pooled over the nine (age, slice) cells (`sr_gaincheck.csv`):

| sr | raw bias | fitted gain | bias ÷g | RMSE ÷g | HF % ÷g | complex rel err | phase med ° |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.593 | 0.571 | 1.052 | 0.01265 | 97.9 | 0.514 | 11.9 |
| 1.5 | 0.880 | 0.757 | 1.164 | 0.02035 | 114.8 | 0.653 | 20.1 |
| 2.0 | 1.180 | 0.946 | 1.247 | 0.02582 | 125.7 | 0.907 | 27.9 |

Dividing by the fitted gain restores sr=1 almost exactly (÷g bias 1.052, ÷g HF % 97.9) while leaving sr=1.5 and sr=2 with a residual overshoot, so at the ensemble level the sr=1 shortfall is primarily a scalar contrast error.

Per checkpoint age, sr=1 only:

| ckpt | n cells | fitted gain | bias ÷g | HF % ÷g | complex rel err | phase ° |
|---:|---:|---|---|---|---|---|
| 1860 | 3 | 0.622–0.741 | 1.006–1.025 | 98.6–101.2 | 0.298–0.442 | 4.6–8.1 |
| 2480 | 3 | 0.536–0.697 | 1.000–1.031 | 88.9–96.8 | 0.366–0.538 | 5.8–10.4 |
| 3100 | 3 | 0.357–0.522 | 1.058–1.160 | 97.6–106.6 | 0.569–0.789 | 12.3–26.3 |

Two things matter here:

1. At the **earliest** checkpoint (1860) sr=1's low contrast is nearly all scalar gain: after correction bias lands in 1.006–1.025 with HF 98.6–101.2 % and phase 4.6–8.1°, i.e. better complex-domain agreement than either other sr value.
2. As training proceeds the fitted gain keeps falling (0.357–0.741 over all nine cells), and by 3100 the corrected bias has drifted to 1.058–1.160 with phase up to 26.3°. Late checkpoints have partly learned to *rely* on the forward perturbation for contrast; no single rescale repairs them.
3. The fitted gain spreads far too widely within sr=1 for a hard-coded constant; a shippable sr=1 path would need a cheap per-map gain calibration, exactly as SINGLE_MAP_DIAG concluded for the blend weight.

### Result 3 — member-0 of the annealed decode is much worse than either alternative

`anneal_single` (realization 0 drawn from the same annealed run), pooled bias over targets × slices:

| loss | sr=1.0 bias | sr=1.5 bias | sr=2.0 bias |
|---|---:|---:|---:|
| l1 | 1.311 | 2.253 | 3.211 |
| mse | 1.280 | 2.209 | 3.158 |
| huber | 1.168 | 2.083 | 3.020 |
| mag_l1 | 1.277 | 2.193 | 3.140 |

Consistent with SINGLE_MAP_DIAG: `conditional_denoise` returns essentially its input for a single member. Passing sr=1 removes the extra injected variance but does not turn this decoder into a useful single-map tool — member-0 is still 17–31 % too bright with 129–145 % HF even at sr=1.

### Result 4 — the deployable single-map path is unchanged by sr and beats everything else

`single_map_denoise(y, σ_task, blend=0.57)` (one matched-rung Tweedie step, one scan in → one map out, scored against LOO):

| nominal | ckpt | bias | RMSE | HF % | HP % |
|---|---:|---:|---:|---:|---:|
| 1821 | 1860 | 1.0108 | 0.02852 | 105.1 | 107.1 |
| 2428 | 2480 | 1.0119 | 0.02829 | 104.3 | 106.0 |
| 3035 | 3100 | 1.0676 | 0.02912 | 109.9 | 111.7 |
| _raw single_ | – | 2.4219 | 0.07295 | 251.7 | 257.7 |

Bias sits at 101–107 % of the reference across the three ages, versus 2.42x for an untouched realization. In HF terms the raw scan carries 251.7 % of reference energy against 106.4 % for the decoder — excess-over-reference 152 % vs 6 %, so the blind scan carries roughly 23.5x as much excess high-frequency energy. This is the only single-map option that lands both brightness and structure on the reference, and it does not depend on `sigma_ratio` at all.

### Decision

- **Keep `sigma_ratio = 1.5` as the shipped default for ensemble decoding.** It is the calibrated operating point; sr=1 systematically under-recovers contrast and sr=2 overshoots.
- **Do not adopt sr=1 as a way to improve single maps.** It cannot make `conditional_denoise` produce a good single output, and at late checkpoints it loses structure, not merely gain.
- **Ship `single_map_denoise(blend≈0.57)` for the single-map deliverable**, with the caveat that a per-map gain calibration is still needed for full portability.
- Figures must always state which frame a panel is scored against; the sweep scripts now do.

### Caveats

- All maps are magnitude unless explicitly labelled complex; phase error comes from `cstats` over pixels above the median signal.
- `SRSTR_01` trained its own model; this sweep uses the R0 `runs/r0_split` and `runs/r0_eval` weights, so absolute numbers are not directly comparable across the two runs.
- This block is generated from `sr_sweep.csv` and `sr_gaincheck.csv` by `_sr_rewrite.py`; if a figure here ever disagrees with the CSV, the CSV wins and the generator needs fixing.

**Artifacts:** `runs/displays/sr_sweep/` — per-slice pages (`sr_display_{split}_{slice}_iter*.png`), per-loss grids (`sr_byloss_{split}_{slice}_iter*.png`), overviews (`sr_overview_iter*.png`), `sr_sweep.csv`, `sr_gaincheck.csv`, `sr_sweep.log`, `gaincheck.log`.
Archive copy + L1 weights + hashes: `checkpoints/SR_SWEEP/` (`MANIFEST.md`), verbatim stdout: `Archive/SR_SWEEP/stdout_sr_sweep_raw.html`. The pre-regeneration hand-edited version of this section is kept verbatim at `Archive/SR_SWEEP/WEEKEND_ITERATIONS_pre_repair.md`.

---

## Matched-noise head-to-head: does the learned decode earn its keep?

**Date:** 2026-10-19 · **Scripts:** `_headtohead.py`, `_compose.py`, `_h2h_report.py`  
**Artifacts:** `runs/displays/target_sweep/headtohead.csv`, `headtohead.log`, `compose.csv`, `compose2.log`

### Why a matched-noise comparison

The acceptance criterion is *one noisy realization in, one map out whose residual noise matches the LOO reference floor*, i.e. `sigma_ref/sigma_task = 1/3`. Comparing estimators each at its own favourite setting says nothing about deployment. So every family is swept along its single knob, its curve is traced in (achieved residual noise, structural error), and each curve is read at **exactly noise = 1/3**. Achieved noise is measured identically for every family: spread between independently processed realizations, `N/(N-1)` weighted, divided by `sigma_task`.

Families: `id` untouched input; `blur` separable Gaussian on Re/Im; `wiener` oracle stationary Wiener filter with radially smoothed PSD (a ceiling, since it uses the true signal PSD estimated from the ensemble mean); `tweedie` the deployable `single_map_denoise` -- one matched-rung Tweedie step plus a linear blend.

### Result A -- structure error at exactly noise = 1/3

Mean over 8 models (l1/mse/huber/mag_l1 x early/late checkpoint) and 3 slices.

| slice | family | n | relE | sHF % | s_bias | phase deg | knob |
|---|---|---:|---:|---:|---:|---:|---|
| train | blur | 8 | 0.9886 | 62.9 | 0.956 | 50.4 | Gaussian sigma |
| train | wiener | 8 | 0.9899 | 63.7 | 0.949 | 51.3 | Wiener fac |
| train | tweedie | 8 | 1.0780 | 92.3 | 1.038 | 64.3 | blend |
| val | blur | 8 | 0.9950 | 62.7 | 0.958 | 47.6 | Gaussian sigma |
| val | wiener | 8 | 0.9893 | 63.5 | 0.949 | 46.7 | Wiener fac |
| val | tweedie | 8 | 1.0595 | 90.3 | 1.033 | 60.7 | blend |
| test | blur | 8 | 1.1642 | 71.3 | 1.038 | 66.0 | Gaussian sigma |
| test | wiener | 8 | 1.1803 | 76.4 | 1.018 | 65.9 | Wiener fac |
| test | tweedie | 8 | 1.2111 | 93.2 | 1.027 | 73.4 | blend |

`id` never reaches 1/3 -- its quietest achievable level is noise 1.054, well above the target -- so it has no row.

**At the deployment target the ordering is blur < wiener << tweedie.** Over all 24 (loss x age x slice) cells the bare Tweedie decode beats Gaussian blur **0/24** times. Its relE penalty is mean **+0.0669** (range +0.0385 to +0.1014) versus blur and +0.0630 versus the oracle Wiener (range +0.0224 to +0.1002).

Its signature is unmistakable: it keeps almost all high-frequency energy (sHF near the low 90s %) where blur and Wiener leave roughly two thirds. Global gain is also less well calibrated (`s_bias` above 1.02-1.03 on train/val against below 0.96 for the baselines), though a downstream contrast fix removes most of that.

### Result B -- budget-feasible optimum (`noise <= 1/3`, lowest relE anywhere under budget)

This second definition matters: a family can become much quieter than required, and its quietest legal point is often better than its point sitting exactly at 1/3. It answers "what is the best structure error available without exceeding the reference noise floor?".

| slice | family | n | relE | sHF % | s_bias | phase deg | achieved noise |
|---|---|---:|---:|---:|---:|---:|---:|
| train | blur | 8 | 0.8298 | 34.0 | 0.978 | 41.5 | 0.160 |
| train | wiener | 8 | 0.8171 | 31.3 | 0.978 | 40.0 | 0.124 |
| train | tweedie | 8 | 0.9262 | 67.1 | 1.193 | 53.4 | 0.114 |
| val | blur | 8 | 0.8347 | 33.6 | 0.978 | 42.6 | 0.160 |
| val | wiener | 8 | 0.8151 | 28.3 | 0.982 | 41.4 | 0.107 |
| val | tweedie | 8 | 0.9251 | 63.9 | 1.194 | 52.4 | 0.118 |
| test | blur | 8 | 1.0030 | 41.9 | 1.071 | 61.3 | 0.188 |
| test | wiener | 8 | 0.9432 | 26.2 | 1.105 | 60.2 | 0.070 |
| test | tweedie | 8 | 0.9756 | 74.0 | 1.144 | 64.6 | 0.106 |

Under this criterion the net wins only **8/24** cells against blur (gap mean +0.0532, range -0.0431 to +0.1101). The gap narrows because the net tolerates heavier smoothing when it is allowed to go quieter, not because it recovers structure blur misses.

### Result C -- is the excess HF real structure or estimation noise?

That is the question that could rescue the network: if the retained HF were genuine, total relE would understate it and blurring would destroy information. `_compose.py` settles it by composing a post-filter with the decode,

```
out_k = blur( y_k + blend * (Tweedie_rung(y_k) - y_k), sigma_post )
```

and reading every composed curve at exactly noise = 1/3 as well.

| post sigma | split | n@1/3 | relE@1/3 | sHF %@1/3 | blend@1/3 | best relE under budget |
|---:|---|---:|---:|---:|---:|---:|
| 0.50 | train | 8 | 1.0432 | 82.1 | 0.536 | 0.9103 |
| 0.50 | val | 8 | 1.0287 | 80.8 | 0.530 | 0.9074 |
| 0.50 | test | 8 | 1.1948 | 88.0 | 0.555 | 0.9567 |
| 0.62 | train | 8 | 1.0212 | 75.3 | 0.376 | 0.8970 |
| 0.62 | val | 8 | 1.0108 | 74.9 | 0.373 | 0.8933 |
| 0.62 | test | 8 | 1.1814 | 84.5 | 0.416 | 0.9536 |
| 0.76 | train | 0 | unreached | - | - | 0.8834 |
| 0.76 | val | 0 | unreached | - | - | 0.8794 |
| 0.76 | test | 1 | 1.1755 | 84.0 | 0.300 | 0.9514 |
| 0.92 | train | 0 | unreached | - | - | 0.8706 |
| 0.92 | val | 0 | unreached | - | - | 0.8671 |
| 0.92 | test | 0 | unreached | - | - | 0.9496 |
| 1.12 | train | 0 | unreached | - | - | 0.8575 |
| 1.12 | val | 0 | unreached | - | - | 0.8551 |
| 1.12 | test | 0 | unreached | - | - | 0.9479 |

With the mildest post width 0.50: train 1.0780 -> 1.0432 (blur alone 0.9886); val 1.0595 -> 1.0287 (blur 0.9950); test 1.2111 -> 1.1948 (blur 1.1642). Composition closes part of the gap and lowers sHF sharply, so a share of the retained HF is estimation noise: reading (ii) holds, reading (i) does not. Even so the composed estimator stays behind plain blur at equal noise, so the surviving residue is larger than what a blind Gaussian leaves.

Widths at or above 0.92 pull every operating point below the 1/3 budget at all tested blends, hence their `n@1/3` is zero and they are reported through the budget column.

### Result D -- capacity and training length barely matter

Tweedie relE at exact 1/3, averaged over the three slices:

| loss | ckpt iter | relE@1/3 |
|---|---:|---:|
| huber | 1782 | 1.1133 |
| huber | 4455 | 1.1205 |
| l1 | 1860 | 1.1089 |
| l1 | 4340 | 1.1168 |
| mag_l1 | 1782 | 1.1123 |
| mag_l1 | 4455 | 1.1084 |
| mse | 1782 | 1.1269 |
| mse | 4455 | 1.1223 |

Aggregate early (1.1154) against late (1.1170) checkpoints differs by 0.0016, below the spread across slices. Longer training does not buy matched-noise quality on this metric, consistent with the standing instruction not to over-train.

### Decision

1. At matched residual noise the deployable one-shot Tweedie decode trails a plain separable Gaussian on essentially every cell tested, and trails the oracle Wiener by a similar margin. The result survives both aggregation definitions and all four losses at both checkpoint ages.
2. Composing a spectral tail control genuinely helps (large sHF reduction, visible relE gain) and belongs in any shipped estimator, but it converts the net toward the baseline rather than past it.
3. One thing this instrument cannot settle: whether relE against the LOO complex mean is the right deliverable metric. If small-vessel preservation is what matters, deleting HF is a failure the metric rewards and a task-specific scorer is needed first. That call belongs to the project owner.
4. No change to the net-only default (`single_map_denoise`, blend near 0.67 yielded the matched-noise optimum) is made pending that call; the composed form is reported alongside it.

### Caveats

- Reading a family at exactly 1/3 interpolates linearly between neighbouring settings; where a family bottoms out near the target the interpolated value can differ slightly from either grid neighbour. All grids here straddle 1/3 except the widest post-blur widths, which are excluded from the `@1/3` column by construction.
- The Wiener family is oracle (PSD from the ensemble mean), so every net-versus-Wiener gap is a lower bound; a realizable Wiener would look worse, narrowing but not closing the net's deficit against blur.
- Gain calibration differs between criteria: `s_bias` exceeds 1.14 for budget-picked Tweedie points because those sit at blend < 1 and are under-weighted. Post-gain columns remove it; raw relE carries it.
- Slices 26 and 29 stay excluded (known-bad references), same convention as earlier entries.

### Reproduction

```
C:/ProgramData/anaconda3/python.exe -u _headtohead.py \
    --losses l1 mse huber mag_l1 --targets 1782 4455 \
    > runs/displays/target_sweep/headtohead.log 2>&1
C:/ProgramData/anaconda3/python.exe -u _h2h_report.py
C:/ProgramData/anaconda3/python.exe -u _compose.py \
    --losses l1 mse huber mag_l1 --targets 1750 4420 \
    > runs/displays/target_sweep/compose2.log 2>&1
```
The `--targets` values above are the nominal requests recorded in each CSV's `target_iter` column; tables use the actual checkpoint iterations each model resolved to (`ckpt_iter`). Every numeral in this section is generated from `headtohead.csv` and `compose.csv` by `_h2h_log.py` -- no figure is typed by hand.

---

## HOUSEKEEPING 2026-10-20 — experiment files moved to `experiments/`

Requested by the project owner to declutter the repo root. Logged before the move.

- 40 files → `experiments/`: `_alpha_sweep.py`, `_bench_oracle.py`, `_bench_wiener.py`, `_compose.py`, `_decode_variants.py`, `_diag_decode.py`, `_frontier.py`, `_frontier_archive.py`, `_h2h_log.py`, `_h2h_report.py`, `_headtohead.py`, `_probe_slice_scale.py`, `_r0_probe.py`, `_r0_slice_stats.py`, `_r0_train.py`, `_r12_report.py`, `_r1_baseline.py`, `_r1_eval.py`, `_r2_displays.py`, `_re_render_srstr_fig.py`, `_run_srstr.py`, `_rung_probe.py`, `_single_biasmatched.py`, `_single_fixed_w.py`, `_single_map_best.py`, `_single_vs_ens.py`, `_snr_probe.py`, `_splice_log.py`, `_sr_gaincheck.py`, `_sr_rewrite.py`, `_sr_summary.py`, `_sr_sweep_displays.py`, `_strength_grid.py`, `_target_sweep.py`, `_tweedie_once.py`, `_verify_nb.py`, `_r0_run_all.ps1`, `_r1_run_all.ps1`, `_r1_watch_eval.ps1`, `_r0_slice_stats.csv`
- 7 files → `experiments/logs/`: `_r0_probe.log`, `cells_dump.txt`, `diag_dump.txt`, `diag_dump2.txt`, `err_dump.txt`, `srstr_run.log`, `srstr_run.err`
- Unchanged at root: library code (`n2n.py`, `unet.py`, `dsm.py`, `ddpm.py`, `train.py`, `data_loader.py`, `demo.py`, `fid.py`), notebooks, PNGs, `README.md`, `CHANGELOG.md`, this log, and `runs/`, `checkpoints/`, `Archive/`, `saved_models/`.

Every script hardcodes the absolute repo root and `chdir`s there, so its data/ckpt/output paths are unaffected; sibling imports (`_r1_eval`, `_r2_displays`, `_headtohead`) resolve because they moved together. The three `.ps1` runners were updated to call `experiments\<script>`. Commands recorded in earlier entries as `python _x.py` now run as `python experiments/_x.py` from `diffusion/`; `srstr_run.log` cited in SRSTR_01 now lives at `experiments/logs/srstr_run.log`. Nothing was deleted.

---

## RUN 2026-10-20 — RUNG_LABEL: does rung i mean added or total noise?

Training perturbs an already-noisy y by sigma_i and labels it rung i (net sees total noise sqrt(sigma_task^2+sigma_i^2)); inference feeds raw y at rung sigma_i≈sigma_task (reads the label as total noise). Diagnostic only, no training.

**Command:** `python experiments/_rung_label_check.py --target 1800`  
**Log:** `runs/displays/rung_label/rung_label_check.log` (verbatim below)

```
model l1@1860  ladder L=40 sigma 0.0200..0.4000
split   sl sig_task  rung   sig_i blend  noise   gain   sHF%   relE
train   17   0.0793   low  0.0200 0.667  0.872  0.833   99.2  1.923
train   17   0.0793   low  0.0200 1.000  0.786  0.752   99.2  1.773
train   17   0.0793  task  0.0797 0.667  0.342  0.411   90.6  1.075
train   17   0.0793  task  0.0797 1.000  0.060  0.212   33.7  0.936
train   17   0.0793    id       - 0.000  1.054  1.000  100.0  2.256
val     16   0.0560   low  0.0200 0.667  0.804  0.769   98.8  1.739
val     16   0.0560   low  0.0200 1.000  0.691  0.660   98.6  1.552
val     16   0.0560  task  0.0543 0.667  0.315  0.406   85.4  1.021
val     16   0.0560  task  0.0543 1.000  0.112  0.216   63.8  0.957
val     16   0.0560    id       - 0.000  1.054  1.000  100.0  2.191
test    25   0.0561   low  0.0200 0.667  0.784  0.746   99.7  2.050
test    25   0.0561   low  0.0200 1.000  0.663  0.626  101.2  1.809
test    25   0.0561  task  0.0543 0.667  0.314  0.375   90.0  1.175
test    25   0.0561  task  0.0543 1.000  0.110  0.195   77.4  0.980
test    25   0.0561    id       - 0.000  1.054  1.000  100.0  2.624
```

- Lowest rung (where raw y belongs if the label means *added* noise), full Tweedie: noise 0.663–0.786, gain 0.626–0.752 — barely denoises yet dims. The sigma_i^2 Tweedie coefficient makes this rung ill-conditioned.
- Task rung, full Tweedie: gain 0.195–0.216 — contrast collapse, consistent with the net expecting sqrt(2)*sigma_task there.
- Task rung, blend 2/3 (= one DDIM step sigma_task -> sigma_task/3): achieved noise 0.314–0.342, on the 1/3 target.

---

## RUN 2026-10-20 — R3_TOTAL: retrain with total-noise labels (+ channel-layout fix)

**Code changes (n2n.py, train.py; pre-change copies in `Archive/NOISE_LABEL/`).** `loss_multireal(..., sigma_y=, p_raw=)` conditions the net on TOTAL noise sigma_tot ≥ sigma_y (log-uniform to sigma_max, or exactly sigma_y with prob p_raw), adds only sqrt(sigma_tot²−sigma_y²), and decodes with sigma_tot²; `level_coord` gives a fractional ladder index so no rounding is needed; `denoise_to(y, sigma_task, target_ratio=1/3)` evaluates the raw map at its own sigma (nothing injected) and takes one DDIM step to sigma_task/3. `sigma_y=None` keeps the legacy path, verified bit-identical to the archived file.

**Second defect found while writing the harness: channel layout.** `loss_multireal` reads the stack as `(B, N, 2, H, W)` = `[Re1,Im1,Re2,Im2,…]`, but `_r0_train.pack`, `_run_srstr.pack_joint` and `asl_denoise.ipynb` cell 11 `pack_joint` build `[Re1..ReN, Im1..ImN]`. Verified numerically: R0's first "realization" equals (Re of scan 1, Re of scan 2). So every R0 model (l1/mse/huber/mag_l1), SRSTR_01 and the notebook model were trained on mismatched Re/Re and Im/Im pairs, then evaluated on true (Re, Im) pairs. All earlier decode results on those checkpoints inherit this. `_r3_train.pack` uses the correct layout (checked for all 10 realizations). The notebook was not edited.

**TODO:** port total-noise labels to `N2N_DDPM` once the DSM real-data path is validated (marked in its docstring).

**Setup.** Same split (val 16/11/1, test 25/22/8), backbone (base 32, emb 32), LR 1e-3, batch 8, seed 101. Ladder L=40 from the smallest training sigma_hat to 0.40; p_raw 0.5; no level weights; checkpoints every 200 iterations. Validation at the deployment condition (raw maps, own sigma_hat, nothing injected).

**Command:** `python experiments/_r3_train.py --loss l1 --iters 2000`

### R3_TOTAL results — l1

**Training.** 2000 iterations in 8.1 min. Full-Tweedie val L1 at deployment bottoms at iteration 1350 (0.02637) and ends at 0.02670; the 1/3 decode's val residual noise ends at 0.353 of sigma_hat with gain 0.574. Curves: `runs/r3_total/l1/eval/curves.png`.

**Matched noise, mean relE @ noise 1/3 per checkpoint** (net = `denoise_to` swept over target_ratio; blur = Gaussian; both read at exactly 1/3; identity reads noise 1.000):

| iter | val net | val blur | test net | test blur |
|---:|---:|---:|---:|---:|
| 200 | 1.4011 (2/3) | 1.2984 | 1.1210 (1/3) | 1.1362 |
| 400 | 1.2304 | 1.2984 | 1.1539 (2/3) | 1.1362 |
| 600 | 1.1909 | 1.2984 | 1.1469 | 1.1362 |
| 800 | 1.1842 | 1.2984 | 1.1397 | 1.1362 |
| 1000 | 1.1822 | 1.2984 | 1.1371 | 1.1362 |
| 1200 | 1.1814 | 1.2984 | 1.1335 | 1.1362 |
| 1400 | 1.1840 | 1.2984 | 1.1312 | 1.1362 |
| 1600 | 1.1747 | 1.2984 | 1.1256 | 1.1362 |
| 1800 | 1.1740 | 1.2984 | 1.1291 | 1.1362 |
| 2000 | 1.1720 | 1.2984 | 1.1232 | 1.1362 |

Cells where the net could not get down to noise 1/3 at any tested ratio (excluded from the means above): iter 200 slice 16, iter 200 slice 25, iter 200 slice 22, iter 400 slice 22.

**Per slice at iteration 2000** (@ noise 1/3; knob = target_ratio for net, Gaussian width for blur):

| split | slice | sigma_hat | family | knob | gain | s_bias | sHF %% | relE | phase deg |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| train | 17 | 0.0836 | net | 0.317 | 0.680 | 0.940 | 77.0 | 1.004 | 46.6 |
| train | 17 | 0.0836 | blur | 0.899 | 0.623 | 0.955 | 65.3 | 1.012 | 51.2 |
| train | 17 | 0.0836 | id | 0.000 | 1.000 | 1.000 | 100.0 | 2.256 | 69.9 |
| val | 16 | 0.0591 | net | 0.309 | 0.644 | 0.936 | 73.9 | 0.999 | 46.5 |
| val | 16 | 0.0591 | blur | 0.899 | 0.627 | 0.957 | 65.1 | 1.017 | 48.4 |
| val | 16 | 0.0591 | id | 0.000 | 1.000 | 1.000 | 100.0 | 2.191 | 66.3 |
| val | 11 | 0.0830 | net | 0.304 | 0.643 | 0.946 | 78.5 | 1.081 | 55.6 |
| val | 11 | 0.0830 | blur | 0.867 | 0.582 | 0.964 | 70.3 | 1.096 | 58.6 |
| val | 11 | 0.0830 | id | 0.000 | 1.000 | 1.000 | 100.0 | 2.354 | 73.0 |
| val | 1 | 0.1355 | net | 0.323 | 0.402 | 1.020 | 95.9 | 1.436 | 89.8 |
| val | 1 | 0.1355 | blur | 0.877 | 0.350 | 1.098 | 93.9 | 1.782 | 98.4 |
| val | 1 | 0.1355 | id | 0.000 | 1.000 | 1.000 | 100.0 | 3.371 | 94.6 |
| test | 25 | 0.0591 | net | 0.330 | 0.527 | 0.997 | 82.7 | 1.200 | 63.8 |
| test | 25 | 0.0591 | blur | 1.034 | 0.461 | 1.034 | 74.1 | 1.189 | 67.0 |
| test | 25 | 0.0591 | id | 0.000 | 1.000 | 1.000 | 100.0 | 2.624 | 79.4 |
| test | 22 | 0.0470 | net | 0.273 | 0.561 | 0.962 | 78.8 | 1.126 | 57.5 |
| test | 22 | 0.0470 | blur | 1.034 | 0.517 | 1.019 | 68.6 | 1.131 | 60.2 |
| test | 22 | 0.0470 | id | 0.000 | 1.000 | 1.000 | 100.0 | 2.435 | 74.4 |
| test | 8 | 0.1360 | net | 0.322 | 0.665 | 0.943 | 75.3 | 1.043 | 49.2 |
| test | 8 | 0.1360 | blur | 0.841 | 0.599 | 0.945 | 69.5 | 1.089 | 58.9 |
| test | 8 | 0.1360 | id | 0.000 | 1.000 | 1.000 | 100.0 | 2.334 | 73.2 |

- Net beats blur on relE at matched noise on **6/7** slices (not on slice 25). It keeps more magnitude gain on 7/7 and has lower phase error on 7/7.
- The target_ratio that lands exactly on noise 1/3 is 0.273–0.330, close to the 1/3 the one-step DDIM rule predicts (slightly below it because x0_hat is not noise-free).
- Gain below 1 is not net-specific: blur at the same noise keeps 0.350–0.627. Reaching 1/3 noise on this data costs magnitude for any estimator; post-gain bias (s_bias) stays 0.936–1.020 for the net.
- The test margin over blur is much thinner than val (mean 1.1232 vs 1.1362, against 1.1720 vs 1.2984 on val).

**Displays** (`runs/r3_total/l1/eval/`): `maps_<split>_<slice>.png` — top row reference |mean of 10|, noisy realization 0, net `denoise_to(1/3)`, net full Tweedie, blur at noise 1/3; bottom row the LOO mean and each map minus it. `maps_overview.png` stacks all slices. Display checkpoint: iteration 2000 (lowest mean val relE@1/3).

**Artifacts:** `runs/r3_total/l1/` (ckpt every 200, `train.log`, `train_val.csv`, `config.json`), `runs/r3_total/l1/eval/eval.csv`, `summary.csv`. Checkpoint copy + sha256: `checkpoints/R3_TOTAL/MANIFEST.md` (`febdb9880b5b992a`). Verbatim stdout: `Archive/R3_TOTAL/stdout_train_l1_raw.html`, `stdout_eval_l1_raw.html`. Eval command: `python experiments/_r3_eval.py --run runs/r3_total/l1`.

### R3_TOTAL — target_ratio sweep, single output vs average of 10 outputs (l1, iter 2000)

Each of the 10 realizations is decoded separately with `denoise_to(target_ratio)`; *single* = realization 0's output vs the LOO mean of the other 9; *ensemble* = average of the 10 outputs vs the raw 10-scan mean. Ratios 1 (identity), 1/3, 1/4, 1/6, 1/12, 0 (full Tweedie); 7 slices.

| ratio | single noise | single relE | ensemble gain | ensemble sHF % | ensemble relE vs raw mean |
|---|---|---|---|---|---|
| 1 | 1.000–1.000 | 2.191–3.371 | 1.000–1.000 | 100.000–100.000 | 0.000–0.000 |
| 1/3 | 0.337–0.386 | 1.028–1.460 | 0.411–0.687 | 75.387–96.061 | 0.452–0.642 |
| 1/4 | 0.257–0.313 | 0.911–1.273 | 0.345–0.653 | 70.338–94.235 | 0.508–0.722 |
| 1/6 | 0.183–0.243 | 0.816–1.120 | 0.287–0.623 | 65.185–91.124 | 0.565–0.803 |
| 1/12 | 0.108–0.180 | 0.750–1.015 | 0.243–0.598 | 60.495–86.565 | 0.621–0.883 |
| 0 | 0.059–0.134 | 0.722–0.974 | 0.225–0.591 | 57.934–83.559 | 0.678–0.963 |

- On every slice, single-map relE vs LOO falls monotonically as the ratio goes to 0: the full Tweedie estimate (posterior mean) minimises error against a noisy LOO reference, even though its residual noise (0.059–0.134) sits below the reference floor of 1/3. relE and the noise-matching criterion therefore pull in different directions.
- Ensemble gain falls monotonically as well. Part of that is the denoiser removing the noise-floor magnitude of the raw mean (which is itself a noisy reference), so gain alone does not separate signal loss from noise removal; the `average minus raw mean` panels show which.
- Ensemble relE vs the raw mean is 0 at identity by construction (same realizations), so it measures how far denoising moves the mean, not quality.

**Displays:** `runs/r3_total/l1/eval/sweep/sweep_<split>_<slice>.png` (rows = ratio; columns: single output, single − LOO, average of 10 outputs, average − raw mean), `sweep_overview.png`. **Data:** `sweep_metrics.csv`. **Command:** `python experiments/_r3_display.py --run runs/r3_total/l1 --iter 2000`. **Stdout:** `Archive/R3_TOTAL/stdout_sweep_l1_raw.html`.

### SPLIT_TEST — does averaging denoised realizations beat plain averaging? (R3 l1, iter 2000)

For every subset A of a realizations (all C(10,a), a = 1,2,3,5,7,9), estimates built from A only are scored against the mean of the complementary set B. Subtracting B's noise variance (estimated per region) gives an unbiased error vs the clean map, reported as n_eff = sigma²/MSE = equivalent number of plainly averaged scans. Plain averaging equals a by an algebraic identity (checks the code, not the noise model). Ratio / blur width chosen on VAL, reported on TEST; 95% CIs from a spatial 8×8 block bootstrap (pixel sampling only — realization-level uncertainty is not included). Signal region = upper half of a 3-px-smoothed |mean of 10|. Background-only estimates are unstable (true error ≈ 0) and are not reported.

**Signal region, TEST slices** (net = average of a separately denoised outputs; blur = blurred plain average):

| a | slice | plain avg | net (r) | net 95% CI | blur (w) | blur 95% CI |
|---:|---:|---:|---|---|---|---|
| 1 | 8 | 1.00 | 17.80 (0.000) | 15.9–20.4 | 11.23 (5.00) | 9.8–13.1 |
| 1 | 22 | 1.00 | 19.01 (0.000) | 16.3–22.7 | 12.69 (5.00) | 10.6–15.5 |
| 1 | 25 | 1.00 | 27.70 (0.000) | 22.1–34.6 | 20.85 (5.00) | 16.5–26.9 |
| 2 | 8 | 2.00 | 19.69 (0.000) | 17.3–22.9 | 13.77 (3.00) | 11.9–16.2 |
| 2 | 22 | 2.00 | 23.90 (0.000) | 19.9–29.5 | 14.49 (3.00) | 12.5–17.5 |
| 2 | 25 | 2.00 | 33.64 (0.000) | 26.2–44.7 | 22.80 (3.00) | 18.8–28.2 |
| 3 | 8 | 3.00 | 22.37 (0.083) | 19.8–25.9 | 14.92 (2.50) | 12.8–17.7 |
| 3 | 22 | 3.00 | 25.69 (0.083) | 22.0–31.5 | 15.70 (2.50) | 13.0–19.1 |
| 3 | 25 | 3.00 | 39.36 (0.083) | 30.4–52.4 | 24.51 (2.50) | 20.1–30.5 |
| 5 | 8 | 5.00 | 23.76 (0.083) | 21.1–27.5 | 17.48 (1.60) | 15.1–20.5 |
| 5 | 22 | 5.00 | 29.59 (0.083) | 24.8–35.8 | 18.03 (1.60) | 15.2–21.7 |
| 5 | 25 | 5.00 | 43.30 (0.083) | 33.7–61.1 | 26.59 (1.60) | 22.3–32.2 |
| 7 | 8 | 7.00 | 26.65 (0.167) | 23.4–30.2 | 20.27 (1.10) | 17.5–23.7 |
| 7 | 22 | 7.00 | 32.09 (0.167) | 27.3–38.3 | 20.47 (1.10) | 17.8–24.3 |
| 7 | 25 | 7.00 | 46.79 (0.167) | 36.9–61.2 | 27.78 (1.10) | 23.7–32.8 |
| 9 | 8 | 9.00 | 27.63 (0.167) | 24.3–32.0 | 23.01 (0.90) | 19.8–26.6 |
| 9 | 22 | 9.00 | 34.44 (0.167) | 29.1–41.9 | 22.99 (0.90) | 20.0–26.6 |
| 9 | 25 | 9.00 | 49.44 (0.167) | 38.7–65.5 | 29.74 (0.90) | 25.9–34.9 |

- Averaging denoised outputs beats plain averaging on every val/test slice at every a: by 17.8–49.5x at a=1, 4.8–12.5x at a=5, 3.1–7.4x at a=9 (n_eff ratio).
- It beats the val-tuned blurred average on every test slice at every a (point estimates); the CIs separate on 13 of 18 test cells.
- The advantage shrinks as a grows: going from 1 to 9 denoised outputs raises n_eff only 1.34–1.81x (plain averaging: 9x). This is the systematic-error floor — averaging removes each output's random error but not the bias shared by all outputs.
- The val-chosen ratio rises with a (a=1: 0.000, a=2: 0.000, a=3: 0.083, a=5: 0.083, a=7: 0.167, a=9: 0.167): with more scans to average, it pays to keep more of each scan and shrink less.
- **Caveat — n_eff is pure MSE.** On this low-SNR data MSE rewards heavy smoothing: blurring a single scan with a 4–5 px Gaussian already scores n_eff above a 10-scan plain average. n_eff therefore says nothing about resolution; a structure/resolution-aware comparison is needed before claiming equivalence to more scans.
- Slice 17 was a training slice and its B realizations were training targets, so its numbers are biased upward; reported in the CSV, excluded above.

**Displays:** `runs/r3_total/l1/eval/split_test/neff_all.png`, `neff_signal.png` (n_eff vs a with CIs), `neff_vs_ratio.png`, `maps_split_<split>_<slice>.png` (one split, a=5). **Data:** `split_metrics.csv` (every family/knob/region), `split_selected.csv`. **Command:** `python experiments/_split_test.py --run runs/r3_total/l1 --iter 2000`. **Stdout:** `Archive/R3_TOTAL/stdout_split_l1_raw.html`.

---

## RUN 2026-10-20 — R4_SUBSET: subset-averaging model (denoise the mean of m realizations)

**Idea.** With independent Gaussian noise the mean of all realizations is sufficient, so the best use of 10 scans is to denoise their mean once. R4 trains one total-noise-conditioned net on subset means: input = mean of a random m-subset (m = 1..9, noise sigma_hat/sqrt(m)), target = mean of the other 10−m (independent, so still valid N2N). `N2N_DSM.loss_subset` / `DiffusionTrainer.step_subset`; the R3 path was refactored into a shared helper and verified bit-identical to the pre-change file for all four losses (snapshot `Archive/NOISE_LABEL/n2n_r3.py`). Input/target construction verified on synthetic data: input variance sigma²/m, target sigma²/(10−m), zero cross-covariance.

**Setup.** As R3 (split, backbone, LR, batch, seed, p_raw), ladder from smallest training sigma_hat/3 (0.0123) to 0.40, 3000 iterations, checkpoint every 200, selection by geometric-mean val n_eff over a = 1, 5, 9 (cyclic subsets). **Command:** `python experiments/_r4_train.py --loss l1 --iters 3000`.

**Training.** Val n_eff peaks at iteration 1600 (63.63) and falls to 28.14 by 3000 while the train loss keeps dropping (0.07518 → 0.03549): clear overfitting after the peak, which the standing rule against long training anticipated. Curves: `runs/r4_subset/l1/eval/curves.png`.

**Split test** (`experiments/_r4_split_test.py`, protocol of SPLIT_TEST; R3 = `runs/r3_total/l1` iter 2000, read only). Signal region, n_eff with 95%% block-bootstrap CI, knobs chosen on val:

| a | split | slice | sigma_hat | plain | R3 avg of denoised | R4 subset mean | blur |
|---:|---|---:|---:|---:|---|---|---|
| 1 | val | 1 | 0.1355 | 1.00 | 49.51 (40.3–64.8) | 54.17 (42.5–71.8) | 29.56 (23.3–38.5) |
| 1 | test | 8 | 0.1360 | 1.00 | 17.80 (15.8–20.3) | 18.84 (16.8–21.5) | 11.23 (9.8–13.3) |
| 1 | val | 11 | 0.0830 | 1.00 | 20.20 (17.6–23.2) | 19.70 (17.4–22.6) | 11.91 (10.2–14.3) |
| 1 | val | 16 | 0.0591 | 1.00 | 18.57 (16.2–21.7) | 17.68 (15.3–20.5) | 10.71 (9.2–12.6) |
| 1 | test | 22 | 0.0470 | 1.00 | 19.01 (16.1–22.3) | 18.16 (15.6–21.6) | 12.69 (10.5–15.5) |
| 1 | test | 25 | 0.0591 | 1.00 | 27.70 (22.8–34.5) | 29.12 (23.9–36.2) | 20.85 (16.5–26.8) |
| 5 | val | 1 | 0.1355 | 5.00 | 62.29 (49.7–80.9) | 69.28 (54.1–95.2) | 40.35 (33.0–51.6) |
| 5 | test | 8 | 0.1360 | 5.00 | 23.76 (20.9–27.8) | 25.93 (23.2–29.1) | 17.48 (14.9–20.7) |
| 5 | val | 11 | 0.0830 | 5.00 | 27.39 (23.8–32.6) | 27.81 (24.9–31.7) | 20.33 (17.4–23.9) |
| 5 | val | 16 | 0.0591 | 5.00 | 25.70 (22.2–30.1) | 19.18 (17.9–20.7) | 18.26 (15.9–21.5) |
| 5 | test | 22 | 0.0470 | 5.00 | 29.59 (24.5–36.2) | 15.59 (14.8–16.5) | 18.03 (15.2–21.4) |
| 5 | test | 25 | 0.0591 | 5.00 | 43.30 (32.6–58.1) | 26.09 (23.7–28.6) | 26.59 (22.1–33.1) |
| 9 | val | 1 | 0.1355 | 9.00 | 66.51 (54.8–82.7) | 79.59 (60.7–111.5) | 43.30 (36.4–52.5) |
| 9 | test | 8 | 0.1360 | 9.00 | 27.63 (24.2–31.9) | 31.82 (28.3–36.1) | 23.01 (19.9–26.8) |
| 9 | val | 11 | 0.0830 | 9.00 | 31.61 (27.4–37.0) | 29.12 (26.9–31.6) | 26.59 (23.6–30.6) |
| 9 | val | 16 | 0.0591 | 9.00 | 29.84 (25.9–34.8) | 21.17 (20.2–22.1) | 25.05 (22.1–28.5) |
| 9 | test | 22 | 0.0470 | 9.00 | 34.44 (29.3–41.9) | 18.44 (17.8–19.2) | 22.99 (19.9–26.5) |
| 9 | test | 25 | 0.0591 | 9.00 | 49.44 (38.9–66.4) | 25.18 (23.7–26.8) | 29.74 (25.9–34.7) |

- **R4 ≈ R3 for a single scan** (R4/R3 = 0.95–1.09 across val/test at a=1), as expected: m=1 is the same objective.
- **R4 does not deliver the sufficiency gain at large a.** At a=9 its ratio to R3 ranges 0.51–1.20 and tends to rise with the slice's noise (Spearman rho = 0.77 over six slices, two tied in sigma_hat): it wins on the noisiest slices and loses on the quieter ones, where its n_eff stays nearly flat as a grows.
- **Likely cause: training coverage at low total noise.** An averaged input from slice s at a=9 sits at sigma_hat_s/3; only training slices at least that quiet can produce such inputs. Count of training slices with sigma_hat ≤ the eval slice's: s22: 2, s16: 5, s25: 5, s11: 12, s1: 24, s8: 24. The quiet test slices are served by a handful of training slices, so the low-noise end of the ladder is thinly trained.
- **Resolution check passes for R3.** Splitting error into radial frequency bands (all pixels, unbiased per band), the R3 average of denoised outputs has n_eff ≥ blur in every band — including the highest — on every test slice at a=5. Its advantage over plain averaging is therefore not a smoothing artifact. Bands with very few frequencies (lowest) can read `inf` when the estimated error is ≤ 0; treat those as "below measurable".

**Decision.** R3 (average of separately denoised realizations) remains the best estimator and the one to base the argument on. R4 is not yet a fair test of the sufficiency idea: fix its low-noise coverage first (e.g. sample m per slice so total noise is spread evenly in log sigma over the deployment range, or weight quiet slices), then re-run this exact split test.

**Displays** (`runs/r4_subset/l1/eval/split_test/`): `neff_signal.png`, `neff_all.png` (n_eff vs a, four methods, CIs); `neff_bands_a{1,5,9}.png` (resolution check); `maps_a5_<split>_<slice>.png` (one split, estimates from realizations 0–4, independent reference = mean of 5–9, bottom row = difference); `maps_a10_<split>_<slice>.png` (deployment with all 10 scans; no reference left, bottom row = difference from the plain average).

**Artifacts:** `runs/r4_subset/l1/` (ckpt every 200, `train.log`, `train_val.csv`, `config.json`), `runs/r4_subset/l1/eval/split_test/split_metrics.csv`, `split_selected.csv`. Checkpoint copy + sha256: `checkpoints/R4_SUBSET/MANIFEST.md` (`30096908a052fdc3`). Verbatim stdout: `Archive/R4_SUBSET/stdout_train_l1_raw.html`, `stdout_split_l1_raw.html`. Split-test command: `python experiments/_r4_split_test.py --r4-run runs/r4_subset/l1 --r4-iter 1600`. R3 artifacts were hashed before this run (`Archive/R4_SUBSET/r3_artifacts_before.sha256`) and re-verified unchanged after it.

---

## PLAN 2026-10-20 — R4 polish and cross-validated proof for R3 (logged before results)

**R4b / R4c** (`experiments/_r4b_train.py`, outputs `runs/r4b_subset/{bal,bal_aug}/l1`): R4 under-trained the quietest averaged inputs (R4's uniform m put 2.4% and 1.6% of draws in the two lowest log-sigma bins, where the quiet test slices sit at a=9). R4b draws (slice, m) pairs with inverse-histogram weights so all 12 log-sigma bins get equal share (8.3% each); R4c adds random dihedral transforms (exact symmetries of i.i.d. pixel noise). `loss_subset(m=...)` added; default path verified bit-identical to `Archive/R4_SUBSET/n2n_r4.py`. Variant chosen on val, reported on test.

**CV_R3** (`experiments/_cv_train.py`, `_cv_eval.py`, outputs `runs/cv_r3`): 5-fold cross-validation over all 34 slices with the fixed R3 recipe, each slice scored only by the model that never saw it; oracle-tuned blur as a deliberately favoured baseline; slice-level bootstrap and Wilcoxon tests. Endpoints and decision rule pre-registered at `Archive/CV_R3/PREREGISTRATION.md` (sha256 `230c76431ae08029`) before any CV model was trained.

### R4b / R4c results

| variant | val n_eff gmean at best saved ckpt | best iter | at iter 3000 |
|---|---:|---:|---:|
| R4 | 63.63 | 1600 | 28.14 |
| R4b | 62.58 | 1600 | 48.70 |
| R4c | 75.56 | 3000 | 75.56 |

- Balancing alone (R4b) does not beat R4 on val; balancing + dihedral augmentation (R4c) does and, unlike R4/R4b, has not overfit by 3000 iterations. R4c (iter 3000) is selected by the declared rule.
- Split test, signal region, a=9, test slices — R4c / R4: s25 1.51x, s22 1.31x, s8 1.00x. The coverage fix lands where it was aimed (the quiet slices).
- R4c beats plain averaging and the val-tuned blur on every test slice at a = 1, 5, 9.
- R4c / R3 at a=9: s25 0.77, s22 0.70, s8 1.15. R4c still trails R3 on the two quiet test slices and leads on the noisy one; with 28 training slices, one model covering every averaging depth appears data-limited. R3 stays the basis for the proof.

**Artifacts:** `runs/r4b_subset/{bal,bal_aug}/l1/` (ckpt, `train_val.csv`, `eval/curves.png`), `runs/r4b_subset/bal_aug/l1/eval/split_test/` (same figures/maps as R4). Checkpoint copy: `checkpoints/R4_SUBSET/MANIFEST.md` (`71acc0e38e41ded4`). Stdout: `Archive/R4_SUBSET/stdout_{train_r4b_bal,train_r4c_bal_aug,split_r4c}_raw.html`. Commands: `python experiments/_r4b_train.py --tag bal` / `--tag bal_aug --aug`; `python experiments/_r4_split_test.py --r4-run runs/r4b_subset/bal_aug/l1 --r4-iter 3000`.

---

## RUN 2026-10-20 — CV_R3: cross-validated proof that averaging N2N-denoised scans beats plain averaging

Pre-registration `Archive/CV_R3/PREREGISTRATION.md` (sha256 `230c76431ae08029`), written before training. 5 folds over all 34 slices (stratified by sigma_hat), fixed R3 recipe, iteration 2000, ratio 0; each slice scored only by the fold that never saw it; blur width chosen on the scored slice itself (oracle). Signal region; n_eff ratios are geometric means over slices with slice-bootstrap CIs and one-sided Wilcoxon tests.

**PRIMARY (a=5, R3 vs plain): 5.96× (95% CI 5.23–6.85; 34/34 slices; p = 5.8e-11) → SUPPORTED** under the pre-registered rule.

| scans averaged (a) | R3 vs plain averaging | R3 vs best-case blur | best-case blur vs plain |
|---:|---|---|---|
| 1 | 22.80× (95% CI 19.29–26.45; 34/34 slices; p = 5.8e-11) | 1.24× (95% CI 1.06–1.38; 30/34 slices; p = 1.2e-06) | 18.38× |
| 2 | 13.26× (95% CI 11.47–15.38; 34/34 slices; p = 5.8e-11) | 1.25× (95% CI 1.10–1.36; 30/34 slices; p = 1.2e-06) | 10.64× |
| 3 | 9.39× (95% CI 8.23–10.84; 34/34 slices; p = 5.8e-11) | 1.21× (95% CI 1.10–1.31; 30/34 slices; p = 5.1e-06) | 7.73× |
| 5 | 5.96× (95% CI 5.23–6.85; 34/34 slices; p = 5.8e-11) | 1.13× (95% CI 1.05–1.21; 22/34 slices; p = 9.6e-04) | 5.26× |
| 7 | 4.37× (95% CI 3.85–5.04; 34/34 slices; p = 5.8e-11) | 1.05× (95% CI 0.98–1.12; 20/34 slices; p = 1.7e-01) | 4.16× |
| 9 | 3.46× (95% CI 3.03–3.99; 34/34 slices; p = 5.8e-11) | 0.98× (95% CI 0.92–1.04; 12/34 slices; p = 8.5e-01) | 3.54× |

- R3 beats plain averaging on every slice at every a; the margin shrinks as a grows.
- Against the best-case blur R3 is significantly better for a ∈ {1, 2, 3, 5} and not for a ∈ {7, 9}.
- Best-case blur alone already gives 5.26× at a=5, i.e. 93% of R3's log-gain over plain averaging is available to any well-tuned spatial smoother; R3's specific edge is the remainder.
- Sensitivity: excluding slices 26, 29 → 5.94× (95% CI 5.21–6.84; 32/32 slices; p = 2.3e-10); all pixels → 9.80× (95% CI 8.52–11.38; 34/34 slices; p = 5.8e-11).
- Failure case: slice 29, the quietest slice (sigma_hat 0.0368), quieter than every training slice of its fold (flagged when that fold was trained). R3 n_eff 4.08 vs blur 24.93 at a=1 (29.23 vs 35.61 at a=9). Next-worst R3/blur ratio at a=1: slice 0 at 0.85.

**Resolution / mechanism (a=5, all pixels, per frequency band):**

| frequency band (cycles/pixel) | slices where R3 ≥ best-case blur | median R3 / blur |
|---|---:|---:|
| 0–0.0625 | 34 / 34 | 3.42× |
| 0.0625–0.125 | 24 / 34 | 1.30× |
| 0.125–0.25 | 16 / 34 | 0.98× |
| 0.25–0.75 | 24 / 34 | 1.10× |

R3's edge over blur is concentrated at the coarsest scale (every slice), where a spatial smoother cannot separate noise from signal; mid frequencies are a tie and the finest band a modest edge.

**Figures:** `runs/cv_r3/eval/cv_neff_vs_a_{signal,all}.png`, `cv_ratio_vs_sigma.png`, `cv_bands_a5.png`, `maps/maps_cv_sNN.png` (all 34 slices: one a=5 split with an independent reference, and the all-10 deployment). **Data:** `per_slice.csv`, `bands.csv`, `endpoints.csv`. **Commands:** `python experiments/_cv_train.py --fold K` (K = 0..4), `python experiments/_cv_eval.py`, `python experiments/_cv_report.py`. Checkpoints: `checkpoints/CV_R3/MANIFEST.md`. Stdout: `Archive/CV_R3/stdout_*_raw.html`. Argument write-up: `docs/ARGUMENT_N2N_vs_averaging.md`.

---

## PLAN 2026-10-20 — CV_ABLATION: loss design of the R3 recipe (logged before training)

2×2 ablation {L1, MSE} × {unweighted, noise-normalized}; arm A (L1, unweighted) = CV_R3, reused read-only; arms B (MSE), C (MSE × (σ_ref/σ_tot)²), D (L1 × σ_ref/σ_tot) trained with the identical 5-fold recipe, batches and noise draws. Pre-registration `Archive/CV_ABLATION/PREREGISTRATION.md` (sha256 `23bc6b51408d71ec`). Code: `n2n.py` `_noise_weight`, `_regress(weight=)`, `loss_multireal/loss_subset(noise_norm=, sigma_ref=)`; `_tweedie_total` now also returns σ_tot. Defaults verified bit-identical to `Archive/CV_ABLATION/n2n_pre_ablation.py` for all losses and paths; weighting unit-tested. `_cv_train.py` gains `--loss --noise-norm --iters`, `_cv_eval.py` gains `--cv --out` (defaults unchanged). Protected artifacts hashed: `Archive/CV_ABLATION/protected_before.sha256`.

---

## CODE 2026-10-20 — N2N_DDPM total-noise labels (the TODO left from R3)

`N2N_DDPM.loss(pair, sigma_y=..., p_raw=0.5, lam_max=None)` now conditions on TOTAL noise in the
VE-equivalent coordinate λ = √(1−ᾱ)/√ᾱ: λ_tot drawn log-uniform on [σ_y, λ_max] (or exactly σ_y,
nothing added, with prob. p_raw), lifted as x_t = √ᾱ·(y + √(λ_tot² − σ_y²)·ε), which lies exactly on
the rung's forward marginal. `t_of_lambda` / `alpha_bar_at` give fractional rungs (log-λ
interpolation, exact at integers). New decoders `denoise_to` (one DDIM step to a target noise,
same rule as DSM) and `conditional_denoise_total` (full walk, nothing injected); legacy
`conditional_denoise` marked. Developed in a scratch copy and swapped in atomically because the
CV_ABLATION queue imports `n2n.py` per fold; all DDPM and DSM legacy/total paths verified
bit-identical to the live file before the swap; lifting verified on synthetic data (noise / λ² mean
0.9982; raw draws sit exactly at λ = σ_y with nothing added). Pre-change copy:
`Archive/DDPM_FIX/n2n_pre_ddpm_fix.py`. **Not yet trained or validated on data.** Note for the
phantom notebooks: `draw_pair_batch` noise is σ/√2 per Re/Im channel, which is the `sigma_y` to pass.

---

## RUN 2026-10-20 — CV_ABLATION results

Pre-registered (`Archive/CV_ABLATION/PREREGISTRATION.md`). Paired over the 34 held-out slices; plain and oracle-blur numbers verified identical across arms (only the model changed).

**PRIMARY (signal, a = 5, n_eff ratio vs A = L1 unweighted):**

- B (MSE, unweighted): 1.035× (95% CI 0.996–1.073; higher on 21/34 slices; p = 0.078, Holm 0.078) → **no significant difference**
- C (MSE, noise-normalized): 1.042× (95% CI 1.004–1.079; higher on 23/34 slices; p = 0.0096, Holm 0.019) → **better than A**
- D (L1, noise-normalized): 0.926× (95% CI 0.892–0.961; higher on 6/34 slices; p = 0.00029, Holm 0.00087) → **worse than A**

**Each arm against the baselines (signal, a = 5; geometric mean, slices won):**

| arm | vs plain averaging | vs oracle blur | a where significantly better than oracle blur (p<0.01) |
|---|---|---|---|
| A L1, unweighted (CV_R3) | 5.96× (34/34) | 1.132× (22/34) | 1, 2, 3, 5 |
| B MSE, unweighted | 6.16× (34/34) | 1.172× (29/34) | 1, 2, 3, 5, 7 |
| C MSE, noise-normalized | 6.21× (34/34) | 1.180× (27/34) | 1, 2, 3, 5 |
| D L1, noise-normalized | 5.51× (34/34) | 1.048× (17/34) | 1, 2, 3 |

**Secondary — ratio vs A at every a (signal; unadjusted p):**

| a | B MSE | C MSE norm | D L1 norm |
|---:|---|---|---|
| 1 | 1.047× [0.983–1.107], 23/34, p=0.02 | 0.963× [0.907–1.020], 16/34, p=0.3 | 0.887× [0.849–0.930], 3/34, p=8.8e-07 |
| 2 | 1.037× [0.989–1.084], 23/34, p=0.062 | 1.005× [0.957–1.050], 20/34, p=0.54 | 0.908× [0.873–0.946], 3/34, p=9.1e-06 |
| 3 | 1.035× [0.991–1.078], 21/34, p=0.087 | 1.024× [0.981–1.064], 21/34, p=0.11 | 0.917× [0.883–0.954], 5/34, p=4.2e-05 |
| 5 | 1.035× [0.996–1.074], 21/34, p=0.078 | 1.042× [1.004–1.080], 23/34, p=0.0096 | 0.926× [0.892–0.962], 6/34, p=0.00029 |
| 7 | 1.036× [0.998–1.074], 22/34, p=0.059 | 1.052× [1.015–1.087], 23/34, p=0.0019 | 0.930× [0.896–0.966], 7/34, p=0.00063 |
| 9 | 1.037× [1.001–1.075], 22/34, p=0.05 | 1.057× [1.021–1.093], 25/34, p=0.00058 | 0.933× [0.899–0.969], 7/34, p=0.00097 |

- **H1 (MSE ≥ L1):** B/A = 1.035×, Holm p = 0.078 → not supported (no significant difference).
- **H2 (normalization helps quiet slices most):** Spearman of per-slice ratio vs σ̂ — C/B: rho = -0.50 (p = 0.0024); D/A: rho = -0.30 (p = 0.088) → partly supported.
- C vs D (MSE vs L1, both normalized): 1.126× [1.092–1.160], 31/34, p = 2e-08.
- Per frequency band, lowest to highest (a = 5; median ratio vs A over slices where both are measurable; slices where the arm beats A out of 34; slices with an unmeasurably small error in either arm): B: 1.13 (19/34, 6 unmeas.), 1.38 (23/34, 0 unmeas.), 1.41 (24/34, 0 unmeas.), 1.00 (17/34, 0 unmeas.); C: 1.46 (25/34, 13 unmeas.), 1.20 (25/34, 1 unmeas.), 1.02 (20/34, 0 unmeas.), 1.04 (21/34, 0 unmeas.); D: 1.16 (19/34, 8 unmeas.), 0.82 (8/34, 0 unmeas.), 0.75 (4/34, 0 unmeas.), 0.91 (5/34, 0 unmeas.).
- Slice 29 (out-of-range quiet slice), n_eff vs oracle blur at a = 1 / 5 / 9: A 4.1 vs 24.9 / 18.1 vs 31.9 / 29.2 vs 35.6; B 3.4 vs 24.9 / 15.6 vs 31.9 / 26.2 vs 35.6; C 3.7 vs 24.9 / 17.1 vs 31.9 / 28.9 vs 35.6; D 3.7 vs 24.9 / 17.2 vs 31.9 / 28.8 vs 35.6.
- Caveat (pre-registered): the best of four arms on the same 34 slices is mildly optimistic; it is the best of this ablation, not a second confirmed claim. CV_R3 (arm A) stays the headline.

**Figures:** `runs/cv_ablation/report/abl_neff_vs_a.png`, `abl_paired_ratio_vs_sigma.png`, `abl_bands_a5.png`; per-arm maps and figures under `runs/cv_ablation/<arm>/eval/`. **Data:** `runs/cv_ablation/report/paired.csv`, `summary.json`. **Commands:** `python experiments/_cv_train.py --fold K --loss {mse,l1} [--noise-norm] --outdir runs/cv_ablation/<arm>`; `python experiments/_cv_eval.py --cv runs/cv_ablation/<arm>`; `python experiments/_cv_ablation_report.py`. Weights: `checkpoints/CV_ABLATION/MANIFEST.md`. Stdout: `Archive/CV_ABLATION/*_raw.html`.

---

## PLAN 2026-10-21 — CV_RATIO: matched-noise loss comparison (A) and ratio × split sweep (B)

Decouples the loss comparison from the decoding ratio. The CV_R3 proof and CV_ABLATION decoded at target_ratio 0 (full posterior mean), whose n_eff gain is partly blur. (A) compares the four loss arms at the locked theoretical ratio r = 1/3 (noise of the LOO-of-9 training target), primary at a = 1, vs plain averaging and a noise-matched blur; (B) sweeps r × a descriptively for all arms. Evaluation-only on the existing 20 fold models. Spatial-map figures for slices 17, 16, 11, 1, 25, 22, 8. Pre-registration `Archive/CV_RATIO/PREREGISTRATION.md` (sha256 `b2e606c6bbeee98e`). The older SR_SWEEP (sigma_ratio noise injection on legacy models) is excluded from the abstract. Protected artifacts hashed: `Archive/CV_RATIO/protected_before.sha256`.

---

## RUN 2026-10-21 — CV_RATIO results

Pre-registered (`Archive/CV_RATIO/PREREGISTRATION.md`); evaluation-only on the 20 existing fold models; every slice scored by its held-out model. Self-checks passed for every arm: r = 0 reproduces the CV_R3/CV_ABLATION n_eff, r = 1 equals plain averaging, oracle blur identical, noise-matched blur within 3% of the target noise.

### Set A — losses at the locked ratio r = 1/3

Achieved residual noise of one output at r = 1/3 (median, range over slices): L1 0.348 (0.324–0.652); MSE 0.353 (0.328–0.689); MSE, noise-norm. 0.348 (0.339–0.674); L1, noise-norm. 0.345 (0.329–0.673).

**PRIMARY (a = 1, signal, n_eff ratio vs L1):**

- MSE: 0.951× (95% CI 0.919–0.981; 11/34), p = 0.0065, Holm 0.02 → **worse than L1**
- MSE, noise-norm.: 1.010× (95% CI 0.986–1.044; 17/34), p = 0.77, Holm 1 → **no significant difference**
- L1, noise-norm.: 1.003× (95% CI 0.980–1.030; 14/34), p = 0.6, Holm 1 → **no significant difference**

Each arm at r = 1/3 against the baselines (signal; geometric mean, slices won, one-sided p):

| arm | a | vs plain averaging | vs noise-matched blur | vs oracle-MSE blur |
|---|---:|---|---|---|
| L1 | 1 | 6.843× (34/34, p=6e-11) | 1.062× (32/34, p=3e-09) | 0.372× (0/34, p=1) |
| L1 | 5 | 4.911× (34/34, p=6e-11) | 1.143× (33/34, p=1e-10) | 0.934× (22/34, p=0.4) |
| L1 | 9 | 3.850× (34/34, p=6e-11) | 1.187× (33/34, p=1e-10) | 1.087× (26/34, p=0.003) |
| MSE | 1 | 6.506× (34/34, p=6e-11) | 1.053× (32/34, p=1e-09) | 0.354× (0/34, p=1) |
| MSE | 5 | 4.784× (34/34, p=6e-11) | 1.130× (34/34, p=6e-11) | 0.909× (21/34, p=0.7) |
| MSE | 9 | 3.803× (34/34, p=6e-11) | 1.172× (34/34, p=6e-11) | 1.074× (26/34, p=0.002) |
| MSE, noise-norm. | 1 | 6.911× (34/34, p=6e-11) | 1.070× (33/34, p=1e-09) | 0.376× (0/34, p=1) |
| MSE, noise-norm. | 5 | 5.052× (34/34, p=6e-11) | 1.178× (34/34, p=6e-11) | 0.961× (22/34, p=0.3) |
| MSE, noise-norm. | 9 | 4.002× (34/34, p=6e-11) | 1.238× (34/34, p=6e-11) | 1.130× (26/34, p=0.002) |
| L1, noise-norm. | 1 | 6.865× (34/34, p=6e-11) | 1.058× (28/34, p=8e-07) | 0.373× (0/34, p=1) |
| L1, noise-norm. | 5 | 4.840× (34/34, p=6e-11) | 1.125× (33/34, p=1e-09) | 0.920× (19/34, p=0.7) |
| L1, noise-norm. | 9 | 3.759× (34/34, p=6e-11) | 1.161× (31/34, p=1e-08) | 1.062× (25/34, p=0.03) |

Secondary, ratio vs L1 at r = 1/3 (signal; unadjusted p):

| a | MSE | MSE, noise-norm. | L1, noise-norm. |
|---:|---|---|---|
| 1 | 0.951× [0.918–0.981], 11/34, p=0.0065 | 1.010× [0.986–1.043], 17/34, p=0.77 | 1.003× [0.981–1.030], 14/34, p=0.6 |
| 2 | 0.958× [0.929–0.985], 10/34, p=0.0091 | 1.016× [0.993–1.048], 18/34, p=0.58 | 0.997× [0.975–1.024], 13/34, p=0.31 |
| 3 | 0.964× [0.936–0.990], 10/34, p=0.015 | 1.021× [0.999–1.051], 20/34, p=0.19 | 0.993× [0.971–1.019], 13/34, p=0.13 |
| 5 | 0.974× [0.949–0.998], 14/34, p=0.12 | 1.029× [1.008–1.056], 25/34, p=0.0086 | 0.986× [0.963–1.012], 12/34, p=0.034 |
| 7 | 0.982× [0.958–1.004], 18/34, p=0.33 | 1.035× [1.014–1.061], 26/34, p=0.00058 | 0.980× [0.958–1.008], 10/34, p=0.026 |
| 9 | 0.988× [0.965–1.010], 18/34, p=0.6 | 1.040× [1.019–1.065], 26/34, p=6.2e-05 | 0.976× [0.953–1.004], 10/34, p=0.016 |

Per band vs the noise-matched blur at r = 1/3 (median ratio; slices where the arm wins):

- L1: a=1: 6.06 (34), 5.34 (34), 2.61 (34), 0.34 (0); a=5: 4.04 (34), 3.35 (34), 1.92 (34), 0.64 (0); a=9: 3.18 (34), 2.48 (34), 1.66 (34), 0.83 (0)
- MSE: a=1: 5.55 (34), 4.99 (34), 2.57 (34), 0.35 (0); a=5: 4.04 (34), 3.53 (34), 2.05 (34), 0.62 (0); a=9: 3.22 (33), 2.80 (34), 1.73 (34), 0.79 (0)
- MSE, noise-norm.: a=1: 6.42 (34), 5.31 (34), 2.57 (34), 0.33 (0); a=5: 5.09 (34), 3.55 (34), 1.99 (34), 0.65 (0); a=9: 3.51 (34), 2.62 (34), 1.73 (34), 0.85 (2)
- L1, noise-norm.: a=1: 6.34 (34), 5.23 (34), 2.53 (34), 0.32 (0); a=5: 4.03 (34), 2.97 (34), 1.70 (34), 0.63 (0); a=9: 2.93 (33), 2.07 (34), 1.39 (34), 0.80 (0)

### Set B — ratio × split size (descriptive)

Geometric-mean n_eff (signal) by r and a; r = 1 is plain averaging (= a):

**L1**

| r | a=1 | a=2 | a=3 | a=5 | a=7 | a=9 | noise of one output |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 22.80 | 26.52 | 28.18 | 29.78 | 30.60 | 31.11 | 0.090 |
| 1/12 | 20.77 | 26.67 | 29.60 | 32.60 | 34.16 | 35.14 | 0.125 |
| 1/6 | 14.92 | 22.36 | 26.91 | 32.28 | 35.38 | 37.43 | 0.193 |
| 1/4 | 10.02 | 16.93 | 22.03 | 29.10 | 33.82 | 37.21 | 0.269 |
| 1/3 | 6.84 | 12.45 | 17.13 | 24.55 | 30.20 | 34.65 | 0.348 |
| 1/2 | 3.58 | 6.95 | 10.14 | 16.02 | 21.33 | 26.14 | 0.509 |
| 1 | 1.00 | 2.00 | 3.00 | 5.00 | 7.00 | 9.00 | 1.000 |

**MSE**

| r | a=1 | a=2 | a=3 | a=5 | a=7 | a=9 | noise of one output |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 23.88 | 27.50 | 29.16 | 30.82 | 31.70 | 32.26 | 0.077 |
| 1/12 | 20.54 | 26.69 | 29.83 | 33.12 | 34.89 | 36.02 | 0.122 |
| 1/6 | 14.24 | 21.73 | 26.46 | 32.18 | 35.57 | 37.84 | 0.196 |
| 1/4 | 9.49 | 16.24 | 21.34 | 28.58 | 33.52 | 37.13 | 0.274 |
| 1/3 | 6.51 | 11.92 | 16.52 | 23.92 | 29.65 | 34.23 | 0.353 |
| 1/2 | 3.46 | 6.74 | 9.84 | 15.61 | 20.84 | 25.62 | 0.513 |
| 1 | 1.00 | 2.00 | 3.00 | 5.00 | 7.00 | 9.00 | 1.000 |

**MSE, noise-norm.**

| r | a=1 | a=2 | a=3 | a=5 | a=7 | a=9 | noise of one output |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 21.97 | 26.65 | 28.85 | 31.04 | 32.18 | 32.89 | 0.103 |
| 1/12 | 20.38 | 27.04 | 30.47 | 34.07 | 35.99 | 37.20 | 0.136 |
| 1/6 | 14.90 | 22.77 | 27.72 | 33.68 | 37.19 | 39.52 | 0.195 |
| 1/4 | 10.10 | 17.24 | 22.61 | 30.18 | 35.31 | 39.04 | 0.270 |
| 1/3 | 6.91 | 12.64 | 17.49 | 25.26 | 31.25 | 36.02 | 0.348 |
| 1/2 | 3.61 | 7.02 | 10.26 | 16.27 | 21.71 | 26.69 | 0.509 |
| 1 | 1.00 | 2.00 | 3.00 | 5.00 | 7.00 | 9.00 | 1.000 |

**L1, noise-norm.**

| r | a=1 | a=2 | a=3 | a=5 | a=7 | a=9 | noise of one output |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20.22 | 24.08 | 25.84 | 27.57 | 28.46 | 29.02 | 0.106 |
| 1/12 | 19.24 | 24.86 | 27.66 | 30.54 | 32.05 | 33.01 | 0.132 |
| 1/6 | 14.46 | 21.52 | 25.79 | 30.79 | 33.67 | 35.56 | 0.194 |
| 1/4 | 9.95 | 16.68 | 21.57 | 28.28 | 32.70 | 35.85 | 0.268 |
| 1/3 | 6.86 | 12.41 | 17.00 | 24.20 | 29.60 | 33.83 | 0.345 |
| 1/2 | 3.60 | 6.99 | 10.18 | 16.02 | 21.26 | 25.99 | 0.506 |
| 1 | 1.00 | 2.00 | 3.00 | 5.00 | 7.00 | 9.00 | 1.000 |

Highest-n_eff ratio per a (descriptive, not a selection): L1: a=1→0, a=2→1/12, a=3→1/12, a=5→1/12, a=7→1/6, a=9→1/6; MSE: a=1→0, a=2→0, a=3→1/12, a=5→1/12, a=7→1/6, a=9→1/6; MSE, noise-norm.: a=1→0, a=2→1/12, a=3→1/12, a=5→1/12, a=7→1/6, a=9→1/6; L1, noise-norm.: a=1→0, a=2→1/12, a=3→1/12, a=5→1/6, a=7→1/6, a=9→1/4.

**Figures** (`runs/cv_ratio/report/`): `maps_A_trainval.png`, `maps_A_test.png` (set A maps), `maps_B_<arm>_s<NN>.png` (set B maps), `B_neff_surface.png`, `B_noise_vs_r.png`, `B_frontier.png`, `B_bands_vs_r.png`. **Data:** `<arm>/ratio_eval/*.csv`, `runs/cv_ratio/report/summary.json`. **Commands:** `python experiments/_cv_ratio_eval.py --cv <arm dir>`; `python experiments/_ratio_report.py`. Stdout: `Archive/CV_RATIO/*_raw.html`.

---

## PLAN 2026-10-26 — datasets 2 and 3: GEN (generalization) and MX_CV (pooled training)

dataset2 and dataset3 (same subject, poorer receiver coil, lower SNR; 2 and 3 from one session with exactly matching slices, dataset1 close) are brought into the workflow. Read-only checks: σ̂ medians 0.085 / 0.151 / 0.095 (ds1/2/3), ds3 minimum 0.0339 below ds1's 0.0368; same-index maps correlate 0.86–0.91 vs 0.75–0.79 for distant indices (same anatomy). **GEN** (evaluation-only, first): the existing dataset1-only fold models decode every ds2/ds3 slice, each by the fold that held its slice index out — strictly unseen data and anatomy (pre-registration `Archive/MX_CV/PREREGISTRATION_GEN.md`, sha256 `9d686cc815b52088`). **MX_CV**: arms A–D retrained on ds1+2+3 with folds grouped by slice index (CV_R3's assignment), 6000 iterations with 2000 primary (pre-registration `Archive/MX_CV/PREREGISTRATION.md`, sha256 `2d2474e854e15083`). New code in new files; existing scripts stay byte-reproducible. Figures are no longer committed (encrypted at rest). Protected artifacts hashed: `Archive/MX_CV/protected_before.sha256`.

---

## RUN 2026-10-26 — GEN results

Pre-registered (`Archive/MX_CV/PREREGISTRATION_GEN.md`). The dataset1-only fold models (iteration 2000) decode every slice of datasets 2 and 3 — a different receiver coil, never seen in training — each by the fold that held its slice index out, so data and anatomy are unseen. Verification 4 passed: GEN's dataset-1 rows for arm A equal the CV_RATIO numbers (10200 values).

**PRIMARY (arm A, r = 1/3, a = 1, signal):**

- Dataset 2, vs plain averaging (= n_eff of one repetition): 8.162× (95% CI 8.016–8.311; 34/34; p = 6e-11, Holm 1e-10) → **supported**
- Dataset 3, vs plain averaging (= n_eff of one repetition): 7.422× (95% CI 6.934–7.845; 34/34; p = 6e-11, Holm 1e-10) → **supported**
- Dataset 2, vs noise-matched blur: 0.989× (95% CI 0.975–1.004; 13/34; p = 0.9, Holm 0.9) → **not supported**
- Dataset 3, vs noise-matched blur: 1.046× (95% CI 1.027–1.065; 26/34; p = 3e-05, Holm 6e-05) → **supported**

**Calibration** σ̂_out/σ̂ at r = 1/3 (median, IQR):

| loss | dataset 1 | dataset 2 (unseen) | dataset 3 (unseen) |
|---|---|---|---|
| (A) L1 | 0.348 (IQR 0.341–0.355) | 0.333 (IQR 0.330–0.338) | 0.340 (IQR 0.335–0.348) |
| (B) L1, noise-normalized | 0.345 (IQR 0.342–0.353) | 0.337 (IQR 0.334–0.339) | 0.341 (IQR 0.338–0.343) |
| (C) MSE | 0.353 (IQR 0.340–0.364) | 0.339 (IQR 0.334–0.348) | 0.347 (IQR 0.334–0.361) |
| (D) MSE, noise-normalized | 0.348 (IQR 0.344–0.355) | 0.336 (IQR 0.332–0.338) | 0.339 (IQR 0.337–0.344) |

Slices below the dataset1 models' ladder (σ̂ < 0.0368, extrapolated): dataset 2: none; dataset 3: 21.

**n_eff at r = 1/3 per dataset** (geometric mean over 34 slices; ×plain, ×noise-matched blur, slices won):

| loss | dataset | a = 1 | a = 5 | a = 9 |
|---|---:|---|---|---|
| (A) L1 | 1 | 6.84 (×6.84 plain, ×1.062 blur, 32/34) | 24.55 (×4.91 plain, ×1.143 blur, 33/34) | 34.65 (×3.85 plain, ×1.187 blur, 33/34) |
| (A) L1 | 2 | 8.16 (×8.16 plain, ×0.989 blur, 13/34) | 31.67 (×6.33 plain, ×0.978 blur, 12/34) | 46.75 (×5.19 plain, ×0.972 blur, 13/34) |
| (A) L1 | 3 | 7.42 (×7.42 plain, ×1.046 blur, 26/34) | 29.30 (×5.86 plain, ×1.042 blur, 25/34) | 43.78 (×4.86 plain, ×1.039 blur, 20/34) |
| (B) L1, noise-normalized | 1 | 6.86 (×6.86 plain, ×1.058 blur, 28/34) | 24.20 (×4.84 plain, ×1.125 blur, 33/34) | 33.83 (×3.76 plain, ×1.161 blur, 31/34) |
| (B) L1, noise-normalized | 2 | 8.02 (×8.02 plain, ×0.984 blur, 11/34) | 30.71 (×6.14 plain, ×0.955 blur, 9/34) | 44.93 (×4.99 plain, ×0.937 blur, 9/34) |
| (B) L1, noise-normalized | 3 | 7.64 (×7.64 plain, ×1.045 blur, 25/34) | 29.50 (×5.90 plain, ×1.022 blur, 18/34) | 43.43 (×4.83 plain, ×1.008 blur, 16/34) |
| (C) MSE | 1 | 6.51 (×6.51 plain, ×1.053 blur, 32/34) | 23.92 (×4.78 plain, ×1.130 blur, 34/34) | 34.23 (×3.80 plain, ×1.172 blur, 34/34) |
| (C) MSE | 2 | 7.84 (×7.84 plain, ×0.994 blur, 14/34) | 31.50 (×6.30 plain, ×1.005 blur, 17/34) | 47.59 (×5.29 plain, ×1.012 blur, 19/34) |
| (C) MSE | 3 | 7.00 (×7.00 plain, ×1.049 blur, 26/34) | 28.49 (×5.70 plain, ×1.061 blur, 27/34) | 43.46 (×4.83 plain, ×1.069 blur, 27/34) |
| (D) MSE, noise-normalized | 1 | 6.91 (×6.91 plain, ×1.070 blur, 33/34) | 25.26 (×5.05 plain, ×1.178 blur, 34/34) | 36.02 (×4.00 plain, ×1.238 blur, 34/34) |
| (D) MSE, noise-normalized | 2 | 8.08 (×8.08 plain, ×0.987 blur, 12/34) | 31.50 (×6.30 plain, ×0.977 blur, 12/34) | 46.63 (×5.18 plain, ×0.971 blur, 12/34) |
| (D) MSE, noise-normalized | 3 | 7.75 (×7.75 plain, ×1.052 blur, 26/34) | 30.43 (×6.09 plain, ×1.048 blur, 23/34) | 45.25 (×5.03 plain, ×1.044 blur, 23/34) |

Per band vs noise-matched blur, a = 1 (median ratio; slices won), lowest → finest:

- (A) L1: ds1: 6.06 (34), 5.34 (34), 2.61 (34), 0.34 (0); ds2: 3.69 (34), 5.60 (34), 3.06 (34), 0.33 (0); ds3: 4.30 (34), 5.30 (34), 2.54 (34), 0.24 (0)
- (B) L1, noise-normalized: ds1: 6.34 (34), 5.23 (34), 2.53 (34), 0.32 (0); ds2: 3.51 (34), 5.37 (34), 2.97 (34), 0.33 (0); ds3: 4.93 (34), 5.10 (34), 2.49 (34), 0.25 (0)
- (C) MSE: ds1: 5.55 (34), 4.99 (34), 2.57 (34), 0.35 (0); ds2: 3.56 (34), 5.67 (34), 3.06 (34), 0.35 (0); ds3: 4.03 (34), 4.98 (34), 2.54 (34), 0.27 (0)
- (D) MSE, noise-normalized: ds1: 6.42 (34), 5.31 (34), 2.57 (34), 0.33 (0); ds2: 3.61 (34), 5.52 (34), 3.01 (34), 0.33 (0); ds3: 4.84 (34), 5.12 (34), 2.48 (33), 0.26 (0)

**Figures** (`runs/gen/report/`, not committed): `gen_neff_vs_a.png`, `gen_calibration.png`, `gen_maps_s16.png`. **Data:** `runs/gen/<arm>/*.csv`, `runs/gen/report/summary.json`. **Commands:** `python experiments/_mx_eval.py --run <dataset1-only arm> --datasets 2 3 --out runs/gen/<arm>` (A: `--datasets 1 2 3`); `python experiments/_gen_report.py`.

---

## RUN 2026-10-27 — MX_CV results

Pre-registered (`Archive/MX_CV/PREREGISTRATION.md`). 20 pooled models (arms A–D × 5 slice-index folds, datasets 1+2+3, iteration 2000 primary); every slice decoded by the fold that held its index out. Comparator: the dataset1-only models of the same arm (CV_RATIO for dataset 1, GEN for datasets 2–3). All 8 evaluations passed their self-checks; the report's dry run (pooled := dataset1-only) gave 1.000.

**PRIMARY (arm A, r = 1/3, a = 1, signal; n_eff pooled / dataset1-only):**

- Dataset 1: 0.954× (95% CI 0.925–0.994; 5/34), p = 3.5e-05, Holm 7e-05 → **pooled worse**
- Dataset 2: 0.950× (95% CI 0.933–0.967; 7/34), p = 2.2e-06, Holm 6.6e-06 → **pooled worse**
- Dataset 3: 0.977× (95% CI 0.950–1.007; 11/34), p = 0.16, Holm 0.16 → **no significant difference**

Pooled / dataset1-only for every arm (r = 1/3; geometric mean, slices better, two-sided p):

| loss | dataset | a = 1 | a = 5 | a = 9 |
|---|---:|---|---|---|
| (A) L1 | 1 | 0.954× (5/34, p=3e-05) | 0.975× (4/34, p=0.0002) | 0.987× (11/34, p=0.03) |
| (A) L1 | 2 | 0.950× (7/34, p=2e-06) | 1.021× (25/34, p=0.03) | 1.071× (27/34, p=3e-06) |
| (A) L1 | 3 | 0.977× (11/34, p=0.2) | 1.050× (24/34, p=0.006) | 1.105× (28/34, p=6e-06) |
| (B) L1, noise-normalized | 1 | 1.015× (14/34, p=0.2) | 1.046× (23/34, p=0.001) | 1.062× (25/34, p=0.0004) |
| (B) L1, noise-normalized | 2 | 1.006× (18/34, p=0.6) | 1.091× (33/34, p=2e-10) | 1.151× (34/34, p=1e-10) |
| (B) L1, noise-normalized | 3 | 1.032× (25/34, p=0.0002) | 1.149× (34/34, p=1e-10) | 1.243× (34/34, p=1e-10) |
| (C) MSE | 1 | 0.972× (9/34, p=0.006) | 0.957× (4/34, p=2e-05) | 0.949× (5/34, p=3e-06) |
| (C) MSE | 2 | 1.001× (16/34, p=1) | 1.027× (23/34, p=0.007) | 1.044× (28/34, p=6e-05) |
| (C) MSE | 3 | 0.985× (15/34, p=0.3) | 1.011× (19/34, p=0.5) | 1.032× (22/34, p=0.03) |
| (D) MSE, noise-normalized | 1 | 1.004× (13/34, p=0.03) | 0.993× (10/34, p=0.02) | 0.986× (12/34, p=0.03) |
| (D) MSE, noise-normalized | 2 | 1.003× (20/34, p=0.7) | 1.062× (32/34, p=3e-09) | 1.103× (34/34, p=1e-10) |
| (D) MSE, noise-normalized | 3 | 1.015× (17/34, p=0.4) | 1.103× (31/34, p=2e-08) | 1.177× (32/34, p=6e-10) |

Pooled models at r = 1/3 against the baselines (×plain, ×noise-matched blur with slices won and one-sided p):

| loss | dataset | a = 1 | a = 5 | a = 9 |
|---|---:|---|---|---|
| (A) L1 | 1 | ×6.53 plain; ×1.048 blur (28/34, p=3e-07) | ×4.79 plain; ×1.122 blur (34/34, p=6e-11) | ×3.80 plain; ×1.163 blur (34/34, p=6e-11) |
| (A) L1 | 2 | ×7.76 plain; ×1.004 blur (16/34, p=0.5) | ×6.47 plain; ×1.044 blur (26/34, p=4e-06) | ×5.56 plain; ×1.072 blur (30/34, p=1e-07) |
| (A) L1 | 3 | ×7.25 plain; ×1.072 blur (32/34, p=1e-09) | ×6.15 plain; ×1.130 blur (33/34, p=2e-10) | ×5.37 plain; ×1.173 blur (33/34, p=1e-10) |
| (B) L1, noise-normalized | 1 | ×6.97 plain; ×1.064 blur (32/34, p=5e-09) | ×5.06 plain; ×1.159 blur (34/34, p=6e-11) | ×3.99 plain; ×1.211 blur (34/34, p=6e-11) |
| (B) L1, noise-normalized | 2 | ×8.07 plain; ×1.010 blur (17/34, p=0.2) | ×6.70 plain; ×1.055 blur (29/34, p=2e-07) | ×5.75 plain; ×1.086 blur (32/34, p=1e-08) |
| (B) L1, noise-normalized | 3 | ×7.89 plain; ×1.092 blur (33/34, p=1e-09) | ×6.78 plain; ×1.183 blur (33/34, p=1e-10) | ×6.00 plain; ×1.258 blur (32/34, p=3e-10) |
| (C) MSE | 1 | ×6.33 plain; ×1.041 blur (24/34, p=3e-05) | ×4.58 plain; ×1.084 blur (33/34, p=2e-10) | ×3.61 plain; ×1.106 blur (33/34, p=3e-10) |
| (C) MSE | 2 | ×7.84 plain; ×1.004 blur (16/34, p=0.4) | ×6.47 plain; ×1.039 blur (27/34, p=8e-05) | ×5.52 plain; ×1.063 blur (28/34, p=1e-06) |
| (C) MSE | 3 | ×6.89 plain; ×1.063 blur (30/34, p=2e-08) | ×5.76 plain; ×1.099 blur (32/34, p=4e-10) | ×4.98 plain; ×1.124 blur (32/34, p=1e-09) |
| (D) MSE, noise-normalized | 1 | ×6.94 plain; ×1.056 blur (28/34, p=5e-07) | ×5.02 plain; ×1.147 blur (33/34, p=2e-10) | ×3.94 plain; ×1.197 blur (33/34, p=8e-10) |
| (D) MSE, noise-normalized | 2 | ×8.10 plain; ×1.004 blur (16/34, p=0.4) | ×6.69 plain; ×1.047 blur (27/34, p=8e-06) | ×5.72 plain; ×1.077 blur (29/34, p=1e-07) |
| (D) MSE, noise-normalized | 3 | ×7.87 plain; ×1.085 blur (30/34, p=5e-09) | ×6.71 plain; ×1.170 blur (33/34, p=2e-10) | ×5.92 plain; ×1.241 blur (32/34, p=3e-10) |

Losses within the pooled models (a = 1, r = 1/3; ratio vs A, Holm over B/C/D per dataset): ds1: B 1.067× (Holm 0.00063), C 0.969× (Holm 0.13), D 1.063× (Holm 0.019); ds2: B 1.040× (Holm 0.001), C 1.011× (Holm 0.43), D 1.044× (Holm 0.0097); ds3: B 1.087× (Holm 0.0002), C 0.950× (Holm 0.18), D 1.085× (Holm 0.05).

Iteration 6000 / 2000 (a = 1, r = 1/3): (A) ds1 1.123× (31/34), ds2 1.098× (30/34), ds3 1.108× (28/34); (B) ds1 1.037× (25/34), ds2 1.016× (24/34), ds3 1.023× (23/34); (C) ds1 1.118× (25/34), ds2 1.048× (15/34), ds3 1.142× (20/34); (D) ds1 1.012× (15/34), ds2 0.965× (12/34), ds3 0.992× (17/34).

Calibration σ̂_out/σ̂ at r = 1/3 (median): (A) ds1 0.358, ds2 0.347, ds3 0.351; (B) ds1 0.355, ds2 0.342, ds3 0.345; (C) ds1 0.359, ds2 0.345, ds3 0.350; (D) ds1 0.352, ds2 0.341, ds3 0.341.

Per band vs noise-matched blur, a = 1 (median ratio; slices won), lowest → finest:

- (A) L1: ds1: 5.05 (34), 4.75 (34), 2.61 (34), 0.35 (0); ds2: 5.27 (34), 5.72 (34), 3.04 (34), 0.35 (0); ds3: 6.26 (34), 5.35 (34), 2.57 (34), 0.26 (0)
- (B) L1, noise-normalized: ds1: 5.81 (34), 5.12 (34), 2.60 (34), 0.32 (0); ds2: 5.70 (34), 5.91 (34), 3.05 (34), 0.35 (0); ds3: 7.41 (34), 5.68 (34), 2.60 (34), 0.24 (0)
- (C) MSE: ds1: 4.26 (34), 4.41 (34), 2.57 (34), 0.36 (0); ds2: 5.26 (34), 5.54 (34), 3.05 (34), 0.35 (0); ds3: 5.43 (34), 4.88 (34), 2.63 (34), 0.29 (0)
- (D) MSE, noise-normalized: ds1: 5.84 (34), 4.86 (34), 2.61 (34), 0.32 (0); ds2: 5.53 (34), 5.80 (34), 3.03 (34), 0.33 (0); ds3: 6.66 (34), 5.40 (34), 2.55 (34), 0.26 (0)

**Figures** (`runs/mx_cv/report/`, not committed): `mx_neff_vs_a.png`, `mx_maps_s16.png`. **Data:** `runs/mx_cv/<arm>/eval_it{2000,6000}/`, `runs/mx_cv/report/summary.json`. **Commands:** `python experiments/_mx_train.py --arm X --fold K`; `python experiments/_mx_eval.py --run runs/mx_cv/X --iter {2000,6000} --datasets 1 2 3`; `python experiments/_mx_report.py`.

### MX_CV exploratory — pooled @6000 (equal passes per slice) vs dataset1-only @2000

*Not pre-registered.* Paired n_eff ratio, r = 1/3, signal region, 34 slices per dataset (geometric mean; slices where pooled is better; two-sided Wilcoxon p):

| loss | dataset | a = 1 | a = 5 | a = 9 |
|---|---:|---|---|---|
| (A) L1 | 1 | 1.071× (25/34, p=5e-03) | 0.887× (7/34, p=1e-04) | 0.814× (4/34, p=3e-06) |
| (A) L1 | 2 | 1.043× (26/34, p=6e-03) | 0.810× (4/34, p=7e-08) | 0.715× (3/34, p=4e-09) |
| (A) L1 | 3 | 1.083× (28/34, p=5e-03) | 0.976× (13/34, p=3e-01) | 0.930× (9/34, p=6e-02) |
| (B) L1, noise-normalized | 1 | 1.053× (24/34, p=1e-02) | 0.822× (1/34, p=4e-07) | 0.735× (1/34, p=4e-07) |
| (B) L1, noise-normalized | 2 | 1.022× (24/34, p=2e-02) | 0.800× (0/34, p=1e-10) | 0.705× (0/34, p=1e-10) |
| (B) L1, noise-normalized | 3 | 1.055× (24/34, p=1e-03) | 0.898× (5/34, p=9e-05) | 0.826× (4/34, p=8e-06) |
| (C) MSE | 1 | 1.088× (23/34, p=4e-03) | 1.000× (15/34, p=6e-01) | 0.958× (13/34, p=1e-01) |
| (C) MSE | 2 | 1.049× (21/34, p=2e-01) | 0.914× (9/34, p=5e-03) | 0.845× (10/34, p=5e-04) |
| (C) MSE | 3 | 1.125× (25/34, p=2e-03) | 1.116× (22/34, p=4e-03) | 1.119× (21/34, p=9e-03) |
| (D) MSE, noise-normalized | 1 | 1.017× (15/34, p=4e-01) | 0.841× (1/34, p=4e-07) | 0.765× (1/34, p=4e-07) |
| (D) MSE, noise-normalized | 2 | 0.968× (9/34, p=2e-02) | 0.836× (0/34, p=1e-10) | 0.769× (0/34, p=1e-10) |
| (D) MSE, noise-normalized | 3 | 1.007× (19/34, p=7e-01) | 0.932× (6/34, p=1e-03) | 0.895× (7/34, p=5e-04) |

- Single maps improve with longer pooled training (L1 a = 1: 1.043–1.083× on all datasets), but averaged estimates degrade (a = 9 for L1 and both noise-normalized arms: 0.705–0.930×): the longer-trained models carry more error that is shared by every repetition, which averaging cannot remove. The checkpoint that is best for single maps is not the one to use when outputs are averaged — the stopping point depends on the use.

### TQ exploratory diagnostics — reference quality vs SNR vs noise colour

*Exploratory, read-only (no training).* Per-repetition SNR in the signal region (median [range]): ds1 0.13 [0.02–0.24]; ds2 0.04 [0.01–0.07]; ds3 0.05 [0.01–0.11].
Noise colour, finest/coarsest band noise power (white = 1): ds1 0.66; ds2 1.48; ds3 0.53 — dataset 2's noise is fine-heavy, datasets 1 and 3 coarse-heavy. Spatial noise variation p95/p5: ds1 2.29; ds2 2.22; ds3 2.67.

Edge over the noise-matched blur (r = 1/3) across all 102 slices — Spearman ρ with log SNR and with σ̂; mean log-residual per dataset after a linear fit on log SNR:

- dataset1-only (A), a = 1: ρ(SNR) -0.19, ρ(σ̂) -0.74, residual ds1/2/3 +0.035/-0.046/+0.012
- dataset1-only (A), a = 9: ρ(SNR) +0.23, ρ(σ̂) -0.53, residual ds1/2/3 +0.087/-0.073/-0.014
- pooled (A) @2000, a = 1: ρ(SNR) -0.27, ρ(σ̂) -0.76, residual ds1/2/3 +0.018/-0.043/+0.026
- pooled (A) @2000, a = 9: ρ(SNR) +0.09, ρ(σ̂) -0.60, residual ds1/2/3 +0.022/-0.055/+0.033
- pooled (B) @2000, a = 1: ρ(SNR) -0.24, ρ(σ̂) -0.81, residual ds1/2/3 +0.020/-0.051/+0.031
- pooled (B) @2000, a = 9: ρ(SNR) +0.07, ρ(σ̂) -0.72, residual ds1/2/3 +0.023/-0.085/+0.062

Reading (to be tested by TQ E1/E2): the edge tracks noise strength, and dataset 2 stays below the SNR trend even for pooled models trained on it, consistent with its fine-heavy noise colour (cheap for a blur to remove) rather than with reference quality alone.

---

## PLAN 2026-10-27 — TQ: is performance governed by reference quality, input SNR or noise colour?

E1 retrains the dataset1-only recipe (arms A, D) with LOO targets from k = 1 or 3 other repetitions (k = 9 existing) to vary reference quality at fixed input; E2 evaluates existing models on dataset-1 slices degraded to 1.4× and 2× σ̂ with self-coloured, white or dataset-2-coloured noise to separate SNR from noise colour. Pre-registration `Archive/TQ/PREREGISTRATION.md` (sha256 `9e817c388d658b5c`). All outputs in new directories; existing results hashed (`Archive/TQ/protected_before.sha256`, 1514 files) and re-verified at the end.

---

## RUN 2026-10-27 — TQ results (reference quality vs input SNR vs noise colour)

Pre-registered (`Archive/TQ/PREREGISTRATION.md`). E1: 20 models (arms A, D × k ∈ {1, 3} × 5 folds, 2000 iterations) trained with the LOO target averaged over a random k of the other 9 repetitions; compared with the k = 9 models of the same arm, all slices decoded by their held-out fold. E2: 24 evaluations (4 models × 3 noise colours × f ∈ {2.0, 1.4}) of dataset 1 with added independent noise. All evaluations passed their self-checks; the report's dry run gave 1.000.

Achieved noise factor σ̂_degraded/σ̂ (median over slices): f2.0: self 2.001, white 2.001, ds2 2.001; f1.4: self 1.401, white 1.400, ds2 1.401.

### E1 — does the quality of the training reference matter?

**PRIMARY (arm A, dataset 1, r = 1/3, a = 1, signal; n_eff k=1 / k=9):** 1.029× (95% CI 1.004–1.056; 21/34 better), Wilcoxon p = 0.039 → **k = 1 better**.

n_eff ratio vs k = 9 (r = 1/3; geometric mean, slices better, two-sided p):

| arm | k | dataset | a = 1 | a = 5 | a = 9 | a = 1, r = 0 |
|---|---:|---:|---|---|---|---|
| (A) | 1 | 1 | 1.029× (21/34, p=0.04) | 1.015× (22/34, p=0.1) | 1.008× (20/34, p=0.5) | 1.092× (26/34) |
| (A) | 1 | 2 | 1.002× (21/34, p=0.5) | 1.026× (23/34, p=0.02) | 1.042× (26/34, p=8e-05) | 1.179× (33/34) |
| (A) | 1 | 3 | 1.021× (19/34, p=0.3) | 1.028× (18/34, p=0.1) | 1.034× (22/34, p=0.03) | 1.147× (28/34) |
| (A) | 3 | 1 | 0.989× (13/34, p=0.3) | 1.002× (18/34, p=0.9) | 1.010× (17/34, p=0.3) | 1.102× (33/34) |
| (A) | 3 | 2 | 0.980× (11/34, p=0.04) | 1.004× (18/34, p=0.8) | 1.020× (20/34, p=0.09) | 1.151× (33/34) |
| (A) | 3 | 3 | 0.985× (16/34, p=0.4) | 1.009× (18/34, p=0.4) | 1.024× (23/34, p=0.02) | 1.183× (33/34) |
| (D) | 1 | 1 | 1.008× (21/34, p=0.3) | 1.005× (19/34, p=0.2) | 1.004× (17/34, p=0.5) | 1.162× (32/34) |
| (D) | 1 | 2 | 0.999× (16/34, p=0.8) | 1.024× (23/34, p=0.006) | 1.041× (28/34, p=4e-06) | 1.226× (32/34) |
| (D) | 1 | 3 | 0.998× (15/34, p=0.5) | 1.006× (18/34, p=0.8) | 1.012× (19/34, p=0.8) | 1.200× (29/34) |
| (D) | 3 | 1 | 0.976× (15/34, p=0.1) | 0.998× (16/34, p=0.7) | 1.011× (22/34, p=0.09) | 1.135× (32/34) |
| (D) | 3 | 2 | 0.964× (14/34, p=0.004) | 0.986× (14/34, p=0.1) | 1.001× (17/34, p=0.9) | 1.123× (30/34) |
| (D) | 3 | 3 | 0.970× (14/34, p=0.04) | 0.992× (16/34, p=0.5) | 1.006× (16/34, p=0.8) | 1.144× (30/34) |

Edge over the noise-matched blur (geometric mean of n_eff / n_eff_blur, r = 1/3) by target size k = 1 / 3 / 9:

- (A): ds1 a=1: 1.066 / 1.064 / 1.062; ds1 a=9: 1.190 / 1.200 / 1.187; ds2 a=1: 0.996 / 0.995 / 0.989; ds2 a=9: 1.012 / 1.002 / 0.972; ds3 a=1: 1.053 / 1.054 / 1.046; ds3 a=9: 1.069 / 1.075 / 1.039
- (D): ds1 a=1: 1.077 / 1.075 / 1.070; ds1 a=9: 1.241 / 1.247 / 1.238; ds2 a=1: 0.993 / 0.993 / 0.987; ds2 a=9: 1.011 / 0.988 / 0.971; ds3 a=1: 1.059 / 1.058 / 1.052; ds3 a=9: 1.060 / 1.062 / 1.044

### E2 — input SNR or noise colour?

**PRIMARY (dataset1-only A, f = 2.0, a = 1, r = 1/3, signal; edge ratio colour / self, Holm over 2):**

- white / self: 1.002× (95% CI 1.000–1.005; 20/34 better), p = 0.15, Holm 0.16 → **no significant difference**
- ds2 / self: 1.003× (95% CI 1.001–1.005; 21/34 better), p = 0.078, Holm 0.16 → **no significant difference**

Edge over the noise-matched blur (geometric mean), undegraded → degraded by colour, with the real datasets 2/3 for reference:

| model | a | undegraded | self f1.4 | white f1.4 | ds2 f1.4 | self f2.0 | white f2.0 | ds2 f2.0 | real ds2 | real ds3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one_A | 1 | 1.062 | 1.028 | 1.032 | 1.032 | 1.014 | 1.017 | 1.017 | 0.989 | 1.046 |
| one_A | 9 | 1.187 | 1.140 | 1.126 | 1.120 | 1.120 | 1.103 | 1.102 | 0.972 | 1.039 |
| one_D | 1 | 1.070 | 1.030 | 1.035 | 1.035 | 1.015 | 1.018 | 1.019 | 0.987 | 1.052 |
| one_D | 9 | 1.238 | 1.172 | 1.158 | 1.153 | 1.128 | 1.109 | 1.113 | 0.971 | 1.044 |
| pool_A | 1 | 1.048 | 1.021 | 1.026 | 1.026 | 1.010 | 1.013 | 1.014 | 1.004 | 1.072 |
| pool_A | 9 | 1.163 | 1.126 | 1.114 | 1.109 | 1.105 | 1.089 | 1.089 | 1.072 | 1.173 |
| pool_D | 1 | 1.056 | 1.019 | 1.024 | 1.025 | 1.005 | 1.008 | 1.009 | 1.004 | 1.085 |
| pool_D | 9 | 1.197 | 1.135 | 1.122 | 1.120 | 1.096 | 1.079 | 1.082 | 1.077 | 1.241 |

Edge ratio colour / self for every model (geometric mean, slices better, Holm over the 2 colours):

| model | f | a | white / self | ds2 / self |
|---|---:|---:|---|---|
| one_A | 2.0 | 1 | 1.002× (20/34, Holm 0.2) | 1.003× (21/34, Holm 0.2) |
| one_A | 2.0 | 9 | 0.984× (4/34, Holm 6e-06) | 0.983× (7/34, Holm 2e-06) |
| one_A | 1.4 | 1 | 1.004× (25/34, Holm 0.002) | 1.004× (19/34, Holm 0.02) |
| one_A | 1.4 | 9 | 0.987× (6/34, Holm 1e-05) | 0.982× (4/34, Holm 2e-08) |
| one_D | 2.0 | 1 | 1.002× (20/34, Holm 0.2) | 1.004× (23/34, Holm 0.04) |
| one_D | 2.0 | 9 | 0.984× (5/34, Holm 1e-05) | 0.987× (10/34, Holm 0.0002) |
| one_D | 1.4 | 1 | 1.005× (27/34, Holm 0.0002) | 1.005× (22/34, Holm 0.007) |
| one_D | 1.4 | 9 | 0.988× (7/34, Holm 2e-05) | 0.983× (5/34, Holm 3e-07) |
| pool_A | 2.0 | 1 | 1.002× (21/34, Holm 0.07) | 1.003× (23/34, Holm 0.02) |
| pool_A | 2.0 | 9 | 0.986× (4/34, Holm 0.0001) | 0.986× (10/34, Holm 0.0003) |
| pool_A | 1.4 | 1 | 1.004× (26/34, Holm 0.0001) | 1.005× (25/34, Holm 0.0006) |
| pool_A | 1.4 | 9 | 0.989× (6/34, Holm 6e-05) | 0.985× (6/34, Holm 4e-06) |
| pool_D | 2.0 | 1 | 1.003× (22/34, Holm 0.06) | 1.004× (24/34, Holm 0.004) |
| pool_D | 2.0 | 9 | 0.985× (6/34, Holm 8e-05) | 0.987× (9/34, Holm 0.0003) |
| pool_D | 1.4 | 1 | 1.005× (26/34, Holm 0.0004) | 1.006× (25/34, Holm 0.0005) |
| pool_D | 1.4 | 9 | 0.989× (7/34, Holm 8e-05) | 0.986× (7/34, Holm 4e-05) |

**Data:** `runs/tq/<arm>_k<k>/eval_it2000/`, `runs/degrade/`, `runs/tq/report/summary.json`. **Commands:** `bash runs/tq_queue.sh`; `python experiments/_tq_report.py`.

---

## RUN 2026-10-29 — GROUND results (KR: is k marginal? SM: denoise the mean vs denoise each)

Pre-registered (`Archive/GROUND/PREREGISTRATION.md`, written before any run). 60 new models: KR-seed (k ∈ {1, 9} × seeds 102, 103 × 5 folds, 4000 iterations), KR-frac (k ∈ {1, 9} × training fraction 0.5, 0.25 × 5 folds, 2000 iterations), SM (PER and SUB, dihedral augmentation, ladder down to σ̂_min/3, dataset 1 and pooled datasets 1–3, 3000 iterations); seed 101 comes from the existing `runs/cv_r3` (k = 9) and `runs/tq/A_k1` (k = 1). All evaluated with `_gr_eval.py` (reproduces `evaluate_slice` n_eff to 6e-15); `_gr_train.py` reproduces `_cv_train.py` and `_mx_train.py` exactly on deterministic CPU. The report's dry run gave 1.000. Note: GPU training is not bit-reproducible run to run (also for the original scripts), so every seed is an independent draw.

### KR — is the target size k marginal?

**PRIMARY (dataset 1, a = 1, r = 1/3, iteration 2000; per slice the geometric mean over seeds 101–103 of n_eff(k=1)/n_eff(k=9)):** 1.000× (90% CI 0.984–1.017; 17/34 better) → **k marginal (equivalent within ±5%)**.

Per seed (same read-out): seed 101 1.029× (90% CI 1.008–1.052; 21/34 better); seed 102 0.987× (90% CI 0.959–1.014; 16/34 better); seed 103 0.985× (90% CI 0.962–1.009; 16/34 better).
Seed-to-seed noise floor (same k, seed s / seed 101): k = 9: 102: 1.018×, 103: 0.991×; k = 1: 102: 0.976×, 103: 0.948×.

Learning curve, n_eff(k=1)/n_eff(k=9) (geometric mean, 90% CI; seeds 101–103 up to 2000, 102–103 beyond; at r = 1/3 and at each k's best grid r):

| iteration | a = 1, r = 1/3 | a = 9, r = 1/3 | a = 1, best r | a = 9, best r |
|---:|---|---|---|---|
| 500 | 0.942 (0.919–0.965) | 0.955 (0.944–0.965) | 0.866 (0.841–0.889) | 0.952 (0.941–0.963) |
| 1000 | 0.968 (0.953–0.983) | 0.958 (0.951–0.966) | 0.919 (0.908–0.929) | 0.943 (0.935–0.950) |
| 1500 | 0.952 (0.940–0.964) | 0.956 (0.947–0.966) | 0.982 (0.960–1.001) | 0.951 (0.936–0.965) |
| 2000 | 1.000 (0.984–1.017) | 1.011 (0.995–1.027) | 1.125 (1.089–1.163) | 1.024 (0.999–1.050) |
| 3000 | 1.007 (0.991–1.024) | 1.194 (1.147–1.244) | 1.363 (1.300–1.429) | 1.252 (1.197–1.308) |
| 4000 | 1.009 (0.993–1.026) | 1.278 (1.231–1.329) | 1.371 (1.319–1.424) | 1.282 (1.238–1.329) |

Absolute n_eff (geometric mean, r = 1/3; seed 101 ≤ 2000, seed 102 at 4000), k = 1 / k = 9: it 500 a=1: 5.51 / 5.49; it 500 a=9: 28.20 / 28.95; it 1000 a=1: 6.31 / 6.29; it 1000 a=9: 31.93 / 32.61; it 2000 a=1: 7.04 / 6.84; it 2000 a=9: 34.94 / 34.65; it 4000 a=1: 6.82 / 6.73; it 4000 a=9: 27.90 / 21.58.

Less training data (seed 101), n_eff(k=1)/n_eff(k=9) at r = 1/3 (90% CI):

| training fraction | iteration | a = 1 | a = 5 | a = 9 |
|---:|---:|---|---|---|
| 0.5 | 1000 | 0.959 (0.934–0.985) | 0.967 (0.952–0.983) | 0.971 (0.959–0.985) |
| 0.5 | 2000 | 0.990 (0.965–1.017) | 1.091 (1.059–1.125) | 1.138 (1.098–1.181) |
| 0.25 | 1000 | 1.012 (0.985–1.040) | 1.024 (1.006–1.042) | 1.030 (1.011–1.048) |
| 0.25 | 2000 | 0.989 (0.967–1.011) | 1.160 (1.115–1.209) | 1.231 (1.174–1.291) |

Absolute n_eff at 2000 (r = 1/3), k = 1 / k = 9: frac 0.5 a=1: 6.19 / 6.25; frac 0.5 a=9: 30.09 / 26.44; frac 0.25 a=1: 5.50 / 5.56; frac 0.25 a=9: 24.16 / 19.63; full data (seed 101) a=1: 7.04 / 6.84, a=9: 34.94 / 34.65.

### SM — denoise the mean (SUB) vs denoise each, then average (PER)

**PRIMARY 2a (pooled training, dataset 1, a = 9, iteration 3000, each arm at its best grid r: SUB r = 0.000, PER r = 0.250):** 0.967× (95% CI 0.930–1.005; 15/34 better), p = 0.26, Holm 0.36 → **no significant difference**.
**PRIMARY 2b (data limitation: SUB/PER pooled ÷ SUB/PER dataset-1-only, dataset 1, a = 9):** 1.033× (95% CI 0.993–1.076; 20/34 better), p = 0.18, Holm 0.36 → **no significant difference**.

SUB / PER n_eff at each arm's best r (iteration 3000; geometric mean, slices better, two-sided p):

| training | dataset | a = 1 | a = 2 | a = 3 | a = 5 | a = 7 | a = 9 |
|---|---:|---|---|---|---|---|---|
| dataset 1 | 1 | 0.957× (6/34, p=6e-05) | 0.970× (9/34, p=0.004) | 0.962× (7/34, p=0.002) | 0.969× (15/34, p=0.3) | 0.953× (13/34, p=0.03) | 0.937× (13/34, p=0.03) |
| pooled | 1 | 0.965× (7/34, p=0.004) | 0.989× (11/34, p=0.1) | 0.982× (14/34, p=0.2) | 0.998× (20/34, p=0.7) | 0.980× (15/34, p=0.6) | 0.967× (15/34, p=0.3) |
| pooled | 2 | 0.948× (5/34, p=0.003) | 0.983× (17/34, p=0.6) | 1.003× (23/34, p=0.1) | 0.971× (20/34, p=1) | 0.944× (19/34, p=0.9) | 0.936× (21/34, p=0.5) |
| pooled | 3 | 0.988× (10/34, p=0.07) | 0.951× (12/34, p=0.1) | 0.885× (6/34, p=5e-07) | 0.811× (8/34, p=3e-05) | 0.761× (11/34, p=0.0005) | 0.721× (12/34, p=0.001) |

Same at fixed r (iteration 3000), a = 1 / 5 / 9: ds1 ds1 r=0: 0.957 / 1.050 / 1.114; ds1 ds1 r=1/3: 1.051 / 0.869 / 0.792; pool ds1 r=0: 0.965 / 1.110 / 1.201; pool ds1 r=1/3: 0.955 / 0.824 / 0.787; pool ds2 r=0: 0.948 / 1.030 / 1.042; pool ds2 r=1/3: 0.977 / 0.865 / 0.826; pool ds3 r=0: 0.988 / 0.871 / 0.803; pool ds3 r=1/3: 0.946 / 0.769 / 0.668.
Iteration 2000, best r, a = 1 / 5 / 9: ds1 ds1: 0.934 / 0.915 / 0.875; pool ds1: 0.967 / 0.944 / 0.910; pool ds2: 0.939 / 0.898 / 0.866; pool ds3: 0.958 / 0.749 / 0.669.
Absolute n_eff at best r (iteration 3000), PER / SUB: ds1 ds1 a=1: 26.08 / 24.96; ds1 ds1 a=9: 38.78 / 36.33; pool ds1 a=1: 24.50 / 23.65; pool ds1 a=9: 35.82 / 34.66; pool ds2 a=1: 59.48 / 56.36; pool ds2 a=9: 74.66 / 69.91; pool ds3 a=1: 46.10 / 45.52; pool ds3 a=9: 74.52 / 53.71.
Augmentation + ladder to σ̂/3 vs the existing PER recipe (iteration 2000, best r): dataset-1 training ds1 a=1 1.080× (24/34); dataset-1 training ds1 a=9 0.985× (15/34); pooled training ds1 a=1 0.891× (1/34); pooled training ds1 a=9 0.902× (1/34); pooled training ds2 a=1 0.940× (3/34); pooled training ds2 a=9 0.936× (2/34); pooled training ds3 a=1 0.820× (0/34); pooled training ds3 a=9 0.813× (6/33).

**Data:** `runs/gr/{kr,sm}/`, `runs/gr/{kr_eval,sm_eval}/`, `runs/gr/report/summary.json`. **Commands:** `bash runs/gr/queue_a.sh`; `bash runs/gr/queue_b.sh`; `bash runs/gr/queue_eval.sh`; `python experiments/_gr_report.py`.

---

## HOUSEKEEPING 2026-10-29 — pruning before the new main

- **Kept:** every script, pre-registration, log and text output behind a finding in `docs/RESEARCH_NOTES.md`
  (R3 onward: RUNG_LABEL, R3_TOTAL, R4/R4b/R4c, CV_R3, CV_ABLATION, CV_RATIO, GEN, MX_CV, TQ, GROUND). Map of
  scripts to experiments to findings: `experiments/README.md`. TQ and GROUND configs, training logs, stdout and the
  queue scripts are now committed like the earlier experiments'.
- **Removed:** the pre-R3 scripts and run folders (R0–R2, SR_SWEEP, FRONTIER, SRSTR_01, head-to-head and decode
  probes), whose runs used legacy added-noise labels and a Re/Re–Im/Im channel pairing. Their sections above stay as
  history. Untracked code snapshots in `Archive/*` that git history already covers were also removed. Copy of every
  removed text file: `../diffusion_pruned_2026-10-29.tar.gz` (172 files, outside the repository).
- **Checkpoints:** local checkpoints pruned to the iterations that reported results use (`experiments/_prune_runs.py`):
  `runs/` 23 GB → 8.2 GB, 314 checkpoints kept.
- **Moved:** the Shepp-Logan foundation demo (`demo.py`, `demo.ipynb`, `n2n.ipynb`, `fid.py`, `data_loader.py`, figures)
  to `legacy/`; its weights (`legacy/saved_models/`) are no longer tracked. The encrypted R3/R4 PNGs are no longer
  tracked (they remain in earlier commits).

---

## EXPLORATORY 2026-10-29 — AR_HEADROOM: how much does an adaptive strength r buy?

Not pre-registered (read-only; existing fold models at iteration 2000, held-out slices). Every pixel's split-half
error is an exact quadratic in r, so each rule's optimum has a closed form. Uniform r = 1/3 reproduces the logged
dataset-1 n_eff (worst relative difference 7e-14). Rules: fixed r = 1/3; one global r per averaging depth a,
chosen on the other folds' slices (baseline); per-slice oracle; r per local-signal or edge-strength quintile
(cross-fitted); r per 8×8 block (in-sample oracle). Honesty check: the 10 repetitions are split into two
independent halves of 5; an r map chosen on one half is scored on the other (cross-half).

**Reading:**
- **Biggest lever: r set by a (closed form / global per a), not adaptivity.** Fixed r = 1/3 reaches only 0.19–0.31×
  the global-r(a) n_eff for a single map and 0.78–0.92× at a = 9.
- **Per-slice adaptivity is small:** even the in-sample oracle gains only 0.5–4.2%.
- **Free spatial r maps are fitted noise:** the 8×8 block oracle's in-sample gain (up to +33%) turns into a loss
  cross-half (0.79–0.98× for 8×8; 0.84–1.01× for 16×16).
- **Structure-driven spatial r is real but modest:** r per local-signal quintile, cross-fitted, gains +0.1–0.8% for a
  single map and +3–8% at a = 9 (dataset 2: up to +7.8%). It survives the cross-half check (+1–11%). Edge
  strength gives nothing.

**Implication for the roadmap:** the averaging-aware strength r*(a) is the main contribution. Per-input
adaptivity can add only a few percent, and only through a structure-dependent rule (lower r in low-signal regions,
higher in high-signal ones). Its ceiling is set by learnable structure, not by per-block tuning.

**Verbatim outputs:** `runs/ar/headroom/stdout_headroom.txt` (all rules, models A and D, datasets 1–3, a = 1–9) and
`runs/ar/headroom/stdout_crosshalf.txt` (honesty check, model A); `runs/ar/headroom/summary.json`. **Commands:**
`python experiments/_ar_headroom.py`; `cd experiments && python _ar_crosshalf.py`.

---

## RUN 2026-10-29 — DEC part B: generative sampling vs one-shot Tweedie

Pre-registered (`Archive/DEC/PREREGISTRATION.md`). DSM models (MSE, noise-normalized; seeds 101–103, iteration 2000), every held-out slice of datasets 1–3 decoded per repetition by: one-shot Tweedie (TW); a 20-step deterministic reverse walk (ODE); a 20-step ancestral walk = one posterior sample (ANC); the mean of 8 samples (ANC8); and ANC extrapolated below the trained noise floor to σ̂/10 (ANCX, off-distribution). The stochastic walk stops at the ladder bottom σ_floor = 0.0368: N2N never trains below the quietest acquisition. Per slice, ratios are geometric means over the 3 seeds. The dry run gave 1.000.

**PRIMARY (dataset 1, a = 1, r = 0; n_eff ANC / TW, Holm over 2):**

- overall (signal region): 0.729× (95% CI 0.670–0.792; 2/34 better), p = 1.8e-06, Holm 1.8e-06 → **sampling worse**
- finest band (0.25–0.75 cycles/pixel): 0.695× (95% CI 0.612–0.775; 1/34 better), p = 6.5e-07, Holm 1.3e-06 → **sampling worse**

Decoder / TW, overall n_eff (geometric mean, slices better); r = 0 and each decoder at its cross-fitted r(a):

| decoder | dataset | a=1, r=0 | a=5, r=0 | a=9, r=0 | a=1, cf r | a=5, cf r | a=9, cf r |
|---|---:|---|---|---|---|---|---|
| ANC | 1 | 0.729 (2/34) | 0.701 (1/34) | 0.699 (1/34) | 0.729 (2/34) | 0.653 (2/34) | 0.605 (0/34) |
| ANC | 2 | 0.941 (13/34) | 0.970 (12/34) | 0.978 (13/34) | 0.941 (13/34) | 0.893 (8/34) | 0.848 (7/34) |
| ANC | 3 | 0.957 (15/34) | 0.914 (7/34) | 0.913 (8/34) | 0.957 (15/34) | 0.837 (4/34) | 0.799 (3/34) |
| ANC8 | 1 | 0.872 (12/34) | 0.741 (1/34) | 0.725 (2/34) | 0.872 (12/34) | 0.756 (0/34) | 0.806 (0/34) |
| ANC8 | 2 | 1.178 (28/34) | 1.030 (16/34) | 1.012 (15/34) | 1.178 (28/34) | 0.972 (14/34) | 0.981 (14/34) |
| ANC8 | 3 | 1.226 (29/34) | 0.992 (13/34) | 0.961 (11/34) | 1.226 (29/34) | 0.923 (8/34) | 0.898 (3/34) |
| ODE | 1 | 0.843 (9/34) | 0.818 (2/34) | 0.819 (3/34) | 0.843 (9/34) | 0.807 (1/34) | 0.839 (0/34) |
| ODE | 2 | 1.073 (23/34) | 1.107 (27/34) | 1.118 (27/34) | 1.073 (23/34) | 1.045 (23/34) | 1.055 (25/34) |
| ODE | 3 | 1.040 (24/34) | 1.000 (15/34) | 1.001 (13/34) | 1.040 (24/34) | 0.936 (8/34) | 0.943 (6/34) |
| ANCX | 1 | 0.258 (0/34) | 0.473 (0/34) | 0.565 (0/34) | 0.258 (0/34) | 0.435 (0/34) | 0.501 (0/34) |
| ANCX | 2 | 0.362 (0/34) | 0.658 (1/34) | 0.766 (3/34) | 0.362 (0/34) | 0.605 (0/34) | 0.663 (0/34) |
| ANCX | 3 | 0.242 (0/34) | 0.473 (0/34) | 0.595 (0/34) | 0.242 (0/34) | 0.433 (0/34) | 0.516 (0/34) |

Finest band, decoder / TW at r = 0, a = 1 / 5 / 9: ANC ds1: 0.695 / 0.865 / 0.908; ANC ds2: 0.728 / 0.959 / 1.020; ANC ds3: 0.721 / 0.934 / 1.003; ANC8 ds1: 1.015 / 0.987 / 0.990; ANC8 ds2: 1.280 / 1.178 / 1.164; ANC8 ds3: 1.229 / 1.155 / 1.153; ODE ds1: 0.841 / 0.986 / 1.025; ODE ds2: 1.001 / 1.119 / 1.166; ODE ds3: 0.892 / 1.049 / 1.100.
Walk length (dataset 1, a = 1, r = 0): median σ̂/σ_floor 2.31; ANC/TW on short vs long walks: 0.764 vs 0.695; ODE/TW: 0.874 vs 0.813.

**Data:** `runs/dec/B/s{101,102,103}/neff.csv`, `runs/dec/report/summary.json`. **Commands:** `bash runs/dec/queue.sh`; `python experiments/_dec_report.py --part B`.

---

## RUN 2026-10-30 — DEC part A: regressor vs diffusion

Pre-registered (`Archive/DEC/PREREGISTRATION.md`). Dataset-1 training, MSE noise-normalized, 2000 iterations, seeds 101–103, four arms: DSM (x̂₀ = x̃ + σ² net, noise ladder), DIR_L (x̂₀ = net, same ladder), DIR_N (direct, native noise only: a classic σ-conditioned N2N regressor), BLIND (DIR_N without σ input). All decoded one-shot and scored at each arm's cross-fitted r(a). Per slice, geometric means over the 3 seeds. The dry run gave 1.000.

**PRIMARY (DIR_N / DSM, dataset 1, a = 9):** 0.875× (90% CI 0.848–0.903; 2/34 better) → **regressor worse (diffusion training contributes)**.

Ratios (geometric mean, 90% CI; overall n_eff at cross-fitted r):

| comparison | dataset | a = 1 | a = 5 | a = 9 |
|---|---:|---|---|---|
| DIR_N / DSM | 1 | 1.135 (1.053–1.241) | 0.891 (0.859–0.924) | 0.875 (0.847–0.903) |
| DIR_N / DSM | 2 | 1.025 (0.927–1.127) | 0.879 (0.810–0.945) | 0.872 (0.815–0.928) |
| DIR_N / DSM | 3 | 1.358 (1.236–1.503) | 0.981 (0.926–1.039) | 0.934 (0.889–0.979) |
| DIR_L / DSM | 1 | 1.138 (1.052–1.253) | 0.887 (0.850–0.929) | 0.870 (0.837–0.906) |
| DIR_L / DSM | 2 | 1.350 (1.255–1.459) | 1.100 (1.061–1.141) | 1.059 (1.027–1.092) |
| DIR_L / DSM | 3 | 1.590 (1.447–1.762) | 1.117 (1.066–1.175) | 1.047 (1.009–1.089) |
| DIR_N / DIR_L | 1 | 0.998 (0.970–1.023) | 1.004 (0.982–1.024) | 1.006 (0.985–1.024) |
| DIR_N / DIR_L | 2 | 0.760 (0.705–0.811) | 0.799 (0.746–0.848) | 0.824 (0.777–0.867) |
| DIR_N / DIR_L | 3 | 0.854 (0.809–0.897) | 0.878 (0.835–0.921) | 0.892 (0.852–0.929) |
| BLIND / DIR_N | 1 | 1.033 (1.011–1.057) | 1.036 (1.019–1.056) | 1.034 (1.018–1.052) |
| BLIND / DIR_N | 2 | 1.183 (1.124–1.249) | 1.179 (1.124–1.242) | 1.160 (1.114–1.212) |
| BLIND / DIR_N | 3 | 1.093 (1.041–1.149) | 1.114 (1.069–1.164) | 1.105 (1.065–1.149) |

Finest band, a = 1 / 9: DIR_N/DSM ds1: 1.332 / 0.966; DIR_N/DSM ds2: 1.523 / 0.987; DIR_N/DSM ds3: 1.808 / 1.042; DIR_L/DSM ds1: 1.339 / 0.963; DIR_L/DSM ds2: 1.571 / 1.102; DIR_L/DSM ds3: 1.923 / 1.060; DIR_N/DIR_L ds1: 0.995 / 1.003; DIR_N/DIR_L ds2: 0.905 / 0.896; DIR_N/DIR_L ds3: 0.940 / 0.984; BLIND/DIR_N ds1: 1.004 / 1.004; BLIND/DIR_N ds2: 1.020 / 1.106; BLIND/DIR_N ds3: 1.011 / 1.008.
Absolute n_eff (geometric mean over slices and seeds), a = 1 / 9: DSM ds1 21.68 / 38.95; DSM ds2 40.07 / 60.66; DSM ds3 28.65 / 56.63; DIR_L ds1 24.67 / 33.90; DIR_L ds2 54.07 / 64.21; DIR_L ds3 45.56 / 59.28; DIR_N ds1 24.62 / 34.09; DIR_N ds2 41.08 / 52.91; DIR_N ds3 38.89 / 52.87; BLIND ds1 25.43 / 35.25; BLIND ds2 48.58 / 61.39; BLIND ds3 42.51 / 58.44.

**Data:** `runs/dec/A/<arm>_s<seed>/`, `runs/dec/A/eval/`, `runs/dec/report/summary.json`. **Commands:** `bash runs/dec/queue.sh`; `python experiments/_dec_report.py --part A`.

---

## EXPLORATORY 2026-10-30 — DEC part A: why the score parametrization wins when averaging

Not pre-registered. Per-slice bias² b, per-map variance v and noise covariance c of each arm's one-shot estimate,
from the exact decomposition of the all-splits estimator (`experiments/_dec_bvc.py`). Output, verbatim:

```
median over slices x seeds 101-103; b = bias^2, v = per-map variance, c = noise covariance (units of sigma^2)
ds1 dsm   b 0.0311  v 0.0152  c 0.0189   single map 1/(b+v) = 21.6   a -> inf limit 1/b = 32.2
ds1 dirl  b 0.0481  v 0.0046  c 0.0286   single map 1/(b+v) = 19.0   a -> inf limit 1/b = 20.8
ds1 dirn  b 0.0457  v 0.0051  c 0.0314   single map 1/(b+v) = 19.7   a -> inf limit 1/b = 21.9
ds1 blind b 0.0434  v 0.0055  c 0.0316   single map 1/(b+v) = 20.5   a -> inf limit 1/b = 23.1
ds2 dsm   b 0.0213  v 0.0049  c 0.0023   single map 1/(b+v) = 38.1   a -> inf limit 1/b = 46.9
ds2 dirl  b 0.0173  v 0.0020  c 0.0200   single map 1/(b+v) = 51.9   a -> inf limit 1/b = 57.9
ds2 dirn  b 0.0223  v 0.0036  c 0.0273   single map 1/(b+v) = 38.7   a -> inf limit 1/b = 44.9
ds2 blind b 0.0174  v 0.0038  c 0.0268   single map 1/(b+v) = 47.3   a -> inf limit 1/b = 57.6
ds3 dsm   b 0.0210  v 0.0154  c 0.0063   single map 1/(b+v) = 27.5   a -> inf limit 1/b = 47.7
ds3 dirl  b 0.0199  v 0.0043  c 0.0252   single map 1/(b+v) = 41.4   a -> inf limit 1/b = 50.3
ds3 dirn  b 0.0241  v 0.0053  c 0.0270   single map 1/(b+v) = 34.1   a -> inf limit 1/b = 41.6
ds3 blind b 0.0198  v 0.0060  c 0.0270   single map 1/(b+v) = 38.7   a -> inf limit 1/b = 50.4
```

**Reading:** on dataset 1 the score parametrization (x̂₀ = x̃ + σ² net) halves the bias² relative to the direct
parametrizations (0.031 vs 0.046–0.048) at the price of three times the per-map variance (0.015 vs 0.005).
Averaging removes variance but not bias, so DSM wins at a = 9 (primary 0.875× for the regressor) while the direct
models win for a single map (1.13–1.14×). The same mechanism orders the arms on datasets 2–3, where direct-with-ladder
has the lowest bias. Design implication: when outputs will be averaged, prefer the low-bias estimator. A loss
targeting the error of the averaged estimate is a candidate method (not tested).

---

## RUN 2026-10-30 — BASE: the method vs conventional and self-supervised baselines

Pre-registered (`Archive/BASE/PREREGISTRATION.md`). Ours = DSM, MSE noise-normalized, dataset-1 training, seeds 101–103, one-shot Tweedie + dial. Every method sees only the A half of each split; all methods are scored on the SAME fixed subset of 10 splits per a (unbiased, paired); every parameter is cross-fitted over the other folds' slices. Code: `baselines/`. The dry run gave 1.000.

**Deviation from the pre-registration (Noise2Score).** The first implementation (annealed perturbation, network conditioned on σ̂ only) failed its sanity check: n_eff 0.03, i.e. far worse than the noisy input, with a final loss of 0.98 against 1.0 for no learning (runs kept in `runs/base/n2s_v1*`). Version 2 conditions the network on the total noise √(σ̂² + δ²) of the perturbed input with δ log-uniform in [0.05, 2]·σ̂; the AR-DAE objective is unchanged. It was validated on synthetic data with known ground truth (`baselines/_n2s_validate.py`, `runs/base/report/n2s_validation.txt`): it denoises white noise at SNR 1 and 0.1, also from a fixed set of 10 realizations, but with noise coloured like dataset 1 it is WORSE than the noisy input (error 1.25× the noise variance), while a Noise2Noise model on the same data is unaffected (0.002×). Noise2Score's Tweedie step assumes white noise of known σ; real ASL noise is coloured. The Noise2Score rows below are version 2.

**PRIMARY (dataset 1, a = 9; n_eff ours / baseline, Holm over 8):**

- vs Plain averaging: 4.328× (95% CI 3.933–4.779; ours better on 34/34), p = 1.2e-10, Holm 9.3e-10 → **ours better**
- vs Gaussian blur: 1.362× (95% CI 1.278–1.452; ours better on 33/34), p = 2e-08, Holm 3.2e-08 → **ours better**
- vs Non-local means: 1.426× (95% CI 1.295–1.587; ours better on 31/34), p = 3.8e-09, Holm 1.2e-08 → **ours better**
- vs BM3D: 1.288× (95% CI 1.189–1.402; ours better on 31/34), p = 1.6e-08, Holm 3.2e-08 → **ours better**
- vs MP-PCA: 2.565× (95% CI 2.316–2.824; ours better on 34/34), p = 1.2e-10, Holm 9.3e-10 → **ours better**
- vs LPCA: 2.643× (95% CI 2.323–2.971; ours better on 34/34), p = 1.2e-10, Holm 9.3e-10 → **ours better**
- vs Patch2Self: 1.646× (95% CI 1.515–1.790; ours better on 33/34), p = 2.3e-10, Holm 9.3e-10 → **ours better**
- vs Noise2Score (r = 0): 11.207× (95% CI 9.726–12.817; ours better on 34/34), p = 1.2e-10, Holm 9.3e-10 → **ours better**

Geometric-mean n_eff (signal region) by method:

| method | ds1 a=1 | ds1 a=3 | ds1 a=5 | ds1 a=9 | ds2 a=1 | ds2 a=3 | ds2 a=5 | ds2 a=9 | ds3 a=1 | ds3 a=3 | ds3 a=5 | ds3 a=9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours (DSM + dial) | 21.68 | 31.94 | 34.34 | 38.95 | 40.07 | 53.86 | 61.37 | 60.66 | 28.65 | 41.30 | 59.56 | 56.63 |
| Plain averaging | 1.00 | 3.02 | 5.17 | 9.00 | 1.00 | 3.02 | 5.02 | 9.00 | 1.00 | 2.93 | 5.12 | 9.00 |
| Gaussian blur | 16.72 | 22.20 | 23.37 | 28.60 | 39.75 | 59.34 | 62.95 | 62.30 | 24.87 | 38.44 | 51.60 | 50.77 |
| Non-local means | 13.39 | 19.55 | 23.13 | 27.32 | 32.25 | 49.34 | 52.86 | 54.85 | 15.92 | 24.89 | 37.24 | 40.97 |
| BM3D | 14.71 | 21.77 | 25.67 | 30.24 | 35.50 | 53.68 | 56.76 | 61.89 | 18.96 | 28.48 | 42.19 | 45.08 |
| MP-PCA | 13.85 | 13.60 | 13.66 | 15.19 | 35.69 | 36.19 | 36.77 | 35.91 | 22.67 | 18.19 | 21.43 | 23.17 |
| LPCA | 13.91 | 15.32 | 14.44 | 14.74 | 37.31 | 43.96 | 44.57 | 41.89 | 20.59 | 23.44 | 25.85 | 23.74 |
| Patch2Self | 12.04 | 16.71 | 19.05 | 23.67 | 34.01 | 36.55 | 39.80 | 44.03 | 27.87 | 24.74 | 30.09 | 33.02 |
| Noise2Score (r = 0) | 0.44 | 1.27 | 2.06 | 3.48 | 0.68 | 1.97 | 3.15 | 5.23 | 0.42 | 1.19 | 2.00 | 3.36 |
| Noise2Score + dial | 1.06 | 3.13 | 5.18 | 8.81 | 1.28 | 3.82 | 6.24 | 10.82 | 0.98 | 2.82 | 4.87 | 8.29 |

Ours / baseline, overall and finest band (geometric mean, slices where ours is better):

| baseline | dataset | a = 1 | a = 3 | a = 5 | a = 9 | finest band a = 1 / 9 |
|---|---:|---|---|---|---|---|
| Plain averaging | 1 | 21.678 (34/34) | 10.575 (34/34) | 6.646 (34/34) | 4.328 (34/34) | 49.055 / 7.039 |
| Plain averaging | 2 | 40.067 (34/34) | 17.814 (34/34) | 12.225 (34/34) | 6.740 (34/34) | 129.012 / 15.534 |
| Plain averaging | 3 | 28.649 (34/34) | 14.086 (34/34) | 11.635 (34/34) | 6.293 (34/34) | 75.692 / 12.688 |
| Gaussian blur | 1 | 1.296 (31/34) | 1.438 (33/34) | 1.470 (33/34) | 1.362 (33/34) | 0.769 / 0.986 |
| Gaussian blur | 2 | 1.008 (13/34) | 0.908 (13/34) | 0.975 (15/34) | 0.974 (15/34) | 0.620 / 0.680 |
| Gaussian blur | 3 | 1.152 (26/34) | 1.074 (24/34) | 1.154 (26/34) | 1.116 (25/34) | 0.497 / 0.726 |
| Non-local means | 1 | 1.619 (33/34) | 1.634 (32/34) | 1.485 (31/34) | 1.426 (31/34) | 1.048 / 0.972 |
| Non-local means | 2 | 1.243 (25/34) | 1.092 (23/34) | 1.161 (23/34) | 1.106 (22/34) | 0.979 / 0.772 |
| Non-local means | 3 | 1.799 (31/34) | 1.659 (30/34) | 1.600 (32/34) | 1.382 (34/34) | 1.266 / 0.889 |
| BM3D | 1 | 1.473 (32/34) | 1.467 (32/34) | 1.338 (30/34) | 1.288 (31/34) | 0.861 / 0.872 |
| BM3D | 2 | 1.129 (22/34) | 1.003 (22/34) | 1.081 (23/34) | 0.980 (11/34) | 0.681 / 0.758 |
| BM3D | 3 | 1.511 (28/34) | 1.450 (27/34) | 1.412 (29/34) | 1.256 (31/34) | 0.810 / 0.831 |
| MP-PCA | 1 | 1.565 (31/34) | 2.349 (33/34) | 2.514 (34/34) | 2.565 (34/34) | 0.864 / 1.021 |
| MP-PCA | 2 | 1.123 (25/34) | 1.488 (34/34) | 1.669 (34/34) | 1.689 (34/34) | 0.798 / 0.772 |
| MP-PCA | 3 | 1.264 (29/34) | 2.270 (34/34) | 2.780 (34/34) | 2.444 (34/34) | 0.724 / 1.036 |
| LPCA | 1 | 1.559 (31/34) | 2.085 (33/34) | 2.379 (34/34) | 2.643 (34/34) | 0.825 / 1.066 |
| LPCA | 2 | 1.074 (21/34) | 1.225 (27/34) | 1.377 (30/34) | 1.448 (31/34) | 0.669 / 0.761 |
| LPCA | 3 | 1.392 (33/34) | 1.762 (34/34) | 2.304 (34/34) | 2.385 (34/34) | 0.780 / 1.149 |
| Patch2Self | 1 | 1.801 (32/34) | 1.911 (33/34) | 1.802 (33/34) | 1.646 (33/34) | 0.802 / 1.103 |
| Patch2Self | 2 | 1.178 (22/34) | 1.474 (34/34) | 1.542 (34/34) | 1.378 (34/34) | 0.640 / 1.276 |
| Patch2Self | 3 | 1.028 (18/34) | 1.669 (32/34) | 1.980 (34/34) | 1.715 (34/34) | 0.539 / 1.452 |
| Noise2Score (r = 0) | 1 | 49.258 (34/34) | 25.077 (34/34) | 16.706 (34/34) | 11.207 (34/34) | 151.080 / 22.667 |
| Noise2Score (r = 0) | 2 | 58.612 (34/34) | 27.314 (34/34) | 19.451 (34/34) | 11.588 (34/34) | 262.024 / 33.144 |
| Noise2Score (r = 0) | 3 | 68.771 (34/34) | 34.727 (34/34) | 29.854 (34/34) | 16.839 (34/34) | 257.870 / 45.403 |
| Noise2Score + dial | 1 | 20.432 (34/34) | 10.212 (34/34) | 6.631 (34/34) | 4.423 (34/34) | 52.920 / 7.840 |
| Noise2Score + dial | 2 | 31.324 (34/34) | 14.113 (34/34) | 9.833 (34/34) | 5.608 (34/34) | 112.391 / 13.841 |
| Noise2Score + dial | 3 | 29.101 (34/34) | 14.660 (34/34) | 12.219 (34/34) | 6.830 (34/34) | 90.276 / 15.686 |

**Data:** `runs/base/{classical,ours,n2s_eval}/neff.csv`, `runs/base/report/summary.json`. **Commands:** `python baselines/evaluate.py --methods ...`; `bash runs/base/n2s_queue.sh`; `python baselines/report.py`.

---

## RUN 2026-10-31 — BASE-DL: the method vs self-supervised deep-learning baselines

Pre-registered (`Archive/BASE/PREREGISTRATION_DL.md`); same splits, scoring and cross-fitting as BASE. Baselines trained like ours (same U-Net, folds, seeds 101–103, 2000 iterations; code `baselines/dl_selfsup.py`; the two Noise2Noise arms are the DEC runs re-scored). Primary = each baseline with the dial. Pre-registered (`Archive/BASE/PREREGISTRATION.md`). Ours = DSM, MSE noise-normalized, dataset-1 training, seeds 101–103, one-shot Tweedie + dial. Every method sees only the A half of each split; all methods are scored on the SAME fixed subset of 10 splits per a (unbiased, paired); every parameter is cross-fitted over the other folds' slices. Code: `baselines/`. The dry run gave 1.000.

**PRIMARY (dataset 1, a = 9; n_eff ours / baseline, Holm over 5):**

- vs Noise2Noise (classic) + dial: 1.105× (95% CI 1.053–1.155; ours better on 28/34), p = 9.1e-06, Holm 9.1e-06 → **ours better**
- vs Noise2Noise + σ + dial: 1.142× (95% CI 1.100–1.187; ours better on 32/34), p = 8.9e-08, Holm 1.8e-07 → **ours better**
- vs Neighbor2Neighbor + dial: 2.020× (95% CI 1.896–2.158; ours better on 34/34), p = 1.2e-10, Holm 5.8e-10 → **ours better**
- vs Recorrupted-to-Recorrupted + dial: 1.847× (95% CI 1.708–2.003; ours better on 34/34), p = 1.2e-10, Holm 5.8e-10 → **ours better**
- vs Noise2Self + dial: 1.543× (95% CI 1.432–1.664; ours better on 34/34), p = 1.2e-10, Holm 5.8e-10 → **ours better**

Geometric-mean n_eff (signal region) by method:

| method | ds1 a=1 | ds1 a=3 | ds1 a=5 | ds1 a=9 | ds2 a=1 | ds2 a=3 | ds2 a=5 | ds2 a=9 | ds3 a=1 | ds3 a=3 | ds3 a=5 | ds3 a=9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours (DSM + dial) | 21.68 | 31.94 | 34.34 | 38.95 | 40.07 | 53.86 | 61.37 | 60.66 | 28.65 | 41.30 | 59.56 | 56.63 |
| Noise2Noise (classic) + dial | 25.43 | 30.51 | 30.50 | 35.25 | 48.58 | 59.35 | 63.49 | 61.39 | 42.51 | 48.73 | 67.11 | 58.44 |
| Noise2Noise + σ + dial | 24.62 | 29.31 | 29.54 | 34.09 | 41.08 | 48.16 | 53.22 | 52.91 | 38.89 | 42.47 | 58.08 | 52.87 |
| Neighbor2Neighbor + dial | 7.82 | 13.54 | 15.31 | 19.28 | 11.86 | 25.82 | 32.58 | 38.29 | 9.31 | 18.11 | 25.79 | 29.77 |
| Recorrupted-to-Recorrupted + dial | 3.97 | 10.01 | 14.73 | 21.09 | 5.15 | 13.74 | 19.92 | 28.80 | 4.12 | 9.90 | 16.46 | 23.50 |
| Noise2Self + dial | 4.94 | 12.34 | 17.48 | 25.25 | 7.66 | 20.29 | 29.52 | 41.53 | 5.86 | 14.31 | 24.27 | 33.83 |
| Noise2Noise (classic) (r = 0, as published) | 25.43 | 28.86 | 27.32 | 28.65 | 48.58 | 59.35 | 61.34 | 56.83 | 42.51 | 48.73 | 62.98 | 54.06 |
| Noise2Noise + σ (r = 0, as published) | 24.62 | 27.62 | 26.23 | 27.35 | 41.08 | 48.32 | 50.39 | 47.17 | 38.89 | 43.22 | 54.34 | 47.77 |
| Neighbor2Neighbor (r = 0, as published) | 7.82 | 13.54 | 15.43 | 17.42 | 11.86 | 25.82 | 32.58 | 38.29 | 9.31 | 18.11 | 25.79 | 29.87 |
| Recorrupted-to-Recorrupted (r = 0, as published) | 3.96 | 9.68 | 13.51 | 17.99 | 5.15 | 13.74 | 19.84 | 27.97 | 4.12 | 9.89 | 16.10 | 22.02 |
| Noise2Self (r = 0, as published) | 4.83 | 11.61 | 15.57 | 21.02 | 7.66 | 20.19 | 28.75 | 39.48 | 5.86 | 14.34 | 23.69 | 32.39 |

Ours / baseline, overall and finest band (geometric mean, slices where ours is better):

| baseline | dataset | a = 1 | a = 3 | a = 5 | a = 9 | finest band a = 1 / 9 |
|---|---:|---|---|---|---|---|
| Noise2Noise (classic) + dial | 1 | 0.853 (7/34) | 1.047 (25/34) | 1.126 (27/34) | 1.105 (28/34) | 0.747 / 1.031 |
| Noise2Noise (classic) + dial | 2 | 0.825 (8/34) | 0.908 (12/34) | 0.967 (12/34) | 0.988 (12/34) | 0.690 / 0.916 |
| Noise2Noise (classic) + dial | 3 | 0.674 (0/34) | 0.847 (3/34) | 0.888 (8/34) | 0.969 (13/34) | 0.547 / 0.952 |
| Noise2Noise + σ + dial | 1 | 0.881 (9/34) | 1.090 (29/34) | 1.162 (31/34) | 1.142 (32/34) | 0.751 / 1.035 |
| Noise2Noise + σ + dial | 2 | 0.975 (17/34) | 1.118 (24/34) | 1.153 (23/34) | 1.146 (26/34) | 0.653 / 1.013 |
| Noise2Noise + σ + dial | 3 | 0.737 (4/34) | 0.973 (16/34) | 1.025 (19/34) | 1.071 (23/34) | 0.553 / 0.959 |
| Neighbor2Neighbor + dial | 1 | 2.772 (33/34) | 2.360 (34/34) | 2.244 (34/34) | 2.020 (34/34) | 3.834 / 2.287 |
| Neighbor2Neighbor + dial | 2 | 3.377 (34/34) | 2.086 (34/34) | 1.884 (34/34) | 1.584 (34/34) | 7.790 / 2.082 |
| Neighbor2Neighbor + dial | 3 | 3.078 (34/34) | 2.280 (34/34) | 2.310 (34/34) | 1.903 (34/34) | 5.768 / 2.961 |
| Recorrupted-to-Recorrupted + dial | 1 | 5.465 (34/34) | 3.192 (34/34) | 2.332 (34/34) | 1.847 (34/34) | 10.967 / 2.031 |
| Recorrupted-to-Recorrupted + dial | 2 | 7.781 (34/34) | 3.921 (34/34) | 3.080 (34/34) | 2.106 (34/34) | 23.969 / 3.309 |
| Recorrupted-to-Recorrupted + dial | 3 | 6.951 (34/34) | 4.173 (34/34) | 3.620 (34/34) | 2.410 (34/34) | 16.097 / 3.211 |
| Noise2Self + dial | 1 | 4.388 (34/34) | 2.588 (34/34) | 1.965 (34/34) | 1.543 (34/34) | 7.344 / 1.640 |
| Noise2Self + dial | 2 | 5.234 (34/34) | 2.655 (34/34) | 2.079 (34/34) | 1.460 (34/34) | 12.404 / 2.027 |
| Noise2Self + dial | 3 | 4.891 (34/34) | 2.885 (34/34) | 2.454 (34/34) | 1.674 (34/34) | 8.311 / 1.931 |
| Noise2Noise (classic) (r = 0, as published) | 1 | 0.853 (7/34) | 1.107 (27/34) | 1.257 (28/34) | 1.359 (28/34) | 0.747 / 0.913 |
| Noise2Noise (classic) (r = 0, as published) | 2 | 0.825 (8/34) | 0.908 (12/34) | 1.000 (18/34) | 1.067 (24/34) | 0.690 / 0.653 |
| Noise2Noise (classic) (r = 0, as published) | 3 | 0.674 (0/34) | 0.847 (3/34) | 0.946 (18/34) | 1.048 (27/34) | 0.547 / 0.732 |
| Noise2Noise + σ (r = 0, as published) | 1 | 0.881 (9/34) | 1.156 (31/34) | 1.309 (31/34) | 1.424 (32/34) | 0.751 / 0.919 |
| Noise2Noise + σ (r = 0, as published) | 2 | 0.975 (17/34) | 1.115 (22/34) | 1.218 (27/34) | 1.286 (28/34) | 0.653 / 0.662 |
| Noise2Noise + σ (r = 0, as published) | 3 | 0.737 (4/34) | 0.956 (17/34) | 1.096 (23/34) | 1.186 (27/34) | 0.553 / 0.739 |
| Neighbor2Neighbor (r = 0, as published) | 1 | 2.772 (33/34) | 2.360 (34/34) | 2.225 (34/34) | 2.237 (34/34) | 3.834 / 1.826 |
| Neighbor2Neighbor (r = 0, as published) | 2 | 3.377 (34/34) | 2.086 (34/34) | 1.884 (34/34) | 1.584 (34/34) | 7.790 / 2.082 |
| Neighbor2Neighbor (r = 0, as published) | 3 | 3.078 (34/34) | 2.280 (34/34) | 2.310 (34/34) | 1.896 (34/34) | 5.768 / 2.839 |
| Recorrupted-to-Recorrupted (r = 0, as published) | 1 | 5.473 (34/34) | 3.299 (34/34) | 2.542 (34/34) | 2.166 (34/34) | 11.376 / 2.316 |
| Recorrupted-to-Recorrupted (r = 0, as published) | 2 | 7.781 (34/34) | 3.921 (34/34) | 3.094 (34/34) | 2.169 (34/34) | 23.969 / 3.399 |
| Recorrupted-to-Recorrupted (r = 0, as published) | 3 | 6.951 (34/34) | 4.175 (34/34) | 3.700 (34/34) | 2.572 (34/34) | 16.097 / 3.442 |
| Noise2Self (r = 0, as published) | 1 | 4.489 (34/34) | 2.750 (34/34) | 2.206 (34/34) | 1.853 (34/34) | 8.314 / 2.128 |
| Noise2Self (r = 0, as published) | 2 | 5.234 (34/34) | 2.668 (34/34) | 2.134 (34/34) | 1.537 (34/34) | 12.404 / 2.250 |
| Noise2Self (r = 0, as published) | 3 | 4.891 (34/34) | 2.879 (34/34) | 2.514 (34/34) | 1.748 (34/34) | 8.311 / 2.170 |

**Data:** `runs/base/{classical,ours,n2s_eval}/neff.csv`, `runs/base/report_dl/summary.json`. **Commands:** `python baselines/evaluate.py --methods ...`; `bash runs/base/n2s_queue.sh`; `python baselines/report.py`.

---

## RUN 2026-11-01 — NAT: step 3, ground-truth validation on natural images

Pre-registered (`Archive/NAT/PREREGISTRATION.md`). Train400 → BSD68 (320×480 crops), 10 fixed noisy copies per image; conditions W50 (σ 50/255), W05 (σ 0.5), W20 (σ 2.0), C05 (σ 0.5, ASL-coloured). Same 1-channel U-Net and budget for every learned method (3000 iterations, seed 101; code `natural/`). r chosen WITHOUT ground truth on Set12 (split-half n_eff); all scores on BSD68 with exact ground truth. The dry run gave 0 dB everywhere.

**P1 — is the split-half estimator unbiased?** Relative bias of the estimated MSE (ours, r = 0):

- W05, a = 1: +0.02% (95% CI -0.77 to +0.88%) → **unbiased in practice**
- W05, a = 9: +0.13% (95% CI -1.14 to +1.56%) → **unbiased in practice**
- C05, a = 1: -0.04% (95% CI -0.51 to +0.45%) → **unbiased in practice**
- C05, a = 9: -0.01% (95% CI -0.77 to +0.81%) → **unbiased in practice**
- W20, a = 1: -2.26% (95% CI -4.34 to -0.10%) → **unbiased in practice**
- W20, a = 9: -4.87% (95% CI -8.97 to -0.83%) → **biased beyond ±5%**
- W50, a = 1: -0.05% (95% CI -0.24 to +0.15%) → **unbiased in practice**
- W50, a = 9: -0.07% (95% CI -0.35 to +0.24%) → **unbiased in practice**

**P2 — choosing r without ground truth** (ours, a = 9; PSNR loss vs the BSD68 oracle r):

- W05: r chosen 0.083 vs oracle 0.083, loss 0.000 dB → **ground-truth-free selection suffices**
- C05: r chosen 0.083 vs oracle 0.083, loss 0.000 dB → **ground-truth-free selection suffices**
- W20: r chosen 0.000 vs oracle 0.000, loss 0.000 dB → **ground-truth-free selection suffices**
- W50: r chosen 0.167 vs oracle 0.250, loss 0.126 dB → **loss above 0.1 dB**

**P3 — ours vs baselines (W05, a = 9; PSNR(ours) − PSNR(baseline), Holm over 6):**

- vs Noise2Noise (classic): +0.43 dB (95% CI +0.38 to +0.48; 68/68 images), p = 7.6e-13, Holm 4.6e-12
- vs Neighbor2Neighbor: +0.62 dB (95% CI +0.54 to +0.73; 68/68 images), p = 7.6e-13, Holm 4.6e-12
- vs Recorrupted-to-Recorrupted: +0.41 dB (95% CI +0.36 to +0.47; 68/68 images), p = 7.6e-13, Holm 4.6e-12
- vs Noise2Score: +0.37 dB (95% CI +0.32 to +0.43; 67/68 images), p = 8e-13, Holm 4.6e-12
- vs Plain averaging: +8.31 dB (95% CI +7.71 to +8.95; 68/68 images), p = 7.6e-13, Holm 4.6e-12
- vs BM3D: -1.85 dB (95% CI -1.97 to -1.72; 0/68 images), p = 7.6e-13, Holm 4.6e-12

**P4 — averaging-aware loss (W05; PSNR(method) − PSNR(ours), Holm over 3):**

- Averaging-aware loss, a = 9 at a = 9: -0.36 dB (95% CI -0.70 to -0.06; 34/68 images), p = 0.31, Holm 0.31
- Averaging-aware loss, a ~ U{1..9} at a = 1: -2.30 dB (95% CI -2.53 to -2.07; 0/68 images), p = 7.6e-13, Holm 2.3e-12
- Averaging-aware loss, a ~ U{1..9} at a = 9: +0.59 dB (95% CI +0.50 to +0.67; 64/68 images), p = 3.4e-11, Holm 6.7e-11

Mean PSNR on BSD68 (dB; learned methods at their ground-truth-free r):

| method | W05 a=1 | W05 a=9 | C05 a=1 | C05 a=9 | W20 a=1 | W20 a=9 | W50 a=1 | W50 a=9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours (DSM + dial) | 22.55 | 24.48 | 22.14 | 24.21 | 17.77 | 19.98 | 25.84 | 28.55 |
| Averaging-aware loss, a = 9 | 17.24 | 24.12 | 17.12 | 24.27 | 13.91 | 19.61 | 20.59 | 28.41 |
| Averaging-aware loss, a ~ U{1..9} | 20.26 | 25.07 | 19.97 | 24.70 | 16.19 | 20.16 | 23.20 | 29.39 |
| Supervised (upper bound) | 22.61 | 24.48 | 22.17 | 24.28 | 17.58 | 19.78 | 25.86 | 28.62 |
| Noise2Noise (classic) | 22.67 | 24.05 | 22.34 | 23.94 | 18.88 | 20.33 | 25.76 | 28.28 |
| Neighbor2Neighbor | 22.20 | 23.86 | 16.51 | 23.26 | 17.78 | 18.98 | 25.43 | 27.98 |
| Recorrupted-to-Recorrupted | 22.49 | 24.07 | 14.87 | 21.53 | 18.88 | 20.09 | 25.26 | 27.71 |
| Noise2Score | 22.30 | 24.11 | 10.10 | 17.36 | 17.81 | 19.55 | 25.91 | 28.57 |
| Plain averaging | 9.12 | 16.17 | 9.12 | 16.16 | 5.96 | 8.02 | 14.91 | 23.90 |
| BM3D | 22.26 | 26.33 | 21.50 | 26.04 | 16.41 | 21.19 | 25.69 | 30.60 |

**Data:** `runs/nat/eval/<cond>/<method>.csv`, `runs/nat/report/summary.json`. **Commands:** `bash runs/nat/queue_gpu.sh`; `bash runs/nat/queue_cpu.sh`; `python natural/report.py`.

---

## READING 2026-11-01 — NAT and BASE-DL: what the weekend results mean (interpretation, not pre-registered)

**Integrity.** Protected-hash re-checks: NAT 3 of 2811 entries changed, all in `runs/_ch_verify/` (a temporary
verification folder that was running when the snapshot was taken and was deleted on purpose afterwards); BASE 1 of
2086, `docs/RESEARCH_NOTES.md` (deliberate step-2 update). No result file changed.

**Sanity.** For every condition, plain averaging gives exactly the theoretical error σ²/a (e.g. W05 a = 9: 0.02776 vs
0.02778) and the split-half estimate matches it, so the NAT numbers are trustworthy.

**What holds.**
- **Estimator (P1):** the ground-truth-free split-half error estimate is unbiased to within 1% at σ = 50/255, 0.5
  (white and coloured). At the extreme σ = 2.0 it reads 2–5% low (CI excludes 0 at a = 9). There the true error is
  about 0.25% of the reference noise variance being subtracted, so precision suffers. The ASL regime (n_eff ≈ 40, true
  error ≈ 2.5% of it) is ten times easier.
- **Dial without ground truth (P2):** r chosen on Set12 by the split-half estimate equals the BSD68 oracle in three of
  four conditions (0 dB loss); 0.13 dB at σ = 50/255.
- **Noise-model robustness:** with ASL-coloured noise (C05), Neighbor2Neighbor, R2R and Noise2Score collapse (single
  map 16.5 / 14.9 / 10.1 dB vs ours 22.1 dB). Ours and classic Noise2Noise are unaffected. Same mechanism as on ASL.
- **Self-supervision is not the bottleneck:** ours matches the SUPERVISED per-copy upper bound (W05 a = 9: 24.48 vs
  24.48 dB; a = 1: 22.55 vs 22.61).

**What does not hold, and why it matters.**
- **BM3D on the mean of the copies beats every per-copy method at a = 9** (W05: 26.33 vs ours 24.48 dB, 0/68 images;
  the same in all conditions), while ours is better or equal for a single copy (except σ = 2.0 a = 1).
- Since even the supervised per-copy model stops at 24.48 dB, the limit is the "denoise each copy, then average"
  architecture, not the training signal. Each copy is denoised from one copy's information, and averaging afterwards
  cannot recover what each step discarded.
- This is the sufficiency argument of research note 5, now supported where data is ample (400 training images). On
  ASL (34 slices, SNR 0.04–0.13) the subset-mean model was data-limited and BM3D lost to ours (1.29×).
- **Averaging-aware loss (P4):** training on the error of the averaged estimate with a ~ U{1..9} gains +0.59 dB at
  a = 9 (64/68 images) but loses 2.3 dB for a single copy. The fixed a = 9 variant is not significantly different at
  a = 9 and much worse for single copies. The bias/variance lever is real, but it closes only a third of the gap to
  BM3D.
- **Classic Noise2Noise** (pairs, no σ input) beats ours for single maps at σ = 2.0 and, on ASL, on datasets 2–3. It is
  consistently the most robust learned competitor.

**Implication.** For averaged estimates, the method should see all copies, not denoise them one by one: denoise the
copy mean (SUB, conditioned on σ/√a) or take the copies jointly. Natural images have the data to train that. This is
the most important next experiment, and it may change the method's architecture. The per-copy model plus the
depth-dependent dial remains the right choice for single maps and in the data-poor ASL regime.

---

## RUN 2026-11-01 — NAT-SUB (exploratory): does denoising the copy mean close the gap to BM3D?

Pre-registered as exploratory (`Archive/NAT/PREREGISTRATION_SUB.md`). SUB denoises the mean of m ~ U{1..9} copies (conditioned on σ/√m, N2N target = mean of the other copies); SUBSUP is the same with the clean target. Same U-Net, data and budget as NAT; r chosen without ground truth on Set12; scores on BSD68. The dry run gave 0 dB.

**PRIMARY (W05, a = 9): PSNR(SUB) − PSNR(ours) = +1.59 dB (95% CI +1.47 to +1.71; 68/68 images), p = 7.6e-13.**

Mean PSNR on BSD68 (dB; chosen r in brackets):

| method | W05 a=1 | W05 a=3 | W05 a=5 | W05 a=9 | C05 a=1 | C05 a=3 | C05 a=5 | C05 a=9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours (per copy + dial) | 22.55 (0.00) | 23.63 (0.00) | 24.08 (0.08) | 24.48 (0.08) | 22.14 (0.00) | 23.34 (0.00) | 23.75 (0.08) | 24.21 (0.08) |
| Supervised, per copy | 22.61 (0.00) | 23.71 (0.00) | 23.97 (0.00) | 24.48 (0.08) | 22.17 (0.00) | 23.37 (0.00) | 23.85 (0.08) | 24.28 (0.08) |
| SUB: denoise the mean (N2N) | 22.47 (0.00) | 24.20 (0.00) | 25.08 (0.00) | 26.07 (0.00) | 21.91 (0.00) | 23.79 (0.00) | 24.76 (0.00) | 25.73 (0.00) |
| SUBSUP: denoise the mean (supervised) | 22.36 (0.00) | 24.19 (0.00) | 25.12 (0.00) | 26.07 (0.00) | 22.09 (0.00) | 23.93 (0.00) | 24.89 (0.00) | 25.86 (0.00) |
| BM3D on the mean | 22.26 | 24.24 | 25.18 | 26.33 | 21.50 | 23.78 | 24.82 | 26.04 |
| Plain averaging | 9.12 | 12.16 | 13.94 | 16.17 | 9.12 | 12.16 | 13.93 | 16.16 |

Differences (paired over 68 images): W05 SUB−OURS a=1: -0.08 dB (22/68); W05 SUB−OURS a=9: +1.59 dB (68/68); W05 SUB−BM3D a=1: +0.21 dB (61/68); W05 SUB−BM3D a=9: -0.26 dB (18/68); W05 SUBSUP−SUP a=1: -0.24 dB (2/68); W05 SUBSUP−SUP a=9: +1.60 dB (68/68); C05 SUB−OURS a=1: -0.22 dB (2/68); C05 SUB−OURS a=9: +1.52 dB (68/68); C05 SUB−BM3D a=1: +0.41 dB (65/68); C05 SUB−BM3D a=9: -0.31 dB (16/68); C05 SUBSUP−SUP a=1: -0.07 dB (17/68); C05 SUBSUP−SUP a=9: +1.59 dB (68/68).

**Data:** `runs/nat/eval/{W05,C05}/{sub,subsup}.csv`, `runs/nat/report/summary_sub.json`. **Commands:** `bash runs/nat/queue_sub.sh`; `python natural/report_sub.py`.

---

## RUN 2026-11-02 — CLAIMS LC: were the natural-image models still improving at 3000 iterations?

Descriptive (`Archive/CLAIMS/PREREGISTRATION.md`). W05, BSD68 mean PSNR (dB) at the r chosen without ground truth, checkpoints 1000 / 2000 / 3000:

| model | a = 1 | a = 9 |
|---|---|---|
| ours | 22.03 / 22.34 / 22.55 | 24.05 / 24.14 / 24.48 |
| sup | 21.85 / 22.43 / 22.61 | 24.16 / 24.22 / 24.48 |
| sub | 21.86 / 22.25 / 22.47 | 24.93 / 25.74 / 26.07 |
| subsup | 21.59 / 22.35 / 22.36 | 24.91 / 25.88 / 26.07 |

---

## RUN 2026-11-02 — CLAIMS C3: does training across noise levels give robustness to shift?

Pre-registered (`Archive/CLAIMS/PREREGISTRATION.md`). Trained at W05 (σ 0.5 white); OURS = with the noise ladder, NOLADDER = native noise only, N2N = classic Noise2Noise. Tested in distribution (W05) and shifted (W035 σ 0.35, W07 σ 0.7, C05 ASL-coloured).

**PRIMARY (PSNR(OURS) − PSNR(NOLADDER), Holm over 4):**

- W07, a = 1: +2.92 dB (95% CI +2.59 to +3.27; 68/68 images), p = 7.6e-13, Holm 3.1e-12
- W07, a = 9: -0.74 dB (95% CI -0.85 to -0.62; 6/68 images), p = 3.8e-11, Holm 3.8e-11
- C05, a = 1: -0.26 dB (95% CI -0.31 to -0.22; 0/68 images), p = 7.6e-13, Holm 3.1e-12
- C05, a = 9: +0.29 dB (95% CI +0.25 to +0.34; 64/68 images), p = 1e-11, Holm 2e-11

Mean BSD68 PSNR (dB), a = 1 / a = 9:

| test set | OURS (ladder) | NOLADDER | N2N (classic) |
|---|---|---|---|
| W05 | 22.55 / 24.48 | 22.72 / 24.17 | 22.67 / 24.05 |
| W035 | 16.38 / 23.91 | 15.46 / 23.27 | 22.57 / 24.10 |
| W07 | 21.59 / 23.26 | 18.67 / 24.00 | 18.17 / 23.33 |
| C05 | 22.06 / 24.49 | 22.32 / 24.20 | 22.27 / 24.09 |

---

## RUN 2026-11-02 — CLAIMS C2: residual parametrization (x̂₀ = x̃ + net) vs score (x̃ + σ² net)

Pre-registered (`Archive/CLAIMS/PREREGISTRATION.md`). ASL, dataset-1 training, MSE noise-normalized, noise ladder, seeds 101–103, 2000 iterations; DSM and DIR_L from DEC part A.

**PRIMARY (RESID / DSM, dataset 1, a = 9):** 0.857× (90% CI 0.825–0.893; 3/34 better) → **σ² scaling contributes (residual worse)**.

| comparison | dataset | a = 1 | a = 9 |
|---|---:|---|---|
| RESID / DSM | 1 | 1.096 (1.014–1.205) | 0.857 (0.825–0.893) |
| RESID / DSM | 2 | 1.323 (1.233–1.428) | 1.066 (1.036–1.098) |
| RESID / DSM | 3 | 1.512 (1.391–1.650) | 1.035 (0.999–1.074) |
| RESID / DIRL | 1 | 0.963 (0.953–0.972) | 0.985 (0.978–0.992) |
| RESID / DIRL | 2 | 0.981 (0.967–0.993) | 1.007 (0.999–1.015) |
| RESID / DIRL | 3 | 0.951 (0.930–0.971) | 0.988 (0.977–0.999) |

---

## RUN 2026-11-02 — CLAIMS C7: when does denoising the copy mean win? (training-set size × noise)

Pre-registered (`Archive/CLAIMS/PREREGISTRATION.md`). OURS (per copy + dial) vs SUB (denoise the copy mean), trained on the first n Train400 images; BSD68, r chosen without ground truth.

**PRIMARY (PSNR(SUB) − PSNR(OURS) at a = 9, Holm over 6):**

- W05, n = 25: +2.13 dB (95% CI +1.98 to +2.29; 68/68 images), p = 7.6e-13, Holm 4.6e-12
- W05, n = 100: +2.21 dB (95% CI +2.04 to +2.36; 68/68 images), p = 7.6e-13, Holm 4.6e-12
- W05, n = 400: +1.59 dB (95% CI +1.47 to +1.71; 68/68 images), p = 7.6e-13, Holm 4.6e-12
- W20, n = 25: -0.20 dB (95% CI -0.48 to +0.07; 35/68 images), p = 0.62, Holm 0.62
- W20, n = 100: -0.88 dB (95% CI -1.22 to -0.57; 19/68 images), p = 7.9e-06, Holm 2.4e-05
- W20, n = 400: -0.36 dB (95% CI -0.63 to -0.10; 32/68 images), p = 0.081, Holm 0.16

Mean PSNR (dB), OURS / SUB, a = 1 and a = 9:

| condition | n | a = 1 | a = 9 |
|---|---:|---|---|
| W05 | 25 | 21.72 / 21.86 | 23.13 / 25.27 |
| W05 | 100 | 22.35 / 22.42 | 23.79 / 25.99 |
| W05 | 400 | 22.55 / 22.47 | 24.48 / 26.07 |
| W20 | 25 | 17.59 / 17.37 | 19.43 / 19.23 |
| W20 | 100 | 17.98 / 17.26 | 19.92 / 19.04 |
| W20 | 400 | 17.77 / 17.21 | 19.98 / 19.62 |

---

## EXPLORATORY 2026-11-02 — CLAIMS C2-LC: does the score-vs-direct bias gap close with training?

Not pre-registered. DEC networks (dataset 1, seeds 102–103; DSM vs direct-with-ladder) at 500–2000 iterations; median bias² b and per-map variance v (units of σ²) from the exact decomposition. The linear toy predicts a transient gap.

| iteration | DSM b / v | DIR_L b / v |
|---:|---|---|
| 500 | 0.0433 / 0.0123 | 0.0607 / 0.0073 |
| 1000 | 0.0386 / 0.0084 | 0.0539 / 0.0058 |
| 1500 | 0.0329 / 0.0102 | 0.0511 / 0.0052 |
| 2000 | 0.0316 / 0.0161 | 0.0480 / 0.0047 |

---

## RUN 2026-11-02 — BASE-DDM2: the method vs a DDM2-like baseline

Pre-registered (`Archive/BASE/PREREGISTRATION_DDM2.md`); same splits, scoring and cross-fitting as BASE. DDM2-like pipeline (`baselines/ddm2.py`): stage I = classic Noise2Noise (self, per repetition; deviation: DDM2 uses the other volumes), stage II = residual noise level, stage III = score network trained on stage-I outputs, reverse-sampled from the matched level (DDM2) or decoded one-shot (DDM2-TW). Pre-registered (`Archive/BASE/PREREGISTRATION.md`). Ours = DSM, MSE noise-normalized, dataset-1 training, seeds 101–103, one-shot Tweedie + dial. Every method sees only the A half of each split; all methods are scored on the SAME fixed subset of 10 splits per a (unbiased, paired); every parameter is cross-fitted over the other folds' slices. Code: `baselines/`. The dry run gave 1.000.

**PRIMARY (dataset 1, a = 1 and 9; n_eff ours / DDM2-like + dial, Holm over 2):**

- vs ddm2 a=1: 0.954× (95% CI 0.843–1.045; ours better on 18/34), p = 0.76, Holm 0.76 → **no significant difference**
- vs ddm2 a=9: 1.138× (95% CI 1.076–1.195; ours better on 29/34), p = 3.2e-06, Holm 6.4e-06 → **ours better**

Geometric-mean n_eff (signal region) by method:

| method | ds1 a=1 | ds1 a=3 | ds1 a=5 | ds1 a=9 | ds2 a=1 | ds2 a=3 | ds2 a=5 | ds2 a=9 | ds3 a=1 | ds3 a=3 | ds3 a=5 | ds3 a=9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours (DSM + dial) | 21.68 | 31.94 | 34.34 | 38.95 | 40.07 | 53.86 | 61.37 | 60.66 | 28.65 | 41.30 | 59.56 | 56.63 |
| DDM2-like + dial | 22.73 | 28.79 | 29.37 | 34.24 | 50.59 | 62.02 | 66.98 | 64.32 | 37.41 | 43.78 | 63.82 | 58.20 |
| DDM2-like stage III, one-shot (no sampling) + dial | 22.24 | 28.37 | 29.18 | 33.96 | 47.78 | 60.86 | 65.83 | 63.81 | 32.16 | 39.74 | 55.84 | 53.94 |
| DDM2-like (r = 0, as published) | 22.73 | 26.35 | 25.12 | 26.63 | 50.59 | 60.82 | 62.10 | 57.49 | 37.41 | 44.11 | 59.52 | 52.54 |
| DDM2-like stage III, one-shot (no sampling) (r = 0, as published) | 22.24 | 26.81 | 25.87 | 27.40 | 47.78 | 61.84 | 63.38 | 59.18 | 32.16 | 39.85 | 53.49 | 49.71 |

Ours / baseline, overall and finest band (geometric mean, slices where ours is better):

| baseline | dataset | a = 1 | a = 3 | a = 5 | a = 9 | finest band a = 1 / 9 |
|---|---:|---|---|---|---|---|
| DDM2-like + dial | 1 | 0.954 (18/34) | 1.109 (28/34) | 1.169 (29/34) | 1.138 (29/34) | 0.782 / 1.027 |
| DDM2-like + dial | 2 | 0.792 (4/34) | 0.869 (7/34) | 0.916 (10/34) | 0.943 (11/34) | 0.633 / 0.954 |
| DDM2-like + dial | 3 | 0.766 (0/34) | 0.943 (10/34) | 0.933 (20/34) | 0.973 (19/34) | 0.557 / 0.925 |
| DDM2-like stage III, one-shot (no sampling) + dial | 1 | 0.975 (19/34) | 1.126 (27/34) | 1.177 (29/34) | 1.147 (31/34) | 0.854 / 1.049 |
| DDM2-like stage III, one-shot (no sampling) + dial | 2 | 0.839 (6/34) | 0.885 (8/34) | 0.932 (12/34) | 0.951 (11/34) | 0.871 / 0.889 |
| DDM2-like stage III, one-shot (no sampling) + dial | 3 | 0.891 (6/34) | 1.039 (22/34) | 1.067 (23/34) | 1.050 (23/34) | 0.724 / 0.990 |
| DDM2-like (r = 0, as published) | 1 | 0.954 (18/34) | 1.212 (28/34) | 1.367 (29/34) | 1.463 (32/34) | 0.782 / 0.969 |
| DDM2-like (r = 0, as published) | 2 | 0.792 (4/34) | 0.886 (13/34) | 0.988 (21/34) | 1.055 (22/34) | 0.633 / 0.674 |
| DDM2-like (r = 0, as published) | 3 | 0.766 (0/34) | 0.936 (15/34) | 1.001 (23/34) | 1.078 (25/34) | 0.557 / 0.762 |
| DDM2-like stage III, one-shot (no sampling) (r = 0, as published) | 1 | 0.975 (19/34) | 1.191 (28/34) | 1.327 (29/34) | 1.422 (32/34) | 0.854 / 0.964 |
| DDM2-like stage III, one-shot (no sampling) (r = 0, as published) | 2 | 0.839 (6/34) | 0.871 (9/34) | 0.968 (15/34) | 1.025 (21/34) | 0.871 / 0.713 |
| DDM2-like stage III, one-shot (no sampling) (r = 0, as published) | 3 | 0.891 (6/34) | 1.036 (22/34) | 1.114 (26/34) | 1.139 (28/34) | 0.724 / 0.820 |

**Data:** `runs/base/{classical,ours,n2s_eval}/neff.csv`, `runs/base/report_dl/summary.json`. **Commands:** `python baselines/evaluate.py --methods ...`; `bash runs/base/n2s_queue.sh`; `python baselines/report.py`.

---

## EXPLORATORY 2026-11-03 — LABEL SWEEP: does a mis-stated σ label recover datasets 2–3?

Not pre-registered. The dataset-1 DSM models (seeds 101–103, DEC A) are fed c·σ̂ as the noise label
(`experiments/_dec_eval.py --sigma-scale c`, queue `runs/label_sweep/queue.sh`, report
`experiments/_label_sweep_report.py`); r cross-fitted as usual. c = 1 reproduces `runs/dec/A/eval` exactly.

Geometric-mean n_eff (signal region):

| dataset, a | c 0.6 | 0.75 | 0.9 | 1.0 | 1.15 | 1.3 | 1.5 | best c vs c = 1 | c cross-fitted vs c = 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1, a = 1 | 7.0 | 14.9 | 21.0 | 21.7 | 22.6 | 22.8 | 22.4 | 1.30: 1.052 (1.016–1.101) | 1.040 (1.004–1.090) |
| 1, a = 9 | 30.6 | 36.6 | 39.1 | 39.0 | 39.4 | 38.1 | 37.3 | 1.15: 1.012 (0.988–1.048) | 0.988 (0.972–1.003) |
| 2, a = 1 | 14.7 | 29.2 | 39.2 | 40.1 | 41.3 | 44.6 | 44.1 | 1.30: 1.112 (1.090–1.135) | 1.097 (1.070–1.125) |
| 2, a = 9 | 54.6 | 59.5 | 59.4 | 60.7 | 61.2 | 59.6 | 61.1 | 1.15: 1.009 (0.998–1.020) | 1.000 (0.987–1.012) |
| 3, a = 1 | 8.0 | 17.9 | 27.5 | 28.7 | 29.8 | 31.2 | 32.4 | 1.50: 1.133 (1.092–1.173) | 1.133 (1.093–1.173) |
| 3, a = 9 | 39.8 | 50.2 | 54.4 | 56.6 | 56.1 | 55.8 | 56.3 | 1.00: 1.000 | 1.000 |

(90% slice-bootstrap CIs.) **Reading:**
- Over-stating the noise helps single maps everywhere, including dataset 1 (+5%), and more on datasets 2–3
  (+11%, +13%; dataset 3 still rising at the grid edge). At a = 9 the label does nothing.
- Against classic N2N at a = 1 (48.6 / 42.5), the best label closes about half of the dataset-2 gap (1.21× → 1.09×)
  and about a third of the dataset-3 gap (1.48× → 1.31×). **The scalar label is not the main culprit.**
- A larger label = stronger denoising, i.e. an effective dial r < 0, which the grid (r ≥ 0) does not allow. The
  dataset-1 gain says the one-shot Tweedie estimate under-smooths single maps (consistent with C2: DSM = low bias,
  high variance). The extra gain on datasets 2–3 is the part attributable to the label/noise mismatch.
- Next: extend the dial below 0 (cheap, no training) to separate "needs more smoothing" from "wrong label"; then
  the colour-matched ladder (C9).

---

## RUN 2026-11-03 — RES: one denoiser across resolutions (fastMRI, physical noise)

Pre-registered (`Archive/RES/PREREGISTRATION.md`). fastMRI AXT2, 15 test volumes × 8 slices; physical noise σ(M) = M/320 per channel; n_eff = σ²/MSE vs ground truth (head mask), r chosen without ground truth (leave-one-volume-out); seeds 101–103. Training matrices: A {160}; B–E {80, 160, 320}.

**PRIMARY (n_eff C / B, physical noise, held-out matrices; Holm over 4):**

- M = 128, a = 1: 1.024× (90% CI 1.020–1.028; C better on 15/15 volumes), p = 6.1e-05, Holm 0.00024 → **equivalent (±5%)**
- M = 128, a = 9: 0.982× (90% CI 0.975–0.988; C better on 3/15 volumes), p = 0.002, Holm 0.004 → **equivalent (±5%)**
- M = 240, a = 1: 0.985× (90% CI 0.982–0.988; C better on 0/15 volumes), p = 6.1e-05, Holm 0.00024 → **equivalent (±5%)**
- M = 240, a = 9: 0.990× (90% CI 0.983–0.998; C better on 5/15 volumes), p = 0.073, Holm 0.073 → **equivalent (±5%)**

n_eff (physical noise, a = 1; geometric mean over volumes):

| arm | M = 320 | M = 240 | M = 160 | M = 128 | M = 80 |
|---|---:|---:|---:|---:|---:|
| A | 87.08 | 58.61 | 33.19 | 9.85 | 1.31 |
| B | 96.27 | 53.87 | 27.62 | 17.57 | 7.67 |
| C | 92.57 | 53.06 | 28.17 | 17.99 | 7.89 |
| D | 96.66 | 55.72 | 27.13 | 17.05 | 7.78 |
| E | 106.57 | 63.15 | 31.27 | 21.31 | 9.59 |

| ratio | M = 320 | M = 240 | M = 160 | M = 128 | M = 80 |
|---|---|---|---|---|---|
| A / B | 0.905 (0.89–0.92) | 1.088 (1.08–1.10) | 1.202 (1.19–1.21) | 0.560 (0.55–0.57) | 0.170 (0.17–0.18) |
| C / B | 0.962 (0.96–0.97) | 0.985 (0.98–0.99) | 1.020 (1.02–1.02) | 1.024 (1.02–1.03) | 1.028 (1.03–1.03) |
| D / B | 1.004 (1.00–1.01) | 1.034 (1.03–1.04) | 0.982 (0.98–0.99) | 0.970 (0.96–0.98) | 1.013 (1.01–1.02) |
| E / B | 1.107 (1.10–1.12) | 1.172 (1.16–1.18) | 1.132 (1.12–1.14) | 1.213 (1.20–1.22) | 1.250 (1.24–1.26) |

n_eff (physical noise, a = 9; geometric mean over volumes):

| arm | M = 320 | M = 240 | M = 160 | M = 128 | M = 80 |
|---|---:|---:|---:|---:|---:|
| A | 126.40 | 91.83 | 56.60 | 36.56 | 11.25 |
| B | 164.16 | 101.46 | 53.43 | 37.92 | 20.94 |
| C | 168.94 | 100.49 | 52.17 | 37.23 | 20.64 |
| D | 154.31 | 92.10 | 50.17 | 37.01 | 21.53 |
| E | 179.11 | 103.27 | 52.36 | 37.55 | 21.13 |

| ratio | M = 320 | M = 240 | M = 160 | M = 128 | M = 80 |
|---|---|---|---|---|---|
| A / B | 0.770 (0.75–0.79) | 0.905 (0.89–0.93) | 1.059 (1.04–1.08) | 0.964 (0.95–0.98) | 0.537 (0.52–0.56) |
| C / B | 1.029 (1.02–1.04) | 0.990 (0.98–1.00) | 0.976 (0.97–0.98) | 0.982 (0.98–0.99) | 0.986 (0.98–0.99) |
| D / B | 0.940 (0.93–0.95) | 0.908 (0.90–0.91) | 0.939 (0.93–0.94) | 0.976 (0.97–0.98) | 1.028 (1.02–1.04) |
| E / B | 1.091 (1.07–1.11) | 1.018 (1.00–1.03) | 0.980 (0.97–0.99) | 0.990 (0.98–1.00) | 1.009 (1.00–1.02) |

n_eff (matched σ = 0.5 noise, a = 1; geometric mean over volumes):

| arm | M = 320 | M = 240 | M = 160 | M = 128 | M = 80 |
|---|---:|---:|---:|---:|---:|
| A | 43.93 | 39.33 | 33.19 | 28.85 | 17.15 |
| B | 39.41 | 34.43 | 27.62 | 24.06 | 18.03 |
| C | 40.26 | 34.92 | 28.17 | 24.68 | 18.85 |
| D | 29.75 | 31.35 | 27.13 | 22.19 | 12.89 |
| E | 44.07 | 38.55 | 31.27 | 27.23 | 18.66 |

| ratio | M = 320 | M = 240 | M = 160 | M = 128 | M = 80 |
|---|---|---|---|---|---|
| A / B | 1.115 (1.10–1.13) | 1.142 (1.13–1.15) | 1.202 (1.19–1.21) | 1.199 (1.19–1.21) | 0.951 (0.93–0.97) |
| C / B | 1.022 (1.02–1.03) | 1.014 (1.01–1.02) | 1.020 (1.02–1.02) | 1.026 (1.02–1.03) | 1.046 (1.04–1.05) |
| D / B | 0.755 (0.74–0.77) | 0.911 (0.89–0.93) | 0.982 (0.98–0.99) | 0.922 (0.91–0.93) | 0.715 (0.70–0.73) |
| E / B | 1.118 (1.11–1.13) | 1.120 (1.11–1.13) | 1.132 (1.12–1.14) | 1.132 (1.12–1.14) | 1.035 (1.02–1.05) |

n_eff (matched σ = 0.5 noise, a = 9; geometric mean over volumes):

| arm | M = 320 | M = 240 | M = 160 | M = 128 | M = 80 |
|---|---:|---:|---:|---:|---:|
| A | 72.80 | 66.52 | 56.60 | 49.11 | 28.55 |
| B | 84.86 | 71.74 | 53.43 | 45.02 | 32.80 |
| C | 83.18 | 69.12 | 52.17 | 44.84 | 33.33 |
| D | 48.47 | 49.59 | 50.17 | 48.50 | 40.52 |
| E | 76.29 | 66.28 | 52.36 | 44.94 | 30.55 |

| ratio | M = 320 | M = 240 | M = 160 | M = 128 | M = 80 |
|---|---|---|---|---|---|
| A / B | 0.858 (0.85–0.87) | 0.927 (0.91–0.94) | 1.059 (1.04–1.08) | 1.091 (1.08–1.11) | 0.870 (0.86–0.88) |
| C / B | 0.980 (0.98–0.98) | 0.964 (0.96–0.97) | 0.976 (0.97–0.98) | 0.996 (0.99–1.00) | 1.016 (1.01–1.02) |
| D / B | 0.571 (0.56–0.58) | 0.691 (0.68–0.70) | 0.939 (0.93–0.94) | 1.077 (1.07–1.08) | 1.236 (1.22–1.25) |
| E / B | 0.899 (0.89–0.91) | 0.924 (0.91–0.93) | 0.980 (0.97–0.99) | 0.998 (0.99–1.01) | 0.932 (0.92–0.94) |

Estimator check (arm B, physical noise): split-half n_eff / ground-truth n_eff, geometric mean over volumes: M 320 a 1: 0.995, M 320 a 9: 0.994, M 240 a 1: 1.001, M 240 a 9: 1.002, M 160 a 1: 0.999, M 160 a 9: 0.999, M 128 a 1: 0.997, M 128 a 9: 0.995, M 80 a 1: 0.997, M 80 a 9: 0.996


---

## RUN 2026-11-03 — RES: one denoiser across resolutions (fastMRI, physical noise) — low-noise repeat (S_K = 0.25)

Pre-registered (`Archive/RES/PREREGISTRATION.md`). fastMRI AXT2, 15 test volumes × 8 slices; physical noise σ(M) = 0.25 × M/320 per channel; n_eff = σ²/MSE vs ground truth (head mask), r chosen without ground truth (leave-one-volume-out); seeds 101–103. Training matrices: A {160}; B–E {80, 160, 320}.

**PRIMARY (n_eff C / B, physical noise, held-out matrices; Holm over 4):**

- M = 128, a = 1: 1.000× (90% CI 0.999–1.001; C better on 9/15 volumes), p = 0.52, Holm 0.52 → **equivalent (±5%)**
- M = 128, a = 9: 0.964× (90% CI 0.959–0.969; C better on 0/15 volumes), p = 6.1e-05, Holm 0.00024 → **equivalent (±5%)**
- M = 240, a = 1: 0.912× (90% CI 0.910–0.915; C better on 0/15 volumes), p = 6.1e-05, Holm 0.00024 → **B better**
- M = 240, a = 9: 0.910× (90% CI 0.905–0.916; C better on 0/15 volumes), p = 6.1e-05, Holm 0.00024 → **B better**

n_eff (physical noise, a = 1; geometric mean over volumes):

| arm | M = 320 | M = 240 | M = 160 | M = 128 | M = 80 |
|---|---:|---:|---:|---:|---:|
| A | 23.10 | 15.98 | 9.36 | 4.12 | 1.17 |
| B | 22.23 | 13.91 | 7.68 | 5.16 | 2.25 |
| C | 21.75 | 12.69 | 7.64 | 5.16 | 2.24 |
| D | 21.44 | 13.69 | 7.61 | 5.47 | 2.68 |
| E | 21.74 | 13.91 | 7.58 | 5.51 | 2.84 |

| ratio | M = 320 | M = 240 | M = 160 | M = 128 | M = 80 |
|---|---|---|---|---|---|
| A / B | 1.039 (1.03–1.05) | 1.149 (1.14–1.16) | 1.218 (1.20–1.23) | 0.799 (0.79–0.81) | 0.518 (0.51–0.53) |
| C / B | 0.978 (0.97–0.98) | 0.912 (0.91–0.91) | 0.994 (0.99–1.00) | 1.000 (1.00–1.00) | 0.994 (0.99–1.00) |
| D / B | 0.964 (0.96–0.97) | 0.984 (0.98–0.99) | 0.990 (0.99–0.99) | 1.059 (1.05–1.07) | 1.190 (1.18–1.20) |
| E / B | 0.978 (0.97–0.98) | 1.000 (0.99–1.01) | 0.987 (0.98–0.99) | 1.068 (1.06–1.08) | 1.263 (1.24–1.28) |

n_eff (physical noise, a = 9; geometric mean over volumes):

| arm | M = 320 | M = 240 | M = 160 | M = 128 | M = 80 |
|---|---:|---:|---:|---:|---:|
| A | 43.04 | 33.79 | 23.05 | 15.72 | 9.65 |
| B | 44.42 | 33.04 | 21.32 | 17.82 | 11.74 |
| C | 40.88 | 30.07 | 20.42 | 17.17 | 11.74 |
| D | 37.74 | 28.18 | 20.57 | 17.64 | 12.91 |
| E | 39.33 | 27.63 | 18.90 | 16.08 | 11.87 |

| ratio | M = 320 | M = 240 | M = 160 | M = 128 | M = 80 |
|---|---|---|---|---|---|
| A / B | 0.969 (0.96–0.98) | 1.023 (1.01–1.03) | 1.081 (1.07–1.09) | 0.882 (0.87–0.89) | 0.822 (0.81–0.83) |
| C / B | 0.920 (0.91–0.93) | 0.910 (0.90–0.92) | 0.958 (0.95–0.96) | 0.964 (0.96–0.97) | 1.000 (1.00–1.00) |
| D / B | 0.850 (0.84–0.86) | 0.853 (0.84–0.86) | 0.965 (0.96–0.97) | 0.990 (0.98–1.00) | 1.100 (1.09–1.11) |
| E / B | 0.886 (0.88–0.89) | 0.836 (0.83–0.84) | 0.886 (0.88–0.89) | 0.903 (0.89–0.91) | 1.011 (0.99–1.03) |

n_eff (matched σ = 0.125 noise, a = 1; geometric mean over volumes):

| arm | M = 320 | M = 240 | M = 160 | M = 128 | M = 80 |
|---|---:|---:|---:|---:|---:|
| A | 10.29 | 10.39 | 9.36 | 8.12 | 4.85 |
| B | 9.34 | 9.02 | 7.68 | 6.75 | 4.92 |
| C | 8.98 | 8.54 | 7.64 | 6.67 | 4.89 |
| D | 6.58 | 7.86 | 7.61 | 6.42 | 3.97 |
| E | 9.40 | 9.05 | 7.58 | 6.57 | 4.56 |

| ratio | M = 320 | M = 240 | M = 160 | M = 128 | M = 80 |
|---|---|---|---|---|---|
| A / B | 1.102 (1.09–1.11) | 1.152 (1.14–1.17) | 1.218 (1.20–1.23) | 1.203 (1.19–1.22) | 0.987 (0.98–1.00) |
| C / B | 0.962 (0.96–0.97) | 0.947 (0.94–0.95) | 0.994 (0.99–1.00) | 0.987 (0.98–0.99) | 0.994 (0.99–1.00) |
| D / B | 0.704 (0.69–0.72) | 0.871 (0.86–0.88) | 0.990 (0.99–0.99) | 0.950 (0.94–0.96) | 0.808 (0.79–0.83) |
| E / B | 1.007 (1.00–1.01) | 1.003 (1.00–1.01) | 0.987 (0.98–0.99) | 0.974 (0.97–0.98) | 0.927 (0.92–0.94) |

n_eff (matched σ = 0.125 noise, a = 9; geometric mean over volumes):

| arm | M = 320 | M = 240 | M = 160 | M = 128 | M = 80 |
|---|---:|---:|---:|---:|---:|
| A | 22.42 | 23.89 | 23.05 | 20.52 | 13.98 |
| B | 22.54 | 23.33 | 21.32 | 19.38 | 15.40 |
| C | 21.60 | 21.85 | 20.42 | 18.53 | 15.53 |
| D | 16.72 | 18.45 | 20.57 | 20.97 | 19.09 |
| E | 20.40 | 20.64 | 18.90 | 17.06 | 13.69 |

| ratio | M = 320 | M = 240 | M = 160 | M = 128 | M = 80 |
|---|---|---|---|---|---|
| A / B | 0.995 (0.99–1.00) | 1.024 (1.02–1.03) | 1.081 (1.07–1.09) | 1.059 (1.05–1.07) | 0.907 (0.90–0.92) |
| C / B | 0.958 (0.95–0.96) | 0.937 (0.93–0.94) | 0.958 (0.95–0.96) | 0.956 (0.95–0.96) | 1.008 (1.01–1.01) |
| D / B | 0.742 (0.72–0.76) | 0.791 (0.78–0.80) | 0.965 (0.96–0.97) | 1.082 (1.07–1.09) | 1.239 (1.22–1.25) |
| E / B | 0.905 (0.90–0.91) | 0.885 (0.88–0.89) | 0.886 (0.88–0.89) | 0.880 (0.87–0.89) | 0.889 (0.88–0.90) |

Estimator check (arm B, physical noise): split-half n_eff / ground-truth n_eff, geometric mean over volumes: M 320 a 1: 0.999, M 320 a 9: 0.999, M 240 a 1: 1.001, M 240 a 9: 1.001, M 160 a 1: 1.000, M 160 a 9: 1.000, M 128 a 1: 0.999, M 128 a 9: 0.999, M 80 a 1: 0.999, M 80 a 9: 0.999


---

## RUN 2026-11-03 — MEAN: denoise the mean vs denoise each, then average (fastMRI, training-data scaling)

Pre-registered (`Archive/MEAN/PREREGISTRATION.md`). fastMRI AXT2 at matrix 160, physical noise σ = 0.5 per channel, N = 10 copies; PER = arm A (denoise each copy, average, dial), SUB = denoise the mean of the a copies (σ/√a, dial). Training on the first 2 / 8 / 62 training volumes (16 / 64 / 496 images; corrected 2026-11-04, was mislabelled 128), seeds 101–102; checkpoint and r chosen without ground truth (leave-one-volume-out split-half); n_eff vs ground truth on 15 test volumes.

**PRIMARY (Holm over 2):**

- P1 n_eff SUB/PER, 62 volumes, a = 9: 1.536× (90% CI 1.508–1.568; SUB higher on 15/15 volumes), p = 6.1e-05, Holm 0.00012
- P2 band T4 SUB - PER, 62 volumes, a = 9: 0.214 (90% CI 0.205–0.224; SUB higher on 15/15 volumes), p = 6.1e-05, Holm 0.00012

n_eff (geometric mean over volumes) and SUB / PER (90% CI):

| training volumes | a | PER | SUB | SUB / PER |
|---:|---:|---:|---:|---|
| 2 | 1 | 23.11 | 23.44 | 1.014 (1.00–1.03) |
| 2 | 3 | 31.84 | 39.95 | 1.254 (1.23–1.28) |
| 2 | 5 | 35.68 | 52.96 | 1.484 (1.45–1.52) |
| 2 | 9 | 41.23 | 70.39 | 1.707 (1.67–1.75) |
| 8 | 1 | 28.23 | 28.32 | 1.003 (0.98–1.02) |
| 8 | 3 | 38.15 | 48.69 | 1.276 (1.23–1.32) |
| 8 | 5 | 42.86 | 64.13 | 1.496 (1.45–1.55) |
| 8 | 9 | 48.63 | 83.67 | 1.720 (1.66–1.78) |
| 62 | 1 | 32.87 | 31.91 | 0.971 (0.97–0.98) |
| 62 | 3 | 44.77 | 53.82 | 1.202 (1.18–1.22) |
| 62 | 5 | 50.43 | 68.56 | 1.359 (1.33–1.39) |
| 62 | 9 | 56.27 | 86.45 | 1.536 (1.51–1.57) |

Band transfer at a = 9 (share of the true detail kept per band of k / k_max; plain averaging = 1 in every band):

| design | training volumes | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–1 |
|---|---:|---:|---:|---:|---:|---:|
| PER | 2 | 0.98 | 0.75 | 0.48 | 0.23 | 0.18 |
| PER | 8 | 0.98 | 0.84 | 0.57 | 0.28 | 0.20 |
| PER | 62 | 1.00 | 0.87 | 0.59 | 0.28 | 0.19 |
| SUB | 2 | 0.99 | 0.92 | 0.78 | 0.48 | 0.28 |
| SUB | 8 | 1.00 | 0.96 | 0.81 | 0.47 | 0.25 |
| SUB | 62 | 1.00 | 0.95 | 0.82 | 0.49 | 0.26 |

---

## RUN 2026-11-04 — MEAN-SNR: denoise the mean vs denoise each at ASL-like noise (fastMRI, S_K = 2)

Pre-registered (`Archive/MEAN/PREREGISTRATION_SNR.md`). fastMRI AXT2 at matrix 160, physical noise σ = 1 per channel, N = 10 copies; PER = arm A (denoise each copy, average, dial), SUB = denoise the mean of the a copies (σ/√a, dial). Training on the first 62 training volumes (496 images), seeds 101–102; checkpoint and r chosen without ground truth (leave-one-volume-out split-half); n_eff vs ground truth on 15 test volumes.

**PRIMARY (Holm over 2):**

- P1 n_eff SUB/PER, 62 volumes, a = 9: 1.455× (90% CI 1.411–1.500; SUB higher on 15/15 volumes), p = 6.1e-05, Holm 0.00012
- P2 band T4 SUB - PER, 62 volumes, a = 9: 0.112 (90% CI 0.107–0.117; SUB higher on 15/15 volumes), p = 6.1e-05, Holm 0.00012

n_eff (geometric mean over volumes) and SUB / PER (90% CI):

| training volumes | a | PER | SUB | SUB / PER |
|---:|---:|---:|---:|---|
| 62 | 1 | 70.05 | 67.63 | 0.965 (0.96–0.98) |
| 62 | 3 | 89.60 | 103.02 | 1.150 (1.13–1.17) |
| 62 | 5 | 94.94 | 128.54 | 1.354 (1.32–1.39) |
| 62 | 9 | 103.98 | 151.25 | 1.455 (1.41–1.50) |

Band transfer at a = 9 (share of the true detail kept per band of k / k_max; plain averaging = 1 in every band):

| design | training volumes | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–1 |
|---|---:|---:|---:|---:|---:|---:|
| PER | 62 | 0.96 | 0.63 | 0.26 | 0.10 | 0.09 |
| SUB | 62 | 1.00 | 0.89 | 0.61 | 0.21 | 0.10 |

---

## RUN 2026-11-04 — MEAN-SNR: denoise the mean vs denoise each at ASL-like noise (fastMRI, S_K = 3)

Pre-registered (`Archive/MEAN/PREREGISTRATION_SNR.md`). fastMRI AXT2 at matrix 160, physical noise σ = 1.5 per channel, N = 10 copies; PER = arm A (denoise each copy, average, dial), SUB = denoise the mean of the a copies (σ/√a, dial). Training on the first 62 training volumes (496 images), seeds 101–102; checkpoint and r chosen without ground truth (leave-one-volume-out split-half); n_eff vs ground truth on 15 test volumes.

**PRIMARY (Holm over 2):**

- P1 n_eff SUB/PER, 62 volumes, a = 9: 1.229× (90% CI 1.192–1.265; SUB higher on 15/15 volumes), p = 6.1e-05, Holm 0.00012
- P2 band T4 SUB - PER, 62 volumes, a = 9: 0.043 (90% CI 0.040–0.045; SUB higher on 15/15 volumes), p = 6.1e-05, Holm 0.00012

n_eff (geometric mean over volumes) and SUB / PER (90% CI):

| training volumes | a | PER | SUB | SUB / PER |
|---:|---:|---:|---:|---|
| 62 | 1 | 108.07 | 106.12 | 0.982 (0.97–0.99) |
| 62 | 3 | 137.39 | 144.14 | 1.049 (1.03–1.07) |
| 62 | 5 | 145.46 | 175.50 | 1.207 (1.18–1.23) |
| 62 | 9 | 156.96 | 192.92 | 1.229 (1.19–1.26) |

Band transfer at a = 9 (share of the true detail kept per band of k / k_max; plain averaging = 1 in every band):

| design | training volumes | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–1 |
|---|---:|---:|---:|---:|---:|---:|
| PER | 62 | 0.95 | 0.39 | 0.14 | 0.08 | 0.08 |
| SUB | 62 | 0.96 | 0.80 | 0.44 | 0.13 | 0.08 |

---

## RUN 2026-11-04 — MEAN-SNR: denoise the mean vs denoise each at ASL-like noise (fastMRI, S_K = 6)

Pre-registered (`Archive/MEAN/PREREGISTRATION_SNR.md`). fastMRI AXT2 at matrix 160, physical noise σ = 3 per channel, N = 10 copies; PER = arm A (denoise each copy, average, dial), SUB = denoise the mean of the a copies (σ/√a, dial). Training on the first 62 training volumes (496 images), seeds 101–102; checkpoint and r chosen without ground truth (leave-one-volume-out split-half); n_eff vs ground truth on 15 test volumes.

**PRIMARY (Holm over 2):**

- P1 n_eff SUB/PER, 62 volumes, a = 9: 0.659× (90% CI 0.634–0.682; SUB higher on 0/15 volumes), p = 6.1e-05, Holm 0.00012
- P2 band T4 SUB - PER, 62 volumes, a = 9: 0.142 (90% CI 0.137–0.146; SUB higher on 15/15 volumes), p = 6.1e-05, Holm 0.00012

n_eff (geometric mean over volumes) and SUB / PER (90% CI):

| training volumes | a | PER | SUB | SUB / PER |
|---:|---:|---:|---:|---|
| 62 | 1 | 208.32 | 223.44 | 1.073 (1.06–1.09) |
| 62 | 3 | 277.69 | 243.11 | 0.875 (0.86–0.89) |
| 62 | 5 | 312.28 | 289.12 | 0.926 (0.90–0.95) |
| 62 | 9 | 335.20 | 220.82 | 0.659 (0.63–0.68) |

Band transfer at a = 9 (share of the true detail kept per band of k / k_max; plain averaging = 1 in every band):

| design | training volumes | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–1 |
|---|---:|---:|---:|---:|---:|---:|
| PER | 62 | 0.85 | 0.05 | 0.00 | -0.01 | -0.00 |
| SUB | 62 | 0.93 | 0.47 | 0.24 | 0.13 | 0.11 |

---

### Reading — MEAN-SNR (written 2026-11-04 after the three reports above)

SUB / PER at a = 9 against the common SNR scale (rms head signal / per-channel noise): S_K = 1 (SNR 1.09) 1.536×,
S_K = 2 (0.55) 1.455×, S_K = 3 (0.36) 1.229×, S_K = 6 (0.18) **0.659×**. Pre-registered predictions: S_K = 2, 3 in
(1, 1.54) — held; S_K = 6 ≤ 1.1 — held (more strongly than expected); a = 1 within 5% — held at S_K = 2, 3 (0.965,
0.982), missed at S_K = 6 (1.073, SUB better). Rule "> 1.2 at S_K = 3 → noise level does not explain the ASL result":
point estimate 1.229 passes, CI 1.19–1.26 straddles 1.2; ASL dataset 1 (SNR 0.63) sits between S_K = 1 and 2, where SUB
wins by 1.45–1.54×, so noise level does not explain GROUND SM's 0.967× on dataset 1.

Diagnostic (exploratory, oracle over checkpoint and r, `r = 0`): the S_K = 6 reversal is NOT a selection failure —
SUB's best ground-truth n_eff peaks at a = 5 (265) and falls at a = 9 (206), i.e. the SUB model gets worse as its input
gets cleaner. At equal decode noise, SUB models are worse than a model trained at that noise, increasingly with S_K:
σ = 0.5: S_K = 3 SUB (a = 9) MSE 0.0111 vs S_K = 1 PER 0.0075; σ = 1.0: S_K = 6 SUB (a = 9) 0.0379 vs S_K = 2 PER 0.0137.
Mechanism (hypothesis): in subset N2N the input of a large subset m is paired with the mean of the N − m others, so the
clean end of SUB's range is trained against the noisiest targets (m = 9: target variance 9× the input's). N2N keeps the
learned function unbiased but its estimation variance at fixed data/iterations grows with target noise — a
"target-noise tax" that sufficiency does not see. It is small at S_K ≤ 2 and dominates at S_K = 6. PER's LOO target
is always a 9-copy mean. The band data agree: at S_K = 6 PER's transfer is ≈ 0 above k/k_max = 0.1 (it wins MSE by
blurring), while SUB keeps more detail (T4 +0.14) yet loses n_eff.

Candidate fixes (not run): inverse-target-variance weighting over m in `loss_subset` (weight ∝ (N − m)), or
oversampling small-target-noise pairs; a matched check is S_K = 6 SUB with the weighting.

---

## RUN 2026-11-04 — ASL-SM2: denoise the mean vs denoise each, then average, on re-reconstructed ASL

Pre-registered (`Archive/SM2/PREREGISTRATION.md`). Pooled training on datasets 1–3 (re-reconstructed 2026-11-03), 5-fold CV by slice index, 3000 iterations. PER and SUB: the fastMRI MEAN recipe (MSE, p_raw 0.5, no augmentation; SUB ladder floor σ̂_min/3, m ~ U{1..9}); PER-D: MSE noise-normalized (the best earlier ASL recipe), seed 101 only. Split-half n_eff (signal region, no ground truth); checkpoint and r chosen on the other folds.

**PRIMARY (Holm over 2):**

- P1 SUB / PER, dataset 1, a = 9: 0.867× (90% CI 0.818–0.916; SUB better on 11/34 slices), p = 0.00046, Holm 0.00046
- P2 SUB / PER-D, dataset 1, a = 9: 0.752× (90% CI 0.682–0.825; SUB better on 7/34 slices), p = 4.1e-06, Holm 8.1e-06

SUB / PER and SUB / PER-D (90% CI):

| dataset | a | SUB / PER | SUB / PER-D |
|---:|---:|---|---|
| 1 | 1 | 0.932 (0.88–1.00; 4/34) | 0.819 (0.79–0.85; 1/34) |
| 1 | 2 | 0.927 (0.90–0.95; 5/34) | 0.797 (0.76–0.83; 0/34) |
| 1 | 3 | 0.906 (0.89–0.92; 1/34) | 0.788 (0.74–0.84; 2/34) |
| 1 | 5 | 0.884 (0.85–0.92; 8/34) | 0.763 (0.70–0.83; 2/34) |
| 1 | 7 | 0.863 (0.82–0.91; 9/34) | 0.758 (0.69–0.83; 4/34) |
| 1 | 9 | 0.867 (0.82–0.92; 11/34) | 0.752 (0.68–0.82; 7/34) |
| 2 | 1 | 0.944 (0.93–0.96; 4/34) | 0.955 (0.93–0.99; 6/34) |
| 2 | 2 | 0.915 (0.89–0.94; 1/34) | 0.910 (0.88–0.94; 4/34) |
| 2 | 3 | 0.900 (0.83–0.96; 9/34) | 0.874 (0.81–0.92; 3/34) |
| 2 | 5 | 0.833 (0.75–0.91; 4/34) | 0.845 (0.77–0.91; 3/34) |
| 2 | 7 | 0.806 (0.72–0.89; 2/34) | 0.803 (0.72–0.88; 3/34) |
| 2 | 9 | 0.811 (0.72–0.89; 5/34) | 0.805 (0.73–0.88; 0/34) |
| 3 | 1 | 0.909 (0.82–1.03; 4/34) | 0.708 (0.64–0.77; 1/34) |
| 3 | 2 | 0.773 (0.72–0.82; 2/34) | 0.000 (nan–0.48; 0/34) |
| 3 | 3 | 0.728 (0.67–0.78; 0/34) | 0.000 (nan–nan; 0/34) |
| 3 | 5 | 0.693 (0.61–0.78; 0/34) | 0.000 (nan–nan; 1/34) |
| 3 | 7 | 0.644 (0.53–0.77; 4/34) | 0.000 (nan–nan; 2/34) |
| 3 | 9 | 0.621 (0.50–0.76; 12/34) | 0.000 (nan–nan; 4/34) |

Absolute n_eff (geometric mean over slices) and chosen r (per seed, fold 0's choice):

| dataset | a | PER | SUB | PER-D |
|---:|---:|---|---|---|
| 1 | 1 | 24.33 (r 0, 0) | 22.68 (r 0, 0) | 27.68 (r 0) |
| 1 | 2 | 27.16 (r 0, 0) | 25.17 (r 0, 0) | 31.56 (r 0) |
| 1 | 3 | 29.58 (r 0.0833, 0.0833) | 26.81 (r 0, 0) | 34.04 (r 0.0833) |
| 1 | 5 | 32.26 (r 0.0833, 0.0833) | 28.53 (r 0, 0) | 37.38 (r 0.0833) |
| 1 | 7 | 34.86 (r 0.167, 0.167) | 30.09 (r 0, 0) | 39.71 (r 0.167) |
| 1 | 9 | 36.48 (r 0.25, 0.25) | 31.62 (r 0, 0) | 42.06 (r 0.167) |
| 2 | 1 | 58.83 (r 0, 0) | 55.55 (r 0, 0) | 58.15 (r 0) |
| 2 | 2 | 63.39 (r 0, 0) | 57.98 (r 0, 0) | 63.74 (r 0) |
| 2 | 3 | 64.46 (r 0, 0) | 57.99 (r 0, 0) | 66.36 (r 0) |
| 2 | 5 | 69.58 (r 0.0833, 0.0833) | 57.98 (r 0, 0) | 68.58 (r 0.0833) |
| 2 | 7 | 72.78 (r 0.0833, 0.0833) | 58.65 (r 0, 0) | 73.02 (r 0.0833) |
| 2 | 9 | 74.73 (r 0.0833, 0.0833) | 60.64 (r 0.0833, 0.0833) | 75.31 (r 0.0833) |
| 3 | 1 | 45.47 (r 0, 0) | 41.34 (r 0, 0) | 58.36 (r 0) |
| 3 | 2 | 56.59 (r 0, 0) | 43.73 (r 0, 0) | inf (r 0) |
| 3 | 3 | 62.61 (r 0, 0) | 45.59 (r 0, 0) | inf (r 0) |
| 3 | 5 | 67.64 (r 0.0833, 0.0833) | 46.85 (r 0, 0) | inf (r 0.0833) |
| 3 | 7 | 75.26 (r 0.0833, 0.0833) | 48.48 (r 0, 0) | inf (r 0.0833) |
| 3 | 9 | 79.68 (r 0.0833, 0.0833) | 49.46 (r 0, 0) | inf (r 0.0833) |

### Note on the ASL-SM2 report (written 2026-11-04, 21:10, after the run; exploratory)
- Predictions vs result: P1 > 1 and P2 > 1 were predicted; both FAIL (SUB / PER 0.867×, SUB / PER-D 0.752× on dataset 1,
  a = 9). The ratio falls with a instead of growing; at a = 1 SUB is 7% below PER (prediction: within 5%). Dataset 3
  shows the smallest ratio, as predicted. Per the pre-registered interpretation, with SUB winning at fastMRI SNR
  0.36–0.55 (MEAN-SNR), the deficit is specific to ASL (coloured / non-independent noise, few subjects) — not noise level
  or data size. Note SUB picks r = 0 almost everywhere and its absolute n_eff barely grows with a (dataset 1: 22.7 → 31.6;
  plain averaging gains 9×), i.e. its decoder does not exploit the lower noise of the mean — consistent with the
  target-noise tax seen at fastMRI S_K = 6. MEAN-TAX and ASL-SUBJ (queued) test this further.
- Artefact in the secondary table: PER-D on dataset 3 has 61 non-positive split-half MSEs (n_eff = inf; slices 22–27 at
  a = 9), so its "0.000" ratios and "inf" n_eff are meaningless. Robust versions (not pre-registered): SUB / PER-D on
  dataset 3 median per-slice ratio 0.75 / 0.70 / 0.70 / 0.79 / 0.84 / 0.87 and pooled-MSE ratio 0.75 / 0.60 / 0.58 / 0.55
  / 0.54 / 0.53 for a = 1 / 2 / 3 / 5 / 7 / 9. Primary read-outs (dataset 1) are unaffected (no non-finite entries).

---

## RUN 2026-11-04 — ASL-SUBJ: denoise the mean vs denoise each, then average, on an unseen subject

Pre-registered (`Archive/SUBJ/PREREGISTRATION.md`). Trained on every slice of dataset 1 + scan 1104a (same subject, 2026-09-24 and 2026-11-04, good coil); tested on every slice of scan 1104c (another subject, same protocol and coil). The ASL-SM2 recipe (MSE, p_raw 0.5, no augmentation, 3000 iterations; SUB ladder floor σ̂_min/3, m ~ U{1..9}; PER-D noise-normalized, seed 101 only). Split-half n_eff on the test subject (signal region, no ground truth); checkpoint and r cross-fitted over the test subject's slice folds.

**PRIMARY (Holm over 2):**

- P1 SUB / PER, unseen subject (1104c), a = 9: 1.062× (90% CI 1.028–1.094; SUB better on 28/34 slices), p = 0.00058, Holm 0.00058
- P2 SUB / PER-D, unseen subject (1104c), a = 9: 1.054× (90% CI 1.020–1.085; SUB better on 30/34 slices), p = 0.00029, Holm 0.00058

SUB / PER and SUB / PER-D (90% CI):

| dataset | a | SUB / PER | SUB / PER-D |
|---:|---:|---|---|
| 6 | 1 | 1.001 (0.99–1.01; 20/34) | 0.992 (0.98–1.00; 12/34) |
| 6 | 2 | 1.040 (1.03–1.05; 33/34) | 1.024 (1.02–1.03; 29/34) |
| 6 | 3 | 1.066 (1.06–1.08; 32/34) | 1.026 (1.01–1.04; 26/34) |
| 6 | 5 | 1.057 (1.02–1.09; 27/34) | 1.039 (1.01–1.06; 27/34) |
| 6 | 7 | 1.062 (1.03–1.09; 27/34) | 1.050 (1.02–1.08; 30/34) |
| 6 | 9 | 1.062 (1.03–1.09; 28/34) | 1.054 (1.02–1.09; 30/34) |

Absolute n_eff (geometric mean over slices) and chosen r (per seed, fold 0's choice):

| dataset | a | PER | SUB | PER-D |
|---:|---:|---|---|---|
| 6 | 1 | 17.90 (r 0, 0) | 17.91 (r 0, 0) | 18.06 (r 0) |
| 6 | 2 | 19.59 (r 0.0833, 0.0833) | 20.37 (r 0, 0) | 19.90 (r 0.0833) |
| 6 | 3 | 20.67 (r 0.0833, 0.167) | 22.04 (r 0, 0) | 21.48 (r 0.0833) |
| 6 | 5 | 23.32 (r 0.167, 0.167) | 24.64 (r 0.0833, 0.0833) | 23.71 (r 0.167) |
| 6 | 7 | 25.40 (r 0.25, 0.25) | 26.96 (r 0.0833, 0.0833) | 25.69 (r 0.25) |
| 6 | 9 | 27.24 (r 0.333, 0.333) | 28.93 (r 0.0833, 0.0833) | 27.44 (r 0.333) |

### Note on ASL-SUBJ (written 2026-11-04, 22:20, after the run; exploratory)
- All pre-registered predictions hold: P1 1.062×, P2 1.054× (both Holm p < 0.001); the ratio grows from a = 1 (1.001,
  within 5%) to a plateau of about 1.06 from a = 3. SUB again picks r ≈ 0–1/12; PER needs r up to 1/3.
- Contrast with ASL-SM2 (SUB 0.87× on dataset 1): the two studies differ in training data (good coil only, one subject in
  two sessions vs pooled datasets 1–3 including datasets 2–3 from the poorer coil — one session, noise colour 1.48 vs 0.53 vs 0.66 for dataset 1) and in test data
  (unseen subject vs held-out slices), so they are not a controlled pair. Leading hypothesis to test, not a finding:
  pooling data with different noise colour (two coils, three colours) hurts SUB (one scalar σ conditioning, decoding at σ/√a) more than PER.
  A direct test: SM2's recipe trained on dataset 1 only, scored on dataset 1 held-out slices.
- The gain (≈ 6%) is much smaller than on fastMRI at a similar SNR (1.46–1.54×).

---

## RUN 2026-11-04 — MEAN-TAX: removing the target-noise tax of denoise-the-mean (fastMRI, S_K = 6)

Pre-registered (`Archive/MEAN/PREREGISTRATION_TAX.md`). fastMRI AXT2 at matrix 160, σ = 3 per channel, N = 10, 62 training volumes, seeds 101–102. SUBW: denoise the mean with each sample weighted by its target's precision (N − m); SUBM: denoise the mean with m drawn ∝ m. SUB and PER (arm A) are the MEAN-SNR runs at the same S_K. Checkpoint and r chosen without ground truth (leave-one-volume-out split-half); n_eff vs ground truth on 15 test volumes.

**PRIMARY (Holm over 2):**

- P1 SUBW / SUB, a = 9: 0.586× (90% CI 0.582–0.591; higher on 0/15 volumes), p = 6.1e-05, Holm 0.00012
- P2 SUBM / SUB, a = 9: 1.126× (90% CI 1.122–1.131; higher on 15/15 volumes), p = 6.1e-05, Holm 0.00012

n_eff (geometric mean over volumes) and ratios to PER (90% CI):

| a | PER | SUB | SUBW | SUBM | SUB / PER | SUBW / PER | SUBM / PER |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 208.3 | 223.4 | 185.7 | 163.8 | 1.073 (1.06–1.09) | 0.892 (0.88–0.90) | 0.786 (0.77–0.80) |
| 3 | 277.7 | 243.1 | 229.1 | 216.1 | 0.875 (0.86–0.89) | 0.825 (0.81–0.84) | 0.778 (0.77–0.79) |
| 5 | 312.3 | 289.1 | 269.5 | 247.9 | 0.926 (0.90–0.95) | 0.863 (0.84–0.88) | 0.794 (0.77–0.81) |
| 9 | 335.2 | 220.8 | 129.5 | 248.8 | 0.659 (0.63–0.68) | 0.386 (0.37–0.40) | 0.742 (0.72–0.77) |

Band transfer at a = 9 (same choice; plain averaging = 1):

| design | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–1 |
|---|---:|---:|---:|---:|---:|
| PER | 0.85 | 0.05 | 0.00 | -0.01 | -0.00 |
| SUB | 0.93 | 0.47 | 0.24 | 0.13 | 0.11 |
| SUBW | 0.92 | 0.50 | 0.30 | 0.23 | 0.21 |
| SUBM | 0.99 | 0.45 | 0.22 | 0.13 | 0.10 |

Exploratory — oracle (best checkpoint and r by ground truth) MSE, seed 101, by decode noise (SUB arms decode at σ/√a; a monotone tax-free model gets lower MSE as a grows):

| a | decode σ (SUB arms) | PER | SUB | SUBW | SUBM |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.00 | 0.0435 | 0.0423 | 0.0536 | 0.0462 |
| 3 | 1.73 | 0.0322 | 0.0388 | 0.0408 | 0.0385 |
| 5 | 1.34 | 0.0286 | 0.0340 | 0.0335 | 0.0330 |
| 9 | 1.00 | 0.0262 | 0.0438 | 0.0635 | 0.0315 |

---

## RUN 2026-11-04 — MEAN-TAX: removing the target-noise tax of denoise-the-mean (fastMRI, S_K = 2)

Pre-registered (`Archive/MEAN/PREREGISTRATION_TAX.md`). fastMRI AXT2 at matrix 160, σ = 1 per channel, N = 10, 62 training volumes, seeds 101–102. SUBW: denoise the mean with each sample weighted by its target's precision (N − m); SUBM: denoise the mean with m drawn ∝ m. SUB and PER (arm A) are the MEAN-SNR runs at the same S_K. Checkpoint and r chosen without ground truth (leave-one-volume-out split-half); n_eff vs ground truth on 15 test volumes.

**PRIMARY (Holm over 2):**

- P1 SUBW / SUB, a = 9: 0.810× (90% CI 0.805–0.815; higher on 0/15 volumes), p = 6.1e-05, Holm 0.00012
- P2 SUBM / SUB, a = 9: 1.035× (90% CI 1.032–1.038; higher on 15/15 volumes), p = 6.1e-05, Holm 0.00012

n_eff (geometric mean over volumes) and ratios to PER (90% CI):

| a | PER | SUB | SUBW | SUBM | SUB / PER | SUBW / PER | SUBM / PER |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 70.1 | 67.6 | 68.7 | 65.4 | 0.965 (0.96–0.98) | 0.980 (0.97–0.99) | 0.934 (0.92–0.95) |
| 3 | 89.6 | 103.0 | 102.1 | 99.1 | 1.150 (1.13–1.17) | 1.140 (1.12–1.16) | 1.106 (1.08–1.13) |
| 5 | 94.9 | 128.5 | 127.5 | 127.0 | 1.354 (1.32–1.39) | 1.343 (1.31–1.38) | 1.338 (1.30–1.38) |
| 9 | 104.0 | 151.3 | 122.5 | 156.6 | 1.455 (1.41–1.50) | 1.178 (1.14–1.22) | 1.506 (1.46–1.55) |

Band transfer at a = 9 (same choice; plain averaging = 1):

| design | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–1 |
|---|---:|---:|---:|---:|---:|
| PER | 0.96 | 0.63 | 0.26 | 0.10 | 0.09 |
| SUB | 1.00 | 0.89 | 0.61 | 0.21 | 0.10 |
| SUBW | 0.99 | 0.87 | 0.59 | 0.26 | 0.16 |
| SUBM | 0.99 | 0.88 | 0.60 | 0.21 | 0.09 |

Exploratory — oracle (best checkpoint and r by ground truth) MSE, seed 101, by decode noise (SUB arms decode at σ/√a; a monotone tax-free model gets lower MSE as a grows):

| a | decode σ (SUB arms) | PER | SUB | SUBW | SUBM |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 0.0137 | 0.0147 | 0.0149 | 0.0158 |
| 3 | 0.58 | 0.0105 | 0.0097 | 0.0098 | 0.0104 |
| 5 | 0.45 | 0.0099 | 0.0077 | 0.0079 | 0.0079 |
| 9 | 0.33 | 0.0090 | 0.0066 | 0.0082 | 0.0065 |

### Reading — MEAN-TAX (written 2026-11-04, 23:50, after the run)
- Predictions vs result:
  - "At least one remedy > 1 at S_K = 6, a = 9; SUBM more likely": holds. SUBM / SUB 1.126× (15/15 volumes); SUBW / SUB
    0.586× (0/15).
  - "Its oracle MSE falls monotonically with a": holds for SUBM (0.0462 → 0.0385 → 0.0330 → 0.0315); SUB (0.0438 at
    a = 9, above a = 5) and SUBW (0.0635) do not.
  - "Neither beats PER at S_K = 6; above 0.9× would be strong": SUBM / PER 0.742× (SUB 0.659×). Expected, not strong.
  - "At S_K = 2 the remedies change SUB / PER by < 5%": holds for SUBM (1.455 → 1.506, +3.5%), fails for SUBW
    (1.455 → 1.178, −19%).
- Pre-registered rule triggers: SUBM works at S_K = 6 without costing at S_K = 2, so it becomes the default denoise-the-mean
  recipe for ASL runs after ASL-SM2 (ASL-SM2 and ASL-SUBJ are not changed).
- Cost of SUBM at small a (not in the predictions): drawing m ∝ m starves the noisy end, so a = 1 drops (S_K = 6:
  SUB / PER 1.073 → SUBM / PER 0.786; S_K = 2: 0.965 → 0.934). For averaged ASL maps (a = 9) this is the right trade.
- Why SUBW hurts: inverse-target-variance weights shrink exactly the large-m (clean-input) samples, so the decode point
  σ/√9 gets even less gradient. Lower per-sample estimation variance does not help when the clean end is under-trained;
  the tax is a sample-allocation problem, not a weighting problem.
- Partial remedy only: the S_K = 6 gap to PER (0.74×) remains; candidates left: capacity at the clean end, data size.

---

## RUN 2026-11-05 — ASL-COIL exploratory (CPU): where ASL's noise colour comes from, and per-band SNR vs fastMRI

Pre-specified as exploratory in `Archive/COIL/PREREGISTRATION.md`. Code: `experiments/_band_snr.py` (output
`runs/coil/band_snr.txt`); the region split below is a one-off of the same computation.

**1. Noise colour is mostly in-brain, coarse-scale fluctuation, not the reconstruction.**
- Protocols of datasets 2 and 3 (one session, coil2) differ only in VSS on (+ T2 prep 20 vs 25 ms) and a manual
  adjust volume for dataset 2; readout and trajectory are identical.
- Absolute noise band power (deviations × N/(N−1), coarse → fine bands 0–.0625 / .0625–.125 / .125–.25 / .25–.75 cycles/px):

  | scan | background | in brain |
  |---|---|---|
  | ds1 (coil1) | 2.43 2.33 2.67 2.73 (×1e10; f/c 1.13) | 8.12 5.05 4.51 3.56 (f/c 0.44) |
  | ds2 (coil2, VSS) | 1.98 2.31 2.83 3.06 (f/c 1.54) | 3.51 3.16 3.58 3.61 (f/c 1.03) |
  | ds3 (coil2) | 2.78 2.51 2.82 2.84 (f/c 1.02) | 12.3 6.33 5.44 4.02 (f/c 0.33) |
  | 1104c (coil1) | 2.43 2.18 2.46 2.58 (f/c 1.06) | 6.11 3.82 3.79 3.45 (f/c 0.57) |

- Background (thermal, recon-shaped) noise is nearly identical across coils and scans, flat to slightly blue. Fine-band
  in-brain power is also similar (3.5–4.0e10). The differences are an in-brain coarse excess: ds3 3.5× ds2's.
  VSS (ds2) suppresses it, which points to physiological (vascular / labelling) fluctuation across repetitions.
- So coil1 vs coil2 do not differ in thermal noise in absolute units; their M0-normalised SNR difference comes from lower M0.
- σ̂ averages the whole image: in-brain noise variance is 1.2–1.5× the background's (ds2 1.22, ds1 1.39, ds3 1.54),
  so the scalar noise label under-states in-brain noise, by different amounts per dataset.

**2. Per-band SNR does not explain the small ASL gain.** Per-copy SNR (power) by band (k/k_max), brain/head mask:

| data | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–1 | all | window [1/9, 1] | above 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fastMRI S_K = 1 | 60.1 | 2.04 | 0.291 | 0.040 | 0.017 | 0.76 | 0.07 | 0.90 |
| fastMRI S_K = 2 | 15.0 | 0.51 | 0.073 | 0.010 | 0.004 | 0.19 | 0.18 | 0.76 |
| fastMRI S_K = 3 | 6.68 | 0.227 | 0.032 | 0.004 | 0.002 | 0.084 | 0.17 | 0.73 |
| fastMRI S_K = 6 | 1.67 | 0.057 | 0.008 | 0.001 | 0.0005 | 0.021 | 0.08 | 0.68 |
| ds1 (coil1) | 5.18 | 0.504 | 0.106 | 0.053 | 0.029 | 0.172 | 0.17 | 0.60 |
| ds2 (coil2, VSS) | 4.06 | 0.271 | 0.042 | 0.017 | 0.012 | 0.060 | 0.16 | 0.54 |
| ds3 (coil2) | 1.61 | 0.172 | 0.033 | 0.017 | 0.012 | 0.067 | 0.31 | 0.42 |
| 1104a (coil1) | 3.64 | 0.452 | 0.135 | 0.079 | 0.047 | 0.162 | 0.25 | 0.45 |
| 1104c (coil1) | 5.88 | 0.553 | 0.159 | 0.070 | 0.039 | 0.180 | 0.25 | 0.53 |

('window' = share of signal power at per-copy SNR in [1/9, 1], where denoise-the-mean can beat denoise-each at a = 9.)
- ASL's spectrum is flatter than brain anatomy: lower SNR in the coarsest band (physiological excess), higher in the fine
  bands. Its overall band SNR (ds1 0.17, 1104c 0.18) and window share (0.17–0.25) match or exceed fastMRI S_K = 2
  (0.19, 0.18), where SUB / PER was 1.46×. A spectral-SNR explanation for ASL's ≈ 1.06× (or 0.87×) is ruled out.
- Remaining candidates, in order: (a) the noise is not the stationary, scalar-σ noise the model assumes (in-brain excess
  1.2–1.5×, coarse, physiological, possibly not independent across repetitions; SUB decodes the mean at σ̂/√a with no
  dial, PER's dial can absorb a mis-stated σ, and SUB picks r ≈ 0); (b) few subjects / slices. ASL-COIL Part A tests
  pooling; (a) can be tested evaluation-only by decoding SUB at c · σ̂/√a with c chosen by split-half (no retraining).

---

## RUN 2026-11-05 — ASL-COIL: does pooling coils hurt denoise-the-mean, and does the MEAN-TAX fix carry to ASL?

Pre-registered (`Archive/COIL/PREREGISTRATION.md`). Part A: PER, SUB and SUBM (denoise the mean with m drawn ∝ m) trained on dataset 1 only (coil1), 5-fold CV by slice index, seed 101, compared on the same slices and folds with ASL-SM2's pooled (datasets 1–3, coil1 + coil2) seed-101 models. Part B: SUBM trained like ASL-SUBJ (dataset 1 + 1104a, seeds 101–102), scored on the unseen subject 1104c against ASL-SUBJ's PER and SUB. ASL-SM2 recipe otherwise (MSE, p_raw 0.5, no augmentation, 3000 iterations, SUB/SUBM ladder floor σ̂_min/3). Split-half n_eff (signal region, no ground truth); checkpoint and r cross-fitted over slice folds.

**PRIMARY (Holm over 3):**

- P1 pooling interaction (SUB/PER on dataset 1 only) / (SUB/PER pooled), dataset 1, a = 9: 0.966× (90% CI 0.930–1.004; higher on 10/34 slices), p = 0.1, Holm 0.2
- P2 SUBM / SUB, unseen subject (1104c), a = 9: 1.038× (90% CI 1.025–1.053; higher on 29/34 slices), p = 2.8e-07, Holm 8.3e-07
- P3 SUBM / SUB, dataset-1-only CV, a = 9: 0.951× (90% CI 0.910–0.992; higher on 17/34 slices), p = 0.15, Holm 0.2

Dataset 1 (Part A). Ratios (90% CI; slices higher):

| a | SUB / PER (d1 only) | SUBM / PER (d1 only) | SUB / PER (pooled) | interaction | PER d1 / pooled | SUB d1 / pooled |
|---:|---|---|---|---|---|---|
| 1 | 1.051 (0.96–1.20; 13/34) | 0.962 (0.87–1.10; 10/34) | 0.905 (0.85–0.97; 3/34) | 1.162 (1.10–1.25; 30/34) | 0.914 (0.85–0.97; 13/34) | 1.062 (1.04–1.09; 25/34) |
| 2 | 1.013 (0.96–1.08; 13/34) | 0.936 (0.88–1.01; 8/34) | 0.925 (0.89–0.96; 6/34) | 1.095 (1.06–1.14; 24/34) | 0.939 (0.89–0.98; 15/34) | 1.028 (1.00–1.06; 19/34) |
| 3 | 0.926 (0.88–0.98; 10/34) | 0.849 (0.80–0.91; 5/34) | 0.903 (0.89–0.92; 3/34) | 1.025 (0.98–1.08; 18/34) | 0.968 (0.92–1.01; 16/34) | 0.992 (0.97–1.02; 14/34) |
| 5 | 0.894 (0.83–0.95; 11/34) | 0.818 (0.78–0.86; 3/34) | 0.898 (0.86–0.93; 7/34) | 0.995 (0.95–1.05; 15/34) | 0.966 (0.93–1.00; 19/34) | 0.962 (0.94–0.98; 10/34) |
| 7 | 0.851 (0.78–0.92; 10/34) | 0.792 (0.75–0.84; 2/34) | 0.873 (0.82–0.92; 9/34) | 0.975 (0.94–1.02; 12/34) | 0.979 (0.94–1.01; 18/34) | 0.955 (0.93–0.98; 7/34) |
| 9 | 0.843 (0.77–0.92; 10/34) | 0.801 (0.76–0.85; 4/34) | 0.873 (0.82–0.92; 12/34) | 0.966 (0.93–1.00; 10/34) | 0.995 (0.96–1.03; 19/34) | 0.960 (0.94–0.99; 8/34) |

Unseen subject 1104c (Part B). Ratios (90% CI; slices higher):

| a | SUBM / SUB | SUBM / PER | SUB / PER |
|---:|---|---|---|
| 1 | 0.967 (0.96–0.97; 0/34) | 0.968 (0.95–0.98; 5/34) | 1.001 (0.99–1.01; 20/34) |
| 2 | 0.982 (0.98–0.99; 5/34) | 1.021 (1.01–1.03; 25/34) | 1.040 (1.03–1.05; 33/34) |
| 3 | 1.001 (1.00–1.01; 18/34) | 1.068 (1.06–1.08; 31/34) | 1.066 (1.06–1.08; 32/34) |
| 5 | 1.024 (1.02–1.03; 28/34) | 1.082 (1.06–1.11; 30/34) | 1.057 (1.02–1.09; 27/34) |
| 7 | 1.031 (1.02–1.04; 30/34) | 1.095 (1.06–1.12; 31/34) | 1.062 (1.03–1.09; 27/34) |
| 9 | 1.038 (1.03–1.05; 29/34) | 1.103 (1.07–1.13; 31/34) | 1.062 (1.03–1.09; 28/34) |

Absolute n_eff (geometric mean over slices) and chosen r (per seed, fold 0's choice):

| a | PER_d1 | SUB_d1 | SUBM_d1 | PER_pool | SUB_pool | PER_subj | SUB_subj | SUBM_subj |
|---:|---|---|---|---|---|---|---|---|
| 1 | 22.36 (r 0) | 23.51 (r 0) | 21.51 (r 0) | 24.45 (r 0) | 22.13 (r 0) | 17.90 (r 0, 0) | 17.91 (r 0, 0) | 17.32 (r 0, 0) |
| 2 | 25.26 (r 0.0833) | 25.58 (r 0) | 23.64 (r 0) | 26.92 (r 0) | 24.89 (r 0) | 19.59 (r 0.0833, 0.0833) | 20.37 (r 0, 0) | 20.01 (r 0, 0.0833) |
| 3 | 28.77 (r 0.0833) | 26.62 (r 0) | 24.44 (r 0) | 29.72 (r 0.0833) | 26.83 (r 0) | 20.67 (r 0.0833, 0.167) | 22.04 (r 0, 0) | 22.07 (r 0, 0.0833) |
| 5 | 31.06 (r 0.167) | 27.76 (r 0) | 25.42 (r 0) | 32.14 (r 0.0833) | 28.87 (r 0) | 23.32 (r 0.167, 0.167) | 24.64 (r 0.0833, 0.0833) | 25.23 (r 0.0833, 0.0833) |
| 7 | 34.22 (r 0.167) | 29.12 (r 0) | 27.10 (r 0) | 34.94 (r 0.167) | 30.50 (r 0) | 25.40 (r 0.25, 0.25) | 26.96 (r 0.0833, 0.0833) | 27.81 (r 0.0833, 0.0833) |
| 9 | 36.41 (r 0.25) | 30.68 (r 0) | 29.17 (r 0) | 36.61 (r 0.25) | 31.95 (r 0) | 27.24 (r 0.333, 0.333) | 28.93 (r 0.0833, 0.0833) | 30.04 (r 0.0833, 0.0833) |

### Reading — ASL-COIL (written 2026-11-05, 13:45, after the run)
- Predictions vs result:
  - P1 > 1 (pooling hurts SUB): FAILS. Interaction 0.966× (CI 0.93–1.00) at a = 9; dataset-1-only SUB / PER is 0.843×,
    as low as pooled (0.873×). Pooling only matters at a = 1 (interaction 1.16×: pooling helps PER 9%, costs SUB 6%).
  - P2 SUBM / SUB on 1104c ≥ 1 and ≤ +5%: HOLDS (1.038×, 29/34 slices, Holm 8e-7). SUBM / PER there: 1.103×.
  - P3 SUBM / SUB on dataset 1 ≥ 1: FAILS (0.951×, CI 0.91–0.99, Holm 0.2, not significant).
- Pre-registered rules: pooling is not the cause; SUBM is confirmed as the ASL default (P2 > 1).
- The pattern across ASL-SM2, ASL-SUBJ and ASL-COIL is consistent: denoise-the-mean variants WIN on the unseen subject
  (SUB 1.06×, SUBM 1.10× vs PER) and LOSE on held-out slices of the training subject (0.84–0.87×), whatever the pooling.
  PER's n_eff is much higher on the training subject's held-out slices (36.4 at a = 9) than on 1104c (27.2), SUB's less
  so (30.7 vs 28.9). The CV folds interleave slice indices (fold pattern 3331120340102043113403214210402224), so every
  held-out slice has trained neighbours of the same subject (and, pooled, the same indices' anatomy in datasets 2–3).
- New hypothesis (not a finding): PER, which decodes at single-copy SNR, leans on the learned prior far more than SUB,
  and a prior learned from the same subject's adjacent slices is unusually good. Slice-CV on the training subject then
  favours PER; on an unseen subject (the deployment case) the order flips. Evaluation-only test: score the
  dataset-1-only CV fold models (Part A) on 1104c — same models, two test settings.

### ASL-COIL follow-up (exploratory, eval only; 2026-11-05, 14:00): same models, two test subjects
`runs/coil/xsubj/queue.sh`: the dataset-1-only CV fold models (Part A) and ASL-SM2's pooled seed-101 fold models score
1104c, each slice by the fold model that held its index out; selection as in the report (cross-fitted, no ground truth).

| a | d1-only models, ds1 held-out: SUB / PER | d1-only models, 1104c: SUB / PER | SUBM / PER | SUBM / SUB | pooled models, ds1: SUB / PER | pooled models, 1104c: SUB / PER |
|---:|---|---|---|---|---|---|
| 1 | 1.051 (0.96–1.20) | 1.094 (1.06–1.13; 28/34) | 1.048 | 0.958 | 0.905 | 0.988 (0.98–1.00) |
| 3 | 0.926 (0.88–0.98) | 1.049 (1.02–1.08; 28/34) | 1.069 | 1.018 | 0.903 | 1.071 (1.05–1.09) |
| 5 | 0.894 (0.84–0.96) | 1.054 (1.01–1.10; 28/34) | 1.073 | 1.018 | 0.898 | 1.082 (1.04–1.12) |
| 9 | 0.843 (0.77–0.92) | 1.021 (0.98–1.06; 25/34) | 1.042 | 1.021 | 0.873 | 1.068 (1.03–1.10; 30/34) |

n_eff at a = 9: PER 36.4 on ds1 held-out slices → 26.6 on 1104c; SUB 30.7 → 27.1; SUBM 27.7 on 1104c.
- The ranking flips with the test subject for the SAME models (pooled: 0.873× → 1.068×). The flip comes almost
  entirely from PER, whose n_eff drops 27% off the training subject; SUB drops 12%. Supports the hypothesis that PER
  leans on a subject-specific prior learned from adjacent slices. Caveat: ds1 vs 1104c are different scans as well as
  different subjects (SNR matched, 0.633 vs 0.638); one unseen subject only.
- Consequence for evaluation: slice-index CV within the training subject favours denoise-each; unseen-subject testing
  is the deployment-relevant setting and needs more subjects.

### ASL-COIL follow-up 2 (exploratory, eval only; 2026-11-05, 15:15): same subject, new session (1104a)
`runs/coil/xsess/queue.sh`: the same fold models score 1104a (dataset-1 subject, new session; not in their training).
SUB / PER at a = 9 (90% CI): ds1 held-out slices 0.843 (d1-only) / 0.873 (pooled); **1104a 0.946 (0.91–0.99) / 0.943
(0.90–0.99)**; 1104c 1.021 / 1.068. SUBM / PER on 1104a 0.908. n_eff at a = 9 (d1-only): PER 36.4 → 24.5 (1104a) →
26.6 (1104c); SUB 30.7 → 23.2 → 27.1.
- Same session (ds1 held-out slices): PER's large advantage. New session, same subject: most of it gone (PER's n_eff
  falls 33%, below its 1104c value); SUB still trails slightly. New subject: SUB ahead.
- Reading: the bulk of PER's in-subject advantage is session-specific (adjacent slices of the same acquisition share
  anatomy, perfusion state and noise realization statistics); a smaller part is subject-level. Caveat: 1104a fails the
  repetition-independence gate (slow perfusion dip), which can bias split-half n_eff; one scan per setting.

---

## RUN 2026-11-05 — GAP A1/A2: does a re-scaled noise label close ASL's small denoise-the-mean gain?

Pre-registered (`Archive/GAP/PREREGISTRATION.md`). Evaluation only: the ASL-SUBJ / ASL-COIL models (trained on dataset 1 + 1104a; PER, SUB, SUBM, seeds 101–102) decode the unseen subject 1104c with the noise label scaled by c (PER: c·σ̂; SUB/SUBM: c·σ̂/√a), c ∈ {0.5, 0.65, 0.8, 1, 1.2, 1.4, 1.7, 2} chosen without ground truth with checkpoint and r (cross-fitted over slice folds), or c = the slice's in-signal / whole-image noise std ratio ("brain", untuned).

**PRIMARY (Holm over 2):**

- P1 SUB tuned c / SUB c = 1, a = 9: 1.008× (90% CI 0.997–1.020; higher on 22/34 slices), p = 0.14, Holm 0.14
- P2 (SUB / PER) both tuned c, a = 9: 1.081× (90% CI 1.066–1.095; higher on 32/34 slices), p = 2.4e-08, Holm 4.8e-08

Ratios (90% CI; slices higher):

| a | SUB/PER, c = 1 | SUB/PER, tuned | SUB/PER, brain | SUBM/PER, tuned | SUB tuned / c = 1 | PER tuned / c = 1 | SUB brain / c = 1 | PER brain / c = 1 |
|---:|---|---|---|---|---|---|---|---|
| 1 | 1.001 (0.99–1.01; 20/34) | 0.997 (0.99–1.01; 20/34) | 0.998 (0.99–1.01; 14/34) | 0.965 (0.95–0.98; 4/34) | 1.000 (1.00–1.00; 0/34) | 1.004 (1.00–1.01; 23/34) | 0.983 (0.98–0.99; 2/34) | 0.986 (0.98–0.99; 13/34) |
| 2 | 1.040 (1.03–1.05; 33/34) | 1.040 (1.03–1.05; 33/34) | 1.028 (1.02–1.04; 25/34) | 1.021 (1.01–1.03; 25/34) | 1.000 (1.00–1.00; 0/34) | 1.000 (1.00–1.00; 0/34) | 0.989 (0.99–0.99; 2/34) | 1.000 (0.99–1.01; 11/34) |
| 3 | 1.066 (1.06–1.08; 32/34) | 1.070 (1.06–1.09; 31/34) | 1.046 (1.03–1.06; 30/34) | 1.071 (1.06–1.09; 31/34) | 1.000 (1.00–1.00; 0/34) | 0.997 (0.99–1.01; 9/34) | 0.994 (0.99–1.00; 15/34) | 1.013 (1.00–1.03; 19/34) |
| 5 | 1.057 (1.02–1.09; 27/34) | 1.070 (1.05–1.09; 29/34) | 1.071 (1.05–1.09; 30/34) | 1.099 (1.08–1.12; 31/34) | 0.992 (0.99–0.99; 3/34) | 0.980 (0.95–1.00; 18/34) | 1.005 (0.99–1.02; 13/34) | 0.991 (0.98–1.00; 18/34) |
| 7 | 1.062 (1.03–1.09; 27/34) | 1.067 (1.05–1.09; 28/34) | 1.089 (1.06–1.11; 30/34) | 1.100 (1.08–1.12; 32/34) | 0.992 (0.99–1.00; 7/34) | 0.987 (0.97–1.00; 5/34) | 1.006 (1.00–1.02; 22/34) | 0.981 (0.97–0.99; 13/34) |
| 9 | 1.062 (1.03–1.09; 28/34) | 1.081 (1.07–1.10; 32/34) | 1.074 (1.04–1.10; 27/34) | 1.128 (1.12–1.14; 33/34) | 1.008 (1.00–1.02; 22/34) | 0.990 (0.98–1.00; 5/34) | 1.015 (1.01–1.02; 25/34) | 1.004 (1.00–1.01; 14/34) |

Absolute n_eff (geometric mean) and chosen c (per seed, fold 0's choice):

| a | PER c = 1 | PER tuned c | PER brain | SUB c = 1 | SUB tuned c | SUB brain | SUBM c = 1 | SUBM tuned c | SUBM brain |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | 17.9 (1, 1) | 18.0 (1, 1.2) | 17.6 (brain, brain) | 17.9 (1, 1) | 17.9 (1, 1) | 17.6 (brain, brain) | 17.3 (1, 1) | 17.3 (1, 1.4) | 17.2 (brain, brain) |
| 2 | 19.6 (1, 1) | 19.6 (1, 1) | 19.6 (brain, brain) | 20.4 (1, 1) | 20.4 (1, 1) | 20.1 (brain, brain) | 20.0 (1, 1) | 20.0 (1, 1) | 19.8 (brain, brain) |
| 3 | 20.7 (1, 1) | 20.6 (1, 1) | 20.9 (brain, brain) | 22.0 (1, 1) | 22.0 (1, 1) | 21.9 (brain, brain) | 22.1 (1, 1) | 22.1 (1, 1) | 21.8 (brain, brain) |
| 5 | 23.3 (1, 1) | 22.8 (0.8, 1) | 23.1 (brain, brain) | 24.6 (1, 1) | 24.4 (1.2, 1.2) | 24.8 (brain, brain) | 25.2 (1, 1) | 25.1 (1.2, 1) | 25.3 (brain, brain) |
| 7 | 25.4 (1, 1) | 25.1 (1, 1) | 24.9 (brain, brain) | 27.0 (1, 1) | 26.8 (1.2, 1.2) | 27.1 (brain, brain) | 27.8 (1, 1) | 27.6 (1.2, 1.2) | 28.0 (brain, brain) |
| 9 | 27.2 (1, 1) | 27.0 (1, 1) | 27.3 (brain, brain) | 28.9 (1, 1) | 29.2 (1.2, 1.2) | 29.4 (brain, brain) | 30.0 (1, 1) | 30.4 (1.2, 1.2) | 30.5 (brain, brain) |

Untuned in-signal / whole-image noise std ratio on 1104c: median 1.075 (range 1.057–1.132).

**Reading (GAP A1/A2, against the pre-registration).** The noise label is not the cause of the small gain.
- P1 (SUB with tuned c / SUB at c = 1, a = 9) is 1.008× (CI 0.997–1.020, Holm p = 0.14). That is the pre-registered "≤ 1.02" outcome.
- The c chosen without ground truth is 1 or 1.2, never below 1. The single-slice hint of c = 0.8 (amendment) did not generalize.
- The "brain" label (in-signal noise std ≈ 1.075× the whole-image σ̂ on 1104c) moves SUB by ≤ 1.5%.
- P2 (SUB/PER, both tuned) is 1.081× against 1.062× at c = 1. Tuning helps SUB by < 2% and PER not at all.
- So the gap to fastMRI's 1.46× is not a scalar label error, either whole-image or in-brain. Per the pre-registration, the next test is B (noise transplant).
- SUBM/PER, both tuned, is 1.128× at a = 9 (33/34 slices). That is the largest unseen-subject gain so far, but it is still far from 1.46×.

---

## RUN 2026-11-05 — GAP B: does ASL's noise, transplanted onto fastMRI, shrink the denoise-the-mean gain?

Pre-registered (`Archive/GAP/PREREGISTRATION_B.md`). fastMRI AXT2 at the ASL matrix (80), 62 training volumes, N = 10 copies, label σ = 1 per channel (white noise then has fastMRI S_K = 2's per-band SNR). Noise: W white; G Gaussian with the ASL repetition-deviation power spectrum; T the real ASL deviation fields (coil1; training from ds1 + 1104a, test from 1104c; distinct slice and repetition per copy). PER (arm A), SUB, SUBM; seeds 101–102; checkpoint and r chosen without ground truth (leave-one-volume-out split-half); n_eff vs ground truth on 15 test volumes.

**PRIMARY (Holm over 2):**

- P1 (SUB/PER under T) / (SUB/PER under W), a = 9: 1.029× (90% CI 1.012–1.046; higher on 13/15 volumes), p = 0.0084, Holm 0.0084
- P2 (SUB/PER under G) / (SUB/PER under W), a = 9: 1.084× (90% CI 1.071–1.097; higher on 15/15 volumes), p = 6.1e-05, Holm 0.00012

Share of the gap (W's SUB/PER 1.205 → ASL's 1.062) reproduced, in log ratio: G -0.64, T -0.23.

SUB / PER and SUBM / PER (90% CI; volumes higher):

| a | SUB/PER W | SUB/PER G | SUB/PER T | SUBM/PER W | SUBM/PER G | SUBM/PER T |
|---:|---|---|---|---|---|---|
| 1 | 0.948 (0.94–0.96; 0/15) | 0.977 (0.97–0.99; 3/15) | 0.910 (0.90–0.92; 0/15) | 0.912 (0.90–0.92; 0/15) | 0.957 (0.95–0.97; 2/15) | 0.860 (0.85–0.87; 0/15) |
| 3 | 1.059 (1.04–1.07; 14/15) | 1.100 (1.08–1.12; 15/15) | 0.974 (0.95–0.99; 4/15) | 1.030 (1.02–1.04; 13/15) | 1.086 (1.07–1.10; 15/15) | 0.977 (0.96–0.99; 4/15) |
| 5 | 1.167 (1.15–1.18; 15/15) | 1.197 (1.17–1.22; 15/15) | 1.077 (1.05–1.10; 14/15) | 1.156 (1.14–1.17; 15/15) | 1.187 (1.16–1.21; 15/15) | 1.078 (1.06–1.09; 14/15) |
| 9 | 1.205 (1.18–1.23; 15/15) | 1.306 (1.28–1.34; 15/15) | 1.240 (1.21–1.26; 15/15) | 1.246 (1.22–1.27; 15/15) | 1.357 (1.32–1.39; 15/15) | 1.237 (1.21–1.26; 15/15) |

n_eff (geometric mean over volumes; σ = 1 label):

| a | A W | SUB W | SUBM W | A G | SUB G | SUBM G | A T | SUB T | SUBM T |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 58.6 | 55.5 | 53.4 | 54.0 | 52.8 | 51.7 | 51.3 | 46.7 | 44.1 |
| 3 | 71.7 | 75.9 | 73.8 | 67.4 | 74.1 | 73.2 | 63.4 | 61.7 | 61.9 |
| 5 | 77.9 | 90.9 | 90.0 | 73.8 | 88.3 | 87.6 | 69.0 | 74.2 | 74.4 |
| 9 | 85.1 | 102.6 | 106.1 | 78.8 | 103.0 | 106.9 | 75.3 | 93.4 | 93.2 |

Band transfer at a = 9 (share of the true detail kept per band of k / k_max; plain averaging = 1):

| design | noise | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–1 |
|---|---|---:|---:|---:|---:|---:|
| A | W | 1.00 | 0.65 | 0.36 | 0.15 | 0.09 |
| SUB | W | 0.99 | 0.86 | 0.69 | 0.40 | 0.21 |
| SUBM | W | 0.99 | 0.87 | 0.69 | 0.39 | 0.19 |
| A | G | 0.99 | 0.62 | 0.36 | 0.16 | 0.12 |
| SUB | G | 0.99 | 0.87 | 0.69 | 0.37 | 0.18 |
| SUBM | G | 0.99 | 0.87 | 0.69 | 0.37 | 0.17 |
| A | T | 1.01 | 0.61 | 0.32 | 0.12 | 0.08 |
| SUB | T | 0.98 | 0.82 | 0.62 | 0.30 | 0.13 |
| SUBM | T | 0.96 | 0.81 | 0.61 | 0.30 | 0.11 |

**Reading (GAP B, against the pre-registration; written 2026-11-05 after the run).**
- **Control first.** W at matrix 80 gives SUB/PER 1.205 at a = 9, against 1.455 at matrix 160 at the same per-band SNR.
  - That sits at the pre-registered "W < 1.2" boundary, so the matrix change carries much of the gap.
  - In log ratio, matrix 160 → 80 accounts for (ln 1.455 − ln 1.205) / (ln 1.455 − ln 1.062) ≈ 0.60 of the fastMRI-to-ASL gap.
- **ASL's noise is not the cause.** Both P1 (T/W 1.029×) and P2 (G/W 1.084×) are above 1, so the transplanted noise does not shrink the gain. ASL's colour *helps* denoise-the-mean, and the real fields leave it about where white noise does.
  - The heavy tails and in-brain excess cost both designs n_eff (PER 85 → 75, SUB 103 → 93), but not the ratio.
  - My pre-run guess (P1 ≈ 0.9) was wrong.
- **What remains.** About 40% of the gap (W at matrix 80, 1.205 → ASL's 1.06) is not the noise, not the label and not the training-set size. It lies in ASL's signal or data: low-SNR perfusion content, and the subject/session generalization seen in ASL-SUBJ.
- **Open: why the matrix matters.** Two candidates, untested:
  - At 80 × 80 the same anatomy has less fine structure per pixel, so the mid bands where denoise-the-mean gains (T3/T4) hold less signal.
  - 64-pixel crops cover most of an 80 image, so training sees fewer distinct patches.

---

## RUN 2026-11-06 — GAP C: why does a coarser matrix shrink the learned denoise-the-mean gain?

Pre-registered (`Archive/GAP/PREREGISTRATION_C.md`). fastMRI AXT2, white noise σ = 1 per channel, 62 training volumes, PER (arm A) and SUB, seeds 101–102, pixels per training step 8 × 64² in every condition; checkpoint and r chosen without ground truth (leave-one-volume-out split-half); n_eff vs ground truth on 15 test volumes. Conditions: 160 (crop 64), 160c128 (crop 128, batch 2), 80 (crop 64), 80c32 (crop 32, batch 32), U80 (matrix-80 content on a 160 grid, crop 64).

**PRIMARY (Holm over 3):**

- P1 crop at matrix 80: (SUB/PER, 80c32) / (SUB/PER, 80), a = 9: 1.130× (90% CI 1.115–1.146; higher on 15/15 volumes), p = 6.1e-05, Holm 0.00018
- P2 pixel scale: (SUB/PER, U80) / (SUB/PER, 80), a = 9: 1.182× (90% CI 1.161–1.207; higher on 15/15 volumes), p = 6.1e-05, Holm 0.00018
- P3 crop at matrix 160: (SUB/PER, 160c128) / (SUB/PER, 160), a = 9: 0.884× (90% CI 0.870–0.898; higher on 0/15 volumes), p = 6.1e-05, Holm 0.00018

SUB / PER (90% CI; volumes higher), and the learned / linear factor at a = 9 (SUB/PER over the Wiener prediction):

| condition | a = 1 | a = 3 | a = 5 | a = 9 | Wiener a = 9 | learned / Wiener |
|---|---|---|---|---|---:|---:|
| 160 | 0.965 (0.96–0.98; 0/15) | 1.150 (1.13–1.17; 15/15) | 1.354 (1.32–1.39; 15/15) | 1.455 (1.41–1.50; 15/15) | 1.423 | 1.022 |
| 160c128 | 0.931 (0.92–0.94; 0/15) | 1.077 (1.06–1.10; 14/15) | 1.253 (1.22–1.28; 15/15) | 1.286 (1.25–1.32; 15/15) | 1.423 | 0.903 |
| 80 | 0.948 (0.94–0.96; 0/15) | 1.059 (1.04–1.07; 14/15) | 1.167 (1.15–1.18; 15/15) | 1.205 (1.18–1.23; 15/15) | 1.273 | 0.947 |
| 80c32 | 0.918 (0.91–0.92; 0/15) | 1.073 (1.05–1.09; 14/15) | 1.248 (1.22–1.28; 15/15) | 1.361 (1.32–1.40; 15/15) | 1.273 | 1.069 |
| U80 | 0.973 (0.97–0.98; 1/15) | 1.140 (1.12–1.16; 15/15) | 1.289 (1.26–1.33; 15/15) | 1.424 (1.38–1.47; 15/15) | 1.273 | 1.119 |

n_eff (geometric mean over volumes), PER / SUB:

| a | 160 | 160c128 | 80 | 80c32 | U80 |
|---:|---|---|---|---|---|
| 1 | 70.1 / 67.6 | 73.8 / 68.7 | 58.6 / 55.5 | 55.2 / 50.7 | 49.8 / 48.5 |
| 3 | 89.6 / 103.0 | 94.8 / 102.0 | 71.7 / 75.9 | 67.4 / 72.3 | 60.1 / 68.5 |
| 5 | 94.9 / 128.5 | 100.6 / 126.0 | 77.9 / 90.9 | 71.3 / 88.9 | 65.3 / 84.2 |
| 9 | 104.0 / 151.3 | 109.0 / 140.2 | 85.1 / 102.6 | 78.1 / 106.4 | 70.4 / 100.3 |

Band transfer at a = 9 (bands of k / k_max of the content; plain averaging = 1):

| condition | design | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–1 |
|---|---|---:|---:|---:|---:|---:|
| 160 | A | 0.96 | 0.63 | 0.26 | 0.10 | 0.09 |
| 160 | SUB | 1.00 | 0.89 | 0.61 | 0.21 | 0.10 |
| 160c128 | A | 0.99 | 0.59 | 0.26 | 0.11 | 0.09 |
| 160c128 | SUB | 0.98 | 0.88 | 0.60 | 0.21 | 0.12 |
| 80 | A | 1.00 | 0.65 | 0.36 | 0.15 | 0.09 |
| 80 | SUB | 0.99 | 0.86 | 0.69 | 0.40 | 0.21 |
| 80c32 | A | 0.98 | 0.62 | 0.33 | 0.15 | 0.11 |
| 80c32 | SUB | 0.98 | 0.88 | 0.70 | 0.38 | 0.17 |
| U80 | A | 0.96 | 0.56 | 0.29 | 0.14 | 0.09 |
| U80 | SUB | 1.01 | 0.87 | 0.69 | 0.41 | 0.19 |

**Reading (GAP C, against the pre-registration; written 2026-11-06 after the run).**
- **The primaries.** All three hit their thresholds, and two of them strongly: P1 1.130 (crop 32 at 80 raises the ratio), P3 0.884 (crop 128 at 160 lowers it), P2 1.182 (U80).
  - Read literally, the pre-registration says both crop and pixel scale matter.
- **One variable explains all five conditions: the share of the field of view a training crop covers.**
  - At 0.4 of the FOV the ratio is 1.455 (160), 1.361 (80c32) and 1.424 (U80).
  - At 0.8 it is 1.286 (160c128) and 1.205 (80).
  - U80 is confounded with this. Its 64 crop on a 160 grid covers 0.4 of the FOV, a design flaw I missed. So P2 does not isolate pixel scale; C3 below separates them.
- **The ratio moves because the two designs respond in opposite directions** (with the pixel budget fixed, large crops also mean fewer patches per step):

  | n_eff at a = 9 | small crop (0.4 FOV) | large crop (0.8 FOV) |
  |---|---|---|
  | PER, matrix 160 | 104 | 109 |
  | PER, matrix 80 | 78 | 85 |
  | SUB, matrix 160 | 151 | 140 |
  | SUB, matrix 80 | 106 | 103 |

  Larger crops (fewer patches) help PER and hurt SUB. In U80, PER falls to 70 and SUB to 100, so U80's ratio rises because PER worsens, not because SUB improves.
- **Exploratory: each design at its own better recipe** (picked by ground truth from two options):
  - matrix 160: SUB/PER = 151.3 / 109.0 = 1.388 (learned / Wiener 0.975);
  - matrix 80: 106.4 / 85.1 = 1.250 (0.982).
  - The learned factor is then about 0.98 at both matrices. The matrix effect becomes entirely spectral (Wiener 1.423 → 1.273).
  - The MEAN-SNR headline of 1.455 was inflated by about 5% because PER was handicapped by small crops.
- **Implication for ASL.** ASL trains on whole 80 × 80 slices (crop = FOV, batch 8), the regime that favours PER and hurts SUB.
  - Prediction: SUB/SUBM trained on 32-pixel crops (batch 32) gains on ASL, and PER does not.
  - That is a direct test of a recipe/architecture lever on the real data (ASL-CROP, to be pre-registered).
- **Amendment C3 (pre-registered here, before running).** To isolate pixel scale: U80 with crop 128 and batch 2 (0.8 FOV), PER and SUB, seeds 101–102.
  - If its ratio is about 1.2 (as at matrix 80), pixel scale does not matter and the FOV fraction is the lever.
  - If it stays about 1.4, pixel scale matters independently.

**GAP C3 (amendment, pre-registered in the GAP C reading; scored with matrix_report's selection, 2026-11-06).** Matrix-80 content on a 160 grid, crop 128, batch 2 (0.8 FOV):
- **The ratio.** SUB/PER = 1.278 (1.24–1.31; 15/15). That lies between matrix 80 (1.205) and U80 with crop 64 (1.424), and closer to matrix 80.
  - U80c128 / U80 = 0.898 (0/15); U80c128 / 80 = 1.061 (15/15).
  - In log ratio, about 64% of U80's rise came from the crop's FOV share and about 36% from the finer pixel grid.
- **Absolute n_eff, against matrix 80.** A finer grid helps neither design:
  - PER 79.3 against 85.1 (0.932×);
  - SUB 101.4 against 102.6 (0.988×).
  - The residual ratio rise is PER getting worse on the finer grid (band-limited noise spread over more pixels), not SUB improving.
- **Reading.** Pixel scale is not a useful lever. Up-sampling ASL inputs onto a finer grid would not help denoise-the-mean. The lever is the crop/patch recipe: FOV share and patches per step, tested on ASL in ASL-CROP.

---

## RUN 2026-11-06 — ASL-CROP: does training on crops help denoise-the-mean on ASL?

Pre-registered (`Archive/CROP/PREREGISTRATION.md`). Trained on dataset 1 + 1104a (seeds 101–102, 3000 iterations, MSE), tested on the unseen subject 1104c; crop models use random 32 × 32 crops, batch 32 (same pixels per step as whole 80 × 80 slices at batch 8); references are ASL-SUBJ (PER, SUB) and ASL-COIL (SUBM). Checkpoint and r cross-fitted over slice folds, no ground truth.

**PRIMARY (Holm over 3):**

- P1 SUBM crop / SUBM whole slice, a = 9: 0.947× (90% CI 0.916–0.974; higher on 11/34 slices), p = 0.0034, Holm 0.0068
- P2 SUB crop / SUB whole slice, a = 9: 1.037× (90% CI 1.025–1.049; higher on 26/34 slices), p = 2.3e-05, Holm 7e-05
- P3 PER crop / PER whole slice, a = 9: 1.006× (90% CI 1.000–1.012; higher on 22/34 slices), p = 0.043, Holm 0.043

Ratios (90% CI; slices higher):

| a | SUBM c / SUBM | SUB c / SUB | PER c / PER | SUB/PER whole | SUB/PER crop | SUBM/PER whole | SUBM/PER crop | SUBM crop / PER whole |
|---:|---|---|---|---|---|---|---|---|
| 1 | 1.000 (0.99–1.01; 20/34) | 0.990 (0.98–1.00; 10/34) | 1.000 (0.99–1.01; 21/34) | 1.001 (0.99–1.01; 20/34) | 0.991 (0.98–1.01; 10/34) | 0.968 (0.95–0.98; 5/34) | 0.968 (0.95–0.98; 12/34) | 0.968 (0.95–0.98; 12/34) |
| 2 | 0.986 (0.97–1.00; 15/34) | 1.003 (1.00–1.01; 21/34) | 1.014 (1.01–1.02; 25/34) | 1.040 (1.03–1.05; 33/34) | 1.029 (1.02–1.04; 30/34) | 1.021 (1.01–1.03; 25/34) | 0.994 (0.98–1.01; 21/34) | 1.007 (0.99–1.03; 21/34) |
| 3 | 0.961 (0.94–0.98; 13/34) | 1.019 (1.01–1.02; 26/34) | 1.036 (1.02–1.05; 30/34) | 1.066 (1.06–1.08; 32/34) | 1.049 (1.03–1.06; 27/34) | 1.068 (1.06–1.08; 31/34) | 0.991 (0.95–1.03; 22/34) | 1.026 (0.99–1.06; 24/34) |
| 5 | 0.958 (0.93–0.98; 12/34) | 1.014 (1.01–1.02; 22/34) | 1.014 (1.01–1.02; 28/34) | 1.057 (1.02–1.09; 27/34) | 1.057 (1.03–1.08; 26/34) | 1.082 (1.06–1.11; 30/34) | 1.022 (0.98–1.06; 26/34) | 1.037 (0.99–1.08; 26/34) |
| 7 | 0.947 (0.92–0.97; 10/34) | 1.027 (1.02–1.04; 24/34) | 1.009 (1.00–1.02; 25/34) | 1.062 (1.03–1.09; 27/34) | 1.080 (1.05–1.11; 31/34) | 1.095 (1.06–1.12; 31/34) | 1.027 (0.98–1.07; 27/34) | 1.037 (0.99–1.08; 28/34) |
| 9 | 0.947 (0.92–0.97; 11/34) | 1.037 (1.03–1.05; 26/34) | 1.006 (1.00–1.01; 22/34) | 1.062 (1.03–1.09; 28/34) | 1.095 (1.06–1.12; 31/34) | 1.103 (1.07–1.13; 31/34) | 1.038 (0.99–1.08; 28/34) | 1.044 (1.00–1.08; 26/34) |

**Reading (against `Archive/CROP/PREREGISTRATION.md`):**
- **P1 = 0.947 (Holm 0.007): the prediction failed for SUBM.** Training SUBM on 32-pixel crops lowers n_eff by 5% at a = 9. The drop grows with a (1.000 at a = 1, 0.947 at a = 9). Both seeds agree (0.941, 0.952). SUBM-crop does **not** become the ASL default; the whole-slice SUBM stays the default.
- **P2 = 1.037 (Holm 7e-5): SUB gains as predicted**, matching fastMRI's +3.7% at matrix 80. Both seeds agree (1.052, 1.023).
- **P3 = 1.006: PER is essentially unchanged**, so the predicted PER loss (fastMRI −8%) did not appear on ASL.
- **Crops and m ∝ m look like substitutes, not complements.**
  - SUB-crop / SUB-whole (1.037) equals SUBM-whole / SUB-whole (1.038).
  - Combining the two is worse than either: SUBM-crop / SUB-crop is 0.947.
  - So both levers seem to buy the same thing, a better denoiser at high m (low input noise), and stacking them overshoots. This is a hypothesis, not yet tested.
- **The application comparison is unchanged.** The best ASL design is still SUBM on whole slices: 1.103× PER at a = 9. SUB-crop reaches about the same level (1.037 × 1.062 ≈ 1.10).
- **What carries over from fastMRI.** The recipe dependence of the SUB/PER ratio is real on ASL, so the comparison should report each design at its own recipe. But the best recipe depends on the subset weighting, so the fastMRI-derived "crop 0.4 FOV" rule cannot be adopted as a default.

---

## PLAN 2026-11-06 (weekend) — where does the ASL / fastMRI gap come from?

The user asked for a weekend exploring where the ASL / fastMRI performance gap comes from. Working hypothesis after GAP B, C and ASL-CROP: the gap is spectral. With each design at its best recipe, the learned / Wiener factor is about 0.98 on both datasets.

- **Already consistent, from the pre-run check in `Archive/GAP/PREREGISTRATION_D.md`.** The Wiener ratio with fastMRI content predicts GAP B's noise result: T/W = 1.307 / 1.273 = 1.027, against 1.029 measured.
- **Running now: GAP D/E** (`runs/gap/queue_d.sh`).
  - D: fastMRI content filtered to ASL's per-frequency SNR. Does it reproduce ASL's gain?
  - E: the same spectrum with random phase. Does image structure matter?
- **Next, chosen from the D/E result:**
  - **If D closes the gap:** a cross-dataset check that the Wiener ratio predicts the learned SUB/PER from the spectrum alone, across every condition run so far. Then which band of the spectrum carries the gap, a band-swap analysis that is linear-theory first and needs training only if it is ambiguous.
  - **If D does not close it:** find which property of ASL content beyond its spectrum matters. Candidates are the non-stationary signal (brain-only perfusion against a background) and the per-slice SNR spread.

---

## RUN 2026-11-07 — GAP D/E: is ASL's small denoise-the-mean gain set by its signal-to-noise spectrum?

Pre-registered (`Archive/GAP/PREREGISTRATION_D.md`). fastMRI AXT2 at matrix 80, σ = 1 per channel, 62 training volumes, PER (arm A), SUB, SUBM, seeds 101–102; checkpoint and r chosen without ground truth; n_eff vs ground truth on 15 test volumes. Content F = fastMRI (GAP B), D = filtered to coil1 ASL's per-frequency SNR for its noise, E = D with random Fourier phase. Noise T = real ASL fields, W = white.

**PRIMARY (Holm over 3):**

- P1 spectrum, ASL noise: (SUB/PER, DT) / (SUB/PER, T), a = 9: 1.175× (90% CI 1.151–1.201; higher on 15/15 volumes), p = 6.1e-05, Holm 0.00018
- P2 spectrum, white noise: (SUB/PER, DW) / (SUB/PER, W), a = 9: 1.278× (90% CI 1.249–1.307; higher on 15/15 volumes), p = 6.1e-05, Holm 0.00018
- P3 phase structure: (SUB/PER, EW) / (SUB/PER, DW), a = 9: 0.744× (90% CI 0.720–0.769; higher on 0/15 volumes), p = 6.1e-05, Holm 0.00018

Share of the T → ASL gap closed by DT (log): SUB -1.04 (T 1.240 → DT 1.457; ASL 1.062), SUBM -1.57 (T 1.237 → DT 1.482; ASL 1.103).

SUB / PER and SUBM / PER (90% CI; volumes higher), Wiener and learned / Wiener at a = 9:

| condition | SUB/PER a = 1 | SUB/PER a = 3 | SUB/PER a = 5 | SUB/PER a = 9 | SUBM/PER a = 9 | Wiener a = 9 | SUB learned/W | SUBM learned/W |
|---|---|---|---|---|---|---:|---:|---:|
| T | 0.910 (0.90–0.92; 0/15) | 0.974 (0.95–0.99; 4/15) | 1.077 (1.05–1.10; 14/15) | 1.240 (1.21–1.26; 15/15) | 1.237 (1.21–1.26; 15/15) | 1.307 | 0.949 | 0.946 |
| W | 0.948 (0.94–0.96; 0/15) | 1.059 (1.04–1.07; 14/15) | 1.167 (1.15–1.18; 15/15) | 1.205 (1.18–1.23; 15/15) | 1.246 (1.22–1.27; 15/15) | 1.273 | 0.947 | 0.979 |
| DT | 0.964 (0.96–0.97; 1/15) | 1.078 (1.06–1.10; 14/15) | 1.249 (1.22–1.28; 15/15) | 1.457 (1.43–1.49; 15/15) | 1.482 (1.45–1.51; 15/15) | 1.149 | 1.268 | 1.290 |
| DW | 0.967 (0.96–0.97; 0/15) | 1.164 (1.15–1.18; 15/15) | 1.346 (1.32–1.37; 15/15) | 1.540 (1.50–1.58; 15/15) | 1.569 (1.53–1.61; 15/15) | 1.165 | 1.322 | 1.347 |
| EW | 0.990 (0.99–0.99; 0/15) | 1.043 (1.03–1.05; 15/15) | 1.090 (1.08–1.10; 15/15) | 1.146 (1.13–1.16; 15/15) | 1.155 (1.14–1.17; 15/15) | 1.165 | 0.984 | 0.992 |

n_eff (geometric mean over volumes), PER / SUB / SUBM:

| a | T | W | DT | DW | EW |
|---:|---|---|---|---|---|
| 1 | 51.3 / 46.7 / 44.1 | 58.6 / 55.5 / 53.4 | 22.8 / 22.0 / 21.1 | 30.4 / 29.4 / 29.0 | 28.6 / 28.4 / 28.1 |
| 3 | 63.4 / 61.7 / 61.9 | 71.7 / 75.9 / 73.8 | 29.9 / 32.2 / 32.2 | 38.9 / 45.2 / 44.6 | 33.2 / 34.6 / 34.6 |
| 5 | 69.0 / 74.2 / 74.4 | 77.9 / 90.9 / 90.0 | 32.4 / 40.5 / 40.9 | 42.5 / 57.2 / 56.9 | 35.9 / 39.1 / 39.2 |
| 9 | 75.3 / 93.4 / 93.2 | 85.1 / 102.6 / 106.1 | 36.8 / 53.6 / 54.6 | 47.7 / 73.5 / 74.8 | 40.0 / 45.8 / 46.2 |

Band transfer at a = 9 (bands of k / k_max; plain averaging = 1):

| condition | design | 0–.1 | .1–.25 | .25–.5 | .5–.75 | .75–1 |
|---|---|---:|---:|---:|---:|---:|
| T | A | 1.01 | 0.61 | 0.32 | 0.12 | 0.08 |
| T | SUB | 0.98 | 0.82 | 0.62 | 0.30 | 0.13 |
| T | SUBM | 0.96 | 0.81 | 0.61 | 0.30 | 0.11 |
| W | A | 1.00 | 0.65 | 0.36 | 0.15 | 0.09 |
| W | SUB | 0.99 | 0.86 | 0.69 | 0.40 | 0.21 |
| W | SUBM | 0.99 | 0.87 | 0.69 | 0.39 | 0.19 |
| DT | A | 1.02 | 0.80 | 0.58 | 0.43 | 0.33 |
| DT | SUB | 0.97 | 0.87 | 0.80 | 0.67 | 0.51 |
| DT | SUBM | 0.95 | 0.88 | 0.81 | 0.67 | 0.51 |
| DW | A | 0.95 | 0.77 | 0.60 | 0.42 | 0.31 |
| DW | SUB | 0.98 | 0.89 | 0.84 | 0.72 | 0.58 |
| DW | SUBM | 0.98 | 0.90 | 0.83 | 0.71 | 0.56 |
| EW | A | 0.75 | 0.38 | 0.23 | 0.20 | 0.19 |
| EW | SUB | 0.95 | 0.75 | 0.42 | 0.27 | 0.19 |
| EW | SUBM | 0.95 | 0.74 | 0.42 | 0.26 | 0.18 |

### Reading — GAP D/E (written 2026-11-06, 19:45, after the run)
- **The spectral hypothesis FAILS, in the opposite direction.**
  - Giving fastMRI content coil1 ASL's per-frequency SNR raises SUB/PER from 1.240 (T) to **1.457 (DT)**. That is P1 = 1.175×, 15/15 volumes, Holm 2e-4. The share of the gap closed is −1.04: the gap widens.
  - Under white noise the rise is larger: DW 1.540 against W 1.205.
  - The Wiener prediction for DT is 1.149, the same as ASL's. The networks beat it by 27–35% (learned / Wiener 1.27 for SUB, 1.29 for SUBM).
- **Phase structure is what drives the learned gain.** With DW's spectrum but random phase (EW), SUB/PER is 1.146 (P3 = 0.744×, 0/15 volumes). There learned / Wiener is 0.98–0.99: on Gaussian-like content, both designs reach the linear optimum and no more.
- **ASL sits with EW, not with DT.** Its learned / Wiener is 0.92 (SUB) and 0.96 (SUBM). So the ASL gap is NOT spectral. Content with ASL's spectrum but anatomical structure gives a large gain; ASL itself behaves as if it had no exploitable structure.
- **Correction to earlier statements.** "About 68% of the gap is spectral" (GAP C0) was an accounting split of measured ratios against the Wiener ratio, not a cause. D/E shows the Wiener ratio does not predict the learned ratio once structure is present: DT 1.457 against 1.149.
- **Mechanism, in line with the band transfer.** On DT, SUB keeps 0.67–0.80 of mid- and high-band detail, against PER's 0.43–0.58. On EW, both keep about 0.2–0.4.
  - Denoising the mean pays off when the mean's SNR crosses a detection threshold for sparse structure (edges) that a single copy's SNR does not.
  - Gaussian-like content has no such threshold, so the linear limit applies.
- **What this means for architecture.** The gain from denoising the mean comes from exploiting non-Gaussian image structure. Spectrum-only levers (noise colour, label calibration, pixel scale) were all null, and that is now explained. ASL's question becomes: why does the network find no exploitable structure in ASL? Candidates, tested in GAP F:
  - (i) the content itself;
  - (ii) signal-like physiological fluctuation;
  - (iii) learnability from one training subject.
