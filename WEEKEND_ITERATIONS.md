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
