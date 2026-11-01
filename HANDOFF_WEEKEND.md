# Weekend handoff (2026-10-30 → Monday 2026-11-02)

Written so that work can resume from disk alone (e.g. after a context compaction). Update the checklist as steps
complete; the final deliverable is a decision-ready summary for the user on Monday (log section in
`WEEKEND_ITERATIONS.md` + a short message).

## Running (all self-contained; each script skips completed steps, so re-running resumes)
| Queue | Command (from repo root) | Does | Finishes with |
|---|---|---|---|
| BASE-DL | `bash runs/base/dl_queue.sh` | trains Neighbor2Neighbor / R2R / Noise2Self (3 seeds × 5 folds), scores them + the two Noise2Noise arms | `runs/base/dl_eval/neff.csv` |
| NAT CPU | `bash runs/nat/queue_cpu.sh` | plain averaging + BM3D on Set12/BSD68, 4 conditions | `runs/nat/cpu.done` |
| NAT GPU | `bash runs/nat/queue_gpu.sh` | WAITS for `runs/base/dl_eval/neff.csv` AND `runs/nat/_ch_verified`; trains 8 methods × 4 conditions (3000 it), evaluates | `runs/nat/gpu.done` |

Check progress: `ls runs/base/dl_eval runs/nat/eval/*`, the `train.log` files, `runs/*/logs/`. Never start a second
GPU job while one runs (8 GB card: two trainings page and crawl; a memory squeeze once killed a run).

## Gate to release (do this first)
`runs/nat/_ch_verified` (RELEASED 2026-10-30: check passed, loss identical) is created ONLY after the CPU determinism check of the `N2N_DSM(ch=...)` change passes:
run (CPU, single thread, CUDA_VISIBLE_DEVICES=-1, `torch.use_deterministic_algorithms(True)`)
`experiments/_cv_train.py --fold 0 --iters 50` and confirm iteration-50 `train_loss_avg50` == `0.08874751314520836`.
If it passed: `touch runs/nat/_ch_verified`. If not: do NOT release; fix `n2n.py` (diff vs commit before the change).

## Then, in order
1. ~~DONE 2026-10-31 03:05 (logged; ours wins all 5 at a = 9; classic N2N wins single maps and datasets 2–3)~~ **BASE-DL report** when `runs/base/dl_eval/neff.csv` exists: `cd baselines && python report.py --set dl --dry`
   (must print DRY RUN OK), then `python report.py --set dl` (logs its section). Pre-registration:
   `Archive/BASE/PREREGISTRATION_DL.md` (primary: ours vs each baseline + dial, dataset 1, a = 9, Holm over 5).
2. ~~DONE 2026-11-01~~ **NAT report** — `natural/report.py` (written, compiles, NOT yet run on data): when `runs/nat/gpu.done` and
   `runs/nat/cpu.done` exist, run `python natural/report.py --dry` (must print DRY RUN OK), then
   `python natural/report.py` (logs its section only when all conditions x methods are present). Sanity-check its
   numbers before trusting them (e.g. plain averaging PSNR must match σ²/a).
3. ~~DONE (benign: _ch_verify scratch, RESEARCH_NOTES edit)~~ **Verify** protected hashes: `sha256sum -c Archive/NAT/protected_before.sha256` (and BASE's); report mismatches.
4. **Commit** text outputs only (code, logs, json, md; never csv/png/pkl/npy), on branch `dev`. Do not push without
   asking.
5. (2026-11-01: NAT report logged, reading logged, notes 6c/6d updated; only the user summary remains) **Monday summary:** update `docs/RESEARCH_NOTES.md` (step 3 results, notes; NO venue, NO data plan — user's
   request) and give the user a decision-ready summary: estimator validated?, GT-free r selection good enough?,
   ours vs baselines with exact PSNR, does the averaging-aware loss beat ours?

## Context for whoever resumes
- Research notes: `docs/RESEARCH_NOTES.md`; experiment log: `WEEKEND_ITERATIONS.md`; baselines: `baselines/README.md`.
- Natural images: `../natural_images/{train400,bsd68,set12}` (DnCNN split; source in `SOURCE.txt`).
- fastMRI (user-provided, for the later resolution study; read-only): `E:/fastMRI/data` (311 multicoil brain h5).
- External packages for baselines are in `../_pydeps` (bm3d, dipy), added to the path by `baselines/_common.py`.
- .csv/.png written by Python are encrypted at rest on this machine: read them with Python only.
