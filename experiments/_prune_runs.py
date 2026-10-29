# -*- coding: utf-8 -*-
"""Housekeeping 2026-10-29: prune local checkpoints and superseded run folders (dry run by default).

Keeps a checkpoint only if a reported result uses it: an evaluation exists at that iteration, or it is the
declared primary / selected iteration. Removes the pre-fix folders (before the R3 total-noise label and
channel-layout fix), whose results no longer bear on any current finding. Text outputs (logs, json, csv)
of kept experiments are untouched. Untracked code and notes were archived first in
../diffusion_pruned_2026-10-29.tar.gz.

    python experiments/_prune_runs.py            # dry run: list and size
    python experiments/_prune_runs.py --execute
"""
import os, re, sys, glob, shutil, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
os.chdir(ROOT)

DROP_DIRS = ["runs/r0_eval", "runs/r0_split", "runs/displays", "runs/compare", "runs/_smoke",
             "checkpoints/SR_SWEEP", "checkpoints/SRSTR_01", "checkpoints/FRONTIER"]
DROP_FILES = ["runs/_sr_dry.txt", "runs/_sr_block_preview.md"]
# run-dir glob -> iterations whose checkpoints are kept (all folds)
KEEP = {
    "runs/r3_total/l1": {2000},                       # R3 reported iteration
    "runs/r4_subset/l1": {1600},                      # R4 selected
    "runs/r4b_subset/bal/l1": {1600},                 # R4b best
    "runs/r4b_subset/bal_aug/l1": {3000},             # R4c selected
    "runs/cv_r3": {500, 1000, 1500, 2000},            # CV_R3 primary + GROUND learning curve (seed 101)
    "runs/cv_ablation/*": {2000},                     # CV_ABLATION primary
    "runs/mx_cv/*": {2000, 6000},                     # MX_CV primary + exploratory
    "runs/tq/A_k1": {500, 1000, 1500, 2000},          # TQ primary + GROUND learning curve (seed 101)
    "runs/tq/A_k3": {2000}, "runs/tq/D_k1": {2000}, "runs/tq/D_k3": {2000},
    "runs/gr/kr/k*_s*": {500, 1000, 1500, 2000, 3000, 4000},
    "runs/gr/kr/k*_f*": {1000, 2000},
    "runs/gr/sm/*": {2000, 3000},
}


def size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)


ap = argparse.ArgumentParser()
ap.add_argument("--execute", action="store_true")
EX = ap.parse_args().execute
victims, kept = [], 0
for p in DROP_DIRS + DROP_FILES:
    if os.path.exists(p):
        victims.append(p)
covered = set()
for pat, its in KEEP.items():
    for run in glob.glob(pat):
        for ck in glob.glob(os.path.join(run, "**", "*.pkl"), recursive=True):
            covered.add(os.path.normpath(ck))
            m = re.search(r"_(\d{6})\.pkl$", ck)
            if m and int(m.group(1)) in its:
                kept += 1
            else:                                     # other iterations and *_final.pkl duplicates
                victims.append(ck)
allck = {os.path.normpath(p) for p in glob.glob("runs/**/*.pkl", recursive=True)}
inside_drop = {p for p in allck if any(os.path.normpath(p).startswith(os.path.normpath(d)) for d in DROP_DIRS)}
uncovered = sorted(allck - covered - inside_drop)
assert not uncovered, ("checkpoints with no rule", uncovered[:5])
tot = sum(size(v) for v in victims)
print(f"{'EXECUTE' if EX else 'DRY RUN'}: {len(victims)} paths, {tot / 2**30:.2f} GiB; checkpoints kept: {kept}")
for d in DROP_DIRS + DROP_FILES:
    if d in victims:
        print(f"  folder/file {d}: {size(d) / 2**20:.0f} MiB")
by = {}
for v in victims:
    if v.endswith(".pkl"):
        k = re.sub(r"[\\/]fold\d+[\\/].*|[\\/]ckpt[\\/].*", "", v)
        by[k] = by.get(k, 0) + 1
for k, n in sorted(by.items()):
    print(f"  checkpoints removed from {k}: {n}")
if EX:
    for v in victims:
        shutil.rmtree(v) if os.path.isdir(v) else os.remove(v)
    print("done")
