#!/usr/bin/env bash
# Noise2Score baseline: waits for the DEC queue's last output (GPU free), trains 3 seeds x 5 folds, evaluates.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }

mkdir -p runs/base/logs
for s in 101 102 103; do for f in 0 1 2 3 4; do
  grep -qs DONE runs/base/n2s2/s$s/fold$f/train.log && continue
  run "N2S s$s fold$f" $PY baselines/noise2score.py --fold $f --seed $s --outdir runs/base/n2s2/s$s > runs/base/logs/n2s_s${s}_f$f.txt 2>&1
done; done
run "N2S eval" $PY baselines/evaluate.py --methods n2s --out runs/base/n2s2_eval > runs/base/logs/n2s2_eval.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE"
