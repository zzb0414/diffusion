#!/usr/bin/env bash
# BASE-DDM2 (Archive/BASE/PREREGISTRATION_DDM2.md): waits for the CLAIMS queues (GPU free), trains stage III
# (3 seeds x 5 folds), evaluates DDM2 and DDM2-TW on the baseline splits. Resumable.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }
until [ -f runs/claims/queue.done ] && [ -f runs/claims/queue2.done ]; do sleep 60; done
mkdir -p runs/base/logs
for s in 101 102 103; do for f in 0 1 2 3 4; do
  grep -qs DONE runs/base/ddm2/s$s/fold$f/train.log && continue
  rm -rf runs/base/ddm2/s$s/fold$f
  run "DDM2 stage III s$s f$f" $PY baselines/ddm2.py --fold $f --seed $s --outdir runs/base/ddm2/s$s > runs/base/logs/ddm2_s${s}_f$f.txt 2>&1
done; done
[ -f runs/base/ddm2_eval/neff.csv ] || run "DDM2 eval" $PY baselines/evaluate.py --methods ddm2 ddm2_tw --device cuda --out runs/base/ddm2_eval > runs/base/logs/ddm2_eval.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE"; touch runs/base/ddm2.done
