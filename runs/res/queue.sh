#!/usr/bin/env bash
# RES (Archive/RES/PREREGISTRATION.md): 5 arms x 3 seeds, train then evaluate. Resumable.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }
for s in 101 102 103; do for a in A B C D E; do
  d=runs/res/${a}_s$s
  grep -qs DONE $d/train.log || { rm -rf $d; run "train $a s$s" $PY resolution/train.py --arm $a --seed $s --outdir $d > runs/res/logs/train_${a}_s$s.txt 2>&1; }
  [ -f runs/res/eval/${a}_s$s.csv ] || run "eval $a s$s" $PY resolution/evaluate.py --run $d --out runs/res/eval/${a}_s$s.csv > runs/res/logs/eval_${a}_s$s.txt 2>&1
done; done
echo "[$(date +%H:%M:%S)] ALL_DONE"; touch runs/res/queue.done
