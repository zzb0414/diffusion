#!/usr/bin/env bash
# MEAN (Archive/MEAN/PREREGISTRATION.md). Waits for the RES low-noise queue (one GPU job at a time). Resumable.
set -u
while [ ! -f runs/res_sk0.25/queue.done ]; do sleep 60; done
unset RES_SK
R=runs/mean; mkdir -p $R/logs $R/eval
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }
for s in 101 102; do for n in 2 8 62; do for a in A SUB; do
  if [ $a = A ] && [ $n = 62 ]; then d=runs/res/A_s$s; else
    d=$R/${a}_v${n}_s$s
    grep -qs DONE $d/train.log || { rm -rf $d; run "train $a v$n s$s" $PY resolution/train.py --arm $a --seed $s --n-vol $n --outdir $d > $R/logs/train_${a}_v${n}_s$s.txt 2>&1; }
  fi
  for it in 1000 2000 3000; do
    o=$R/eval/${a}_v${n}_s${s}_it$it.csv
    [ -f $o ] || run "eval $a v$n s$s it$it" $PY resolution/mean_eval.py --run $d --iter $it --out $o > $R/logs/eval_${a}_v${n}_s${s}_it$it.txt 2>&1
  done
done; done; done
run "report dry" $PY resolution/mean_report.py --dry
run "report" $PY resolution/mean_report.py > $R/logs/report.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE"; touch $R/queue.done
