#!/usr/bin/env bash
# RES low-noise repeat (Archive/RES/PREREGISTRATION.md, addendum): RES_SK=0.25, 5 arms x 3 seeds, train then evaluate,
# then the report and figures. Outputs runs/res_sk0.25/. Resumable.
set -u
export RES_SK=0.25
R=runs/res_sk0.25; mkdir -p $R/logs $R/eval
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }
for s in 101 102 103; do for a in A B C D E; do
  d=$R/${a}_s$s
  grep -qs DONE $d/train.log || { rm -rf $d; run "train $a s$s" $PY resolution/train.py --arm $a --seed $s --outdir $d > $R/logs/train_${a}_s$s.txt 2>&1; }
  [ -f $R/eval/${a}_s$s.csv ] || run "eval $a s$s" $PY resolution/evaluate.py --run $d --out $R/eval/${a}_s$s.csv > $R/logs/eval_${a}_s$s.txt 2>&1
done; done
run "report dry" $PY resolution/report.py --dry
run "report" $PY resolution/report.py > $R/logs/report.txt 2>&1
run "figures" $PY resolution/figures.py > $R/logs/figures.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE"; touch $R/queue.done
