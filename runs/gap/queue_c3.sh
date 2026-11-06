#!/usr/bin/env bash
# GAP C3 (amendment in WEEKEND_ITERATIONS GAP C reading): U80 with crop 128, batch 2. One GPU job at a time. Resumable.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
R=runs/gap/c; Q=$R/queue_c3.log
run() { local t=$1; shift; echo "[$(date +%H:%M:%S)] START $t" >> $Q; "$@" || { echo "[$(date +%H:%M:%S)] FAILED $t" >> $Q; exit 1; }; }
for s in 101 102; do for a in A SUB; do
  d=$R/${a}_U80c128_s$s
  grep -qs DONE $d/train.log || { rm -rf $d; run "train $a s$s" $PY resolution/transplant_train.py --arm $a --noise W --grid 160 --crop 128 --batch 2 --seed $s --outdir $d > $R/logs/train_${a}_U80c128_s$s.txt 2>&1; }
  for it in 1000 2000 3000; do
    o=$R/eval/${a}_U80c128_s${s}_it$it.csv
    [ -f $o ] || run "eval $a s$s it$it" $PY resolution/transplant_eval.py --run $d --iter $it --out $o > $R/logs/eval_${a}_U80c128_s${s}_it$it.txt 2>&1
  done
done; done
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch $R/queue_c3.done
