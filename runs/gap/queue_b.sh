#!/usr/bin/env bash
# GAP B (Archive/GAP/PREREGISTRATION_B.md): noise transplant, W (control) then T then G; arms A, SUB, SUBM; seeds 101-102.
# One GPU job at a time. Resumable. Progress lines go to runs/gap/b/queue.log.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
R=runs/gap/b; Q=$R/queue.log
run() { local t=$1; shift; echo "[$(date +%H:%M:%S)] START $t" >> $Q; "$@" || { echo "[$(date +%H:%M:%S)] FAILED $t" >> $Q; exit 1; }; }
for n in W T G; do for s in 101 102; do for a in A SUB SUBM; do
  d=$R/${a}_${n}_s$s
  grep -qs DONE $d/train.log || { rm -rf $d; run "train $a $n s$s" $PY resolution/transplant_train.py --arm $a --noise $n --seed $s --outdir $d > $R/logs/train_${a}_${n}_s$s.txt 2>&1; }
  for it in 1000 2000 3000; do
    o=$R/eval/${a}_${n}_s${s}_it$it.csv
    [ -f $o ] || run "eval $a $n s$s it$it" $PY resolution/transplant_eval.py --run $d --iter $it --out $o > $R/logs/eval_${a}_${n}_s${s}_it$it.txt 2>&1
  done
done; done; done
run "report dry" $PY resolution/transplant_report.py --dry >> $Q 2>&1
run "report" $PY resolution/transplant_report.py > $R/logs/report.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch $R/queue.done
