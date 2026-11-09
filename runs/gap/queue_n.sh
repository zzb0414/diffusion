#!/usr/bin/env bash
# GAP-N (Archive/GAP/PREREGISTRATION_N.md): DT content D, one slice per volume at n_vol 8 / 32 / 62. One GPU job at a time.
# Resumable. Progress lines go to runs/gap/n/queue.log.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
R=runs/gap/n; Q=$R/queue.log
mkdir -p $R/logs $R/eval
run() { local t=$1; shift; echo "[$(date +%H:%M:%S)] START $t" >> $Q; "$@" || { echo "[$(date +%H:%M:%S)] FAILED $t" >> $Q; exit 1; }; }
for n in 8 32 62; do for s in 101 102; do for a in A SUB SUBM; do
  d=$R/${a}_DTo${n}_s$s
  grep -qs DONE $d/train.log || { rm -rf $d; run "train $a DTo$n s$s" $PY resolution/transplant_train.py --arm $a --noise T --content D --n-vol $n --one-slice --seed $s --outdir $d > $R/logs/train_${a}_DTo${n}_s$s.txt 2>&1; }
  for it in 1000 2000 3000; do
    o=$R/eval/${a}_DTo${n}_s${s}_it$it.csv
    [ -f $o ] || run "eval $a DTo$n s$s it$it" $PY resolution/transplant_eval.py --run $d --iter $it --out $o > $R/logs/eval_${a}_DTo${n}_s${s}_it$it.txt 2>&1
  done
done; done; done
run "report dry" $PY resolution/subjects_report.py --dry >> $Q 2>&1
run "report" $PY resolution/subjects_report.py > $R/logs/report.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch $R/queue.done
