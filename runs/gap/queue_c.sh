#!/usr/bin/env bash
# GAP C (Archive/GAP/PREREGISTRATION_C.md): U80, 80c32, 160c128 x A, SUB x seeds 101-102. One GPU job at a time. Resumable.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
R=runs/gap/c; Q=$R/queue.log
run() { local t=$1; shift; echo "[$(date +%H:%M:%S)] START $t" >> $Q; "$@" || { echo "[$(date +%H:%M:%S)] FAILED $t" >> $Q; exit 1; }; }
for c in U80 80c32 160c128; do for s in 101 102; do for a in A SUB; do
  d=$R/${a}_${c}_s$s
  case $c in
    U80)     TR="resolution/transplant_train.py --arm $a --noise W --grid 160"; EV=resolution/transplant_eval.py; export RES_SK=1;;
    80c32)   TR="resolution/transplant_train.py --arm $a --noise W --crop 32 --batch 32"; EV=resolution/transplant_eval.py; export RES_SK=1;;
    160c128) TR="resolution/train.py --arm $a --n-vol 62 --crop 128 --batch 2"; EV=resolution/mean_eval.py; export RES_SK=2;;
  esac
  grep -qs DONE $d/train.log || { rm -rf $d; run "train $a $c s$s" $PY $TR --seed $s --outdir $d > $R/logs/train_${a}_${c}_s$s.txt 2>&1; }
  for it in 1000 2000 3000; do
    o=$R/eval/${a}_${c}_s${s}_it$it.csv
    [ -f $o ] || run "eval $a $c s$s it$it" $PY $EV --run $d --iter $it --out $o > $R/logs/eval_${a}_${c}_s${s}_it$it.txt 2>&1
  done
done; done; done
unset RES_SK
run "report dry" $PY resolution/matrix_report.py --dry >> $Q 2>&1
run "report" $PY resolution/matrix_report.py > $R/logs/report.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch $R/queue.done
