#!/usr/bin/env bash
# GAP H (Archive/GAP/PREREGISTRATION_H.md): ASL training-set dose-response (--train-frac 0.25, 0.5). Waits for GAP G. Resumable.
set -u
while [ ! -f runs/gap/g/queue.done ]; do sleep 60; done
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
R=runs/gap/h; Q=$R/queue.log
mkdir -p $R/logs
run() { local t=$1; shift; echo "[$(date +%H:%M:%S)] START $t" >> $Q; "$@" || { echo "[$(date +%H:%M:%S)] FAILED $t" >> $Q; exit 1; }; }
COMMON="--fold -1 --datasets 1 5 --iters 3000 --loss mse"
for fr in 0.25 0.5; do for s in 101 102; do for arm in SUBM SUB PER; do
  d=$R/${arm}_f${fr}_s$s
  case $arm in
    PER)  mode=per; extra="" ;;
    SUB)  mode=sub; extra="--ladder-div 3 --sub-uniform" ;;
    SUBM) mode=sub; extra="--ladder-div 3 --sub-uniform --m-dist linear" ;;
  esac
  grep -qs DONE $d/foldall/train.log || { rm -rf $d/foldall; run "train $arm f$fr s$s" $PY experiments/_gr_train.py --mode $mode --seed $s --outdir $d --train-frac $fr $COMMON $extra > $R/logs/train_${arm}_f${fr}_s$s.txt 2>&1; }
  for it in 1000 2000 3000; do
    [ -f $d/eval_it$it/neff.csv ] || { rm -rf $d/eval_it$it; run "eval $arm f$fr s$s it$it" $PY experiments/_gr_eval.py --run $d --iter $it --mode $mode --datasets 6 --single --out $d/eval_it$it > $R/logs/eval_${arm}_f${fr}_s${s}_it$it.txt 2>&1; }
  done
done; done; done
run "report dry" $PY experiments/_h_report.py --dry >> $Q 2>&1
run "report" $PY experiments/_h_report.py > $R/logs/report.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch $R/queue.done
