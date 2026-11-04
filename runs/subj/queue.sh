#!/usr/bin/env bash
# ASL-SUBJ (Archive/SUBJ/PREREGISTRATION.md). Waits for ASL-SM2 (one GPU job at a time). Resumable.
# Progress lines go to runs/subj/queue.log.
set -u
while [ ! -f runs/sm2/queue.done ]; do sleep 60; done
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
R=runs/subj; Q=$R/queue.log; mkdir -p $R/logs
run() { local t=$1; shift; echo "[$(date +%H:%M:%S)] START $t" >> $Q; "$@" || { echo "[$(date +%H:%M:%S)] FAILED $t" >> $Q; exit 1; }; }
COMMON="--fold -1 --datasets 1 5 --iters 3000 --loss mse"
for spec in "PER 101" "SUB 101" "PERD 101" "PER 102" "SUB 102"; do
  set -- $spec; arm=$1; s=$2; d=$R/${arm}_s$s
  case $arm in
    PER)  mode=per; extra="" ;;
    PERD) mode=per; extra="--noise-norm" ;;
    SUB)  mode=sub; extra="--ladder-div 3 --sub-uniform" ;;
  esac
  grep -qs DONE $d/foldall/train.log || { rm -rf $d/foldall; run "train $arm s$s" $PY experiments/_gr_train.py --mode $mode --seed $s --outdir $d $COMMON $extra > $R/logs/train_${arm}_s$s.txt 2>&1; }
  for it in 1000 2000 3000; do
    [ -f $d/eval_it$it/neff.csv ] || { rm -rf $d/eval_it$it; run "eval $arm s$s it$it" $PY experiments/_gr_eval.py --run $d --iter $it --mode $mode --datasets 6 --single --out $d/eval_it$it > $R/logs/eval_${arm}_s${s}_it$it.txt 2>&1; }
  done
done
run "report dry" $PY experiments/_subj_report.py --dry >> $Q 2>&1
run "report" $PY experiments/_subj_report.py > $R/logs/report.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch $R/queue.done
