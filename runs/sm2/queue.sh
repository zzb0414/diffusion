#!/usr/bin/env bash
# ASL-SM2 (Archive/SM2/PREREGISTRATION.md). Waits for MEAN-SNR (one GPU job at a time). Resumable.
# Progress lines go to runs/sm2/queue.log.
set -u
while [ ! -f runs/mean_snr/queue.done ]; do sleep 60; done
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
R=runs/sm2; Q=$R/queue.log; mkdir -p $R/logs
run() { local t=$1; shift; echo "[$(date +%H:%M:%S)] START $t" >> $Q; "$@" || { echo "[$(date +%H:%M:%S)] FAILED $t" >> $Q; exit 1; }; }
COMMON="--datasets 1 2 3 --iters 3000 --loss mse"
for spec in "PER 101" "SUB 101" "PERD 101" "PER 102" "SUB 102"; do
  set -- $spec; arm=$1; s=$2; d=$R/${arm}_s$s
  case $arm in
    PER)  mode=per; extra="" ;;
    PERD) mode=per; extra="--noise-norm" ;;
    SUB)  mode=sub; extra="--ladder-div 3 --sub-uniform" ;;
  esac
  for k in 0 1 2 3 4; do
    grep -qs DONE $d/fold$k/train.log || { rm -rf $d/fold$k; run "train $arm s$s fold$k" $PY experiments/_gr_train.py --mode $mode --fold $k --seed $s --outdir $d $COMMON $extra > $R/logs/train_${arm}_s${s}_f$k.txt 2>&1; }
  done
  for it in 1000 2000 3000; do
    [ -f $d/eval_it$it/neff.csv ] || { rm -rf $d/eval_it$it; run "eval $arm s$s it$it" $PY experiments/_gr_eval.py --run $d --iter $it --mode $mode --datasets 1 2 3 --out $d/eval_it$it > $R/logs/eval_${arm}_s${s}_it$it.txt 2>&1; }
  done
done
run "report dry" $PY experiments/_sm2_report.py --dry >> $Q 2>&1
run "report" $PY experiments/_sm2_report.py > $R/logs/report.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch $R/queue.done
