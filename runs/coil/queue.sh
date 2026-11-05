#!/usr/bin/env bash
# ASL-COIL (Archive/COIL/PREREGISTRATION.md). One GPU job at a time. Resumable. Progress lines go to runs/coil/queue.log.
# Part B first (2 trainings), then Part A (15 trainings).
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
R=runs/coil; Q=$R/queue.log; mkdir -p $R/logs
run() { local t=$1; shift; echo "[$(date +%H:%M:%S)] START $t" >> $Q; "$@" || { echo "[$(date +%H:%M:%S)] FAILED $t" >> $Q; exit 1; }; }
SUBM="--mode sub --ladder-div 3 --sub-uniform --m-dist linear"
for s in 101 102; do                                        # Part B: SUBM like ASL-SUBJ
  d=$R/subj/SUBM_s$s
  grep -qs DONE $d/foldall/train.log || { rm -rf $d/foldall; run "B train SUBM s$s" $PY experiments/_gr_train.py $SUBM --seed $s --outdir $d --fold -1 --datasets 1 5 --iters 3000 --loss mse > $R/logs/B_train_SUBM_s$s.txt 2>&1; }
  for it in 1000 2000 3000; do
    [ -f $d/eval_it$it/neff.csv ] || { rm -rf $d/eval_it$it; run "B eval SUBM s$s it$it" $PY experiments/_gr_eval.py --run $d --iter $it --mode sub --datasets 6 --single --out $d/eval_it$it > $R/logs/B_eval_SUBM_s${s}_it$it.txt 2>&1; }
  done
done
for arm in PER SUB SUBM; do                                 # Part A: dataset 1 only, seed 101
  d=$R/cv/${arm}_s101
  case $arm in
    PER)  args="--mode per"; mode=per ;;
    SUB)  args="--mode sub --ladder-div 3 --sub-uniform"; mode=sub ;;
    SUBM) args=$SUBM; mode=sub ;;
  esac
  for k in 0 1 2 3 4; do
    grep -qs DONE $d/fold$k/train.log || { rm -rf $d/fold$k; run "A train $arm fold$k" $PY experiments/_gr_train.py $args --seed 101 --outdir $d --fold $k --datasets 1 --iters 3000 --loss mse > $R/logs/A_train_${arm}_f$k.txt 2>&1; }
  done
  for it in 1000 2000 3000; do
    [ -f $d/eval_it$it/neff.csv ] || { rm -rf $d/eval_it$it; run "A eval $arm it$it" $PY experiments/_gr_eval.py --run $d --iter $it --mode $mode --datasets 1 --out $d/eval_it$it > $R/logs/A_eval_${arm}_it$it.txt 2>&1; }
  done
done
run "report dry" $PY experiments/_coil_report.py --dry >> $Q 2>&1
run "report" $PY experiments/_coil_report.py > $R/logs/report.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch $R/queue.done
