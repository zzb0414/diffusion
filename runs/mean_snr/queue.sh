#!/usr/bin/env bash
# MEAN-SNR (Archive/MEAN/PREREGISTRATION_SNR.md): PER (arm A) vs SUB at S_K = 2, 3, 6, 62 volumes, seeds 101-102.
# One GPU job at a time. Resumable. Progress lines go to runs/mean_snr/queue.log.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
Q=runs/mean_snr/queue.log
run() { local t=$1; shift; echo "[$(date +%H:%M:%S)] START $t" >> $Q; "$@" || { echo "[$(date +%H:%M:%S)] FAILED $t" >> $Q; exit 1; }; }
for sk in 2 3 6; do
  export RES_SK=$sk
  R=runs/mean_sk$sk; mkdir -p $R/logs $R/eval
  for s in 101 102; do for a in A SUB; do
    d=$R/${a}_v62_s$s
    grep -qs DONE $d/train.log || { rm -rf $d; run "sk$sk train $a s$s" $PY resolution/train.py --arm $a --seed $s --n-vol 62 --outdir $d > $R/logs/train_${a}_s$s.txt 2>&1; }
    for it in 1000 2000 3000; do
      o=$R/eval/${a}_v62_s${s}_it$it.csv
      [ -f $o ] || run "sk$sk eval $a s$s it$it" $PY resolution/mean_eval.py --run $d --iter $it --out $o > $R/logs/eval_${a}_s${s}_it$it.txt 2>&1
    done
  done; done
  run "sk$sk report dry" $PY resolution/mean_report.py --dry >> $Q 2>&1
  run "sk$sk report" $PY resolution/mean_report.py > $R/logs/report.txt 2>&1
done
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch runs/mean_snr/queue.done
