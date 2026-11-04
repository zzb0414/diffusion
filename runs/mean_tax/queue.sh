#!/usr/bin/env bash
# MEAN-TAX (Archive/MEAN/PREREGISTRATION_TAX.md): SUBW, SUBM at S_K = 6 then 2, 62 volumes, seeds 101-102.
# Waits for ASL-SUBJ, which waits for ASL-SM2 (one GPU job at a time). Resumable. Progress lines go to runs/mean_tax/queue.log.
set -u
while [ ! -f runs/subj/queue.done ]; do sleep 60; done
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
Q=runs/mean_tax/queue.log
run() { local t=$1; shift; echo "[$(date +%H:%M:%S)] START $t" >> $Q; "$@" || { echo "[$(date +%H:%M:%S)] FAILED $t" >> $Q; exit 1; }; }
for sk in 6 2; do
  export RES_SK=$sk
  R=runs/mean_sk$sk
  for s in 101 102; do for a in SUBW SUBM; do
    d=$R/${a}_v62_s$s
    grep -qs DONE $d/train.log || { rm -rf $d; run "sk$sk train $a s$s" $PY resolution/train.py --arm $a --seed $s --n-vol 62 --outdir $d > $R/logs/train_${a}_s$s.txt 2>&1; }
    for it in 1000 2000 3000; do
      o=$R/eval/${a}_v62_s${s}_it$it.csv
      [ -f $o ] || run "sk$sk eval $a s$s it$it" $PY resolution/mean_eval.py --run $d --iter $it --out $o > $R/logs/eval_${a}_s${s}_it$it.txt 2>&1
    done
  done; done
  run "sk$sk report dry" $PY resolution/tax_report.py --dry >> $Q 2>&1
  run "sk$sk report" $PY resolution/tax_report.py > $R/logs/report_tax.txt 2>&1
done
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch runs/mean_tax/queue.done
