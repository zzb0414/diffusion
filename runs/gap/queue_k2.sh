#!/usr/bin/env bash
# GAP K2 (Archive/GAP/PREREGISTRATION_K2.md): leakage-free re-scoring of the GAP K SUB / SUBM models. Evaluation only.
# Waits for GAP J2 (one GPU job at a time). Resumable.
set -u
while [ ! -f runs/gap/j2/queue.done ]; do sleep 60; done
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
R=runs/gap/k2; Q=$R/queue.log
mkdir -p $R/logs
run() { local t=$1; shift; echo "[$(date +%H:%M:%S)] START $t" >> $Q; "$@" || { echo "[$(date +%H:%M:%S)] FAILED $t" >> $Q; exit 1; }; }
for s in 101 102; do for arm in SUBM SUB; do for it in 1000 2000 3000; do
  o=$R/${arm}_s$s/eval_it$it
  [ -f $o/neff.csv ] || { rm -rf $o; run "eval $arm s$s it$it" $PY experiments/_wh_eval_clean.py --run runs/gap/k/${arm}_s$s --iter $it --datasets 6 --out $o > $R/logs/eval_${arm}_s${s}_it$it.txt 2>&1; }
done; done; done
run "report dry" $PY experiments/_wh_clean_report.py --dry >> $Q 2>&1
run "report" $PY experiments/_wh_clean_report.py > $R/logs/report.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch $R/queue.done
