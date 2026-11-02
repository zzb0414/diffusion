#!/usr/bin/env bash
# CLAIMS follow-up (exploratory, not pre-registered): does the DSM-vs-direct bias/variance gap close with training
# (as in the linear toy) or persist? DEC checkpoints at 500/1000/1500 iterations (2000 already evaluated).
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }
until [ -f runs/claims/queue.done ]; do sleep 60; done
for it in 500 1000 1500; do for arm in dsm dirl; do for s in 102 103; do      # DSM seed 101 checkpoints <2000 were pruned
  run_dir=runs/dec/A/${arm}_s$s
  out=runs/claims/c2_lc/${arm}_s$s/it$it
  [ -f $out/neff.csv ] || run "C2-LC $arm s$s it$it" $PY experiments/_dec_eval.py --run $run_dir --iter $it --loss mse --datasets 1 --out $out > runs/claims/logs/c2lc_${arm}_s${s}_$it.txt 2>&1
done; done; done
echo "[$(date +%H:%M:%S)] ALL_DONE"; touch runs/claims/queue2.done
