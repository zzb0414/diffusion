#!/usr/bin/env bash
# ASL-COIL follow-up (exploratory, eval only): the dataset-1-only CV fold models (Part A) and ASL-SM2's pooled seed-101
# models score the unseen subject 1104c (data id 6), each slice by the fold model that held its index out.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"; Q=runs/coil/xsubj/queue.log
for spec in "runs/coil/cv/PER_s101 per d1_PER" "runs/coil/cv/SUB_s101 sub d1_SUB" "runs/coil/cv/SUBM_s101 sub d1_SUBM" "runs/sm2/PER_s101 per pool_PER" "runs/sm2/SUB_s101 sub pool_SUB"; do
  set -- $spec
  for it in 1000 2000 3000; do
    o=runs/coil/xsubj/$3/eval_it$it
    [ -f $o/neff.csv ] && continue
    rm -rf $o; echo "[$(date +%H:%M:%S)] START $3 it$it" >> $Q
    $PY experiments/_gr_eval.py --run $1 --iter $it --mode $2 --datasets 6 --out $o > runs/coil/xsubj/logs/$3_it$it.txt 2>&1 || { echo FAILED >> $Q; exit 1; }
  done
done
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q
