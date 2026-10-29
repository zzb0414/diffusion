#!/usr/bin/env bash
# GROUND evaluation: waits for queues A and B, then evaluates every run (refuses to overwrite).
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }
until [ -f runs/gr/queue_a.done ] && [ -f runs/gr/queue_b.done ]; do sleep 60; done
ev() { run "eval $1 it$2 $3" $PY experiments/_gr_eval.py --run $1 --iter $2 --mode $3 --datasets ${4:-1} --out $5 \
         > runs/gr/logs/eval_$(echo $5 | tr / _).txt 2>&1; }
# KR: seed 101 (existing runs, outputs under runs/gr), new seeds, fractions
for it in 500 1000 1500 2000; do
  ev runs/cv_r3 $it per 1 runs/gr/kr_eval/k9_s101/it$it
  ev runs/tq/A_k1 $it per 1 runs/gr/kr_eval/k1_s101/it$it
done
for s in 102 103; do for k in 1 9; do for it in 500 1000 1500 2000 3000 4000; do
  ev runs/gr/kr/k${k}_s$s $it per 1 runs/gr/kr_eval/k${k}_s$s/it$it
done; done; done
for fr in 0.5 0.25; do for k in 1 9; do for it in 1000 2000; do
  ev runs/gr/kr/k${k}_f$fr $it per 1 runs/gr/kr_eval/k${k}_f$fr/it$it
done; done; done
# SM
for it in 2000 3000; do
  ev runs/gr/sm/per_ds1 $it per 1 runs/gr/sm_eval/per_ds1/it$it
  ev runs/gr/sm/sub_ds1 $it sub 1 runs/gr/sm_eval/sub_ds1/it$it
  ev runs/gr/sm/per_pool $it per "1 2 3" runs/gr/sm_eval/per_pool/it$it
  ev runs/gr/sm/sub_pool $it sub "1 2 3" runs/gr/sm_eval/sub_pool/it$it
done
ev runs/mx_cv/A 2000 per "1 2 3" runs/gr/sm_eval/per_pool_noaug/it2000
echo "[$(date +%H:%M:%S)] ALL_DONE"
