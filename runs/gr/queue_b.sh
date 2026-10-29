#!/usr/bin/env bash
# GROUND queue B: KR-frac (k in {1,9} x frac {0.5,0.25}, 2000 it), then SM PER-aug ds1, SUB ds1, SUB pooled.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }
for fr in 0.5 0.25; do for k in 1 9; do for f in 0 1 2 3 4; do
  TK=""; [ $k = 1 ] && TK="--target-k 1"
  run "KR k$k frac$fr fold$f" $PY experiments/_gr_train.py --mode per --fold $f --iters 2000 --train-frac $fr $TK \
      --outdir runs/gr/kr/k${k}_f$fr > runs/gr/logs/kr_k${k}_f${fr}_f$f.txt 2>&1
done; done; done
for spec in "per_ds1:per:1" "sub_ds1:sub:1" "sub_pool:sub:1 2 3"; do
  IFS=: read name mode ds <<< "$spec"
  for f in 0 1 2 3 4; do
    run "SM $name fold$f" $PY experiments/_gr_train.py --mode $mode --aug --ladder-div 3 --datasets $ds --fold $f \
        --iters 3000 --outdir runs/gr/sm/$name > runs/gr/logs/sm_${name}_f$f.txt 2>&1
  done
done
echo "[$(date +%H:%M:%S)] QUEUE_B_DONE"; touch runs/gr/queue_b.done
