#!/usr/bin/env bash
# GROUND queue A: KR-seed (k in {1,9} x seeds {102,103}, 4000 it), then SM PER-aug pooled. Stops at first failure.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }
for s in 102 103; do for k in 1 9; do for f in 0 1 2 3 4; do
  TK=""; [ $k = 1 ] && TK="--target-k 1"
  run "KR k$k s$s fold$f" $PY experiments/_gr_train.py --mode per --fold $f --iters 4000 --seed $s $TK \
      --outdir runs/gr/kr/k${k}_s$s > runs/gr/logs/kr_k${k}_s${s}_f$f.txt 2>&1
done; done; done
for f in 0 1 2 3 4; do
  run "SM per_pool fold$f" $PY experiments/_gr_train.py --mode per --aug --ladder-div 3 --datasets 1 2 3 --fold $f \
      --iters 3000 --outdir runs/gr/sm/per_pool > runs/gr/logs/sm_per_pool_f$f.txt 2>&1
done
echo "[$(date +%H:%M:%S)] QUEUE_A_DONE"; touch runs/gr/queue_a.done
