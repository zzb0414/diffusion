#!/usr/bin/env bash
# NAT, CPU part: plain averaging and BM3D on Set12 + BSD68 for every condition (numpy-only pool, 6 workers).
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }
for cond in W05 C05 W20 W50; do for m in avg bm3d; do
  [ -f runs/nat/eval/$cond/$m.csv ] || run "eval $cond $m" $PY natural/evaluate.py --cond $cond --methods $m --workers 6 > runs/nat/logs/eval_${cond}_$m.txt 2>&1
done; done
echo "[$(date +%H:%M:%S)] CPU_ALL_DONE"; touch runs/nat/cpu.done
