#!/usr/bin/env bash
# NAT (Archive/NAT/PREREGISTRATION.md), GPU part: waits until (1) the BASE-DL queue has finished (GPU free) and
# (2) runs/nat/_ch_verified exists (library channel change verified identical on CPU); then trains every learned
# method x condition (3000 it) and evaluates them. Completed steps are skipped, so it can be re-run to resume.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }
until [ -f runs/base/dl_eval/neff.csv ] && [ -f runs/nat/_ch_verified ]; do sleep 60; done
for cond in W05 C05 W20 W50; do for m in ours avg9 avgu sup n2n ne2ne r2r n2s; do
  grep -qs DONE runs/nat/$cond/$m/train.log && continue
  rm -rf runs/nat/$cond/$m
  run "train $cond $m" $PY natural/train.py --method $m --cond $cond --outdir runs/nat/$cond/$m > runs/nat/logs/train_${cond}_$m.txt 2>&1
done; done
for cond in W05 C05 W20 W50; do for m in ours avg9 avgu sup n2n ne2ne r2r n2s; do
  [ -f runs/nat/eval/$cond/$m.csv ] || run "eval $cond $m" $PY natural/evaluate.py --cond $cond --methods $m > runs/nat/logs/eval_${cond}_$m.txt 2>&1
done; done
echo "[$(date +%H:%M:%S)] GPU_ALL_DONE"; touch runs/nat/gpu.done
