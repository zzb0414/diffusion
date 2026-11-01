#!/usr/bin/env bash
# NAT-SUB (Archive/NAT/PREREGISTRATION_SUB.md): train SUB and SUBSUP on W05 and C05, then evaluate. Resumable.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }
for cond in W05 C05; do for m in sub subsup; do
  grep -qs DONE runs/nat/$cond/$m/train.log && continue
  rm -rf runs/nat/$cond/$m
  run "train $cond $m" $PY natural/train.py --method $m --cond $cond --outdir runs/nat/$cond/$m > runs/nat/logs/train_${cond}_$m.txt 2>&1
done; done
for cond in W05 C05; do for m in sub subsup; do
  [ -f runs/nat/eval/$cond/$m.csv ] || run "eval $cond $m" $PY natural/evaluate.py --cond $cond --methods $m > runs/nat/logs/eval_${cond}_$m.txt 2>&1
done; done
echo "[$(date +%H:%M:%S)] SUB_ALL_DONE"; touch runs/nat/sub.done
