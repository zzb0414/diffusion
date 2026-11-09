#!/usr/bin/env bash
# GAP-M (Archive/GAP/PREREGISTRATION_M.md): DL, DS, DLw; arms A, SUB, SUBM; seeds 101-102. DW is GAP-D's (read only).
# One GPU job at a time. Resumable. Progress lines go to runs/gap/m/queue.log.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
R=runs/gap/m; Q=$R/queue.log
mkdir -p $R/logs $R/eval
run() { local t=$1; shift; echo "[$(date +%H:%M:%S)] START $t" >> $Q; "$@" || { echo "[$(date +%H:%M:%S)] FAILED $t" >> $Q; exit 1; }; }
for cn in DL DS DLw; do n=${cn:1:1}; w=""; [ $cn = DLw ] && w="--whiten"; for s in 101 102; do for a in A SUB SUBM; do
  d=$R/${a}_${cn}_s$s
  grep -qs DONE $d/train.log || { rm -rf $d; run "train $a $cn s$s" $PY resolution/transplant_train.py --arm $a --noise $n --content D $w --seed $s --outdir $d > $R/logs/train_${a}_${cn}_s$s.txt 2>&1; }
  for it in 1000 2000 3000; do
    o=$R/eval/${a}_${cn}_s${s}_it$it.csv
    [ -f $o ] || run "eval $a $cn s$s it$it" $PY resolution/transplant_eval.py --run $d --iter $it --out $o > $R/logs/eval_${a}_${cn}_s${s}_it$it.txt 2>&1
  done
done; done; done
run "report dry" $PY resolution/coloc_report.py --dry >> $Q 2>&1
run "report" $PY resolution/coloc_report.py > $R/logs/report.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch $R/queue.done
