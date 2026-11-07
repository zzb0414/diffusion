#!/usr/bin/env bash
# GAP F (Archive/GAP/PREREGISTRATION_F.md). F1: T and DT models on ASL 1104c (eval only). F2: DT at n_vol 1 and 4.
# One GPU job at a time. Resumable. Progress lines go to runs/gap/f/queue.log.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
R=runs/gap/f; Q=$R/queue.log
mkdir -p $R/logs $R/eval $R/asl
run() { local t=$1; shift; echo "[$(date +%H:%M:%S)] START $t" >> $Q; "$@" || { echo "[$(date +%H:%M:%S)] FAILED $t" >> $Q; exit 1; }; }
for c in DT T; do src=runs/gap/d; [ $c = T ] && src=runs/gap/b; for s in 101 102; do for a in A SUB SUBM; do for it in 1000 2000 3000; do
  o=$R/asl/${a}_${c}_s$s/eval_it$it
  [ -f $o/neff.csv ] || run "asl eval $a $c s$s it$it" $PY experiments/_tr_eval.py --run $src/${a}_${c}_s$s --iter $it --datasets 6 --out $o > $R/logs/asl_${a}_${c}_s${s}_it$it.txt 2>&1
done; done; done; done
for n in 1 4; do for s in 101 102; do for a in A SUB SUBM; do
  d=$R/${a}_DTn${n}_s$s
  grep -qs DONE $d/train.log || { rm -rf $d; run "train $a DTn$n s$s" $PY resolution/transplant_train.py --arm $a --noise T --content D --n-vol $n --seed $s --outdir $d > $R/logs/train_${a}_DTn${n}_s$s.txt 2>&1; }
  for it in 1000 2000 3000; do
    o=$R/eval/${a}_DTn${n}_s${s}_it$it.csv
    [ -f $o ] || run "eval $a DTn$n s$s it$it" $PY resolution/transplant_eval.py --run $d --iter $it --out $o > $R/logs/eval_${a}_DTn${n}_s${s}_it$it.txt 2>&1
  done
done; done; done
run "report dry" $PY resolution/gap_f_report.py --dry >> $Q 2>&1
run "report" $PY resolution/gap_f_report.py > $R/logs/report.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch $R/queue.done
