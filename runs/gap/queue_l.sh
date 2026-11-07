#!/usr/bin/env bash
# GAP L (Archive/GAP/PREREGISTRATION_L.md): SC vs reference recipe, both early-stopped (6 checkpoints cross-fitted), fresh seeds.
# One GPU job at a time (nothing else queued). Resumable.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
R=runs/gap/l; Q=$R/queue.log
mkdir -p $R/logs
run() { local t=$1; shift; echo "[$(date +%H:%M:%S)] START $t" >> $Q; "$@" || { echo "[$(date +%H:%M:%S)] FAILED $t" >> $Q; exit 1; }; }
COMMON="--fold -1 --datasets 1 5 --iters 3000 --loss mse"
for s in 103 104; do for arm in SUBM SUB; do for rec in SC ref; do
  d=$R/${rec}_${arm}_s$s
  extra="--ladder-div 3 --sub-uniform"; [ $arm = SUBM ] && extra="$extra --m-dist linear"
  if [ $rec = SC ]; then tr=experiments/_ft_train.py; ev=experiments/_ft_eval.py; mk=ft.json
  else tr=experiments/_gr_train.py; ev=experiments/_gr_eval.py; mk=config.json; fi
  { grep -qs DONE $d/foldall/train.log && [ -f $d/foldall/$mk ]; } || { rm -rf $d/foldall; run "train $rec $arm s$s" $PY $tr --mode sub --seed $s --outdir $d $COMMON $extra > $R/logs/train_${rec}_${arm}_s$s.txt 2>&1; }
  for it in 500 1000 1500 2000 2500 3000; do
    [ -f $d/eval_it$it/neff.csv ] || { rm -rf $d/eval_it$it; run "eval $rec $arm s$s it$it" $PY $ev --run $d --iter $it --mode sub --datasets 6 --single --out $d/eval_it$it > $R/logs/eval_${rec}_${arm}_s${s}_it$it.txt 2>&1; }
  done
done; done; done
run "report dry" $PY experiments/_l_report.py --dry >> $Q 2>&1
run "report" $PY experiments/_l_report.py > $R/logs/report.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch $R/queue.done
