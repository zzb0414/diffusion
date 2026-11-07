#!/usr/bin/env bash
# GAP J (Archive/GAP/PREREGISTRATION_J.md): fastMRI-initialized (FT) vs random-init (SC) ASL models, unit-noise units.
# Waits for GAP I (one GPU job at a time). Resumable.
set -u
while [ ! -f runs/gap/i/queue.done ]; do sleep 60; done
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
R=runs/gap/j; Q=$R/queue.log
mkdir -p $R/logs
run() { local t=$1; shift; echo "[$(date +%H:%M:%S)] START $t" >> $Q; "$@" || { echo "[$(date +%H:%M:%S)] FAILED $t" >> $Q; exit 1; }; }
COMMON="--fold -1 --datasets 1 5 --iters 3000 --loss mse"
for s in 101 102; do for init in FT SC; do for arm in SUBM SUB PER; do
  d=$R/${init}_${arm}_s$s
  case $arm in
    PER)  mode=per; extra=""; src=A ;;
    SUB)  mode=sub; extra="--ladder-div 3 --sub-uniform"; src=SUB ;;
    SUBM) mode=sub; extra="--ladder-div 3 --sub-uniform --m-dist linear"; src=SUBM ;;
  esac
  ini=""; [ $init = FT ] && ini="--init runs/gap/d/${src}_DT_s$s/ckpt/res_003000.pkl"
  { grep -qs DONE $d/foldall/train.log && [ -f $d/foldall/ft.json ]; } || { rm -rf $d/foldall; run "train $init $arm s$s" $PY experiments/_ft_train.py $ini --mode $mode --seed $s --outdir $d $COMMON $extra > $R/logs/train_${init}_${arm}_s$s.txt 2>&1; }
  for it in 1000 2000 3000; do
    [ -f $d/eval_it$it/neff.csv ] || { rm -rf $d/eval_it$it; run "eval $init $arm s$s it$it" $PY experiments/_ft_eval.py --run $d --iter $it --mode $mode --datasets 6 --single --out $d/eval_it$it > $R/logs/eval_${init}_${arm}_s${s}_it$it.txt 2>&1; }
  done
done; done; done
run "report dry" $PY experiments/_ft_report.py --dry >> $Q 2>&1
run "report" $PY experiments/_ft_report.py > $R/logs/report.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch $R/queue.done
