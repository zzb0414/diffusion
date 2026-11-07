#!/usr/bin/env bash
# GAP J2 (EXPLORATORY, after the GAP J reading): checkpoint trajectory. Scores the extra saved checkpoints (500, 1500) of the
# GAP J FT / SC models and the original-units references on 1104c. Evaluation only. Waits for GAP K. Resumable.
set -u
while [ ! -f runs/gap/k/queue.done ]; do sleep 60; done
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
R=runs/gap/j2; Q=$R/queue.log
mkdir -p $R/logs
run() { local t=$1; shift; echo "[$(date +%H:%M:%S)] START $t" >> $Q; "$@" || { echo "[$(date +%H:%M:%S)] FAILED $t" >> $Q; exit 1; }; }
for s in 101 102; do for arm in SUBM SUB PER; do
  mode=sub; [ $arm = PER ] && mode=per
  ref=runs/subj/${arm}_s$s; [ $arm = SUBM ] && ref=runs/coil/subj/SUBM_s$s
  for it in 500 1500; do
    for init in FT SC; do
      o=$R/${init}_${arm}_s$s/eval_it$it
      [ -f $o/neff.csv ] || { rm -rf $o; run "eval $init $arm s$s it$it" $PY experiments/_ft_eval.py --run runs/gap/j/${init}_${arm}_s$s --iter $it --mode $mode --datasets 6 --single --out $o > $R/logs/eval_${init}_${arm}_s${s}_it$it.txt 2>&1; }
    done
    o=$R/ref_${arm}_s$s/eval_it$it
    [ -f $o/neff.csv ] || { rm -rf $o; run "eval ref $arm s$s it$it" $PY experiments/_gr_eval.py --run $ref --iter $it --mode $mode --datasets 6 --single --out $o > $R/logs/eval_ref_${arm}_s${s}_it$it.txt 2>&1; }
  done
done; done
run "trajectory" $PY experiments/_j2_traj.py > $R/logs/report.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch $R/queue.done
