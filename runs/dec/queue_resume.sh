#!/usr/bin/env bash
# DEC queue, resume after the 18:19 transient I/O failure: identical steps, completed ones skipped.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }
done_train() { grep -qs DONE "$1/train.log"; }
B="--decoders tw ode anc anc8 ancx --datasets 1 2 3 --loss mse --iter 2000"
[ -f runs/dec/B/s101/neff.csv ] || run "B eval DSM s101" $PY experiments/_dec_eval.py --run runs/cv_ablation/mse_norm $B --out runs/dec/B/s101 > runs/dec/logs/B_s101.txt 2>&1
declare -A ARGS=( [dsm]="" [dirl]="--param direct" [dirn]="--param direct --p-raw 1" [blind]="--param direct --p-raw 1 --blind" )
for arm in dsm dirl dirn blind; do for s in 101 102 103; do
  [ $arm = dsm ] && [ $s = 101 ] && continue
  for f in 0 1 2 3 4; do
    done_train runs/dec/A/${arm}_s$s/fold$f && continue
    run "A train $arm s$s fold$f" $PY experiments/_gr_train.py --mode per --fold $f --iters 2000 --seed $s \
        --loss mse --noise-norm ${ARGS[$arm]} --outdir runs/dec/A/${arm}_s$s > runs/dec/logs/train_${arm}_s${s}_f$f.txt 2>&1
  done
done; done
for s in 102 103; do
  [ -f runs/dec/B/s$s/neff.csv ] || run "B eval DSM s$s" $PY experiments/_dec_eval.py --run runs/dec/A/dsm_s$s $B --out runs/dec/B/s$s > runs/dec/logs/B_s$s.txt 2>&1
done
[ -f runs/dec/A/eval/dsm_s101/neff.csv ] || run "A eval dsm s101" $PY experiments/_dec_eval.py --run runs/cv_ablation/mse_norm --loss mse --datasets 1 2 3 --out runs/dec/A/eval/dsm_s101 > runs/dec/logs/A_eval_dsm_s101.txt 2>&1
for arm in dsm dirl dirn blind; do for s in 101 102 103; do
  [ $arm = dsm ] && [ $s = 101 ] && continue
  [ -f runs/dec/A/eval/${arm}_s$s/neff.csv ] || run "A eval $arm s$s" $PY experiments/_dec_eval.py --run runs/dec/A/${arm}_s$s --loss mse \
      --datasets 1 2 3 --out runs/dec/A/eval/${arm}_s$s > runs/dec/logs/A_eval_${arm}_s$s.txt 2>&1
done; done
echo "[$(date +%H:%M:%S)] ALL_DONE"
