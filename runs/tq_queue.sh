#!/usr/bin/env bash
# TQ: E2 (degradation, eval-only) -> E1 training (target_k) -> E1 evaluation. Stops at the first failure.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }
declare -A MODEL=( [one_A]=runs/cv_r3 [one_D]=runs/cv_ablation/mse_norm [pool_A]=runs/mx_cv/A [pool_D]=runs/mx_cv/D )
# E2: primary first (dataset1-only A, f = 2.0, all colours), then the rest
for spec in one_A:2.0 one_A:1.4 one_D:2.0 one_D:1.4 pool_A:2.0 pool_A:1.4 pool_D:2.0 pool_D:1.4; do
  m=${spec%%:*}; f=${spec#*:}
  for mode in self white ds2; do
    out=runs/degrade/${m}_${mode}_f${f}
    run "E2 $m $mode f$f" $PY experiments/_mx_eval.py --run ${MODEL[$m]} --iter 2000 --datasets 1 \
        --degrade $mode:$f --out $out > runs/degrade/stdout_${m}_${mode}_f${f}.txt 2>&1
  done
done
echo "[$(date +%H:%M:%S)] E2_DONE"
# E1: training with k-subset LOO targets
for k in 1 3; do for k_fold in 0 1 2 3 4; do
  run "E1 train A k$k fold$k_fold" $PY experiments/_cv_train.py --fold $k_fold --loss l1 --target-k $k \
      --outdir runs/tq/A_k$k > runs/tq/stdout_A_k${k}_fold$k_fold.txt 2>&1
  run "E1 train D k$k fold$k_fold" $PY experiments/_cv_train.py --fold $k_fold --loss mse --noise-norm --target-k $k \
      --outdir runs/tq/D_k$k > runs/tq/stdout_D_k${k}_fold$k_fold.txt 2>&1
done; done
echo "[$(date +%H:%M:%S)] E1_TRAIN_DONE"
for r in A_k1 A_k3 D_k1 D_k3; do
  run "E1 eval $r" $PY experiments/_mx_eval.py --run runs/tq/$r --iter 2000 --datasets 1 2 3 \
      --out runs/tq/$r/eval_it2000 > runs/tq/eval_stdout_$r.txt 2>&1
done
echo "[$(date +%H:%M:%S)] ALL_DONE"
