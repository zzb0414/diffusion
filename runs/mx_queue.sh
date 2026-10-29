#!/usr/bin/env bash
# GEN -> MX_CV training -> MX_CV evaluation. Stops at the first failure.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }
# 1. GEN: dataset1-only models on unseen datasets (A also on ds1 = verification 4)
run "GEN A"  $PY experiments/_mx_eval.py --run runs/cv_r3                --datasets 1 2 3 --out runs/gen/A > runs/gen/stdout_A.txt 2>&1
run "GEN B"  $PY experiments/_mx_eval.py --run runs/cv_ablation/l1_norm  --datasets 2 3   --out runs/gen/B > runs/gen/stdout_B.txt 2>&1
run "GEN C"  $PY experiments/_mx_eval.py --run runs/cv_ablation/mse      --datasets 2 3   --out runs/gen/C > runs/gen/stdout_C.txt 2>&1
run "GEN D"  $PY experiments/_mx_eval.py --run runs/cv_ablation/mse_norm --datasets 2 3   --out runs/gen/D > runs/gen/stdout_D.txt 2>&1
echo "[$(date +%H:%M:%S)] GEN_DONE"
# 2. MX_CV training
for arm in A B C D; do for k in 0 1 2 3 4; do
  run "TRAIN $arm fold$k" $PY experiments/_mx_train.py --arm $arm --fold $k > runs/mx_cv/stdout_${arm}_fold$k.txt 2>&1
done; done
echo "[$(date +%H:%M:%S)] TRAIN_DONE"
# 3. MX_CV evaluation: primary (2000, full sweep) then secondary (6000, r in {0, 1/3, 1})
for arm in A B C D; do
  run "EVAL $arm 2000" $PY experiments/_mx_eval.py --run runs/mx_cv/$arm --iter 2000 --datasets 1 2 3 --out runs/mx_cv/$arm/eval_it2000 > runs/mx_cv/eval_stdout_${arm}_2000.txt 2>&1
done
for arm in A B C D; do
  run "EVAL $arm 6000" $PY experiments/_mx_eval.py --run runs/mx_cv/$arm --iter 6000 --datasets 1 2 3 --ratios 0 0.3333333 1 --out runs/mx_cv/$arm/eval_it6000 > runs/mx_cv/eval_stdout_${arm}_6000.txt 2>&1
done
echo "[$(date +%H:%M:%S)] ALL_DONE"
