#!/usr/bin/env bash
# CLAIMS (Archive/CLAIMS/PREREGISTRATION.md): LC -> C3 -> C2 (gated on runs/dec/_res_verified) -> C7. Resumable.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
L=runs/claims/logs
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }
trained() { grep -qs DONE "$1/train.log"; }
# LC: learning curve of the W05 natural-image models
for it in 1000 2000; do for m in ours sup sub subsup; do
  [ -f runs/nat/lc/it$it/W05/$m.csv ] || run "LC $m it$it" $PY natural/evaluate.py --cond W05 --methods $m --iter $it --out-dir runs/nat/lc/it$it/W05 > $L/lc_${m}_$it.txt 2>&1
done; done
# C3: no-ladder model at W05, then all three models on in-distribution and shifted test sets
trained runs/nat/c3/noladder/ours || { rm -rf runs/nat/c3/noladder/ours; run "C3 train noladder" $PY natural/train.py --method ours --cond W05 --p-raw 1.0 --outdir runs/nat/c3/noladder/ours > $L/c3_train.txt 2>&1; }
for t in W05 W035 W07 C05; do
  [ -f runs/nat/eval_c3/$t/ladder/n2n.csv ] || run "C3 eval ladder+n2n on $t" $PY natural/evaluate.py --cond $t --methods ours n2n --run-root runs/nat/W05 --out-dir runs/nat/eval_c3/$t/ladder > $L/c3_eval_ladder_$t.txt 2>&1
  [ -f runs/nat/eval_c3/$t/noladder/ours.csv ] || run "C3 eval noladder on $t" $PY natural/evaluate.py --cond $t --methods ours --run-root runs/nat/c3/noladder --out-dir runs/nat/eval_c3/$t/noladder > $L/c3_eval_noladder_$t.txt 2>&1
done
# C2: residual parametrization on ASL (after the CPU determinism check of the library change)
until [ -f runs/dec/_res_verified ]; do sleep 60; done
for s in 101 102 103; do for f in 0 1 2 3 4; do
  trained runs/dec/A/resid_s$s/fold$f && continue
  rm -rf runs/dec/A/resid_s$s/fold$f
  run "C2 train resid s$s f$f" $PY experiments/_gr_train.py --mode per --fold $f --iters 2000 --seed $s --loss mse --noise-norm \
      --param residual --outdir runs/dec/A/resid_s$s > $L/c2_train_s${s}_f$f.txt 2>&1
done; done
for s in 101 102 103; do
  [ -f runs/dec/A/eval/resid_s$s/neff.csv ] || run "C2 eval s$s" $PY experiments/_dec_eval.py --run runs/dec/A/resid_s$s --loss mse --datasets 1 2 3 --out runs/dec/A/eval/resid_s$s > $L/c2_eval_s$s.txt 2>&1
done
# C7: per-copy (ours) vs denoise-the-mean (sub) across training-set size and noise
for c in W05 W20; do for n in 25 100 400; do for m in ours sub; do
  if [ $n = 400 ] && [ -d runs/nat/$c/$m ] && grep -qs DONE runs/nat/$c/$m/train.log; then continue; fi    # NAT run reused
  trained runs/nat/c7/$c/n$n/$m && continue
  rm -rf runs/nat/c7/$c/n$n/$m
  run "C7 train $c n$n $m" $PY natural/train.py --method $m --cond $c --n-train $n --outdir runs/nat/c7/$c/n$n/$m > $L/c7_train_${c}_n${n}_$m.txt 2>&1
done; done; done
for c in W05 W20; do for n in 25 100 400; do for m in ours sub; do
  root=runs/nat/c7/$c/n$n
  if [ $n = 400 ] && grep -qs DONE runs/nat/$c/$m/train.log; then root=runs/nat/$c; fi
  [ -f runs/nat/eval_c7/$c/n$n/$m.csv ] || run "C7 eval $c n$n $m" $PY natural/evaluate.py --cond $c --methods $m --run-root $root --out-dir runs/nat/eval_c7/$c/n$n > $L/c7_eval_${c}_n${n}_$m.txt 2>&1
done; done; done
echo "[$(date +%H:%M:%S)] ALL_DONE"; touch runs/claims/queue.done
