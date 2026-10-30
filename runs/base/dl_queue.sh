#!/usr/bin/env bash
# BASE-DL (Archive/BASE/PREREGISTRATION_DL.md): train Neighbor2Neighbor, R2R, Noise2Self (3 seeds x 5 folds), then
# score them with the two re-scored Noise2Noise arms on the baseline splits. Completed runs are skipped.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"
run() { echo "[$(date +%H:%M:%S)] START $1"; shift; "$@" || { echo "[$(date +%H:%M:%S)] FAILED"; exit 1; }; }
mkdir -p runs/base/logs
for m in ne2ne r2r n2self; do for s in 101 102 103; do for f in 0 1 2 3 4; do
  grep -qs DONE runs/base/$m/s$s/fold$f/train.log && continue
  rm -rf runs/base/$m/s$s/fold$f
  run "$m s$s fold$f" $PY baselines/dl_selfsup.py --method $m --fold $f --seed $s --outdir runs/base/$m/s$s > runs/base/logs/${m}_s${s}_f$f.txt 2>&1
done; done; done
[ -f runs/base/dl_eval/neff.csv ] || run "DL eval" $PY baselines/evaluate.py --methods n2n n2n_sigma ne2ne r2r n2self --out runs/base/dl_eval > runs/base/logs/dl_eval.txt 2>&1
echo "[$(date +%H:%M:%S)] ALL_DONE"
