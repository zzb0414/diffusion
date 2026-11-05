#!/usr/bin/env bash
# GAP A1/A2 (Archive/GAP/PREREGISTRATION.md): re-scaled noise label, evaluation only, on 1104c. Resumable.
set -u
PY="C:/ProgramData/anaconda3/python.exe -X utf8 -u"; Q=runs/gap/queue_scale.log
for spec in "runs/subj/PER_s101 per PER_s101" "runs/subj/SUB_s101 sub SUB_s101" "runs/coil/subj/SUBM_s101 sub SUBM_s101" \
            "runs/subj/PER_s102 per PER_s102" "runs/subj/SUB_s102 sub SUB_s102" "runs/coil/subj/SUBM_s102 sub SUBM_s102"; do
  set -- $spec
  for it in 1000 2000 3000; do
    o=runs/gap/scale/$3/eval_it$it
    [ -f $o/neff.csv ] && continue
    rm -rf $o; echo "[$(date +%H:%M:%S)] START $3 it$it" >> $Q
    $PY experiments/_scale_eval.py --run $1 --iter $it --mode $2 --datasets 6 --out $o > runs/gap/logs/scale_$3_it$it.txt 2>&1 || { echo "[$(date +%H:%M:%S)] FAILED $3 it$it" >> $Q; exit 1; }
  done
done
$PY experiments/_gap_report.py --dry >> $Q 2>&1 || { echo FAILED report dry >> $Q; exit 1; }
$PY experiments/_gap_report.py > runs/gap/logs/report_scale.txt 2>&1 || { echo FAILED report >> $Q; exit 1; }
echo "[$(date +%H:%M:%S)] ALL_DONE" >> $Q; touch runs/gap/queue_scale.done
