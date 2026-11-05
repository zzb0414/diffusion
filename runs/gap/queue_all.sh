#!/usr/bin/env bash
# finish the ASL-COIL 1104a follow-up (two evals failed when the data moved), then GAP A1/A2. One GPU job at a time.
bash runs/coil/xsess/queue.sh && bash runs/gap/queue_scale.sh
