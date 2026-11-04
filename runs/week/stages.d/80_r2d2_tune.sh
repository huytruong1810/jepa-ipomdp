#!/bin/bash
# PROTOCOL.md C-f development screen: R2D2-lite arms F1-F3 on training seeds 0-2, analysis seed 12345 (development
# games; the script prints the development returns). One run at a time; gate + GPU_BUSY around each run (~13-19 min).
# A finished run is skipped by the baseline script itself (final.pt exists), so a restarted stage resumes.
cd /home/andyj1810/projects/jepa_ipomdp
set -o pipefail
declare -A ARMS=([r2d2_f1]="batch_size=64" [r2d2_f2]="hidden_dim=512" [r2d2_f3]="nstep=10 batch_size=64")
for seed in 0 1 2; do
  for name in r2d2_f1 r2d2_f2 r2d2_f3; do
    [ -e runs/baselines/$name/seed$seed/analysis/report.json ] && continue
    gate
    echo "$name seed $seed $(date '+%F %T')" > runs/week/GPU_BUSY
    nice -n 10 .venv/bin/python reports/2026-10-19_eval_protocol/recurrent_dqn_baseline.py $name $seed \
      --set ${ARMS[$name]} > runs/baselines/${name}_seed${seed}.log 2>&1
    rc=$?
    rm -f runs/week/GPU_BUSY
    [ $rc = 0 ] || { echo "$name seed $seed failed rc=$rc"; exit 1; }
  done
done
