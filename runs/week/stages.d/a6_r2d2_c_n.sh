#!/bin/bash
# PROTOCOL.md C-n development screen: R2D2-lite arms N1-N3 (exploration=anneal, value_scale=linear, both) on training
# seeds 0-2, analysis seed 12345 (development games). One run at a time; gate + GPU_BUSY around each run (~15-20 min).
# A finished run is skipped by the baseline script itself (final.pt exists), so a restarted stage resumes.
cd /home/andyj1810/projects/jepa_ipomdp
export CUDA_LAUNCH_BLOCKING=1   # WSL GPU driver faults on async launches since 2026-11-14 18:30; same numerics
set -o pipefail
declare -A ARMS=([r2d2_n1]="exploration=anneal" [r2d2_n2]="value_scale=linear" [r2d2_n3]="exploration=anneal value_scale=linear")
for seed in 0 1 2; do
  for name in r2d2_n1 r2d2_n2 r2d2_n3; do
    [ -e runs/baselines/$name/seed$seed/analysis/report.json ] && continue
    gate
    echo "$name seed $seed $(date '+%F %T')" > runs/week/GPU_BUSY
    OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python reports/2026-10-19_eval_protocol/recurrent_dqn_baseline.py $name $seed \
      --set ${ARMS[$name]} > runs/baselines/${name}_seed${seed}.log 2>&1
    rc=$?
    rm -f runs/week/GPU_BUSY
    [ $rc = 0 ] || { echo "$name seed $seed failed rc=$rc"; exit 1; }
  done
done
