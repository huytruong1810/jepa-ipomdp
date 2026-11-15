#!/bin/bash
# Submission audit W7: median-constant baseline of the held-out Wumpus probes (seeds 5-9); about 2 GPU-minutes.
cd /home/andyj1810/projects/jepa_ipomdp
L=reports/2026-11-14_submission_audit/logs/w7_median_constant_$(date +%F).log
gate
echo "W7 median constant $(date '+%F %T')" > runs/week/GPU_BUSY
{ echo "commit $(git log -1 --format=%H)"; OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python \
  reports/2026-11-14_submission_audit/w7_median_constant.py; } > $L 2>&1
rc=$?
rm -f runs/week/GPU_BUSY
exit $rc
