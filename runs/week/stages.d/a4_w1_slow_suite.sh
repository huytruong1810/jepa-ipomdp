#!/bin/bash
# Submission audit W1 (AUDIT_LOG.md E1): the slow acceptance suite (pytest -m slow: certified V* solve, exact-model
# planner vs optimal, world-model acceptance on the GPU) at the current main, through the gate; log into the audit's
# logs/ folder with the commit. Last run 2026-10-19.
cd /home/andyj1810/projects/jepa_ipomdp
L=reports/2026-11-14_submission_audit/logs/slow_suite_$(date +%F).log
gate
echo "W1 slow suite $(date '+%F %T')" > runs/week/GPU_BUSY
{ echo "commit $(git log -1 --format=%H)  started $(date '+%F %T')"; \
  OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python -m pytest -m slow -v -p no:cacheprovider 2>&1; rc=$?; \
  echo "finished $(date '+%F %T') rc=$rc"; exit $rc; } > $L 2>&1
rc=$?
rm -f runs/week/GPU_BUSY
exit $rc
