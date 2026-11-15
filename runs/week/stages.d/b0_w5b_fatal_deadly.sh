#!/bin/bash
# Submission audit W5(b): risk rows for the recipe's best.pt on its 512 development analysis games (seeds 0-4), then
# the share of fatal FORWARDs into provably deadly cells (reports/2026-11-14_submission_audit/w5b_fatal_deadly.py).
cd /home/andyj1810/projects/jepa_ipomdp
O=runs/wumpus/w5b
mkdir -p $O
L=reports/2026-11-14_submission_audit/logs/w5b_fatal_deadly_$(date +%F).log
for s in 0 1 2 3 4; do
  R=runs/sweeps/wumpus_retrain/seed$s
  [ -f $O/seed${s}_best.pt ] || .venv/bin/python reports/2026-11-14_submission_audit/w5b_fatal_deadly.py export $s || exit 1
  if [ ! -f $O/seed${s}_best.risk.npz ]; then
    gate
    echo "W5b risk seed $s $(date '+%F %T')" > runs/week/GPU_BUSY
    OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python reports/2026-10-02_wumpus_diagnostics/risk_localise.py $R $O/seed${s}_best.pt \
      > $O/risk_seed$s.log 2>&1
    rc=$?
    rm -f runs/week/GPU_BUSY
    [ $rc = 0 ] || exit 1
  fi
done
.venv/bin/python reports/2026-11-14_submission_audit/w5b_fatal_deadly.py read 0 1 2 3 4 > $L 2>&1
