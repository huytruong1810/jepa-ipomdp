#!/bin/bash
# PROTOCOL.md C-i (smoke S-C): the rows of seeds 1-2 (target_rule_smoke.py: best.pt greedy planner on the first 512
# control games, labels from the shadow heuristic), then sc_smoke.py on the CPU. Seed 0's rows exist. Gate + GPU_BUSY
# around each GPU play; a seed whose output exists is skipped. Seed 0 is re-run for the within-row AUCs (sc_smoke.py
# header 4); every smoke passes the gate first (CPU-heavy, so it also waits while ipomcp's R-final runs).
cd /home/andyj1810/projects/jepa_ipomdp
mkdir -p runs/wumpus/sc_smoke
for s in 0 1 2; do
  R=runs/sweeps/wumpus_retrain/seed$s
  if [ ! -f runs/wumpus/target/smoke_seed$s.npz ]; then
    gate
    echo "sc_rows seed $s $(date '+%F %T')" > runs/week/GPU_BUSY
    nice -n 10 .venv/bin/python reports/2026-10-02_wumpus_diagnostics/target_rule_smoke.py $R runs/wumpus/target/smoke_seed$s.npz \
      > runs/wumpus/target/smoke_seed$s.log 2>&1
    rc=$?
    rm -f runs/week/GPU_BUSY
    [ $rc = 0 ] || { echo "rows seed $s failed rc=$rc"; exit 1; }
  fi
  if ! grep -q "within non-wall" runs/wumpus/sc_smoke/seed$s.log 2>/dev/null; then
    gate                                                  # CPU-heavy (~3 h): also waits for ipomcp's R-final
    CUDA_VISIBLE_DEVICES= nice -n 15 .venv/bin/python reports/2026-11-09_augmentations/sc_smoke.py $R runs/wumpus/target/smoke_seed$s.npz \
      > runs/wumpus/sc_smoke/seed$s.log 2>&1 || exit 1
  fi
done
