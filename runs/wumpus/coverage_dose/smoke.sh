#!/bin/bash
# Dose smoke of coverage_games.py section 7: own arm, seed 0, 1 x 512 games at trial probability 0.5 and 1.0.
cd /home/andyj1810/projects/jepa_ipomdp
export OMP_NUM_THREADS=2
for eps in 0.5 1.0; do
  avail=$(free -m | awk '/Mem:/{print $7}')
  if [ "$avail" -lt 8000 ]; then echo "[$(date '+%F %T')] only ${avail} MB available, stopping before eps $eps"; exit 3; fi
  nice -n 10 uv run --no-sync python reports/2026-10-02_wumpus_diagnostics/coverage_games.py \
    runs/sweeps/wumpus_retrain/seed0 own runs/wumpus/coverage_dose/games_own${eps}_seed0.pt 1 512 $eps \
    > runs/wumpus/coverage_dose/games_own${eps}_seed0.log 2>&1
  echo "[$(date '+%F %T')] eps $eps exit $?"
done
