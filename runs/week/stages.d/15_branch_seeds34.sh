#!/bin/bash
# Stage A2 (conditional, branch_games.py header 5, R-CONFIRM): random and branch arms on seeds 3-4.
# Condition met 2026-10-31: R-PLAY random - control >= +30 on seeds 0 (+78.1) and 1 (+201.8), 2 of 3 whatever seed 2 gives.
cd /home/andyj1810/projects/jepa_ipomdp
O=runs/wumpus/branch
D=reports/2026-10-02_wumpus_diagnostics
SEEDS="3 4" bash $O/branch_chain.sh   # its final compare lines repeat seeds 0-2 (cheap); the 5-seed compares follow
for s in 3 4; do grep -q "random_seed$s" $O/play_seed$s.log || exit 1; done
RUNS="runs/sweeps/wumpus_retrain/seed0 runs/sweeps/wumpus_retrain/seed1 runs/sweeps/wumpus_retrain/seed2 runs/sweeps/wumpus_retrain/seed3 runs/sweeps/wumpus_retrain/seed4"
nice -n 10 .venv/bin/python $D/feature_localise.py compare control branch $RUNS > $O/compare_branch_5seeds.log 2>&1 || exit 1
nice -n 10 .venv/bin/python $D/feature_localise.py compare control random $RUNS > $O/compare_random_5seeds.log 2>&1 || exit 1
nice -n 10 .venv/bin/python $D/feature_localise.py compare random branch $RUNS > $O/compare_branch_vs_random_5seeds.log 2>&1
