#!/bin/bash
# Stage A: restore-and-branch screen, seeds 1-2 (branch_games.py header 3); resumes at seed 1.
cd /home/andyj1810/projects/jepa_ipomdp
SEEDS="1 2" bash runs/wumpus/branch/branch_chain.sh  # seed 0 is complete (2026-10-30)
grep -q "CHAIN DONE" runs/wumpus/branch/branch_chain.log
