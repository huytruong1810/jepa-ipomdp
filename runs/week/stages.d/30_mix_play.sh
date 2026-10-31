#!/bin/bash
# Stage C: controlled coupling experiment (exact_model_play.py header 5): EPR and MIX0.5 on the held-out analysis
# games, recipe (double) and abl_max (max) networks, training seeds 5-9. Each play is skipped once its npz exists.
cd /home/andyj1810/projects/jepa_ipomdp
P="nice -n 10 .venv/bin/python reports/2026-10-02_wumpus_diagnostics/exact_model_play.py"
O=runs/week/mix; mkdir -p $O
for sweep in wumpus_retrain_heldout abl_max; do
  R=runs/sweeps/$sweep/seed5
  [ -e $R/analysis_heldout/exact_play_learned.npz ] || { gate; $P $R learned --heldout > $O/${sweep}_seed5_learned.log 2>&1 || exit 1; }
done
for k in 5 6 7 8 9; do
  for sweep in wumpus_retrain_heldout abl_max; do
    R=runs/sweeps/$sweep/seed$k
    [ -e $R/analysis_heldout/report.json ] || { echo "missing $R"; exit 1; }
    for arm in EPR MIX0.5; do
      [ -e $R/analysis_heldout/exact_play_$arm.npz ] || { gate; $P $R $arm --heldout > $O/${sweep}_seed${k}_$arm.log 2>&1 || exit 1; }
    done
  done
done
