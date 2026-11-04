#!/bin/bash
# PROTOCOL.md C-g: the recipe's held-out networks play the held-out games with their learned model at 10 and 100
# simulations (50 is C-a). One play at a time; gate + GPU_BUSY around each. The 100-simulation play peaks at ~9.5 GB
# VRAM (smoke 2026-11-04: 18 MiB per game), above our usual 4 GB, so it runs only behind the gate and holding
# GPU_BUSY (peers wait for it). A finished play (its npz exists) is skipped, so a restarted stage resumes.
cd /home/andyj1810/projects/jepa_ipomdp
for sims in 10 100; do
  for seed in 5 6 7 8 9; do
    run=runs/sweeps/wumpus_retrain_heldout/seed$seed
    [ -e $run/analysis_heldout/exact_play_learned_sims$sims.npz ] && continue
    gate
    echo "search_budget seed $seed sims $sims $(date '+%F %T')" > runs/week/GPU_BUSY
    nice -n 10 .venv/bin/python reports/2026-10-02_wumpus_diagnostics/exact_model_play.py $run learned --heldout \
      --sims=$sims > runs/week/search_budget_seed${seed}_sims${sims}.log 2>&1
    rc=$?
    rm -f runs/week/GPU_BUSY
    [ $rc = 0 ] || { echo "seed $seed sims $sims failed rc=$rc"; exit 1; }
  done
done
