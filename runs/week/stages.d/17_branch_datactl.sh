#!/bin/bash
# Stage A3: data-quantity control R-DATA of branch_games.py section 7 (arm "more": 7 x 512 fresh own control games),
# seeds 0-4; needs stage 15's random/branch retrains for seeds 3-4. Resumable (skips existing files).
cd /home/andyj1810/projects/jepa_ipomdp
O=runs/wumpus/branch
C=runs/wumpus/coverage
D=reports/2026-10-02_wumpus_diagnostics
PY="nice -n 10 .venv/bin/python"
export OMP_NUM_THREADS=2
log() { echo "[$(date '+%F %T')] $*" >> $O/branch_chain.log; }
log "datactl commit $(git log -1 --format=%h)"
for s in 0 1 2 3 4; do
  R=runs/sweeps/wumpus_retrain/seed$s
  [ -f $O/games_more_seed$s.pt ] || { gate; BATCH_OFFSET=4 $PY $D/coverage_games.py $R control $O/games_more_seed$s.pt 7 > $O/games_more_seed$s.log 2>&1; log "seed $s games more exit $?"; }
  [ -f $O/games_more_seed$s.pt ] || exit 1
  [ -f $O/more_seed$s.pt ] || { gate; $PY $D/objective_ab.py --config $R/.hydra/config.yaml --representation decoder --horizon 5 \
      --pool-cache runs/wumpus/supervised_filter/pools.pt --seed $s --train-buffer $R \
      --extra-games $C/games_control_seed$s.pt $O/games_more_seed$s.pt --save $O/more_seed$s.pt \
      > $O/train_more_seed$s.log 2>&1; log "seed $s retrain more exit $?"; }
  [ -f $O/more_seed$s.pt ] || exit 1
  [ -f $R/analysis/feature_localise_more_seed$s.json ] || { gate; $PY $D/feature_localise.py $R cuda --networks $O/more_seed$s.pt > $O/fl_more_seed$s.log 2>&1; log "seed $s feature_localise more exit $?"; }
  grep -q "random_seed$s" $O/play_more_seed$s.log 2>/dev/null || { gate; $PY $D/retrain_play.py $R $O/more_seed$s.pt $O/random_seed$s.pt $O/branch_seed$s.pt > $O/play_more_seed$s.log 2>&1; log "seed $s play more exit $?"; }
  grep -q "random_seed$s" $O/play_more_seed$s.log || exit 1
done
log "DATACTL DONE"
