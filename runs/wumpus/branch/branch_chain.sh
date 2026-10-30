#!/bin/bash
# branch_games.py section 3 screen, seeds 0-2: arms branch and random vs the coverage control retrains.
# One GPU job at a time; each step waits for >= 9 GB MemAvailable (another project shares the machine).
cd /home/andyj1810/projects/jepa_ipomdp
O=runs/wumpus/branch
C=runs/wumpus/coverage
D=reports/2026-10-02_wumpus_diagnostics
PY="nice -n 10 .venv/bin/python"
export OMP_NUM_THREADS=2
log() { echo "[$(date '+%F %T')] $*" >> $O/branch_chain.log; }
memgate() { until [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -ge 9000 ]; do sleep 120; done; }
log "commit $(git log -1 --format=%h)"
for s in 0 1 2; do
  R=runs/sweeps/wumpus_retrain/seed$s
  for arm in branch random; do
    [ -f $O/games_${arm}_seed$s.pt ] || { memgate; $PY $D/branch_games.py $R $arm $O/games_${arm}_seed$s.pt > $O/games_${arm}_seed$s.log 2>&1; log "seed $s games $arm exit $?"; }
    [ -f $O/${arm}_seed$s.pt ] || { memgate; $PY $D/objective_ab.py --config $R/.hydra/config.yaml --representation decoder --horizon 5 \
        --pool-cache runs/wumpus/supervised_filter/pools.pt --seed $s --train-buffer $R \
        --extra-games $C/games_control_seed$s.pt $O/games_${arm}_seed$s.pt --save $O/${arm}_seed$s.pt \
        > $O/train_${arm}_seed$s.log 2>&1; log "seed $s retrain $arm exit $?"; }
    memgate; $PY $D/feature_localise.py $R cuda --networks $O/${arm}_seed$s.pt > $O/fl_${arm}_seed$s.log 2>&1
    log "seed $s feature_localise $arm exit $?"
  done
  memgate; $PY $D/retrain_play.py $R $C/control_seed$s.pt $O/branch_seed$s.pt $O/random_seed$s.pt > $O/play_seed$s.log 2>&1
  log "seed $s play exit $?"
  memgate; $PY $D/risk_localise.py $R $O/branch_seed$s.pt $O/random_seed$s.pt > $O/risk_seed$s.log 2>&1
  log "seed $s risk exit $?"
done
$PY $D/feature_localise.py compare control branch runs/sweeps/wumpus_retrain/seed{0,1,2} > $O/compare_branch.log 2>&1
$PY $D/feature_localise.py compare control random runs/sweeps/wumpus_retrain/seed{0,1,2} > $O/compare_random.log 2>&1
$PY $D/feature_localise.py compare random branch runs/sweeps/wumpus_retrain/seed{0,1,2} > $O/compare_branch_vs_random.log 2>&1
log "CHAIN DONE"
