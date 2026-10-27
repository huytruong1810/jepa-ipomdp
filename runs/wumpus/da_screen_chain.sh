#!/bin/bash
# objective_ab.py 2o: decision-aware weighting retrain screen. Per seed, one GPU job at a time: retrain DA50, play
# control + DA50 with retrain_play.py (paired with the 2i control), then feature_localise.py --networks DA50.
cd /home/andyj1810/projects/jepa_ipomdp
O=runs/wumpus/da_screen
log() { echo "[$(date '+%F %T')] $*" >> $O/chain.log; }
log "commit $(git log -1 --format=%h)"
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=temperature.gpu,memory.used,power.draw --format=csv,noheader) | $(free -m | awk '/Mem/{print $3"MB"} /Swap/{print "swap "$3}' | tr '\n' ' ') | $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 60; done ) > $O/sysmon.log 2>&1 &
MON=$!
for s in 0 1 2; do
  R=runs/sweeps/wumpus_retrain/seed$s
  OMP_NUM_THREADS=4 .venv/bin/python reports/2026-10-02_wumpus_diagnostics/objective_ab.py --config $R/.hydra/config.yaml \
    --representation decoder --horizon 5 --pool-cache runs/wumpus/supervised_filter/pools.pt --seed $s --train-buffer $R \
    --set model.decision_aware_weight=0.5 --save $O/DA50_seed$s.pt > $O/train_DA50_seed$s.log 2>&1
  log "seed $s train exit $?"
  ln -sf ../ve_screen/control_seed$s.pt $O/control_seed$s.pt
  OMP_NUM_THREADS=4 .venv/bin/python reports/2026-10-02_wumpus_diagnostics/retrain_play.py $R $O/control_seed$s.pt \
    $O/DA50_seed$s.pt > $O/play_seed$s.log 2>&1
  log "seed $s play exit $?"
  OMP_NUM_THREADS=4 .venv/bin/python reports/2026-10-02_wumpus_diagnostics/feature_localise.py $R cuda \
    --networks $O/DA50_seed$s.pt > $O/fl_seed$s.log 2>&1
  log "seed $s feature_localise exit $?"
done
kill $MON
log "ALL DONE"
