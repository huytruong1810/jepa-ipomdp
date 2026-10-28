#!/bin/bash
# objective_ab.py 2p: observation-conditioned reward (OR) and Q-equivalent term (QE<w>) retrain screen. One GPU job at
# a time. Per seed: retrain the arms (OR seed 0 already trained for the dose smoke), play best.pt + OR + QE with
# retrain_play.py --set (QE paired in-script with OR), then feature_localise.py --networks for each arm.
# Usage: bash runs/wumpus/qe_screen_chain.sh <w>
cd /home/andyj1810/projects/jepa_ipomdp
W=$1
[ -n "$W" ] || { echo "usage: $0 <w>"; exit 2; }
O=runs/wumpus/qe_screen
QE=QE$W
SET="--set model.observation_reward=true"
log() { echo "[$(date '+%F %T')] $*" >> $O/chain.log; }
log "commit $(git log -1 --format=%h), w = $W"
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=temperature.gpu,memory.used,power.draw --format=csv,noheader) | $(free -m | awk '/Mem/{print $3"MB"} /Swap/{print "swap "$3}' | tr '\n' ' ') | $(cut -d' ' -f1-3 /proc/loadavg)"; sleep 60; done ) > $O/sysmon.log 2>&1 &
MON=$!
train() {   # <seed> <arm> <extra --set items>
  local s=$1 arm=$2; shift 2
  local R=runs/sweeps/wumpus_retrain/seed$s
  if [ -f $O/${arm}_seed$s.pt ]; then log "seed $s $arm exists, skipped"; return; fi
  OMP_NUM_THREADS=4 .venv/bin/python reports/2026-10-02_wumpus_diagnostics/objective_ab.py --config $R/.hydra/config.yaml \
    --representation decoder --horizon 5 --pool-cache runs/wumpus/supervised_filter/pools.pt --seed $s --train-buffer $R \
    --set model.observation_reward=true "$@" --save $O/${arm}_seed$s.pt > $O/train_${arm}_seed$s.log 2>&1
  log "seed $s $arm train exit $?"
}
for s in 0 1 2; do
  R=runs/sweeps/wumpus_retrain/seed$s
  train $s OR
  train $s $QE model.q_equivalent_weight=$W
  OMP_NUM_THREADS=4 .venv/bin/python reports/2026-10-02_wumpus_diagnostics/retrain_play.py $R $O/OR_seed$s.pt \
    $O/${QE}_seed$s.pt $SET > $O/play_seed$s.log 2>&1
  log "seed $s play exit $?"
  for arm in OR $QE; do
    OMP_NUM_THREADS=4 .venv/bin/python reports/2026-10-02_wumpus_diagnostics/feature_localise.py $R cuda \
      --networks $O/${arm}_seed$s.pt $SET > $O/fl_${arm}_seed$s.log 2>&1
    log "seed $s $arm feature_localise exit $?"
  done
done
kill $MON
log "ALL DONE"
