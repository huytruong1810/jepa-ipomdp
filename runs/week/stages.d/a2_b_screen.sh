#!/bin/bash
# PROTOCOL.md C-j / objective_ab.py 2s: Stage-B architecture retrain screen on development seeds 0-2 (arms TR, CH, TRCE
# against runs/wumpus/ve_screen/control_seed<s>.pt). Smoke first (TRCE, 20 updates, feature_localise on 8 games). Gate +
# GPU_BUSY around every step; each step is skipped when its output exists, so a restarted stage resumes.
cd /home/andyj1810/projects/jepa_ipomdp
O=runs/wumpus/b_screen
mkdir -p $O
D=reports/2026-10-02_wumpus_diagnostics
log() { echo "[$(date '+%F %T')] $*" >> $O/chain.log; }
step() {
  local label=$1; shift
  gate
  echo "b_screen $label $(date '+%F %T')" > runs/week/GPU_BUSY
  nice -n 10 "$@"
  local rc=$?
  rm -f runs/week/GPU_BUSY
  log "$label exit $rc"
  [ $rc = 0 ] || exit 1
}
sets_of() {
  case $1 in
    TR) echo "model.filter=transformer" ;;
    CH) echo "model.observation_head=chain" ;;
    TRCE) echo "model.filter=transformer model.observation_head=chain model.kinematic_weight=0.1" ;;
  esac
}
setflags() { local f=""; for x in $(sets_of $1); do f="$f --set $x"; done; echo $f; }
log "commit $(git log -1 --format=%h)"
R0=runs/sweeps/wumpus_retrain/seed0
if [ ! -f $O/smoke.done ]; then
  step "smoke train" env OMP_NUM_THREADS=4 .venv/bin/python $D/objective_ab.py --config $R0/.hydra/config.yaml --representation decoder \
    --horizon 5 --pool-cache runs/wumpus/supervised_filter/pools.pt --seed 0 --train-buffer $R0 --updates 20 \
    --set $(sets_of TRCE) --save $O/smoke_TRCE.pt > $O/smoke_train.log 2>&1
  step "smoke localise" env OMP_NUM_THREADS=4 .venv/bin/python $D/feature_localise.py $R0 cuda 8 --networks $O/smoke_TRCE.pt \
    $(setflags TRCE) > $O/smoke_fl.log 2>&1
  touch $O/smoke.done
fi
for s in 0 1 2; do
  R=runs/sweeps/wumpus_retrain/seed$s
  for arm in TR CH TRCE; do
    [ -f $O/${arm}_seed$s.pt ] || step "seed $s $arm train" env OMP_NUM_THREADS=4 .venv/bin/python $D/objective_ab.py \
      --config $R/.hydra/config.yaml --representation decoder --horizon 5 --pool-cache runs/wumpus/supervised_filter/pools.pt \
      --seed $s --train-buffer $R --set $(sets_of $arm) --save $O/${arm}_seed$s.pt > $O/train_${arm}_seed$s.log 2>&1
    [ -f $O/${arm}_seed$s.returns.npy ] || step "seed $s $arm play" env OMP_NUM_THREADS=4 .venv/bin/python $D/retrain_play.py $R \
      $O/${arm}_seed$s.pt $(setflags $arm) > $O/play_${arm}_seed$s.log 2>&1
    [ -f $O/fl_${arm}_seed$s.done ] || { step "seed $s $arm feature_localise" env OMP_NUM_THREADS=4 .venv/bin/python \
      $D/feature_localise.py $R cuda --networks $O/${arm}_seed$s.pt $(setflags $arm) > $O/fl_${arm}_seed$s.log 2>&1; touch $O/fl_${arm}_seed$s.done; }
  done
done
log "ALL DONE"
