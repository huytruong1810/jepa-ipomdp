#!/bin/bash
# PROTOCOL.md C-m / objective_ab.py 2t: dead-reckoned ego map, retrain screen on development seeds 0-2 (arms EGO1,
# EGO01 against runs/wumpus/ve_screen/control_seed<s>.pt). Smoke first (EGO1, 20 updates; risk_localise and
# feature_localise on 8 games). Gate + GPU_BUSY around every step; each step is skipped when its output exists, so a
# restarted stage resumes. Machine contract (2026-11-13): <= 8 heavy CPU threads, MemAvailable >= 8 GB (the gate).
cd /home/andyj1810/projects/jepa_ipomdp
O=runs/wumpus/ego_screen
C=runs/wumpus/ve_screen
mkdir -p $O
D=reports/2026-10-02_wumpus_diagnostics
EGO="--set model.ego_map_weight=1.0"
log() { echo "[$(date '+%F %T')] $*" >> $O/chain.log; }
step() {
  local label=$1; shift
  gate
  echo "ego_screen $label $(date '+%F %T')" > runs/week/GPU_BUSY
  nice -n 10 "$@"
  local rc=$?
  rm -f runs/week/GPU_BUSY
  log "$label exit $rc"
  [ $rc = 0 ] || exit 1
}
log "commit $(git log -1 --format=%h)"
R0=runs/sweeps/wumpus_retrain/seed0
if [ ! -f $O/smoke.done ]; then
  step "smoke train" env OMP_NUM_THREADS=4 .venv/bin/python $D/objective_ab.py --config $R0/.hydra/config.yaml \
    --representation decoder --horizon 5 --pool-cache runs/wumpus/supervised_filter/pools.pt --seed 0 --train-buffer $R0 \
    --updates 20 --set model.ego_map_weight=1.0 --save $O/smoke_EGO1.pt > $O/smoke_train.log 2>&1
  step "smoke risk" env OMP_NUM_THREADS=4 .venv/bin/python $D/risk_localise.py $R0 $O/smoke_EGO1.pt 8 $EGO > $O/smoke_risk.log 2>&1
  step "smoke localise" env OMP_NUM_THREADS=4 .venv/bin/python $D/feature_localise.py $R0 cuda 8 --networks $O/smoke_EGO1.pt \
    $EGO > $O/smoke_fl.log 2>&1
  touch $O/smoke.done
fi
for s in 0 1 2; do
  R=runs/sweeps/wumpus_retrain/seed$s
  [ -f $C/control_seed$s.risk.npz ] || step "seed $s control risk" env OMP_NUM_THREADS=4 .venv/bin/python $D/risk_localise.py \
    $R $C/control_seed$s.pt > $O/risk_control_seed$s.log 2>&1
  for arm in EGO1 EGO01; do
    w=1.0; [ $arm = EGO01 ] && w=0.1
    [ -f $O/${arm}_seed$s.pt ] || step "seed $s $arm train" env OMP_NUM_THREADS=4 .venv/bin/python $D/objective_ab.py \
      --config $R/.hydra/config.yaml --representation decoder --horizon 5 --pool-cache runs/wumpus/supervised_filter/pools.pt \
      --seed $s --train-buffer $R --set model.ego_map_weight=$w --save $O/${arm}_seed$s.pt > $O/train_${arm}_seed$s.log 2>&1
  done
  [ -f $O/EGO01_seed$s.returns.npy ] || step "seed $s play" env OMP_NUM_THREADS=4 .venv/bin/python $D/retrain_play.py $R \
    $O/EGO1_seed$s.pt $O/EGO01_seed$s.pt $EGO > $O/play_seed$s.log 2>&1
  [ -f $O/EGO01_seed$s.risk.npz ] || step "seed $s risk" env OMP_NUM_THREADS=4 .venv/bin/python $D/risk_localise.py $R \
    $O/EGO1_seed$s.pt $O/EGO01_seed$s.pt $EGO > $O/risk_seed$s.log 2>&1
  for arm in EGO1 EGO01; do
    [ -f $O/fl_${arm}_seed$s.done ] || { step "seed $s $arm feature_localise" env OMP_NUM_THREADS=4 .venv/bin/python \
      $D/feature_localise.py $R cuda --networks $O/${arm}_seed$s.pt $EGO > $O/fl_${arm}_seed$s.log 2>&1; touch $O/fl_${arm}_seed$s.done; }
  done
done
log "ALL DONE"
