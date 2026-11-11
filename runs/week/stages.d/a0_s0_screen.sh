#!/bin/bash
# PROTOCOL.md C-h / objective_ab.py 2r: Stage-0 estimation fixes, offline retrain screen on development seeds 0-2.
# Arms CB, CBP, SLOT, VCLIP retrained on each seed's own buffer, played on the 512 analysis games (retrain_play.py),
# localised (feature_localise.py). Paired control: runs/wumpus/ve_screen/control_seed<s>.pt (existing). One GPU job at a
# time; gate + GPU_BUSY around each step. Every step is skipped when its output exists, so a restarted stage resumes.
# ~48 min per arm per seed (2p timings) -> ~10 h.
cd /home/andyj1810/projects/jepa_ipomdp
O=runs/wumpus/s0_screen
mkdir -p $O
D=reports/2026-10-02_wumpus_diagnostics
CODEC="--set model.codec_bound=absorbing"
log() { echo "[$(date '+%F %T')] $*" >> $O/chain.log; }
step() {   # <label> <command...>: gate, hold GPU_BUSY, run, release; abort the stage on failure
  local label=$1; shift
  gate
  echo "s0_screen $label $(date '+%F %T')" > runs/week/GPU_BUSY
  nice -n 10 "$@"
  local rc=$?
  rm -f runs/week/GPU_BUSY
  log "$label exit $rc"
  [ $rc = 0 ] || exit 1
}
log "commit $(git log -1 --format=%h)"
for s in 0 1 2; do
  R=runs/sweeps/wumpus_retrain/seed$s
  train_args="--config $R/.hydra/config.yaml --representation decoder --horizon 5 --pool-cache runs/wumpus/supervised_filter/pools.pt --seed $s --train-buffer $R"
  for arm in CB CBP SLOT VCLIP; do
    [ -f $O/${arm}_seed$s.pt ] && continue
    case $arm in
      CB) sets="model.codec_bound=absorbing" ;;
      CBP) sets="model.codec_bound=absorbing model.target_prune=true" ;;
      SLOT) sets="model.double_partition=slot" ;;
      VCLIP) sets="trainer.separate_value_clip=true" ;;
    esac
    step "seed $s $arm train" env OMP_NUM_THREADS=4 .venv/bin/python $D/objective_ab.py $train_args --set $sets \
      --save $O/${arm}_seed$s.pt > $O/train_${arm}_seed$s.log 2>&1
  done
  if [ ! -f $O/CBP_seed$s.returns.npy ]; then
    step "seed $s play CB CBP" env OMP_NUM_THREADS=4 .venv/bin/python $D/retrain_play.py $R $O/CB_seed$s.pt $O/CBP_seed$s.pt \
      $CODEC > $O/play_codec_seed$s.log 2>&1
  fi
  if [ ! -f $O/VCLIP_seed$s.returns.npy ]; then
    step "seed $s play SLOT VCLIP" env OMP_NUM_THREADS=4 .venv/bin/python $D/retrain_play.py $R $O/SLOT_seed$s.pt \
      $O/VCLIP_seed$s.pt > $O/play_other_seed$s.log 2>&1
  fi
  for arm in CB CBP SLOT VCLIP; do
    [ -f $O/fl_${arm}_seed$s.done ] && continue
    extra=""
    case $arm in CB|CBP) extra=$CODEC ;; esac
    step "seed $s $arm feature_localise" env OMP_NUM_THREADS=4 .venv/bin/python $D/feature_localise.py $R cuda \
      --networks $O/${arm}_seed$s.pt $extra > $O/fl_${arm}_seed$s.log 2>&1
    touch $O/fl_${arm}_seed$s.done
  done
done
log "ALL DONE"
