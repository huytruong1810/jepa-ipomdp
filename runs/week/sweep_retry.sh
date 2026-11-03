#!/bin/bash
# Sourced by week stages. sweep_retry <name> --seeds a,b,c <sweep.py args...>
# - Runs the sweep ONE SEED AT A TIME (sweep.py --seeds <prefix up to that seed>; sweep.json records the overrides and
#   analysis settings, not the seed list, and finished seeds are only reloaded), so the queue gate (PAUSE file, RAM,
#   foreign GPU use) is checked between seeds, not only between stages.
# - While a seed trains/analyses, runs/week/GPU_BUSY exists (holds the stage name and seed). Other projects' GPU jobs
#   (agreed with the forge/imagegen session, 2026-11-02): create runs/week/PAUSE, wait until GPU_BUSY is gone (at most
#   one seed, ~1-2 h), run, delete PAUSE. The gate keeps us waiting while PAUSE exists.
# - If sweep.py fails because the VRAM guardrail aborted a seed (that seed dir has checkpoints/emergency.pt and no
#   TRAINED marker; another process on the GPU pushed device VRAM over vram_trip_mb), the seed dir moves to
#   runs/week/aborted/ and the seed reruns from scratch after the gate (training is deterministic per seed). Any other
#   failure is returned unchanged (fail loudly). At most 5 attempts per seed; every move is logged to
#   runs/week/aborted.log.
# - The first call of every sweep has a one-seed prefix, and sweep.py's in-process aggregate refuses a Student-t
#   interval over one seed ("needs at least 2 seeds") AFTER that seed has trained and been analysed (both stages 50 and
#   60 failed on this, 2026-11-02 22:11). That one error is accepted, and only when the prefix is one seed and the seed
#   dir has TRAINED and <analysis output>/report.json; the next call (two seeds) writes the aggregate. Any other error
#   still fails.
sweep_retry() {
  local name=$1; shift
  [ "$1" = "--seeds" ] || { echo "sweep_retry: --seeds must follow the name" >&2; return 2; }
  local seeds=$2; shift 2
  local prefix="" seed attempt d moved ok out="analysis" a prev=""
  for a in "$@"; do [ "$prev" = "--analysis-output" ] && out=$a; prev=$a; done
  mkdir -p runs/week/aborted
  local errlog=runs/week/sweep_retry_last.err
  for seed in ${seeds//,/ }; do
    prefix=${prefix:+$prefix,}$seed
    ok=0
    for attempt in 1 2 3 4 5; do
      gate
      echo "$name seed $seed $(date '+%F %T')" > runs/week/GPU_BUSY
      if nice -n 10 .venv/bin/python sweep.py "$name" --seeds "$prefix" "$@" 2> >(tee $errlog >&2); then ok=1; rm -f runs/week/GPU_BUSY; break; fi
      rm -f runs/week/GPU_BUSY
      sleep 1  # let the stderr tee flush
      d=runs/sweeps/$name/seed$seed
      if [ "$prefix" = "$seed" ] && grep -q "needs at least 2 seeds, got 1" $errlog \
         && [ -e "$d/TRAINED" ] && [ -e "$d/$out/report.json" ]; then
        echo "sweep_retry: one-seed prefix $prefix trained and analysed; aggregate deferred to the next seed" >&2
        ok=1; break
      fi
      moved=0
      for d in runs/sweeps/$name/seed*; do
        if [ -d "$d" ] && [ ! -e "$d/TRAINED" ] && [ -e "$d/checkpoints/emergency.pt" ]; then
          local to=runs/week/aborted/${name}_$(basename $d)_$(date +%Y%m%d_%H%M%S)
          mv "$d" "$to"; moved=1
          echo "[$(date '+%F %T')] guardrail abort: moved $d -> $to (attempt $attempt)" >> runs/week/aborted.log
        fi
      done
      [ $moved = 1 ] || return 1
    done
    [ $ok = 1 ] || return 1
  done
}
