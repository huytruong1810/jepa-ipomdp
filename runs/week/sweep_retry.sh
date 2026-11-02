#!/bin/bash
# Sourced by week stages. sweep_retry <name> <sweep.py args...>: run sweep.py; if it fails because the VRAM guardrail
# aborted a seed (that seed dir has checkpoints/emergency.pt and no TRAINED marker; another process on the GPU, e.g. a
# foreign llama-server, pushed device VRAM over vram_trip_mb), move that seed dir to runs/week/aborted/, wait for the
# queue gate, and rerun (training is deterministic per seed, so a from-scratch rerun is the same run). Any other
# failure is returned unchanged (fail loudly). At most 4 retries per call; every move is logged to runs/week/aborted.log.
sweep_retry() {
  local name=$1; shift
  local attempt d moved
  mkdir -p runs/week/aborted
  for attempt in 1 2 3 4 5; do
    nice -n 10 .venv/bin/python sweep.py "$name" "$@" && return 0
    [ $attempt = 5 ] && return 1
    moved=0
    for d in runs/sweeps/$name/seed*; do
      if [ -d "$d" ] && [ ! -e "$d/TRAINED" ] && [ -e "$d/checkpoints/emergency.pt" ]; then
        local to=runs/week/aborted/${name}_$(basename $d)_$(date +%Y%m%d_%H%M%S)
        mv "$d" "$to"; moved=1
        echo "[$(date '+%F %T')] guardrail abort: moved $d -> $to (attempt $attempt)" >> runs/week/aborted.log
      fi
    done
    [ $moved = 1 ] || return 1
    gate
  done
}
