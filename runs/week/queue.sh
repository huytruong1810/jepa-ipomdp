#!/bin/bash
# Week queue (2026-10-31 -> 11-07): runs the stage scripts in runs/week/stages.d/ in lexical order, ONE at a time.
# Design decisions:
# - The directory is re-scanned after every stage, so stages can be added (or conditional ones dropped) while the
#   queue runs, without restarting it. A stage is done when runs/week/done/<stage> exists; a failed stage is logged
#   and retried only after its marker file runs/week/failed/<stage> is deleted by hand (fail loudly, no auto-retry).
# - Before every stage (and inside long stages, via gate()) the queue waits until: no runs/week/PAUSE file (another
#   project's agent may request a large-RAM window), MemAvailable >= 8000 MB (WSL has 16 GB total; ipomcp and the
#   Blender agents share it), GPU memory in use < 4000 MiB (no foreign GPU job), and no ipomcp quiet-machine rerun job
#   running (a python process with "final_20261108/" in its arguments other than the always-on machine_log.py; that
#   campaign needs an otherwise idle machine).
# - Detached with setsid nohup so a reaped Claude Code shell cannot kill it; each stage is resumable.
cd /home/andyj1810/projects/jepa_ipomdp
W=runs/week
mkdir -p $W/done $W/failed
export OMP_NUM_THREADS=2
log() { echo "[$(date '+%F %T')] $*" >> $W/queue.log; }
gate() {
  local said=0
  while :; do
    local mem gpu
    mem=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    local rfinal=n
    pgrep -af "final_20261108/" | grep -v -e machine_log -e "bash -c" | grep -q python && rfinal=y
    if [ ! -e $W/PAUSE ] && [ "$mem" -ge 8000 ] && [ "$gpu" -lt 4000 ] && [ $rfinal = n ]; then return 0; fi
    [ $said = 0 ] && { log "gate waiting (PAUSE=$([ -e $W/PAUSE ] && echo y || echo n) mem=${mem}MB gpu=${gpu}MiB rfinal=$rfinal)"; said=1; }
    sleep 300
  done
}
export -f gate log
export W
log "queue start, commit $(git log -1 --format=%h)"
while :; do
  next=""
  for f in $(ls $W/stages.d/*.sh 2>/dev/null | sort); do
    s=$(basename $f .sh)
    [ -e $W/done/$s ] || [ -e $W/failed/$s ] || { next=$f; break; }
  done
  [ -z "$next" ] && { log "no pending stage; idle 30 min"; sleep 1800; continue; }
  s=$(basename $next .sh)
  gate
  log "stage $s START"
  bash $next > $W/logs_$s.out 2>&1
  rc=$?
  if [ $rc = 0 ]; then touch $W/done/$s; log "stage $s DONE"; else touch $W/failed/$s; log "stage $s FAILED rc=$rc"; fi
done
