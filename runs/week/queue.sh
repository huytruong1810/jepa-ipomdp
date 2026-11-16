#!/bin/bash
# Week queue: runs the stage scripts in runs/week/stages.d/ in lexical order, ONE at a time.
# Launch (detached, survives a reaped shell):
#   (setsid nohup bash runs/week/queue.sh > runs/week/queue_stdout.log 2>&1 < /dev/null &)
#
# Design decisions:
# - The directory is re-scanned after every stage, so stages can be added while the queue runs. A stage is done when
#   runs/week/done/<stage> exists; a failed stage is logged and retried only after its marker runs/week/failed/<stage>
#   is deleted by hand (fail loudly). The ONE exception is the intermittent WSL driver fault ("CUDA error: unknown
#   error", since 2026-11-14): a stage, or a gstep step, that fails with that text is rerun once after the gate. The
#   rerun is the same command under the same conditions, so no result can differ from a clean first run.
# - One queue only: an flock on runs/week/queue.lock (two queues once wrote into queue.log on 2026-10-31).
# - Gate (before every stage, and before every step through gate / gstep / sweep_retry). It waits while any holds:
#     runs/week/PAUSE exists (another project asks for a quiet machine; its one line says owner, since, until);
#     MemAvailable < 8000 MB; GPU memory in use >= 4000 MiB (a foreign GPU job; one Wumpus training uses ~3.6 GB,
#     so a hand-run jepa step is caught by the next rule instead); runs/week/GPU_BUSY names a live pid other than
#     the calling stage (a hand-run jepa GPU step). A GPU_BUSY whose pid is dead is stale and ignored.
#   While the gate waits on PAUSE, or the queue is idle under PAUSE, runs/week/PAUSED_ACK exists: no queue step is
#   running, so the owner of PAUSE can start its timing run. (PAUSE alone only stops NEW steps; a running seed can
#   take ~3 h.) The gate logs when a wait starts and how long it lasted.
# - Polling is 60 s (it was 30 min idle and 5 min in the gate: a new or re-armed stage started up to 33 min late).
# - Status for people and for the other projects' sessions: runs/week/STATUS is rewritten every minute (state,
#   stage, stage pid, since, expected end, the GPU step, PAUSE, memory, GPU, load). runs/week/machine.log gets one
#   line per minute for post-mortems (a crash on 2026-10-16 left no evidence). A "STALL?" line is logged when a
#   running stage has changed no file for 60 minutes.
# - A stage runs in its own session with SIGINT at its default disposition (a background job of a non-interactive
#   shell otherwise ignores SIGINT, and so would main.py). After TIMEOUT_HOURS (stage header, default 48) the queue
#   sends SIGINT to the stage's process group, so main.py writes latest.pt, then SIGTERM 10 minutes later.
# - Stage output is APPENDED to runs/week/logs_<stage>.out with a header per attempt (date, commit, clean or dirty
#   tree); the old truncation lost the evidence of failed attempts.
# - Every stage inherits CUDA_LAUNCH_BLOCKING=1 (same kernels and numerics, slower; three stages forgot it) and
#   JEPA_GPU_NICE=5 (the tree search is one Python thread; under CPU contention a seed took 9 h instead of 2.5 h;
#   agreed with the ipomcp session on 2026-11-15, whose non-timing workers run at nice 10).
# - After every stage the audit manifest is verified (sha256sum -c), so an overwritten pinned artifact is seen at once.
#
# Stage header lines the queue reads (optional):  # TIMEOUT_HOURS=<int>   # EXPECTED_HOURS=<number>
# Test hooks (environment): WEEK_DIR, QUEUE_POLL_SECONDS, QUEUE_HEARTBEAT_SECONDS, QUEUE_TIMEOUT_UNIT_SECONDS.
cd /home/andyj1810/projects/jepa_ipomdp || exit 1
W=${WEEK_DIR:-runs/week}
mkdir -p $W/done $W/failed $W/stages.d
exec 9> $W/queue.lock
flock -n 9 || { echo "[$(date '+%F %T')] another queue holds $W/queue.lock; not starting" >&2; exit 1; }
POLL=${QUEUE_POLL_SECONDS:-60}
export W POLL OMP_NUM_THREADS=2 CUDA_LAUNCH_BLOCKING=1 JEPA_GPU_NICE=5
MANIFEST=reports/2026-11-14_submission_audit/MANIFEST.sha256

qlog() { echo "[$(date '+%F %T')] $*" >> $W/queue.log; }

gate() {
  local t0 said=0 mem gpu why bpid
  t0=$(date +%s)
  while :; do
    mem=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    why=""
    if [ -e $W/PAUSE ]; then why="PAUSE"
    elif [ "$mem" -lt 8000 ]; then why="mem=${mem}MB"
    elif ! [[ "$gpu" =~ ^[0-9]+$ ]]; then why="nvidia-smi gave no number"
    elif [ "$gpu" -ge 4000 ]; then why="gpu=${gpu}MiB"
    elif [ -e $W/GPU_BUSY ]; then
      bpid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' $W/GPU_BUSY | head -1)
      if [ -n "$bpid" ]; then
        if [ "$bpid" != "$$" ] && kill -0 "$bpid" 2>/dev/null; then why="GPU_BUSY pid=$bpid"; fi
      elif [ -n "$(find $W/GPU_BUSY -mmin -360 2>/dev/null)" ]; then why="GPU_BUSY without a pid, younger than 6 h"
      fi
    fi
    if [ -e $W/PAUSE ]; then
      [ -e $W/PAUSED_ACK ] || echo "no jepa queue step is running since $(date '+%F %T')" > $W/PAUSED_ACK
    else
      rm -f $W/PAUSED_ACK
    fi
    if [ -z "$why" ]; then
      rm -f $W/.gate
      [ $said = 1 ] && qlog "gate passed after $(( ($(date +%s) - t0) / 60 )) min"
      return 0
    fi
    [ $said = 0 ] && { qlog "gate waiting ($why)"; said=1; }
    echo "$why" > $W/.gate
    sleep $POLL 9>&-
  done
}

# gstep <label> <logfile> <command...>: one GPU step of a stage. Gate, GPU_BUSY (label, the stage's pid, start time),
# the command at JEPA_GPU_NICE with its output appended to <logfile>, and one rerun if it failed with the driver fault.
gstep() {
  local label=$1 logf=$2 rc try
  shift 2
  for try in 1 2; do
    gate
    echo "$label pid=$$ since=$(date '+%F %T')" > $W/GPU_BUSY
    nice -n $JEPA_GPU_NICE "$@" >> "$logf" 2>&1
    rc=$?
    rm -f $W/GPU_BUSY
    qlog "step $label exit $rc (try $try)"
    [ $rc = 0 ] && return 0
    { [ $try = 1 ] && tail -c 20000 "$logf" | grep -q "CUDA error: unknown error"; } || return $rc
    qlog "step $label: CUDA unknown error (driver fault), one rerun"
  done
  return $rc
}
export -f gate gstep qlog

heartbeat() {
  local qpid=$1 n=0 stalled=0 st stage since exp spid mem gpu util load end busy
  exec 9>&-   # the heartbeat must not hold the queue's lock: a killed queue frees it at once
  while kill -0 $qpid 2>/dev/null; do
    mem=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    IFS=', ' read -r gpu util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    load=$(cut -d' ' -f1-3 /proc/loadavg)
    echo "$(date '+%F %T') mem_avail_mb=$mem gpu_used_mib=$gpu gpu_util_pct=$util load=$load" >> $W/machine.log
    IFS='|' read -r st stage since exp spid < $W/.state
    end=""
    # Epoch arithmetic: GNU date reads "<time> + 42 minutes" as a time-zone offset and fails.
    [ -n "$exp" ] && [ -n "$since" ] && end=$(date -d "@$(( $(date -d "$since" +%s) + $(awk -v h="$exp" 'BEGIN {print int(h * 3600)}') ))" '+%F %T' 2>/dev/null)
    busy=none; [ -e $W/GPU_BUSY ] && busy=$(head -1 $W/GPU_BUSY)
    {
      echo "project=jepa_ipomdp"
      echo "updated=$(date '+%F %T')"
      echo "queue_pid=$qpid"
      echo "state=$st"
      echo "stage=${stage:-none}"
      echo "stage_pid=${spid:-none}"
      echo "since=${since:-none}"
      echo "expected_end=${end:-unknown}"
      echo "gate_waiting_on=$([ -e $W/.gate ] && head -1 $W/.gate || echo nothing)"
      echo "gpu_step=$busy"
      echo "pause=$([ -e $W/PAUSE ] && echo "yes ($(head -1 $W/PAUSE))" || echo no)"
      echo "paused_ack=$([ -e $W/PAUSED_ACK ] && echo yes || echo no)"
      echo "mem_available_mb=$mem"
      echo "gpu_used_mib=$gpu"
      echo "gpu_util_pct=$util"
      echo "load=$load"
      echo "contract=PAUSE stops new jepa steps; PAUSED_ACK means none is running; GPU_BUSY names the jepa GPU step; only jepa writes STATUS, GPU_BUSY, PAUSED_ACK"
    } > $W/STATUS.tmp && mv $W/STATUS.tmp $W/STATUS
    n=$((n + 1))
    if [ "$st" = running ] && [ $((n % 10)) = 0 ] && [ ! -e $W/.gate ]; then
      if [ -z "$(find runs reports -path "$W" -prune -o -type f -mmin -60 -print -quit 2>/dev/null)" ] \
         && [ -z "$(find $W/logs_$stage.out -mmin -60 2>/dev/null)" ]; then
        [ $stalled = 0 ] && { qlog "STALL? stage $stage has changed no file for 60 min"; stalled=1; }
      else
        stalled=0
      fi
    fi
    sleep ${QUEUE_HEARTBEAT_SECONDS:-60}
  done
}

set_state() { echo "$1" > $W/.state; }

set_state "idle||||"
rm -f $W/.gate
heartbeat $$ &
qlog "queue start, pid $$, commit $(git log -1 --format=%h), poll ${POLL}s"
idle_said=0
while :; do
  next=""
  for f in $(ls $W/stages.d/*.sh 2>/dev/null | sort); do
    s=$(basename $f .sh)
    [ -e $W/done/$s ] || [ -e $W/failed/$s ] || { next=$f; break; }
  done
  if [ -z "$next" ]; then
    [ $idle_said = 0 ] && { qlog "no pending stage; idle"; idle_said=1; }
    set_state "idle||||"
    if [ -e $W/PAUSE ]; then
      [ -e $W/PAUSED_ACK ] || echo "no jepa queue step is running since $(date '+%F %T')" > $W/PAUSED_ACK
    else
      rm -f $W/PAUSED_ACK
    fi
    sleep $POLL 9>&-
    continue
  fi
  idle_said=0
  s=$(basename $next .sh)
  tmo=$(sed -n 's/^# TIMEOUT_HOURS=\([0-9][0-9]*\).*/\1/p' $next | head -1); tmo=${tmo:-48}
  exp=$(sed -n 's/^# EXPECTED_HOURS=\([0-9.][0-9.]*\).*/\1/p' $next | head -1)
  for attempt in 1 2; do
    set_state "gate_wait|$s|$(date '+%F %T')||"
    gate
    head=$(git log -1 --format=%h)
    dirty=$(git diff --quiet HEAD -- src conf reports main.py sweep.py analyze.py && echo clean || echo dirty)
    off=$(stat -c %s $W/logs_$s.out 2>/dev/null || echo 0)
    echo "===== $s attempt $attempt, $(date '+%F %T'), commit $head ($dirty tree) =====" >> $W/logs_$s.out
    qlog "stage $s START attempt $attempt (commit $head, $dirty tree, timeout ${tmo} h)"
    setsid env --default-signal=INT bash $next >> $W/logs_$s.out 2>&1 &
    spid=$!
    set_state "running|$s|$(date '+%F %T')|$exp|$spid"
    deadline=$(( $(date +%s) + tmo * ${QUEUE_TIMEOUT_UNIT_SECONDS:-3600} ))
    timed_out=0
    while kill -0 $spid 2>/dev/null; do
      sleep 5 9>&-
      now=$(date +%s)
      if [ $timed_out = 0 ] && [ $now -ge $deadline ]; then
        timed_out=1; kill_at=$((now + 600))
        qlog "stage $s TIMEOUT after ${tmo} h: SIGINT to its process group"
        kill -INT -- -$spid 2>/dev/null
      elif [ $timed_out = 1 ] && [ $now -ge $kill_at ]; then
        timed_out=2
        qlog "stage $s still alive 10 min after SIGINT: SIGTERM"
        kill -TERM -- -$spid 2>/dev/null
      fi
    done
    wait $spid
    rc=$?
    if [ -e $W/GPU_BUSY ]; then
      bpid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' $W/GPU_BUSY | head -1)
      { [ -n "$bpid" ] && kill -0 "$bpid" 2>/dev/null; } || { qlog "removing the GPU_BUSY that stage $s left behind"; rm -f $W/GPU_BUSY; }
    fi
    if [ $rc = 0 ]; then touch $W/done/$s; qlog "stage $s DONE"; break; fi
    if [ $attempt = 1 ] && [ $timed_out = 0 ] && tail -c +$((off + 1)) $W/logs_$s.out | grep -q "CUDA error: unknown error"; then
      qlog "stage $s failed rc=$rc with the CUDA unknown error (driver fault): one automatic rerun"
      continue
    fi
    touch $W/failed/$s
    qlog "stage $s FAILED rc=$rc$([ $timed_out != 0 ] && echo ' (timeout)')"
    break
  done
  if [ -f $MANIFEST ]; then
    bad=$(sha256sum -c --quiet $MANIFEST 2>&1 | grep -c -v '^sha256sum: WARNING')
    [ "$bad" = 0 ] && qlog "manifest OK after $s" || qlog "MANIFEST MISMATCH after $s: $bad lines (sha256sum -c --quiet $MANIFEST)"
  fi
done
