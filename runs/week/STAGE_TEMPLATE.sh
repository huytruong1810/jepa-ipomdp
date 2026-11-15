#!/bin/bash
# Template for a week-queue stage (copy to runs/week/stages.d/<prefix>_<name>.sh; stages run in lexical order).
# One-paragraph purpose, the PROTOCOL.md registration it belongs to, and what a failure means.
#
# Header lines the queue reads:
# TIMEOUT_HOURS=24      (default 48; after it the queue sends SIGINT to the stage's process group)
# EXPECTED_HOURS=3      (shown as expected_end in runs/week/STATUS for the other projects' sessions)
#
# Conventions (queue.sh header has the reasons):
# - `set -o pipefail`, and no `| tail` after a command whose exit code matters: two old stages reported DONE on a
#   failed pipe.
# - The queue exports CUDA_LAUNCH_BLOCKING=1, JEPA_GPU_NICE, OMP_NUM_THREADS=2, and the functions gate, gstep, qlog.
# - Every GPU step goes through `gstep <label> <logfile> <command...>`: gate, GPU_BUSY with the stage pid, the
#   command at JEPA_GPU_NICE, one rerun on the driver fault. Sweeps go through sweep_retry (runs/week/sweep_retry.sh).
# - CPU-only steps: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 <command>`.
# - Resumable: skip a step when its `.done` marker exists, and touch the marker only after the step succeeded
#   (never skip on the output file itself: a reboot mid-write leaves a truncated file).
# - A smoke goes first. Print `commit $(git log -1 --format=%H)` into the stage's own log.
cd /home/andyj1810/projects/jepa_ipomdp || exit 1
set -o pipefail
O=runs/wumpus/<name>
mkdir -p $O
echo "commit $(git log -1 --format=%H)"

step() {  # step <marker> <label> <command...>
  local marker=$O/$1.done label=$2; shift 2
  [ -e $marker ] && return 0
  OMP_NUM_THREADS=4 gstep "<stage> $label" $O/$label.log "$@" || exit 1
  touch $marker
}

# step smoke   "smoke"   .venv/bin/python <script> <small arguments>
# step seed0   "seed 0"  .venv/bin/python <script> <arguments>
