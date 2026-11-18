#!/bin/bash
# ABSOLUTE PATH: tools/fingerprint/gpu_run.sh
# Runs ONE hand-started jepa GPU step under the week queue's contract:
#     bash tools/fingerprint/gpu_run.sh "<label>" <command...>
# ==============================================================================
# DESIGN DECISIONS
# ==============================================================================
# 1. The Queue's Gate, Without Waiting:
#    - The machine is shared (other projects run multi-GB CPU jobs; the GPU belongs to this project, one job at a
#      time). The week queue (runs/week/queue.sh) gates its own GPU steps; a step started by hand must obey the same
#      contract or two GPU jobs collide. This script refuses (exit 3) instead of waiting when runs/week/PAUSE
#      exists, MemAvailable < 8000 MB, the GPU already holds >= 4000 MiB, or runs/week/GPU_BUSY names a live pid.
#    - Otherwise it writes runs/week/GPU_BUSY ("<label> pid=<pid> since=<time>"), so the queue and the other sessions
#      see the step, and removes it when the command ends, whatever its exit code.
# 2. Launch Settings:
#    - nice 5 (the GPU-driving thread, as agreed with the other projects), OMP_NUM_THREADS=2 unless set, and
#      CUDA_LAUNCH_BLOCKING=1 (an intermittent WSL driver fault on asynchronous launches; same kernels and numerics,
#      slower; JEPA_LAUNCH_BLOCKING=0 turns it off).
# ==============================================================================
cd "$(dirname "$0")/../.." || exit 1
W=runs/week; label=$1; shift
mem=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ -e $W/PAUSE ] && { echo "refused: PAUSE ($(head -1 $W/PAUSE))" >&2; exit 3; }
[ "$mem" -ge 8000 ] || { echo "refused: MemAvailable ${mem} MB < 8000" >&2; exit 3; }
[ "$gpu" -lt 4000 ] || { echo "refused: GPU has ${gpu} MiB in use" >&2; exit 3; }
if [ -e $W/GPU_BUSY ]; then
  bpid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' $W/GPU_BUSY | head -1)
  { [ -n "$bpid" ] && ! kill -0 "$bpid" 2>/dev/null; } || { echo "refused: GPU_BUSY ($(head -1 $W/GPU_BUSY))" >&2; exit 3; }
fi
echo "$label pid=$$ since=$(date '+%F %T')" > $W/GPU_BUSY
trap 'rm -f $W/GPU_BUSY' EXIT
CUDA_LAUNCH_BLOCKING=${JEPA_LAUNCH_BLOCKING:-1} OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} nice -n 5 "$@"
