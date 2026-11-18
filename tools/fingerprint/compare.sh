#!/bin/bash
# ABSOLUTE PATH: tools/fingerprint/compare.sh
# Compares the working tree with a reference commit on fingerprint cases:
#     bash tools/fingerprint/compare.sh <commit> <case|cpu|options> [--device cuda] [--run-dir <stored recipe run>]
# Prints "FINGERPRINT IDENTICAL <case>" per case and exits 0 only when every case is byte-equal and non-empty.
# ==============================================================================
# DESIGN DECISIONS
# ==============================================================================
# 1. One Harness, Two Checkouts:
#    - The reference side runs THIS tree's tools/fingerprint/fingerprint.py against a git worktree of <commit>: the
#      worktree's src/ is first on PYTHONPATH and --code-root composes the worktree's conf/. So both sides run the
#      same cases with the same harness and differ only in the code under test.
#    - Reference outputs are cached per commit, device and case in runs/fingerprints/ref/<commit>/<device>/, so a
#      gate that compares many edits against one base commit computes the base once. The worktree is removed after
#      use.
# 2. GPU Cases Go Through the Queue's Gate:
#    - With --device cuda each side runs under tools/fingerprint/gpu_run.sh (PAUSE, memory, GPU_BUSY contract).
#    - CPU cases are also covered by `pytest -m golden` (tests/golden/fingerprints); this script is the tool for the
#      GPU cases and for comparing against any commit other than the golden one.
# 3. Never "Identical" on Nothing:
#    - Both outputs must be non-empty and byte-equal (an earlier check once passed on two empty files).
# ==============================================================================
set -u
cd "$(dirname "$0")/../.." || exit 1
commit=$(git rev-parse --short "$1") || exit 2; selector=$2; shift 2
device=cpu; extra=()
while [ $# -gt 0 ]; do
  case $1 in --device) device=$2; shift 2;; --run-dir) extra+=(--run-dir "$(realpath "$2")"); shift 2;; *) echo "unknown argument $1"; exit 2;; esac
done
py=.venv/bin/python; fp=tools/fingerprint/fingerprint.py
case $selector in
  cpu) cases=$($py $fp --list | grep -v '^wumpus_gpu$');;
  options) cases=$($py $fp --list | grep -v -E '^(tiger|dqn|wumpus_gpu)$');;
  *) cases=$selector;;
esac
run() {  # run <label> <command...>: through the GPU gate on cuda, at nice 19 on one thread otherwise
  local label=$1; shift
  if [ "$device" = cuda ]; then bash tools/fingerprint/gpu_run.sh "$label" "$@"; else env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 "$@"; fi
}
ref=runs/fingerprints/ref/$commit/$device; new=runs/fingerprints/new/$device; mkdir -p "$ref" "$new"
missing=""; for c in $cases; do [ -s "$ref/$c.txt" ] || missing="$missing $c"; done
if [ -n "$missing" ]; then
  tmp=$(mktemp -d); wt=$tmp/wt
  git worktree add -q "$wt" "$commit" || exit 1
  for c in $missing; do
    run "fingerprint $c @$commit" env PYTHONPATH="$wt/src" $py $fp --case "$c" --code-root "$wt" --device "$device" "${extra[@]}" \
      > "$ref/$c.part" 2> "$ref/$c.err" && mv "$ref/$c.part" "$ref/$c.txt" || { echo "reference side of $c failed: $ref/$c.err"; git worktree remove --force "$wt"; rm -rf "$tmp"; exit 1; }
  done
  git worktree remove --force "$wt"; rm -rf "$tmp"
fi
status=0
for c in $cases; do
  run "fingerprint $c working tree" env PYTHONPATH="$PWD/src" $py $fp --case "$c" --device "$device" "${extra[@]}" > "$new/$c.txt" 2> "$new/$c.err" \
    || { echo "working-tree side of $c failed: $new/$c.err"; status=1; continue; }
  if [ -s "$new/$c.txt" ] && cmp -s "$new/$c.txt" "$ref/$c.txt"; then echo "FINGERPRINT IDENTICAL $c"; else
    echo "FINGERPRINT DIFFERS $c: diff $new/$c.txt $ref/$c.txt"; status=1; fi
done
exit $status
