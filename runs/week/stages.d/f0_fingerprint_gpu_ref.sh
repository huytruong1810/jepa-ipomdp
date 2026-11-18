#!/bin/bash
# Code audit 2026-11-16, phase 0: GPU reference fingerprints at 5bea8c3 (reports/2026-11-16_code_audit/REVIEW.md).
# The trainer's forward passes run under bf16 autocast only on CUDA, so the CPU golden files cannot see a refactor that
# moves a loss term out of the autocast region. This stage records, from a git worktree of the reference commit, the
# CUDA output of every fingerprint case (Tiger, the baseline, the Wumpus base and recipe, one case per remedy option)
# and of the wumpus_gpu case (two recipe collections, an evaluation and a forced retrain from the stored state of
# development seed 0). Later stages compare the refactored code against these files. The wumpus_gpu output is also
# compared with the earlier tool's reference for c454e1b (src unchanged since), which validates the new harness on the
# GPU path. No registered quantity; a failure means a harness or machine problem, not a result.
# TIMEOUT_HOURS=3
# EXPECTED_HOURS=0.7
cd /home/andyj1810/projects/jepa_ipomdp || exit 1
set -o pipefail
REF=5bea8c3
O=runs/fingerprints/ref/$REF/cuda
mkdir -p $O
echo "commit $(git log -1 --format=%H)"
WT=runs/fingerprints/wt_$REF
[ -d $WT ] || git worktree add -q $WT $REF || exit 1
FP="env PYTHONPATH=$PWD/$WT/src .venv/bin/python tools/fingerprint/fingerprint.py --code-root $WT --device cuda"
step() {  # step <name> <label> <command...>: skipped when <name>.done exists
  local name=$1 label=$2; shift 2
  [ -e $O/$name.done ] && return 0
  OMP_NUM_THREADS=2 gstep "f0 $label" $O/$name.log "$@" || exit 1
  touch $O/$name.done
}
step smoke  "smoke tiger"     $FP --case tiger --out $O/smoke
step cases  "all cases"       $FP --case cpu --out $O
step wumpus "wumpus_gpu"      bash -c "$FP --case wumpus_gpu --run-dir runs/sweeps/wumpus_retrain/seed0 > $O/wumpus_gpu.txt"
git worktree remove --force $WT
cmp $O/smoke/tiger.txt $O/tiger.txt || { echo "tiger differs between two runs of the same code on the GPU"; exit 1; }
if cmp -s $O/wumpus_gpu.txt runs/fingerprints/wumpus_c454e1b.txt; then echo "wumpus_gpu equals the c454e1b reference of the earlier tool"
else echo "wumpus_gpu differs from the earlier tool's c454e1b reference: diff $O/wumpus_gpu.txt runs/fingerprints/wumpus_c454e1b.txt"; exit 1; fi
ls $O/*.txt | wc -l
