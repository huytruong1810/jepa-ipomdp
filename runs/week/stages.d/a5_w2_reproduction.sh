#!/bin/bash
# Submission audit W2: reproduction check. (i) Held-out recipe seed 5 re-analysed (best.pt, analysis seed 20261019,
# the sweep's settings) into analysis_repro_w2 and its games.npz compared array by array, bit for bit, with
# analysis_heldout/games.npz. (ii) Tiger polyak seed 0: config and best.pt copied to a scratch run dir (analyze.py has
# no output option for Tiger, so the stored analysis is never overwritten), re-analysed with the sweep's settings,
# report.json returns compared exactly. Comparison: reports/2026-11-14_submission_audit/repro_check.py.
cd /home/andyj1810/projects/jepa_ipomdp
L=reports/2026-11-14_submission_audit/logs/reproduction_$(date +%F).log
W=runs/sweeps/wumpus_retrain_heldout/seed5
T=runs/repro_w2/tiger_polyak_seed0
mkdir -p $T/.hydra $T/checkpoints
cp runs/sweeps/polyak/seed0/.hydra/config.yaml $T/.hydra/config.yaml
cp runs/sweeps/polyak/seed0/checkpoints/best.pt $T/checkpoints/best.pt
gate
echo "W2 reproduction $(date '+%F %T')" > runs/week/GPU_BUSY
{ echo "commit $(git log -1 --format=%H)  started $(date '+%F %T')"
  [ -e $W/analysis_repro_w2/games.npz ] || OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python analyze.py $W --seed 20261019 \
    --analysis-output analysis_repro_w2 || exit 1
  [ -e $T/analysis/report.json ] || OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python analyze.py $T --seed 12345 || exit 1
  .venv/bin/python reports/2026-11-14_submission_audit/repro_check.py $W/analysis_heldout/games.npz \
    $W/analysis_repro_w2/games.npz runs/sweeps/polyak/seed0/analysis/report.json $T/analysis/report.json
  rc=$?; echo "finished $(date '+%F %T') rc=$rc"; exit $rc; } > $L 2>&1
rc=$?
rm -f runs/week/GPU_BUSY
exit $rc
