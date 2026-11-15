#!/bin/bash
# Submission audit W8: Tiger polyak and decoder seeds 0-4 were analysed before a1d7696 (the unroll hoist, 2026-10-04);
# seeds 5-9 after. Re-analyse seeds 0-4 at HEAD with the same analysis seed 12345 (W2b showed two fresh analyses agree
# bit for bit). Each old report is kept as analysis/report_pre_hoist.json. Guard: returns.optimal and
# returns.learned_planner must be unchanged for every seed; otherwise the old report is restored and the stage stops.
# Then sweep.py regenerates each sweep's aggregate.json over seeds 0-9 (no training: every report exists).
cd /home/andyj1810/projects/jepa_ipomdp
export CUDA_LAUNCH_BLOCKING=1   # WSL GPU driver faults on async launches since 2026-11-14 18:30; same numerics
L=reports/2026-11-14_submission_audit/logs/w8_tiger_reanalysis_$(date +%F).log
{
echo "commit $(git log -1 --format=%H)"
for sweep in polyak decoder; do
  for s in 0 1 2 3 4; do
    R=runs/sweeps/$sweep/seed$s
    [ -f $R/analysis/report_pre_hoist.json ] && [ -f $R/analysis/.w8_done ] && continue
    [ -f $R/analysis/report_pre_hoist.json ] || cp $R/analysis/report.json $R/analysis/report_pre_hoist.json
    gate
    echo "W8 $sweep seed $s $(date '+%F %T')" > runs/week/GPU_BUSY
    OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python analyze.py $R --seed 12345 > $R/analysis/w8_analyze.log 2>&1
    rc=$?
    rm -f runs/week/GPU_BUSY
    [ $rc = 0 ] || { cp $R/analysis/report_pre_hoist.json $R/analysis/report.json; echo "$sweep seed $s analysis failed rc=$rc; old report restored"; exit 1; }
    .venv/bin/python - $R <<'PY' || { cp $R/analysis/report_pre_hoist.json $R/analysis/report.json; echo "GUARD FAILED; old report restored"; exit 1; }
import json, sys
r = sys.argv[1]
old, new = json.load(open(f"{r}/analysis/report_pre_hoist.json")), json.load(open(f"{r}/analysis/report.json"))
for k in ("optimal", "learned_planner"):
    if old["returns"][k] != new["returns"][k]:
        print(f"{r}: returns.{k} changed: {old['returns'][k]} -> {new['returns'][k]}"); raise SystemExit(1)
changed = {k: (old["returns"][k]["mean"], new["returns"][k]["mean"]) for k in old["returns"] if old["returns"][k] != new["returns"][k]}
print(f"{r}: optimal and learned_planner unchanged; changed returns {changed}")
PY
    touch $R/analysis/.w8_done
  done
  overrides=""; [ $sweep = decoder ] && overrides="model.representation=decoder"
  cp runs/sweeps/$sweep/aggregate.json runs/sweeps/$sweep/aggregate_pre_hoist.json 2>/dev/null
  OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python sweep.py $sweep --seeds 0,1,2,3,4,5,6,7,8,9 $overrides 2>&1 | tail -30
done
echo "ledger:"; .venv/bin/python reports/2026-11-14_submission_audit/ledger.py 2>&1 | tail -30
} > $L 2>&1
