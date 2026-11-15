#!/bin/bash
# Submission audit W2 follow-up: is the Tiger decoded-return mismatch (decoded_linear 15.4047 vs stored 15.3848,
# decoded_mlp 18.6660 vs 18.6559; optimal and learned_planner equal) drift since the stored analysis or
# nondeterminism? Analyse the same copy of polyak seed 0 a second time at HEAD (fresh output: a second copy dir) and
# compare the two fresh reports with each other, then each with the stored one. Stored analyses are not touched.
cd /home/andyj1810/projects/jepa_ipomdp
L=reports/2026-11-14_submission_audit/logs/reproduction_tiger_twice_$(date +%F).log
A=runs/repro_w2/tiger_polyak_seed0
B=runs/repro_w2/tiger_polyak_seed0_run2
mkdir -p $B/.hydra $B/checkpoints
cp $A/.hydra/config.yaml $B/.hydra/config.yaml
cp $A/checkpoints/best.pt $B/checkpoints/best.pt
gate
echo "W2b tiger twice $(date '+%F %T')" > runs/week/GPU_BUSY
{ echo "commit $(git log -1 --format=%H)  started $(date '+%F %T')"
  [ -e $B/analysis/report.json ] || OMP_NUM_THREADS=4 nice -n 10 .venv/bin/python analyze.py $B --seed 12345 || exit 1
  .venv/bin/python - <<'PY'
import json
fresh1 = json.load(open("runs/repro_w2/tiger_polyak_seed0/analysis/report.json"))
fresh2 = json.load(open("runs/repro_w2/tiger_polyak_seed0_run2/analysis/report.json"))
stored = json.load(open("runs/sweeps/polyak/seed0/analysis/report.json"))
for name, other in (("fresh run 2", fresh2), ("stored", stored)):
    for k in sorted(fresh1["returns"]):
        a, b = fresh1["returns"][k], other["returns"][k]
        print(f"  returns.{k}: fresh run 1 {a['mean']:.10f} vs {name} {b['mean']:.10f} -> {'EQUAL' if a == b else 'DIFFER'}")
    for k in sorted(fresh1["probes"]):
        a, b = fresh1["probes"][k], other["probes"][k]
        print(f"  probes.{k}: {'EQUAL' if a == b else 'DIFFER'} (mean L1 {a.get('mean_l1')} vs {b.get('mean_l1')})")
same = fresh1["returns"] == fresh2["returns"] and fresh1["probes"] == fresh2["probes"]
print("VERDICT: two fresh runs at HEAD " + ("AGREE bit for bit -> the mismatch is drift since the stored analysis"
                                            if same else "DIFFER -> nondeterminism in the analysis"))
PY
  echo "finished $(date '+%F %T')"; } > $L 2>&1
rc=$?
rm -f runs/week/GPU_BUSY
exit $rc
