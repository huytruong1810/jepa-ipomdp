#!/bin/bash
# Submission audit W4: accepted-state control of the localisation (descriptive). q_localise_diag.py section 5 and
# feature_localise.py section 7 (--states accepted|all) on development seeds 0-4, best.pt, 512 games.
# Step 0 (identity): the default (declined) output of both scripts is unchanged by the option: run the pre-change
# scripts (commit dbae0cb, a git worktree) and the current ones on 8 games of a scratch copy of seed 0 and require
# equal outputs; otherwise the stage stops. Summary: reports/2026-11-14_submission_audit/w4_accepted_states.py.
cd /home/andyj1810/projects/jepa_ipomdp
export CUDA_LAUNCH_BLOCKING=1   # WSL GPU driver faults on async launches since 2026-11-14 18:30; same numerics
D=reports/2026-10-02_wumpus_diagnostics
L=reports/2026-11-14_submission_audit/logs
O=runs/wumpus/w4
mkdir -p $O
step() {
  local label=$1; shift
  gate
  echo "W4 $label $(date '+%F %T')" > runs/week/GPU_BUSY
  OMP_NUM_THREADS=4 nice -n 10 "$@"
  local rc=$?
  rm -f runs/week/GPU_BUSY
  echo "[$(date '+%F %T')] $label exit $rc" >> $O/chain.log
  [ $rc = 0 ] || exit 1
}
if [ ! -f $O/identity.done ]; then
  W=/tmp/w4_identity_wt
  rm -rf $W; git worktree add -q $W dbae0cb || exit 1
  for ver in old new; do
    R=runs/w4_identity/$ver/seed0
    rm -rf $R; mkdir -p $R/analysis
    ln -s $PWD/runs/sweeps/wumpus_retrain/seed0/.hydra $R/.hydra
    ln -s $PWD/runs/sweeps/wumpus_retrain/seed0/checkpoints $R/checkpoints
    src=$D; [ $ver = old ] && src=$W/$D
    step "identity $ver q_localise" env PYTHONPATH=$([ $ver = old ] && echo $W/src || echo $PWD/src) .venv/bin/python \
      $src/q_localise_diag.py $R 8 > $O/identity_q_$ver.log 2>&1
    step "identity $ver feature_localise" env PYTHONPATH=$([ $ver = old ] && echo $W/src || echo $PWD/src) .venv/bin/python \
      $src/feature_localise.py $R cuda 8 > $O/identity_fl_$ver.log 2>&1
  done
  git worktree remove --force $W
  .venv/bin/python - <<'PY' > $L/w4_identity_$(date +%F).log 2>&1 || exit 1
import json, numpy as np, torch
a, b = "runs/w4_identity/old/seed0/analysis", "runs/w4_identity/new/seed0/analysis"
ok = True
for f in ("q_localise.npz", "feature_localise_smoke.npz"):
    x, y = dict(np.load(f"{a}/{f}", allow_pickle=True)), dict(np.load(f"{b}/{f}", allow_pickle=True))
    same = set(x) == set(y) and all(np.array_equal(x[k], y[k]) for k in x)
    ok &= same; print(f"{f}: {'EQUAL' if same else 'DIFFER'}")
x, y = json.load(open(f"{a}/feature_localise_smoke.json")), json.load(open(f"{b}/feature_localise_smoke.json"))
x.pop("networks", None); y.pop("networks", None)
ok &= x == y; print(f"feature_localise_smoke.json: {'EQUAL' if x == y else 'DIFFER'}")
p, q = torch.load(f"{a}/declined_states_smoke.pt", weights_only=False), torch.load(f"{b}/declined_states_smoke.pt", weights_only=False)
same = all(torch.equal(p[k], q[k]) for k in p if isinstance(p[k], torch.Tensor)) and p["meta"] == q["meta"]
ok &= same; print(f"declined_states_smoke.pt: {'EQUAL' if same else 'DIFFER'}")
print("IDENTITY " + ("OK: the default output is unchanged" if ok else "FAILED")); raise SystemExit(0 if ok else 1)
PY
  touch $O/identity.done
fi
for s in 0 1 2 3 4; do
  R=runs/sweeps/wumpus_retrain/seed$s
  for st in accepted all; do
    [ -f $R/analysis/q_localise_$st.npz ] || step "seed $s q_localise $st" .venv/bin/python $D/q_localise_diag.py $R 512 \
      --states $st > $O/q_${st}_seed$s.log 2>&1
    [ -f $R/analysis/feature_localise_$st.json ] || step "seed $s feature_localise $st" .venv/bin/python $D/feature_localise.py \
      $R cuda --states $st > $O/fl_${st}_seed$s.log 2>&1
  done
done
.venv/bin/python reports/2026-11-14_submission_audit/w4_accepted_states.py > $L/w4_accepted_states_$(date +%F).log 2>&1
