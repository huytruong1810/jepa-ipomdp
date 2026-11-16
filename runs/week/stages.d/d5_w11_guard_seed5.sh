#!/bin/bash
# W11 guard (WORK_ORDERS.md, order W11 item 1), run now because the tree search was restructured (c454e1b): train the
# frozen recipe on held-out seed 5 at HEAD for the first 5 collections and require the same numbers as the stored run
# runs/sweeps/wumpus_retrain_heldout/seed5. Checked: the main.log line of collection 5 (the order's definition) and,
# stricter, every Train/* scalar of collections 1-5 and every Eval/* scalar of collection 5 in TensorBoard (float32,
# exact). The stored run used asynchronous launches; this one runs with CUDA_LAUNCH_BLOCKING=1, so the guard also
# checks that the launch mode leaves the numbers unchanged. A mismatch fails the stage: stop and report.
# TIMEOUT_HOURS=3
# EXPECTED_HOURS=0.7
cd /home/andyj1810/projects/jepa_ipomdp || exit 1
set -o pipefail
O=runs/wumpus/w11_guard
R=$O/seed5
mkdir -p $O
echo "commit $(git log -1 --format=%H)"
rm -rf $R
RUN='env --default-signal=INT .venv/bin/python main.py seed=5 hydra.run.dir='$R' env=wumpus model.representation=decoder model.observation_horizon=5 training.updates_per_collection=64 training.total_episodes=6400 training.store_eval_episodes=true agent.epsilon=0.25 agent.epsilon_min=0.1 model.value_estimator=double training.retrain_every=50 training.retrain_updates=6400'
# Run until the collection-5 evaluation is logged, then SIGINT (main.py saves latest.pt and exits 130).
OMP_NUM_THREADS=4 gstep "W11 guard seed 5" $O/train.log bash -c "$RUN & pid=\$!; while ! grep -q 'collection 5:' $R/main.log 2>/dev/null; do kill -0 \$pid 2>/dev/null || exit 1; sleep 10; done; kill -INT \$pid; wait \$pid; [ \$? = 130 ]" || exit 1
CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python - $R <<'PY' > $O/compare.log 2>&1 || { cat $O/compare.log; exit 1; }
import glob, re, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
new, old = sys.argv[1], "runs/sweeps/wumpus_retrain_heldout/seed5"
line = lambda d: re.search(r"collection 5: (.*)", open(f"{d}/main.log").read()).group(1)
ok = line(new) == line(old)
print("main.log collection 5:", "EQUAL" if ok else "DIFFERENT"); print("  stored:", line(old)); print("  new:   ", line(new))
def scalars(d):
    ea = EventAccumulator(glob.glob(f"{d}/tensorboard/*")[0], size_guidance={"scalars": 0}); ea.Reload()
    return {t: {e.step: e.value for e in ea.Scalars(t) if e.step <= 5} for t in ea.Tags()["scalars"] if t.startswith(("Train/", "Eval/"))}
a, b = scalars(new), scalars(old)
bad = [(t, s, b[t][s], a.get(t, {}).get(s)) for t in b for s in b[t] if a.get(t, {}).get(s) != b[t][s]]
print(f"TensorBoard Train/* and Eval/* through collection 5: {sum(len(v) for v in b.values())} stored values, {len(bad)} differ")
for row in bad[:20]: print("  ", row)
print("GUARD " + ("PASSED: the first 5 collections reproduce the stored seed 5" if ok and not bad else "FAILED"))
raise SystemExit(0 if ok and not bad else 1)
PY
cat $O/compare.log
