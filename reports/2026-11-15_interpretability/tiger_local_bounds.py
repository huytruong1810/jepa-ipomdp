# ABSOLUTE PATH: reports/2026-11-15_interpretability/tiger_local_bounds.py
# ==============================================================================
# C-q.4: THE LOCAL FORM OF THE SPAN BOUNDS ON THE CANONICAL TIGER (POST HOC; STORED RUNS, CPU)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Why, and Why Post Hoc:
#    - The error bounds of interpretability/error_bounds.py use one constant for a whole value function: the largest
#      half-span of its vectors. Section 5 of that module (2026-11-23) adds the local form: the same lemma applied to
#      the vectors that are active at the exact and at the decoded belief. It holds pair by pair and is never above
#      the global form.
#    - It was derived while the Interactive Tiger experiment was prepared, after every canonical Tiger number had
#      been read, and the stored Tiger reports hold the global form only. This script measures the local form on the
#      canonical Tiger. It is post hoc and is reported as such (PROTOCOL.md, C-q.4); for the Interactive Tiger the
#      same quantities are registered before any run.
#
# 2. Method:
#    - The ten canonical JEPA runs runs/sweeps/polyak/seed0..9, best.pt, the acting (Polyak) copies, loaded strictly
#      (experiments/runs.py, load_trained_run). For each, interpretability.analyze_beliefs is called exactly as
#      analyze_run calls it: the certified solution at tolerance 0.01 (one solve for all seeds), the greedy planner
#      at the run's budget, 512 episodes, 3,000 probe steps, analysis seed 12345.
#    - Nothing is written into a run directory. One JSON per seed goes to runs/interpretability/tiger_local_bounds/,
#      written atomically; a seed whose file exists is skipped, so an interrupted run continues.
#    - On the CPU. The stored reports were made on the GPU, and a simulator's generator lives on its device
#      (domain/env.py), so the same analysis seed draws other episodes here. The global values of this script are
#      therefore a replication on other episodes of the stored ones, not a recomputation; both are printed.
#
# 3. Quantities (`readout`, once, after the ten seeds; the probe fitted and scored on the agent's own histories):
#    - per seed, and as mean with a Student-t 95% interval over the ten seeds: the mean L1 decoding error; the
#      half-span of the vector active at the exact belief against L_V; the measured value error, its local bound and
#      its global bound; the same three for the one-step regret; the two policy forms; the ratio local / global.
#    - No pass rule. That the local form is not above the global one is a theorem, tested in
#      tests/test_interpretability.py; how much tighter it is on the Tiger is what this measures.
#
# 4. Use (from the repository root; one thread per process, two processes side by side):
#      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 uv run python \
#          reports/2026-11-15_interpretability/tiger_local_bounds.py play --seeds 0,1,2,3,4
#      CUDA_VISIBLE_DEVICES= uv run python reports/2026-11-15_interpretability/tiger_local_bounds.py readout \
#          > reports/2026-11-15_interpretability/tiger_local_bounds.log
# ==============================================================================

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from ipomdp.agents import PlanningAgent
from ipomdp.domain import action_value_functions, build_tiger_pomdp, solve_infinite_horizon
from ipomdp.experiments import AnalysisSettings
from ipomdp.experiments.runs import load_trained_run
from ipomdp.interpretability import analyze_beliefs
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training.seeding import stream_seed

CPU = torch.device("cpu")
RUNS = Path("runs/sweeps/polyak")
OUT = Path("runs/interpretability/tiger_local_bounds")
SEEDS = tuple(range(10))
SETTINGS = AnalysisSettings()                      # 512 episodes, tolerance 0.01, 3,000 probe steps, seed 12345
PROBES = ("mlp/agent", "linear/agent")
ROWS = (("value error", "value_error_mean", "value_error_bound_local_mean", "value_error_bound_mean"),
        ("one-step regret", "regret_mean", "regret_bound_local_mean", "regret_bound_mean"))


def play(seeds: list[int]) -> None:
    torch.set_num_threads(1)
    commit = subprocess.run(["git", "log", "-1", "--format=%H"], check=True, capture_output=True, text=True).stdout.strip()
    todo = [seed for seed in seeds if not (OUT / f"seed{seed}.json").exists()]
    if not todo:
        print("every requested seed is done", flush=True)
        return
    tiger = build_tiger_pomdp()
    start = time.time()
    solution = solve_infinite_horizon(tiger, tolerance=SETTINGS.solver_tolerance, prune_epsilon=1e-6)
    action_values = action_value_functions(tiger, solution.value_function, 1e-6)
    print(f"certified solution: {len(solution.value_function.vectors)} vectors, error bound {solution.error_bound:.2e} "
          f"({time.time() - start:.0f} s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in todo:
        start = time.time()
        trained = load_trained_run(RUNS / f"seed{seed}", CPU)
        stored = json.loads((RUNS / f"seed{seed}" / "analysis" / "report.json").read_text())
        if stored["settings"] != asdict(SETTINGS):
            raise ValueError(f"seed {seed} was analysed with {stored['settings']}, not {asdict(SETTINGS)}.")
        run, cfg = trained.run, trained.cfg
        model = run.eval_agent.model
        # As experiments/runs.py analyze_run builds the planner and calls the analysis.
        planner = PlanningAgent(
            model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                    dirichlet_epsilon=0.0, seed=stream_seed(SETTINGS.seed, 7)),
            SETTINGS.episodes, temperature=0.0, seed=stream_seed(SETTINGS.seed, 8), device=CPU)
        analysis = analyze_beliefs(tiger, run.acting_filter, planner, solution.value_function, action_values,
                                   solution.error_bound, cfg.env.max_steps, SETTINGS.episodes,
                                   SETTINGS.mlp_probe_steps, SETTINGS.seed, CPU)
        row = {"seed": seed, "commit": commit, "seconds": time.time() - start,
               "bounds": {name: asdict(report) for name, report in analysis.bounds.items()},
               "stored_bounds": {name: stored["bounds"][name] for name in PROBES}}
        tmp = OUT / f"seed{seed}.json.tmp"
        tmp.write_text(json.dumps(row, indent=2))
        os.replace(tmp, OUT / f"seed{seed}.json")
        # Timing only: the values are read once, by `readout`.
        print(f"seed {seed}: analysed in {time.time() - start:.0f} s", flush=True)


def _interval(values: list[float]) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))


def _line(label: str, values: list[float], digits: int = 4) -> str:
    mean, half = _interval(values)
    return (f"    {label:<50} {mean:.{digits}f} [{mean - half:.{digits}f}, {mean + half:.{digits}f}]  per seed "
            f"{np.round(values, digits).tolist()}")


def readout() -> None:
    missing = [seed for seed in SEEDS if not (OUT / f"seed{seed}.json").exists()]
    if missing:
        raise SystemExit(f"not read: seeds {missing} are not analysed yet")
    rows = [json.loads((OUT / f"seed{seed}.json").read_text()) for seed in SEEDS]
    print(f"C-q.4 (post hoc): the local form of the span bounds on the canonical Tiger; ten JEPA runs of {RUNS}, CPU, "
          f"analysis seed {SETTINGS.seed}, {SETTINGS.episodes} episodes; analysed at commits "
          f"{sorted({row['commit'][:7] for row in rows})}")
    print("mean [Student-t 95% over the ten seeds]; the stored reports (GPU, other episodes of the same seed) hold the "
          "global form only")
    for probe in PROBES:
        new = [row["bounds"][probe] for row in rows]
        old = [row["stored_bounds"][probe] for row in rows]
        print(f"\n  probe {probe}")
        print(_line("mean L1 decoding error", [b["belief_l1_mean"] for b in new]))
        print(_line("  the same in the stored reports", [b["belief_l1_mean"] for b in old]))
        print(_line("L_V (largest half-span of V*)", [b["lipschitz_value"] for b in new], 2))
        print(_line("half-span of the active vector (mean)", [b["active_half_span_mean"] for b in new], 2))
        for label, measured, local, bound in ROWS:
            print(_line(f"{label}: measured", [b[measured] for b in new]))
            print(_line(f"{label}: local bound", [b[local] for b in new]))
            print(_line(f"{label}: global bound", [b[bound] for b in new]))
            print(_line(f"{label}:   global bound, stored reports", [b[bound] for b in old]))
            print(_line(f"{label}: local / global", [b[local] / b[bound] for b in new], 3))
            print(_line(f"{label}: local bound / measured", [b[local] / b[measured] for b in new], 2))
        print(_line("policy form, local (estimate)", [b["policy_loss_bound_local_mean"] for b in new], 2))
        print(_line("policy form, global (estimate)", [b["policy_loss_bound_mean"] for b in new], 2))
        holds = sum(b["value_error_max"] <= b["value_error_bound_local_max"] + 1e-9
                    and b["regret_max"] <= b["regret_bound_local_max"] + 1e-9 for b in new)
        print(f"    the largest measured value error and regret are under the largest local bounds on {holds} of "
              f"{len(new)} seeds (pair by pair it is a theorem; tests/test_interpretability.py)")
    print(f"\ncost: {np.mean([row['seconds'] for row in rows]) / 60:.1f} min per seed on one CPU thread")


def main() -> None:
    parser = argparse.ArgumentParser(description="C-q.4 (module header).")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("play")
    p.add_argument("--seeds", required=True, type=lambda text: [int(k) for k in text.split(",")])
    sub.add_parser("readout")
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("hide the GPU (CUDA_VISIBLE_DEVICES=): this analysis is registered on the CPU (section 2).")
    if args.command == "play":
        play(args.seeds)
    else:
        readout()


if __name__ == "__main__":
    main()
