# ABSOLUTE PATH: reports/2026-11-15_interpretability/tiger_reanalysis.py
# ==============================================================================
# C-q.7: THE CANONICAL TIGER RE-ANALYSED WITH THE SELECTED-ITERATE MLP PROBE (POST HOC; STORED RUNS, CPU)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Why:
#    - Until 2026-11-24 the MLP probe returned the last iterate of its fit, which lands on a late rise of the loss
#      about once in 25 to 50 fits (PROTOCOL.md C-q.5). It now returns the lowest-training-loss iterate of the
#      fit's tail (interpretability/belief_probe.py, section 5; tested on fresh fits in C-q.6). The stored Tiger
#      reports were made with the old fit. This item analyses the same twenty runs with the new one, so that one
#      instrument serves every probe number of the paper.
#    - It never writes into a run directory: the stored reports stay what the ledger reads until the audit decides
#      otherwise. One JSON per run goes to runs/interpretability/tiger_reanalysis/<sweep>/, written atomically; a
#      run whose file exists is skipped.
#
# 2. Method:
#    - The ten JEPA runs runs/sweeps/polyak/seed0..9 and the ten decoder runs runs/sweeps/decoder/seed0..9, best.pt,
#      the acting (Polyak) copies, loaded strictly. interpretability.analyze_beliefs is called exactly as analyze_run
#      calls it (the certified solution at tolerance 0.01, one solve for all runs; the greedy planner at the run's
#      budget; 512 episodes; 3,000 probe steps; analysis seed 12345). This is tiger_local_bounds.py's call (C-q.4).
#    - On the CPU, one thread. The stored reports were made on the GPU, and a simulator's generator lives on its
#      device, so the same analysis seed draws other episodes here: against the stored reports this is a replication
#      on other episodes AND a change of the fit. The clean comparison of the two fits is with C-q.4, which analysed
#      the ten JEPA runs on these very CPU episodes with the old fit: there only the MLP rows can differ.
#    - Faithfulness, checked per JEPA run and raising on failure: the linear probe is untouched by the change, so
#      every field of its bound report must equal the one C-q.4 stored for the run. (The decoder runs have no CPU
#      analysis to check against.)
#
# 3. Quantities (`readout`, once, after the twenty runs), per sweep, mean with a Student-t 95% interval over its ten
#    seeds and the per-seed values:
#    - the probes' mean L1 and mean KL (linear and MLP, random and agent histories), beside the stored reports';
#    - for the MLP probe on the agent's histories, every field of the bound report, beside the stored reports' and,
#      for the JEPA runs, beside C-q.4's (the old fit on the same episodes);
#    - the returns of the four agents and the paired gaps to the optimal agent, beside the stored reports'.
#    - No pass rule. Which numbers the paper cites is the audit's decision.
#
# 4. Use (from the repository root; one thread per process):
#      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 uv run python \
#          reports/2026-11-15_interpretability/tiger_reanalysis.py play --sweep polyak --seeds 0,1,2,3,4,5,6,7,8,9
#      CUDA_VISIBLE_DEVICES= uv run python reports/2026-11-15_interpretability/tiger_reanalysis.py readout \
#          > reports/2026-11-15_interpretability/tiger_reanalysis.log
#    `--episodes` and `--out` exist for a rehearsal at a small size; the faithfulness check runs only at the
#    registered size, and `readout` reads only the registered directory.
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
SWEEPS = {"polyak": Path("runs/sweeps/polyak"), "decoder": Path("runs/sweeps/decoder")}
OUT = Path("runs/interpretability/tiger_reanalysis")
CQ4 = Path("runs/interpretability/tiger_local_bounds")          # the old fit on the JEPA runs' CPU episodes
SEEDS = tuple(range(10))
SETTINGS = AnalysisSettings()                      # 512 episodes, tolerance 0.01, 3,000 probe steps, seed 12345
PROBES = ("linear/random", "linear/agent", "mlp/random", "mlp/agent")
AGENTS = ("optimal", "learned_planner", "decoded_mlp", "decoded_linear")


def play(sweep: str, seeds: list[int], episodes: int, out: Path) -> None:
    torch.set_num_threads(1)
    registered = episodes == SETTINGS.episodes
    if registered != (out == OUT):
        raise ValueError(f"{OUT} holds the registered size ({SETTINGS.episodes} episodes) and nothing else.")
    commit = subprocess.run(["git", "log", "-1", "--format=%H"], check=True, capture_output=True, text=True).stdout.strip()
    todo = [seed for seed in seeds if not (out / sweep / f"seed{seed}.json").exists()]
    if not todo:
        print("every requested seed is done", flush=True)
        return
    tiger = build_tiger_pomdp()
    solution = solve_infinite_horizon(tiger, tolerance=SETTINGS.solver_tolerance, prune_epsilon=1e-6)
    action_values = action_value_functions(tiger, solution.value_function, 1e-6)
    (out / sweep).mkdir(parents=True, exist_ok=True)
    for seed in todo:
        start = time.time()
        trained = load_trained_run(SWEEPS[sweep] / f"seed{seed}", CPU)
        stored = json.loads((SWEEPS[sweep] / f"seed{seed}" / "analysis" / "report.json").read_text())
        if stored["settings"] != asdict(SETTINGS):
            raise ValueError(f"{sweep} seed {seed} was analysed with {stored['settings']}, not {asdict(SETTINGS)}.")
        run, cfg = trained.run, trained.cfg
        model = run.eval_agent.model
        # As experiments/runs.py analyze_run builds the planner and calls the analysis.
        planner = PlanningAgent(
            model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                    dirichlet_epsilon=0.0, seed=stream_seed(SETTINGS.seed, 7)),
            episodes, temperature=0.0, seed=stream_seed(SETTINGS.seed, 8), device=CPU)
        analysis = asdict(analyze_beliefs(tiger, run.acting_filter, planner, solution.value_function, action_values,
                                          solution.error_bound, cfg.env.max_steps, episodes,
                                          SETTINGS.mlp_probe_steps, SETTINGS.seed, CPU))
        old_fit = None
        if registered and sweep == "polyak":                     # section 2, faithfulness
            cq4 = json.loads((CQ4 / f"seed{seed}.json").read_text())["bounds"]
            if analysis["bounds"]["linear/agent"] != cq4["linear/agent"]:
                raise RuntimeError(f"polyak seed {seed}: the linear probe's bound report differs from the one C-q.4 "
                                   "stored; these are not C-q.4's episodes or not its analysis.")
            old_fit = cq4["mlp/agent"]
        for probe in analysis["probes"].values():                # megabytes of per-posterior rows that no readout uses
            probe["by_posterior"] = None
        row = {"sweep": sweep, "seed": seed, "commit": commit, "episodes": episodes, "seconds": time.time() - start,
               "analysis": analysis,
               "stored": {"returns": stored["returns"], "bounds": stored["bounds"],
                          "probes": {name: {k: v for k, v in report.items() if k != "by_posterior"}
                                     for name, report in stored["probes"].items()}},
               "old_fit_same_episodes": old_fit}
        tmp = out / sweep / f"seed{seed}.json.tmp"
        tmp.write_text(json.dumps(row))
        os.replace(tmp, out / sweep / f"seed{seed}.json")
        # Timing only: the values are read once, by `readout`.
        checked = "linear rows equal to C-q.4" if old_fit is not None else "no CPU reference to check against"
        print(f"{sweep} seed {seed}: {checked}; analysed in {time.time() - start:.0f} s", flush=True)


def _interval(values: list[float], digits: int) -> str:
    x = np.asarray(values, dtype=np.float64)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return f"{x.mean():.{digits}f} [{x.mean() - half:.{digits}f}, {x.mean() + half:.{digits}f}]"


def _line(label: str, values: list[float], digits: int = 4) -> str:
    return f"    {label:<58} {_interval(values, digits)}  per seed {np.round(values, digits).tolist()}"


def readout() -> None:
    missing = [f"{sweep}/seed{seed}" for sweep in SWEEPS for seed in SEEDS if not (OUT / sweep / f"seed{seed}.json").exists()]
    if missing:
        raise SystemExit(f"not read: {missing} are not analysed yet")
    print(f"C-q.7 (post hoc): the canonical Tiger re-analysed with the selected-iterate MLP probe; CPU, one thread, "
          f"analysis seed {SETTINGS.seed}, {SETTINGS.episodes} episodes")
    print("mean [Student-t 95% over the ten seeds]. 'stored' is the run's own report (GPU, other episodes of the same "
          "seed, the last-iterate fit); 'old fit' is C-q.4 (these CPU episodes, the last-iterate fit; JEPA runs only)")
    for sweep in SWEEPS:
        rows = [json.loads((OUT / sweep / f"seed{seed}.json").read_text()) for seed in SEEDS]
        print(f"\n=== {SWEEPS[sweep]} (analysed at commits {sorted({row['commit'][:7] for row in rows})})")
        print("  probes")
        for probe in PROBES:
            for key in ("mean_l1", "mean_kl"):
                digits = 4 if key == "mean_l1" else 5
                print(_line(f"{probe} {key}", [row["analysis"]["probes"][probe][key] for row in rows], digits))
                print(_line(f"  stored {probe} {key}", [row["stored"]["probes"][probe][key] for row in rows], digits))
        print("  the MLP probe's bound report on the agent's histories")
        for key in rows[0]["stored"]["bounds"]["mlp/agent"]:
            print(_line(key, [row["analysis"]["bounds"]["mlp/agent"][key] for row in rows]))
            print(_line(f"  stored {key}", [row["stored"]["bounds"]["mlp/agent"][key] for row in rows]))
            if rows[0]["old_fit_same_episodes"] is not None:
                print(_line(f"  old fit {key}", [row["old_fit_same_episodes"][key] for row in rows]))
        for key in ("active_half_span_mean", "value_error_bound_local_mean", "regret_bound_local_mean",
                    "policy_loss_bound_local_mean"):                 # the local form: not in the stored reports
            print(_line(key, [row["analysis"]["bounds"]["mlp/agent"][key] for row in rows]))
            if rows[0]["old_fit_same_episodes"] is not None:
                print(_line(f"  old fit {key}", [row["old_fit_same_episodes"][key] for row in rows]))
        print("  returns")
        for agent in AGENTS:
            print(_line(agent, [row["analysis"]["returns"][agent]["mean"] for row in rows], 3))
            print(_line(f"  stored {agent}", [row["stored"]["returns"][agent]["mean"] for row in rows], 3))
        for agent in AGENTS[1:]:
            gap = lambda source: [row[source]["returns"][agent]["mean"] - row[source]["returns"]["optimal"]["mean"]  # noqa: E731
                                  for row in rows]
            print(_line(f"{agent} - optimal (paired)", gap("analysis"), 3))
            print(_line(f"  stored {agent} - optimal (paired)", gap("stored"), 3))
        print(f"  cost: {np.mean([row['seconds'] for row in rows]) / 60:.1f} min per run on one CPU thread")


def main() -> None:
    parser = argparse.ArgumentParser(description="C-q.7 (module header).")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("play")
    p.add_argument("--sweep", required=True, choices=sorted(SWEEPS))
    p.add_argument("--seeds", required=True, type=lambda text: [int(k) for k in text.split(",")])
    p.add_argument("--episodes", type=int, default=SETTINGS.episodes)
    p.add_argument("--out", type=Path, default=OUT)
    sub.add_parser("readout")
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("hide the GPU (CUDA_VISIBLE_DEVICES=): this analysis is registered on the CPU (section 2).")
    if args.command == "play":
        play(args.sweep, args.seeds, args.episodes, args.out)
    else:
        readout()


if __name__ == "__main__":
    main()
