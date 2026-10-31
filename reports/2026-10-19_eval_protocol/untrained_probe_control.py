# ABSOLUTE PATH: reports/2026-10-19_eval_protocol/untrained_probe_control.py
# ==============================================================================
# PROBE CONTROL FOR C-b: THE SAME PROBES ON THE UNTRAINED FILTER OF EACH HELD-OUT RUN
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question: how much of the C-b probe accuracy (cell, heading, pit hazard on random histories) is learned, and
#    how much a recurrent network gives for free? A randomly initialised GRU already carries much of its input
#    history (on Tiger an untrained GRU reaches belief R^2 = 0.93; interpretability/belief_probe.py), so a probe
#    result is only meaningful against the same probe on the untrained network (probe controls, Hewitt & Liang
#    2019).
#
# 2. Method (CPU, except the random games, drawn on CUDA as the analysis drew them; usage: uv run python <this> runs/sweeps/wumpus_retrain_heldout 5,6,7,8,9):
#    - Per held-out run: the trained acting filter (best.pt, as C-b) and the run's INITIAL filter (a fresh
#      TrainingRun from the run's own config and seed, i.e. the networks training started from).
#    - The random histories of the held-out analysis: UniformRandomAgent with stream_seed(20261019, 4) on the
#      simulator stream stream_seed(20261019, 1), 512 games, split in halves for fit / held-out, exactly as
#      experiments/wumpus.py (analyze_wumpus_run). The linear and MLP probes and their evaluation are the same
#      functions with the same settings (3000 MLP steps, probe seed 0).
#    - Check: the trained filter's numbers must reproduce report.json's linear_random / mlp_random exactly (on the
#      CPU the probe fit can differ in the last digits from the GPU run that wrote report.json: the check tolerates
#      0.01 on accuracies and prints both).
#    - Agent histories are not replayed (the planner's games are not stored); the control is reported on random
#      histories, the C-b split the paper cites first.
#
# 3. Readout (descriptive; no threshold): per run and mean over runs, trained vs untrained MLP and linear probes:
#    cell and heading accuracy, pit and wumpus hazard error, and the best constant predictor's hazard error.
#
# 4. Result 2026-10-31 (held-out runs 5-9, random histories; log runs/week/untrained_probe_control.log; the trained
#    probes reproduce report.json on every seed). Mean over the 5 runs:
#    | Filter    | Probe  | cell  | heading | pit error | wumpus error |
#    | trained   | MLP    | 0.759 | 0.833   | 0.059     | 0.040        |
#    | trained   | linear | 0.699 | 0.781   | 0.059     | 0.050        |
#    | untrained | MLP    | 0.460 | 0.431   | 0.073     | 0.054        |
#    | untrained | linear | 0.462 | 0.384   | 0.071     | 0.052        |
#    | constant  | -      | -     | -       | 0.083     | 0.060        |
#    - Training adds +30 points of cell and +40 points of heading accuracy over the network it started from; the
#      untrained spread over seeds is under 1.5 points. For the pit hazard the untrained filter already closes about
#      40% of the gap from the constant predictor (0.083 -> 0.073; the latest percepts alone carry breeze), and
#      training closes about as much again (0.059).
# ==============================================================================

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from ipomdp.agents import UniformRandomAgent
from ipomdp.domain import WumpusSimulator
from ipomdp.experiments import build_training_run
from ipomdp.experiments.runs import load_trained_run
from ipomdp.interpretability import (build_wumpus_dataset, evaluate_wumpus_probe, fit_linear_wumpus_probe,
                                     fit_mlp_wumpus_probe)
from ipomdp.training import EpisodeBatch, play_episodes, stream_seed

ANALYSIS_SEED, EPISODES, MLP_STEPS = 20261019, 512, 3000
dev, gpu = torch.device("cpu"), torch.device("cuda")


def halves(episodes: EpisodeBatch) -> tuple[EpisodeBatch, EpisodeBatch]:
    half = episodes.actions.shape[0] // 2
    pick = lambda rows: EpisodeBatch(episodes.actions[rows], episodes.observations[rows],  # noqa: E731
                                     episodes.rewards[rows], episodes.initial_observations[rows])
    return pick(slice(0, half)), pick(slice(half, None))


def probes(world, belief_filter, episodes: EpisodeBatch) -> dict[str, dict]:
    fit_eps, held_eps = halves(episodes)
    with torch.no_grad():
        fit = build_wumpus_dataset(world, belief_filter, fit_eps)
        held = build_wumpus_dataset(world, belief_filter, held_eps)
    return {"linear": asdict(evaluate_wumpus_probe(fit_linear_wumpus_probe(fit), fit, held)),
            "mlp": asdict(evaluate_wumpus_probe(fit_mlp_wumpus_probe(fit, steps=MLP_STEPS), fit, held))}


def line(name: str, r: dict) -> str:
    return (f"{name:16s} cell {r['cell_accuracy']:.3f} heading {r['heading_accuracy']:.3f} hazard error "
            + " ".join(f"{k} {v:.3f}" for k, v in r["hazard_error"].items())
            + " | constant " + " ".join(f"{k} {v:.3f}" for k, v in r["constant_error"].items()))


def main() -> None:
    sweep, seeds = Path(sys.argv[1]), [int(s) for s in sys.argv[2].split(",")]
    table: dict[str, list[dict]] = {}
    for seed in seeds:
        run_dir = sweep / f"seed{seed}"
        trained = load_trained_run(run_dir, dev)
        world, cfg = trained.setup.domain, trained.cfg
        _, fresh = build_training_run(OmegaConf.load(run_dir / ".hydra" / "config.yaml"), dev)
        fresh.belief_filter.eval()
        # The games are drawn on CUDA as in the C-b analysis (torch generators differ between devices), then moved.
        gpu_world = load_trained_run(run_dir, gpu).setup.domain
        simulator = WumpusSimulator(gpu_world, EPISODES, cfg.env.max_steps, stream_seed(ANALYSIS_SEED, 1), gpu)
        played, _ = play_episodes(simulator, UniformRandomAgent(world.num_actions, EPISODES,
                                                                stream_seed(ANALYSIS_SEED, 4), gpu))
        episodes = EpisodeBatch(played.actions.cpu(), played.observations.cpu(), played.rewards.cpu(),
                                played.initial_observations.cpu())
        reported = json.loads((run_dir / "analysis_heldout" / "report.json").read_text())["probes"]
        print(f"== seed {seed}", flush=True)
        for name, belief_filter in (("trained", trained.run.acting_filter), ("untrained", fresh.acting_filter)):
            result = probes(world, belief_filter, episodes)
            for kind in ("linear", "mlp"):
                print("  " + line(f"{name} {kind}", result[kind]), flush=True)
                table.setdefault(f"{name} {kind}", []).append(result[kind])
            if name == "trained":
                for kind in ("linear", "mlp"):
                    ref = reported[f"{kind}_random"]
                    for metric in ("cell_accuracy", "heading_accuracy"):
                        if abs(ref[metric] - result[kind][metric]) > 0.01:
                            raise RuntimeError(f"seed {seed} {kind} {metric}: {result[kind][metric]:.4f} vs "
                                               f"report.json {ref[metric]:.4f}: the replay is not the C-b analysis.")
                print("  trained probes reproduce report.json (random histories, tolerance 0.01)", flush=True)
    print("\n== mean over seeds")
    for name, rows in table.items():
        mean = {"cell_accuracy": np.mean([r["cell_accuracy"] for r in rows]),
                "heading_accuracy": np.mean([r["heading_accuracy"] for r in rows]),
                "hazard_error": {k: np.mean([r["hazard_error"][k] for r in rows]) for k in rows[0]["hazard_error"]},
                "constant_error": {k: np.mean([r["constant_error"][k] for r in rows]) for k in rows[0]["constant_error"]}}
        print("  " + line(name, mean))


if __name__ == "__main__":
    main()
