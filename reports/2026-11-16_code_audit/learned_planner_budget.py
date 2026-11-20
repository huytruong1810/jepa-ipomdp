# ABSOLUTE PATH: reports/2026-11-16_code_audit/learned_planner_budget.py
# ==============================================================================
# THE LEARNED TIGER PLANNER AGAINST ITS SEARCH BUDGET (CODE AUDIT, PHASE 4; STORED RUNS, CPU)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. The Question:
#    - reports/2026-11-16_code_audit/planner_budget.py shows, on the EXACT Tiger model, that the search's error bound
#      is monotone in the number of simulations N but its decisions are not: with biased leaves the greedy return can
#      fall as N grows (the horizon effect), while near-exact leaves play optimally at every N. The learned agent
#      plans with learned leaves (the value head) over a learned model. This script measures where it sits: does its
#      return change with N, and in which direction?
#    - It also separates the paper's Wumpus search-budget result (PROTOCOL.md, C-g: more search, more deaths) into
#      what the search does on a model it plans well with (Tiger) and what is left for the Wumpus model's error.
#
# 2. Method (registered in reports/2026-10-19_eval_protocol/PROTOCOL.md before the first run):
#    - Runs: the ten canonical Tiger JEPA runs runs/sweeps/polyak/seed0..9, best.pt, the acting (Polyak) copies,
#      loaded strictly (experiments/runs.py, load_trained_run).
#    - The greedy learned planner of the analysis (dirichlet_epsilon = 0, temperature 0, the run's c_puct and
#      observation_prune) at N in {1, 2, 4, 8, 16, 32, 50, 64, 128}; N = 50 is the trained budget.
#    - Episodes: 256 x 100 steps on simulator seed 11, the episodes of planner_budget.py part B, so every arm and the
#      exact optimal agent (exact filter, argmax Q* of the certified solution) see the same environment numbers.
#    - Quantities: per seed and N, the mean paired gap (learned minus optimal) over the 256 episodes; per N, the mean
#      over the ten seeds with a Student-t 95% interval on the per-seed gaps (the project's convention, statistics.md),
#      the per-seed values printed beside it.
#
# 3. Use:
#      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 uv run python reports/2026-11-16_code_audit/learned_planner_budget.py \
#          > reports/2026-11-16_code_audit/learned_planner_budget.log
#    - One CPU thread; the certified solve takes about 6 min and each seed about 10 min.
# ==============================================================================

import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from ipomdp.agents import PlanningAgent
from ipomdp.domain import BatchedPOMDPEnv, belief_update, build_tiger_pomdp, solve_infinite_horizon
from ipomdp.experiments.runs import load_trained_run
from ipomdp.planning import BeliefTreeSearch

CPU = torch.device("cpu")
RUNS = Path("runs/sweeps/polyak")
SEEDS = tuple(range(10))
BUDGETS = (1, 2, 4, 8, 16, 32, 50, 64, 128)
EPISODES, STEPS, SIMULATOR_SEED = 256, 100, 11


class OptimalAgent:
    """Exact Bayes filter and argmax Q* of the certified solution (the analysis's optimal agent)."""

    def __init__(self, tiger, value_function):
        self.tiger, self.value_function, self.state = tiger, value_function, None

    def reset(self, initial_observation):
        self.state = self.tiger.initial_belief.expand(EPISODES, -1).clone()

    def act(self):
        return self.value_function.greedy_action(self.state)

    def update(self, action, observation):
        self.state = belief_update(self.tiger, self.state, action, observation)


def play(tiger, agent):
    """Discounted returns over EPISODES episodes on SIMULATOR_SEED (the episodes of planner_budget.py part B)."""
    env = BatchedPOMDPEnv(tiger, EPISODES, STEPS, SIMULATOR_SEED, CPU)
    env.reset()
    agent.reset(None)
    returns = torch.zeros(EPISODES, dtype=torch.float64)
    for t in range(STEPS):
        action = agent.act()
        out = env.step(action)
        agent.update(action, out.observation)
        returns += tiger.discount ** t * out.reward.double()
    return returns


def main():
    tiger = build_tiger_pomdp()
    start = time.time()
    certified = solve_infinite_horizon(tiger, tolerance=1e-3, prune_epsilon=1e-6)
    optimal = play(tiger, OptimalAgent(tiger, certified.value_function))
    print(f"optimal agent: {float(optimal.mean()):.3f} over {EPISODES} episodes (certified solve "
          f"{time.time() - start:.0f} s, error bound {certified.error_bound:.1e})", flush=True)
    gaps = np.zeros((len(SEEDS), len(BUDGETS)))
    for i, seed in enumerate(SEEDS):
        trained = load_trained_run(RUNS / f"seed{seed}", CPU)
        model, mcts = trained.run.eval_agent.model, trained.cfg.mcts
        for j, sims in enumerate(BUDGETS):
            planner = BeliefTreeSearch(model, sims, mcts.c_puct, mcts.dirichlet_alpha, dirichlet_epsilon=0.0, seed=0,
                                       observation_prune=mcts.observation_prune)
            agent = PlanningAgent(model, planner, EPISODES, temperature=0.0, seed=0, device=CPU)
            gaps[i, j] = float((play(tiger, agent) - optimal).mean())
        print(f"seed {seed}: gap (learned - optimal) per episode at N = {BUDGETS}: "
              f"{np.round(gaps[i], 3).tolist()}", flush=True)
    print("\nper budget, over the ten seeds: mean gap, Student-t 95% interval on the per-seed gaps")
    t = stats.t.ppf(0.975, len(SEEDS) - 1)
    for j, sims in enumerate(BUDGETS):
        column = gaps[:, j]
        half = t * column.std(ddof=1) / math.sqrt(len(SEEDS))
        print(f"  N={sims:4d}: {column.mean():+.3f} [{column.mean() - half:+.3f}, {column.mean() + half:+.3f}]")


if __name__ == "__main__":
    main()
