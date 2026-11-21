# ABSOLUTE PATH: reports/2026-10-19_eval_protocol/ba_pomcp_point_mass_check.py
# ==============================================================================
# POST HOC DIAGNOSTIC OF C-r: IS BA-POMCP'S TIGER DEFICIT IN ITS LEARNING OR IN ITS SEARCH? (CPU; NOT REGISTERED)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Why (written after the C-r Tiger readout, read once on 2026-11-21; PROTOCOL.md, C-r):
#    - The C-r main arm loses 6.76 per episode against the optimal agent, although its post-learning posterior of the
#      listening accuracy is 0.84-0.86 on every run (truth 0.85): the model it plans with is essentially exact. If the
#      deficit is the search, BA-POMCP planning with the TRUE model must lose about as much on the same episodes.
#    - This separates the two by giving BA-POMCP the true tables at point-mass confidence (counts 1e9 x the true
#      probabilities), everything else as the main arm: EXPECTED_MDP rollouts, 1000 simulations, 1000 particles, planning
#      horizon 20 (src/ipomdp/baselines/ba_pomcp.py; reports/2026-10-19_eval_protocol/ba_pomcp_baseline.py).
#    - It is a diagnostic written after the result it explains; it is reported as such, not as a registered claim.
#
# 2. Method:
#    - The 256 paired episodes of simulator seed 11 (the episodes of C-q.3 and C-r), played in 8 chunks of 32 live
#      agents as C-r does (exact: tests/test_ba_pomcp.py, test_simulator_rows_do_not_depend_on_other_rows_actions);
#      agent copy of row i seeded with stream 10,000 + i of the base 20261122.
#    - The optimal agent is the exact filter plus argmax Q* of the certified solution, as in C-r's readout.
#    - Reported: the paired gap per episode, mean +- 1.96 s.e. over the 256 episodes (one deterministic planner on one set
#      of episodes, not a seed interval), against C-r main's -6.76.
#
# 3. Use:
#      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 uv run python reports/2026-10-19_eval_protocol/ba_pomcp_point_mass_check.py 4 \
#          > reports/2026-10-19_eval_protocol/ba_pomcp_point_mass_check.log
#    - The argument is the number of worker processes (at most 8).
# ==============================================================================

import math
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import torch

from ipomdp.baselines.ba_pomcp import BAPOMCPAgent, BAPOMCPBatch, DirichletPrior, Rollout
from ipomdp.domain import BatchedPOMDPEnv, TigerAction, belief_update, build_tiger_pomdp, solve_infinite_horizon
from ipomdp.training.seeding import stream_seed

CPU = torch.device("cpu")
BASE, EPISODES, STEPS, SEED, CHUNK = 20261122, 256, 100, 11, 32
LISTEN = int(TigerAction.LISTEN)


class _Listen:
    def act(self, steps_left):
        return LISTEN

    def begin_episode(self):
        pass

    def update(self, action, observation):
        pass


def chunk_returns(chunk: int) -> np.ndarray:
    torch.set_num_threads(1)
    tiger = build_tiger_pomdp()
    point_mass = DirichletPrior.from_model(tiger, 1e9, 1e9)
    rows = range(chunk * CHUNK, (chunk + 1) * CHUNK)
    agents = [BAPOMCPAgent(tiger, point_mass, 1000, 1000, 20, Rollout.EXPECTED_MDP, stream_seed(BASE, 10_000 + row), 1000)
              if row in rows else _Listen() for row in range(EPISODES)]
    batch = BAPOMCPBatch(agents, STEPS)
    env = BatchedPOMDPEnv(tiger, EPISODES, STEPS, SEED, CPU)
    env.reset()
    batch.reset(None)
    returns = np.zeros(EPISODES)
    for t in range(STEPS):
        action = batch.act()
        out = env.step(action)
        batch.update(action, out.observation)
        returns += tiger.discount ** t * out.reward.double().numpy()
    return returns[list(rows)]


def optimal_returns() -> np.ndarray:
    tiger = build_tiger_pomdp()
    solution = solve_infinite_horizon(tiger, tolerance=1e-3, prune_epsilon=1e-6)
    env = BatchedPOMDPEnv(tiger, EPISODES, STEPS, SEED, CPU)
    env.reset()
    belief = tiger.initial_belief.expand(EPISODES, -1).clone()
    returns = np.zeros(EPISODES)
    for t in range(STEPS):
        action = solution.value_function.greedy_action(belief)
        out = env.step(action)
        belief = belief_update(tiger, belief, action, out.observation)
        returns += tiger.discount ** t * out.reward.double().numpy()
    return returns


def main() -> None:
    workers = int(sys.argv[1])
    if not 1 <= workers <= 8:
        raise ValueError("at most 8 workers (the machine contract).")
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        planned = np.concatenate(list(pool.map(chunk_returns, range(EPISODES // CHUNK))))
    optimal = optimal_returns()
    gap = planned - optimal
    print(f"optimal agent: {optimal.mean():.3f} on the {EPISODES} paired episodes (C-r readout: 22.375)")
    print(f"BA-POMCP with the TRUE model at point-mass confidence (EXPECTED_MDP, 1000 simulations): return "
          f"{planned.mean():.3f}; gap {gap.mean():+.3f} +- {1.96 * gap.std(ddof=1) / math.sqrt(EPISODES):.3f}")
    print(f"episodes with a wrong door opened at least once: share of gap < -50 = {float((gap < -50).mean()):.3f}; "
          f"episodes identical to the optimal agent's return: {float((np.abs(gap) < 1e-9).mean()):.3f}")
    print("C-r main arm (learned model, read once): -6.756 [-8.408, -5.104] over 10 runs")


if __name__ == "__main__":
    main()
