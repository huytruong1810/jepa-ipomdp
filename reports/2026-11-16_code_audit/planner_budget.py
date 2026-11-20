# ABSOLUTE PATH: reports/2026-11-16_code_audit/planner_budget.py
# ==============================================================================
# DOES THE BELIEF-TREE SEARCH IMPROVE MONOTONICALLY WITH ITS BUDGET? (CODE AUDIT, PHASE 4; EXACT MODEL, CPU)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. The Question:
#    - A planner that is given more simulations should not get worse. This script separates what the search
#      guarantees from what it does not, on a model with no learning error: the exact canonical Tiger
#      (ExactSearchModel), with leaf estimates of known quality (planning/mcts.py: PUCT selection, exact observation
#      branching, expectimax backups).
#
# 2. What Is Guaranteed (checked here and in tests/test_planner_and_agent.py):
#    - If every leaf estimate is within eps of V* (eps = ||leaf - V*||_inf), every root edge satisfies
#         |Q_tree(b, a) - Q*(b, a)| <= gamma^{d_a} eps,
#      d_a = 1 + the shallowest positive-probability frontier depth below the edge. Proof by induction: a leaf is off
#      by at most eps; a chance edge averages its children's errors, times gamma; a decision node's max moves by at
#      most its largest edge error.
#    - The tree after N simulations is a prefix of the tree after N' > N (fixed seed, one random stream), so d_a never
#      decreases and the bound is non-increasing in N. PUCT visits every action and observation sampling every
#      positive-probability branch infinitely often, so d_a grows without limit and the search is consistent:
#      Q_tree -> Q* as N -> infinity.
#
# 3. What Is Not Guaranteed:
#    - The DECISION argmax_a Q_tree. PUCT deepens the promising branch first, so at a finite budget the actions are
#      compared on subtrees of different depths. With biased leaves the deeper subtree carries less of the bias, and
#      the comparison can flip back and forth as N grows (the horizon effect of depth-limited search). How fast d_a
#      grows (part A of the log) and whether the flips cost return (part B) are measured, not assumed.
#
# 4. Method:
#    - Leaves: zero, the finite-horizon optima V_1, V_3, V_8 and V_12 (domain/solver.py, solve_finite_horizon), and
#      V* itself (the certified solve, about 6 min). eps is the exact sup-norm distance to V*
#      (domain/solver.py, sup_norm_distance) plus the solver's certified error.
#    - Part A: root Q against Q* at six beliefs P(tiger left) in {0.5, 0.85, 0.9, 0.95, 0.97, 0.99}, budgets 1 to
#      4096, no root noise, seed 0. It prints the largest error, the largest bound, the range of d_a, the beliefs
#      where the greedy action is not optimal, and the decision sequence over the budgets where it changes.
#    - Part B: the greedy planning agent (agents/planning_agent.py) plays 256 episodes of 100 steps on one simulator
#      seed (11) for every leaf and budget. Every arm sees the same environment random numbers, so the differences
#      are paired per episode against the exact optimal agent (V* leaves, one simulation, which is argmax Q*). The
#      interval is mean +- 1.96 standard errors over the 256 paired episodes: a property of one deterministic planner
#      on one set of episodes, not a training-seed interval.
#
# 5. Use:
#      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 uv run python reports/2026-11-16_code_audit/planner_budget.py \
#          > reports/2026-11-16_code_audit/planner_budget.log
#    - Runs on one CPU thread in about one hour (the certified solve and the budget-128 plays dominate). Results are
#      recorded in reports/2026-11-16_code_audit/REVIEW.md (Phase 4).
# ==============================================================================

import math
import time

import numpy as np
import torch

from ipomdp.agents import PlanningAgent
from ipomdp.domain import (BatchedPOMDPEnv, belief_update, build_tiger_pomdp, observation_distribution,
                           solve_finite_horizon, solve_infinite_horizon, sup_norm_distance)
from ipomdp.planning import BeliefTreeSearch, ExactSearchModel

CPU = torch.device("cpu")
BELIEFS = (0.5, 0.85, 0.9, 0.95, 0.97, 0.99)
TREE_BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
PLAY_BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128)
EPISODES, STEPS, SIMULATOR_SEED = 256, 100, 11


def exact_q(tiger, leaf, beliefs):
    """Q(b, a) = b . R[a] + gamma sum_o P(o | b, a) leaf(tau(b, a, o)), shape (N, A)."""
    rows = []
    for a in range(tiger.num_actions):
        action = torch.full((len(beliefs),), a)
        probs = observation_distribution(tiger, beliefs, action)
        q = beliefs @ tiger.reward[a]
        for o in range(tiger.num_observations):
            posterior = belief_update(tiger, beliefs, action, torch.full((len(beliefs),), o))
            q = q + tiger.discount * probs[:, o] * leaf.value(posterior)
        rows.append(q)
    return torch.stack(rows, dim=-1)


def frontier_depth(node):
    """Shallowest frontier depth below `node` through positive-probability branches (0 if unexpanded)."""
    if node is None or node.edges is None:
        return 0
    return min(1 + frontier_depth(child) for edge in node.edges
               for p, child in zip(edge._probs, edge._children) if p > 0)


def edge_depth(edge):
    return min(1 + frontier_depth(child) for p, child in zip(edge._probs, edge._children) if p > 0)


def play(tiger, leaf, sims):
    """Discounted returns of the greedy planning agent over EPISODES episodes on SIMULATOR_SEED."""
    model = ExactSearchModel(tiger, leaf, CPU)
    agent = PlanningAgent(model, BeliefTreeSearch(model, sims, 1.25, 0.3, 0.0, seed=0), EPISODES, temperature=0.0,
                          seed=0, device=CPU)
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
    vstar, delta = certified.value_function, certified.error_bound
    finite = solve_finite_horizon(tiger, 12)
    print(f"V*: {certified.iterations} iterations, certified error {delta:.2e}, {len(vstar.vectors)} vectors, "
          f"V*(b0) = {float(vstar.value(tiger.initial_belief[None])):.4f} ({time.time() - start:.0f} s)")
    corners = vstar.value(torch.eye(2, dtype=torch.float64))
    leaves = {"zero": (None, float(corners.max()) + delta)}
    for h in (1, 3, 8, 12):
        leaves[f"V{h}"] = (finite[h - 1], sup_norm_distance(finite[h - 1], vstar) + delta)
    print("eps = ||leaf - V*||_inf + delta: " + ", ".join(f"{k} {eps:.3f}" for k, (_, eps) in leaves.items()))

    beliefs = torch.tensor([[p, 1 - p] for p in BELIEFS], dtype=torch.float64)
    q_star = exact_q(tiger, vstar, beliefs).numpy()
    optimal = q_star.argmax(1)
    print("Q*(b, a) [LISTEN, OPEN_LEFT, OPEN_RIGHT] at P(left) = "
          + "; ".join(f"{p}: {np.round(q, 3).tolist()}" for p, q in zip(BELIEFS, q_star)))

    print("\nPART A: root Q against Q* (exact model; no root noise; seed 0)")
    worst = -math.inf
    for name, (leaf, eps) in leaves.items():
        print(f"\nleaves {name} (eps {eps:.3f})")
        decisions, previous = [], None
        for sims in TREE_BUDGETS:
            planner = BeliefTreeSearch(ExactSearchModel(tiger, leaf, CPU), sims, 1.25, 0.3, 0.0, seed=0)
            planner.search(beliefs, temperature=0.0)
            q = planner.statistics.q_values
            depths = np.array([[edge_depth(edge) for edge in root.edges] for root in planner.roots])
            if previous is not None and not (depths >= previous).all():
                raise AssertionError(f"a frontier depth decreased at N = {sims}")
            previous = depths
            error = np.abs(q - q_star)
            bound = tiger.discount ** depths * eps + tiger.discount * delta
            worst = max(worst, float((error - bound).max()))
            decision = q.argmax(1)
            decisions.append(decision)
            wrong = [p for p, d, o in zip(BELIEFS, decision, optimal) if d != o]
            print(f"  N={sims:5d}  max|Q-Q*| {error.max():8.4f}  max bound {bound.max():8.3f}  "
                  f"d_a {depths.min()}..{depths.max()}  non-optimal at P(left) = {wrong}")
        sequence = np.array(decisions)
        changes = {p: sequence[:, i].tolist() for i, p in enumerate(BELIEFS) if len(set(sequence[:, i])) > 1}
        print(f"  decisions over the budgets where they change (0 LISTEN, 1 OPEN_LEFT, 2 OPEN_RIGHT): {changes}")
    print(f"\nlargest (error - bound) over every case: {worst:.2e} (the bound holds if <= 0)")

    print(f"\nPART B: discounted return, {EPISODES} episodes x {STEPS} steps, simulator seed {SIMULATOR_SEED}, "
          "paired per episode with the optimal agent")
    reference = play(tiger, vstar, 1)
    print(f"optimal agent (V* leaves, 1 simulation): {float(reference.mean()):.3f} "
          f"+- {1.96 * float(reference.std()) / math.sqrt(EPISODES):.3f}")
    for name in ("zero", "V3", "V8"):
        leaf = leaves[name][0]
        for sims in PLAY_BUDGETS:
            tick = time.time()
            returns = play(tiger, leaf, sims)
            gap = returns - reference
            print(f"  leaves {name:4s} N={sims:4d}: return {float(returns.mean()):8.3f}   minus optimal "
                  f"{float(gap.mean()):+8.3f} +- {1.96 * float(gap.std()) / math.sqrt(EPISODES):.3f}   "
                  f"({time.time() - tick:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
