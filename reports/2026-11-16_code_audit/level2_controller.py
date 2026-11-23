# ABSOLUTE PATH: reports/2026-11-16_code_audit/level2_controller.py
# ==============================================================================
# WHICH CONTROLLER MODELS AGENT j AT LEVEL 1? THE MEASUREMENTS BEHIND interactive.py, SECTION 7 (CPU)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. The Question:
#    - Agent i's level-2 problem folds agent j's level-1 controller. At level 0 that controller is extracted from a
#      tight HSVI bracket (interactive.py, section 6). At level 1 no tight bracket exists: HSVI's upper end stalls
#      (itiger_benchmark.py). This script measures what the available constructions give on j's level-1 problem, the
#      26-state fold whose optimum is bracketed in [16.5510, 19.6692] (runs/benchmarks/itiger_level1.pt).
#
# 2. What Is Measured:
#    a. j's level-1 fold against i's: the game is symmetric, so their digests must be equal. The benchmark's
#       lower-bound vectors, computed on i's problem, are then lower-bound vectors of j's problem too.
#    b. policy_graph.stationary_graph on the search's first lower-bound vectors (any prefix of them is a valid set of
#       lower-bound vectors: the search only appends) at several prune epsilons: nodes and exact value at b0.
#    c. The lower bound's own pointer controller (bounds.py, section 2), from the search's checkpoint: its size, the
#       part reachable from its initial node, the same after merging equivalent nodes (Moore's partition refinement
#       on action and successors), and the number of steps before it falls into a blind policy.
#    d. The declared family of growl counters (interactive.tiger_growl_counters): exact value of each member.
#    e. How close the best counter is to the best SMALL deterministic controller: an exact local search that takes,
#       sweep by sweep, the best single change of one node's action or of one successor, from the best counter with
#       0 to 12 spare nodes (copies of its nodes) and from random controllers.
#    - Every value is exact: one linear solve over (node, state), checked against policy_graph.evaluate.
#
# 3. Use (about 15 minutes on one thread; reads runs/benchmarks/itiger_level1.search.npz, the search's last checkpoint):
#      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 uv run python reports/2026-11-16_code_audit/level2_controller.py \
#          > reports/2026-11-16_code_audit/level2_controller.log
# ==============================================================================

import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from ipomdp.domain.interactive import (CREAK_RULES, GROWL_COUNTER_THRESHOLDS, TIGER_LEVEL_ZERO_NOISE, fold, growl_counter,
                                       nested_problem, tiger_growl_counters)
from ipomdp.domain.multiagent_tiger import Agent, build_multiagent_tiger
from ipomdp.domain.policy_graph import evaluate, stationary_graph
from ipomdp.domain.solver import AlphaVectorSet
from ipomdp.domain.tiger import TigerAction
from ipomdp.experiments.interactive_analysis import fold_digest

CHECKPOINT = Path("runs/benchmarks/itiger_level1.search.npz")
LISTEN = int(TigerAction.LISTEN)
PREFIXES = (828, 2157)                  # the vectors after 10 and after 30 trials (itiger_benchmark_artifact_rerun.log)
PRUNE_EPSILONS = (0.5, 0.2, 0.05)
SPARE_NODES = (0, 2, 4, 8, 12)
RANDOM_NODES, RANDOM_STARTS, RANDOM_SEED = (7, 9, 12), 8, 0


def main() -> None:
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)   # noqa: E731
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    log(f"commit {commit}{' (uncommitted changes)' if dirty else ''}")
    physics = build_multiagent_tiger(0.95)
    family = tiger_growl_counters()
    problem_i = nested_problem(physics, Agent.I, 1, TIGER_LEVEL_ZERO_NOISE, 0.05, 0.5, LISTEN, family)
    problem_j = nested_problem(physics, Agent.J, 1, TIGER_LEVEL_ZERO_NOISE, 0.05, 0.5, LISTEN, family)
    model = problem_j.pomdp
    log(f"(a) level-1 fold digests: i {fold_digest(problem_i.pomdp)[:16]}, j {fold_digest(model)[:16]}, equal "
        f"{fold_digest(problem_i.pomdp) == fold_digest(model)}; {model.num_states} states")
    b0 = model.initial_belief.numpy()
    with np.load(CHECKPOINT) as stored:
        vectors, actions, successors, trials = stored["vectors"], stored["actions"], stored["successors"], int(stored["trial"])
    log(f"checkpoint: {trials} trials, {len(vectors)} lower-bound vectors, L(b0) = {float((vectors @ b0).max()):.4f}")

    # (b) stationary_graph on prefixes of the lower bound.
    for count in PREFIXES:
        lower = float((vectors[:count] @ b0).max())
        for epsilon in PRUNE_EPSILONS:
            graph = stationary_graph(model, AlphaVectorSet(torch.from_numpy(vectors[:count].copy()),
                                                           torch.from_numpy(actions[:count].copy())), epsilon)
            value = float(graph.initial_law @ (evaluate(model, graph) @ model.initial_belief))
            log(f"(b) stationary_graph, first {count} vectors (lower bound {lower:.4f}), prune epsilon {epsilon}: "
                f"{graph.num_nodes} nodes, exact value {value:.4f}")

    # (c) the pointer controller.
    k = len(vectors)
    start = int((vectors @ b0).argmax())
    seen = np.zeros(k, dtype=bool)
    seen[start], stack = True, [start]
    while stack:
        for nxt in successors[stack.pop()]:
            if not seen[nxt]:
                seen[nxt] = True
                stack.append(int(nxt))
    reachable = np.flatnonzero(seen)
    blind = len(TigerAction)                                # the first |A| vectors are the blind policies (self-loops)
    older = bool((successors[blind:] < np.arange(blind, k)[:, None]).all())
    shortest, longest = np.zeros(k, dtype=np.int64), np.zeros(k, dtype=np.int64)
    for node in range(blind, k):
        shortest[node], longest[node] = 1 + shortest[successors[node]].min(), 1 + longest[successors[node]].max()
    classes = {int(node): int(actions[node]) for node in reachable}
    while True:
        signature = {node: (classes[node], tuple(classes[int(nxt)] for nxt in successors[node])) for node in classes}
        index = {sig: i for i, sig in enumerate(sorted(set(signature.values())))}
        refined = {node: index[signature[node]] for node in classes}
        if len(set(refined.values())) == len(set(classes.values())):
            break
        classes = refined
    log(f"(c) pointer controller: {k} nodes, {len(reachable)} reachable from its initial node, {len(set(classes.values()))} "
        f"after merging equivalent nodes; every successor is an older node: {older}; steps from the initial node to a "
        f"blind policy: {shortest[start]} at least, {longest[start]} at most; its fold would have up to "
        f"{2 * len(reachable) * 3} states")

    # (d) the declared family.
    values = [float(member.initial_law @ (evaluate(model, member) @ model.initial_belief)) for member in family]
    for member, value in zip(family, values):
        folded = fold(physics.view(Agent.I), member, LISTEN)
        log(f"(d) {member.node_names[0].rsplit('.', 1)[0]}: {member.num_nodes} nodes, exact value {value:.4f}; i's level-2 "
            f"fold {folded.pomdp.num_states} states")
    best = values.index(max(values))
    log(f"(d) best of {len(family)} (thresholds {GROWL_COUNTER_THRESHOLDS}, rules {CREAK_RULES}): "
        f"{family[best].node_names[0].rsplit('.', 1)[0]}, exact value {values[best]:.4f}")

    # (e) exact local search over small deterministic controllers.
    transition, observation, reward = model.transition.numpy(), model.observation.numpy(), model.reward.numpy()
    num_s, num_z, gamma = model.num_states, model.num_observations, model.discount
    step = np.einsum("ast,ato->aost", transition, observation)                     # M[a, o, s, t]

    def exact(act: np.ndarray, succ: np.ndarray, first: int) -> float:
        n = len(act)
        chain = np.zeros((n, num_s, n, num_s))
        for node in range(n):
            for o in range(num_z):
                chain[node, :, succ[node, o], :] += step[act[node], o]
        value = np.linalg.solve(np.eye(n * num_s) - gamma * chain.reshape(n * num_s, n * num_s), reward[act].reshape(n * num_s))
        return float(value.reshape(n, num_s)[first] @ b0)

    def search(act: np.ndarray, succ: np.ndarray, first: int) -> tuple[float, int]:
        act, succ, n = act.copy(), succ.copy(), len(act)
        best_value, sweeps = exact(act, succ, first), 0
        while True:
            sweeps += 1
            move = None
            for node in range(n):
                for a in range(len(TigerAction)):
                    if a != act[node]:
                        old, act[node] = act[node], a
                        value = exact(act, succ, first)
                        act[node] = old
                        if value > best_value + 1e-9 and (move is None or value > move[0]):
                            move = (value, node, None, a)
                for o in range(num_z):
                    for target in range(n):
                        if target != succ[node, o]:
                            old, succ[node, o] = succ[node, o], target
                            value = exact(act, succ, first)
                            succ[node, o] = old
                            if value > best_value + 1e-9 and (move is None or value > move[0]):
                                move = (value, node, o, target)
            if move is None:
                return best_value, sweeps
            best_value = move[0]
            if move[2] is None:
                act[move[1]] = move[3]
            else:
                succ[move[1], move[2]] = move[3]

    seed = family[best]
    seed_act = seed.action_law.argmax(-1).numpy()
    seed_succ = seed.successor_law[torch.arange(seed.num_nodes), torch.from_numpy(seed_act)].argmax(-1).numpy()   # (N, O)
    seed_first = int(seed.initial_law.argmax())
    if abs(exact(seed_act, seed_succ, seed_first) - values[best]) > 1e-9:
        raise RuntimeError("the search's evaluator disagrees with policy_graph.evaluate.")
    for spare in SPARE_NODES:
        copies = np.arange(spare) % seed.num_nodes
        value, sweeps = search(np.concatenate([seed_act, seed_act[copies]]), np.vstack([seed_succ, seed_succ[copies]]), seed_first)
        log(f"(e) local search from the best counter with {spare} spare nodes ({seed.num_nodes + spare} nodes): "
            f"{value:.4f} after {sweeps} sweeps")
    rng = np.random.default_rng(RANDOM_SEED)
    for n in RANDOM_NODES:
        found = []
        for _ in range(RANDOM_STARTS):
            act = rng.integers(0, len(TigerAction), size=n)
            act[0] = LISTEN
            found.append(search(act, rng.integers(0, n, size=(n, num_z)), 0)[0])
        log(f"(e) local search from {RANDOM_STARTS} random controllers of {n} nodes: " + ", ".join(f"{v:.2f}" for v in sorted(found)))


if __name__ == "__main__":
    main()
