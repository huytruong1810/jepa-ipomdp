# ABSOLUTE PATH: reports/2026-11-16_code_audit/benchmark_improvability.py
# ==============================================================================
# THE LOWER BOUND OF AN INTERACTIVE TIGER BENCHMARK IS UNIFORMLY IMPROVABLE (CODE AUDIT, PHASE 6; CPU, MINUTES)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. What Rests on This Property:
#    - A benchmark file (experiments/interactive_analysis.py) holds the lower bound of a folded problem as vectors
#      alpha_k with actions a_k: L(b) = max_k alpha_k . b. Two statements of the analysis are made about it:
#        (i)  the reference agent (exact belief, action of the best vector) has value at least L(b0);
#        (ii) the policy form of the span bounds (interpretability/error_bounds.py, section 4): an agent that acts
#             greedily on a decoded belief with error eps_t loses at most 2 L_Q sum_t gamma^t E[eps_t] against L(b0).
#    - Both need L to be uniformly improvable action by action: with Q_L(b, a) the maximum over the vectors of action a,
#           Q_L(b, a) <= R(b, a) + gamma sum_o P(o | b, a) L(tau(b, a, o))     for every belief b and action a.     (*)
#      Then L(b_t) = Q_L(b_t, a_t) + regret_t <= r_t + gamma E[L(b_{t+1})] + regret_t along any policy, and unrolling
#      gives L(b0) <= V^pi(b0) + sum_t gamma^t E[regret_t]; the reference agent has regret 0.
#
# 2. The Exact Check (all beliefs at once):
#    - (*) holds for every belief if every vector is the value of its pointer plan in the FINAL set,
#           alpha_k = R(., a_k) + gamma sum_o M_{a_k, o} alpha_{succ(k, o)},      M_{a,o}[s, s'] = T(s'|s,a) O(o|s',a),
#      because alpha_{succ(k,o)} . u <= max_j alpha_j . u for the unnormalised successor belief u = b M_{a,o}.
#    - The benchmark file stores vectors and actions; the search checkpoint beside it
#      (runs/benchmarks/itiger_level<l>.search.npz; domain/bounds.py, section 7) also stores the pointers succ(k, o).
#      The script first requires the checkpoint's vectors and actions to equal the file's, then reports the largest
#      absolute residual of the equation over all vectors and states. The initial blind vectors point to themselves and
#      satisfy the same equation.
#
# 3. The Sampled Check (the consequence itself, independent of the pointers):
#    - (*) is evaluated at b0 and at the beliefs the reference agent and a uniformly random agent visit on 64 episodes
#      of 100 steps (simulator seed 11), for every action. Reported: the smallest slack of (*) over beliefs and
#      actions, which must not be negative beyond round-off.
#
# 4. Use:
#      CUDA_VISIBLE_DEVICES= uv run python reports/2026-11-16_code_audit/benchmark_improvability.py \
#          > reports/2026-11-16_code_audit/benchmark_improvability.log
# ==============================================================================

import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from ipomdp.domain import BatchedPOMDPEnv, belief_update
from ipomdp.domain.solver import AlphaVectorSet
from ipomdp.experiments.interactive_analysis import InteractiveBenchmark, interactive_problem, reference_sets

CPU = torch.device("cpu")
EPISODES, STEPS, SIMULATOR_SEED = 64, 100, 11


def visited_beliefs(pomdp, policy) -> torch.Tensor:
    """Exact beliefs along EPISODES episodes under `policy` (a map from beliefs to actions); (EPISODES x STEPS, S)."""
    env = BatchedPOMDPEnv(pomdp, EPISODES, STEPS, SIMULATOR_SEED, CPU)
    env.reset()
    belief = pomdp.initial_belief.expand(EPISODES, -1).clone()
    seen = []
    for _ in range(STEPS):
        seen.append(belief)
        action = policy(belief)
        belief = belief_update(pomdp, belief, action, env.step(action).observation)
    return torch.cat(seen)


def main() -> None:
    torch.set_num_threads(2)
    commit = subprocess.run(["git", "log", "-1", "--format=%h"], check=True, capture_output=True, text=True).stdout.strip()
    print(f"commit {commit}")
    for level in (1, 2):
        start = time.time()
        cfg = OmegaConf.load("conf/env/itiger.yaml").env
        cfg.level = level
        pomdp = interactive_problem(cfg).pomdp
        benchmark = InteractiveBenchmark.load(Path(f"runs/benchmarks/itiger_level{level}.pt"))
        value, per_action = reference_sets(benchmark, pomdp)                # refuses another fold's file
        stored = np.load(f"runs/benchmarks/itiger_level{level}.search.npz")
        vectors, actions = torch.from_numpy(stored["vectors"]), torch.from_numpy(stored["actions"]).long()
        successors = torch.from_numpy(stored["successors"]).long()
        if not (torch.equal(vectors, benchmark.vectors) and torch.equal(actions, benchmark.actions)):
            raise ValueError(f"level {level}: the checkpoint's vectors or actions differ from the benchmark file's.")
        gamma, T, O, R = pomdp.discount, pomdp.transition, pomdp.observation, pomdp.reward
        step = torch.einsum("ast,ato->aost", T, O)                           # M_{a,o}[s, s']
        print(f"\nlevel {level}: {pomdp.num_states} states, {len(vectors)} vectors, bracket [{benchmark.lower:.4f}, "
              f"{benchmark.upper:.4f}], fold digest {benchmark.digest[:16]}")

        # Section 2: every vector against its pointer equation.
        worst = 0.0
        for a in range(pomdp.num_actions):
            rows = actions == a
            backed = R[a][None].expand(int(rows.sum()), -1).clone()
            for o in range(pomdp.num_observations):
                backed += gamma * vectors[successors[rows, o]] @ step[a, o].T
            worst = max(worst, float((vectors[rows] - backed).abs().max()))
            print(f"  action {pomdp.action_names[a]}: {int(rows.sum())} vectors, largest |alpha_k - pointer backup| "
                  f"{float((vectors[rows] - backed).abs().max()):.3e}")
        print(f"  exact check: largest residual over all vectors and states {worst:.3e} "
              f"(values reach {float(vectors.abs().max()):.1f})")

        # Section 3: (*) at visited beliefs, every action.
        generator = torch.Generator().manual_seed(0)
        beliefs = torch.cat([pomdp.initial_belief[None], visited_beliefs(pomdp, value.greedy_action),
                             visited_beliefs(pomdp, lambda b: torch.randint(pomdp.num_actions, (len(b),),
                                                                            generator=generator))])
        slack = []
        for a in range(pomdp.num_actions):
            lookahead = beliefs @ R[a]
            for o in range(pomdp.num_observations):
                lookahead = lookahead + gamma * value.value(beliefs @ step[a, o])      # L is positively homogeneous
            slack.append(lookahead - per_action[a].value(beliefs))
        slack = torch.stack(slack, -1)
        best = AlphaVectorSet(benchmark.vectors, benchmark.actions).greedy_action(beliefs)
        own = slack.gather(1, best[:, None]).squeeze(1)
        print(f"  sampled check at {len(beliefs)} beliefs (b0, the reference agent's, a random agent's): smallest "
              f"slack of (*) over all actions {float(slack.min()):.3e}; at the reference agent's own action: smallest "
              f"{float(own.min()):.3e}, mean {float(own.mean()):.4f}")
        print(f"  ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
