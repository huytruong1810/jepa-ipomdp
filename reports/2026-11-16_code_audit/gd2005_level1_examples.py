# ABSOLUTE PATH: reports/2026-11-16_code_audit/gd2005_level1_examples.py
# ==============================================================================
# THE LEVEL-1 EXAMPLES OF GMYTRASIEWICZ & DOSHI (2005), SECTION 7.3, AGAINST OUR FOLD (CODE AUDIT, PHASE 1; CPU, SECONDS)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. What the Paper Prints and What Was Already Checked:
#    - Table 1 (the joint transition, observation and reward tables) is reproduced exactly
#      (src/ipomdp/domain/multiagent_tiger.py; tests/test_interactive.py, TestTableOne), and an independent
#      implementation agrees on Q_h to 2.8e-14 (ipomcp_crosscheck.py).
#    - Fig. 4, the horizon-3 policy of the POMDP with noise (level 0, undiscounted), is reproduced within 0.003
#      (tests/test_interactive.py, test_level_zero_policy_matches_the_published_noise_pomdp_thresholds).
#    - Two level-1 statements of Section 7.3 were open. Both are for agent i at level 1 with horizon 3, undiscounted,
#      when i has no information about j's belief (a uniform density over p_j(TL), the paper's Fig. 5 (ii)):
#        (a) "with two steps left to go, j will perform its actions OL, L, and OR with probabilities 0.009076, 0.96591
#            and 0.02501", after i listened and heard <GL,S>;
#        (b) Fig. 11, i's policy graph: seven regions of p_i(TL) with boundaries 0.029, 0.089, 0.211, 0.789, 0.911,
#            0.971, of which the two outer regions open a door.
#
# 2. Method:
#    - j at level 0 is the POMDP with noise (the other agent listens with 0.8 and opens each door with 0.1) with as
#      many steps to go as the interaction has left: its horizon-3 plan graph (policy_graph.plan_graph; 3 + 5 + 7
#      vectors and a sink, 16 nodes). i's prior over j's first node is the length of each top-layer region of p_j
#      (node_law_on_segment). The fold (interactive.fold) is i's exact POMDP over (tiger side, j's node, j's last
#      action): 44 states at horizon 3.
#    - (a) is one exact Bayes update of the fold from p_i = 0.5 and a sum over j's nodes.
#    - (b) does NOT solve the fold by alpha vectors: an earlier attempt at that solve was killed for memory
#      (solver.prune compares all pairs of vectors). Fig. 11 needs the value only along one segment of beliefs,
#      b(p) = p (TL x law) + (1 - p) (TR x law). The undiscounted h-step value is positively homogeneous, so
#      W_h(u) = max_a [u . R_a + sum_o W_{h-1}(u T_a diag(O_a[., o]))] is evaluated on unnormalised beliefs by exact
#      expectimax (3 actions x 6 observations, depth 3) on a grid of p. The value is piecewise linear in p: a region
#      boundary is a kink, located exactly by intersecting the lines on its two sides.
#    - Checks of the method: at horizon 1 the kinks are 0.1 and 0.9 and the value overlaps the noisy POMDP (the
#      paper's Fig. 8); V_h(0) = V_h(1) = 10 - (h - 1), an immediate opening followed by listening.
#    - A second reading of level 0 is printed beside ours for (a): j as the single-agent POMDP (the other agent
#      always listens), because the paper's belief-update example refers to the single-agent value function.
#
# 3. Result (the log beside this file): neither statement is reproduced.
#    - (a) ours gives OL 0.0259, L 0.8983, OR 0.0758; the single-agent reading gives L 1. The paper's 0.966 lies
#      between the two.
#    - (b) ours opens a door below 0.0182 and above 0.9818 (the paper: 0.029 and 0.971), and the value has 16 kinks,
#      not 6. V_3(0) = V_3(1) = 8, as the axis of the paper's Fig. 10 shows.
#    - The difference is a convention of the paper's level-1 examples that we could not identify (how j's model and
#      j's steps to go enter i's update). It is reported as such. Nothing in the repository depends on it: the
#      experiments use Table 1, a discount of 0.95 and controllers certified by exact evaluation.
#
# 4. Use:
#      CUDA_VISIBLE_DEVICES= uv run python reports/2026-11-16_code_audit/gd2005_level1_examples.py \
#          > reports/2026-11-16_code_audit/gd2005_level1_examples.log
# ==============================================================================

import subprocess

import numpy as np
import torch

from ipomdp.domain import TigerAction, TigerState, belief_update, solve_finite_horizon
from ipomdp.domain.interactive import TIGER_LEVEL_ZERO_NOISE, InteractiveModel, fold, noise_model
from ipomdp.domain.multiagent_tiger import Agent, build_multiagent_tiger
from ipomdp.domain.policy_graph import PolicyGraph, envelope, node_law_on_segment, plan_graph, segment_breakpoints

F64 = torch.float64
LISTEN, TL, TR = int(TigerAction.LISTEN), int(TigerState.TIGER_LEFT), int(TigerState.TIGER_RIGHT)
GRID = 20001
PAPER_LAW = {"OL": 0.009076, "L": 0.96591, "OR": 0.02501}
PAPER_FIG_11 = [0.029, 0.089, 0.211, 0.789, 0.911, 0.971]


def onehot(index: int, size: int) -> torch.Tensor:
    vector = torch.zeros(size, dtype=F64)
    vector[index] = 1.0
    return vector


def level_one(noise: tuple[float, ...], horizon: int) -> tuple[PolicyGraph, InteractiveModel, list[list[float]]]:
    """j's undiscounted horizon-`horizon` plan graph under `noise`, i's fold of it, and j's thresholds per horizon."""
    physics = build_multiagent_tiger(0.5)                     # the discount is replaced below: the examples have none
    model_j = noise_model(physics.view(Agent.J), noise)
    object.__setattr__(model_j, "discount", 1.0)
    value_functions = [envelope(g) for g in solve_finite_horizon(model_j, horizon)]
    cuts = [[round(p, 4) for p, _ in segment_breakpoints(v.vectors, onehot(TR, 2), onehot(TL, 2))[:-1]]
            for v in value_functions]
    graph = plan_graph(model_j, value_functions)
    graph = graph.with_initial_law(node_law_on_segment(value_functions[-1].vectors, onehot(TR, 2), onehot(TL, 2), 0,
                                                       graph.num_nodes))
    return graph, fold(physics.view(Agent.I), graph, LISTEN), cuts


def action_law_after_growl_left_silence(graph: PolicyGraph, folded: InteractiveModel) -> tuple[float, list[float]]:
    """Statement (a): P_i(TL) and j's action law (L, OL, OR) after i listens and hears <GL,S> from p_i = 0.5."""
    model = folded.pomdp
    observation = model.observation_names.index("GROWL_LEFT|SILENCE")
    belief = belief_update(model, model.initial_belief[None], torch.tensor([LISTEN]), torch.tensor([observation]))
    law = (belief[0][:, None] * graph.action_law[folded.other_node]).sum(0)
    return float(folded.physical_marginal(belief)[0][TL]), [float(x) for x in law]


def segment_value(folded: InteractiveModel, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Statement (b): p, the undiscounted value V_h(b(p)) and the first action on a grid of p_i(TL) (section 2)."""
    model = folded.pomdp
    step = torch.einsum("ast,ato->aost", model.transition, model.observation)          # T_a diag(O_a[., o])

    def value(u: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        best, first = None, None
        for a in range(model.num_actions):
            q = u @ model.reward[a]
            if steps > 1:
                for o in range(model.num_observations):
                    q = q + value(u @ step[a, o], steps - 1)[0]
            if best is None:
                best, first = q, torch.zeros_like(q, dtype=torch.int64)
            else:
                better = q > best
                best, first = torch.where(better, q, best), torch.where(better, torch.full_like(first, a), first)
        return best, first

    p = torch.linspace(0.0, 1.0, GRID, dtype=F64)
    left = (folded.physical_state == TL).to(F64)
    law = model.initial_belief / 0.5                          # P(j's node, j's last action) on each side's states
    beliefs = p[:, None] * (law * left)[None] + (1.0 - p)[:, None] * (law * (1.0 - left))[None]
    best, first = value(beliefs, horizon)
    return p.numpy(), best.numpy(), first.numpy()


def kinks(p: np.ndarray, v: np.ndarray) -> list[float]:
    """The breakpoints of a piecewise-linear function sampled on a grid: the lines on the two sides, intersected."""
    slopes = np.diff(v) / np.diff(p)
    change = np.flatnonzero(np.abs(np.diff(slopes)) > 1e-7)
    found, k = [], 0
    while k < len(change):
        left = right = change[k]
        while k + 1 < len(change) and change[k + 1] == right + 1:      # a kink inside a cell touches two differences
            k += 1
            right = change[k]
        a, b = left, right + 1                                          # cells with the clean slope of each side
        found.append(float((v[b + 1] - v[a] + slopes[a] * p[a] - slopes[b] * p[b + 1]) / (slopes[a] - slopes[b])))
        k += 1
    return found


def main() -> None:
    commit = subprocess.run(["git", "log", "-1", "--format=%h"], check=True, capture_output=True, text=True).stdout.strip()
    print(f"commit {commit}; Gmytrasiewicz & Doshi (2005), Section 7.3: agent i at level 1, undiscounted, no information "
          "about j's belief (uniform over p_j)")
    print(f"\n(a) j's action law with two steps to go, after i listens and hears <GL,S> from p_i = 0.5; paper {PAPER_LAW}")
    for label, noise in (("j = the POMDP with noise (0.8, 0.1, 0.1), our level 0", TIGER_LEVEL_ZERO_NOISE),
                         ("j = the single-agent POMDP (the other agent always listens)", (1.0, 0.0, 0.0))):
        graph, folded, cuts = level_one(noise, 3)
        p_left, law = action_law_after_growl_left_silence(graph, folded)
        first = graph.initial_law @ graph.action_law
        print(f"  {label}")
        print(f"    j's thresholds over p_j(TL) with 1, 2, 3 steps to go: {cuts}")
        print(f"    j's action law at the first step: L {float(first[0]):.6f}, OL {float(first[1]):.6f}, OR {float(first[2]):.6f}")
        print(f"    after <GL,S>: P_i(TL) = {p_left:.5f}; j's law: OL {law[1]:.6f}, L {law[0]:.6f}, OR {law[2]:.6f}")

    print("\n(b) i's value along p_i(TL) with j = the POMDP with noise; kinks are the boundaries between optimal plans")
    for horizon in (1, 2, 3):
        graph, folded, _ = level_one(TIGER_LEVEL_ZERO_NOISE, horizon)
        p, v, first = segment_value(folded, horizon)
        names = folded.pomdp.action_names
        opens = [round(float(p[i]), 4) for i in range(1, GRID) if first[i] != first[i - 1]]
        print(f"  horizon {horizon}: fold of {folded.pomdp.num_states} states (j has {graph.num_nodes} nodes); "
              f"V(0) {v[0]:.4f}, V(0.5) {v[GRID // 2]:.4f}, V(1) {v[-1]:.4f}; first action {names[first[0]]} / "
              f"{names[first[GRID // 2]]} / {names[first[-1]]}, changing at {opens}")
        print(f"    kinks: {[round(c, 4) for c in kinks(p, v)]}")
    print(f"  paper, Fig. 11 (horizon 3): {PAPER_FIG_11}, opening a door outside [0.029, 0.971]")


if __name__ == "__main__":
    main()
