# ABSOLUTE PATH: src/ipomdp/domain/interactive.py
# ==============================================================================
# FINITELY NESTED I-POMDPS AS POMDPS: FOLDING THE OTHER AGENT'S CONTROLLER INTO THE PHYSICS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. The Reduction:
#    - In a finitely nested I-POMDP_{i,l} (Gmytrasiewicz & Doshi 2005, Definition 4), agent i's interactive state is
#      IS_{i,l} = S x M_{j,l-1}: the physical state and a model of j. When the models of j that i entertains are
#      the nodes of one finite policy graph (domain/policy_graph.py; j's beliefs are summarised by the node its
#      controller is in, and i's uncertainty about j's beliefs is a law over nodes), i's problem is an ordinary
#      POMDP over S x N_j x A_j:
#         T~(s', n', a'_j | s, n, a_prev; a_i)
#             = psi[n, a'_j] T(s' | s, a_i, a'_j) sum_{o_j} O_j(o_j | s', a_i, a'_j) eta[n, a'_j, o_j, n'],
#         O~(o_i | s', n', a'_j; a_i) = O_i(o_i | s', a_i, a'_j),
#         R~(s, n, a_prev; a_i)       = sum_{a_j} psi[n, a_j] R_i(s, a_i, a_j).
#      The previous action of j sits in the post-state only because i's observation (its creak) depends on it and
#      FinitePOMDP emits observations from the arrival state; T~ and R~ never read it. So the exact Bayes filter,
#      the exact solver, the batched simulator and the learned agent of this package apply to the interactive
#      problem without any change: the I-POMDP extension path that pomdp.py's header promised.
#    - j's node moves on j's OWN observation, drawn from the true physics with i's true action. That is the
#      semantics of "j executes its controller": j's controller was computed against j's model of i (noise at level
#      0, i's level-(l-2) controller above), but it reacts to what j actually hears.
#    - o_i and o_j are conditionally independent given (s', a_i, a_j) (multiagent_tiger.py, section 2), so i's
#      observation law and j's node update multiply.
#
# 2. Levels and the Recursion:
#    - Level 0 (Gmytrasiewicz & Doshi 2005, footnotes 13 and 17: "the other agent's actions are folded into the T, O
#      and R functions as noise"; "the POMDP with noise is the same as level-0 I-POMDP"): noise_model averages each
#      table separately over a fixed action law of the other agent. For the multi-agent tiger game the paper's noise
#      is OL 0.1, L 0.8, OR 0.1, which is what a horizon-1 level-0 agent with a uniform prior over its belief does.
#      Averaging the tables separately is the paper's definition and it is what reproduces the paper's Fig. 4
#      thresholds; it is not the same problem as fold() with a one-node noise controller, which keeps the
#      correlation between a noise action, the reset it causes and the creak that reports it (a better-informed
#      level-0 agent than the paper's).
#    - Level l: solve the other agent's level-(l-1) problem, extract its controller (exactly, as a plan tree, for a
#      finite horizon; stationary and certified for the infinite horizon), and fold that controller into this
#      agent's view. Each level is therefore an ordinary POMDP, and the exactness claim is precise: the solution is
#      optimal against the other agent executing its extracted controller.
#
# 3. Unreachable Interactive States Are Dropped:
#    - The full grid S x N x A is built, then only the states reachable from the initial belief under some sequence
#      of actions are kept (breadth-first over the support of T~). With a deterministic controller most (n', a'_j)
#      pairs cannot occur (a'_j is the action of the node j left); keeping them would only inflate the solver's
#      linear programs. Reachable rows of T~ put all their mass on reachable states, so the kept model is exact.
#
# 4. The Initial Previous Action Is a Convention:
#    - Before the first step there is no previous action of j; the initial belief puts it on one declared action.
#      Nothing reads it (T~ and R~ ignore it, and no observation precedes the first action), so values are
#      unaffected, but it is part of the joint belief; probes therefore read the S x N_j marginal.
# ==============================================================================

from dataclasses import dataclass

import torch
from torch import Tensor

from .bounds import Bracket, hsvi
from .multiagent_tiger import Agent, AgentView, TwoAgentModel
from .policy_graph import PolicyGraph, evaluate, stationary_graph
from .pomdp import FinitePOMDP
from .reduction import expand_observations, lump_observations
from .solver import AlphaVectorSet

# The level-0 noise of the two-agent tiger game (Gmytrasiewicz & Doshi 2005, Section 7.3 and footnote 3): the other
# agent listens with probability 0.8 and opens each door with probability 0.1, in TigerAction order (L, OL, OR).
TIGER_LEVEL_ZERO_NOISE = (0.8, 0.1, 0.1)

# A lower level's controller is extracted from its bracket's lower-bound vectors pruned at this fraction of the
# bracket width (policy_graph.stationary_graph): small enough to fold one level up, and its exact value stays
# within the bracket's scale of the lower bound. The controller is certified by U - V_controller in either case.
CONTROLLER_PRUNE_FRACTION = 0.1


@dataclass(frozen=True)
class InteractiveModel:
    """
    An agent's decision problem against a modelled other agent, as a FinitePOMDP over the reachable interactive
    states (section 3).

    Attributes:
        pomdp: The folded problem.
        physical_state: s of each interactive state, int64 (IS,).
        other_node: The other agent's controller node, int64 (IS,).
        other_previous_action: The other agent's previous action, int64 (IS,).
        other: The folded controller.
        num_physical_states: |S|.
    """

    pomdp: FinitePOMDP
    physical_state: Tensor
    other_node: Tensor
    other_previous_action: Tensor
    other: PolicyGraph
    num_physical_states: int

    def physical_marginal(self, belief: Tensor) -> Tensor:
        """P(s) of interactive beliefs of shape (B, IS); returns (B, S)."""
        index = self.physical_state.to(belief.device).expand_as(belief)
        return torch.zeros(belief.shape[0], self.num_physical_states, dtype=belief.dtype, device=belief.device
                           ).scatter_add_(1, index, belief)

    def other_node_marginal(self, belief: Tensor) -> Tensor:
        """P(other agent's node) of interactive beliefs of shape (B, IS); returns (B, N)."""
        index = self.other_node.to(belief.device).expand_as(belief)
        return torch.zeros(belief.shape[0], self.other.num_nodes, dtype=belief.dtype, device=belief.device
                           ).scatter_add_(1, index, belief)


def noise_model(view: AgentView, noise: tuple[float, ...]) -> FinitePOMDP:
    """
    A level-0 problem as the paper defines it (section 2): the other agent's actions folded into T, O and R as a fixed
    action law, each table averaged separately. Its creaks then depend on the noise only, not on the state, so they
    carry no evidence (domain/reduction.py lumps them). The law is taken as Python floats and built in float64 here:
    a float32 tensor of 0.8 is 0.800000011920929, which the stochasticity check rightly rejects.
    """
    noise = torch.tensor(noise, dtype=torch.float64)
    return FinitePOMDP(torch.einsum("y,xyst->xst", noise, view.transition),
                       torch.einsum("y,xyto->xto", noise, view.own_observation),
                       torch.einsum("y,xys->xs", noise, view.own_reward), view.initial_state, view.discount,
                       view.state_names, view.action_names, view.observation_names)


def fold(view: AgentView, other: PolicyGraph, initial_previous_action: int) -> InteractiveModel:
    """
    The agent of `view` facing the other agent executing `other` (section 1), reachable states only (section 3).

    Args:
        view: The joint physics from this agent's side (multiagent_tiger.AgentView).
        other: The other agent's controller over the other agent's actions and observations; its initial law is
            this agent's prior over the other's first node, independent of the physical state.
        initial_previous_action: The convention of section 4.
    """
    num_s, num_a = view.transition.shape[2], view.transition.shape[0]
    num_n = other.num_nodes
    if (other.num_actions, other.num_observations) != (view.transition.shape[1], view.other_observation.shape[-1]):
        raise ValueError("the controller's actions and observations must be the other agent's.")
    psi, eta = other.action_law, other.successor_law                                    # (N, A_j), (N, A_j, O_j, N)
    # node[x, y, t, n, m] = sum_o O_j(o | t, x, y) eta[n, y, o, m]: x own action, y other's, t arrival state.
    node = torch.einsum("xyto,nyom->xytnm", view.other_observation, eta)
    # core[x, s, n, t, m, y] = psi[n, y] T(t | s, x, y) node[x, y, t, n, m]
    core = torch.einsum("ny,xyst,xytnm->xsntmy", psi, view.transition, node)
    full = core.unsqueeze(3).expand(num_a, num_s, num_n, psi.shape[1], num_s, num_n, psi.shape[1])
    size = num_s * num_n * psi.shape[1]
    transition = full.reshape(num_a, size, size)
    # Index helpers over the (s, n, a_j) grid, in that order.
    grid_s = torch.arange(num_s).repeat_interleave(num_n * psi.shape[1])
    grid_n = torch.arange(num_n).repeat_interleave(psi.shape[1]).repeat(num_s)
    grid_a = torch.arange(psi.shape[1]).repeat(num_s * num_n)
    observation = view.own_observation[:, grid_a, grid_s, :]                            # (A, size, O)
    reward = torch.einsum("ny,xys->xsn", psi, view.own_reward)[:, grid_s, grid_n]       # (A, size)
    initial = (view.initial_state[grid_s] * other.initial_law[grid_n]
               * (grid_a == initial_previous_action).to(torch.float64))
    # Breadth-first reachability over the support of T~ under any action (section 3).
    reachable = initial > 0
    frontier = reachable.clone()
    support = (transition > 0).any(dim=0)                                               # (size, size)
    while frontier.any():
        nxt = support[frontier].any(dim=0) & ~reachable
        reachable |= nxt
        frontier = nxt
    keep = torch.nonzero(reachable).flatten()
    names = tuple(f"{view.state_names[s]}|{other.node_names[n]}|{view.action_names[a]}"
                  for s, n, a in zip(grid_s[keep].tolist(), grid_n[keep].tolist(), grid_a[keep].tolist()))
    pomdp = FinitePOMDP(transition[:, keep][:, :, keep].contiguous(), observation[:, keep].contiguous(),
                        reward[:, keep].contiguous(), initial[keep].contiguous(), view.discount, names,
                        view.action_names, view.observation_names)
    return InteractiveModel(pomdp, grid_s[keep], grid_n[keep], grid_a[keep], other, num_s)


@dataclass(frozen=True)
class Level:
    """
    One certified rung of the nested hierarchy below the modelled agent's own problem (section 2).

    Attributes:
        level: Strategy level l of this agent's model.
        agent: Whose problem this is.
        problem: This agent's decision problem: the paper's noise model at level 0 (its uninformative creaks merged,
            domain/reduction.py), the fold of the other agent's level-(l-1) controller above.
        bracket: Certified [L, U] of this problem's V*(b0) (domain/bounds.py).
        controller: This agent's compact stationary controller over its own six observations: the model of this
            agent that the other agent folds one level up.
        controller_value: The controller's exact value at b0; bracket.upper - controller_value bounds its
            suboptimality.
    """

    level: int
    agent: Agent
    problem: FinitePOMDP
    bracket: Bracket
    controller: PolicyGraph
    controller_value: float


@dataclass(frozen=True)
class NestedProblem:
    """
    An agent's level-l problem (level >= 1) with the certified levels 0..l-1 that define it.

    Attributes:
        agent: The modelling agent.
        level: l.
        interactive: The fold of the level-(l-1) controller of the other agent into this agent's view.
        below: Levels 0..l-1, alternating agents, the last one the other agent's.
    """

    agent: Agent
    level: int
    interactive: InteractiveModel
    below: tuple[Level, ...]

    @property
    def pomdp(self) -> FinitePOMDP:
        return self.interactive.pomdp


def _certified_level(level: int, agent: Agent, problem: FinitePOMDP, group: Tensor | None, epsilon: float) -> Level:
    """Bracket the problem, extract the agent's compact controller from it and evaluate it exactly (section 2)."""
    bracket = hsvi(problem, epsilon)
    compact = stationary_graph(problem, AlphaVectorSet(bracket.lower_vectors, bracket.controller.action_law.argmax(-1)),
                               CONTROLLER_PRUNE_FRACTION * epsilon)
    value = float(compact.initial_law @ (evaluate(problem, compact) @ problem.initial_belief))
    controller = expand_observations(compact, group) if group is not None else compact
    return Level(level, agent, problem, bracket, controller, value)


def nested_problem(physics: TwoAgentModel, agent: Agent, level: int, noise: tuple[float, ...], epsilon: float,
                   initial_previous_action: int) -> NestedProblem:
    """
    `agent`'s level-`level` problem by the recursion of section 2, bottom-up: level 0 is the noise model of the agent
    at the bottom of the chain, and each level folds the controller of the level below into the other agent's view.
    Each level below is bracketed to `epsilon` and contributes its compact controller; the top problem itself is
    not bracketed here (the benchmark of an experiment brackets it once, and training needs only the problem).
    """
    if level < 1:
        raise ValueError(f"a nested problem has level >= 1 (level 0 is noise_model), got {level}.")
    below: list[Level] = []
    for rung in range(level):
        rung_agent = agent if (level - rung) % 2 == 0 else Agent(1 - int(agent))
        view = physics.view(rung_agent)
        if rung == 0:
            lumped = lump_observations(noise_model(view, noise))
            below.append(_certified_level(rung, rung_agent, lumped.model, lumped.group, epsilon))
        else:
            folded = fold(view, below[-1].controller, initial_previous_action)
            below.append(_certified_level(rung, rung_agent, folded.pomdp, None, epsilon))
    interactive = fold(physics.view(agent), below[-1].controller, initial_previous_action)
    return NestedProblem(agent, level, interactive, tuple(below))
