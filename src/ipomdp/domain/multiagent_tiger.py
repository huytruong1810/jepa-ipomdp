# ABSOLUTE PATH: src/ipomdp/domain/multiagent_tiger.py
# ==============================================================================
# TWO-AGENT TIGER: THE JOINT PHYSICS OF GMYTRASIEWICZ & DOSHI (2005), TABLE 1
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. What This Module Is (and Is Not):
#    - The physics of the multi-agent tiger game: joint transition, each agent's observation and reward tables, as
#      Table 1 of Gmytrasiewicz & Doshi, "A Framework for Sequential Planning in Multi-Agent Settings", JAIR 24
#      (2005), Section 7.1, states them (also Table 1 of Doshi & Gmytrasiewicz, JAIR 34, 2009). It holds no model of
#      either agent's mind. An agent's decision problem against a modelled other agent is a FinitePOMDP built by
#      domain/interactive.py, which folds the other agent's controller into these tables.
#    - The canonical single-agent Tiger (domain/tiger.py, Cassandra's tiger.95.POMDP) is not touched and not reused
#      as a table: the joint tables are written here entry by entry in the paper's layout, and the degenerate fold
#      (the other agent always listens) must reproduce the canonical values, which is the test that ties the two.
#
# 2. The Tables (Table 1; every entry is a decimal literal, so float64 sums are exact to ~1e-16):
#    - States TL, TR; actions L, OL, OR for each agent; prior 0.5 / 0.5.
#    - Transition T(s' | s, a_i, a_j): if either agent opens a door, the tiger is re-placed uniformly; if both
#      listen, it stays.
#    - Observation of agent k, O_k(o_k | s', a_i, a_j), over the six pairs (growl, creak) in Table 1's column order
#      <GL,CL>, <GL,CR>, <GL,S>, <GR,CL>, <GR,CR>, <GR,S>. It depends on the ARRIVAL state s' and on both actions:
#      * k opened a door: all six are equally likely (1/6);
#      * k listened: the growl names the tiger's side with probability 0.85, and independently the creak names the
#        other agent's action (CL for OL, CR for OR, S for L) with probability 0.9, each other creak 0.05.
#    - Reward R_k(s, a_i, a_j): k's own action only, on the pre-transition state, as in the single-agent game:
#      listen -1, open the tiger's door -100, open the other door +10. The other agent matters to k only through
#      the state it causes (resets) and the creaks it produces.
#    - o_i and o_j are conditionally independent given (s', a_i, a_j): the paper specifies O_i and O_j separately,
#      and the fold multiplies them (interactive.py, section 2).
#
# 3. Joint Layout and Perspectives:
#    - Every table is indexed joint-action first, (a_i, a_j, ...), as Table 1 is. An agent's decision problem needs
#      the tables from its own side: AgentView reorders the action axes so the agent's own action comes first
#      (own, other, ...). The game is symmetric, but the code never relies on it: both views are built from the
#      joint tables, so an asymmetric variant needs no other change.
#
# 4. Discount:
#    - The paper's examples are finite-horizon; Doshi & Gmytrasiewicz (2009) use gamma = 0.9 for their published
#      values. The discount is therefore a parameter of the build, not a constant; the learned-agent experiments use
#      the canonical Tiger's 0.95.
# ==============================================================================

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

from .tiger import LISTEN_ACCURACY, LISTEN_ERROR, LISTEN_REWARD, TIGER_REWARD, TREASURE_REWARD, TigerAction, TigerState

# Creak accuracy (Table 1): the correct creak 0.9, each of the two others 0.05. Written as literals, like the growl
# accuracy in tiger.py, so the products below are the paper's entries to the last bit.
CREAK_ACCURACY = 0.9
CREAK_ERROR = 0.05

_STOCHASTIC_ATOL = 1e-12


class Agent(IntEnum):
    """The two agents; their joint actions are indexed (I, J) in every table (section 3)."""

    I = 0  # noqa: E741 - the paper's name for the modelling agent
    J = 1


class Growl(IntEnum):
    GROWL_LEFT = 0
    GROWL_RIGHT = 1


class Creak(IntEnum):
    CREAK_LEFT = 0
    CREAK_RIGHT = 1
    SILENCE = 2


# The creak that correctly reports each action of the other agent (Table 1).
TRUE_CREAK = {TigerAction.LISTEN: Creak.SILENCE, TigerAction.OPEN_LEFT: Creak.CREAK_LEFT,
              TigerAction.OPEN_RIGHT: Creak.CREAK_RIGHT}
# Table 1's observation columns: growl-major, creak-minor.
OBSERVATIONS = tuple((g, c) for g in Growl for c in Creak)
OBSERVATION_NAMES = tuple(f"{g.name}|{c.name}" for g, c in OBSERVATIONS)


@dataclass(frozen=True)
class AgentView:
    """
    The joint tables seen from one agent's side: action axes ordered (own, other).

    Attributes:
        agent: Whose side this is.
        transition: T(s' | s, a_own, a_other), shape (A, A, S, S).
        own_observation: O(o_own | s', a_own, a_other), shape (A, A, S, O).
        other_observation: O(o_other | s', a_own, a_other), shape (A, A, S, O).
        own_reward: R_own(s, a_own, a_other), shape (A, A, S).
        initial_state: b0(s), shape (S,).
        discount: gamma.
        state_names, action_names, observation_names: Shared by both agents.
    """

    agent: Agent
    transition: Tensor
    own_observation: Tensor
    other_observation: Tensor
    own_reward: Tensor
    initial_state: Tensor
    discount: float
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    observation_names: tuple[str, ...]


@dataclass(frozen=True)
class TwoAgentModel:
    """
    Joint physics of a two-agent POMDP, float64, joint actions indexed (a_i, a_j) (section 3).

    Attributes:
        transition: T(s' | s, a_i, a_j), shape (A, A, S, S).
        observation: O_k(o_k | s', a_i, a_j) per agent, shape (2, A, A, S, O).
        reward: R_k(s, a_i, a_j) per agent, shape (2, A, A, S).
        initial_state: b0(s), shape (S,).
        discount: gamma in (0, 1).
    """

    transition: Tensor
    observation: Tensor
    reward: Tensor
    initial_state: Tensor
    discount: float
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    observation_names: tuple[str, ...]

    def __post_init__(self) -> None:
        s, a, o = len(self.state_names), len(self.action_names), len(self.observation_names)
        shapes = {"transition": (a, a, s, s), "observation": (2, a, a, s, o), "reward": (2, a, a, s),
                  "initial_state": (s,)}
        for name, shape in shapes.items():
            tensor = getattr(self, name)
            if tensor.dtype != torch.float64 or tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must be float64 of shape {shape}, got {tensor.dtype} {tuple(tensor.shape)}.")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite entries.")
        for name in ("transition", "observation", "initial_state"):
            tensor = getattr(self, name)
            sums = tensor.sum(-1)
            if (tensor < 0).any() or not torch.allclose(sums, torch.ones_like(sums), atol=_STOCHASTIC_ATOL, rtol=0.0):
                raise ValueError(f"{name} must be non-negative with rows summing to 1.")
        if not 0.0 < self.discount < 1.0:
            raise ValueError(f"discount must lie in (0, 1), got {self.discount}.")

    def view(self, agent: Agent) -> AgentView:
        """The tables from `agent`'s side, own action first (section 3)."""
        own, other = int(agent), 1 - int(agent)
        swap = (lambda t: t) if agent == Agent.I else (lambda t: t.transpose(0, 1).contiguous())
        return AgentView(agent, swap(self.transition), swap(self.observation[own]), swap(self.observation[other]),
                         swap(self.reward[own]), self.initial_state, self.discount, self.state_names,
                         self.action_names, self.observation_names)


def _listener_observation(tiger: TigerState, other_action: TigerAction) -> Tensor:
    """O(. | s', listener, other's action) over Table 1's six columns (section 2)."""
    growl = {Growl.GROWL_LEFT: LISTEN_ACCURACY if tiger == TigerState.TIGER_LEFT else LISTEN_ERROR,
             Growl.GROWL_RIGHT: LISTEN_ACCURACY if tiger == TigerState.TIGER_RIGHT else LISTEN_ERROR}
    creak = {c: CREAK_ACCURACY if c == TRUE_CREAK[other_action] else CREAK_ERROR for c in Creak}
    return torch.tensor([growl[g] * creak[c] for g, c in OBSERVATIONS], dtype=torch.float64)


def build_multiagent_tiger(discount: float) -> TwoAgentModel:
    """
    The multi-agent tiger game of Gmytrasiewicz & Doshi (2005), Table 1 (section 2).

    Args:
        discount: gamma (section 4).

    Returns:
        TwoAgentModel with |S| = 2, |A| = 3 per agent and |O| = 6 per agent.
    """
    states, actions = list(TigerState), list(TigerAction)
    num_s, num_a, num_o = len(states), len(actions), len(OBSERVATIONS)
    transition = torch.empty(num_a, num_a, num_s, num_s, dtype=torch.float64)
    observation = torch.empty(2, num_a, num_a, num_s, num_o, dtype=torch.float64)
    reward = torch.empty(2, num_a, num_a, num_s, dtype=torch.float64)
    own_reward = {TigerAction.LISTEN: {TigerState.TIGER_LEFT: LISTEN_REWARD, TigerState.TIGER_RIGHT: LISTEN_REWARD},
                  TigerAction.OPEN_LEFT: {TigerState.TIGER_LEFT: TIGER_REWARD, TigerState.TIGER_RIGHT: TREASURE_REWARD},
                  TigerAction.OPEN_RIGHT: {TigerState.TIGER_LEFT: TREASURE_REWARD, TigerState.TIGER_RIGHT: TIGER_REWARD}}
    for a_i in actions:
        for a_j in actions:
            both_listen = a_i == TigerAction.LISTEN and a_j == TigerAction.LISTEN
            transition[a_i, a_j] = (torch.eye(num_s, dtype=torch.float64) if both_listen
                                    else torch.full((num_s, num_s), 0.5, dtype=torch.float64))
            for agent, own, other in ((Agent.I, a_i, a_j), (Agent.J, a_j, a_i)):
                for s in states:
                    observation[agent, a_i, a_j, s] = (_listener_observation(s, other) if own == TigerAction.LISTEN
                                                       else torch.full((num_o,), 1.0 / 6.0, dtype=torch.float64))
                    reward[agent, a_i, a_j, s] = own_reward[own][s]
    return TwoAgentModel(transition, observation, reward, torch.full((num_s,), 0.5, dtype=torch.float64), discount,
                         tuple(s.name for s in states), tuple(a.name for a in actions), OBSERVATION_NAMES)
