# ABSOLUTE PATH: src/ipomdp/baselines/ba_pomcp.py
# ==============================================================================
# BAYES-ADAPTIVE POMCP (KATT, OLIEHOEK & AMATO 2017): THE MODEL-LEARNING PLANNING BASELINE
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Why This Baseline:
#    - Our agent learns a model of an unknown POMDP from experience and plans with it as if it were exact
#      (certainty equivalence): it never values an action for what it would teach about the model. The
#      Bayes-adaptive POMDP (BA-POMDP; Ross, Chaib-draa & Pineau 2007; Ross et al. 2011) plans in the belief over
#      the model itself, so exploring to learn the model is part of the optimal plan. BA-POMCP (Katt, Oliehoek &
#      Amato, ICML 2017; arXiv 1806.05631, the extended version, read at the source) is its Monte Carlo tree
#      search solver: the state-of-the-art competitor for learning while planning in POMDPs.
#    - It is told what our agent is not: the state space, the reward function and a prior over the dynamics. Every
#      comparison states that advantage.
#
# 2. The BA-POMDP With Factored Counts (paper, section 2.2 and footnote 1):
#    - The augmented state is (s, chi), chi the Dirichlet counts of the dynamics. The paper writes one count vector
#      per (s, a) over (s', z) and uses the usual factorisation in its experiments; so does this module:
#         chi_T[a, s, s']  transition counts,     chi_O[a, s', z]  observation counts.
#    - Expected dynamics (paper, Eq. 1):  P(s' | s, a) = chi_T[a, s, s'] / sum_s'' chi_T[a, s, s''],
#                                         P(z | a, s') = chi_O[a, s', z] / sum_z' chi_O[a, s', z'].
#      A step of the BA-POMDP samples from them and adds one count to each entry it used. The reward R(s, a) is
#      known (paper, section 2.2).
#    - Zero counts are structural zeros, known impossibilities: they stay zero for ever. An observation that every
#      particle's counts make impossible cannot be accepted by the belief update (section 3), which then raises.
#
# 3. Beliefs Are Particles Over (s, chi), Updated by Rejection Sampling (paper, sections 2.1 and 3):
#    - K particles (paper, Table 1: 1000), each a hidden state with its own counts. After a real step (a, z): draw a
#      particle uniformly, sample s' and z' from its expected dynamics, and keep (s', chi + the two increments) when
#      z' = z, until K particles are kept. A cap on the attempts raises instead of reinvigorating the particle set:
#      a silent fallback would change the method.
#    - Between episodes the counts carry over and every particle's state is redrawn from b0: the agent keeps what it
#      has learned about the dynamics and starts each game uninformed about the hidden state.
#
# 4. Search (paper, Algorithms 1, 2 and 5; POMCP, Silver & Veness 2010):
#    - Each simulation samples a root particle and simulates the BA-POMDP from it on a copy of its counts. A step
#      uses the EXPECTED-MODEL variant (paper, Algorithm 5): sample s' and z from the expected dynamics and add the
#      counts to the copy. In distribution this is the basic BA-POMCP step (Algorithm 3: draw D ~ Dir(chi), then
#      (s', z) ~ D), because a Dirichlet draw followed by a categorical draw is the Dirichlet-multinomial
#      predictive; the paper reports that its variants give the same Tiger results.
#    - Action selection is UCB1: U(h, a) = Q(h, a) + c sqrt(log(N(h) + 1) / N(h, a)), untried actions first in
#      index order, with c = H (max R - min R) for the planning horizon H (paper, Table 1).
#    - One node is added per simulation. Below it, a rollout runs to the planning horizon on the same BA-POMDP; the
#      paper does not state its rollout policy, so it is an argument (section 4b).
#    - The action played is the greedy argmax_a Q(root, a) (paper, Algorithm 1), first maximum on ties.
#
# 4b. Rollout Policies (Rollout):
#    - UNIFORM: uniformly random actions, POMCP's default (Silver & Veness 2010) and the faithful reading of the paper.
#    - EXPECTED_MDP: the optimal policy of the MDP of the posterior-mean dynamics (the particle average of the expected
#      dynamics, section 2) with the known reward, solved by value iteration once per decision and applied to the
#      rollout's own (simulated, hence known) state. It uses only what the agent itself has: its prior, its data and
#      the reward function; it is the rollout counterpart of the QMDP heuristic (Littman, Cassandra & Kaelbling 1995).
#    - Why both: on the canonical, CONTINUING Tiger (a door re-places the tiger and play goes on) a uniformly random
#      continuation is worth about -380 over 20 steps, whatever the belief. Opening a door hands the search that
#      continuation one step earlier than listening does, so the search prefers to listen even when the tiger's side
#      is almost certain: on the TRUE model (point-mass prior), at P(tiger left) = 0.9999 and 10,000 simulations,
#      Q(LISTEN) = -326 against Q(OPEN_RIGHT) = -333 (3 of 3 seeds listen; at 1,000 simulations 1 of 3), and at
#      P = 0.97, where the optimum opens by a margin of 0.72, it listens at both budgets. A door that ENDS the game
#      (the episodic Tiger of many BA-POMDP studies) would not have this bias. The EXPECTED_MDP rollout values every
#      continuation by a policy that opens the right door, which removes it; the comparison reports both.
#
# 5. The Planning Horizon Is an Argument:
#    - The paper's Tiger horizon is 20 steps. The protocol that compares this baseline with our agent fixes H; a
#      search never plans past the steps left in the episode window (act's steps_left).
#
# 6. Reproducibility and Precision:
#    - One seeded numpy Generator per agent draws every random number (particle draws, simulations, rollouts and the
#      belief update). Counts and probabilities are float64.
# ==============================================================================

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch
from torch import Tensor

from ..domain import FinitePOMDP

# Value iteration of the EXPECTED_MDP rollout policy (section 4b) stops when the sup-norm change is below this.
MDP_TOLERANCE = 1e-10


class Rollout(StrEnum):
    """The rollout policy below the search tree (section 4b)."""

    UNIFORM = "uniform"                # uniformly random actions (POMCP's default)
    EXPECTED_MDP = "expected_mdp"      # the optimal policy of the posterior-mean MDP


@dataclass(frozen=True)
class DirichletPrior:
    """
    Dirichlet counts of the dynamics (section 2).

    Attributes:
        transition: chi_T[a, s, s'], float64 (A, S, S), non-negative, every row with a positive sum.
        observation: chi_O[a, s', z], float64 (A, S, Z), non-negative, every row with a positive sum.
    """

    transition: np.ndarray
    observation: np.ndarray

    def __post_init__(self) -> None:
        for name, counts in (("transition", self.transition), ("observation", self.observation)):
            if counts.dtype != np.float64 or counts.ndim != 3:
                raise ValueError(f"{name} counts must be a float64 array of rank 3, got {counts.dtype} {counts.shape}.")
            if not np.isfinite(counts).all() or (counts < 0).any() or not (counts.sum(-1) > 0).all():
                raise ValueError(f"{name} counts must be finite and non-negative, with a positive sum in every row.")
        if self.transition.shape[:2] != self.observation.shape[:2] or self.transition.shape[1] != self.transition.shape[2]:
            raise ValueError(f"shapes disagree: transition {self.transition.shape}, observation {self.observation.shape}.")

    @staticmethod
    def from_model(model: FinitePOMDP, transition_total: float, observation_total: float) -> "DirichletPrior":
        """Counts proportional to the true tables, each row summing to its total: a point mass as the total grows."""
        if not (transition_total > 0 and observation_total > 0):
            raise ValueError("the count totals must be positive.")
        return DirichletPrior(model.transition.numpy() * transition_total, model.observation.numpy() * observation_total)

    def with_observation_counts(self, action: int, counts: np.ndarray) -> "DirichletPrior":
        """A copy whose observation counts after `action` are replaced by `counts`, float64 (S, Z)."""
        observation = self.observation.copy()
        observation[action] = counts
        return DirichletPrior(self.transition.copy(), observation)

    def with_transition_counts(self, action: int, counts: np.ndarray) -> "DirichletPrior":
        """A copy whose transition counts of `action` are replaced by `counts`, float64 (S, S)."""
        transition = self.transition.copy()
        transition[action] = counts
        return DirichletPrior(transition, self.observation.copy())


class _Node:
    """A history node of the search tree: visit counts, action values and children keyed by (action, observation)."""

    __slots__ = ("visits", "action_visits", "values", "children")

    def __init__(self, num_actions: int):
        self.visits = 0
        self.action_visits = [0] * num_actions
        self.values = [0.0] * num_actions
        self.children: dict[tuple[int, int], "_Node"] = {}


def _draw(row: np.ndarray, u: float) -> int:
    """
    Index i of a count row with probability row[i] / sum(row), by inversion at u in [0, 1). The total and the running
    sum are accumulated in the same order, so the running sum reaches the total exactly at the last positive entry and
    a structural zero is never returned.
    """
    values = row.tolist()
    total, last_positive = 0.0, -1
    for i, v in enumerate(values):
        total += v
        if v > 0.0:
            last_positive = i
    if last_positive < 0:
        raise ValueError("a count row with a zero sum was sampled (section 2: every row has a positive sum).")
    target, acc = u * total, 0.0
    for i, v in enumerate(values):
        acc += v
        if target < acc:
            return i
    return last_positive          # u * total rounded up to the total itself: the top of the last positive interval


def _draw_rows(rows: np.ndarray, u: np.ndarray) -> np.ndarray:
    """_draw for a batch of count rows (n, k) at u (n,): the first index whose running sum exceeds u * total."""
    cumulative = rows.cumsum(-1)
    above = cumulative > u[:, None] * cumulative[:, -1:]
    index = above.argmax(-1)
    rounded = ~above.any(-1)                                   # u * total rounded up to the total (see _draw)
    if rounded.any():
        positive = rows[rounded] > 0.0
        index[rounded] = rows.shape[-1] - 1 - positive[:, ::-1].argmax(-1)
    return index


class BAPOMCPAgent:
    """One BA-POMCP learner over a FinitePOMDP's state, action and observation spaces (module header)."""

    def __init__(self, model: FinitePOMDP, prior: DirichletPrior, num_particles: int, num_simulations: int,
                 horizon: int, rollout: Rollout, seed: int, rejection_cap: int):
        """
        Args:
            model: Supplies S, A, Z, the known reward R(s, a), b0 and gamma; its T and O are NOT read.
            prior: Dirichlet counts of the dynamics (section 2).
            num_particles: K particles over (s, chi) (section 3).
            num_simulations: Simulations per decision.
            horizon: Planning horizon H (section 5); c = H (max R - min R).
            rollout: The rollout policy (section 4b).
            seed: Seed of the agent's generator (section 6).
            rejection_cap: Attempts per kept particle before the belief update raises (section 3).
        """
        num_a, num_s = model.reward.shape
        if prior.transition.shape != (num_a, num_s, num_s) or prior.observation.shape != (num_a, num_s,
                                                                                         model.num_observations):
            raise ValueError("the prior's shapes do not match the model's state, action and observation spaces.")
        if num_particles < 1 or num_simulations < 1 or horizon < 1 or rejection_cap < 1:
            raise ValueError("particles, simulations, horizon and rejection cap must be >= 1.")
        self.num_actions, self.num_states, self.num_observations = num_a, num_s, model.num_observations
        self.reward = model.reward.numpy().tolist()                      # R[a][s], known (section 2)
        self.initial_belief = model.initial_belief.numpy()
        self.discount = model.discount
        self.num_particles, self.num_simulations, self.horizon = num_particles, num_simulations, horizon
        self.exploration = horizon * float(model.reward.max() - model.reward.min())
        self.rollout, self.rejection_cap = Rollout(rollout), rejection_cap
        self._reward_table = model.reward.numpy()                       # (A, S), for the rollout MDP (section 4b)
        self._rollout_policy: list[int] | None = None
        self.rng = np.random.default_rng(seed)
        self.transition_counts = np.broadcast_to(prior.transition, (num_particles, *prior.transition.shape)).copy()
        self.observation_counts = np.broadcast_to(prior.observation, (num_particles, *prior.observation.shape)).copy()
        self.states = self.rng.choice(num_s, size=num_particles, p=self.initial_belief)
        self.root_values: list[float] | None = None                      # Q(root, a) of the last search

    # -- episodes and beliefs (section 3) --------------------------------------------------------------------------
    def redraw_states(self, belief: np.ndarray) -> None:
        """Redraws every particle's hidden state from `belief` (S,), keeping its counts: b0 at an episode start."""
        self.states = self.rng.choice(self.num_states, size=self.num_particles, p=belief)

    def begin_episode(self) -> None:
        """A new game: counts carry over, hidden states restart from b0 (section 3)."""
        self.redraw_states(self.initial_belief)

    def update(self, action: int, observation: int) -> None:
        """Rejection-sampling update of the particles after the real step (action, observation) (section 3)."""
        kept = 0
        states = np.empty(self.num_particles, dtype=np.int64)
        transition = np.empty_like(self.transition_counts)
        observation_counts = np.empty_like(self.observation_counts)
        attempts, limit, batch = 0, self.rejection_cap * self.num_particles, 4 * self.num_particles
        while kept < self.num_particles:
            if attempts >= limit:
                raise RuntimeError(f"rejection sampling kept {kept} of {self.num_particles} particles in {attempts} "
                                   f"attempts: the particles' counts make observation {observation} near impossible.")
            parents = self.rng.integers(self.num_particles, size=batch)
            source = self.states[parents]
            successor = _draw_rows(self.transition_counts[parents, action, source], self.rng.random(batch))
            emitted = _draw_rows(self.observation_counts[parents, action, successor], self.rng.random(batch))
            attempts += batch
            match = np.flatnonzero(emitted == observation)[:self.num_particles - kept]
            if match.size == 0:
                continue
            index = np.arange(kept, kept + match.size)
            states[index] = successor[match]
            transition[index] = self.transition_counts[parents[match]]
            observation_counts[index] = self.observation_counts[parents[match]]
            transition[index, action, source[match], successor[match]] += 1.0
            observation_counts[index, action, successor[match], observation] += 1.0
            kept += match.size
        self.states, self.transition_counts, self.observation_counts = states, transition, observation_counts

    def expected_dynamics(self) -> tuple[np.ndarray, np.ndarray]:
        """The particle average of the expected dynamics (paper, Eq. 1): T (A, S, S) and O (A, S, Z)."""
        transition = self.transition_counts / self.transition_counts.sum(-1, keepdims=True)
        observation = self.observation_counts / self.observation_counts.sum(-1, keepdims=True)
        return transition.mean(0), observation.mean(0)

    # -- search (section 4) ------------------------------------------------------------------------------------------
    def act(self, steps_left: int) -> int:
        """Plans from the current particles with num_simulations simulations and returns argmax_a Q(root, a)."""
        if steps_left < 1:
            raise ValueError(f"act() needs at least one step left, got {steps_left}.")
        depth_limit = min(self.horizon, steps_left)
        if self.rollout is Rollout.EXPECTED_MDP:
            self._rollout_policy = self.expected_mdp_policy()
        root = _Node(self.num_actions)
        for _ in range(self.num_simulations):
            i = int(self.rng.integers(self.num_particles))
            self._simulate(int(self.states[i]), self.transition_counts[i].copy(), self.observation_counts[i].copy(),
                           root, 0, depth_limit)
        self.root_values = list(root.values)
        best, best_value = 0, -math.inf
        for a in range(self.num_actions):
            if root.action_visits[a] > 0 and root.values[a] > best_value:
                best, best_value = a, root.values[a]
        return best

    def _select(self, node: _Node) -> int:
        for a in range(self.num_actions):
            if node.action_visits[a] == 0:
                return a
        log_visits = math.log(node.visits + 1)
        best, best_score = 0, -math.inf
        for a in range(self.num_actions):
            score = node.values[a] + self.exploration * math.sqrt(log_visits / node.action_visits[a])
            if score > best_score:
                best, best_score = a, score
        return best

    def _step(self, state: int, action: int, transition: np.ndarray, observation: np.ndarray) -> tuple[int, int]:
        """One BA-POMDP step on the given counts, which it increments (section 2)."""
        successor = _draw(transition[action, state], self.rng.random())
        emitted = _draw(observation[action, successor], self.rng.random())
        transition[action, state, successor] += 1.0
        observation[action, successor, emitted] += 1.0
        return successor, emitted

    def _simulate(self, state: int, transition: np.ndarray, observation: np.ndarray, node: _Node, depth: int,
                  depth_limit: int) -> float:
        if depth >= depth_limit:
            return 0.0
        action = self._select(node)
        reward = self.reward[action][state]
        successor, emitted = self._step(state, action, transition, observation)
        child = node.children.get((action, emitted))
        if child is None:
            node.children[(action, emitted)] = _Node(self.num_actions)
            future = self._rollout(successor, transition, observation, depth + 1, depth_limit)
        else:
            future = self._simulate(successor, transition, observation, child, depth + 1, depth_limit)
        total = reward + self.discount * future
        node.visits += 1
        node.action_visits[action] += 1
        node.values[action] += (total - node.values[action]) / node.action_visits[action]
        return total

    def expected_mdp_policy(self) -> list[int]:
        """
        The optimal policy of the MDP of the posterior-mean dynamics with the known reward (section 4b), by value
        iteration to MDP_TOLERANCE in the sup norm; the first maximising action per state.
        """
        transition, _ = self.expected_dynamics()                                       # (A, S, S)
        value = np.zeros(self.num_states)
        while True:
            q = self._reward_table + self.discount * transition @ value                # (A, S)
            nxt = q.max(axis=0)
            if np.abs(nxt - value).max() < MDP_TOLERANCE:
                return q.argmax(axis=0).tolist()
            value = nxt

    def _rollout(self, state: int, transition: np.ndarray, observation: np.ndarray, depth: int, depth_limit: int) -> float:
        total, weight = 0.0, 1.0
        policy = self._rollout_policy if self.rollout is Rollout.EXPECTED_MDP else None
        for _ in range(depth, depth_limit):
            action = int(self.rng.integers(self.num_actions)) if policy is None else policy[state]
            total += weight * self.reward[action][state]
            state, _ = self._step(state, action, transition, observation)
            weight *= self.discount
        return total


class BAPOMCPBatch:
    """
    B independent BA-POMCP learners behind the batched Agent protocol (training/rollout.py), one per row of a
    BatchedPOMDPEnv. Each row's learner keeps its own particles, counts and generator.
    """

    def __init__(self, agents: list[BAPOMCPAgent], episode_length: int):
        if not agents:
            raise ValueError("a batch needs at least one agent.")
        self.agents, self.batch_size, self.episode_length = agents, len(agents), episode_length
        self.step = 0

    def reset(self, initial_observation: Tensor | None) -> None:
        if initial_observation is not None:
            raise ValueError("a FinitePOMDP emits no initial observation.")
        for agent in self.agents:
            agent.begin_episode()
        self.step = 0

    def act(self) -> Tensor:
        steps_left = self.episode_length - self.step
        return torch.tensor([agent.act(steps_left) for agent in self.agents], dtype=torch.int64)

    def update(self, action: Tensor, observation: Tensor) -> None:
        for agent, a, o in zip(self.agents, action.tolist(), observation.tolist()):
            agent.update(a, o)
        self.step += 1
