# ABSOLUTE PATH: src/ipomdp/planning/mcts.py
# ==============================================================================
# BELIEF-TREE MONTE CARLO TREE SEARCH WITH EXACT OBSERVATION BRANCHING
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Tree Structure (a belief tree, as in exact POMDP planning):
#      decision node  s          (a belief b or a learned latent z; see search_model.py)
#        -> edge (s, a)          reward R(s, a), observation probabilities P(o | s, a)
#          -> child  tau(s,a,o)  one decision node per observation o (exact branching)
#    - Expanding a node calls SearchModel.expand once and creates ALL |A| x |O| children, each
#      with a leaf estimate V(child). Every action therefore has a one-step lookahead value
#      from the moment its parent is expanded.
#
# 2. Values: Expectimax Backups on the Expanded Tree:
#      V^(s)    = max_a Q(s, a)            if s is expanded, else its leaf estimate V(s)
#      Q(s, a)  = R(s, a) + gamma * sum_o P(o | s, a) * V^(tau(s, a, o))
#    - Decision nodes take the MAX over actions and chance edges the EXACT EXPECTATION over
#      observations, so the expanded tree computes precisely the finite belief-tree values the
#      exact solver computes, with the leaf estimates at its frontier. With exact leaves the
#      search can only improve on them; it never mixes in the values of exploratory actions.
#    - Phase-4 finding: the earlier mean backup (V^ = average of all returns backed up through
#      a node, as in MuZero) made estimates WORSE with more search. At b = (0.97, 0.03) with
#      exact V_8 leaves, root Q(OPEN_RIGHT) fell from the exact 12.80 (1 simulation) to -2.98
#      (1000 simulations) because returns of exploratory door openings (Q = -90) were averaged
#      in; the search then preferred LISTEN over the optimal OPEN_RIGHT. MuZero tolerates mean
#      backups only because a learned policy prior keeps exploration narrow; here the prior is
#      uniform and the branch probabilities are exact, so expectimax is the principled choice.
#    - Only nodes on the simulated path change, so both Q(s, a) per edge and V^ per node are
#      cached and recomputed bottom-up along the path after each expansion. Selection then costs
#      O(|A|) per node instead of recomputing every Q from its children (profiling: ~18k Q
#      evaluations per 16-root, 50-simulation search before caching).
#
# 3. Selection (PUCT, AlphaZero/MuZero form):
#      a* = argmax_a  Q_norm(s, a) + c_puct * P(a) * sqrt(N(s) + 1) / (1 + N(s, a))
#    - P(a) is uniform (no policy network); at the root it is mixed with Dirichlet(alpha)
#      noise with weight epsilon when exploration is requested (training), and left uniform
#      for evaluation (dirichlet_epsilon = 0).
#    - Q_norm is Q min-max normalised over all Q values computed in the current tree, so
#      c_puct is independent of the reward scale (-100 .. +10 on Tiger).
#    - Descent below the chosen edge samples o ~ P(o | s, a), so simulations concentrate on
#      likely observation branches.
#    - Bugs of the previous search fixed here: unvisited actions were scored by V(child)
#      WITHOUT the immediate reward, so opening a door (r = -45 in expectation at b0) looked
#      as good as listening; Python's `random`/NumPy global generators were unseeded; a
#      silent random-action fallback existed; Q telemetry ignored rewards.
#
# 4. Reproducibility:
#    - All randomness (Dirichlet noise, observation sampling) comes from one seeded
#      numpy Generator owned by the planner.
#
# 5. Output:
#    - temperature = 0 (evaluation): a one-hot on argmax_a Q(root, a), the Bellman-optimal
#      action of the searched tree.
#    - temperature > 0 (training collection): visit counts shaped by counts^(1/T), which
#      follow PUCT's exploration and give a smoothed, exploratory policy.
#    - Root visit counts and Q values are recorded for telemetry and tests.
#
# 6. Pruning Negligible Observations (Wumpus World, design decision 2026-10-01):
#    - Wumpus World has 33 observations, so exact branching would create 5 x 33 = 165 Python
#      child nodes per expansion although, in any given belief, most percept combinations are
#      impossible. With observation_prune = eps > 0 an edge keeps only the observations with
#      P(o | s, a) >= eps and renormalises their probabilities, so its Q is the exact
#      expectimax over the outcomes the model considers possible. The neglected mass is at
#      most |O| * eps per edge (0.033 at eps = 1e-3); the largest neglected mass of a search is
#      reported in SearchStatistics.max_pruned_mass.
#    - observation_prune = 0 keeps every observation, zero-probability ones included, and does
#      NOT renormalise: canonical Tiger's search is bit-identical to the unpruned search.
#
# 7. Cost of the Python Tree (2026-11-15; results bit-identical to the earlier implementation):
#    - A Wumpus recipe seed spends about three quarters of its wall time in this module's Python,
#      not on the GPU (profile_recipe.py). Four costs were removed without changing one number
#      (checked by the GPU Wumpus fingerprint: every metric, parameter sum and the full-state hash):
#        a. Children are created LAZILY. An expansion creates about 15 kept children per node, of
#           which the search later visits one or two. An edge now holds the children's leaf values
#           as plain floats and builds a DecisionNode only when the descent selects that child (or
#           when `Edge.children` is read from outside, which materialises all of them).
#        b. A node's state is a HANDLE (the expansion's next_states block and an index), turned into
#           a tensor only when the node is expanded. The earlier code created a tensor view per
#           child (about a thousand GPU-tensor views per simulation step that were never read).
#        c. Selection works on Python floats: sqrt(N + 1) once per node, c_puct * P(a) once per
#           search, a first-maximum loop for the argmax, and the cumulative branch probabilities
#           cached per edge (bisect on a list). Each expression keeps its order of operations.
#        d. Edge.refresh accumulates p * V^ in an explicit left-to-right loop.
#    - What must NOT be "simplified" (each would change rounding):
#        * Python 3.12's built-in sum() uses compensated summation for exact floats, while the
#          earlier generator of np.float64 products was added plainly, left to right. The explicit
#          `acc += p * v` loop is the same plain addition; sum() over Python floats is not.
#        * The pruned mass and the renormalisation stay per-edge numpy calls on the compacted
#          array: numpy's reduction is not a left-to-right sum, and its order depends on the length.
#        * math.sqrt and np.sqrt agree (both correctly rounded), and a product of a Python float
#          and an np.float64 is the same IEEE double product.
#    - The model's contract is unchanged (one batched expand per simulation step; states are rows
#      of next_states), so every SearchModel and wrapper works as before. Edge.refresh reads a
#      materialised child's CURRENT value, so code that edits a child's leaf_value and refreshes
#      the edge (reports/.../self_loop_diag.py) behaves as before.
#
# 8. What a Larger Budget Guarantees, and What It Does Not (code audit 2026-11-19;
#    reports/2026-11-16_code_audit/planner_budget.py and its log; tests/test_planner_and_agent.py):
#    - Guaranteed, for the model the search is given: if every leaf estimate is within eps of that model's optimal
#      value, every root edge satisfies |Q_tree(s, a) - Q*(s, a)| <= gamma^{d_a} eps, where d_a = 1 + the shallowest
#      positive-probability frontier depth below the edge (a leaf errs by at most eps, a chance edge averages its
#      children's errors, a decision node's max moves by at most its largest edge error). With a fixed seed the
#      tree after N simulations is a prefix of the tree after N' > N, so d_a never decreases and the bound is
#      non-increasing in N; PUCT reaches every branch eventually, so Q_tree converges to Q* (consistency).
#    - Not guaranteed: the decision at a finite budget. PUCT deepens the promising action first, so actions are
#      compared on subtrees of different depths, and with biased leaves the comparison can flip as N grows (the
#      horizon effect of best-first search). On the exact canonical Tiger with zero leaves, the greedy agent plays
#      the optimal actions at N = 1, then loses 5.4 per episode at N = 2 and 5.9 at N = 4 (paired over 256
#      episodes, 95% half-widths about 3) and recovers to -1.1 at N = 32; with V_8 leaves the root decision is
#      optimal at every budget tested. The bound tightens slowly: after 4096 simulations d_a is 2 or 3 at every root.
#    - Consequence: at practical budgets the leaf value and the model set the decision quality, not the number of
#      simulations, and more search converges to the optimal policy of the MODEL it is given. With a learned model
#      that policy exploits the model's errors (the optimizer's curse), so a learned planner need not improve with
#      its budget (PROTOCOL.md: C-g on Wumpus; C-q.3 measures the learned Tiger planner).
# ==============================================================================

from bisect import bisect_right
from dataclasses import dataclass, field
import math

import numpy as np
import torch
from torch import Tensor

from .search_model import SearchModel


class MinMaxStats:
    """Running minimum/maximum of Q values in one tree, for PUCT normalisation."""

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value: float) -> None:
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        """Maps value into [0, 1]; 0.5 until two distinct values have been seen."""
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return 0.5


class DecisionNode:
    """A belief (or latent) node; `edges` is None until the node is expanded.

    The state is either a tensor (roots, nodes built by callers) or a handle into the expansion that created
    the node (section 7b): `source[index]`, read through the `state` property only when it is needed.
    """

    __slots__ = ("_state", "_source", "_index", "leaf_value", "visits", "edges", "cached_value", "__dict__")

    def __init__(self, state: Tensor | None, leaf_value: float, source: Tensor | None = None,
                 index: tuple[int, ...] | None = None):
        self._state, self._source, self._index = state, source, index
        self.leaf_value = leaf_value
        self.visits = 0
        self.edges: list["Edge"] | None = None
        self.cached_value: float | None = None

    @property
    def state(self) -> Tensor:
        """The node's state tensor (materialised from the expansion block on first use)."""
        if self._state is None:
            self._state = self._source[self._index]
        return self._state

    def value(self) -> float:
        """V^: max_a Q(s, a) once expanded (cached, see refresh), else the leaf estimate."""
        return self.leaf_value if self.cached_value is None else self.cached_value

    def refresh(self) -> None:
        """Recomputes the cached max_a Q(s, a) from the edges' cached Q values."""
        self.cached_value = max(edge.q for edge in self.edges)


class Edge:
    """Action edge (s, a): expected reward, observation distribution, one child per kept o (sections 6 and 7)."""

    __slots__ = ("reward", "observation_probs", "observations", "visits", "q", "_probs", "_leaf_values", "_children",
                 "_source", "_row", "_cumulative")

    def __init__(self, reward: float, observation_probs: np.ndarray, observations: np.ndarray, leaf_values: list[float],
                 source: Tensor, row: tuple[int, int]):
        self.reward = reward
        self.observation_probs = observation_probs          # kept (renormalised) probabilities, np.float64
        self.observations = observations                    # the kept observation indices
        self.visits = 0
        self.q = 0.0
        self._probs: list[float] = observation_probs.tolist()
        self._leaf_values = leaf_values
        self._children: list[DecisionNode | None] = [None] * len(leaf_values)
        self._source, self._row = source, row
        self._cumulative: list[float] | None = None

    def child(self, k: int) -> DecisionNode:
        """The k-th kept child, created on first use (section 7a)."""
        node = self._children[k]
        if node is None:
            node = self._children[k] = DecisionNode(None, self._leaf_values[k], self._source,
                                                    (*self._row, int(self.observations[k])))
        return node

    @property
    def children(self) -> list[DecisionNode]:
        """Every kept child as a node (materialises the ones the search never visited)."""
        return [self.child(k) for k in range(len(self._children))]

    def cumulative(self) -> list[float]:
        """Cumulative kept probabilities for sampling a branch (np.cumsum once per edge, then a list)."""
        if self._cumulative is None:
            self._cumulative = np.cumsum(self.observation_probs).tolist()
        return self._cumulative

    def refresh(self, discount: float) -> float:
        """Recomputes and caches Q = R(s, a) + gamma * sum_o P(o | s, a) V^(child_o) (left-to-right, section 7)."""
        acc = 0.0
        for p, leaf, node in zip(self._probs, self._leaf_values, self._children):
            acc += p * (leaf if node is None else node.value())
        self.q = self.reward + discount * acc
        return self.q


@dataclass
class SearchStatistics:
    """Root-level results of the last search (one row per root)."""

    visit_counts: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    q_values: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    mean_depth: float = 0.0
    max_depth: int = 0
    max_pruned_mass: float = 0.0


class BeliefTreeSearch:
    """Batched PUCT search over belief trees with exact observation branching."""

    def __init__(
        self,
        model: SearchModel,
        num_simulations: int,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        seed: int,
        observation_prune: float = 0.0,
    ):
        """
        Args:
            model: SearchModel providing expansions (exact or learned).
            num_simulations: Simulations per root per search (>= 1).
            c_puct: Exploration constant.
            dirichlet_alpha: Concentration of the root prior noise.
            dirichlet_epsilon: Weight of the root prior noise (0 disables it).
            seed: Seed of the planner's random generator.
            observation_prune: Minimum P(o | s, a) of a kept observation branch (section 6); 0 keeps all.
        """
        if num_simulations < 1:
            raise ValueError(f"num_simulations must be >= 1, got {num_simulations}.")
        if not 0.0 <= observation_prune < 1.0 / model.num_observations:
            raise ValueError(f"observation_prune must lie in [0, 1/|O|), got {observation_prune}.")
        self.observation_prune = observation_prune
        self._max_pruned_mass = 0.0
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.rng = np.random.default_rng(seed)
        self.roots: list[DecisionNode] = []
        self.statistics = SearchStatistics()

    def state_dict(self) -> dict:
        """Random-generator state (checkpointing); trees are rebuilt on every search."""
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        """Restores a state produced by state_dict()."""
        self.rng.bit_generator.state = state["rng"]

    def _expand(self, nodes: list[DecisionNode], stats: list[MinMaxStats]) -> None:
        """Expands all `nodes` with one batched model call and records their Q values."""
        expansion = self.model.expand(torch.stack([node.state for node in nodes]))
        rewards = expansion.rewards.double().cpu().numpy()
        probs = expansion.observation_probs.double().cpu().numpy()
        values = expansion.next_values.double().cpu().numpy()
        next_states = expansion.next_states
        all_observations = np.arange(self.model.num_observations)
        for i, (node, node_stats) in enumerate(zip(nodes, stats)):
            node.edges = []
            for a in range(self.model.num_actions):
                kept, kept_probs = all_observations, probs[i, a]
                if self.observation_prune > 0.0:
                    # Always non-empty: the largest of |O| probabilities is >= 1/|O| > observation_prune.
                    kept = np.flatnonzero(kept_probs >= self.observation_prune)
                    mass = kept_probs[kept].sum()
                    self._max_pruned_mass = max(self._max_pruned_mass, 1.0 - mass / kept_probs.sum())
                    kept_probs = kept_probs[kept] / mass
                node.edges.append(Edge(float(rewards[i, a]), kept_probs, kept, values[i, a][kept].tolist(),
                                       next_states, (i, a)))
            for edge in node.edges:
                node_stats.update(edge.refresh(self.model.discount))
            node.refresh()

    def _select(self, node: DecisionNode, c_prior: list[float], stats: MinMaxStats) -> int:
        """PUCT action selection (module header, section 3); `c_prior` holds c_puct * P(a) (section 7c)."""
        edges = node.edges
        total_visits = 0
        for edge in edges:
            total_visits += edge.visits
        root = math.sqrt(total_visits + 1)
        best, best_score = 0, -math.inf
        for a, edge in enumerate(edges):
            score = stats.normalize(edge.q) + c_prior[a] * root / (1 + edge.visits)
            if score > best_score:                        # strict: the first maximum, as np.argmax
                best, best_score = a, score
        return best

    @torch.no_grad()
    def search(self, root_states: Tensor, temperature: float) -> Tensor:
        """
        Runs num_simulations simulations from every root.

        Args:
            root_states: States of shape (B, ...), as produced by the SearchModel.
            temperature: 0 for a one-hot on argmax_a Q(root, a), > 0 for visit counts^(1/T).

        Returns:
            Action distributions of shape (B, |A|), float32 on the states' device.
        """
        num_roots, num_a = root_states.shape[0], self.model.num_actions
        self._max_pruned_mass = 0.0
        roots = [DecisionNode(root_states[i], leaf_value=0.0) for i in range(num_roots)]
        stats = [MinMaxStats() for _ in range(num_roots)]
        self._expand(roots, stats)
        uniform = np.full(num_a, 1.0 / num_a)
        priors = [
            (1.0 - self.dirichlet_epsilon) * uniform
            + self.dirichlet_epsilon * self.rng.dirichlet([self.dirichlet_alpha] * num_a)
            if self.dirichlet_epsilon > 0 else uniform
            for _ in range(num_roots)
        ]

        # c_puct * P(a) once per search, as Python floats (section 7c): the same product the score computed first.
        c_uniform = [self.c_puct * p for p in uniform.tolist()]
        c_priors = [[self.c_puct * p for p in prior.tolist()] for prior in priors]
        random = self.rng.random
        depths = []
        for _ in range(self.num_simulations):
            paths, leaves = [], []
            for root, c_prior, root_stats in zip(roots, c_priors, stats):
                node, path = root, []
                while node.edges is not None:
                    edge = node.edges[self._select(node, c_prior if node is root else c_uniform, root_stats)]
                    cumulative = edge.cumulative()
                    observation = min(bisect_right(cumulative, random() * cumulative[-1]), len(cumulative) - 1)
                    path.append((node, edge))
                    node = edge.child(observation)
                paths.append(path)
                leaves.append(node)
                depths.append(len(path))
            self._expand(leaves, stats)
            for leaf, path, root_stats in zip(leaves, paths, stats):
                leaf.visits += 1
                for node, edge in reversed(path):
                    root_stats.update(edge.refresh(self.model.discount))
                    edge.visits += 1
                    node.visits += 1
                    node.refresh()

        self.roots = roots
        counts = np.array([[edge.visits for edge in root.edges] for root in roots], dtype=np.float64)
        self.statistics = SearchStatistics(
            visit_counts=counts,
            q_values=np.array([[edge.q for edge in root.edges] for root in roots]),
            mean_depth=float(np.mean(depths)),
            max_depth=int(np.max(depths)),
            max_pruned_mass=self._max_pruned_mass,
        )
        if temperature == 0.0:
            policy = np.zeros_like(counts)
            policy[np.arange(num_roots), self.statistics.q_values.argmax(axis=1)] = 1.0
        else:
            shaped = (counts / counts.max(axis=1, keepdims=True)) ** (1.0 / temperature)
            policy = shaped / shaped.sum(axis=1, keepdims=True)
        return torch.as_tensor(policy, dtype=torch.float32, device=root_states.device)
