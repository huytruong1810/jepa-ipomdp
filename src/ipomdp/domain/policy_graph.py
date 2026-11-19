# ABSOLUTE PATH: src/ipomdp/domain/policy_graph.py
# ==============================================================================
# POLICY GRAPHS: FINITE-STATE CONTROLLERS EXTRACTED FROM EXACT SOLUTIONS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Why Controllers:
#    - A finitely nested I-POMDP of agent i at level l models agent j by j's level-(l-1) solution. The fold of
#      domain/interactive.py needs that solution in a form that a POMDP over i's interactive states can carry: a
#      finite set of controller nodes, an action law per node and a successor law driven by j's own observations.
#      A policy graph is exactly that. With a finite node set, I-POMDP_{i,l} becomes a POMDP over
#      S x N_j x A_j, so the exact Bayes filter, the solver and the simulator of this package apply unchanged
#      (Panella & Gmytrasiewicz 2017 use the same reduction for finite-state models of other agents).
#
# 2. The Successor Law Has Four Indices:
#    - eta[n, a, o, n'] = P(next node n' | node n, action taken a, observation o). With a deterministic action law
#      the action index is redundant, but Doshi & Gmytrasiewicz (2009) let a modelled agent break ties uniformly
#      over its optimal actions; the node it moves to then depends on the action it actually took. Entries for
#      actions a node never takes are self-loops: they carry no probability and keep every row stochastic.
#
# 3. Two Extractions:
#    - plan_graph (finite horizon, exact): every alpha vector of Gamma_k is a k-step conditional plan, and Bellman
#      optimality fixes its continuation: at an interior witness belief b of alpha's region, each observation's
#      continuation is a maximiser of Gamma_{k-1} at tau(b, a, o). The continuation is chosen among the maximisers
#      so that R(a) + gamma sum_o M_{a,o} alpha_succ(o) reproduces alpha exactly (1e-9); if no combination does,
#      extraction raises. Nodes are (k, index) for k = H..1 plus one sink after the last step: the modelled agent's
#      horizon equals the modelling agent's (Gmytrasiewicz & Doshi 2005, Section 7.1), so the sink is never acted
#      from within the horizon.
#    - stationary_graph (infinite horizon, certified): one node per vector of the envelope of a value function
#      (re-pruned at a declared epsilon; at the exact epsilon epsilon-pruning cannot leave an asymmetric mirror
#      pair); successors are
#      the maximisers at tau(witness, a, o), ties split uniformly. This is exact only for finitely transient
#      policies, so it is never trusted on its own: evaluate() gives the controller's exact value, and an upper
#      bound of V* (domain/bounds.py) certifies its distance to the optimum at b0.
#
# 4. Witnesses and the Envelope:
#    - prune() keeps no witness beliefs (the seeded members come from sample beliefs, not LPs), so witnesses are
#      recomputed here against the final set with solver.witness.
#    - A pruned set can still hold a vector that owns no region of the final envelope: Lark's filter adds members to
#      W as it finds them and never re-examines an early member once later members cover its region. Such a vector
#      changes no value, so envelope() drops every vector whose witness margin against all the others is not
#      positive. Both extractions take envelopes; plan_graph raises on a vector without a region, so a caller that
#      forgot envelope() fails loudly instead of building a node no belief ever starts in.
#
# 5. Initial Nodes From a Prior Over Beliefs:
#    - The modelling agent's uncertainty about the other's beliefs becomes a law over the other's first node:
#      node_law_on_beliefs for a finite set of beliefs (ties split uniformly), node_law_on_segment for a uniform
#      density on a segment of beliefs (the "no information about j's belief" prior of Gmytrasiewicz & Doshi 2005,
#      Fig. 5 (ii)), computed exactly from the upper envelope along the segment.
# ==============================================================================

import itertools
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor

from .belief import belief_update, observation_distribution
from .pomdp import FinitePOMDP
from .solver import EXACT_PRUNE_EPSILON, AlphaVectorSet, prune, witness

_STOCHASTIC_ATOL = 1e-9
_RECONSTRUCTION_ATOL = 1e-9
_MAX_TIE_COMBINATIONS = 4096


@dataclass(frozen=True)
class PolicyGraph:
    """
    A finite-state controller (section 2).

    Attributes:
        action_law: psi[n, a], float64 (N, A).
        successor_law: eta[n, a, o, n'], float64 (N, A, O, N).
        initial_law: P(first node), float64 (N,).
        node_names: One name per node.
    """

    action_law: Tensor
    successor_law: Tensor
    initial_law: Tensor
    node_names: tuple[str, ...]

    def __post_init__(self) -> None:
        n, a = self.action_law.shape
        if tuple(self.successor_law.shape[:2]) != (n, a) or self.successor_law.shape[3] != n:
            raise ValueError(f"successor_law must be (N, A, O, N) with N={n}, A={a}; got {tuple(self.successor_law.shape)}.")
        if tuple(self.initial_law.shape) != (n,) or len(self.node_names) != n:
            raise ValueError("initial_law and node_names must have one entry per node.")
        for name in ("action_law", "successor_law", "initial_law"):
            t = getattr(self, name)
            sums = t.sum(-1)
            if t.dtype != torch.float64 or (t < 0).any() or not torch.allclose(sums, torch.ones_like(sums), atol=_STOCHASTIC_ATOL, rtol=0.0):
                raise ValueError(f"{name} must be float64, non-negative, rows summing to 1.")

    @property
    def num_nodes(self) -> int:
        return self.action_law.shape[0]

    @property
    def num_actions(self) -> int:
        return self.action_law.shape[1]

    @property
    def num_observations(self) -> int:
        return self.successor_law.shape[2]

    def with_initial_law(self, law: Tensor) -> "PolicyGraph":
        return replace(self, initial_law=law)


def constant_policy(action_probs: Tensor, num_observations: int, name: str) -> PolicyGraph:
    """A one-node controller acting by a fixed float64 action law: a test fixture or a subintentional model."""
    if action_probs.dtype != torch.float64:
        raise ValueError(f"the action law must be float64 (a float32 0.8 is 0.800000011920929), got {action_probs.dtype}.")
    probs = action_probs.reshape(1, -1)
    return PolicyGraph(probs, torch.ones(1, probs.shape[1], num_observations, 1, dtype=torch.float64),
                       torch.ones(1, dtype=torch.float64), (name,))


def envelope(value_function: AlphaVectorSet) -> AlphaVectorSet:
    """The vectors that own a region of max_k alpha_k . b, in their original order (section 4)."""
    vectors = value_function.vectors.numpy()
    if len(vectors) == 1:
        return value_function
    keep = [k for k in range(len(vectors)) if witness(vectors[k], np.delete(vectors, k, axis=0))[0] > 0.0]
    return AlphaVectorSet(value_function.vectors[keep].clone(), value_function.actions[keep].clone())


def _witnesses(vectors: np.ndarray) -> np.ndarray:
    """An interior belief of each vector's region of the upper envelope (section 4)."""
    if len(vectors) == 1:
        return np.full((1, vectors.shape[1]), 1.0 / vectors.shape[1])
    beliefs = []
    for k in range(len(vectors)):
        delta, belief = witness(vectors[k], np.delete(vectors, k, axis=0))
        if delta <= 0.0:
            raise ValueError(f"vector {k} owns no region of the envelope (margin {delta}); pass envelope(...) (section 4).")
        beliefs.append(belief)
    return np.stack(beliefs)


def _maximisers(vectors: Tensor, belief: Tensor) -> list[int]:
    scores = vectors @ belief
    best = float(scores.max())
    return torch.nonzero(scores >= best - _RECONSTRUCTION_ATOL * max(1.0, abs(best))).flatten().tolist()


def _uniform(indices: list[int], size: int) -> Tensor:
    law = torch.zeros(size, dtype=torch.float64)
    law[indices] = 1.0 / len(indices)
    return law


def _projections(model: FinitePOMDP, action: int) -> Tensor:
    """M_{a,o}[s, s'] = T(s'|s,a) O(o|s',a) for every o, shape (O, S, S)."""
    return torch.einsum("st,to->ost", model.transition[action], model.observation[action])


def plan_graph(model: FinitePOMDP, value_functions: list[AlphaVectorSet]) -> PolicyGraph:
    """
    The exact plan tree behind [Gamma_1, ..., Gamma_H] as a controller (section 3), with an initial law on the
    top-level plans that is uniform until a prior over beliefs replaces it (section 5). Every set must be an
    envelope (section 4): pass [envelope(g) for g in solve_finite_horizon(model, H)].

    Returns:
        Nodes (k, index) for k = H, ..., 1, then a sink; node names "h<k>.<index>" and "sink".
    """
    horizon = len(value_functions)
    offsets, names = {}, []
    for k in range(horizon, 0, -1):
        offsets[k] = len(names)
        names += [f"h{k}.{i}" for i in range(len(value_functions[k - 1].vectors))]
    sink = len(names)
    names.append("sink")
    n, a_size, o_size = len(names), model.num_actions, model.num_observations
    action_law = torch.zeros(n, a_size, dtype=torch.float64)
    successor = torch.zeros(n, a_size, o_size, n, dtype=torch.float64)
    successor[:, :, :, :] = torch.eye(n, dtype=torch.float64)[:, None, None, :]        # self-loops (section 2)
    action_law[sink, 0] = 1.0
    for k in range(horizon, 0, -1):
        gamma_k = value_functions[k - 1]
        beliefs = torch.from_numpy(_witnesses(gamma_k.vectors.numpy()))
        for i, (alpha, a) in enumerate(zip(gamma_k.vectors, gamma_k.actions.tolist())):
            node = offsets[k] + i
            action_law[node, a] = 1.0
            successor[node, a] = 0.0
            if k == 1:
                if not torch.allclose(alpha, model.reward[a], atol=_RECONSTRUCTION_ATOL, rtol=0.0):
                    raise ValueError(f"Gamma_1 vector {i} is not R(., {a}).")
                successor[node, a, :, sink] = 1.0
                continue
            below = value_functions[k - 2].vectors
            belief = beliefs[i:i + 1]
            probs = observation_distribution(model, belief, torch.tensor([a]))[0]
            choices = []
            for o in range(o_size):
                if probs[o] <= 0.0:
                    raise ValueError(f"observation {o} has zero probability at the witness of h{k}.{i}.")
                posterior = belief_update(model, belief, torch.tensor([a]), torch.tensor([o]))[0]
                choices.append(_maximisers(below, posterior))
            combinations = list(itertools.islice(itertools.product(*choices), _MAX_TIE_COMBINATIONS + 1))
            if len(combinations) > _MAX_TIE_COMBINATIONS:
                raise ValueError(f"too many tied continuations at h{k}.{i}.")
            projection = _projections(model, a)
            for combo in combinations:
                rebuilt = model.reward[a] + model.discount * sum(projection[o] @ below[c] for o, c in enumerate(combo))
                if torch.allclose(rebuilt, alpha, atol=_RECONSTRUCTION_ATOL * max(1.0, float(alpha.abs().max())), rtol=0.0):
                    for o, c in enumerate(combo):
                        successor[node, a, o, offsets[k - 1] + c] = 1.0
                    break
            else:
                raise ValueError(f"no continuation of maximisers reproduces plan h{k}.{i} (section 3).")
    top = len(value_functions[horizon - 1].vectors)
    initial = torch.zeros(n, dtype=torch.float64)
    initial[:top] = 1.0 / top
    return PolicyGraph(action_law, successor, initial, tuple(names))


def stationary_graph(model: FinitePOMDP, value_function: AlphaVectorSet, prune_epsilon: float) -> PolicyGraph:
    """
    A controller with one node per vector of the epsilon-pruned envelope of a value function (section 3). Not exact by
    itself: certify it with an upper bound of V* (domain/bounds.py). Initial law: the maximisers at the model's
    initial belief.

    prune_epsilon trades nodes for value: the pruned envelope is within prune_epsilon of the input everywhere
    (solver.py, section 4). A converged exact value function is extracted at EXACT_PRUNE_EPSILON; a point-based lower
    bound (thousands of nearly tangent vectors, each a node) is extracted at a fraction of its bracket width, which
    keeps the controller small enough to fold into the other agent's problem. Either way the controller's value is
    computed exactly by evaluate() and certified against the upper bound, so the choice never affects a correctness
    claim, only the controller's size and how close to optimal it is.
    """
    exact = envelope(AlphaVectorSet(*prune(value_function.vectors, value_function.actions, prune_epsilon)))
    vectors, actions = exact.vectors, exact.actions
    beliefs = torch.from_numpy(_witnesses(vectors.numpy()))
    n, a_size, o_size = len(vectors), model.num_actions, model.num_observations
    action_law = torch.zeros(n, a_size, dtype=torch.float64)
    successor = torch.zeros(n, a_size, o_size, n, dtype=torch.float64)
    successor[:, :, :, :] = torch.eye(n, dtype=torch.float64)[:, None, None, :]        # self-loops (section 2)
    for node, a in enumerate(actions.tolist()):
        action_law[node, a] = 1.0
        belief = beliefs[node:node + 1]
        probs = observation_distribution(model, belief, torch.tensor([a]))[0]
        for o in range(o_size):
            if probs[o] <= 0.0:
                raise ValueError(f"observation {o} has zero probability at the witness of node {node}.")
            posterior = belief_update(model, belief, torch.tensor([a]), torch.tensor([o]))[0]
            successor[node, a, o] = _uniform(_maximisers(vectors, posterior), n)
    initial = _uniform(_maximisers(vectors, model.initial_belief), n)
    return PolicyGraph(action_law, successor, initial, tuple(f"n{k}" for k in range(n)))


def evaluate(model: FinitePOMDP, graph: PolicyGraph) -> Tensor:
    """
    Exact value of a controller acting in `model`: V[n, s] = sum_a psi[n,a] (R[a,s] + gamma sum_{s',o} T O
    sum_{n'} eta V[n', s']), one linear solve over (n, s). Returns V of shape (N, S).
    """
    if (graph.num_actions, graph.num_observations) != (model.num_actions, model.num_observations):
        raise ValueError("the controller's action and observation sets must be the model's.")
    n, s = graph.num_nodes, model.num_states
    reward = graph.action_law @ model.reward                                                    # (N, S)
    # P[n, s, n', s'] = sum_a psi[n,a] sum_o T[a,s,s'] O[a,s',o] eta[n,a,o,n']
    step = torch.einsum("na,ast,ato,naom->nsmt", graph.action_law, model.transition, model.observation,
                        graph.successor_law).reshape(n * s, n * s)
    system = torch.eye(n * s, dtype=torch.float64) - model.discount * step
    return torch.linalg.solve(system, reward.reshape(n * s)).reshape(n, s)


def node_law_on_beliefs(vectors: Tensor, beliefs: Tensor, weights: Tensor, offset: int, size: int) -> Tensor:
    """P(first node) for a finite prior over beliefs: each belief starts at its maximiser, ties split (section 5)."""
    law = torch.zeros(size, dtype=torch.float64)
    for belief, weight in zip(beliefs, weights):
        law[offset:offset + len(vectors)] += float(weight) * _uniform(_maximisers(vectors, belief), len(vectors))
    return law


def node_law_on_segment(vectors: Tensor, start: Tensor, end: Tensor, offset: int, size: int) -> Tensor:
    """
    P(first node) for a uniform density on the belief segment b(p) = (1 - p) start + p end, p in [0, 1]: the length
    of the interval on which each vector is the maximiser (section 5). Exact up to float64 rounding.
    """
    lo, slope = (vectors @ start).numpy(), (vectors @ (end - start)).numpy()
    cuts = {0.0, 1.0}
    for i, j in itertools.combinations(range(len(lo)), 2):
        if slope[i] != slope[j]:
            p = (lo[j] - lo[i]) / (slope[i] - slope[j])
            if 0.0 < p < 1.0:
                cuts.add(float(p))
    cuts = sorted(cuts)
    law = torch.zeros(size, dtype=torch.float64)
    for left, right in zip(cuts[:-1], cuts[1:]):
        middle = 0.5 * (left + right)
        scores = lo + middle * slope
        tied = np.flatnonzero(scores >= scores.max() - _RECONSTRUCTION_ATOL * max(1.0, abs(scores.max())))
        law[offset + tied] += (right - left) / len(tied)
    return law


def segment_breakpoints(vectors: Tensor, start: Tensor, end: Tensor) -> list[tuple[float, int]]:
    """
    The upper envelope of max_k vectors[k] . b(p) along b(p) = (1 - p) start + p end: a list of (right end of the
    interval, maximiser) in increasing p, merging neighbouring intervals with the same maximiser.
    """
    lo, slope = (vectors @ start).numpy(), (vectors @ (end - start)).numpy()
    cuts = {0.0, 1.0}
    for i, j in itertools.combinations(range(len(lo)), 2):
        if slope[i] != slope[j]:
            p = (lo[j] - lo[i]) / (slope[i] - slope[j])
            if 0.0 < p < 1.0:
                cuts.add(float(p))
    cuts = sorted(cuts)
    pieces: list[tuple[float, int]] = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        best = int(np.argmax(lo + 0.5 * (left + right) * slope))
        if pieces and pieces[-1][1] == best:
            pieces[-1] = (right, best)
        else:
            pieces.append((right, best))
    return pieces
