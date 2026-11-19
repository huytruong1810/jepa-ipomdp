# ABSOLUTE PATH: src/ipomdp/domain/bounds.py
# ==============================================================================
# CERTIFIED BRACKETS OF V*: HEURISTIC SEARCH VALUE ITERATION (HSVI2) WITH AN EXECUTABLE LOWER BOUND
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Why a Bracket:
#    - Exact incremental pruning (solver.py) certifies V* of the canonical Tiger, but its linear programs grow with
#      the number of states and observations; the interactive problems (domain/interactive.py: tens of states, six
#      observations) are out of its reach at gamma = 0.95. HSVI (Smith & Simmons 2004, 2005) keeps a lower bound L
#      and an upper bound U of V* and tightens both along beliefs reachable from b0, until U(b0) - L(b0) <= epsilon.
#      Both bounds are proofs, so the benchmark is the certified interval [L(b0), U(b0)], never an estimate.
#
# 2. The Lower Bound Is a Controller:
#    - Every lower-bound vector is the exact value of a policy that the bracket can execute: the blind vectors
#      alpha_a = (I - gamma T_a)^{-1} R_a are the values of "always do a", and a point backup at b builds
#      alpha = R_a + gamma sum_o M_{a,o} beta_o from existing vectors beta_o, recording the index of beta_o as the
#      successor after observation o. By induction each vector is a lower bound of the value of its pointer plan,
#      so the pointers form a finite-state controller (policy_graph.PolicyGraph) whose exact value (evaluate())
#      dominates the lower bound at every node. That controller is the "optimal agent" of the interactive
#      benchmark: its played return estimates a value inside the certified bracket.
#    - Vectors are never pruned: a pruned vector could be the successor of a kept one. The sets stay small (one vector
#      per backup), and pointwise-dominated vectors cost only time.
#
# 3. The Upper Bound Is a Sawtooth:
#    - Corner values c(s) start at the fast informed bound (Hauskrecht 2000): the fixed point of
#      c_a(s) = R_a(s) + gamma sum_o max_a' sum_s' T(s'|s,a) O(o|s',a) c_a'(s'), a valid upper bound. A point
#      update stores (b, u = max_a Q_U(b, a)). The sawtooth interpolation
#         U(b) = min( c . b, min_k [ c . b + (u_k - c . b_k) min_{s in supp b_k} b(s) / b_k(s) ] )
#      is an upper bound of V* whenever every stored value is (Hauskrecht 2000, section 4.2), and every update keeps
#      it so (a Bellman backup of an upper bound is an upper bound).
#
# 4. Exploration (Smith & Simmons 2005, HSVI2):
#    - A trial descends from b0 while the gap exceeds epsilon * gamma^(-t): action by the upper bound's Q, observation
#      by the largest probability-weighted excess gap; on the way back both bounds are updated at every visited
#      belief. Each trial improves the bounds at b0 or at a belief whose excess gap is positive, which bounds the
#      number of trials (Smith & Simmons 2004, Theorem 1). A cap on trials raises instead of returning an
#      uncertified result.
# ==============================================================================

from dataclasses import dataclass

import numpy as np
import torch

from .policy_graph import PolicyGraph
from .pomdp import FinitePOMDP


@dataclass(frozen=True)
class Bracket:
    """
    A certified interval of V*(b0) and the controller behind its lower end.

    Attributes:
        lower: L(b0), the lower bound (a value the controller provably attains or exceeds).
        upper: U(b0), the upper bound of V*(b0).
        controller: The lower bound's pointer controller (section 2), initial node = best vector at b0.
        lower_vectors: float64 (K, S), the lower-bound vectors (node k's lower bound).
        trials: Exploration trials used.
    """

    lower: float
    upper: float
    controller: PolicyGraph
    lower_vectors: torch.Tensor
    trials: int


class _Rows:
    """A growing float64 row store with doubling capacity, so that adding a vector or point costs O(S), not O(K S)."""

    def __init__(self, width: int):
        self.data, self.size = np.zeros((64, width)), 0

    def append(self, row: np.ndarray) -> None:
        if self.size == len(self.data):
            self.data = np.vstack([self.data, np.zeros_like(self.data)])
        self.data[self.size] = row
        self.size += 1

    @property
    def rows(self) -> np.ndarray:
        return self.data[:self.size]


class _HSVI:
    def __init__(self, model: FinitePOMDP):
        self.T = model.transition.numpy()                    # (A, S, S')
        self.O = model.observation.numpy()                   # (A, S', O)
        self.R = model.reward.numpy()                        # (A, S)
        self.gamma = model.discount
        self.A, self.S = self.R.shape
        self.Z = self.O.shape[2]
        # M[a, o, s, s'] = T(s'|s,a) O(o|s',a)
        self.M = np.einsum("ast,ato->aost", self.T, self.O)
        # Lower bound (section 2): blind vectors with self-loop successors.
        self.vectors, self.actions, self.successors = _Rows(self.S), [], []
        for a in range(self.A):
            self.vectors.append(np.linalg.solve(np.eye(self.S) - self.gamma * self.T[a], self.R[a]))
            self.actions.append(a)
            self.successors.append(np.full(self.Z, self.vectors.size - 1))
        # Upper bound (section 3): fast informed bound corners.
        c = np.full((self.A, self.S), self.R.max() / (1 - self.gamma))
        for _ in range(100_000):
            nxt = self.R + self.gamma * np.einsum("aost,bt->aosb", self.M, c).max(axis=3).sum(axis=1)
            if np.abs(nxt - c).max() < 1e-12:
                break
            c = nxt
        else:
            raise RuntimeError("the fast informed bound did not converge.")
        self.corner = nxt.max(axis=0)
        # Sawtooth points: the belief, 1 / b_k(s) on the support (0 elsewhere), the support mask, and
        # u_k - c . b_k (section 3).
        self.points, self.inverse, self.excess = _Rows(self.S), _Rows(self.S), []

    # -- bounds ---------------------------------------------------------------------------------------------
    def lower(self, beliefs: np.ndarray) -> np.ndarray:
        """L at beliefs of shape (n, S)."""
        return (beliefs @ self.vectors.rows.T).max(axis=1)

    def upper(self, beliefs: np.ndarray) -> np.ndarray:
        """U at beliefs of shape (n, S): the sawtooth interpolation of section 3, all points at once."""
        base = beliefs @ self.corner
        if self.points.size == 0:
            return base
        inverse = self.inverse.rows                                                    # (P, S)
        ratio = np.where(inverse[None] > 0, beliefs[:, None, :] * inverse[None], np.inf).min(axis=2)   # (n, P)
        return np.minimum(base, base + (np.asarray(self.excess)[None] * ratio).min(axis=1))

    def children(self, belief: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """P(o | b, a) of shape (A, O) and the normalised posteriors of shape (A, O, S) (zero rows where P = 0)."""
        unnormalised = np.einsum("s,aost->aot", belief, self.M)
        probs = unnormalised.sum(axis=2)
        return probs, unnormalised / np.where(probs > 0, probs, 1.0)[..., None]

    def upper_q(self, belief: np.ndarray, probs: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        values = self.upper(posteriors.reshape(-1, self.S)).reshape(self.A, self.Z)
        return self.R @ belief + self.gamma * (probs * values).sum(axis=1)

    # -- updates --------------------------------------------------------------------------------------------
    def update(self, belief: np.ndarray) -> None:
        stacked = self.vectors.rows                                        # (K, S)
        unnormalised = np.einsum("s,aost->aot", belief, self.M)            # (A, O, S)
        choice = (unnormalised @ stacked.T).argmax(axis=2)                 # (A, O): beta_o per action
        alphas = self.R + self.gamma * np.einsum("aost,aot->as", self.M, stacked[choice])
        a = int((alphas @ belief).argmax())
        if alphas[a] @ belief > self.lower(belief[None])[0] + 1e-12:
            self.vectors.append(alphas[a]); self.actions.append(a); self.successors.append(choice[a])
        probs, posteriors = self.children(belief)
        value = float(self.upper_q(belief, probs, posteriors).max())
        if value < self.upper(belief[None])[0] - 1e-12:
            self.points.append(belief)
            self.inverse.append(np.where(belief > 0, 1.0 / np.where(belief > 0, belief, 1.0), 0.0))
            self.excess.append(value - float(belief @ self.corner))

    def explore(self, belief: np.ndarray, threshold: float, depth: int) -> None:
        if self.upper(belief[None])[0] - self.lower(belief[None])[0] <= threshold or depth > self.max_depth:
            return
        probs, posteriors = self.children(belief)
        a = int(self.upper_q(belief, probs, posteriors).argmax())
        live = probs[a] > 0
        gaps = self.upper(posteriors[a]) - self.lower(posteriors[a])
        excess = np.where(live, probs[a] * (gaps - threshold / self.gamma), -np.inf)
        o = int(excess.argmax())
        if excess[o] > 0:
            self.explore(posteriors[a, o], threshold / self.gamma, depth + 1)
        self.update(belief)


def hsvi(model: FinitePOMDP, epsilon: float, max_trials: int = 100_000) -> Bracket:
    """
    Certified bracket of V*(b0) with U(b0) - L(b0) <= epsilon (sections 1-4).

    Args:
        model: Exact POMDP specification.
        epsilon: Required width of the bracket at b0.
        max_trials: Exceeding it raises (an uncertified result is never returned).
    """
    search = _HSVI(model)
    span = (search.R.max() - search.R.min()) / (1 - search.gamma)
    search.max_depth = int(np.ceil(np.log(epsilon / span) / np.log(search.gamma))) + 1
    b0 = model.initial_belief.numpy()
    gap = lambda: float(search.upper(b0[None])[0] - search.lower(b0[None])[0])   # noqa: E731
    for trial in range(1, max_trials + 1):
        search.explore(b0, epsilon, 0)
        if gap() <= epsilon:
            break
    else:
        raise RuntimeError(f"HSVI did not reach a bracket of {epsilon} within {max_trials} trials.")
    vectors = search.vectors.rows.copy()
    k, z = len(vectors), search.Z
    action_law = torch.zeros(k, search.A, dtype=torch.float64)
    action_law[torch.arange(k), torch.tensor(search.actions)] = 1.0
    successor = torch.zeros(k, search.A, z, k, dtype=torch.float64)
    successor[:, :, :, :] = torch.eye(k, dtype=torch.float64)[:, None, None, :]   # untaken actions: self-loops
    for node, (a, succ) in enumerate(zip(search.actions, search.successors)):
        successor[node, a] = 0.0
        successor[node, a, torch.arange(z), torch.from_numpy(succ)] = 1.0
    initial = torch.zeros(k, dtype=torch.float64)
    initial[int((vectors @ b0).argmax())] = 1.0
    controller = PolicyGraph(action_law, successor, initial, tuple(f"l{i}" for i in range(k)))
    return Bracket(float(search.lower(b0[None])[0]), float(search.upper(b0[None])[0]), controller,
                   torch.from_numpy(vectors), trial)
