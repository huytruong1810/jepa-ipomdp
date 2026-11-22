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
#
# 5. An Upper Bound by Information Relaxation over a Finite Belief Graph (revealed_upper_bound):
#    - Why: on folded interactive problems the sawtooth stalls. On the level-1 two-agent tiger fold (26 states),
#      U(b0) is still 44.8 after about 1,000 trials, while L(b0) = 16.54. Two causes: the fold's beliefs spread over
#      the other agent's nodes, so a stored point rarely fits inside a new belief's support; and the corners keep their
#      fast-informed values (about 77) because no reachable belief comes near a corner. LP (convex-hull)
#      interpolation over the same points moved U(b0) only from 55.3 to 53.4 in one sweep of backups.
#    - The relaxation: on every transition c -> c' the agent receives an extra symbol y = reveal(a, history). Any
#      policy of the original problem is a policy of the relaxed one (it can ignore y), so V*_relaxed >= V* at every
#      belief. With a well-chosen y, the beliefs reachable from b0 in the relaxed problem form a finite set, because
#      every reset lands on one of finitely many beliefs. Value iteration on that finite graph then gives
#      V*_relaxed(b0), with no interpolation (domain/interactive.py, section 5, chooses y for the tiger game).
#    - Standard form: y may read the departure state and earlier states, while FinitePOMDP emits observations from
#      the arrival state. So the state is augmented with the last `memory` states (a history). This leaves the
#      problem unchanged, because the past never enters T, O or R, and makes y a function of the arrival history.
#    - Certification:
#      * Beliefs are merged when they agree after rounding to `rounding`. Let delta be the largest L1 distance of a
#        merge and span = (max R - min R) / (1 - gamma). Merging costs at most (span / 2) delta / (1 - gamma), by the
#        span-Holder lemma (every alpha vector's components lie in [min R, max R] / (1 - gamma)), and that slack is
#        added.
#      * Value iteration starts at max R / (1 - gamma), which is >= its own Bellman image, so every iterate lies above
#        the merged graph's fixed point.
#      * The result is a proof whatever the enumeration's size; a cap on beliefs raises instead of truncating.
#    - Each enumerated belief's marginal over the original states, with its relaxed value, is an upper-bound point of
#      the original problem: the past components of the history never enter the original problem's future. The
#      values at the roots (b0 and any extra beliefs) are returned for that use.
#
# 6. Seeding HSVI with the Relaxation (hsvi_budget):
#    - The relaxed values at the corners and at the enumerated beliefs are valid upper-bound points of the original
#      problem, so they replace the fast-informed corners and seed the sawtooth. HSVI's own backups on the original
#      problem then tighten the bound below the relaxation (the relaxation ignores the cost of the noise it removes).
#    - Corners are backed up after every trial (_SeededHSVI). hsvi() itself is unchanged: the controllers that
#      define the interactive problems (interactive.py, section 2) are extracted from it, so its arithmetic must stay
#      identical.
#
# 7. Checkpoints of a Budgeted Search (hsvi_vector_bracket):
#    - Why: a trial's cost grows with the stored vectors and points, so the 1,000 trials behind the level-1 benchmark
#      take most of a day. Without a checkpoint an interruption loses all of them, and the run says nothing until
#      its last line.
#    - What is stored: the whole mutable state of the search (the lower-bound vectors with their actions and
#      successors; the sawtooth points with their inverses, values and excesses; the corners), the number of
#      finished trials, the epsilon that set the depth and gap thresholds, and a digest of the model's tables.
#    - A resumed run is the uninterrupted run: trials draw no random numbers and every array is restored bit for bit,
#      so both give identical vectors (tested). A run that had reached its width is not given another trial.
#    - A checkpoint of another model or another epsilon raises, and so does a budget below the stored trials: none
#      of them is the search the caller asked for.
#    - The file is replaced atomically (written beside itself, then renamed), so an interruption during a write
#      leaves the previous checkpoint.
# ==============================================================================

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
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
    return _bracket(search, b0, trial)


def _bracket(search: "_HSVI", b0: np.ndarray, trials: int) -> Bracket:
    """The bracket at b0 and the lower bound's pointer controller (section 2)."""
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
                   torch.from_numpy(vectors), trials)


class _SeededHSVI(_HSVI):
    """
    HSVI whose upper bound starts from given corner values and points (section 6), and whose corners are backed up
    after every trial. A corner backup max_a Q_U(e_s, a) is an upper bound of V*(e_s), and the sawtooth stays valid
    for any corner values that are upper bounds, so the excess of every stored point is recomputed from its value.
    """

    def __init__(self, model: FinitePOMDP, corners: np.ndarray | None, beliefs: np.ndarray | None,
                 values: np.ndarray | None):
        super().__init__(model)
        self.values: list[float] = []
        if corners is not None:
            self.corner = np.minimum(self.corner, corners)
        if beliefs is not None:
            for belief, value in zip(beliefs, values):
                if value < self.upper(belief[None])[0] - 1e-12:
                    self._store(belief, float(value))

    def _store(self, belief: np.ndarray, value: float) -> None:
        self.points.append(belief)
        self.inverse.append(np.where(belief > 0, 1.0 / np.where(belief > 0, belief, 1.0), 0.0))
        self.values.append(value)
        self.excess.append(value - float(belief @ self.corner))

    def update(self, belief: np.ndarray) -> None:
        size = self.points.size
        super().update(belief)
        if self.points.size > size:
            self.values.append(self.excess[-1] + float(belief @ self.corner))

    def back_up_corners(self) -> None:
        # Every backup reads the same, consistent bound; the new corners are applied together and the excesses
        # recomputed at once. Applying one corner while the stored excesses still refer to the old corners would mix
        # the two, and since corners only decrease, that mix can fall below the sawtooth (an invalid bound).
        backed_up = np.empty(self.S)
        for s in range(self.S):
            corner = np.zeros(self.S)
            corner[s] = 1.0
            backed_up[s] = float(self.upper_q(corner, *self.children(corner)).max())
        if (backed_up < self.corner - 1e-12).any():
            self.corner = np.minimum(self.corner, backed_up)
            if self.points.size:
                self.excess = list(np.asarray(self.values) - self.points.rows @ self.corner)


def hsvi_budget(model: FinitePOMDP, epsilon: float, trials: int, corners: torch.Tensor | None = None,
                points: tuple[torch.Tensor, torch.Tensor] | None = None, back_up_corners: bool = True) -> Bracket:
    """
    The certified bracket after at most `trials` HSVI trials toward width `epsilon` (sections 4 and 6). Unlike
    hsvi(), the width is whatever the budget reached: both ends are proofs either way, and the caller reports the
    width. It stops early once the width is at most epsilon.

    Args:
        model: Exact POMDP specification.
        epsilon: Target width; it also sets the trials' depth and gap thresholds, as in hsvi().
        trials: The trial budget.
        corners: Optional float64 (S,) upper bounds of V* at the corners (e.g. RelaxedBound.root_values).
        points: Optional (beliefs (K, S), values (K,)) upper bounds of V* (e.g. RelaxedBound.beliefs, .values).
        back_up_corners: Back up the corners after every trial (section 6). The cost of a sweep grows with the stored
            points (on the level-1 fold, 1,000 trials did not finish in 11 hours with it). Without it and without
            seeds, the trials are hsvi()'s, so a run that needs only the lower end turns it off.
    """
    search, trial = _run_budget(model, epsilon, trials, corners, points, back_up_corners)
    return _bracket(search, model.initial_belief.numpy(), trial)


def _model_digest(model: FinitePOMDP) -> str:
    """SHA-256 of the model's tables and discount, which ties a checkpoint to the problem it searched (section 7)."""
    digest = hashlib.sha256()
    for table in (model.transition, model.observation, model.reward, model.initial_belief):
        digest.update(str(tuple(table.shape)).encode())
        digest.update(np.ascontiguousarray(table.numpy()).tobytes())
    digest.update(repr(float(model.discount)).encode())
    return digest.hexdigest()


def _save_search(search: "_SeededHSVI", path: Path, digest: str, epsilon: float, trial: int) -> None:
    """Write the search's whole mutable state after `trial` trials (section 7), replacing the file atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez(handle, digest=np.array(digest), epsilon=np.float64(epsilon), trial=np.int64(trial),
                 vectors=search.vectors.rows, actions=np.asarray(search.actions, dtype=np.int64),
                 successors=np.stack(search.successors), points=search.points.rows, inverse=search.inverse.rows,
                 excess=np.asarray(search.excess, dtype=np.float64),
                 values=np.asarray(search.values, dtype=np.float64), corner=search.corner)
    temporary.replace(path)


def _restore_search(search: "_SeededHSVI", path: Path, digest: str, epsilon: float) -> int:
    """
    Load a checkpoint into a freshly built search of the same model and return its finished trials (section 7).
    Lists return to Python numbers and rows to fresh row stores, so every later operation reads the values the
    uninterrupted search would read.
    """
    def rows(array: np.ndarray) -> _Rows:
        store = _Rows(search.S)
        for row in array:
            store.append(row)
        return store

    with np.load(path) as stored:
        if str(stored["digest"]) != digest or float(stored["epsilon"]) != epsilon:
            raise ValueError(f"{path} is the checkpoint of another model or another epsilon (section 7).")
        search.vectors, search.points, search.inverse = rows(stored["vectors"]), rows(stored["points"]), rows(
            stored["inverse"])
        search.actions, search.successors = stored["actions"].tolist(), list(stored["successors"])
        search.excess, search.values = stored["excess"].tolist(), stored["values"].tolist()
        search.corner = stored["corner"]
        return int(stored["trial"])


def _run_budget(model: FinitePOMDP, epsilon: float, trials: int, corners, points, back_up_corners: bool,
                checkpoint: Path | None = None, every: int = 10,
                report: Callable[[int, float, float, int], None] | None = None):
    search = _SeededHSVI(model, None if corners is None else corners.numpy(),
                         None if points is None else points[0].numpy(), None if points is None else points[1].numpy())
    span = (search.R.max() - search.R.min()) / (1 - search.gamma)
    search.max_depth = int(np.ceil(np.log(epsilon / span) / np.log(search.gamma))) + 1
    b0 = model.initial_belief.numpy()
    gap = lambda: float(search.upper(b0[None])[0] - search.lower(b0[None])[0])   # noqa: E731
    digest = None if checkpoint is None else _model_digest(model)
    trial = 0
    if checkpoint is not None and checkpoint.exists():
        trial = _restore_search(search, checkpoint, digest, epsilon)
        if trial > trials:
            raise ValueError(f"{checkpoint} holds {trial} trials, more than the budget of {trials} (section 7).")
        if gap() <= epsilon:
            # The stored search had stopped at this trial. Another trial would go beyond the uninterrupted run.
            return search, trial
    for trial in range(trial + 1, trials + 1):
        search.explore(b0, epsilon, 0)
        if back_up_corners:
            search.back_up_corners()
        reached = gap() <= epsilon
        if reached or trial == trials or trial % every == 0:
            if checkpoint is not None:
                _save_search(search, checkpoint, digest, epsilon, trial)
            if report is not None:
                report(trial, float(search.lower(b0[None])[0]), float(search.upper(b0[None])[0]), search.vectors.size)
        if reached:
            break
    return search, trial


@dataclass(frozen=True)
class VectorBracket:
    """The bracket of hsvi_budget without the pointer controller: lower-bound vectors and their actions only."""

    lower: float
    upper: float
    vectors: torch.Tensor
    actions: torch.Tensor
    trials: int


def hsvi_vector_bracket(model: FinitePOMDP, epsilon: float, trials: int, checkpoint: Path | None = None,
                        every: int = 10,
                        report: Callable[[int, float, float, int], None] | None = None) -> VectorBracket:
    """
    hsvi_budget (no seeds, no corner backups) returning the lower-bound vectors and their actions instead of the
    controller. The controller's successor law is dense (K, A, O, K) in float64: with the K = 49,909 vectors that
    1,000 trials left on the level-1 fold (A = 3, O = 6) it needs 8 * 3 * 6 * K^2 bytes = 359 GB, which killed a run
    after all its trials had finished. A reference agent needs only the vectors (the best vector's action at the
    exact belief is worth at least L).

    Args:
        model: Exact POMDP specification.
        epsilon: Target width; it also sets the trials' depth and gap thresholds, as in hsvi().
        trials: The trial budget.
        checkpoint: Optional file (section 7). If it exists the search continues from it, and the result equals the
            uninterrupted run's. It is rewritten every `every` trials and after the last one.
        every: Trials between checkpoints and between progress reports.
        report: Optional callback (trial, L(b0), U(b0), number of lower-bound vectors), called at the same trials.
    """
    search, trial = _run_budget(model, epsilon, trials, None, None, False, checkpoint, every, report)
    b0 = model.initial_belief.numpy()
    return VectorBracket(float(search.lower(b0[None])[0]), float(search.upper(b0[None])[0]),
                         torch.from_numpy(search.vectors.rows.copy()), torch.tensor(search.actions, dtype=torch.int64),
                         trial)


@dataclass(frozen=True)
class RelaxedBound:
    """
    A certified upper bound of V* from an information relaxation over a finite belief graph (section 5).

    Attributes:
        upper: The bound at b0 (merge slack included).
        root_values: The bound at every root, b0 first, then the extra roots in the order given (slack included).
        beliefs: float64 (K, S), every enumerated belief's marginal over the original states.
        values: float64 (K,), an upper bound of V* at each of those marginals (slack included).
        merge_slack: The slack added for merged beliefs (section 5).
    """

    upper: float
    root_values: torch.Tensor
    beliefs: torch.Tensor
    values: torch.Tensor
    merge_slack: float


def revealed_upper_bound(model: FinitePOMDP, reveal: Callable[[int, tuple[int, ...]], int], memory: int,
                         extra_roots: torch.Tensor | None = None, rounding: float = 1e-10,
                         max_beliefs: int = 1_000_000) -> RelaxedBound:
    """
    V*_relaxed at b0 (and at extra roots) when the agent also observes reveal(a, history) on every transition
    (section 5). An upper bound of V* of `model`.

    Args:
        model: Exact POMDP specification.
        reveal: The extra symbol emitted when action a arrives at history[-1]; history holds the last memory + 1
            states, oldest first (at the first step the past is padded with the initial state itself).
        memory: How many past states reveal may read (>= 1).
        extra_roots: Optional float64 (k, S) beliefs whose bounds are wanted as well (e.g. the corners).
        rounding: Merge tolerance of the enumeration (section 5).
        max_beliefs: Exceeding it raises: the bound is never computed on a truncated graph.
    """
    if memory < 1:
        raise ValueError(f"memory must be >= 1, got {memory}.")
    T, O, R = model.transition.numpy(), model.observation.numpy(), model.reward.numpy()
    gamma, (num_a, num_s), num_o = model.discount, model.reward.shape, model.observation.shape[2]
    starts = [model.initial_belief.numpy()] + ([] if extra_roots is None else list(extra_roots.numpy()))
    # Histories reachable from every root's support (standard form, section 5).
    histories: dict[tuple[int, ...], int] = {}
    queue = []
    for start in starts:
        for c in np.nonzero(start > 0)[0]:
            h = (int(c),) * (memory + 1)
            if h not in histories:
                histories[h] = len(histories)
                queue.append(h)
    successors = [np.nonzero(T[:, c].sum(axis=0) > 0)[0] for c in range(num_s)]
    while queue:
        h = queue.pop()
        for c2 in successors[h[-1]]:
            h2 = h[1:] + (int(c2),)
            if h2 not in histories:
                histories[h2] = len(histories)
                queue.append(h2)
    num_x = len(histories)
    keys = list(histories)
    last = np.array([h[-1] for h in keys])
    rows, cols = zip(*[(x, histories[h[1:] + (int(c2),)]) for h, x in histories.items() for c2 in successors[h[-1]]])
    rows, cols = np.array(rows), np.array(cols)
    transition = [sparse.csr_matrix((T[a, last[rows], last[cols]], (rows, cols)), shape=(num_x, num_x))
                  for a in range(num_a)]
    observation = O[:, last, :]                                                        # (A, X, O)
    labels = np.array([[reveal(a, h) for h in keys] for a in range(num_a)])           # (A, X)
    masks = [[(labels[a] == y).astype(np.float64) for y in np.unique(labels[a])] for a in range(num_a)]

    def augment(belief: np.ndarray) -> np.ndarray:
        x = np.zeros(num_x)
        for c in np.nonzero(belief > 0)[0]:
            x[histories[(int(c),) * (memory + 1)]] = belief[c]
        return x

    # Breadth-first enumeration with merging (section 5).
    store, index, merge = _Rows(num_x), {}, 0.0
    edges: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [[] for _ in range(num_a)]

    def insert(beliefs: np.ndarray) -> tuple[np.ndarray, list[int]]:
        nonlocal merge
        rounded = np.round(beliefs / rounding).astype(np.int64)
        found, fresh = np.empty(len(beliefs), dtype=np.int64), []
        for r in range(len(beliefs)):
            digest = hashlib.blake2b(rounded[r].tobytes(), digest_size=16).digest()
            j = index.get(digest)
            if j is None:
                j = store.size
                index[digest] = j
                store.append(beliefs[r])
                fresh.append(j)
            else:
                if not np.array_equal(np.round(store.data[j] / rounding).astype(np.int64), rounded[r]):
                    raise RuntimeError("belief digest collision.")
                merge = max(merge, float(np.abs(store.data[j] - beliefs[r]).sum()))
            found[r] = j
        return found, fresh

    roots, frontier = insert(np.stack([augment(s) for s in starts]))
    frontier = np.array(sorted(set(frontier)), dtype=np.int64)
    while frontier.size:
        parents, fresh = store.data[frontier], []
        for a in range(num_a):
            moved = np.asarray(transition[a].T @ parents.T).T                           # (F, X): b T_a
            for mask in masks[a]:
                part = moved * mask
                for o in range(num_o):
                    joint = part * observation[a, :, o]
                    probs = joint.sum(axis=1)
                    live = np.nonzero(probs > 0)[0]
                    if live.size == 0:
                        continue
                    children, new = insert(joint[live] / probs[live, None])
                    fresh += new
                    edges[a].append((frontier[live], children, probs[live]))
        if store.size > max_beliefs:
            raise RuntimeError(f"the relaxed belief graph exceeds {max_beliefs} beliefs.")
        frontier = np.array(fresh, dtype=np.int64)
    num_b = store.size
    beliefs = store.rows
    step = [sparse.csr_matrix((np.concatenate([e[2] for e in edges[a]]),
                               (np.concatenate([e[0] for e in edges[a]]), np.concatenate([e[1] for e in edges[a]]))),
                              shape=(num_b, num_b)) for a in range(num_a)]
    reward = beliefs @ R[:, last].T                                                   # (K, A)
    value = np.full(num_b, R.max() / (1 - gamma))
    while True:
        new = np.max(np.stack([reward[:, a] + gamma * (step[a] @ value) for a in range(num_a)], axis=1), axis=1)
        change = float(np.abs(new - value).max())
        value = new
        if change < 1e-10:
            break
    span = (R.max() - R.min()) / (1 - gamma)
    slack = span / 2 * merge / (1 - gamma)
    marginal = sparse.csr_matrix((np.ones(num_x), (np.arange(num_x), last)), shape=(num_x, num_s))
    bounded = torch.from_numpy(value + slack)
    return RelaxedBound(float(bounded[roots[0]]), bounded[torch.from_numpy(roots)],
                        torch.from_numpy(np.asarray(beliefs @ marginal)), bounded, slack)
