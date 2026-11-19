# ABSOLUTE PATH: src/ipomdp/domain/reduction.py
# ==============================================================================
# EXACT REDUCTIONS OF A FINITE POMDP: MERGING OBSERVATIONS THAT CARRY THE SAME EVIDENCE
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Observation Lumping Is Value-Preserving:
#    - Two observations o1, o2 carry the same evidence about the arrival state when, for every action a, their
#      likelihood columns are proportional: O(o1 | s', a) = c_a O(o2 | s', a) for all s'. Then
#      tau(b, a, o1) = tau(b, a, o2) for every belief, so in the Bellman backup
#         V_{h+1}(b) = max_a [ b . R_a + gamma sum_o P(o | b, a) V_h(tau(b, a, o)) ]
#      the two terms share V_h(tau) and add their probabilities. The model with the group merged into one
#      observation, O(m | s', a) = sum_{o in group} O(o | s', a), has exactly the same optimal values and policies.
#    - Why it matters here: a level-0 opponent of the two-agent tiger game folds the other agent's actions into its
#      tables as fixed noise (domain/interactive.py, section 2). Its creak then depends only on that noise, not on the
#      state, so the six (growl, creak) pairs carry the evidence of two growls. Incremental pruning performs one
#      cross-sum prune per observation, so the merged model is solved far faster with identical results.
#    - The canonical Tiger has no proportional observations (growl left and right have reversed likelihoods), so
#      nothing in the single-agent benchmark changes.
#
# 2. A Controller of the Reduced Model Acts in the Original One:
#    - A policy graph extracted from the reduced model moves on merged observations. expand_observations maps it
#      back: an original observation moves the controller as its group does. The controller is identical as a
#      function of the observation history, so folding it into another agent's problem (interactive.py) is exact.
#
# 3. Tolerance:
#    - Proportionality is tested on the normalised columns with an absolute tolerance of 1e-12: the tables are
#      products of decimal literals, so equal evidence is equal to round-off, and real differences are large.
# ==============================================================================

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .policy_graph import PolicyGraph
from .pomdp import FinitePOMDP

_PROPORTIONAL_ATOL = 1e-12


@dataclass(frozen=True)
class ObservationLumping:
    """
    Attributes:
        model: The reduced model, one observation per group.
        group: The group of every original observation, int64 (O,).
    """

    model: FinitePOMDP
    group: Tensor


def _evidence(model: FinitePOMDP, o: int) -> Tensor:
    """The likelihood column of o under every action, each normalised to sum 1 (zero columns stay zero), (A, S)."""
    column = model.observation[:, :, o]
    total = column.sum(-1, keepdim=True)
    return torch.where(total > 0, column / total.clamp_min(1e-300), torch.zeros_like(column))


def lump_observations(model: FinitePOMDP) -> ObservationLumping:
    """Merges every group of observations with proportional likelihoods under every action (section 1)."""
    representatives: list[int] = []
    group = torch.empty(model.num_observations, dtype=torch.int64)
    for o in range(model.num_observations):
        evidence = _evidence(model, o)
        for g, r in enumerate(representatives):
            same_support = torch.equal(model.observation[:, :, o].sum(-1) > 0, model.observation[:, :, r].sum(-1) > 0)
            if same_support and torch.allclose(evidence, _evidence(model, r), atol=_PROPORTIONAL_ATOL, rtol=0.0):
                group[o] = g
                break
        else:
            group[o] = len(representatives)
            representatives.append(o)
    merged = torch.zeros(model.num_actions, model.num_states, len(representatives), dtype=torch.float64)
    merged.index_add_(2, group, model.observation)
    names = tuple("+".join(model.observation_names[o] for o in range(model.num_observations) if group[o] == g)
                  for g in range(len(representatives)))
    return ObservationLumping(replace(model, observation=merged, observation_names=names), group)


def expand_observations(graph: PolicyGraph, group: Tensor) -> PolicyGraph:
    """The controller of a reduced model as a controller over the original observations (section 2)."""
    return replace(graph, successor_law=graph.successor_law[:, :, group, :].contiguous())
