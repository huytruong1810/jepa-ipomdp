"""
Domain layer: POMDP specifications, the canonical Tiger, the exact Bayes filter, the exact
value-iteration solver, certified HSVI brackets and a batched simulator; the two-agent tiger game with
policy graphs and the fold of a finitely nested I-POMDP into a POMDP; and Wumpus World (generative
simulator and exact enumeration filter, no solver). Every other layer treats the specifications here (FinitePOMDP,
WumpusWorld.step_rules) as the single source of truth for each environment's dynamics.
"""

from .belief import belief_update, initial_beliefs, observation_distribution, predict_state
from .bounds import Bracket, hsvi
from .env import BatchedPOMDPEnv, StepOutput
from .interactive import (TIGER_LEVEL_ZERO_NOISE, InteractiveModel, Level, NestedProblem, fold, nested_problem,
                          noise_model)
from .multiagent_tiger import Agent, AgentView, TwoAgentModel, build_multiagent_tiger
from .policy_graph import PolicyGraph, constant_policy, envelope, evaluate, plan_graph, stationary_graph
from .pomdp import FinitePOMDP
from .reduction import ObservationLumping, expand_observations, lump_observations
from .solver import (
    EXACT_PRUNE_EPSILON,
    AlphaVectorSet,
    InfiniteHorizonSolution,
    action_value_functions,
    backup,
    prune,
    solve_finite_horizon,
    solve_infinite_horizon,
    sup_norm_distance,
)
from .tiger import TigerAction, TigerObservation, TigerState, build_tiger_pomdp
from .wumpus import (WumpusAction, WumpusBeliefTracker, WumpusMarginals, WumpusSimulator, WumpusState, WumpusWorld,
                     game_outcome_flags, game_outcomes)

__all__ = [
    "EXACT_PRUNE_EPSILON",
    "TIGER_LEVEL_ZERO_NOISE",
    "Agent",
    "AgentView",
    "AlphaVectorSet",
    "BatchedPOMDPEnv",
    "Bracket",
    "FinitePOMDP",
    "InfiniteHorizonSolution",
    "InteractiveModel",
    "Level",
    "NestedProblem",
    "ObservationLumping",
    "PolicyGraph",
    "StepOutput",
    "TwoAgentModel",
    "TigerAction",
    "TigerObservation",
    "TigerState",
    "WumpusAction",
    "WumpusBeliefTracker",
    "WumpusMarginals",
    "WumpusSimulator",
    "WumpusState",
    "WumpusWorld",
    "action_value_functions",
    "backup",
    "belief_update",
    "build_multiagent_tiger",
    "build_tiger_pomdp",
    "constant_policy",
    "envelope",
    "evaluate",
    "expand_observations",
    "fold",
    "game_outcome_flags",
    "game_outcomes",
    "hsvi",
    "initial_beliefs",
    "lump_observations",
    "nested_problem",
    "noise_model",
    "observation_distribution",
    "plan_graph",
    "predict_state",
    "prune",
    "solve_finite_horizon",
    "stationary_graph",
    "solve_infinite_horizon",
    "sup_norm_distance",
]
