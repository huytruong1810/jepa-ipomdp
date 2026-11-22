# ABSOLUTE PATH: tests/test_interactive_analysis.py
"""
The interactive analysis path (src/ipomdp/experiments/interactive_analysis.py) on a small fold.

The fold is agent i against a hand-built two-node controller of j (listen; after an even-indexed observation open the
left door for one step), so the other agent's node marginal is not trivial. The benchmark is valid by construction: the
blind vectors (I - gamma T_a)^-1 R_a are the exact values of "always a" (lower bounds), and R_max / (1 - gamma) bounds
every value from above. What is verified:
    1. A fold's digest identifies it; a benchmark file round-trips and is validated.
    2. The reference sets: each action's vectors plus the constant floor; the greedy action over the sets is the action
       of the best vector; a benchmark of another fold, or one whose vectors do not give its lower bound, is refused.
    3. The marginal divergences vanish for a probe that returns the exact beliefs and are positive for a uniform one.
    4. The whole analysis runs on an untrained agent and reports the bracket, the reference agent and both marginals.
    5. The probe control (module section 5) reads the trained probes' histories: with the trained filter in its place
       it reproduces the trained MLP reports exactly; through a run directory its filter is the run's initial one.
"""

import json
import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ipomdp.agents import PlanningAgent
from ipomdp.domain import TigerAction
from ipomdp.domain.interactive import fold
from ipomdp.domain.multiagent_tiger import Agent, build_multiagent_tiger
from ipomdp.domain.policy_graph import PolicyGraph
from ipomdp.experiments import AnalysisSettings, analyze_run, build_training_run
from ipomdp.experiments.interactive_analysis import (InteractiveAnalysisSettings, InteractiveBenchmark,
                                                     analyze_interactive, analyze_interactive_run, fold_digest,
                                                     initial_acting_filter, interactive_problem, marginal_reports,
                                                     reference_sets)
from ipomdp.experiments.runs import load_trained_run
from ipomdp.interpretability.belief_probe import ProbeDataset
from ipomdp.models import BeliefFilter, ObservationHead, RewardHead, TwoHotSymlog, ValueHead
from ipomdp.planning import BeliefTreeSearch, LearnedSearchModel
from ipomdp.training import save_checkpoint

CPU = torch.device("cpu")
CONF_DIR = str(Path(__file__).resolve().parents[1] / "conf")
L, OL = int(TigerAction.LISTEN), int(TigerAction.OPEN_LEFT)


def _controller(switch_parity: int) -> PolicyGraph:
    """Node 0 listens and moves to node 1 after an observation of the given index parity; node 1 opens left once."""
    num_a, num_o = 3, 6
    action_law = torch.zeros(2, num_a, dtype=torch.float64)
    action_law[0, L], action_law[1, OL] = 1.0, 1.0
    successor = torch.zeros(2, num_a, num_o, 2, dtype=torch.float64)
    successor[:, :, :, :] = torch.eye(2, dtype=torch.float64)[:, None, None, :]      # untaken actions: self-loops
    for o in range(num_o):
        successor[0, L, o] = torch.tensor([0.0, 1.0] if o % 2 == switch_parity else [1.0, 0.0], dtype=torch.float64)
        successor[1, OL, o] = torch.tensor([1.0, 0.0], dtype=torch.float64)
    return PolicyGraph(action_law, successor, torch.tensor([1.0, 0.0], dtype=torch.float64), ("listen", "open"))


@pytest.fixture(scope="module")
def problem():
    return fold(build_multiagent_tiger(0.95).view(Agent.I), _controller(0), L)


def _blind_benchmark(problem) -> InteractiveBenchmark:
    pomdp = problem.pomdp
    eye = torch.eye(pomdp.num_states, dtype=torch.float64)
    vectors = torch.stack([torch.linalg.solve(eye - pomdp.discount * pomdp.transition[a], pomdp.reward[a])
                           for a in range(pomdp.num_actions)])
    actions = torch.arange(pomdp.num_actions)
    lower = float((vectors @ pomdp.initial_belief).max())
    upper = float(pomdp.reward.max()) / (1.0 - pomdp.discount)
    return InteractiveBenchmark(lower, upper, vectors, actions, fold_digest(pomdp), "blind vectors / R_max bound (test)")


class TestBenchmark:

    def test_a_digest_identifies_the_fold(self, problem):
        other = fold(build_multiagent_tiger(0.95).view(Agent.I), _controller(1), L)
        assert fold_digest(problem.pomdp) == fold_digest(fold(build_multiagent_tiger(0.95).view(Agent.I), _controller(0), L).pomdp)
        assert fold_digest(problem.pomdp) != fold_digest(other.pomdp)
        assert problem.other.num_nodes == 2 and problem.pomdp.num_states > 2

    def test_round_trip_and_validation(self, problem, tmp_path):
        benchmark = _blind_benchmark(problem)
        benchmark.save(tmp_path / "benchmark.pt")
        loaded = InteractiveBenchmark.load(tmp_path / "benchmark.pt")
        assert (loaded.lower, loaded.upper, loaded.digest, loaded.source) == (benchmark.lower, benchmark.upper,
                                                                              benchmark.digest, benchmark.source)
        assert torch.equal(loaded.vectors, benchmark.vectors) and torch.equal(loaded.actions, benchmark.actions)
        with pytest.raises(ValueError):
            InteractiveBenchmark(1.0, 0.0, benchmark.vectors, benchmark.actions, benchmark.digest, "")
        with pytest.raises(ValueError):
            InteractiveBenchmark(0.0, 1.0, benchmark.vectors.float(), benchmark.actions, benchmark.digest, "")


class TestReferenceSets:

    def test_groups_add_the_floor_and_the_greedy_action_is_the_best_vectors(self, problem):
        benchmark = _blind_benchmark(problem)
        value, per_action = reference_sets(benchmark, problem.pomdp)
        floor = float(problem.pomdp.reward.min()) / (1.0 - problem.pomdp.discount)
        for a, group in enumerate(per_action):
            assert len(group.vectors) == int((benchmark.actions == a).sum()) + 1
            assert torch.equal(group.vectors[-1], torch.full((problem.pomdp.num_states,), floor, dtype=torch.float64))
        beliefs = torch.distributions.Dirichlet(torch.ones(problem.pomdp.num_states, dtype=torch.float64)).sample((64,))
        grouped = torch.stack([g.value(beliefs) for g in per_action], dim=-1).argmax(-1)
        assert torch.equal(grouped, value.greedy_action(beliefs))

    def test_refuses_another_fold_or_an_inconsistent_lower_bound(self, problem):
        benchmark = _blind_benchmark(problem)
        other = fold(build_multiagent_tiger(0.95).view(Agent.I), _controller(1), L)
        with pytest.raises(ValueError):
            reference_sets(benchmark, other.pomdp)
        shifted = InteractiveBenchmark(benchmark.lower - 1.0, benchmark.upper, benchmark.vectors, benchmark.actions,
                                       benchmark.digest, benchmark.source)
        with pytest.raises(ValueError):
            reference_sets(shifted, problem.pomdp)


class _Probe:
    """A stand-in probe whose decoded log-belief is a fixed function of the latents."""

    def __init__(self, log_probs):
        self._log_probs = log_probs

    def log_probs(self, latents):
        return self._log_probs(latents)


def test_marginal_divergences_vanish_for_the_exact_beliefs(problem):
    beliefs = torch.distributions.Dirichlet(torch.ones(problem.pomdp.num_states, dtype=torch.float64)).sample((200,))
    dataset = ProbeDataset(latents=beliefs.log().float(), posteriors=beliefs)
    exact = marginal_reports(_Probe(lambda z: beliefs.log()), dataset, problem)
    for report in exact.values():
        assert report.kl_mean == pytest.approx(0.0, abs=1e-12) and report.l1_mean == pytest.approx(0.0, abs=1e-12)
    uniform = marginal_reports(_Probe(lambda z: torch.full_like(beliefs, -math.log(problem.pomdp.num_states))), dataset,
                               problem)
    assert all(report.kl_mean > 0.0 for report in uniform.values())


EPISODES, DIM, HIDDEN = 8, 8, 16


def _agent(problem) -> tuple[BeliefFilter, PlanningAgent]:
    """An untrained filter and the greedy planner over it, batch size EPISODES."""
    pomdp = problem.pomdp
    num_a, num_o = pomdp.num_actions, pomdp.num_observations
    belief_filter = BeliefFilter(num_a, num_o, DIM, HIDDEN, 1)
    codec = TwoHotSymlog(31, pomdp.value_bound)
    model = LearnedSearchModel(belief_filter, RewardHead(DIM, num_a, HIDDEN, 1, 31),
                               ObservationHead(DIM, num_a, num_o, HIDDEN, 1), ValueHead(DIM, HIDDEN, 1, 31), codec, num_a,
                               num_o, pomdp.discount)
    planner = PlanningAgent(model, BeliefTreeSearch(model, 4, 1.25, 0.3, 0.0, seed=0), EPISODES, temperature=0.0, seed=0,
                            device=CPU)
    return belief_filter, planner


def _analyze(problem, belief_filter, initial_filter, planner):
    return analyze_interactive(problem, belief_filter, initial_filter, planner, _blind_benchmark(problem), episode_length=6,
                               num_episodes=EPISODES, mlp_probe_steps=20, seed=1, device=CPU)


def test_the_analysis_reports_the_bracket_the_reference_and_both_marginals(problem):
    torch.manual_seed(0)
    belief_filter, planner = _agent(problem)
    initial_filter = BeliefFilter(problem.pomdp.num_actions, problem.pomdp.num_observations, DIM, HIDDEN, 1)
    benchmark = _blind_benchmark(problem)
    analysis = _analyze(problem, belief_filter, initial_filter, planner)
    assert analysis.bracket["width"] == pytest.approx(benchmark.upper - benchmark.lower)
    assert analysis.reference.optimal_value_b0 == pytest.approx(benchmark.lower, abs=1e-12)
    assert {"optimal", "learned_planner", "decoded_linear", "decoded_mlp"} <= set(analysis.reference.returns)
    assert set(analysis.marginals) == {"linear/random", "linear/agent", "mlp/random", "mlp/agent"}
    assert set(analysis.untrained.marginals) == set(analysis.marginals)
    assert set(analysis.untrained.probes) == {"linear/random", "mlp/random"}
    for reports in (*analysis.marginals.values(), *analysis.untrained.marginals.values()):
        assert set(reports) == {"physical", "other_node"}
        assert all(math.isfinite(r.kl_mean) and r.kl_mean >= -1e-12 for r in reports.values())
    assert all(math.isfinite(r.mean_kl) and r.mean_kl >= -1e-12 for r in analysis.untrained.probes.values())


def test_the_control_reads_the_histories_of_the_trained_probes(problem):
    """
    With the trained filter in the control's place, every control MLP report equals its trained counterpart exactly:
    the MLP fit is reseeded per fit, so equality holds only if the control's histories are the trained probes' (the
    replayed streams of analyze_beliefs, and section 3's own episodes). The linear fits start from the global
    generator's state and are not compared.
    """
    torch.manual_seed(0)
    belief_filter, planner = _agent(problem)
    analysis = _analyze(problem, belief_filter, belief_filter, planner)
    assert analysis.untrained.probes["mlp/random"] == analysis.reference.probes["mlp/random"]
    for source in ("random", "agent"):
        assert analysis.untrained.marginals[f"mlp/{source}"] == analysis.marginals[f"mlp/{source}"]


@pytest.mark.slow
def test_a_trained_itiger_run_is_analysed_against_its_benchmark_file(tmp_path):
    """
    End to end through a run directory: a tiny env=itiger run (j's controller extracted and certified at 0.5, so the
    fold is built in seconds) is trained one collection, saved like main.py, reloaded, its fold rebuilt from its config
    and matched by digest, and analysed against a benchmark file; a benchmark of another fold is refused.
    """
    overrides = ["env=itiger", "env.bracket_epsilon=0.5", "env.controller_epsilon=0.5", "training.env_batch_size=4",
                 "training.warmup_episodes=4", "training.batch_size=4", "training.updates_per_collection=2",
                 "training.eval_episodes=4", "env.max_steps=8", "model.latent_dim=8", "model.hidden_dim=16",
                 "mcts.num_simulations=4"]
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    _, run = build_training_run(cfg, CPU)
    initial = {name: value.clone() for name, value in run.acting_filter.state_dict().items()}
    run.collect_and_train()
    (tmp_path / ".hydra").mkdir()
    OmegaConf.save(cfg, tmp_path / ".hydra" / "config.yaml")
    save_checkpoint(tmp_path / "checkpoints" / "best.pt", {
        "collection": 1, "eval_return": 0.0, "networks": {name: net.state_dict() for name, net in run.networks.items()}})
    problem = interactive_problem(cfg.env)
    _blind_benchmark(problem).save(tmp_path / "benchmark.pt")
    settings = InteractiveAnalysisSettings(episodes=4, mlp_probe_steps=10, seed=1)
    analysis = analyze_interactive_run(tmp_path, settings, tmp_path / "benchmark.pt", CPU)
    report = json.loads((tmp_path / "analysis" / "report.json").read_text())
    assert report["bracket"]["lower"] == analysis.bracket["lower"] and report["checkpoint_collection"] == 1
    assert set(report["marginals"]["mlp/agent"]) == {"physical", "other_node"}
    assert set(report["untrained"]["probes"]) == {"linear/random", "mlp/random"}
    assert set(report["untrained"]["marginals"]["mlp/agent"]) == {"physical", "other_node"}
    trained = load_trained_run(tmp_path, CPU)                 # the control's filter is the run's initial one, not best.pt
    control = initial_acting_filter(trained, CPU).state_dict()
    assert all(torch.equal(control[name], value) for name, value in initial.items())
    assert not all(torch.equal(trained.run.acting_filter.state_dict()[name], value) for name, value in initial.items())
    other = fold(build_multiagent_tiger(0.95).view(Agent.I), _controller(0), L)
    _blind_benchmark(other).save(tmp_path / "other.pt")
    with pytest.raises(ValueError):
        analyze_interactive_run(tmp_path, settings, tmp_path / "other.pt", CPU)
    with pytest.raises(TypeError):                         # the exact-solver path refuses the fold before loading it
        analyze_run(tmp_path, AnalysisSettings(), CPU)
