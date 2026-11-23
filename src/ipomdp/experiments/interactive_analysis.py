# ABSOLUTE PATH: src/ipomdp/experiments/interactive_analysis.py
# ==============================================================================
# ANALYSIS OF AN INTERACTIVE-TIGER RUN AGAINST ITS CERTIFIED BRACKET (CONTRIBUTION 2, E1)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Why a Separate Path:
#    - The canonical analysis (experiments/runs.py, analyze_run) scores the learned agent against the exact optimum of
#      a certified infinite-horizon solve. The folded interactive problems (domain/interactive.py) are out of that
#      solver's reach. Their benchmark is a certified bracket [L, U] of V*(b0) (domain/bounds.py), computed ONCE per fold
#      (the level-1 relaxation alone takes about 30 min and 6 GB), stored in a benchmark file and read here, never
#      recomputed per seed. The Tiger path is not touched, so its reports stay byte-identical.
#
# 2. The Reference Agent and What It Guarantees:
#    - The bracket's lower bound is a set of alpha vectors {alpha_k} with actions {a_k}, each the exact value of a pointer
#      plan, so L(b) = max_k alpha_k . b <= V*(b). The reference agent tracks the fold's exact belief and plays the action
#      of the best vector at that belief. The HSVI lower bound is uniformly improvable (L <= T^pi L for that greedy
#      pointer policy), so the reference agent's value is at least L, and its return estimates a value inside [L, U].
#    - The interpretability code (interpretability/analysis.py: ExactBeliefAgent, DecodedBeliefAgent, error_bounds)
#      reads per-action vector sets. They are the vectors grouped by action, each with one constant vector
#      R_min / (1 - gamma): every policy's value is at least that, so every set stays a lower bound of its Q*(., a), none
#      is empty, and the constant never beats a plan vector. The greedy action over the groups is then the action of the
#      best vector, i.e. the reference agent above.
#    - Consequently the span-Holder bounds (interpretability/error_bounds.py, section 4) are statements about the
#      lower bound, not about the unknown optimum: the value error is that of the function L, the regret is measured
#      in the per-action lower bounds, and the policy form bounds a decoded-belief agent's loss against L(b0), the
#      value the reference agent attains or exceeds. The reported bracket width bounds V*(b0) - L(b0); adding it to
#      the policy bound gives a bound on the loss against the optimum. The report names both.
#    - Both the reference agent's guarantee and the policy bound need every vector to be the value of its own
#      pointer plan in the stored set. That is checked on the two benchmark files of the experiments, to 3.4e-13
#      (reports/2026-11-16_code_audit/benchmark_improvability.py and .log), not assumed.
#
# 3. Interactive Marginals (the theory-of-mind readout):
#    - The fold's state is (s, n_j, a_j_prev). The probes of interpretability/analysis.py decode the whole interactive
#      belief. This module adds the divergence of the decoded belief from the exact one on the physical marginal P(s)
#      and on the other agent's node marginal P(n_j) (InteractiveModel.physical_marginal, other_node_marginal). The
#      decoded marginals are summed in log space (logsumexp over each group), so no probability is clamped. The previous
#      action a_j_prev is a convention at t = 0 (domain/interactive.py, section 4) and gets no marginal of its own.
#    - These marginal probes are fitted on their own data sets (streams 11-16 of the analysis seed), separate from the
#      data sets analyze_beliefs builds (streams 1-6), because analyze_beliefs does not return its fitted probes.
#
# 4. A Benchmark File Is Tied to Its Fold:
#    - The file stores a SHA-256 digest of the fold's tables (T, O, R, b0) and gamma. The analysis refuses a file whose
#      digest differs from the run's fold, or whose lower bound does not reproduce L(b0) from its vectors, so a
#      benchmark is never read against another problem.
#
# 5. Probe Control: the Run's Own Untrained Filter:
#    - A recurrent network carries much of its input history before any training (on the canonical Tiger an untrained
#      GRU already reaches belief R^2 = 0.93; interpretability/belief_probe.py), so a probe's divergence is evidence
#      of learning only against the same probe on the untrained network (probe controls, Hewitt & Liang 2019). The
#      control is the run's INITIAL acting filter: a TrainingRun built from the run's own config and seed holds the
#      networks training started from. They are created on the CPU from the run's NETWORK_INIT stream and then moved
#      (training/run.py), so the analysis device does not change them.
#    - Same probes on the same histories, so every control number is paired with a trained one:
#        * the interactive-state probes "<linear|mlp>/random" read analyze_beliefs' random histories, replayed from
#          its streams (1 and then 2, played by one uniform agent on stream 6; interpretability/analysis.py). Its
#          agent histories depend on the planner's generator state at that point and cannot be replayed, so the
#          control has no agent-history state probe;
#        * the four marginal probes of section 3 read section 3's episodes themselves, played once and read by both
#          filters.
#    - The MLP probes are reseeded for every fit (fit_mlp_probe), so with the trained filter in the control's place
#      the control reproduces the trained MLP reports exactly (tests/test_interactive_analysis.py). The linear probes
#      start from the global generator's state and are compared, not reproduced.
#    - Readout: descriptive, no threshold. What the trained probes gain over this control is what training adds.
# ==============================================================================

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from ..agents import PlanningAgent, UniformRandomAgent
from ..domain import BatchedPOMDPEnv, FinitePOMDP, TigerAction
from ..domain.interactive import TIGER_LEVEL_ZERO_NOISE, InteractiveModel, nested_problem, tiger_growl_counters
from ..domain.multiagent_tiger import Agent, build_multiagent_tiger
from ..domain.solver import AlphaVectorSet
from ..interpretability import BeliefAnalysis, analyze_beliefs, build_probe_dataset
from ..interpretability.belief_probe import ProbeDataset, ProbeReport, evaluate_probe, fit_linear_probe, fit_mlp_probe
from ..models import BeliefFilter
from ..planning import BeliefTreeSearch
from ..training import EpisodeBatch, TrainingRun, play_episodes, save_checkpoint
from ..training.seeding import stream_seed
from .runs import TrainedRun, build_run_config, load_trained_run

# L(b0) recomputed from the stored vectors must equal the stored lower bound to this tolerance (section 4).
LOWER_BOUND_TOLERANCE = 1e-9


def fold_digest(pomdp: FinitePOMDP) -> str:
    """SHA-256 of a fold's tables (T, O, R, b0, float64) and its discount (section 4)."""
    digest = hashlib.sha256()
    for table in (pomdp.transition, pomdp.observation, pomdp.reward, pomdp.initial_belief):
        digest.update(str(tuple(table.shape)).encode())
        digest.update(table.detach().cpu().to(torch.float64).contiguous().numpy().tobytes())
    digest.update(repr(float(pomdp.discount)).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class InteractiveBenchmark:
    """
    A certified bracket of one fold's V*(b0) and the lower-bound vectors behind it (sections 2 and 4).

    Attributes:
        lower: L(b0), attained or exceeded by the reference agent.
        upper: U(b0) >= V*(b0).
        vectors: float64 (K, S), the lower bound's alpha vectors.
        actions: int64 (K,), the action of each vector.
        digest: fold_digest of the fold the bracket was computed on.
        source: How the bracket was computed (method, budget, commit), for the report.
    """

    lower: float
    upper: float
    vectors: Tensor
    actions: Tensor
    digest: str
    source: str

    def __post_init__(self) -> None:
        if not self.lower <= self.upper:
            raise ValueError(f"a bracket needs lower <= upper, got [{self.lower}, {self.upper}].")
        if self.vectors.dtype != torch.float64 or self.vectors.ndim != 2 or self.vectors.shape[0] < 1:
            raise ValueError(f"vectors must be float64 of shape (K >= 1, S), got {self.vectors.dtype} {tuple(self.vectors.shape)}.")
        if self.actions.dtype != torch.int64 or self.actions.shape != (self.vectors.shape[0],):
            raise ValueError(f"actions must be int64 of shape ({self.vectors.shape[0]},), got {self.actions.dtype} "
                             f"{tuple(self.actions.shape)}.")

    def save(self, path: Path) -> None:
        """Writes the benchmark atomically (training/checkpointing.py)."""
        save_checkpoint(path, {"lower": self.lower, "upper": self.upper, "vectors": self.vectors,
                               "actions": self.actions, "digest": self.digest, "source": self.source})

    @staticmethod
    def load(path: Path) -> "InteractiveBenchmark":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return InteractiveBenchmark(float(payload["lower"]), float(payload["upper"]), payload["vectors"],
                                    payload["actions"], str(payload["digest"]), str(payload["source"]))


def reference_sets(benchmark: InteractiveBenchmark, pomdp: FinitePOMDP) -> tuple[AlphaVectorSet, list[AlphaVectorSet]]:
    """
    The reference value set and its per-action sets (section 2). Raises if the benchmark was not computed on `pomdp` or
    if its vectors do not reproduce its lower bound at b0.
    """
    if benchmark.digest != fold_digest(pomdp):
        raise ValueError("the benchmark was computed on a different fold (digest mismatch, section 4).")
    if benchmark.vectors.shape[1] != pomdp.num_states:
        raise ValueError(f"benchmark vectors have {benchmark.vectors.shape[1]} states, the fold {pomdp.num_states}.")
    value = AlphaVectorSet(benchmark.vectors, benchmark.actions)
    lower = float(value.value(pomdp.initial_belief.unsqueeze(0)))
    if abs(lower - benchmark.lower) > LOWER_BOUND_TOLERANCE:
        raise ValueError(f"the vectors give L(b0) = {lower}, the benchmark states {benchmark.lower}.")
    floor = torch.full((1, pomdp.num_states), float(pomdp.reward.min()) / (1.0 - pomdp.discount), dtype=torch.float64)
    per_action = []
    for a in range(pomdp.num_actions):
        rows = benchmark.actions == a
        per_action.append(AlphaVectorSet(torch.cat([benchmark.vectors[rows], floor]),
                                         torch.full((int(rows.sum()) + 1,), a, dtype=torch.int64)))
    return value, per_action


def _log_marginal(log_probs: Tensor, groups: Tensor, members: list[int]) -> Tensor:
    """log P(group) of interactive log-beliefs (N, IS) by logsumexp over each group's states; (N, len(members))."""
    return torch.stack([torch.logsumexp(log_probs[:, groups == g], dim=-1) for g in members], dim=-1)


@dataclass(frozen=True)
class MarginalReport:
    """Divergence of decoded from exact marginals on held-out histories (section 3)."""

    kl_mean: float
    kl_max: float
    l1_mean: float


@torch.no_grad()
def marginal_reports(probe, dataset: ProbeDataset, problem: InteractiveModel) -> dict[str, MarginalReport]:
    """
    KL(exact marginal || decoded marginal) and L1 on the physical and the other agent's node marginals. A group with no
    reachable interactive state (a controller node the fold never visits) holds no mass in either belief and is left
    out, so it adds exactly 0 to both divergences.
    """
    log_decoded = probe.log_probs(dataset.latents).double()
    exact = dataset.posteriors.double()
    reports = {}
    for name, groups, num_groups in (("physical", problem.physical_state, problem.num_physical_states),
                                     ("other_node", problem.other_node, problem.other.num_nodes)):
        groups = groups.to(exact.device)
        members = [g for g in range(num_groups) if bool((groups == g).any())]
        log_q = _log_marginal(log_decoded, groups, members)
        p = torch.stack([exact[:, groups == g].sum(-1) for g in members], dim=-1)
        kl = (p * (p.clamp(min=1e-12).log() - log_q)).sum(-1)
        l1 = (p - log_q.exp()).abs().sum(-1)
        reports[name] = MarginalReport(float(kl.mean()), float(kl.max()), float(l1.mean()))
    return reports


@dataclass(frozen=True)
class ProbeControl:
    """
    The probes of the run's untrained filter (section 5), each on the histories of its trained counterpart.

    Attributes:
        probes: "<linear|mlp>/random" interactive-state probe reports on analyze_beliefs' random histories.
        marginals: as InteractiveAnalysis.marginals, on the same episodes as the trained filter's.
    """

    probes: dict[str, ProbeReport]
    marginals: dict[str, dict[str, MarginalReport]]


@dataclass(frozen=True)
class InteractiveAnalysis:
    """
    Everything an interactive run's analysis measures.

    Attributes:
        bracket: lower, upper and width of the benchmark at b0, with its source and digest.
        reference: analyze_beliefs against the reference sets (section 2): its "optimal" return row is the REFERENCE
            agent, its optimal_value_b0 is L(b0) and its solver_error_bound is the bracket width U - L.
        marginals: per probe ("<linear|mlp>/<random|agent>") the physical and other-node marginal reports (section 3).
        untrained: the same probes on the run's untrained filter (section 5).
    """

    bracket: dict
    reference: BeliefAnalysis
    marginals: dict[str, dict[str, MarginalReport]]
    untrained: ProbeControl


def _fitters(mlp_probe_steps: int):
    """The two probe families, in the order analyze_beliefs fits them."""
    return (("linear", fit_linear_probe), ("mlp", lambda d: fit_mlp_probe(d, steps=mlp_probe_steps)))


def _marginal_probes(problem: InteractiveModel, belief_filter: BeliefFilter,
                     episodes: dict[str, tuple[EpisodeBatch, EpisodeBatch]],
                     mlp_probe_steps: int) -> dict[str, dict[str, MarginalReport]]:
    """Section 3: per probe family and history source, a probe fitted on the first set and scored on the second."""
    data = {source: (build_probe_dataset(problem.pomdp, belief_filter, fit_episodes),
                     build_probe_dataset(problem.pomdp, belief_filter, eval_episodes))
            for source, (fit_episodes, eval_episodes) in episodes.items()}
    return {f"{kind}/{source}": marginal_reports(fit(fit_set), eval_set, problem)
            for kind, fit in _fitters(mlp_probe_steps) for source, (fit_set, eval_set) in data.items()}


def _random_state_probes(pomdp: FinitePOMDP, belief_filter: BeliefFilter, episode_length: int, num_episodes: int,
                         mlp_probe_steps: int, seed: int, device: torch.device) -> dict[str, ProbeReport]:
    """
    Section 5: the interactive-state probes on analyze_beliefs' random histories, replayed from its streams (1 and
    then 2, one uniform agent on stream 6, in the order analyze_beliefs plays them).
    """
    env = lambda stream: BatchedPOMDPEnv(pomdp, num_episodes, episode_length, stream_seed(seed, stream), device)  # noqa: E731
    uniform = UniformRandomAgent(pomdp.num_actions, num_episodes, stream_seed(seed, 6), device)
    fit_set = build_probe_dataset(pomdp, belief_filter, play_episodes(env(1), uniform)[0])
    eval_set = build_probe_dataset(pomdp, belief_filter, play_episodes(env(2), uniform)[0])
    return {f"{kind}/random": evaluate_probe(fit(fit_set), eval_set) for kind, fit in _fitters(mlp_probe_steps)}


def analyze_interactive(problem: InteractiveModel, belief_filter: BeliefFilter, initial_filter: BeliefFilter,
                        planner_agent: PlanningAgent, benchmark: InteractiveBenchmark, episode_length: int,
                        num_episodes: int, mlp_probe_steps: int, seed: int, device: torch.device) -> InteractiveAnalysis:
    """
    Sections 2, 3 and 5 on one trained agent.

    Args:
        problem: The fold the agent was trained on (its pomdp must match the benchmark's digest).
        belief_filter: The agent's acting filter.
        initial_filter: The run's untrained acting filter, the probe control (section 5; initial_acting_filter).
        planner_agent: The agent's greedy planner, batch size num_episodes.
        benchmark: The fold's certified bracket.
        episode_length, num_episodes, mlp_probe_steps, seed, device: as interpretability.analyze_beliefs; this module
            also uses streams 11-16 of `seed` (section 3) and replays streams 1, 2 and 6 (section 5).
    """
    pomdp = problem.pomdp
    value, per_action = reference_sets(benchmark, pomdp)
    reference = analyze_beliefs(pomdp, belief_filter, planner_agent, value, per_action, benchmark.upper - benchmark.lower,
                                episode_length, num_episodes, mlp_probe_steps, seed, device)
    env = lambda stream: BatchedPOMDPEnv(pomdp, num_episodes, episode_length, stream_seed(seed, stream), device)  # noqa: E731
    uniform = UniformRandomAgent(pomdp.num_actions, num_episodes, stream_seed(seed, 16), device)
    episodes = {"random": (play_episodes(env(11), uniform)[0], play_episodes(env(12), uniform)[0]),
                "agent": (play_episodes(env(13), planner_agent)[0], play_episodes(env(14), planner_agent)[0])}
    marginals = _marginal_probes(problem, belief_filter, episodes, mlp_probe_steps)
    untrained = ProbeControl(_random_state_probes(pomdp, initial_filter, episode_length, num_episodes, mlp_probe_steps,
                                                  seed, device),
                             _marginal_probes(problem, initial_filter, episodes, mlp_probe_steps))
    bracket = {"lower": benchmark.lower, "upper": benchmark.upper, "width": benchmark.upper - benchmark.lower,
               "source": benchmark.source, "digest": benchmark.digest}
    return InteractiveAnalysis(bracket, reference, marginals, untrained)


def print_interactive_summary(analysis: InteractiveAnalysis) -> None:
    """The console summary of analyze.py for an interactive run."""
    b = analysis.bracket
    print(f"certified bracket of V*(b0): [{b['lower']:.3f}, {b['upper']:.3f}] (width {b['width']:.3f}; {b['source']})")
    for name, estimate in analysis.reference.returns.items():
        label = "reference" if name == "optimal" else name           # section 2: the "optimal" row is the reference
        print(f"  {label:<16} discounted return {estimate.mean:8.3f} +- {estimate.stderr:.3f}")
    for name, probe in analysis.reference.probes.items():
        print(f"  probe {name:<14} interactive-state KL mean {probe.mean_kl:.5f}  L1 mean {probe.mean_l1:.4f}")
    for name, reports in analysis.marginals.items():
        print(f"  marginal probe {name:<12} " + "  ".join(f"{m}: KL {r.kl_mean:.5f} L1 {r.l1_mean:.4f}"
                                                         for m, r in reports.items()))
    # Section 2: bounds on the LOWER BOUND L, global and local form (interpretability/error_bounds.py, 4 and 5).
    for name, bound in analysis.reference.bounds.items():
        print(f"  bounds on L, {name:<12} mean L1 {bound.belief_l1_mean:.4f} | value error {bound.value_error_mean:.3f} "
              f"<= {bound.value_error_bound_local_mean:.3f} (local) <= {bound.value_error_bound_mean:.3f} (L_V "
              f"{bound.lipschitz_value:.1f}) | regret {bound.regret_mean:.4f} <= {bound.regret_bound_local_mean:.3f} "
              f"(local) <= {bound.regret_bound_mean:.3f} | policy form on L(b0) - V: {bound.policy_loss_bound_local_mean:.1f} "
              f"(local), {bound.policy_loss_bound_mean:.1f}")
    print("  control: the same probes on the run's untrained filter (section 5)")
    for name, probe in analysis.untrained.probes.items():
        print(f"  untrained probe {name:<12} interactive-state KL mean {probe.mean_kl:.5f}  L1 mean {probe.mean_l1:.4f}")
    for name, reports in analysis.untrained.marginals.items():
        print(f"  untrained marginal probe {name:<12} " + "  ".join(f"{m}: KL {r.kl_mean:.5f} L1 {r.l1_mean:.4f}"
                                                                   for m, r in reports.items()))


@dataclass(frozen=True)
class InteractiveAnalysisSettings:
    """
    Attributes:
        episodes: Episodes per probe data set and per return estimate.
        mlp_probe_steps: Adam steps of each MLP probe fit.
        seed: Base seed of the analysis streams (streams 1-6 analyze_beliefs, 7-8 the planner, 11-16 section 3).
    """

    episodes: int
    mlp_probe_steps: int
    seed: int


def initial_acting_filter(trained: TrainedRun, device: torch.device) -> BeliefFilter:
    """
    The run's acting filter before its first update (section 5): a TrainingRun built from the run's own config and
    seed over the already built domain (the fold is not rebuilt). Like every TrainingRun it re-seeds the global
    generator from the run's NETWORK_INIT stream, which the linear probes' initial weights then read (section 5).
    """
    fresh = TrainingRun(trained.setup, build_run_config(trained.cfg), device)
    fresh.acting_filter.eval()
    return fresh.acting_filter


def interactive_problem(env_cfg) -> InteractiveModel:
    """The fold of a run's env config, built exactly as experiments/runs.py itiger_setup builds it."""
    problem = nested_problem(build_multiagent_tiger(env_cfg.discount), Agent.I, env_cfg.level, TIGER_LEVEL_ZERO_NOISE,
                             env_cfg.bracket_epsilon, env_cfg.controller_epsilon, int(TigerAction.LISTEN),
                             tiger_growl_counters())
    return problem.interactive


def analyze_interactive_run(run_dir: Path, settings: InteractiveAnalysisSettings, benchmark_path: Path,
                            device: torch.device) -> InteractiveAnalysis:
    """Analyses an env=itiger run's best.pt against a benchmark file and writes <run>/analysis/report.json."""
    trained = load_trained_run(run_dir, device)
    cfg, run = trained.cfg, trained.run
    if cfg.env.name != "itiger":
        raise TypeError(f"analyze_interactive_run needs an itiger run, got {cfg.env.name}.")
    problem = interactive_problem(cfg.env)
    if fold_digest(problem.pomdp) != fold_digest(trained.setup.domain):
        raise ValueError("the rebuilt fold differs from the run's fold: the domain code changed since training.")
    benchmark = InteractiveBenchmark.load(benchmark_path)
    model = run.eval_agent.model
    planner = PlanningAgent(
        model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                dirichlet_epsilon=0.0, seed=stream_seed(settings.seed, 7),
                                observation_prune=cfg.mcts.observation_prune),
        settings.episodes, temperature=0.0, seed=stream_seed(settings.seed, 8), device=device)
    analysis = analyze_interactive(problem, run.acting_filter, initial_acting_filter(trained, device), planner, benchmark,
                                   cfg.env.max_steps, settings.episodes, settings.mlp_probe_steps, settings.seed, device)
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    report = asdict(analysis) | {"checkpoint_collection": trained.checkpoint_collection,
                                 "checkpoint_eval_return": trained.checkpoint_eval_return,
                                 "settings": asdict(settings), "benchmark_file": str(benchmark_path)}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return analysis
