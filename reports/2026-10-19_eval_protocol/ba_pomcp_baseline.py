# ABSOLUTE PATH: reports/2026-10-19_eval_protocol/ba_pomcp_baseline.py
# ==============================================================================
# C-r: BAYES-ADAPTIVE POMCP AGAINST OUR AGENT ON TIGER AND INTERACTIVE TIGER (CPU)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. What Is Compared (PROTOCOL.md, C-r; the user's rulings of 2026-11-19 and 2026-11-20):
#    - BA-POMCP (Katt, Oliehoek & Amato 2017; src/ipomdp/baselines/ba_pomcp.py) is told the states, actions,
#      observations and rewards, and starts from a Dirichlet prior over the dynamics. Our agent is told none of these.
#    - Tiger arms (10 independent runs each; 1000 particles, planning horizon 20 as the paper, capped by the steps left):
#        main      the paper's prior (T known at high confidence, LISTEN's growl 5 / 3, door observations uniform at high
#                  confidence), EXPECTED_MDP rollouts, 1000 simulations;
#        matched   the same at 50 simulations (our planner's budget);
#        uniform   the paper's prior with UNIFORM rollouts, 1000 simulations (the faithful default; it listens where the
#                  optimum opens, module header 4b of ba_pomcp.py);
#        weak      LISTEN's growl 1.25 / 0.75 (the same 62.5% mean at a quarter of the confidence), EXPECTED_MDP, 1000.
#    - Interactive Tiger arms (the 26-state fold of conf/env/itiger.yaml; BA-POMCP is told the interactive state space,
#      which reveals the size and structure of j's controller):
#        main      T uninformed (Dirichlet 1 on every successor), i's own O oriented (section 2), EXPECTED_MDP, 1000;
#        known_t   T known at high confidence as well, the same O, EXPECTED_MDP, 1000 (an upper reference).
#
# 2. The Oriented Observation Prior:
#    - counts = 8 x (w O + (1 - w) / Z) per row, w = 5 / 14: a mixture of the true observation row and the uniform row,
#      scaled to the paper's 8 counts. On Tiger's LISTEN row it is exactly the paper's 5 / 3 (0.625 = w 0.85 + (1 - w) 0.5);
#      on the fold's six (growl, creak) observations it orients growl and creak alike with the same confidence.
#
# 3. One Run = a Learning Period, Then the Paired Episodes:
#    - Learning: 200 episodes of 100 steps on the run's own simulator stream, counts carried over, hidden states reset
#      to b0 (ba_pomcp.py, section 3). Each learning episode's discounted return is a point of the learning curve.
#    - Evaluation: each of the 256 paired episodes (BatchedPOMDPEnv, simulator seed 11: the episodes of C-q.3 and of
#      reports/2026-11-16_code_audit/planner_budget.py) starts from a copy of the run's post-learning posterior and
#      keeps updating within its episode.
#
# 4. Chunked Evaluation Is Exact:
#    - A row of BatchedPOMDPEnv draws its random numbers independently of the other rows' actions (tested:
#      tests/test_ba_pomcp.py, test_simulator_rows_do_not_depend_on_other_rows_actions). The 256 episodes are therefore
#      played in 8 chunks of 32 live agents, the other rows listening, and every chunk row's trajectory is the one the
#      full batch gives. It keeps an Interactive Tiger worker near 1 GB instead of about 5 GB.
#
# 5. Jobs, Seeds and Resumability:
#    - A job is (domain, arm, run, part): part "learn" writes the run's learning returns and its post-learning agent;
#      parts "eval0".."eval7" play one chunk each. Every job writes one file atomically and is skipped when its file
#      exists, so an interrupted campaign continues where it stopped (`parallel`).
#    - Seeds (training/seeding.py, stream_seed of the domain's base): the learning simulator of run r is stream 100 + r
#      and its agent stream 200 + r, the same for every arm (the arms of run r learn on the same episodes); the copy
#      playing evaluation row i is stream 10,000 + 1,000 r + i.
#
# 6. Readout (`readout`, once per domain after every job; PROTOCOL.md, C-r):
#    - Tiger: per arm the per-run paired gap to the exact optimal agent (argmax Q* of the certified solution) on the 256
#      episodes, Student-t 95% over the 10 runs, per-run values beside it; BA-POMCP minus ours by Welch's two-sample t on
#      the per-run against the per-seed gaps, ours being the C-q.3 per-seed gaps at N = 50 (the same episodes and the same
#      optimal agent; reports/2026-11-16_code_audit/learned_planner_budget.log). Learning curves against environment
#      steps beside ours (the greedy evaluations in runs/sweeps/polyak/seed*/main.log, 6,400 steps per collection).
#    - Interactive Tiger: per arm the per-run paired gap to the reference agent of runs/benchmarks/itiger_level1.pt
#      (experiments/interactive_analysis.py: exact filter, best lower-bound vector's action, value >= L), Student-t over
#      the runs, and the arm's mean return against the certified bracket [L, U].
#
# 7. Use (from the repository root; CPU only, at most 8 workers at nice 19):
#      uv run python reports/2026-10-19_eval_protocol/ba_pomcp_baseline.py prepare            # I-Tiger fold, once
#      uv run python reports/2026-10-19_eval_protocol/ba_pomcp_baseline.py smoke  --domain itiger
#      uv run python reports/2026-10-19_eval_protocol/ba_pomcp_baseline.py parallel --domain tiger --workers 8
#      uv run python reports/2026-10-19_eval_protocol/ba_pomcp_baseline.py readout --domain tiger
# ==============================================================================

import argparse
import copy
import math
import os
import pickle
import re
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import stats

from ipomdp.baselines.ba_pomcp import BAPOMCPAgent, BAPOMCPBatch, DirichletPrior, Rollout
from ipomdp.domain import (BatchedPOMDPEnv, FinitePOMDP, TigerAction, belief_update, build_tiger_pomdp,
                           solve_infinite_horizon)
from ipomdp.experiments.interactive_analysis import (InteractiveBenchmark, fold_digest, interactive_problem,
                                                     reference_sets)
from ipomdp.training.seeding import stream_seed

CPU = torch.device("cpu")
OUT = Path("runs/baselines/ba_pomcp")
BENCHMARK = Path("runs/benchmarks/itiger_level1.pt")
FOLD_CACHE = OUT / "itiger_fold.pt"
BASE = {"tiger": 20261120, "itiger": 20261121}
RUNS, LEARN_EPISODES, STEPS = 10, 200, 100
EVAL_EPISODES, EVAL_SEED, CHUNK = 256, 11, 32
PARTICLES, HORIZON, REJECTION_CAP = 1000, 20, 1000
HIGH_CONFIDENCE = 1e6                              # counts per row of a "known" table
ORIENTED_TOTAL, ORIENTED_WEIGHT = 8.0, 5.0 / 14.0   # section 2
LISTEN = int(TigerAction.LISTEN)
OURS_LOG = Path("reports/2026-11-16_code_audit/learned_planner_budget.log")


@dataclass(frozen=True)
class Arm:
    prior: str
    rollout: Rollout
    simulations: int


ARMS = {"tiger": {"main": Arm("paper", Rollout.EXPECTED_MDP, 1000), "matched": Arm("paper", Rollout.EXPECTED_MDP, 50),
                  "uniform": Arm("paper", Rollout.UNIFORM, 1000), "weak": Arm("weak", Rollout.EXPECTED_MDP, 1000)},
        "itiger": {"main": Arm("uninformed_t", Rollout.EXPECTED_MDP, 1000),
                   "known_t": Arm("known_t", Rollout.EXPECTED_MDP, 1000)}}


# -- domains and priors (sections 1 and 2) ---------------------------------------------------------------------------
def oriented_observation_counts(observation: np.ndarray) -> np.ndarray:
    """Section 2: 8 x (w O + (1 - w) / Z) per row."""
    num_z = observation.shape[-1]
    return ORIENTED_TOTAL * (ORIENTED_WEIGHT * observation + (1.0 - ORIENTED_WEIGHT) / num_z)


def domain_model(domain: str) -> FinitePOMDP:
    if domain == "tiger":
        return build_tiger_pomdp()
    payload = torch.load(FOLD_CACHE, map_location="cpu", weights_only=True)
    model = FinitePOMDP(payload["transition"], payload["observation"], payload["reward"], payload["initial_belief"],
                        float(payload["discount"]), tuple(payload["state_names"]), tuple(payload["action_names"]),
                        tuple(payload["observation_names"]))
    if fold_digest(model) != payload["digest"]:
        raise ValueError(f"{FOLD_CACHE} does not reproduce its digest.")
    return model


def prior(domain: str, name: str, model: FinitePOMDP) -> DirichletPrior:
    known = DirichletPrior.from_model(model, HIGH_CONFIDENCE, HIGH_CONFIDENCE)
    if domain == "tiger":
        listen = {"paper": [[5.0, 3.0], [3.0, 5.0]], "weak": [[1.25, 0.75], [0.75, 1.25]]}[name]
        return known.with_observation_counts(LISTEN, np.array(listen))
    if name not in ("uninformed_t", "known_t"):
        raise ValueError(f"unknown Interactive Tiger prior {name}.")
    observation = oriented_observation_counts(model.observation.numpy())
    transition = np.ones_like(known.transition) if name == "uninformed_t" else known.transition
    return DirichletPrior(transition, observation)


def make_agent(domain: str, arm: Arm, model: FinitePOMDP, seed: int) -> BAPOMCPAgent:
    return BAPOMCPAgent(model, prior(domain, arm.prior, model), PARTICLES, arm.simulations, HORIZON, arm.rollout, seed,
                        REJECTION_CAP)


# -- jobs (sections 3-5) ------------------------------------------------------------------------------------------------
def job_path(domain: str, arm: str, run: int, part: str) -> Path:
    return OUT / domain / arm / f"run{run}_{part}.npz"


def agent_path(domain: str, arm: str, run: int) -> Path:
    return OUT / domain / arm / f"run{run}_agent.pkl"


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def learn(domain: str, arm_name: str, run: int) -> None:
    arm, model, base = ARMS[domain][arm_name], domain_model(domain), BASE[domain]
    agent = make_agent(domain, arm, model, stream_seed(base, 200 + run))
    env = BatchedPOMDPEnv(model, 1, STEPS, stream_seed(base, 100 + run), CPU)
    returns, start = np.zeros(LEARN_EPISODES), time.time()
    for episode in range(LEARN_EPISODES):
        env.reset()
        agent.begin_episode()
        for t in range(STEPS):
            action = agent.act(STEPS - t)
            out = env.step(torch.tensor([action]))
            agent.update(action, int(out.observation))
            returns[episode] += model.discount ** t * float(out.reward)
    path = agent_path(domain, arm_name, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as handle:
        pickle.dump(agent, handle)
    os.replace(tmp, path)
    _atomic_npz(job_path(domain, arm_name, run, "learn"), returns=returns, seconds=time.time() - start,
                peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


class _ListenAgent:
    """The rows outside a chunk (section 4): they listen and are not scored."""

    def act(self, steps_left: int) -> int:
        return LISTEN

    def begin_episode(self) -> None:
        pass

    def update(self, action: int, observation: int) -> None:
        pass


def evaluate(domain: str, arm_name: str, run: int, chunk: int) -> None:
    model, base = domain_model(domain), BASE[domain]
    with open(agent_path(domain, arm_name, run), "rb") as handle:
        learned = pickle.load(handle)
    rows = range(chunk * CHUNK, (chunk + 1) * CHUNK)
    agents = []
    for row in range(EVAL_EPISODES):
        if row in rows:
            agent = copy.deepcopy(learned)
            agent.rng = np.random.default_rng(stream_seed(base, 10_000 + 1_000 * run + row))
            agents.append(agent)
        else:
            agents.append(_ListenAgent())
    batch = BAPOMCPBatch(agents, STEPS)
    env = BatchedPOMDPEnv(model, EVAL_EPISODES, STEPS, EVAL_SEED, CPU)
    env.reset()
    batch.reset(None)
    returns, start = np.zeros(EVAL_EPISODES), time.time()
    for t in range(STEPS):
        action = batch.act()
        out = env.step(action)
        batch.update(action, out.observation)
        returns += model.discount ** t * out.reward.double().numpy()
    _atomic_npz(job_path(domain, arm_name, run, f"eval{chunk}"), returns=returns[list(rows)], rows=np.array(list(rows)),
                seconds=time.time() - start, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def run_job(job: tuple[str, str, int, str]) -> str:
    torch.set_num_threads(1)
    domain, arm, run, part = job
    if job_path(domain, arm, run, part).exists():
        return f"skip {job}"
    start = time.time()
    if part == "learn":
        learn(domain, arm, run)
    else:
        evaluate(domain, arm, run, int(part.removeprefix("eval")))
    return f"done {job} in {time.time() - start:.0f} s"


def jobs(domain: str, parts: str) -> list[tuple[str, str, int, str]]:
    names = ["learn"] if parts == "learn" else [f"eval{c}" for c in range(EVAL_EPISODES // CHUNK)]
    return [(domain, arm, run, part) for arm in ARMS[domain] for run in range(RUNS) for part in names]


def parallel(domain: str, workers: int) -> None:
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        for parts in ("learn", "eval"):
            for line in pool.map(run_job, [j for j in jobs(domain, parts) if not job_path(*j).exists()]):
                print(time.strftime("%H:%M:%S"), line, flush=True)


# -- preparation and smoke -----------------------------------------------------------------------------------------------
def prepare() -> None:
    """
    Builds the Interactive Tiger fold once (conf/env/itiger.yaml, as experiments/runs.py itiger_setup) and caches its
    tables with their digest. The benchmark file is checked against the digest here when it exists; the readout checks
    it in any case (experiments/interactive_analysis.py, reference_sets refuses another fold's file).
    """
    env_cfg = OmegaConf.load("conf/env/itiger.yaml").env
    pomdp = interactive_problem(env_cfg).pomdp
    digest = fold_digest(pomdp)
    if BENCHMARK.exists() and InteractiveBenchmark.load(BENCHMARK).digest != digest:
        raise ValueError(f"{BENCHMARK} was computed on another fold.")
    FOLD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"transition": pomdp.transition, "observation": pomdp.observation, "reward": pomdp.reward,
                "initial_belief": pomdp.initial_belief, "discount": pomdp.discount, "state_names": list(pomdp.state_names),
                "action_names": list(pomdp.action_names), "observation_names": list(pomdp.observation_names),
                "digest": digest}, FOLD_CACHE)
    checked = "matches the benchmark file" if BENCHMARK.exists() else "benchmark file not written yet; checked at readout"
    print(f"fold cached: {pomdp.num_states} states, {pomdp.num_observations} observations, digest {digest[:12]} "
          f"({checked})")


def smoke(domain: str, decisions: int) -> None:
    """Seconds per decision and peak memory of every arm, from b0 (not data: no episode is scored)."""
    torch.set_num_threads(1)
    model = domain_model(domain)
    for name, arm in ARMS[domain].items():
        agent = make_agent(domain, arm, model, 0)
        env = BatchedPOMDPEnv(model, 1, STEPS, 0, CPU)
        env.reset()
        start = time.time()
        for t in range(decisions):
            action = agent.act(STEPS - t)
            out = env.step(torch.tensor([action]))
            agent.update(action, int(out.observation))
        per = (time.time() - start) / decisions
        episode_hours = per * STEPS / 3600
        jobs_hours = RUNS * (LEARN_EPISODES + EVAL_EPISODES) * episode_hours
        print(f"{domain}/{name}: {per:.3f} s per decision; {episode_hours * 60:.1f} min per episode; about "
              f"{jobs_hours:.1f} CPU-hours for the arm; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB",
              flush=True)


# -- readout (section 6) ---------------------------------------------------------------------------------------------------
def _t_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean, half = float(values.mean()), float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - half, mean + half


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff, half = float(a.mean() - b.mean()), float(stats.t.ppf(0.975, df) * math.sqrt(va + vb))
    return diff, diff - half, diff + half, float(df)


def _arm_returns(domain: str, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Evaluation returns (runs, 256) in row order and learning returns (runs, 200); raises if a job is missing."""
    evaluation, learning = np.zeros((RUNS, EVAL_EPISODES)), np.zeros((RUNS, LEARN_EPISODES))
    for run in range(RUNS):
        learning[run] = np.load(job_path(domain, arm, run, "learn"))["returns"]
        for chunk in range(EVAL_EPISODES // CHUNK):
            data = np.load(job_path(domain, arm, run, f"eval{chunk}"))
            evaluation[run, data["rows"]] = data["returns"]
    return evaluation, learning


def _paired_returns(model: FinitePOMDP, agent) -> np.ndarray:
    env = BatchedPOMDPEnv(model, EVAL_EPISODES, STEPS, EVAL_SEED, CPU)
    env.reset()
    agent.reset(None)
    returns = np.zeros(EVAL_EPISODES)
    for t in range(STEPS):
        action = agent.act()
        out = env.step(action)
        agent.update(action, out.observation)
        returns += model.discount ** t * out.reward.double().numpy()
    return returns


class _ExactGreedy:
    """Exact Bayes filter and the greedy action of an alpha-vector set (the optimal or the reference agent)."""

    def __init__(self, model: FinitePOMDP, value_function):
        self.model, self.value_function, self.batch_size = model, value_function, EVAL_EPISODES

    def reset(self, initial_observation) -> None:
        self.state = self.model.initial_belief.expand(EVAL_EPISODES, -1).clone()

    def act(self):
        return self.value_function.greedy_action(self.state)

    def update(self, action, observation) -> None:
        self.state = belief_update(self.model, self.state, action, observation)


def _ours_tiger() -> np.ndarray:
    """C-q.3's per-seed paired gaps at N = 50 (the same episodes and optimal agent), from its committed log."""
    gaps = {}
    for line in OURS_LOG.read_text().splitlines():
        match = re.match(r"seed (\d+): gap \(learned - optimal\) per episode at N = \(([^)]*)\): \[([^]]*)\]", line)
        if match:
            budgets = [int(x) for x in match.group(2).split(",")]
            gaps[int(match.group(1))] = float(match.group(3).split(",")[budgets.index(50)])
    if sorted(gaps) != list(range(10)):
        raise ValueError(f"{OURS_LOG} does not hold the ten seeds' gaps.")
    return np.array([gaps[s] for s in range(10)])


def _ours_curve() -> dict[int, list[float]]:
    """Greedy evaluation returns of the ten polyak runs by environment steps (64 episodes x 100 steps per collection)."""
    curve: dict[int, list[float]] = {}
    for seed in range(10):
        for line in Path(f"runs/sweeps/polyak/seed{seed}/main.log").read_text().splitlines():
            match = re.search(r"collection (\d+): greedy eval discounted return (-?[\d.]+)", line)
            if match:
                curve.setdefault(int(match.group(1)) * 64 * STEPS, []).append(float(match.group(2)))
    return curve


def readout(domain: str) -> None:
    model = domain_model(domain)
    if domain == "tiger":
        certified = solve_infinite_horizon(model, tolerance=1e-3, prune_epsilon=1e-6)
        reference = _paired_returns(model, _ExactGreedy(model, certified.value_function))
        print(f"optimal agent on the {EVAL_EPISODES} paired episodes: {reference.mean():.3f} "
              f"(C-q.3 logged 22.375; certified error {certified.error_bound:.1e})")
        ours = _ours_tiger()
        mean, low, high = _t_interval(ours)
        print(f"ours (C-q.3, N = 50): gap {mean:+.3f} [{low:+.3f}, {high:+.3f}]; per seed {np.round(ours, 3).tolist()}")
    else:
        benchmark = InteractiveBenchmark.load(BENCHMARK)
        value, _ = reference_sets(benchmark, model)
        reference = _paired_returns(model, _ExactGreedy(model, value))
        print(f"certified bracket [{benchmark.lower:.4f}, {benchmark.upper:.4f}] ({benchmark.source}); reference agent "
              f"{reference.mean():.3f} on the {EVAL_EPISODES} paired episodes")
    for arm in ARMS[domain]:
        evaluation, learning = _arm_returns(domain, arm)
        gaps = (evaluation - reference[None]).mean(axis=1)
        mean, low, high = _t_interval(gaps)
        print(f"\n{domain}/{arm}: return {evaluation.mean():.3f}; gap to the {'optimal' if domain == 'tiger' else 'reference'} "
              f"agent {mean:+.3f} [{low:+.3f}, {high:+.3f}] (Student-t over {RUNS} runs); per run {np.round(gaps, 3).tolist()}")
        if domain == "tiger":
            diff, dlow, dhigh, df = _welch(gaps, ours)
            print(f"  BA-POMCP minus ours: {diff:+.3f} [{dlow:+.3f}, {dhigh:+.3f}] (Welch, {df:.1f} df)")
        else:
            print(f"  mean return against the bracket: {evaluation.mean():.3f} in [{benchmark.lower:.4f}, {benchmark.upper:.4f}]")
        blocks = learning.reshape(RUNS, LEARN_EPISODES // 20, 20).mean(axis=2).mean(axis=0)
        print("  learning curve (mean over runs, per 2,000 steps): " + ", ".join(f"{x:.2f}" for x in blocks))
    if domain == "tiger":
        curve = _ours_curve()
        print("\nours, greedy evaluation during training (mean over seeds): "
              + ", ".join(f"{steps}: {np.mean(v):.2f}" for steps, v in sorted(curve.items())))


def main() -> None:
    parser = argparse.ArgumentParser(description="C-r: BA-POMCP against our agent (module header).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    for name in ("smoke", "parallel", "readout"):
        p = sub.add_parser(name)
        p.add_argument("--domain", choices=sorted(ARMS), required=True)
        if name == "smoke":
            p.add_argument("--decisions", type=int, required=True)
        if name == "parallel":
            p.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "smoke":
        smoke(args.domain, args.decisions)
    elif args.command == "parallel":
        if not 1 <= args.workers <= 8:
            raise ValueError("at most 8 workers (the machine contract).")
        parallel(args.domain, args.workers)
    else:
        readout(args.domain)


if __name__ == "__main__":
    main()
