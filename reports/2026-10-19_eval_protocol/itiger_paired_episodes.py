# ABSOLUTE PATH: reports/2026-10-19_eval_protocol/itiger_paired_episodes.py
# ==============================================================================
# OUR AGENT ON THE 256 PAIRED EPISODES OF C-r (INTERACTIVE TIGER, LEVEL 1; CPU)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. The Question:
#    - C-r played BA-POMCP on the level-1 Interactive Tiger fold (ba_pomcp_baseline.py): both arms end about 30 per
#      episode below the reference agent of runs/benchmarks/itiger_level1.pt on the 256 episodes of simulator seed 11
#      (PROTOCOL.md, "C-r result, I-Tiger"). The Tiger half of C-r compared BA-POMCP with our agent on the same
#      episodes, because C-q.3 had played them. The Interactive Tiger half could not: no run of ours existed.
#    - This script closes that half. Every trained run of a level-1 sweep plays the same 256 episodes with its
#      greedy learned planner, paired per episode with the same reference agent, so "BA-POMCP minus ours" is a
#      difference of two gaps measured on identical environment numbers.
#    - BA-POMCP is told the interactive state space (the size and structure of j's controller) and the rewards.
#      Our agent is told neither. Every line of the readout says so.
#
# 2. The Planner Is the One the Analysis Scores (as C-q.3 on the canonical Tiger):
#    - best.pt, the acting (Polyak) copies, loaded strictly (experiments/runs.py, load_trained_run); the run's own
#      search budget, c_puct and observation_prune; dirichlet_epsilon 0 and temperature 0, so nothing is sampled and
#      the two seeds of the planner and its agent (0, as in learned_planner_budget.py) select nothing.
#
# 3. The Episodes Are Drawn on the CPU:
#    - A simulator's generator lives on its device (domain/env.py), and C-r ran on the CPU. The same seed on the GPU
#      draws other episodes, so this script hides the GPU and loads GPU-trained runs on the CPU. One thread, as
#      C-q.3: a threaded matrix kernel may round differently, which could flip a near-tied decision.
#    - Checked, not assumed: the reference agent's return on the episodes must equal the value in C-r's committed
#      readout to its three printed decimals, and reference_sets refuses a run whose fold is not the benchmark's.
#
# 4. Jobs and Continuation:
#    - `play` writes one file per seed, <sweep>/seed<k>/paired_seed11.npz (per-episode returns of the learned planner
#      and of the reference agent, seconds, commit), atomically, and skips a seed whose file exists. A seed costs one
#      build of the fold (25 to 50 s, about 0.8 GB) and the play.
#    - `readout` raises unless every requested seed has its file. It prints the per-seed paired gaps, their mean with
#      a Student-t 95% interval over the seeds (the interval to cite), and BA-POMCP minus ours by Welch's two-sample
#      t against the per-run gaps of each C-r arm, read from C-r's committed readout
#      (ba_pomcp_itiger_readout.log; three decimals, which is far below the spread of either sample).
#    - It reads data: on registered seeds it is run once, after `play` has finished for all of them.
#
# 5. Use (from the repository root):
#      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 uv run python \
#          reports/2026-10-19_eval_protocol/itiger_paired_episodes.py play runs/sweeps/<name> --seeds 0,1,2,3,4,5,6,7,8,9
#      CUDA_VISIBLE_DEVICES= uv run python reports/2026-10-19_eval_protocol/itiger_paired_episodes.py \
#          readout runs/sweeps/<name> --seeds 0,1,2,3,4,5,6,7,8,9
# ==============================================================================

import argparse
import math
import os
import re
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from ipomdp.agents import PlanningAgent
from ipomdp.domain import BatchedPOMDPEnv, FinitePOMDP, belief_update
from ipomdp.experiments.interactive_analysis import InteractiveBenchmark, reference_sets
from ipomdp.experiments.runs import load_trained_run
from ipomdp.planning import BeliefTreeSearch

CPU = torch.device("cpu")
BENCHMARK = Path("runs/benchmarks/itiger_level1.pt")
C_R_READOUT = Path("reports/2026-10-19_eval_protocol/ba_pomcp_itiger_readout.log")
EPISODES, STEPS, SIMULATOR_SEED = 256, 100, 11
OUTPUT = f"paired_seed{SIMULATOR_SEED}.npz"


class ReferenceAgent:
    """Exact Bayes filter of the fold and the action of the best lower-bound vector (interactive_analysis.py, 2)."""

    def __init__(self, model: FinitePOMDP, value_function):
        self.model, self.value_function, self.state = model, value_function, None

    def reset(self, initial_observation) -> None:
        self.state = self.model.initial_belief.expand(EPISODES, -1).clone()

    def act(self):
        return self.value_function.greedy_action(self.state)

    def update(self, action, observation) -> None:
        self.state = belief_update(self.model, self.state, action, observation)


def play_episodes(model: FinitePOMDP, agent) -> np.ndarray:
    """Discounted returns of `agent` on the EPISODES paired episodes of SIMULATOR_SEED (section 3); (EPISODES,)."""
    env = BatchedPOMDPEnv(model, EPISODES, STEPS, SIMULATOR_SEED, CPU)
    env.reset()
    agent.reset(None)
    returns = torch.zeros(EPISODES, dtype=torch.float64)
    for t in range(STEPS):
        action = agent.act()
        out = env.step(action)
        agent.update(action, out.observation)
        returns += model.discount ** t * out.reward.double()
    return returns.numpy()


def c_r_readout() -> tuple[float, dict[str, np.ndarray]]:
    """The reference agent's return and each arm's per-run gaps, from C-r's committed readout (section 4)."""
    text = C_R_READOUT.read_text()
    reference = re.search(r"reference agent (-?[\d.]+) on the 256 paired episodes", text)
    arms = {name: np.array([float(x) for x in values.split(",")])
            for name, values in re.findall(r"^itiger/(\w+): .*per run \[([^]]*)\]", text, flags=re.MULTILINE)}
    if reference is None or sorted(arms) != ["known_t", "main"] or any(len(gaps) != 10 for gaps in arms.values()):
        raise ValueError(f"{C_R_READOUT} does not hold the reference return and the ten per-run gaps of both arms.")
    return float(reference.group(1)), arms


def check_reference(returns: np.ndarray) -> None:
    """Section 3: the same episodes and the same benchmark as C-r's readout."""
    logged, _ = c_r_readout()
    if round(float(returns.mean()), 3) != logged:
        raise ValueError(f"the reference agent scores {returns.mean():.3f} on these episodes, C-r's readout logged "
                         f"{logged:.3f}: other episodes, another benchmark file or another fold.")


def play(sweep: Path, seeds: list[int]) -> None:
    torch.set_num_threads(1)
    commit = subprocess.run(["git", "log", "-1", "--format=%H"], check=True, capture_output=True, text=True).stdout.strip()
    benchmark = InteractiveBenchmark.load(BENCHMARK)
    for seed in seeds:
        path = sweep / f"seed{seed}" / OUTPUT
        if path.exists():
            print(f"seed {seed}: {path} exists, skipped", flush=True)
            continue
        start = time.time()
        trained = load_trained_run(sweep / f"seed{seed}", CPU)
        model, mcts = trained.setup.domain, trained.cfg.mcts
        value, _ = reference_sets(benchmark, model)                     # refuses a fold that is not the benchmark's
        reference = play_episodes(model, ReferenceAgent(model, value))
        check_reference(reference)
        search_model = trained.run.eval_agent.model
        planner = BeliefTreeSearch(search_model, mcts.num_simulations, mcts.c_puct, mcts.dirichlet_alpha,
                                   dirichlet_epsilon=0.0, seed=0, observation_prune=mcts.observation_prune)
        learned = play_episodes(model, PlanningAgent(search_model, planner, EPISODES, temperature=0.0, seed=0, device=CPU))
        seconds, peak = time.time() - start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, learned=learned, reference=reference, simulations=mcts.num_simulations,
                 checkpoint_collection=trained.checkpoint_collection, seconds=seconds, peak_rss_mb=peak, commit=commit)
        os.replace(tmp, path)
        # Timing and memory only: no return is printed here, the seeds are read once by `readout` (section 4).
        print(f"seed {seed}: played in {seconds:.0f} s at N = {mcts.num_simulations}, peak RSS {peak:.0f} MB", flush=True)


def _t_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - half, mean + half


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff, half = float(a.mean() - b.mean()), float(stats.t.ppf(0.975, df) * math.sqrt(va + vb))
    return diff, diff - half, diff + half, float(df)


def readout(sweep: Path, seeds: list[int]) -> None:
    if len(seeds) < 2:
        raise ValueError("an interval over seeds needs at least two seeds.")
    logged, arms = c_r_readout()
    benchmark = InteractiveBenchmark.load(BENCHMARK)
    rows = [np.load(sweep / f"seed{seed}" / OUTPUT) for seed in seeds]          # raises on a missing seed
    for row in rows:
        check_reference(row["reference"])
        if not np.array_equal(row["reference"], rows[0]["reference"]):
            raise ValueError("the seeds were not played on the same episodes: their reference returns differ.")
    returns = np.stack([row["learned"] for row in rows])
    gaps = (returns - rows[0]["reference"][None]).mean(axis=1)
    mean, low, high = _t_interval(gaps)
    print(f"certified bracket [{benchmark.lower:.4f}, {benchmark.upper:.4f}]; reference agent {logged:.3f} on the "
          f"{EPISODES} paired episodes of simulator seed {SIMULATOR_SEED}, as in C-r's readout")
    print(f"ours ({sweep}, seeds {seeds}; told neither the states nor the rewards): return {returns.mean():.3f}; gap to "
          f"the reference agent {mean:+.3f} [{low:+.3f}, {high:+.3f}] (Student-t over {len(seeds)} seeds); per seed "
          f"{np.round(gaps, 3).tolist()}")
    for name, theirs in arms.items():
        diff, dlow, dhigh, df = _welch(theirs, gaps)
        print(f"  BA-POMCP {name} (told the interactive state space and the rewards) minus ours: {diff:+.3f} "
              f"[{dlow:+.3f}, {dhigh:+.3f}] (Welch, {df:.1f} df)")
    seconds = [float(row["seconds"]) for row in rows]
    print(f"cost of the play per seed: {np.mean(seconds):.0f} s on one CPU thread (fold build included), peak RSS "
          f"{max(float(row['peak_rss_mb']) for row in rows):.0f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Our agent on the paired episodes of C-r (module header).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("play", "readout"):
        p = sub.add_parser(name)
        p.add_argument("sweep", type=Path, help="Directory holding seed<k>/ run directories (a sweep).")
        p.add_argument("--seeds", required=True, type=lambda s: [int(k) for k in s.split(",")])
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("hide the GPU (CUDA_VISIBLE_DEVICES=): the paired episodes are drawn on the CPU (section 3).")
    (play if args.command == "play" else readout)(args.sweep, args.seeds)


if __name__ == "__main__":
    main()
