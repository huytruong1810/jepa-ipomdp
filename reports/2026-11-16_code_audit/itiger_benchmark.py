# ABSOLUTE PATH: reports/2026-11-16_code_audit/itiger_benchmark.py
# ==============================================================================
# THE LEVEL-1 INTERACTIVE TIGER BENCHMARK: A CERTIFIED BRACKET OF AGENT i's V*(b0)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. What This Script Certifies:
#    - Agent i's level-1 problem is the fold of j's certified level-0 controller (src/ipomdp/domain/interactive.py).
#      E1 compares the learned agent with the optimal agent of this problem, so it needs V*(b0), or a certified bracket
#      [L, U] of it.
#    - The exact solver is out of reach (26 to 80 states, six observations, gamma = 0.95). HSVI's lower bound converges
#      (about 16.5), but its sawtooth upper bound stalls (44.8 after about 1,000 trials; bounds.py, section 5).
# 2. The Ends of the Bracket:
#    - Upper end: an information relaxation with a finite reachable belief graph (bounds.revealed_upper_bound with
#      interactive.relaxation_reveal). j's previous action is revealed `delay` steps late, and j's node before the
#      step is revealed on i's door openings. Its corner and point values then seed HSVI on the true problem
#      (bounds.hsvi_budget), which tightens it further.
#    - Lower end: HSVI's lower bound, which its pointer controller provably attains.
#    - Both ends are proofs, whatever the trial budget.
# 3. Reported Beside the Bracket, Never as a Bound:
#    - The Monte-Carlo value of threshold policies on the exact belief ("listen until P(tiger side) >= theta"). This
#      is an estimate (20,000 episodes, 260 steps, seed 0) that shows where the lower end sits against simple policies.
# 4. Decision Rule (fixed before measuring; WORK_ORDERS.md, status of 2026-11-18):
#    - A bracket of 0.5 or less makes it the benchmark.
#    - Otherwise the bracket is reported as it is, E1 claims "within the certified bracket", and the audit session
#      decides how it enters the paper.
# 5. Usage (CPU only, about 1 GB, nice 10):
#      uv run reports/2026-11-16_code_audit/itiger_benchmark.py --delays 0,1,2 --seed-delay 1 --trials 60
#    - The defaults are conf/env/itiger.yaml's: j's controller extracted at 0.5 and certified at 0.05
#      (interactive.py, section 6), which gives the 26-state fold. Delay 2 needs about 4.5 GB and 30 min, and its
#      368,396 points are too many to seed the sawtooth, so HSVI is seeded with --seed-delay.
#    - --lower-only skips the relaxation and runs unseeded HSVI for --trials trials, without corner backups (hsvi()'s
#      trials). Its lower end tightens with the number of trials: a seeded run of 40 trials reached L = 16.27, and an
#      earlier untracked run of about 1,000 trials reached 16.54.
# 6. E1's Benchmark File (--artifact, with --lower-only):
#    - E1 must not recompute the bracket per seed, because the delay-2 relaxation takes 30 min and 5.8 GB. The file
#      holds L, U, the lower-bound vectors and their actions, and the fold's digest
#      (experiments/interactive_analysis.py). The analysis plays the reference agent (the best vector's action at the
#      exact belief, value >= L) and checks the digest against the run's fold.
#    - U is the smaller of this run's HSVI upper end and --upper, a certified value from a tracked log. Its source is
#      written into the file.
# ==============================================================================

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from ipomdp.domain.bounds import _HSVI, hsvi_budget, hsvi_vector_bracket, revealed_upper_bound
from ipomdp.domain.interactive import TIGER_LEVEL_ZERO_NOISE, nested_problem, relaxation_reveal
from ipomdp.domain.multiagent_tiger import Agent, build_multiagent_tiger
from ipomdp.domain.tiger import TigerAction
from ipomdp.experiments.interactive_analysis import InteractiveBenchmark, fold_digest


def threshold_values(problem, thetas, episodes=20_000, steps=260, seed=0):
    """Monte-Carlo value of threshold policies on the exact folded belief (section 3); estimates, not bounds."""
    m, it = problem.pomdp, problem.interactive
    T, O, R = m.transition.numpy(), m.observation.numpy(), m.reward.numpy()
    b0, gamma, (num_s, num_o) = m.initial_belief.numpy(), m.discount, (m.num_states, m.num_observations)
    left = (it.physical_state.numpy() == 0).astype(float)
    rng = np.random.default_rng(seed)
    out = {}
    for theta in thetas:
        s = rng.choice(num_s, size=episodes, p=b0)
        b = np.tile(b0, (episodes, 1))
        ret, discount = np.zeros(episodes), 1.0
        for _ in range(steps):
            p_left = b @ left
            a = np.where(p_left >= theta, int(TigerAction.OPEN_RIGHT),
                         np.where(p_left <= 1 - theta, int(TigerAction.OPEN_LEFT), int(TigerAction.LISTEN)))
            ret += discount * R[a, s]
            discount *= gamma
            s2 = np.minimum((rng.random(episodes)[:, None] > np.cumsum(T[a, s], axis=1)).sum(1), num_s - 1)
            o = np.minimum((rng.random(episodes)[:, None] > np.cumsum(O[a, s2], axis=1)).sum(1), num_o - 1)
            nb = np.einsum("es,est->et", b, T[a]) * O[a, :, o]
            b, s = nb / nb.sum(1, keepdims=True), s2
        out[theta] = (ret.mean(), ret.std(ddof=1) / np.sqrt(episodes))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bracket-epsilon", type=float, default=0.05)
    parser.add_argument("--controller-epsilon", type=float, default=0.5)
    parser.add_argument("--seed-delay", type=int, default=1)
    parser.add_argument("--delays", type=str, default="0,1")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--max-beliefs", type=int, default=1_000_000)
    parser.add_argument("--lower-only", action="store_true",
                        help="skip the relaxation; run unseeded HSVI for --trials trials (section 5)")
    parser.add_argument("--artifact", type=str, default=None,
                        help="with --lower-only: save E1's benchmark file here (section 6)")
    parser.add_argument("--upper", type=float, default=None, help="with --artifact: a certified U(b0) from a log")
    parser.add_argument("--upper-source", type=str, default=None, help="with --artifact: where --upper comes from")
    args = parser.parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)   # noqa: E731
    log(f"commit {commit}{' (uncommitted changes)' if dirty else ''}; args {vars(args)}")
    start = time.time()
    problem = nested_problem(build_multiagent_tiger(0.95), Agent.I, 1, TIGER_LEVEL_ZERO_NOISE,
                             args.bracket_epsilon, args.controller_epsilon, int(TigerAction.LISTEN))
    (zero,) = problem.below
    m = problem.pomdp
    log(f"level 0 (j): bracket [{zero.bracket.lower:.4f}, {zero.bracket.upper:.4f}], controller "
        f"{zero.controller.num_nodes} nodes, exact value {zero.controller_value:.4f}; level-1 fold: {m.num_states} "
        f"states, {m.num_observations} observations ({time.time() - start:.0f}s)")
    b0 = m.initial_belief.numpy()
    log(f"fast informed bound U(b0) = {float(b0 @ _HSVI(m).corner):.4f}")
    if args.lower_only:
        t = time.time()
        bracket = hsvi_vector_bracket(m, args.epsilon, args.trials)
        log(f"HSVI, unseeded, as hsvi(): L(b0) = {bracket.lower:.4f} (certified lower end), "
            f"U(b0) = {bracket.upper:.4f} after {bracket.trials} trials ({time.time() - t:.0f}s)")
        if args.artifact is not None:
            if args.upper is None or args.upper_source is None:
                raise ValueError("--artifact needs --upper and --upper-source.")
            upper = min(args.upper, bracket.upper)
            benchmark = InteractiveBenchmark(
                lower=bracket.lower, upper=upper, vectors=bracket.vectors, actions=bracket.actions,
                digest=fold_digest(m),
                source=(f"lower: unseeded HSVI, {bracket.trials} trials, epsilon {args.epsilon}, commit {commit}; "
                        f"upper: {args.upper_source}"))
            benchmark.save(Path(args.artifact))
            log(f"E1 benchmark saved to {args.artifact}: [{benchmark.lower:.4f}, {benchmark.upper:.4f}], "
                f"{benchmark.vectors.shape[0]} vectors, fold digest {benchmark.digest[:16]}")
        log(f"total {time.time() - start:.0f}s")
        return
    corners = torch.eye(m.num_states, dtype=torch.float64)
    best, seed = None, None
    for delay in [int(d) for d in args.delays.split(",")]:
        t = time.time()
        reveal, memory = relaxation_reveal(problem.interactive, delay, (int(TigerAction.OPEN_LEFT),
                                                                        int(TigerAction.OPEN_RIGHT)))
        relaxed = revealed_upper_bound(m, reveal, memory, extra_roots=corners, max_beliefs=args.max_beliefs)
        log(f"relaxation, delay {delay}: U(b0) = {relaxed.upper:.4f} (merge slack {relaxed.merge_slack:.1e}), "
            f"{relaxed.values.numel()} beliefs, corners [{float(relaxed.root_values[1:].min()):.2f}, "
            f"{float(relaxed.root_values[1:].max()):.2f}] ({time.time() - t:.0f}s)")
        if best is None or relaxed.upper < best.upper:
            best = relaxed
        if delay == args.seed_delay:
            seed = relaxed
    t = time.time()
    bracket = hsvi_budget(m, args.epsilon, args.trials, corners=seed.root_values[1:],
                          points=(seed.beliefs, seed.values))
    log(f"HSVI seeded with the delay-{args.seed_delay} relaxation: L(b0) = {bracket.lower:.4f}, "
        f"U(b0) = {bracket.upper:.4f} after {bracket.trials} trials ({time.time() - t:.0f}s)")
    lower, upper = bracket.lower, min(bracket.upper, best.upper)
    log(f"CERTIFIED BRACKET of V*(b0): [{lower:.4f}, {upper:.4f}], width {upper - lower:.4f}; decision rule: "
        f"{'benchmark (width <= 0.5)' if upper - lower <= 0.5 else 'report the bracket (width > 0.5)'}")
    for theta, (mean, se) in threshold_values(problem, (0.9, 0.93, 0.95, 0.96, 0.97)).items():
        log(f"estimate, threshold policy theta = {theta}: {mean:.3f} +- {se:.3f} (Monte Carlo, not a bound)")
    log(f"total {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
