# ABSOLUTE PATH: reports/2026-10-19_eval_protocol/ba_pomcp_itiger_diagnostics.py
# ==============================================================================
# POST HOC DIAGNOSTICS OF C-r ON INTERACTIVE TIGER: WHAT DID BA-POMCP LEARN, AND IS ITS DEFICIT THE SEARCH? (CPU; NOT REGISTERED)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Why (written after the C-r I-Tiger readout, read once on 2026-11-23; PROTOCOL.md, "C-r result, I-Tiger"):
#    - Both arms end about 30 per episode below the reference agent: main -14.10 and known_t -18.50, against a
#      certified bracket of [16.55, 19.67]. Always listening is worth -19.88 over the 100-step window, so both arms sit
#      between "never open" and a few points above it.
#    - Two numbers need an explanation. The upper reference (known_t, told T) is not above the main arm, and its
#      return falls during its own learning period (-5.06 to -13.83 per 2,000-step block) while main's rises.
#    - These are diagnostics written after the result they explain. They are reported as such, never as registered
#      claims, and they change no registered number.
#
# 2. `models`: What Each Run Learned (seconds; reads runs/baselines/ba_pomcp/itiger/<arm>/run<r>_agent.pkl):
#    - The posterior after the learning period is a set of 1,000 particles, each with its own Dirichlet counts. For
#      every run the script prints, as the mean over particles of the posterior-mean tables:
#        * the growl accuracy after LISTEN, P(growl names the tiger's side | arrival state), truth 0.85. A value under
#          0.5 is the MIRRORED model, which the Tiger readout found once (rewards are not observed, so a weakly
#          oriented prior can tip into the mirror image and lock in);
#        * the creak accuracy after LISTEN, P(creak names j's action | arrival state), truth 0.9;
#        * the mean total-variation distance of the LISTEN observation rows and of every transition row to the truth.
#      Rows are weighted by the evidence they received (posterior counts minus prior counts), so that rows the run
#      never visited do not dilute the picture; the evidence itself is printed.
#
# 3. `true_model`: BA-POMCP Planning With the True Tables (about one hour on 4 workers, about 1.1 GB each):
#    - If the deficit is the search, BA-POMCP with the TRUE model must lose about as much on the same episodes. The
#      true tables are given at point-mass confidence (counts 1e9 x the true probabilities), everything else as the
#      main arm: EXPECTED_MDP rollouts, 1,000 simulations, 1,000 particles, planning horizon 20. This is the check
#      that ba_pomcp_point_mass_check.py made on the canonical Tiger.
#    - The 256 paired episodes of simulator seed 11, in 8 chunks of 32 live agents (exact: a simulator row's random
#      numbers do not depend on the other rows' actions, tests/test_ba_pomcp.py); the agent of row i is seeded with
#      stream 10,000 + i of the base 20261123.
#    - Reported: the paired gap per episode to the reference agent of the benchmark file, mean +- 1.96 s.e. over the
#      256 episodes (one planner on one set of episodes, not an interval over runs), the share of episodes in which
#      it never opens a door (return equal to always listening), and the share with a wrong door (gap under -50).
#
# 4. Use (from the repository root):
#      CUDA_VISIBLE_DEVICES= uv run python reports/2026-10-19_eval_protocol/ba_pomcp_itiger_diagnostics.py models
#      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 uv run python \
#          reports/2026-10-19_eval_protocol/ba_pomcp_itiger_diagnostics.py true_model 4
# ==============================================================================

import math
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from ipomdp.baselines.ba_pomcp import BAPOMCPAgent, BAPOMCPBatch, DirichletPrior, Rollout
from ipomdp.domain import BatchedPOMDPEnv, FinitePOMDP, TigerAction, belief_update
from ipomdp.domain.multiagent_tiger import OBSERVATIONS, TRUE_CREAK
from ipomdp.experiments.interactive_analysis import InteractiveBenchmark, fold_digest, reference_sets
from ipomdp.training.seeding import stream_seed

CPU = torch.device("cpu")
OUT = Path("runs/baselines/ba_pomcp")
BENCHMARK = Path("runs/benchmarks/itiger_level1.pt")
FOLD_CACHE = OUT / "itiger_fold.pt"
BASE, RUNS, EPISODES, STEPS, SEED, CHUNK = 20261123, 10, 256, 100, 11, 32
LISTEN = int(TigerAction.LISTEN)
ORIENTED_TOTAL, ORIENTED_WEIGHT = 8.0, 5.0 / 14.0          # the registered observation prior (ba_pomcp_baseline.py)


def fold_model() -> FinitePOMDP:
    """The 26-state fold the campaign ran on, from its cache, checked against its digest."""
    payload = torch.load(FOLD_CACHE, map_location="cpu", weights_only=True)
    model = FinitePOMDP(payload["transition"], payload["observation"], payload["reward"], payload["initial_belief"],
                        float(payload["discount"]), tuple(payload["state_names"]), tuple(payload["action_names"]),
                        tuple(payload["observation_names"]))
    if fold_digest(model) != payload["digest"]:
        raise ValueError(f"{FOLD_CACHE} does not reproduce its digest.")
    return model


# -- section 2 ---------------------------------------------------------------------------------------------------------
def models() -> None:
    model = fold_model()
    true_t, true_o = model.transition.numpy(), model.observation.numpy()
    side = np.array([0 if name.split("|")[0] == "TIGER_LEFT" else 1 for name in model.state_names])
    previous = np.array([int(TigerAction[name.split("|")[2]]) for name in model.state_names])
    growl = np.array([int(g) for g, _ in OBSERVATIONS])
    creak = np.array([int(c) for _, c in OBSERVATIONS])
    names_growl = growl[None, :] == side[:, None]                                    # (S, Z): the growl names the side
    names_creak = creak[None, :] == np.array([int(TRUE_CREAK[TigerAction(a)]) for a in previous])[:, None]
    check = lambda table: (float((true_o[LISTEN] * table).sum(1).min()), float((true_o[LISTEN] * table).sum(1).max()))  # noqa: E731
    print(f"truth on the fold: growl accuracy after LISTEN {check(names_growl)}, creak accuracy {check(names_creak)}")
    prior_o = ORIENTED_TOTAL * (ORIENTED_WEIGHT * true_o + (1.0 - ORIENTED_WEIGHT) / true_o.shape[-1])
    print(f"the registered prior's own means: growl accuracy {float((prior_o[LISTEN] / prior_o[LISTEN].sum(1, keepdims=True) * names_growl).sum(1).mean()):.3f}, "
          f"creak accuracy {float((prior_o[LISTEN] / prior_o[LISTEN].sum(1, keepdims=True) * names_creak).sum(1).mean()):.3f}")
    for arm in ("main", "known_t"):
        print(f"\nitiger/{arm}: per run, means over the 1,000 particles (rows weighted by the evidence they received)")
        rows = []
        for run in range(RUNS):
            with open(OUT / "itiger" / arm / f"run{run}_agent.pkl", "rb") as handle:
                agent = pickle.load(handle)
            counts_o = agent.observation_counts[:, LISTEN]                           # (P, S, Z)
            mean_o = counts_o / counts_o.sum(-1, keepdims=True)
            evidence_o = np.maximum(counts_o.sum(-1) - prior_o[LISTEN].sum(-1)[None], 0.0)          # (P, S)
            weight_o = evidence_o / evidence_o.sum(-1, keepdims=True)
            growl_accuracy = ((mean_o * names_growl[None]).sum(-1) * weight_o).sum(-1)              # (P,)
            creak_accuracy = ((mean_o * names_creak[None]).sum(-1) * weight_o).sum(-1)
            tv_o = (0.5 * np.abs(mean_o - true_o[LISTEN][None]).sum(-1) * weight_o).sum(-1)
            counts_t = agent.transition_counts                                       # (P, A, S, S)
            mean_t = counts_t / counts_t.sum(-1, keepdims=True)
            prior_t = 1.0 * true_t.shape[-1] if arm == "main" else None              # Dirichlet 1 on every successor
            if prior_t is None:
                tv_t, evidence_t = float("nan"), float("nan")                        # T is known at high confidence
            else:
                evidence = np.maximum(counts_t.sum(-1) - prior_t, 0.0)                # (P, A, S)
                weight_t = evidence / evidence.sum((-1, -2), keepdims=True)
                tv_t = float((0.5 * np.abs(mean_t - true_t[None]).sum(-1) * weight_t).sum((-1, -2)).mean())
                evidence_t = float(evidence.sum((-1, -2)).mean())
            rows.append((growl_accuracy.mean(), growl_accuracy.min(), growl_accuracy.max(), creak_accuracy.mean(), tv_o.mean(),
                         evidence_o.sum(-1).mean(), tv_t, evidence_t))
            print(f"  run {run}: growl accuracy {rows[-1][0]:.3f} (particles {rows[-1][1]:.3f} to {rows[-1][2]:.3f}), creak accuracy "
                  f"{rows[-1][3]:.3f}, TV of the LISTEN rows {rows[-1][4]:.3f} on {rows[-1][5]:.0f} observations; TV of T "
                  f"{rows[-1][6]:.3f} on {rows[-1][7]:.0f} transitions")
        table = np.array(rows)
        print(f"  over runs: growl accuracy {table[:, 0].min():.3f} to {table[:, 0].max():.3f}; creak accuracy "
              f"{table[:, 3].min():.3f} to {table[:, 3].max():.3f}; mirrored runs (growl accuracy under 0.5): "
              f"{int((table[:, 0] < 0.5).sum())} of {RUNS}")


# -- section 3 ---------------------------------------------------------------------------------------------------------
class _Listen:
    def act(self, steps_left):
        return LISTEN

    def begin_episode(self):
        pass

    def update(self, action, observation):
        pass


def chunk_returns(chunk: int) -> np.ndarray:
    torch.set_num_threads(1)
    model = fold_model()
    point_mass = DirichletPrior.from_model(model, 1e9, 1e9)
    rows = range(chunk * CHUNK, (chunk + 1) * CHUNK)
    agents = [BAPOMCPAgent(model, point_mass, 1000, 1000, 20, Rollout.EXPECTED_MDP, stream_seed(BASE, 10_000 + row), 1000)
              if row in rows else _Listen() for row in range(EPISODES)]
    batch = BAPOMCPBatch(agents, STEPS)
    env = BatchedPOMDPEnv(model, EPISODES, STEPS, SEED, CPU)
    env.reset()
    batch.reset(None)
    returns = np.zeros(EPISODES)
    for t in range(STEPS):
        action = batch.act()
        out = env.step(action)
        batch.update(action, out.observation)
        returns += model.discount ** t * out.reward.double().numpy()
    return returns[list(rows)]


def reference_returns(model: FinitePOMDP) -> np.ndarray:
    value, _ = reference_sets(InteractiveBenchmark.load(BENCHMARK), model)
    env = BatchedPOMDPEnv(model, EPISODES, STEPS, SEED, CPU)
    env.reset()
    belief = model.initial_belief.expand(EPISODES, -1).clone()
    returns = np.zeros(EPISODES)
    for t in range(STEPS):
        action = value.greedy_action(belief)
        out = env.step(action)
        belief = belief_update(model, belief, action, out.observation)
        returns += model.discount ** t * out.reward.double().numpy()
    return returns


def true_model(workers: int) -> None:
    if not 1 <= workers <= 8:
        raise ValueError("at most 8 workers (the machine contract).")
    model = fold_model()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        planned = np.concatenate(list(pool.map(chunk_returns, range(EPISODES // CHUNK))))
    reference = reference_returns(model)
    gap = planned - reference
    listening = -(1 - model.discount ** STEPS) / (1 - model.discount)
    print(f"reference agent: {reference.mean():.3f} on the {EPISODES} paired episodes (C-r readout: 16.306); always listening: {listening:.3f}")
    print(f"BA-POMCP with the TRUE model at point-mass confidence (EXPECTED_MDP, 1000 simulations): return {planned.mean():.3f}; "
          f"gap {gap.mean():+.3f} +- {1.96 * gap.std(ddof=1) / math.sqrt(EPISODES):.3f}")
    print(f"episodes in which it never opens a door (return equal to always listening): {float((np.abs(planned - listening) < 1e-9).mean()):.3f}; "
          f"episodes with gap < -50: {float((gap < -50).mean()):.3f}; episodes at or above the reference agent: {float((gap >= 0).mean()):.3f}")
    print("C-r arms with learned models (read once): main -30.411 [-33.433, -27.388], known_t -34.809 [-39.712, -29.906] over 10 runs")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "models":
        models()
    elif len(sys.argv) == 3 and sys.argv[1] == "true_model":
        true_model(int(sys.argv[2]))
    else:
        raise SystemExit("usage: ba_pomcp_itiger_diagnostics.py models | true_model <workers>")
