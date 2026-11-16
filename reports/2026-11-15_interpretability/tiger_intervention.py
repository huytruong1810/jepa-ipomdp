# ABSOLUTE PATH: reports/2026-11-15_interpretability/tiger_intervention.py
# ==============================================================================
# DOES THE PLANNER USE THE BELIEF THE LATENT ENCODES? A WRITE-IN INTERVENTION ON TIGER
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (PROTOCOL.md C-q.2, registered 2026-11-15 before the first run):
#    - Probes show that the exact belief is DECODABLE from the latent. Decodable is not the same as
#      USED: a probe can read a feature the planner ignores (Elazar et al. 2021; Belinkov 2022). The
#      standard remedy is a causal intervention on the representation, scored on behaviour (Li et al.
#      2023 on Othello; Taufeeque et al. 2024 and Bush et al. 2025 on Sokoban planners).
#    - Tiger gives the cleanest version of that test: the exact optimal action is a known function of
#      the belief (the certified Q* alpha vectors), with thresholds between LISTEN and the two doors.
#      If the planner acts on the belief the latent encodes, then writing another belief into the
#      latent must move the planner's action to the optimal action of the WRITTEN belief.
#
# 2. Method (CPU; trained Tiger runs, best.pt, the acting filter and the greedy 50-simulation planner):
#    - Data: two sets of random-policy histories (analysis seed 20261115, streams 1 and 2; 128 episodes
#      of 100 steps each), replayed through the exact Bayes filter and the learned filter
#      (interpretability/belief_probe.py). Set 1 fits the maps, set 2 is intervened on.
#    - Belief-to-latent map mu: an MLP regression from the exact belief (as clipped log-odds) to the
#      latent, fitted on set 1 (the inverse of the usual probe; the "belief to latent" direction of
#      Mikulik et al. 2020). The forward MLP probe psi (latent to belief) is fitted on set 1 as well and
#      is used only to read the written belief back.
#    - Interventions on a latent z of set 2 with exact belief b, towards the belief b' of another row
#      of set 2 (a fixed permutation, so targets follow the natural belief distribution):
#        write-in   z' = z - mu(b) + mu(b')          moves only the belief-coded component; the
#                                                    residual z - mu(b) is kept
#        replace    z' = mu(b')                      the belief manifold alone
#        random     z' = z + |mu(b') - mu(b)| u      a random unit direction u of the same length
#        none       z' = z
#    - Readout: the planner's greedy action at z' (BeliefTreeSearch on the run's own search model, as in
#      analyze.py), against a*(b') = argmax_a Q*(b', a) and against a*(b). The FLIP SET is the rows with
#      a*(b') != a*(b): there the intervention must change the action.
#    - Reported per seed: the share of the flip set whose action becomes a*(b') (flip rate), the share
#      still at a*(b), agreement with a*(b') over all rows, the L1 error of psi(z') against b', and, for
#      the figure, the planner's action shares per bin of the written belief with the exact thresholds.
#    - No pass rule (descriptive). Expectation stated before the run: write-in and replace flip the
#      action on most of the flip set; the random direction does not.
#
# 3. Usage (CPU, a few minutes for ten seeds):
#      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 .venv/bin/python \
#          reports/2026-11-15_interpretability/tiger_intervention.py polyak 0,1,2,3,4,5,6,7,8,9
#    Writes runs/interpretability/tiger_intervention/<sweep>_seed<s>.npz and <sweep>.json.
# ==============================================================================

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from ipomdp.agents import UniformRandomAgent
from ipomdp.domain import BatchedPOMDPEnv
from ipomdp.domain.solver import action_value_functions, solve_infinite_horizon
from ipomdp.experiments.runs import load_trained_run
from ipomdp.interpretability import build_probe_dataset
from ipomdp.interpretability.belief_probe import fit_mlp_probe
from ipomdp.interpretability.error_bounds import q_values
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training.rollout import play_episodes
from ipomdp.training.seeding import stream_seed

SEED, EPISODES, ROWS, BINS = 20261115, 128, 2048, 20
OUT = Path("runs/interpretability/tiger_intervention")


def log_odds(beliefs: torch.Tensor) -> torch.Tensor:
    """Clipped log-odds of P(state 0), shape (N, 1) float32: the belief coordinate the map mu reads."""
    p = beliefs[:, :1].clamp(1e-6, 1 - 1e-6)
    return (p / (1 - p)).log().float()


def fit_belief_to_latent(beliefs: torch.Tensor, latents: torch.Tensor, seed: int) -> nn.Module:
    """mu: clipped log-odds -> latent, an MLP fitted by mean squared error on standardised latents (section 2)."""
    torch.manual_seed(seed)
    mean, std = latents.mean(0, keepdim=True), latents.std(0, keepdim=True).clamp_min(1e-6)
    net = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, latents.shape[1]))
    x, y = log_odds(beliefs) / 6.0, (latents - mean) / std
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(3000):
        optimizer.zero_grad(); loss = (net(x) - y).pow(2).mean(); loss.backward(); optimizer.step()

    class Map(nn.Module):
        @torch.no_grad()
        def forward(self, b: torch.Tensor) -> torch.Tensor:
            return net(log_odds(b) / 6.0) * std + mean

    fitted = Map(); fitted.residual = float(loss)
    return fitted


def main() -> None:
    sweep, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    device = torch.device("cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    summary, action_values, names = {}, None, None
    for seed in seeds:
        trained = load_trained_run(Path(f"runs/sweeps/{sweep}/seed{seed}"), device)
        pomdp, run, cfg = trained.setup.domain, trained.run, trained.cfg
        if action_values is None:                                  # one certified solve; the domain is the same for all seeds
            solution = solve_infinite_horizon(pomdp, tolerance=0.01, prune_epsilon=1e-6)
            action_values, names = action_value_functions(pomdp, solution.value_function, 1e-6), list(pomdp.action_names)
        env = lambda stream: BatchedPOMDPEnv(pomdp, EPISODES, cfg.env.max_steps, stream_seed(SEED, stream), device)  # noqa: E731
        uniform = UniformRandomAgent(pomdp.num_actions, EPISODES, stream_seed(SEED, 3), device)
        fit_set = build_probe_dataset(pomdp, run.acting_filter, play_episodes(env(1), uniform)[0])
        eval_set = build_probe_dataset(pomdp, run.acting_filter, play_episodes(env(2), uniform)[0])
        mu = fit_belief_to_latent(fit_set.posteriors, fit_set.latents, seed)
        psi = fit_mlp_probe(fit_set)

        generator = torch.Generator().manual_seed(stream_seed(SEED, 4) % (2 ** 31))
        rows = torch.randperm(eval_set.latents.shape[0], generator=generator)[:ROWS]
        z, b = eval_set.latents[rows], eval_set.posteriors[rows]
        target = b[torch.randperm(ROWS, generator=generator)]
        delta = mu(target) - mu(b)
        u = torch.randn(z.shape, generator=generator); u = u / u.norm(dim=1, keepdim=True)
        arms = {"none": z, "write_in": z + delta, "replace": mu(target), "random": z + delta.norm(dim=1, keepdim=True) * u}

        optimal_source = q_values(action_values, b).argmax(-1).numpy()
        optimal_target = q_values(action_values, target).argmax(-1).numpy()
        flip = optimal_source != optimal_target
        model = run.eval_agent.model
        result = {"rows": ROWS, "flip_set": int(flip.sum()), "mu_fit_mse": mu.residual, "arms": {}}
        saved = {"belief_source": b[:, 0].numpy(), "belief_target": target[:, 0].numpy(),
                 "optimal_source": optimal_source, "optimal_target": optimal_target}
        for arm, latents in arms.items():
            search = BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                      dirichlet_epsilon=0.0, seed=stream_seed(SEED, 5))
            action = search.search(latents.float(), temperature=0.0).argmax(-1).numpy()
            readback = float((psi(latents) - target).abs().sum(-1).mean())
            result["arms"][arm] = {"flip_rate": float((action[flip] == optimal_target[flip]).mean()),
                                   "stays_at_source": float((action[flip] == optimal_source[flip]).mean()),
                                   "agreement_with_target": float((action == optimal_target).mean()),
                                   "probe_l1_to_target": readback}
            saved[f"action_{arm}"] = action
        np.savez(OUT / f"{sweep}_seed{seed}.npz", **saved)
        summary[seed] = result
        print(f"seed {seed}: flip set {result['flip_set']} of {ROWS}; " + "; ".join(
            f"{arm} flip {r['flip_rate']:.3f} (agree {r['agreement_with_target']:.3f}, probe L1 {r['probe_l1_to_target']:.3f})"
            for arm, r in result["arms"].items()), flush=True)

    aggregate = {}
    for arm in ("none", "write_in", "replace", "random"):
        for key in ("flip_rate", "stays_at_source", "agreement_with_target", "probe_l1_to_target"):
            values = np.array([summary[s]["arms"][arm][key] for s in seeds])
            half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
            aggregate[f"{arm}.{key}"] = {"mean": float(values.mean()), "t95": half, "min": float(values.min()), "max": float(values.max())}
    for key, value in aggregate.items():
        print(f"{key:36s} {value['mean']:.4f} +- {value['t95']:.4f}  [{value['min']:.4f}, {value['max']:.4f}]")
    (OUT / f"{sweep}.json").write_text(json.dumps({"seeds": seeds, "actions": names, "per_seed": summary, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
