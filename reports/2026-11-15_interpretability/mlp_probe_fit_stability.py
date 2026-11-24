# ABSOLUTE PATH: reports/2026-11-15_interpretability/mlp_probe_fit_stability.py
# ==============================================================================
# C-q.5: IS THE MLP PROBE'S FIT STABLE? (POST HOC; STORED TIGER RUNS, CPU)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. What Prompted It:
#    - C-q.4 (tiger_local_bounds.py) replicated the stored Tiger analysis on the CPU, where the same analysis seed
#      draws other episodes. The linear probe's decoding error replicated (0.0160 against 0.0161). The MLP probe's
#      replicated on eight of the ten runs (within 0.001) and did not on two: seed 1 gave a mean L1 of 0.0176 against
#      a stored 0.0049, seed 5 gave 0.0267 against 0.0051. On both the LARGEST error is smaller than the stored one,
#      so the excess is spread over many histories and is not a handful of rare beliefs.
#    - The MLP probe is fitted by 3,000 full-batch Adam steps at a constant learning rate of 3e-3, and the last
#      iterate is taken (interpretability/belief_probe.py, fit_mlp_probe). Full-batch Adam at a constant rate is not
#      a descent method: its loss can rise for a stretch late in a fit. A last iterate that falls inside such a
#      stretch would give exactly this picture. The linear probe is fitted by L-BFGS with a line search, which
#      cannot do this.
#    - If so, the MLP probe's reported error is partly a draw of the fit, and every analysis that uses it inherits
#      the draw: the stored Tiger reports and the Interactive Tiger experiment that is still to be registered.
#
# 2. Method:
#    - Runs: the ten canonical JEPA runs runs/sweeps/polyak/seed0..9, best.pt, the acting copies, on the CPU.
#    - Data: the two data sets of the agent's own histories that analyze_beliefs builds (simulator streams 3 and 4
#      of analysis seed 12345, the greedy planner at the run's budget, 512 episodes), built here by the same calls in
#      the same order. They are stored, so that a later question about a probe needs no planner.
#    - Faithfulness, checked before anything is fitted for the trace: fit_mlp_probe itself, called as the analysis
#      calls it, must reproduce the mean L1 that C-q.4 stored for the run (to 1e-9), and the traced loop of this
#      script must end at bit-identical parameters. Either failure raises.
#    - Fits: for fit seeds 0 (the analysis's) to 4, the same network, optimizer and step count under two schedules:
#      "constant" (the code as it is) and "cosine" (the rate follows 3e-3 x (1 + cos(pi k / 3000)) / 2 and so ends
#      near zero). Every 25 steps the trace records the training loss and the mean L1 on the evaluation set.
#
# 3. Quantities (`readout`, once, after the ten runs):
#    - per run and schedule: the last iterate's mean L1 for each of the five fit seeds;
#    - per run, fit seed 0, constant rate: the last iterate against the median of the trace over the last 1,000
#      steps (a ratio far above 1 means that the last iterate sits in a rise of the loss, near 1 that the fit is on a
#      plateau);
#    - the number of the 50 fits per schedule whose last iterate is above twice the median of all 50;
#    - over the ten runs, mean with a Student-t 95% interval: the last iterate's mean L1 at fit seed 0 and as the
#      mean over the five fit seeds, under each schedule; the paired difference cosine minus constant.
#    - No pass rule. The reading stated in advance: if the outlying fits are late rises of the loss and the decayed
#      rate removes them at every fit seed, the instability is the constant-rate last iterate. A change of the fit is
#      then a decision about the instrument, because the stored reports were made with the constant rate; this script
#      changes nothing in src/.
#
# 4. Use (from the repository root; one thread per process):
#      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 uv run python \
#          reports/2026-11-15_interpretability/mlp_probe_fit_stability.py play --seeds 1,5,0,2,3
#      CUDA_VISIBLE_DEVICES= uv run python reports/2026-11-15_interpretability/mlp_probe_fit_stability.py readout \
#          > reports/2026-11-15_interpretability/mlp_probe_fit_stability.log
#    `--episodes` and `--out` exist for a rehearsal of the chain at a small size; the faithfulness check against
#    C-q.4 runs only at the registered size, and `readout` reads only the registered directory.
# ==============================================================================

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch import nn

from ipomdp.agents import PlanningAgent
from ipomdp.domain import BatchedPOMDPEnv, build_tiger_pomdp
from ipomdp.experiments import AnalysisSettings
from ipomdp.experiments.runs import load_trained_run
from ipomdp.interpretability.belief_probe import (BeliefProbe, ProbeDataset, _cross_entropy, _standardisation,
                                                  build_probe_dataset, evaluate_probe, fit_mlp_probe)
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training.rollout import play_episodes
from ipomdp.training.seeding import stream_seed

CPU = torch.device("cpu")
RUNS = Path("runs/sweeps/polyak")
OUT = Path("runs/interpretability/mlp_probe_fit")
CQ4 = Path("runs/interpretability/tiger_local_bounds")
SEEDS = tuple(range(10))
SETTINGS = AnalysisSettings()                      # 512 episodes, 3,000 probe steps, seed 12345
FIT_SEEDS = (0, 1, 2, 3, 4)
SCHEDULES = ("constant", "cosine")
HIDDEN, RATE = 64, 3e-3                            # fit_mlp_probe's own; the faithfulness check ties them to it
TRACE_EVERY = 25
TAIL_STEPS = 1000


def _datasets(seed: int, episodes: int) -> tuple[ProbeDataset, ProbeDataset]:
    """The agent's fit and evaluation sets, as analyze_beliefs builds them (streams 3 and 4, in this order)."""
    tiger = build_tiger_pomdp()
    trained = load_trained_run(RUNS / f"seed{seed}", CPU)
    run, cfg = trained.run, trained.cfg
    model = run.eval_agent.model
    planner = PlanningAgent(
        model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                dirichlet_epsilon=0.0, seed=stream_seed(SETTINGS.seed, 7)),
        episodes, temperature=0.0, seed=stream_seed(SETTINGS.seed, 8), device=CPU)
    env = lambda stream: BatchedPOMDPEnv(tiger, episodes, cfg.env.max_steps, stream_seed(SETTINGS.seed, stream), CPU)  # noqa: E731
    fit_set = build_probe_dataset(tiger, run.acting_filter, play_episodes(env(3), planner)[0])
    eval_set = build_probe_dataset(tiger, run.acting_filter, play_episodes(env(4), planner)[0])
    return fit_set, eval_set


def _traced_fit(fit_set: ProbeDataset, eval_set: ProbeDataset, fit_seed: int, schedule: str, steps: int):
    """fit_mlp_probe's loop, statement for statement, with the schedule and a trace added."""
    torch.manual_seed(fit_seed)
    mean, std = _standardisation(fit_set.latents)
    network = nn.Sequential(nn.Linear(fit_set.latents.shape[1], HIDDEN), nn.ELU(),
                            nn.Linear(HIDDEN, fit_set.posteriors.shape[1]))
    probe = BeliefProbe(network, mean, std).to(fit_set.latents.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=RATE)
    trace = []
    for step in range(steps):
        if schedule == "cosine":
            for group in optimizer.param_groups:
                group["lr"] = RATE * 0.5 * (1.0 + math.cos(math.pi * step / steps))
        optimizer.zero_grad()
        loss = _cross_entropy(probe, fit_set)
        loss.backward()
        optimizer.step()
        if (step + 1) % TRACE_EVERY == 0:
            # The loss is the one of the iterate before this step; the L1 is the one of the iterate after it.
            trace.append([step + 1, float(loss.detach()), evaluate_probe(probe, eval_set).mean_l1])
    return probe, trace


def play(seeds: list[int], episodes: int, out: Path) -> None:
    torch.set_num_threads(1)
    registered = episodes == SETTINGS.episodes
    if registered != (out == OUT):
        raise ValueError(f"{OUT} holds the registered size ({SETTINGS.episodes} episodes) and nothing else.")
    steps = SETTINGS.mlp_probe_steps
    commit = subprocess.run(["git", "log", "-1", "--format=%H"], check=True, capture_output=True, text=True).stdout.strip()
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        if (out / f"seed{seed}.json").exists():
            print(f"seed {seed}: done before", flush=True)
            continue
        start = time.time()
        data_path = out / f"seed{seed}_data.pt"
        if data_path.exists():
            stored = torch.load(data_path, weights_only=True)
            fit_set = ProbeDataset(stored["fit_latents"], stored["fit_posteriors"])
            eval_set = ProbeDataset(stored["eval_latents"], stored["eval_posteriors"])
        else:
            fit_set, eval_set = _datasets(seed, episodes)
            tmp = out / f"seed{seed}_data.pt.tmp"
            torch.save({"fit_latents": fit_set.latents, "fit_posteriors": fit_set.posteriors,
                        "eval_latents": eval_set.latents, "eval_posteriors": eval_set.posteriors}, tmp)
            os.replace(tmp, data_path)
        data_seconds = time.time() - start

        # Faithfulness (section 2): the analysis's own fit, then the traced loop against it.
        reference = fit_mlp_probe(fit_set, steps=steps)
        reference_l1 = evaluate_probe(reference, eval_set).mean_l1
        cq4_l1 = None
        if registered:
            cq4_l1 = json.loads((CQ4 / f"seed{seed}.json").read_text())["bounds"]["mlp/agent"]["belief_l1_mean"]
            if abs(reference_l1 - cq4_l1) > 1e-9:
                raise RuntimeError(f"seed {seed}: the rebuilt data give a mean L1 of {reference_l1!r}, C-q.4 stored "
                                   f"{cq4_l1!r}; the data sets are not the ones of the analysis.")
        fits = {schedule: {} for schedule in SCHEDULES}
        for schedule in SCHEDULES:
            for fit_seed in FIT_SEEDS:
                probe, trace = _traced_fit(fit_set, eval_set, fit_seed, schedule, steps)
                if schedule == "constant" and fit_seed == 0:
                    same = all(torch.equal(a, b) for a, b in zip(probe.state_dict().values(),
                                                                 reference.state_dict().values(), strict=True))
                    if not same:
                        raise RuntimeError(f"seed {seed}: the traced loop does not end at fit_mlp_probe's parameters.")
                report = evaluate_probe(probe, eval_set)
                fits[schedule][str(fit_seed)] = {"mean_l1": report.mean_l1, "max_l1": report.max_l1,
                                                 "mean_kl": report.mean_kl, "trace": trace}
        row = {"seed": seed, "commit": commit, "episodes": episodes, "steps": steps, "cq4_mean_l1": cq4_l1,
               "reference_mean_l1": reference_l1, "data_seconds": data_seconds, "seconds": time.time() - start,
               "fits": fits}
        tmp = out / f"seed{seed}.json.tmp"
        tmp.write_text(json.dumps(row))
        os.replace(tmp, out / f"seed{seed}.json")
        # Timing only: the values are read once, by `readout`.
        checked = "faithful to C-q.4" if registered else "rehearsal size, not checked against C-q.4"
        print(f"seed {seed}: {checked}; data {data_seconds:.0f} s, all {time.time() - start:.0f} s", flush=True)


def _interval(values: list[float]) -> str:
    x = np.asarray(values, dtype=np.float64)
    half = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return f"{x.mean():.4f} [{x.mean() - half:.4f}, {x.mean() + half:.4f}]"


def readout() -> None:
    missing = [seed for seed in SEEDS if not (OUT / f"seed{seed}.json").exists()]
    if missing:
        raise SystemExit(f"not read: seeds {missing} are not done yet")
    rows = [json.loads((OUT / f"seed{seed}.json").read_text()) for seed in SEEDS]
    print(f"C-q.5 (post hoc): the stability of the MLP probe's fit; ten JEPA runs of {RUNS}, the agent's own histories "
          f"of C-q.4 (CPU, analysis seed {SETTINGS.seed}, {SETTINGS.episodes} episodes), {SETTINGS.mlp_probe_steps} "
          f"full-batch Adam steps at {RATE}; run at commits {sorted({row['commit'][:7] for row in rows})}")
    print("every run reproduced C-q.4's mean L1 with fit_mlp_probe itself, and the traced loop ended at its parameters")
    last = {schedule: np.array([[row["fits"][schedule][str(k)]["mean_l1"] for k in FIT_SEEDS] for row in rows])
            for schedule in SCHEDULES}
    for schedule in SCHEDULES:
        print(f"\n  schedule {schedule}: the last iterate's mean L1 on the evaluation set, fit seeds {list(FIT_SEEDS)}")
        for row, values in zip(rows, last[schedule], strict=True):
            print(f"    run seed {row['seed']}: {np.round(values, 4).tolist()}")
        median = float(np.median(last[schedule]))
        print(f"    median of the 50 fits {median:.4f}; fits above twice the median: "
              f"{int((last[schedule] > 2 * median).sum())} of 50; largest {last[schedule].max():.4f}")
    print(f"\n  constant rate, fit seed 0 (the analysis's fit): the last iterate against the trace of the last "
          f"{TAIL_STEPS} steps (every {TRACE_EVERY} steps)")
    for row in rows:
        trace = np.array(row["fits"]["constant"]["0"]["trace"])
        tail = trace[trace[:, 0] > row["steps"] - TAIL_STEPS]
        print(f"    run seed {row['seed']}: last {tail[-1, 2]:.4f}; tail min {tail[:, 2].min():.4f}, median "
              f"{np.median(tail[:, 2]):.4f}, max {tail[:, 2].max():.4f}; last / median "
              f"{tail[-1, 2] / np.median(tail[:, 2]):.2f}; training loss last {tail[-1, 1]:.6f}, tail min "
              f"{tail[:, 1].min():.6f}, max {tail[:, 1].max():.6f}")
    print("\n  over the ten runs, mean [Student-t 95%]")
    for schedule in SCHEDULES:
        print(f"    {schedule:<9} fit seed 0: {_interval(last[schedule][:, 0].tolist())};  mean of the five fit seeds: "
              f"{_interval(last[schedule].mean(1).tolist())}")
    print(f"    cosine minus constant, paired per run: fit seed 0 "
          f"{_interval((last['cosine'][:, 0] - last['constant'][:, 0]).tolist())};  mean of the five fit seeds "
          f"{_interval((last['cosine'].mean(1) - last['constant'].mean(1)).tolist())}")
    spread = {schedule: last[schedule].std(axis=1, ddof=1) for schedule in SCHEDULES}
    print(f"    standard deviation over the five fit seeds, mean over runs: constant {spread['constant'].mean():.4f}, "
          f"cosine {spread['cosine'].mean():.4f}")
    print(f"\ncost: {np.mean([row['seconds'] for row in rows]) / 60:.1f} min per run on one CPU thread, of which "
          f"{np.mean([row['data_seconds'] for row in rows]) / 60:.1f} min for the two data sets")


def main() -> None:
    parser = argparse.ArgumentParser(description="C-q.5 (module header).")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("play")
    p.add_argument("--seeds", required=True, type=lambda text: [int(k) for k in text.split(",")])
    p.add_argument("--episodes", type=int, default=SETTINGS.episodes)
    p.add_argument("--out", type=Path, default=OUT)
    sub.add_parser("readout")
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("hide the GPU (CUDA_VISIBLE_DEVICES=): this analysis is registered on the CPU (section 2).")
    if args.command == "play":
        play(args.seeds, args.episodes, args.out)
    else:
        readout()


if __name__ == "__main__":
    main()
