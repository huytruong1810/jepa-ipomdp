# ABSOLUTE PATH: reports/2026-11-15_interpretability/tiger_reanalysis.py
# ==============================================================================
# C-q.7: THE CANONICAL TIGER RE-ANALYSED WITH THE SELECTED-ITERATE MLP PROBE (POST HOC; STORED RUNS, CPU)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Why:
#    - Until 2026-11-24 the MLP probe returned the last iterate of its fit, which lands on a late rise of the loss
#      about once in 25 to 50 fits (PROTOCOL.md C-q.5). It now returns the lowest-training-loss iterate of the
#      fit's tail (interpretability/belief_probe.py, section 5; tested on fresh fits in C-q.6). The stored Tiger
#      reports were made with the old fit. This item analyses the same twenty runs with the new one, so that one
#      instrument serves every probe number of the paper.
#    - It never writes into a run directory: the stored reports stay what the ledger reads until the audit decides
#      otherwise. One JSON per run goes to runs/interpretability/tiger_reanalysis/<sweep>/, written atomically; a
#      run whose file exists is skipped.
#
# 2. Method:
#    - The ten JEPA runs runs/sweeps/polyak/seed0..9 and the ten decoder runs runs/sweeps/decoder/seed0..9, best.pt,
#      the acting (Polyak) copies, loaded strictly. interpretability.analyze_beliefs is called exactly as analyze_run
#      calls it (the certified solution at tolerance 0.01, one solve for all runs; the greedy planner at the run's
#      budget; 512 episodes; 3,000 probe steps; analysis seed 12345). This is tiger_local_bounds.py's call (C-q.4).
#    - On the CPU, one thread. The stored reports were made on the GPU, and a simulator's generator lives on its
#      device, so the same analysis seed draws other episodes here: against the stored reports this is a replication
#      on other episodes AND a change of the fit. The clean comparison of the two fits is with C-q.4, which analysed
#      the ten JEPA runs on these very CPU episodes with the old fit: there only the MLP rows can differ.
#    - Faithfulness, checked per JEPA run and raising on failure: the linear probe is untouched by the change, so
#      every field of its bound report must equal the one C-q.4 stored for the run. (The decoder runs have no CPU
#      analysis to check against.)
#
# 3. Quantities (`readout`, once, after the twenty runs), per sweep, mean with a Student-t 95% interval over its ten
#    seeds and the per-seed values:
#    - the probes' mean L1 and mean KL (linear and MLP, random and agent histories), beside the stored reports';
#    - for the MLP probe on the agent's histories, every field of the bound report, beside the stored reports' and,
#      for the JEPA runs, beside C-q.4's (the old fit on the same episodes);
#    - the returns of the four agents and the paired gaps to the optimal agent, beside the stored reports'.
#    - No pass rule. Which numbers the paper cites is the audit's decision.
#
# 4. Use (from the repository root; one thread per process):
#      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 uv run python \
#          reports/2026-11-15_interpretability/tiger_reanalysis.py play --sweep polyak --seeds 0,1,2,3,4,5,6,7,8,9
#      CUDA_VISIBLE_DEVICES= uv run python reports/2026-11-15_interpretability/tiger_reanalysis.py readout \
#          > reports/2026-11-15_interpretability/tiger_reanalysis.log
#    `--episodes` and `--out` exist for a rehearsal at a small size; the faithfulness check runs only at the
#    registered size, and `readout` reads only the registered directory.
#
# 5. C-q.8, the Control for the Decoded-Belief Agent (added 2026-11-25, after C-q.7 was read):
#    - C-q.7 found the decoded-belief agent (MLP probe) 0.96 under the optimal agent on these CPU episodes, where the
#      stored reports (GPU episodes, the last-iterate fit) have 0.20. Fit and episodes both differ, and C-q.4 kept no
#      returns, so the cause cannot be read off. `play --fit last` runs the same analysis with the OLD fit on the
#      same CPU episodes: it sets belief_probe.MLP_SELECTION_STEPS to 0, which leaves no candidate but the last
#      iterate. Output: runs/interpretability/tiger_reanalysis_lastfit/. JEPA runs only.
#    - Faithfulness, raising on failure: with the old fit BOTH bound reports (linear and MLP) must equal C-q.4's
#      field for field. That also proves that the switch restores the old fit exactly.
#    - `compare` (once, after its ten runs) prints, per seed and over the seeds, the decoded-belief agent's paired
#      gap to the optimal agent under each fit and their difference (selected minus last, the same episodes), with
#      the random-history probe's mean L1 under each fit. The optimal agent's and the learned planner's returns
#      must be equal under both fits (they do not read the probe); the script raises if they are not.
#        CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 uv run python \
#            reports/2026-11-15_interpretability/tiger_reanalysis.py play --fit last --sweep polyak --seeds 0,1,2,3,4
#        CUDA_VISIBLE_DEVICES= uv run python reports/2026-11-15_interpretability/tiger_reanalysis.py compare \
#            > reports/2026-11-15_interpretability/tiger_reanalysis_fit_control.log
#
# 6. C-q.9, the Episode Set as a Second Random Factor (added 2026-11-27, after C-q.8 was read):
#    - Every run of a sweep is analysed on the same 512 episodes, so an interval over the training seeds holds
#      nothing of the episode set. C-q.8 showed that this matters: the decoded-belief agent's paired gap to the
#      optimal agent is about -0.2 on one episode set and about -1.0 on another, with the same runs and the same fit.
#    - `play --analysis-seed S` runs the analysis of section 2 (the selected-iterate fit) on the ten JEPA runs with
#      another analysis seed: other episodes for the probes' data sets and for the agents' returns. Output:
#      runs/interpretability/tiger_reanalysis_seeds/<S>/polyak/. The registered seeds are ANALYSIS_SEEDS; with
#      C-q.7's own seed 12345 the table has ten runs by five episode sets.
#    - `seeds` (once, after the forty runs) prints for each quantity the ten-by-five table's margins and a crossed
#      analysis: run and episode set as two random factors without replication. With mean squares MS_run, MS_set
#      and MS_res, the variance of the grand mean is estimated by (MS_run + MS_set - MS_res) / (R S), its degrees
#      of freedom by Satterthwaite's formula, and the components by (MS_run - MS_res) / S, (MS_set - MS_res) / R and
#      MS_res. The interval over the training seeds for one episode set, which is what the stored reports give, is
#      printed beside it for each of the five sets.
#        CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 19 uv run python \
#            reports/2026-11-15_interpretability/tiger_reanalysis.py play --sweep polyak --analysis-seed 20261127 --seeds 0,1,2,3,4
#        CUDA_VISIBLE_DEVICES= uv run python reports/2026-11-15_interpretability/tiger_reanalysis.py seeds \
#            > reports/2026-11-15_interpretability/tiger_reanalysis_episode_sets.log
# ==============================================================================

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from ipomdp.agents import PlanningAgent
from ipomdp.domain import action_value_functions, build_tiger_pomdp, solve_infinite_horizon
from ipomdp.experiments import AnalysisSettings
from ipomdp.experiments.runs import load_trained_run
from ipomdp.interpretability import analyze_beliefs, belief_probe
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training.seeding import stream_seed

CPU = torch.device("cpu")
SWEEPS = {"polyak": Path("runs/sweeps/polyak"), "decoder": Path("runs/sweeps/decoder")}
OUT = Path("runs/interpretability/tiger_reanalysis")
OUT_LAST = Path("runs/interpretability/tiger_reanalysis_lastfit")   # section 5: the old fit on the same episodes
OUT_SEEDS = Path("runs/interpretability/tiger_reanalysis_seeds")    # section 6: other analysis seeds, one folder each
ANALYSIS_SEEDS = (20261127, 20261128, 20261129, 20261130)           # section 6: registered before their first run
CQ4 = Path("runs/interpretability/tiger_local_bounds")          # the old fit on the JEPA runs' CPU episodes
SEEDS = tuple(range(10))
SETTINGS = AnalysisSettings()                      # 512 episodes, tolerance 0.01, 3,000 probe steps, seed 12345
PROBES = ("linear/random", "linear/agent", "mlp/random", "mlp/agent")
AGENTS = ("optimal", "learned_planner", "decoded_mlp", "decoded_linear")


def play(sweep: str, seeds: list[int], episodes: int, out: Path | None, fit: str, analysis_seed: int) -> None:
    torch.set_num_threads(1)
    home = OUT if fit == "selected" else OUT_LAST
    if analysis_seed != SETTINGS.seed:                           # section 6
        if fit != "selected" or sweep != "polyak" or analysis_seed not in ANALYSIS_SEEDS:
            raise ValueError(f"other analysis seeds are registered for the JEPA runs with the selected-iterate fit, "
                             f"and only {ANALYSIS_SEEDS}.")
        home = OUT_SEEDS / str(analysis_seed)
    out = home if out is None else out
    registered = episodes == SETTINGS.episodes
    if registered != (out == home):
        raise ValueError(f"{home} holds the registered size ({SETTINGS.episodes} episodes) and nothing else.")
    if fit == "last":                                            # section 5
        if sweep != "polyak":
            raise ValueError("the control with the old fit is defined for the JEPA runs (C-q.4 is its reference).")
        belief_probe.MLP_SELECTION_STEPS = 0                     # no candidate but the last iterate: the fit before 8321456
    commit = subprocess.run(["git", "log", "-1", "--format=%H"], check=True, capture_output=True, text=True).stdout.strip()
    todo = [seed for seed in seeds if not (out / sweep / f"seed{seed}.json").exists()]
    if not todo:
        print("every requested seed is done", flush=True)
        return
    tiger = build_tiger_pomdp()
    solution = solve_infinite_horizon(tiger, tolerance=SETTINGS.solver_tolerance, prune_epsilon=1e-6)
    action_values = action_value_functions(tiger, solution.value_function, 1e-6)
    (out / sweep).mkdir(parents=True, exist_ok=True)
    for seed in todo:
        start = time.time()
        trained = load_trained_run(SWEEPS[sweep] / f"seed{seed}", CPU)
        stored = json.loads((SWEEPS[sweep] / f"seed{seed}" / "analysis" / "report.json").read_text())
        if stored["settings"] != asdict(SETTINGS):
            raise ValueError(f"{sweep} seed {seed} was analysed with {stored['settings']}, not {asdict(SETTINGS)}.")
        run, cfg = trained.run, trained.cfg
        model = run.eval_agent.model
        # As experiments/runs.py analyze_run builds the planner and calls the analysis.
        planner = PlanningAgent(
            model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                    dirichlet_epsilon=0.0, seed=stream_seed(analysis_seed, 7)),
            episodes, temperature=0.0, seed=stream_seed(analysis_seed, 8), device=CPU)
        analysis = asdict(analyze_beliefs(tiger, run.acting_filter, planner, solution.value_function, action_values,
                                          solution.error_bound, cfg.env.max_steps, episodes,
                                          SETTINGS.mlp_probe_steps, analysis_seed, CPU))
        old_fit = None
        if registered and sweep == "polyak" and analysis_seed == SETTINGS.seed:     # section 2, faithfulness
            cq4 = json.loads((CQ4 / f"seed{seed}.json").read_text())["bounds"]
            if analysis["bounds"]["linear/agent"] != cq4["linear/agent"]:
                raise RuntimeError(f"polyak seed {seed}: the linear probe's bound report differs from the one C-q.4 "
                                   "stored; these are not C-q.4's episodes or not its analysis.")
            if fit == "last" and analysis["bounds"]["mlp/agent"] != cq4["mlp/agent"]:
                raise RuntimeError(f"polyak seed {seed}: with the old fit the MLP probe's bound report differs from "
                                   "the one C-q.4 stored; the switch does not restore the old fit.")
            old_fit = cq4["mlp/agent"]
        for probe in analysis["probes"].values():                # megabytes of per-posterior rows that no readout uses
            probe["by_posterior"] = None
        row = {"sweep": sweep, "seed": seed, "fit": fit, "analysis_seed": analysis_seed, "commit": commit, "episodes": episodes,
               "seconds": time.time() - start,
               "analysis": analysis,
               "stored": {"returns": stored["returns"], "bounds": stored["bounds"],
                          "probes": {name: {k: v for k, v in report.items() if k != "by_posterior"}
                                     for name, report in stored["probes"].items()}},
               "old_fit_same_episodes": old_fit}
        tmp = out / sweep / f"seed{seed}.json.tmp"
        tmp.write_text(json.dumps(row))
        os.replace(tmp, out / sweep / f"seed{seed}.json")
        # Timing only: the values are read once, by `readout`.
        checked = "no CPU reference to check against" if old_fit is None else (
            "linear rows equal to C-q.4" if fit == "selected" else "linear and MLP rows equal to C-q.4")
        print(f"{sweep} seed {seed} ({fit} fit, analysis seed {analysis_seed}): {checked}; analysed in "
              f"{time.time() - start:.0f} s", flush=True)


def _interval(values: list[float], digits: int) -> str:
    x = np.asarray(values, dtype=np.float64)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return f"{x.mean():.{digits}f} [{x.mean() - half:.{digits}f}, {x.mean() + half:.{digits}f}]"


def _line(label: str, values: list[float], digits: int = 4) -> str:
    return f"    {label:<58} {_interval(values, digits)}  per seed {np.round(values, digits).tolist()}"


def readout() -> None:
    missing = [f"{sweep}/seed{seed}" for sweep in SWEEPS for seed in SEEDS if not (OUT / sweep / f"seed{seed}.json").exists()]
    if missing:
        raise SystemExit(f"not read: {missing} are not analysed yet")
    print(f"C-q.7 (post hoc): the canonical Tiger re-analysed with the selected-iterate MLP probe; CPU, one thread, "
          f"analysis seed {SETTINGS.seed}, {SETTINGS.episodes} episodes")
    print("mean [Student-t 95% over the ten seeds]. 'stored' is the run's own report (GPU, other episodes of the same "
          "seed, the last-iterate fit); 'old fit' is C-q.4 (these CPU episodes, the last-iterate fit; JEPA runs only)")
    for sweep in SWEEPS:
        rows = [json.loads((OUT / sweep / f"seed{seed}.json").read_text()) for seed in SEEDS]
        print(f"\n=== {SWEEPS[sweep]} (analysed at commits {sorted({row['commit'][:7] for row in rows})})")
        print("  probes")
        for probe in PROBES:
            for key in ("mean_l1", "mean_kl"):
                digits = 4 if key == "mean_l1" else 5
                print(_line(f"{probe} {key}", [row["analysis"]["probes"][probe][key] for row in rows], digits))
                print(_line(f"  stored {probe} {key}", [row["stored"]["probes"][probe][key] for row in rows], digits))
        print("  the MLP probe's bound report on the agent's histories")
        for key in rows[0]["stored"]["bounds"]["mlp/agent"]:
            print(_line(key, [row["analysis"]["bounds"]["mlp/agent"][key] for row in rows]))
            print(_line(f"  stored {key}", [row["stored"]["bounds"]["mlp/agent"][key] for row in rows]))
            if rows[0]["old_fit_same_episodes"] is not None:
                print(_line(f"  old fit {key}", [row["old_fit_same_episodes"][key] for row in rows]))
        for key in ("active_half_span_mean", "value_error_bound_local_mean", "regret_bound_local_mean",
                    "policy_loss_bound_local_mean"):                 # the local form: not in the stored reports
            print(_line(key, [row["analysis"]["bounds"]["mlp/agent"][key] for row in rows]))
            if rows[0]["old_fit_same_episodes"] is not None:
                print(_line(f"  old fit {key}", [row["old_fit_same_episodes"][key] for row in rows]))
        print("  returns")
        for agent in AGENTS:
            print(_line(agent, [row["analysis"]["returns"][agent]["mean"] for row in rows], 3))
            print(_line(f"  stored {agent}", [row["stored"]["returns"][agent]["mean"] for row in rows], 3))
        for agent in AGENTS[1:]:
            gap = lambda source: [row[source]["returns"][agent]["mean"] - row[source]["returns"]["optimal"]["mean"]  # noqa: E731
                                  for row in rows]
            print(_line(f"{agent} - optimal (paired)", gap("analysis"), 3))
            print(_line(f"  stored {agent} - optimal (paired)", gap("stored"), 3))
        print(f"  cost: {np.mean([row['seconds'] for row in rows]) / 60:.1f} min per run on one CPU thread")


def compare() -> None:
    """Section 5 (C-q.8): the decoded-belief agent under the two fits on the same episodes, JEPA runs."""
    missing = [f"{home}/polyak/seed{seed}" for home in (OUT, OUT_LAST) for seed in SEEDS
               if not (home / "polyak" / f"seed{seed}.json").exists()]
    if missing:
        raise SystemExit(f"not read: {missing} are not analysed yet")
    new = [json.loads((OUT / "polyak" / f"seed{seed}.json").read_text()) for seed in SEEDS]
    old = [json.loads((OUT_LAST / "polyak" / f"seed{seed}.json").read_text()) for seed in SEEDS]
    for a, b in zip(new, old):
        for agent in ("optimal", "learned_planner"):
            if a["analysis"]["returns"][agent] != b["analysis"]["returns"][agent]:
                raise RuntimeError(f"seed {a['seed']}: the {agent} agent's return differs between the two fits; the "
                                   "two analyses did not play the same episodes.")
    print("C-q.8 (post hoc): the decoded-belief agent under the selected-iterate fit and under the last-iterate fit, "
          f"the ten JEPA runs on the same CPU episodes (analysis seed {SETTINGS.seed}, {SETTINGS.episodes} episodes); "
          "with the old fit every run reproduced C-q.4's bound reports, and the optimal agent's and the learned "
          "planner's returns are equal under both fits")
    gap = lambda rows, agent: [row["analysis"]["returns"][agent]["mean"] - row["analysis"]["returns"]["optimal"]["mean"]  # noqa: E731
                               for row in rows]
    for agent in ("decoded_mlp", "decoded_linear"):
        print(f"\n  {agent} - optimal (paired within the run)")
        print(_line("selected-iterate fit", gap(new, agent), 3))
        print(_line("last-iterate fit", gap(old, agent), 3))
        print(_line("selected minus last", [x - y for x, y in zip(gap(new, agent), gap(old, agent))], 3))
        print(_line("stored reports (GPU episodes, last-iterate fit)",
                    [row["stored"]["returns"][agent]["mean"] - row["stored"]["returns"]["optimal"]["mean"] for row in new], 3))
    print("\n  the MLP probe on random histories (the probe the decoded-belief agent acts on), mean L1")
    print(_line("selected-iterate fit", [row["analysis"]["probes"]["mlp/random"]["mean_l1"] for row in new]))
    print(_line("last-iterate fit", [row["analysis"]["probes"]["mlp/random"]["mean_l1"] for row in old]))
    print(f"\n  cost of the control: {np.mean([row['seconds'] for row in old]) / 60:.1f} min per run on one CPU thread")


def _crossed(table: np.ndarray) -> dict[str, float]:
    """Section 6: runs (rows) by episode sets (columns) as two crossed random factors without replication."""
    runs, sets = table.shape
    grand, row_means, column_means = table.mean(), table.mean(axis=1), table.mean(axis=0)
    ms_run = sets * ((row_means - grand) ** 2).sum() / (runs - 1)
    ms_set = runs * ((column_means - grand) ** 2).sum() / (sets - 1)
    residual = table - row_means[:, None] - column_means[None, :] + grand
    df_res = (runs - 1) * (sets - 1)
    ms_res = (residual ** 2).sum() / df_res
    variance = (ms_run + ms_set - ms_res) / (runs * sets)        # of the grand mean; may be estimated at or below 0
    out = {"mean": float(grand), "sd_run": math.sqrt(max((ms_run - ms_res) / sets, 0.0)),
           "sd_set": math.sqrt(max((ms_set - ms_res) / runs, 0.0)), "sd_res": math.sqrt(ms_res),
           "half": float("nan"), "df": float("nan")}
    if variance > 0:
        df = (ms_run + ms_set - ms_res) ** 2 / (ms_run ** 2 / (runs - 1) + ms_set ** 2 / (sets - 1) + ms_res ** 2 / df_res)
        out["half"], out["df"] = float(stats.t.ppf(0.975, df) * math.sqrt(variance)), float(df)
    return out


def seeds_readout() -> None:
    """Section 6 (C-q.9): ten JEPA runs by five episode sets."""
    homes = {SETTINGS.seed: OUT, **{seed: OUT_SEEDS / str(seed) for seed in ANALYSIS_SEEDS}}
    missing = [f"{home}/polyak/seed{run}" for home in homes.values() for run in SEEDS
               if not (home / "polyak" / f"seed{run}.json").exists()]
    if missing:
        raise SystemExit(f"not read: {missing} are not analysed yet")
    rows = {seed: [json.loads((home / "polyak" / f"seed{run}.json").read_text()) for run in SEEDS]
            for seed, home in homes.items()}
    sets = list(homes)
    print(f"C-q.9 (post hoc): the ten JEPA runs of {SWEEPS['polyak']} analysed on five episode sets (analysis seeds {sets}; "
          f"{SETTINGS.seed} is C-q.7's), CPU, one thread, {SETTINGS.episodes} episodes, the selected-iterate MLP probe")
    print("per quantity: the interval over the ten training seeds within each episode set (Student-t 95%; what one report "
          "gives), then the crossed analysis: grand mean with a 95% interval for runs and episode sets as two random "
          "factors, and the standard deviations of the run effect, the episode-set effect and the residual")
    returns = lambda row, agent: row["analysis"]["returns"][agent]["mean"]                                   # noqa: E731
    quantities = [(f"{agent} - optimal (paired)", lambda row, agent=agent: returns(row, agent) - returns(row, "optimal"), 3)
                  for agent in AGENTS[1:]]
    quantities += [("optimal agent's return", lambda row: returns(row, "optimal"), 3)]
    quantities += [(f"{probe} mean L1", lambda row, probe=probe: row["analysis"]["probes"][probe]["mean_l1"], 4)
                   for probe in PROBES]
    quantities += [(f"mlp/agent {key}", lambda row, key=key: row["analysis"]["bounds"]["mlp/agent"][key], 4)
                   for key in ("value_error_mean", "value_error_bound_mean", "value_error_bound_local_mean", "regret_mean",
                               "regret_bound_mean", "regret_bound_local_mean", "policy_loss_bound_mean",
                               "policy_loss_bound_local_mean", "suboptimal_decision_rate")]
    for label, value, digits in quantities:
        table = np.array([[value(row) for row in rows[seed]] for seed in sets]).T          # (runs, episode sets)
        print(f"\n  {label}")
        for column, seed in enumerate(sets):
            if np.ptp(table[:, column]) == 0:
                print(f"    episode set {seed:<9} {table[0, column]:.{digits}f} (the same for every run)")
            else:
                print(f"    episode set {seed:<9} {_interval(table[:, column].tolist(), digits)}")
        c = _crossed(table)
        interval = (f"[{c['mean'] - c['half']:.{digits}f}, {c['mean'] + c['half']:.{digits}f}] (Satterthwaite df {c['df']:.1f})"
                    if not math.isnan(c["half"]) else "[variance of the grand mean estimated at or below 0]")
        print(f"    crossed: {c['mean']:.{digits}f} {interval}; sd run {c['sd_run']:.{digits}f}, episode set "
              f"{c['sd_set']:.{digits}f}, residual {c['sd_res']:.{digits}f}")
        print(f"    per run, mean over the five sets: {np.round(table.mean(axis=1), digits).tolist()}")
    new = [row for seed in ANALYSIS_SEEDS for row in rows[seed]]
    print(f"\ncost: {np.mean([row['seconds'] for row in new]) / 60:.1f} min per run on one CPU thread for the four new "
          f"analysis seeds; commits {sorted({row['commit'][:7] for row in new})}")


def main() -> None:
    parser = argparse.ArgumentParser(description="C-q.7, C-q.8 and C-q.9 (module header).")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("play")
    p.add_argument("--sweep", required=True, choices=sorted(SWEEPS))
    p.add_argument("--seeds", required=True, type=lambda text: [int(k) for k in text.split(",")])
    p.add_argument("--episodes", type=int, default=SETTINGS.episodes)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fit", choices=("selected", "last"), default="selected")
    p.add_argument("--analysis-seed", type=int, default=SETTINGS.seed)
    sub.add_parser("readout")
    sub.add_parser("compare")
    sub.add_parser("seeds")
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("hide the GPU (CUDA_VISIBLE_DEVICES=): this analysis is registered on the CPU (section 2).")
    if args.command == "play":
        play(args.sweep, args.seeds, args.episodes, args.out, args.fit, args.analysis_seed)
    elif args.command == "readout":
        readout()
    elif args.command == "compare":
        compare()
    else:
        seeds_readout()


if __name__ == "__main__":
    main()
