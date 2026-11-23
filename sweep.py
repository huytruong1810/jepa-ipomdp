# ABSOLUTE PATH: sweep.py
# ==============================================================================
# MULTI-SEED EXPERIMENT: TRAIN, ANALYSE AND AGGREGATE OVER TRAINING SEEDS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Usage:
#      uv run sweep.py <name> --seeds 0,1,2,3,4 [hydra overrides ...]
#    e.g. `uv run sweep.py baseline --seeds 0,1,2,3,4` or
#         `uv run sweep.py short --seeds 0,1,2 training.total_episodes=1280`.
#    Every seed k is trained by main.py into runs/sweeps/<name>/seed<k>/ (the overrides are
#    passed through unchanged), analysed by the same analyze_run as analyze.py, and all seeds
#    are aggregated into runs/sweeps/<name>/aggregate.json with Student-t 95% intervals
#    (ipomdp/experiments/aggregate.py explains why the seed is the unit of replication).
#
# 2. Sequential, One Process per Training Run:
#    - The single GPU is the bottleneck (search is Python-bound), so seeds run one after another. Each training run is its own main.py
#      process, so it is launched exactly as a hand-started run would be (same Hydra config
#      composition, same run-directory contents) and its GPU memory is released before the
#      next seed.
#    - The analysis, by contrast, runs in THIS process, so its CUDA cache is released right after it
#      (2026-10-24). Before that, the ~5.6 GB cached by one seed's analysis stayed allocated while the
#      next seed's main.py trained beside it, which with a second sweep's work tripped main.py's
#      device-wide VRAM guardrail on 2026-10-22. Releasing the cache changes no result.
#    - main.py exits with 0 only when every collection completed (main.py, section 4). Any
#      other status stops the sweep.
#
# 3. Continuing an Interrupted Sweep:
#    - runs/sweeps/<name>/sweep.json records the overrides and analysis settings. Re-running
#      the command with the SAME overrides skips every seed whose analysis/report.json exists,
#      analyses seeds marked as trained but not yet analysed, and trains the rest; more seeds
#      may be added this way. Different overrides or settings raise: one sweep directory is one
#      experimental condition. A seed directory without the trained marker is an interrupted
#      training run and raises too: delete it, or finish it by hand with
#      `main.py resume=<seed dir>/checkpoints/latest.pt hydra.run.dir=<seed dir>` and create the
#      marker. Resuming is bit-exact only from the state at which latest.pt was written, so
#      that choice is left to the user.
#
# 4. One Analysis per Domain (2026-10-10; Interactive Tiger in section 7):
#    - The domain is the last `env=<name>` override, else conf/config.yaml's default (tiger).
#      Tiger seeds are analysed by analyze_run (exact solver), Wumpus seeds by analyze_wumpus_run,
#      and aggregate_reports uses that domain's metric table (experiments/aggregate.py, section 5).
#      --solver-tolerance is Tiger-only and raises for the other domains.
#    - Runs trained elsewhere join a sweep by symlinking their directory as seed<k> with the
#      trained marker: a seed whose analysis/report.json exists is aggregated as is. The recorded
#      overrides must then be the ones those runs were trained with.
#
# 5. Held-Out Evaluation (2026-10-19; reports/2026-10-19_eval_protocol/PROTOCOL.md):
#    - --analysis-output (Wumpus) names the directory each seed's analysis is written to and read
#      from (experiments/wumpus.py, section 3), so a sweep over symlinked development runs can be
#      re-analysed on a held-out game seed without touching their development reports.
#    - Wumpus aggregates add the hierarchical bootstrap over seeds and games
#      (experiments/aggregate.py, section 6), read from every seed's games.npz. A Wumpus seed
#      analysed before games.npz existed raises: re-analyse it into a new output directory.
#    - --exact-planner-games (Wumpus) adds the exact-belief planner reference, played once per
#      analysis seed by reports/2026-10-19_eval_protocol/exact_planner_play.py. Its games.npz
#      must come from the sweep's analysis seed (checked) and be played on the same games
#      (checked against the heuristic's per-game returns, experiments/aggregate.py section 6).
#      It changes the aggregate only, so it is not part of the recorded condition.
#    - --checkpoint (Wumpus; default best.pt) names the networks each seed's analysis reads
#      (experiments/wumpus.py, section 3). It is part of the recorded analysis condition. With a
#      final_retrain run (main.py, section 5), best.pt and best_eval.pt are compared by two sweeps
#      over the same seed directories (symlinked, section 4), each with its own --analysis-output.
#    - --baseline-runs (Wumpus) adds the model-free baseline (PROTOCOL.md C-c), trained per seed by
#      reports/2026-10-19_eval_protocol/recurrent_dqn_baseline.py: seed k's games from
#      <dir>/seed<k>/<analysis output>/games.npz, same checks, aggregate only.
#
# 6. A Stored Report Must Match the Sweep's Analysis Settings (2026-11-14, submission audit W9):
#    - A seed whose report.json exists is aggregated, not re-analysed. check_report_settings raises if that report's
#      recorded settings differ from the sweep's in any field both record (analysis seed, episodes, probe steps, solver
#      tolerance, reference thresholds, output, checkpoint), so a report from another analysis condition can never
#      enter an aggregate.
#    - Two fields are younger than some stored reports, and each had a single possible value before it existed. Reports
#      written before 2026-10-19 record no output; the analysis then always wrote to "analysis". Reports written before
#      2026-10-27 record no checkpoint; the analysis then read only best.pt. A missing output or checkpoint is checked
#      as that value; any other missing field raises.
#
# 7. Interactive Tiger (2026-11-23):
#    - `env=itiger` seeds are analysed by analyze_interactive_run against --benchmark, the certified bracket of the
#      fold (experiments/interactive_analysis.py; one file per fold, e.g. runs/benchmarks/itiger_level1.pt). The
#      option is required for this domain and refused for the others. An unknown domain raises: before this section
#      every domain but Wumpus World took the Tiger path, which would have trained an itiger seed and then failed in
#      its analysis.
#    - The benchmark is part of the condition. sweep.json records its lower bound, upper bound and fold digest (not
#      its path: a copy of the file is the same benchmark), and a stored report whose bracket differs from the file's
#      raises, as a report with other analysis settings does (section 6). The analysis itself refuses a file computed
#      on another fold.
#    - The aggregate carries the bracket beside the statistics: every gap of this domain is a gap to the reference
#      agent, and the bracket's width bounds what that agent may lose to the optimum (experiments/aggregate.py,
#      section 8).
# ==============================================================================

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from omegaconf import OmegaConf

import numpy as np

from ipomdp.experiments import (AnalysisSettings, WumpusAnalysisSettings, aggregate_games, aggregate_reports, analyze_run,
                                analyze_wumpus_run)
from ipomdp.experiments.interactive_analysis import (InteractiveAnalysisSettings, InteractiveBenchmark,
                                                     analyze_interactive_run)

REPO = Path(__file__).resolve().parent
TRAINED_MARKER = "TRAINED"
Settings = AnalysisSettings | WumpusAnalysisSettings | InteractiveAnalysisSettings


def train(seed_dir: Path, seed: int, overrides: list[str]) -> None:
    command = [sys.executable, str(REPO / "main.py"), f"seed={seed}", f"hydra.run.dir={seed_dir}", *overrides]
    print(f"[sweep] training: {' '.join(command[1:])}", flush=True)
    start = time.monotonic()
    subprocess.run(command, cwd=REPO, check=True)
    (seed_dir / TRAINED_MARKER).write_text(f"{time.monotonic() - start:.0f} s\n")


def domain_of(overrides: list[str]) -> str:
    """Section 4: the last env=<name> override, else the default env of conf/config.yaml."""
    chosen = [o.split("=", 1)[1] for o in overrides if o.split("=", 1)[0] == "env"]
    if chosen:
        return chosen[-1]
    (default,) = [entry.env for entry in OmegaConf.load(REPO / "conf" / "config.yaml").defaults
                  if OmegaConf.is_dict(entry) and "env" in entry]
    return default


def benchmark_identity(benchmark: InteractiveBenchmark) -> dict:
    """Section 7: what identifies a benchmark in a condition and in a stored report's bracket."""
    return {"lower": benchmark.lower, "upper": benchmark.upper, "digest": benchmark.digest}


def record_condition(sweep_dir: Path, overrides: list[str], settings: Settings,
                     benchmark: InteractiveBenchmark | None = None) -> None:
    """
    Section 3: one sweep directory is one experimental condition. `benchmark` (Interactive Tiger, section 7) adds
    its identity; without it the recorded condition is the one Tiger and Wumpus sweeps have always written.
    """
    # Normalised through JSON: tuples in the settings (Wumpus reference thresholds) come back as lists.
    condition = json.loads(json.dumps({"overrides": overrides, "analysis": asdict(settings)}
                                      | ({} if benchmark is None else {"benchmark": benchmark_identity(benchmark)})))
    path = sweep_dir / "sweep.json"
    if path.exists():
        recorded = json.loads(path.read_text())
        if recorded != condition:
            raise ValueError(f"{sweep_dir} was started with {recorded}, not {condition}. Use a new sweep name.")
    else:
        sweep_dir.mkdir(parents=True, exist_ok=True)   # may already hold symlinked seeds (section 4)
        path.write_text(json.dumps(condition, indent=2))


def check_report_benchmark(report: dict, benchmark: InteractiveBenchmark, path: Path) -> None:
    """Section 7: the stored report was analysed against this benchmark (same bounds, same fold)."""
    recorded, wanted = report["bracket"], benchmark_identity(benchmark)
    differ = sorted(k for k in wanted if recorded[k] != wanted[k])
    if differ:
        raise ValueError(f"{path} was analysed against another benchmark: "
                         + ", ".join(f"{k} {recorded[k]!r} != {wanted[k]!r}" for k in differ) + " (section 7).")


def check_report_settings(report: dict, settings: Settings, path: Path) -> None:
    """Section 6: the stored report's analysis settings equal the sweep's (missing output / checkpoint: their only old value)."""
    recorded, wanted = dict(report.get("settings", {})), json.loads(json.dumps(asdict(settings)))
    for field, only_value in (("output", "analysis"), ("checkpoint", "best.pt")):       # section 6: younger fields
        if field in wanted and field not in recorded:
            recorded[field] = only_value
    missing = sorted(set(wanted) - set(recorded))
    differ = sorted(k for k in wanted if k in recorded and recorded[k] != wanted[k])
    if missing or differ:
        raise ValueError(f"{path} was analysed with other settings: missing {missing}, differing "
                         + ", ".join(f"{k} {recorded[k]!r} != {wanted[k]!r}" for k in differ) + " (section 6).")


def main() -> None:
    defaults = AnalysisSettings()
    parser = argparse.ArgumentParser(description="Train, analyse and aggregate over seeds (see module header).")
    parser.add_argument("name", help="Sweep directory name under runs/sweeps/.")
    parser.add_argument("--seeds", required=True, type=lambda s: [int(k) for k in s.split(",")],
                        help="Comma-separated training seeds, e.g. 0,1,2,3,4.")
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--solver-tolerance", type=float, default=None,
                        help=f"Tiger only (default {defaults.solver_tolerance}).")
    parser.add_argument("--mlp-probe-steps", type=int, default=defaults.mlp_probe_steps)
    parser.add_argument("--analysis-seed", type=int, default=defaults.seed)
    parser.add_argument("--analysis-output", default=None,
                        help="Wumpus only: analysis directory under each seed (section 5; default analysis).")
    parser.add_argument("--checkpoint", default=None,
                        help="Wumpus only: checkpoint file analysed under each seed's checkpoints/ (section 5; default best.pt).")
    parser.add_argument("--exact-planner-games", type=Path, default=None,
                        help="Wumpus only: the exact planner's games.npz of the same analysis seed (section 6).")
    parser.add_argument("--baseline-runs", type=Path, default=None,
                        help="Wumpus only: runs/baselines/<name>, read seed<k>/<analysis output>/games.npz (section 5).")
    parser.add_argument("--benchmark", type=Path, default=None,
                        help="Interactive Tiger only, required there: the fold's benchmark file (section 7).")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides passed to every main.py run.")
    args = parser.parse_intermixed_args()
    for override in args.overrides:
        if override.split("=")[0].lstrip("+~") in ("seed", "hydra.run.dir", "resume"):
            raise ValueError(f"{override!r} is set by the sweep itself.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"Duplicate seeds in {args.seeds}.")

    domain = domain_of(args.overrides)
    wumpus_only = (args.analysis_output is not None or args.checkpoint is not None or args.exact_planner_games is not None
                   or args.baseline_runs is not None)
    if domain != "wumpus" and wumpus_only:
        raise ValueError("--analysis-output, --checkpoint, --exact-planner-games and --baseline-runs apply to Wumpus World only.")
    if domain != "tiger" and args.solver_tolerance is not None:
        raise ValueError("--solver-tolerance applies to Tiger only: the other domains have no exact solver.")
    if (domain == "itiger") != (args.benchmark is not None):
        raise ValueError("--benchmark is required for env=itiger and applies to env=itiger only (section 7).")
    benchmark = None
    if domain == "wumpus":
        settings = WumpusAnalysisSettings(args.episodes, args.mlp_probe_steps, args.analysis_seed)
        if args.analysis_output is not None:
            settings = replace(settings, output=args.analysis_output)
        if args.checkpoint is not None:
            settings = replace(settings, checkpoint=args.checkpoint)
        output = settings.output
        analyze = analyze_wumpus_run
    elif domain == "itiger":
        output = "analysis"
        settings = InteractiveAnalysisSettings(args.episodes, args.mlp_probe_steps, args.analysis_seed)
        benchmark = InteractiveBenchmark.load(args.benchmark)

        def analyze(run_dir, run_settings, run_device):
            return analyze_interactive_run(run_dir, run_settings, args.benchmark, run_device)
    elif domain == "tiger":
        output = "analysis"
        tolerance = defaults.solver_tolerance if args.solver_tolerance is None else args.solver_tolerance
        settings = AnalysisSettings(args.episodes, tolerance, args.mlp_probe_steps, args.analysis_seed)
        analyze = analyze_run
    else:
        raise ValueError(f"sweep.py has no analysis for env={domain} (sections 4 and 7).")
    sweep_dir = REPO / "runs" / "sweeps" / args.name
    record_condition(sweep_dir, args.overrides, settings, benchmark)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reports = []
    for seed in args.seeds:
        seed_dir = sweep_dir / f"seed{seed}"
        report_path = seed_dir / output / "report.json"
        if not report_path.exists():
            if seed_dir.exists() and not (seed_dir / TRAINED_MARKER).exists():
                raise RuntimeError(f"{seed_dir} holds an interrupted training run (module header, section 3).")
            if not seed_dir.exists():
                train(seed_dir, seed, args.overrides)
            print(f"[sweep] analysing seed {seed}", flush=True)
            analyze(seed_dir, settings, device)
            if device.type == "cuda":
                torch.cuda.empty_cache()     # section 2: the analysis runs in this process; free its cache
        report = json.loads(report_path.read_text())
        check_report_settings(report, settings, report_path)                       # section 6
        if benchmark is not None:
            check_report_benchmark(report, benchmark, report_path)                 # section 7
        reports.append(report)

    statistics = aggregate_reports(reports, domain)
    reference = {"tiger": lambda: {"optimal_value_b0": reports[0]["optimal_value_b0"]},
                 "itiger": lambda: {"bracket": reports[0]["bracket"]},
                 "wumpus": lambda: {}}[domain]()
    hierarchical = {}
    if domain == "wumpus":
        games = []
        for seed in args.seeds:
            with np.load(sweep_dir / f"seed{seed}" / output / "games.npz") as rows:
                games.append(dict(rows))
        exact_planner = None
        if args.exact_planner_games is not None:
            with np.load(args.exact_planner_games) as rows:
                exact_planner = dict(rows)
            if int(exact_planner["analysis_seed"]) != settings.seed:
                raise ValueError(f"{args.exact_planner_games} was played on analysis seed "
                                 f"{int(exact_planner['analysis_seed'])}, not {settings.seed}.")
        baseline = None
        if args.baseline_runs is not None:
            baseline = []
            for seed in args.seeds:
                with np.load(args.baseline_runs / f"seed{seed}" / output / "games.npz") as rows:
                    baseline.append(dict(rows))
                if int(baseline[-1]["analysis_seed"]) != settings.seed:
                    raise ValueError(f"{args.baseline_runs}/seed{seed} was analysed on another analysis seed.")
        hierarchical = aggregate_games(games, exact_planner, baseline)
    aggregate = {"domain": domain, "seeds": args.seeds, **reference,
                 "metrics": {name: asdict(stat) for name, stat in statistics.items()},
                 "hierarchical": {name: asdict(stat) for name, stat in hierarchical.items()}}
    # One aggregate per analysis directory, so a held-out aggregate never replaces the development one.
    aggregate_path = sweep_dir / f"aggregate{'' if output == 'analysis' else '_' + output}.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2))
    headline = {"tiger": lambda: f", V*(b0) = {reference['optimal_value_b0']:.3f}",
                "itiger": lambda: (f", certified bracket of V*(b0) [{reference['bracket']['lower']:.4f}, "
                                   f"{reference['bracket']['upper']:.4f}] (gaps are to the reference agent)"),
                "wumpus": lambda: ""}[domain]()
    print(f"\n{args.name} ({domain}): {len(args.seeds)} seeds {args.seeds}{headline}")
    print(f"  {'metric':<40} {'mean':>10} {'t 95% CI':>10} {'std':>10} {'IQM':>10} {'IQM 95% bootstrap':>22}")
    for name, stat in statistics.items():
        print(f"  {name:<40} {stat.mean:10.4f} {stat.ci95:>10.4f} {stat.std:10.4f} {stat.iqm:10.4f} "
              f"[{stat.iqm_ci95[0]:9.4f}, {stat.iqm_ci95[1]:9.4f}]")
    if hierarchical:
        print(f"  hierarchical bootstrap (seeds, then games): {'mean':>10} {'95% interval':>22}")
        for name, stat in hierarchical.items():
            print(f"  {name:<40} {stat.mean:10.4f} [{stat.ci95[0]:9.4f}, {stat.ci95[1]:9.4f}]")
    print(f"aggregate: {aggregate_path}")


if __name__ == "__main__":
    main()
