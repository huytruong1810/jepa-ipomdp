# ABSOLUTE PATH: reports/2026-10-19_eval_protocol/w11_readout.py
# ==============================================================================
# C-o READOUT: THE HELD-OUT WUMPUS NUMBERS ON TEN TRAINING SEEDS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. What Is Read:
#    - PROTOCOL.md C-o extends the held-out set from training seeds 5-9 to 5-14 for the frozen recipe. Four quantities
#      are read, all on analysis seed 20261019 (512 games, the same games for every seed and agent):
#        C-o.1  the recipe's return per game, its rates of games won, died and unfinished, and its paired gaps per
#               game to the heuristic (risk 0.4) and to the exact-belief planner;
#        C-o.2  the recipe minus the model-free baseline, paired per game;
#        C-o.3  the exact one-step model in play (EPR) minus the learned planner, paired per game, with EPR's death
#               rate and the number of seeds on which the gain is positive;
#        C-o.4  the probes of the latent (cell and heading accuracy, pit hazard error) on random and on the agent's own
#               histories, beside the constant predictors of the analysis seed's random histories.
#
# 2. Three Reports of Each Quantity:
#    - Seeds 10-14 alone (the replication), seeds 5-9 (read earlier), and all ten (the numbers the paper cites).
#
# 3. Statistics:
#    - Per-seed values are always printed. The interval to cite is Student-t 95% on the per-seed means (per-seed
#      paired means for gaps). The hierarchical bootstrap (seeds, then games) and the crossed bootstrap (seeds x games)
#      are printed beside it and labelled.
#    - Every table and interval comes from the functions the submission audit's ledger uses (ipomdp.experiments:
#      game_metric_rows, seed_statistic, hierarchical_bootstrap, crossed_bootstrap), so the report of seeds 5-9 must
#      equal the ledger's `heldout.*` values. `--check` verifies exactly that, and reads nothing of seeds 10-14.
#
# 4. No Partial Read:
#    - Without `--check` the script prints nothing unless every file of every new seed exists (the recipe's games and
#      report, the baseline's games, the EPR play). A missing file is listed and the script exits with status 2.
#    - Every seed must have played the same games: the heuristic's per-game returns are compared across all seeds.
#
# 5. The Prediction of C-o.3 (stated in the registration, before any run on seeds 10-14):
#    - The paired gain is positive on at least 4 of the 5 new seeds, and the Student-t interval over the ten seeds
#      excludes zero. The script prints which parts hold; either outcome is reported.
#
# 6. Usage (CPU only, from the repository root):
#      uv run python reports/2026-10-19_eval_protocol/w11_readout.py --check     # seeds 5-9 against the ledger
#      uv run python reports/2026-10-19_eval_protocol/w11_readout.py > runs/wumpus/w11/readout.log
#    - The second form also writes every printed value to runs/wumpus/w11/readout.json, for the ledger's new ids.
# ==============================================================================

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from ipomdp.experiments import crossed_bootstrap, game_metric_rows, hierarchical_bootstrap, seed_statistic

OLD, NEW = tuple(range(5, 10)), tuple(range(10, 15))
RUNS = Path("runs")
SWEEP = {**{seed: "wumpus_retrain_heldout" for seed in OLD}, **{seed: "wumpus_retrain_heldout_w11" for seed in NEW}}
EXACT_PLANNER = RUNS / "references" / "wumpus_exact_planner" / "seed20261019" / "games.npz"
CONSTANTS = RUNS / "wumpus" / "w7" / "median_constant.json"
OUTPUT = RUNS / "wumpus" / "w11" / "readout.json"
LEDGER = Path("reports/2026-11-14_submission_audit/ledger.py")


def recipe_dir(seed: int) -> Path:
    return RUNS / "sweeps" / SWEEP[seed] / f"seed{seed}" / "analysis_heldout"


def baseline_dir(seed: int) -> Path:
    return RUNS / "baselines" / "r2d2_heldout" / f"seed{seed}" / "analysis_heldout"


def required_files(seed: int) -> list[Path]:
    return [recipe_dir(seed) / "games.npz", recipe_dir(seed) / "report.json", recipe_dir(seed) / "exact_play_EPR.npz",
            baseline_dir(seed) / "games.npz"]


def npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def interval(table: np.ndarray) -> dict[str, float]:
    """Mean and the three 95% intervals of one (seeds, games) table (section 3)."""
    student = seed_statistic([float(v) for v in table.mean(axis=1)])
    nested, crossed = hierarchical_bootstrap(table), crossed_bootstrap(table)
    return {"mean": student.mean, "t_lo": student.mean - student.ci95, "t_hi": student.mean + student.ci95,
            "lo": nested.ci95[0], "hi": nested.ci95[1], "crossed_lo": crossed.ci95[0], "crossed_hi": crossed.ci95[1]}


def show(label: str, table: np.ndarray) -> dict[str, float]:
    row = interval(table)
    per_seed = ", ".join(f"{v:+.1f}" for v in table.mean(axis=1))
    print(f"  {label}: {row['mean']:+.2f}; per seed {per_seed}")
    print(f"      Student-t 95% on the per-seed means [{row['t_lo']:+.2f}, {row['t_hi']:+.2f}] (the interval to cite); "
          f"hierarchical bootstrap [{row['lo']:+.2f}, {row['hi']:+.2f}]; crossed bootstrap "
          f"[{row['crossed_lo']:+.2f}, {row['crossed_hi']:+.2f}]")
    return row


def report(title: str, seeds: tuple[int, ...]) -> dict[str, float]:
    """Print C-o.1 to C-o.4 for one set of training seeds and return every value under the ledger's key names."""
    games = [npz(recipe_dir(seed) / "games.npz") for seed in seeds]
    baseline = [npz(baseline_dir(seed) / "games.npz") for seed in seeds]
    plays = [npz(recipe_dir(seed) / "exact_play_EPR.npz") for seed in seeds]
    reports = [json.loads((recipe_dir(seed) / "report.json").read_text()) for seed in seeds]
    for seed, seed_games in zip(seeds, games):
        if not np.array_equal(seed_games["reference_risk0.4.return"], games[0]["reference_risk0.4.return"]):
            raise ValueError(f"seed {seed} was not analysed on the games of seed {seeds[0]} (section 4).")
    rows = game_metric_rows(games, npz(EXACT_PLANNER), baseline)
    values: dict[str, float] = {}

    def keep(key: str, row: dict[str, float]) -> None:
        values.update({f"{key}.{suffix}": value for suffix, value in row.items()})

    print(f"\n=== {title}: training seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} seeds x {rows['return/learned_planner'].shape[1]} games)")
    print("C-o.1 the recipe")
    keep("ours.return", show("return per game", rows["return/learned_planner"]))
    won = np.stack([g["learned_planner.won"].astype(float) for g in games])
    died = np.stack([g["learned_planner.died"].astype(float) for g in games])
    values.update({"ours.won_pct": 100 * won.mean(), "ours.died_pct": 100 * died.mean(),
                   "ours.unfinished_pct": 100 * (1 - won - died).mean()})
    print(f"  games won {values['ours.won_pct']:.1f}%, died {values['ours.died_pct']:.1f}%, unfinished "
          f"{values['ours.unfinished_pct']:.1f}%; per seed won " + ", ".join(f"{100 * v:.1f}" for v in won.mean(axis=1))
          + "; died " + ", ".join(f"{100 * v:.1f}" for v in died.mean(axis=1)))
    keep("ours_minus_heuristic", show("recipe minus the heuristic (risk 0.4), paired", rows["gap/learned_planner - reference_risk0.4"]))
    keep("ours_minus_exact_planner", show("recipe minus the exact-belief planner, paired", rows["gap/learned_planner - exact_planner"]))
    print("C-o.2 the model-free baseline")
    keep("ours_minus_baseline", show("recipe minus the baseline, paired", rows["gap/learned_planner - recurrent_dqn"]))
    print("C-o.3 the exact one-step model in play (EPR)")
    gain = np.stack([p["returns"] for p in plays]) - rows["return/learned_planner"]
    keep("exact_model_minus_ours", show("EPR minus the learned planner, paired", gain))
    deaths = [100 * float(p["death_rate"]) for p in plays]
    values["exact_model.died_pct"] = float(np.mean(deaths))
    values["exact_model.seeds_positive"] = int((gain.mean(axis=1) > 0).sum())
    print(f"  EPR deaths {values['exact_model.died_pct']:.2f}% (per seed " + ", ".join(f"{d:.2f}" for d in deaths)
          + f"); the gain is positive on {values['exact_model.seeds_positive']} of {len(seeds)} seeds")
    print("C-o.4 the probes of the latent (MLP probe; mean over seeds, per-seed values beside it)")
    constants = json.loads(CONSTANTS.read_text())
    probes = {"probe.cell_random_pct": ("mlp_random", ("cell_accuracy",), 100), "probe.heading_random_pct": ("mlp_random", ("heading_accuracy",), 100),
              "probe.pit_error": ("mlp_random", ("hazard_error", "pit"), 1), "probe.cell_agent_pct": ("mlp_agent", ("cell_accuracy",), 100),
              "probe.heading_agent_pct": ("mlp_agent", ("heading_accuracy",), 100), "probe.pit_error_agent": ("mlp_agent", ("hazard_error", "pit"), 1)}
    for key, (split, path, scale) in probes.items():
        per_seed = []
        for seed_report in reports:
            value = seed_report["probes"][split]
            for name in path:
                value = value[name]
            per_seed.append(scale * float(value))
        values[key] = float(np.mean(per_seed))
        student = seed_statistic(per_seed)
        print(f"  {key}: {values[key]:.4g}; per seed " + ", ".join(f"{v:.4g}" for v in per_seed)
              + f"; Student-t 95% [{student.mean - student.ci95:.4g}, {student.mean + student.ci95:.4g}]")
    print(f"  constant predictors of the pit hazard on random histories: median {constants['median_constant']['pit']:.4g}, "
          f"mean {constants['mean_constant']['pit']:.4g} ({CONSTANTS})")
    return values


def check() -> None:
    """Section 3: the report of seeds 5-9 against the audit ledger's `heldout.*` values."""
    spec = importlib.util.spec_from_file_location("ledger", LEDGER)
    ledger_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ledger_module)
    ledger = ledger_module.Ledger()
    ledger_module.wumpus_heldout(ledger)
    mine = report("already read (the check against the ledger)", OLD)
    compared, bad = 0, []
    for key, value in mine.items():
        if key == "probe.pit_error_agent":            # the ledger has no id for the pit error on own histories
            continue
        compared += 1
        if abs(ledger.values[f"heldout.{key}"] - value) > 1e-9:
            bad.append((key, value, ledger.values[f"heldout.{key}"]))
    for row in bad:
        print("  DIFFERENT:", row)
    print(f"\nCHECK {'PASSED' if not bad else 'FAILED'}: {compared - len(bad)} of {compared} values of seeds 5-9 equal the ledger's")
    sys.exit(1 if bad else 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="seeds 5-9 only, compared with the ledger (section 3)")
    args = parser.parse_args()
    if args.check:
        check()
    missing = [path for seed in NEW for path in required_files(seed) if not path.exists()]
    if missing:
        print(f"C-o is not read: {len(missing)} files of the new seeds are missing (section 4), the first being {missing[0]}",
              file=sys.stderr)
        sys.exit(2)
    out = {"new": report("the replication", NEW), "old": report("read earlier", OLD), "all": report("all ten seeds, the numbers to cite", OLD + NEW)}
    new, everything = out["new"], out["all"]
    positive, excludes = new["exact_model.seeds_positive"] >= 4, everything["exact_model_minus_ours.t_lo"] > 0
    print(f"\nC-o.3 prediction: positive on at least 4 of the 5 new seeds: {'HOLDS' if positive else 'DOES NOT HOLD'} "
          f"({int(new['exact_model.seeds_positive'])} of 5); the ten-seed Student-t interval excludes zero: "
          f"{'HOLDS' if excludes else 'DOES NOT HOLD'} ([{everything['exact_model_minus_ours.t_lo']:+.2f}, "
          f"{everything['exact_model_minus_ours.t_hi']:+.2f}])")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
