# ABSOLUTE PATH: reports/2026-11-14_submission_audit/ledger.py
# ==============================================================================
# NUMBER LEDGER: EVERY NUMBER OF THE PAPER, RECOMPUTED FROM THE STORED RAW ARTIFACTS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. What It Is For:
#    - A number in the paper is trusted only if this script recomputes it from a raw artifact
#      (per-game returns, per-seed report.json, per-state diagnostics) and the recomputed value
#      rounds to what the paper prints. Aggregates written earlier (aggregate*.json), readout logs
#      and result tables in script headers are NOT sources here: they are what is being checked.
#    - claims.yaml lists the paper's numbers (id, printed value, printed decimals, where it stands).
#      compute() below produces a value for every id. The ledger prints OK or DIFF per claim, writes
#      ledger.md next to this file and exits 1 if any claim differs or has no computed value.
#
# 2. Sources (all under runs/, which git does not track; MANIFEST.sha256 pins them):
#    - Tiger: runs/sweeps/{polyak,decoder}/seed*/analysis/report.json.
#    - Wumpus held-out: runs/sweeps/<sweep>/seed{5..9}/analysis_heldout/{games.npz,report.json} for the
#      recipe and the three ablation arms, the exact-belief planner's games
#      (runs/references/wumpus_exact_planner/seed20261019/games.npz), the model-free baseline's games
#      (runs/baselines/r2d2_heldout/seed*/analysis_heldout/games.npz), and exact_play_*.npz.
#    - Wumpus development: runs/sweeps/wumpus_retrain/seed{0..4}/analysis/ (exact_play_*.npz,
#      feature_localise*.json), runs/wumpus/q_localise_seed*.log (per-state shares; the only log read,
#      because q_localise.npz stores the per-state Q values and the log line is their summary), and the
#      per-game returns of the retrain screens under runs/wumpus/{ve,da,qe}_screen, coverage and branch.
#    - The untrained-filter probe control exists only as a log (copied to logs/); its claims are parsed
#      from that copy and are marked "log" in the ledger.
#
# 3. Outcomes From Returns:
#    - The retrain screens stored per-game discounted returns only. A death pays at most
#      -1001 * 0.99^49 = -612 before step costs, an unfinished game at least -50 - 10 = -60 and a win
#      is positive, so died = return < -500 and won = return > 0 classify every game. The rule is
#      checked against every exact_play_*.npz, which stores both the returns and the rates.
#
# 4. Three Intervals per Contrast (src/ipomdp/experiments/aggregate.py, sections 2, 6 and 7):
#    - t: Student-t 95% on the per-seed means (for a gap: per-seed means of the paired per-game
#      differences). Small-sample interval for the seed factor; conditions on the evaluation games.
#    - nested: hierarchical_bootstrap, the interval the protocol registered and the paper cites.
#    - crossed: crossed_bootstrap, seeds and games resampled as the crossed design they are; for an
#      agent played once per analysis seed it is the game-level bootstrap.
#    - The statistics table (statistics.md) prints all three with the per-seed values for every
#      held-out contrast, so the paper's verdict words can be checked against each.
#
# 5. Usage (CPU only, numpy and scipy; about half a minute):
#      uv run python reports/2026-11-14_submission_audit/ledger.py            # ledger.md, statistics.md
#      uv run python reports/2026-11-14_submission_audit/ledger.py --manifest # also rewrite MANIFEST.sha256
# ==============================================================================

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from ipomdp.experiments import crossed_bootstrap, game_metric_rows, hierarchical_bootstrap, seed_statistic

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "runs"
DEV, HELD = range(5), range(5, 10)
ARTIFACTS: set[Path] = set()          # every file read, for the manifest


def _track(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    ARTIFACTS.add(path)
    return path


def load_json(path: Path) -> dict:
    return json.loads(_track(path).read_text())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(_track(path)) as rows:
        return dict(rows)


def load_npy(path: Path) -> np.ndarray:
    return np.load(_track(path))


def t_interval(values) -> tuple[float, float, float]:
    """Mean and Student-t 95% bounds over seeds (section 4)."""
    stat = seed_statistic([float(v) for v in values])
    return stat.mean, stat.mean - stat.ci95, stat.mean + stat.ci95


def outcomes(returns: np.ndarray) -> dict[str, float]:
    """Section 3: percentages of games won, died and unfinished, from per-game discounted returns."""
    won, died = returns > 0, returns < -500
    return {"won": 100 * won.mean(), "died": 100 * died.mean(), "unfinished": 100 * (~won & ~died).mean()}


class Ledger:
    def __init__(self):
        self.values: dict[str, float] = {}
        self.sources: dict[str, str] = {}
        self.contrasts: list[tuple[str, np.ndarray]] = []      # (name, (seeds, games) table) for statistics.md

    def put(self, key: str, value: float, source: str) -> None:
        if key in self.values:
            raise KeyError(f"duplicate ledger id {key}")
        self.values[key], self.sources[key] = float(value), source

    def interval(self, key: str, table: np.ndarray, source: str) -> None:
        """mean, the registered (nested) bounds, and the t and crossed bounds of one (seeds, games) table."""
        nested, crossed = hierarchical_bootstrap(table), crossed_bootstrap(table)
        mean, t_low, t_high = t_interval(table.mean(axis=1))
        for suffix, value in (("mean", mean), ("lo", nested.ci95[0]), ("hi", nested.ci95[1]), ("t_lo", t_low),
                              ("t_hi", t_high), ("crossed_lo", crossed.ci95[0]), ("crossed_hi", crossed.ci95[1])):
            self.put(f"{key}.{suffix}", value, source)
        self.contrasts.append((key, table))


# ------------------------------------------------------------------------------------------ Tiger
def tiger(ledger: Ledger) -> None:
    def reports(sweep: str) -> list[dict]:
        return [load_json(RUNS / "sweeps" / sweep / f"seed{s}" / "analysis" / "report.json") for s in range(10)]

    polyak, decoder = reports("polyak"), reports("decoder")
    src = "runs/sweeps/polyak/seed{0..9}/analysis/report.json"
    ret = lambda reps, agent: [r["returns"][agent]["mean"] for r in reps]  # noqa: E731
    optimal = ret(polyak, "optimal")
    if len(set(optimal)) != 1 or len({json.dumps(r["settings"], sort_keys=True) for r in polyak + decoder}) != 1:
        raise ValueError("Tiger seeds were not analysed on one common set of games.")
    ledger.put("tiger.seeds", len(polyak), src)
    ledger.put("tiger.optimal.return", optimal[0], src)
    ledger.put("tiger.optimal.value_b0", polyak[0]["optimal_value_b0"], src)
    for name, agent in (("learned", "learned_planner"), ("decoded_mlp", "decoded_mlp"), ("decoded_linear", "decoded_linear")):
        stat = seed_statistic(ret(polyak, agent))
        ledger.put(f"tiger.{name}.return", stat.mean, src)
        ledger.put(f"tiger.{name}.return_ci", stat.ci95, src)
        gap = seed_statistic([v - o for v, o in zip(ret(polyak, agent), optimal)])
        ledger.put(f"tiger.{name}.gap", gap.mean, src)
        ledger.put(f"tiger.{name}.gap_ci", gap.ci95, src)
    mean = lambda path: float(np.mean([_dig(r, path) for r in polyak]))  # noqa: E731
    ledger.put("tiger.mlp_agent.l1", mean(("probes", "mlp/agent", "mean_l1")), src)
    ledger.put("tiger.linear_agent.l1", mean(("probes", "linear/agent", "mean_l1")), src)
    ledger.put("tiger.mlp_random.l1", mean(("probes", "mlp/random", "mean_l1")), src)
    ledger.put("tiger.suboptimal_pct", 100 * mean(("bounds", "mlp/agent", "suboptimal_decision_rate")), src)
    for name, key in (("value_bound_mean", "value_error_bound_mean"), ("value_error_mean", "value_error_mean"),
                      ("regret_bound_mean", "regret_bound_mean"), ("regret_mean", "regret_mean"),
                      ("value_bound_max", "value_error_bound_max"), ("regret_bound_max", "regret_bound_max"),
                      ("policy_loss_bound_mean", "policy_loss_bound_mean")):
        ledger.put(f"tiger.bounds.{name}", mean(("bounds", "mlp/agent", key)), src)
    ledger.put("tiger.bounds.value_bound_max_over_seeds",
               max(_dig(r, ("bounds", "mlp/agent", "value_error_bound_max")) for r in polyak), src)
    ledger.put("tiger.bounds.regret_bound_max_over_seeds",
               max(_dig(r, ("bounds", "mlp/agent", "regret_bound_max")) for r in polyak), src)
    both = "runs/sweeps/{polyak,decoder}/seed{0..9}/analysis/report.json"
    gap = seed_statistic([v - o for v, o in zip(ret(decoder, "learned_planner"), ret(decoder, "optimal"))])
    ledger.put("tiger.decoder.gap", gap.mean, both)
    ledger.put("tiger.decoder.gap_ci", gap.ci95, both)
    diff = seed_statistic([d - p for d, p in zip(ret(decoder, "learned_planner"), ret(polyak, "learned_planner"))])
    ledger.put("tiger.decoder_minus_jepa", diff.mean, both)
    ledger.put("tiger.decoder_minus_jepa_ci", diff.ci95, both)
    geometry = [r["geometry"] for r in polyak]
    for name, values in (("pc12_pct", [100 * sum(g["explained_variance_ratio"][:2]) for g in geometry]),
                         ("side_pc2", [g["log_odds_spearman"][1] for g in geometry]),
                         ("confidence_pc1", [g["confidence_spearman"][0] for g in geometry]),
                         ("unexplained_pct", [100 * g["minimality_ratio"] for g in geometry])):
        ledger.put(f"tiger.geometry.{name}.min", min(values), src)
        ledger.put(f"tiger.geometry.{name}.max", max(values), src)


def _dig(report: dict, path: tuple[str, ...]) -> float:
    for key in path:
        report = report[key]
    return float(report)


# ------------------------------------------------------------------------------------------ Wumpus, held-out
def heldout_dir(sweep: str, seed: int) -> Path:
    return RUNS / "sweeps" / sweep / f"seed{seed}" / "analysis_heldout"


def wumpus_heldout(ledger: Ledger) -> dict[int, dict[str, np.ndarray]]:
    games = {s: load_npz(heldout_dir("wumpus_retrain_heldout", s) / "games.npz") for s in HELD}
    exact = load_npz(RUNS / "references" / "wumpus_exact_planner" / "seed20261019" / "games.npz")
    baseline = [load_npz(RUNS / "baselines" / "r2d2_heldout" / f"seed{s}" / "analysis_heldout" / "games.npz") for s in HELD]
    rows = game_metric_rows([games[s] for s in HELD], exact, baseline)
    src = "runs/sweeps/wumpus_retrain_heldout/seed{5..9}/analysis_heldout/games.npz (+ exact planner, baseline games)"
    for name, metric in (("ours", "learned_planner"), ("random", "random"), ("heuristic", "reference_risk0.4"),
                         ("exact_planner", "exact_planner"), ("baseline", "recurrent_dqn")):
        ledger.interval(f"heldout.{name}.return", rows[f"return/{metric}"], src)
    won = lambda agent: np.stack([games[s][f"{agent}.won"].astype(float) for s in HELD])  # noqa: E731
    died = lambda agent: np.stack([games[s][f"{agent}.died"].astype(float) for s in HELD])  # noqa: E731
    extra = {"exact_planner": (exact["exact_planner.won"][None].astype(float), exact["exact_planner.died"][None].astype(float)),
             "baseline": (np.stack([b["recurrent_dqn.won"].astype(float) for b in baseline]),
                          np.stack([b["recurrent_dqn.died"].astype(float) for b in baseline]))}
    for name, agent in (("ours", "learned_planner"), ("random", "random"), ("heuristic", "reference_risk0.4"),
                        ("exact_planner", None), ("baseline", None)):
        w, d = (won(agent), died(agent)) if agent is not None else extra[name]
        ledger.put(f"heldout.{name}.won_pct", 100 * w.mean(), src)
        ledger.put(f"heldout.{name}.died_pct", 100 * d.mean(), src)
        ledger.put(f"heldout.{name}.unfinished_pct", 100 * (1 - w - d).mean(), src)
    ledger.interval("heldout.ours_minus_baseline", rows["gap/learned_planner - recurrent_dqn"], src)
    ledger.interval("heldout.ours_minus_exact_planner", rows["gap/learned_planner - exact_planner"], src)
    ledger.interval("heldout.ours_minus_heuristic", rows["gap/learned_planner - reference_risk0.4"], src)

    reports = [load_json(heldout_dir("wumpus_retrain_heldout", s) / "report.json") for s in HELD]
    psrc = "runs/sweeps/wumpus_retrain_heldout/seed{5..9}/analysis_heldout/report.json"
    probe = lambda reps, split, key: float(np.mean([_dig(r, ("probes", split, *key)) for r in reps]))  # noqa: E731
    ledger.put("heldout.probe.cell_random_pct", 100 * probe(reports, "mlp_random", ("cell_accuracy",)), psrc)
    ledger.put("heldout.probe.heading_random_pct", 100 * probe(reports, "mlp_random", ("heading_accuracy",)), psrc)
    ledger.put("heldout.probe.cell_agent_pct", 100 * probe(reports, "mlp_agent", ("cell_accuracy",)), psrc)
    ledger.put("heldout.probe.heading_agent_pct", 100 * probe(reports, "mlp_agent", ("heading_accuracy",)), psrc)
    ledger.put("heldout.probe.pit_error", probe(reports, "mlp_random", ("hazard_error", "pit")), psrc)
    ledger.put("heldout.probe.pit_constant", probe(reports, "mlp_random", ("constant_error", "pit")), psrc)

    line = [text for text in _track(HERE / "logs" / "untrained_probe_control.log").read_text().splitlines()
            if text.strip().startswith("untrained mlp")][-1]
    found = re.search(r"cell ([\d.]+) heading ([\d.]+) hazard error pit ([\d.]+)", line)
    for name, value, scale in zip(("cell_pct", "heading_pct", "pit_error"), found.groups(), (100, 100, 1)):
        ledger.put(f"heldout.untrained_probe.{name}", scale * float(value), "logs/untrained_probe_control.log (log)")

    for arm in ("abl_max", "abl_noretrain", "abl_jepa"):
        arm_games = {s: load_npz(heldout_dir(arm, s) / "games.npz") for s in HELD}
        for s in HELD:
            if not np.array_equal(arm_games[s]["reference_risk0.4.return"], games[s]["reference_risk0.4.return"]):
                raise ValueError(f"{arm} seed {s} was not analysed on the recipe's games.")
        asrc = f"runs/sweeps/{{wumpus_retrain_heldout,{arm}}}/seed{{5..9}}/analysis_heldout/games.npz"
        arm_returns = np.stack([arm_games[s]["learned_planner.return"] for s in HELD])
        ledger.put(f"heldout.{arm}.return", arm_returns.mean(), asrc)
        ledger.interval(f"heldout.ours_minus_{arm}", rows["return/learned_planner"] - arm_returns, asrc)
        ledger.put(f"heldout.{arm}.died_pct", 100 * np.mean([arm_games[s]["learned_planner.died"].mean() for s in HELD]), asrc)
        arm_reports = [load_json(heldout_dir(arm, s) / "report.json") for s in HELD]
        ledger.put(f"heldout.{arm}.cell_random_pct", 100 * probe(arm_reports, "mlp_random", ("cell_accuracy",)), asrc)
        ledger.put(f"heldout.{arm}.heading_random_pct", 100 * probe(arm_reports, "mlp_random", ("heading_accuracy",)), asrc)
        ledger.put(f"heldout.{arm}.cell_agent_pct", 100 * probe(arm_reports, "mlp_agent", ("cell_accuracy",)), asrc)

    learned = rows["return/learned_planner"]
    for tag, name in (("EPR", "exact_model"), ("MIX0.5", "mixture"), ("learned_sims100", "sims100"), ("learned_sims10", "sims10")):
        plays = [load_npz(heldout_dir("wumpus_retrain_heldout", s) / f"exact_play_{tag}.npz") for s in HELD]
        _check_outcome_rule(plays)
        esrc = f"runs/sweeps/wumpus_retrain_heldout/seed{{5..9}}/analysis_heldout/exact_play_{tag}.npz"
        table = np.stack([p["returns"] for p in plays])
        ledger.put(f"heldout.{name}.return", table.mean(), esrc)
        ledger.interval(f"heldout.{name}_minus_ours", table - learned, esrc)
        deaths = [100 * float(p["death_rate"]) for p in plays]
        ledger.put(f"heldout.{name}.died_pct", float(np.mean(deaths)), esrc)
        ledger.put(f"heldout.{name}.seeds_positive", int((table.mean(1) > learned.mean(1)).sum()), esrc)
        ledger.put(f"heldout.{name}.seeds_deaths_up",
                   int(sum(d > 100 * games[s]["learned_planner.died"].mean() for d, s in zip(deaths, HELD))), esrc)
    return games


def _check_outcome_rule(plays: list[dict[str, np.ndarray]]) -> None:
    for play in plays:
        rule = outcomes(play["returns"])
        for key in ("won", "died", "unfinished"):
            stored = 100 * float(play[{"won": "win_rate", "died": "death_rate", "unfinished": "unfinished_rate"}[key]])
            if abs(rule[key] - stored) > 1e-9:
                raise ValueError(f"the outcome rule of section 3 disagrees with a stored rate ({key}: {rule[key]} vs {stored}).")


# ------------------------------------------------------------------------------------------ Wumpus, development
def dev_dir(seed: int) -> Path:
    return RUNS / "sweeps" / "wumpus_retrain" / f"seed{seed}" / "analysis"


def wumpus_development(ledger: Ledger) -> None:
    plays = {tag: [load_npz(dev_dir(s) / f"exact_play_{tag}.npz") for s in DEV] for tag in ("learned", "EP", "EPR")}
    for tag in plays:
        _check_outcome_rule(plays[tag])
    src = "runs/sweeps/wumpus_retrain/seed{0..4}/analysis/exact_play_{learned,EP,EPR}.npz"
    learned = np.stack([p["returns"] for p in plays["learned"]])
    reports = [load_json(dev_dir(s) / "report.json") for s in DEV]
    for seed, report in zip(DEV, reports):
        if abs(report["returns"]["learned_planner"]["discounted_return"] - learned[seed].mean()) > 1e-6:
            raise ValueError(f"development seed {seed}: the learned replay does not reproduce report.json.")
    ledger.put("dev.learned.return", learned.mean(), src)
    ledger.put("dev.learned.unfinished_pct", np.mean([outcomes(r)["unfinished"] for r in learned]), src)
    for tag, name in (("EPR", "exact_model"), ("EP", "exact_p")):
        table = np.stack([p["returns"] for p in plays[tag]])
        ledger.put(f"dev.{name}.return", table.mean(), src)
        mean, low, high = t_interval((table - learned).mean(axis=1))
        ledger.put(f"dev.{name}_minus_learned", mean, src)
        ledger.put(f"dev.{name}_minus_learned_ci", (high - low) / 2, src)
        ledger.put(f"dev.{name}.died_pct_max", max(outcomes(r)["died"] for r in table), src)
        ledger.put(f"dev.{name}.unfinished_pct", np.mean([outcomes(r)["unfinished"] for r in table]), src)

    shares = {"M": [], "EP": [], "EPR": []}
    for s in DEV:
        text = _track(RUNS / "wumpus" / f"q_localise_seed{s}.log").read_text()
        found = re.search(r"all \(\d+ states\): Q_mac > Q2\(chosen\): M ([\d.]+) .*?EP ([\d.]+) .*?EPR ([\d.]+)", text)
        for key, value in zip(shares, found.groups()):
            shares[key].append(100 * float(value))
    for key, values in shares.items():
        ledger.put(f"dev.flip_share.{key}.min", min(values), "runs/wumpus/q_localise_seed{0..4}.log")
        ledger.put(f"dev.flip_share.{key}.max", max(values), "runs/wumpus/q_localise_seed{0..4}.log")

    features = [load_json(dev_dir(s) / "feature_localise.json") for s in DEV]
    fsrc = "runs/sweeps/wumpus_retrain/seed{0..4}/analysis/feature_localise.json"
    for name, values in (("glitter_ratio", [f["joint_glitter_ratio_c"] for f in features]),
                         ("gold_calibration", [f["mlp"]["c"]["gold"]["calibration"] for f in features]),
                         ("visited_misread_pct", [100 * (1 - f["mlp"]["c"]["unvisited"]["acc"]) for f in features])):
        ledger.put(f"dev.declined.{name}.min", min(values), fsrc)
        ledger.put(f"dev.declined.{name}.max", max(values), fsrc)

    # Retrain-to-retrain spread: each control retrain against the agent whose buffer it was retrained on.
    for name, folder, seeds in (("data", "coverage", DEV), ("objective", "ve_screen", range(3))):
        spread = [(load_npy(RUNS / "wumpus" / folder / f"control_seed{s}.returns.npy") - learned[s]).mean() for s in seeds]
        csrc = f"runs/wumpus/{folder}/control_seed*.returns.npy vs exact_play_learned.npz"
        ledger.put(f"dev.control_minus_agent.{name}.min", min(spread), csrc)
        ledger.put(f"dev.control_minus_agent.{name}.max", max(spread), csrc)

    # Retrain screens: arm minus its control, paired per game (seeds as stored).
    screens = {  # arm -> (directory of the arm's returns, directory and stem of its control, seeds, feature json of the control)
        "VE25": ("ve_screen", "ve_screen/control", range(3), "ctl"), "VE100": ("ve_screen", "ve_screen/control", range(3), "ctl"),
        "DA50": ("da_screen", "ve_screen/control", range(3), "ctl"), "QE5": ("qe_screen", "ve_screen/control", range(3), "ctl"),
        "cover": ("coverage", "coverage/control", DEV, "control"), "own": ("coverage", "coverage/control", DEV, "control"),
        "union": ("coverage", "coverage/control", DEV, "control"), "branch": ("branch", "coverage/control", DEV, "control"),
        "random": ("branch", "coverage/control", DEV, "control"),
    }
    for arm, (folder, control, seeds, control_json) in screens.items():
        rsrc = f"runs/wumpus/{folder}/{arm}_seed*.returns.npy vs runs/wumpus/{control}_seed*.returns.npy"
        diffs, death_change, glitter_change = [], [], []
        for s in seeds:
            arm_returns = load_npy(RUNS / "wumpus" / folder / f"{arm}_seed{s}.returns.npy")
            control_returns = load_npy(RUNS / "wumpus" / f"{control}_seed{s}.returns.npy")
            diffs.append((arm_returns - control_returns).mean())
            death_change.append(outcomes(arm_returns)["died"] - outcomes(control_returns)["died"])
            arm_json, base_json = (dev_dir(s) / f"feature_localise_{name}_seed{s}.json" for name in (arm, control_json))
            if arm_json.exists() and base_json.exists():
                glitter_change.append(load_json(arm_json)["joint_glitter_ratio_c"] - load_json(base_json)["joint_glitter_ratio_c"])
        for s, value in zip(seeds, diffs):
            ledger.put(f"dev.remedy.{arm}.seed{s}", value, rsrc)
        ledger.put(f"dev.remedy.{arm}.deaths_seed0", death_change[0], rsrc)
        ledger.put(f"dev.remedy.{arm}.deaths_mean", float(np.mean(death_change)), rsrc)
        ledger.put(f"dev.remedy.{arm}.deaths_max", max(death_change), rsrc)
        if len(diffs) == 5:
            mean, low, high = t_interval(diffs)
            for suffix, value in (("mean", mean), ("lo", low), ("hi", high)):
                ledger.put(f"dev.remedy.{arm}.{suffix}", value, rsrc)
            ledger.put(f"dev.remedy.{arm}.seeds_positive", sum(d > 0 for d in diffs), rsrc)
        if glitter_change:
            gsrc = f"runs/sweeps/wumpus_retrain/seed*/analysis/feature_localise_{{{arm},{control_json}}}_seed*.json"
            ledger.put(f"dev.remedy.{arm}.glitter_change.n", len(glitter_change), gsrc)
            ledger.put(f"dev.remedy.{arm}.glitter_change.min", min(glitter_change), gsrc)
            ledger.put(f"dev.remedy.{arm}.glitter_change.max", max(glitter_change), gsrc)
            ledger.put(f"dev.remedy.{arm}.glitter_change.seeds_lower", sum(g < 0 for g in glitter_change), gsrc)
            ledger.put(f"dev.remedy.{arm}.glitter_change.seeds_repaired", sum(g >= 0.15 for g in glitter_change), gsrc)


# ------------------------------------------------------------------------------------------ report
def main() -> None:
    ledger = Ledger()
    tiger(ledger)
    wumpus_heldout(ledger)
    wumpus_development(ledger)

    claims = yaml.safe_load((HERE / "claims.yaml").read_text())
    lines = ["| id | paper | recomputed | where | source | |", "|---|---|---|---|---|---|"]
    bad = 0
    for claim in claims:
        key, paper, decimals = claim["id"], claim["paper"], claim["decimals"]
        if key not in ledger.values:
            verdict, shown = "NO VALUE", "-"
        else:
            value = ledger.values[key]
            shown = f"{value:.{decimals + 2}f}"
            verdict = "OK" if abs(round(value, decimals) - paper) < 0.5 * 10 ** -decimals else "DIFF"
        bad += verdict != "OK"
        lines.append(f"| {key} | {paper} | {shown} | {claim['where']} | {ledger.sources.get(key, '-')} | {verdict} |")
    (HERE / "ledger.md").write_text("# Number ledger (written by ledger.py; do not edit)\n\n" + "\n".join(lines) + "\n")
    for line in lines[2:]:
        if not line.endswith("| OK |"):
            print(line)
    print(f"\n{len(claims)} claims, {len(claims) - bad} OK, {bad} not OK; {len(ledger.values)} values computed "
          f"({len(set(ledger.values) - {c['id'] for c in claims})} without a claim)")

    out = ["# Held-out contrasts: per-seed means and three 95% intervals (written by ledger.py; do not edit)", "",
           "t = Student-t on the per-seed means; nested = registered hierarchical bootstrap; crossed = seeds x games.", "",
           "| contrast | mean | per seed | t | nested (registered) | crossed |", "|---|---|---|---|---|---|"]
    for name, table in ledger.contrasts:
        v = ledger.values
        per_seed = ", ".join(f"{x:+.1f}" for x in table.mean(axis=1))
        out.append(f"| {name} | {v[name + '.mean']:+.1f} | {per_seed} | [{v[name + '.t_lo']:+.1f}, {v[name + '.t_hi']:+.1f}] | "
                   f"[{v[name + '.lo']:+.1f}, {v[name + '.hi']:+.1f}] | [{v[name + '.crossed_lo']:+.1f}, {v[name + '.crossed_hi']:+.1f}] |")
    (HERE / "statistics.md").write_text("\n".join(out) + "\n")

    if "--manifest" in sys.argv:
        rows = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT)}" for path in sorted(ARTIFACTS)]
        (HERE / "MANIFEST.sha256").write_text("\n".join(rows) + "\n")
        print(f"MANIFEST.sha256: {len(rows)} artifacts")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
