# ABSOLUTE PATH: reports/2026-10-19_eval_protocol/ablation_readout.py
# ==============================================================================
# CLAIM C-e READOUT: THE RECIPE AGAINST EACH LEAVE-ONE-OUT ARM, PAIRED PER GAME
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (PROTOCOL.md, C-e): does each design decision of the frozen recipe (double value, whole-buffer
#    retraining, decoder representation) contribute on held-out training seeds? Per training seed k, the recipe
#    (<base sweep>/seed<k>/analysis_heldout/games.npz) and the arm (<arm sweep>/seed<k>/analysis_heldout/games.npz)
#    played the same 512 layouts (analysis seed 20261019). The quantity is the per-game difference recipe minus
#    arm, with the hierarchical bootstrap of experiments/aggregate.py, section 6 (seeds, then games): the interval
#    PROTOCOL.md cites for confirmatory claims.
#
# 2. Usage (CPU only; nothing is played):
#      CUDA_VISIBLE_DEVICES= uv run reports/2026-10-19_eval_protocol/ablation_readout.py \
#          runs/sweeps/wumpus_retrain_heldout runs/sweeps/abl_max 5,6,7,8,9
#    Prints per seed both arms' return, death and unfinished rates and the paired difference with its standard
#    error; then the pooled hierarchical-bootstrap intervals, the C-b probes of both sweeps and the registered
#    verdict. Seeds the arm has not finished are listed and the verdict is withheld (C-e "Order and stopping").
#
# 3. Checks: pairing (the heuristic's per-game returns must be equal in both files, so both played the same
#    layouts); both analyses on analysis seed 20261019 (report.json settings).
# ==============================================================================

import json
import sys
from pathlib import Path

import numpy as np

from ipomdp.experiments import hierarchical_bootstrap

OUTPUT = "analysis_heldout"
HELD_OUT_SEED = 20261019


def load(sweep: Path, seed: int) -> tuple[dict[str, np.ndarray], dict] | None:
    folder = sweep / f"seed{seed}" / OUTPUT
    if not (folder / "games.npz").exists():
        return None
    report = json.loads((folder / "report.json").read_text())
    if report["settings"]["seed"] != HELD_OUT_SEED:
        raise ValueError(f"{folder} was not analysed on the held-out analysis seed.")
    with np.load(folder / "games.npz") as data:
        return dict(data), report


def outcomes(games: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    won, died = games["learned_planner.won"], games["learned_planner.died"]
    return {"return": games["learned_planner.return"], "death": died.astype(np.float64),
            "unfinished": (~won & ~died).astype(np.float64)}


def main() -> None:
    base_sweep, arm_sweep = Path(sys.argv[1]), Path(sys.argv[2])
    seeds = [int(s) for s in sys.argv[3].split(",")]
    rows: dict[str, list[np.ndarray]] = {}
    missing, probes = [], {"recipe": [], "arm": []}
    for seed in seeds:
        base, arm = load(base_sweep, seed), load(arm_sweep, seed)
        if base is None:
            raise FileNotFoundError(f"the recipe has no held-out analysis for seed {seed}.")
        if arm is None:
            missing.append(seed)
            continue
        (g0, r0), (g1, r1) = base, arm
        if not np.array_equal(g0["reference_risk0.4.return"], g1["reference_risk0.4.return"]):
            raise ValueError(f"seed {seed}: the two analyses did not play the same games.")
        o0, o1 = outcomes(g0), outcomes(g1)
        diff = o0["return"] - o1["return"]
        print(f"seed {seed}: recipe {o0['return'].mean():6.1f} (deaths {o0['death'].mean():.1%}, unfinished "
              f"{o0['unfinished'].mean():.1%}) | arm {o1['return'].mean():6.1f} (deaths {o1['death'].mean():.1%}, "
              f"unfinished {o1['unfinished'].mean():.1%}) | recipe - arm {diff.mean():+6.1f} +- "
              f"{diff.std(ddof=1) / np.sqrt(diff.size):.1f}", flush=True)
        for key, value in (("difference", diff), ("recipe", o0["return"]), ("arm", o1["return"]),
                           ("recipe deaths", o0["death"]), ("arm deaths", o1["death"]),
                           ("recipe unfinished", o0["unfinished"]), ("arm unfinished", o1["unfinished"])):
            rows.setdefault(key, []).append(value)
        probes["recipe"].append(r0["probes"])
        probes["arm"].append(r1["probes"])
    if not rows:
        print(f"no finished arm seeds yet (missing {missing})")
        return
    print(f"\nhierarchical bootstrap over {len(rows['difference'])} seeds, then games:")
    for key, values in rows.items():
        stat = hierarchical_bootstrap(np.stack(values))
        mean, (low, high) = stat.mean, stat.ci95
        print(f"  {key:18s} {mean:8.3f} [{low:8.3f}, {high:8.3f}]")
    for name, reports in probes.items():
        print(f"  probes ({name}): ", end="")
        flat = {}
        for report in reports:
            for split, values in report.items():
                if isinstance(values, dict):
                    for metric, value in values.items():
                        if isinstance(value, (int, float)):
                            flat.setdefault(f"{split}.{metric}", []).append(value)
        print(", ".join(f"{k} {np.mean(v):.3f}" for k, v in sorted(flat.items())))
    if missing:
        print(f"\nINCOMPLETE: arm seeds {missing} not analysed; no verdict (PROTOCOL.md C-e).")
        return
    low, high = hierarchical_bootstrap(np.stack(rows["difference"])).ci95
    verdict = "CONTRIBUTES" if low > 0 else "HURTS" if high < 0 else "NO DETECTABLE CONTRIBUTION"
    print(f"\nC-e verdict for {arm_sweep.name}: {verdict}")


if __name__ == "__main__":
    main()
