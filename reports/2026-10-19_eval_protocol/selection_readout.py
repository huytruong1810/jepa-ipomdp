# ABSOLUTE PATH: reports/2026-10-19_eval_protocol/selection_readout.py
# ==============================================================================
# CLAIM C-d READOUT: FINAL-RETRAIN NETWORKS AGAINST best.pt, PAIRED PER GAME
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (PROTOCOL.md, C-d):
#    - Per training seed, two analyses of the same run on the same games: <sweep>/seed<k>/<base>/
#      games.npz (best.pt) and <sweep>/seed<k>/<arm>/games.npz (final_retrain.pt). The quantity is
#      the per-game difference arm minus base, with the hierarchical bootstrap of
#      experiments/aggregate.py, section 6 (seeds, then games), the interval cited for
#      confirmatory claims.
#
# 2. Usage (CPU only; nothing is played):
#      CUDA_VISIBLE_DEVICES= uv run reports/2026-10-19_eval_protocol/selection_readout.py \
#          runs/sweeps/wumpus_retrain_heldout 5,6,7,8,9 analysis_heldout analysis_heldout_final
#    Prints, per seed, the mean return and death and unfinished rates of both arms, their paired
#    difference with its standard error, and the selected collections. Then it prints the pooled
#    hierarchical-bootstrap intervals and the registered verdict.
#
# 3. Checks:
#    - Pairing: the heuristic's per-game returns must be equal in both files (aggregate.py,
#      _check_pairing), so both arms played the same layouts.
#    - Seeds whose two analyses read the same collection must agree game for game (C-d: a
#      mismatch voids the read); the script raises otherwise.
# ==============================================================================

import json
from pathlib import Path
import sys

import numpy as np

from ipomdp.experiments import hierarchical_bootstrap


def outcomes(games: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    won, died = games["learned_planner.won"], games["learned_planner.died"]
    return {"return": games["learned_planner.return"], "death": died.astype(np.float64),
            "unfinished": (~won & ~died).astype(np.float64)}


def main() -> None:
    sweep, seeds, base, arm = Path(sys.argv[1]), [int(s) for s in sys.argv[2].split(",")], sys.argv[3], sys.argv[4]
    rows: dict[str, list[np.ndarray]] = {}
    for seed in seeds:
        pair = []
        for output in (base, arm):
            with np.load(sweep / f"seed{seed}" / output / "games.npz") as data:
                games = dict(data)
            collection = json.loads((sweep / f"seed{seed}" / output / "report.json").read_text())["checkpoint_collection"]
            pair.append((games, collection))
        (g0, c0), (g1, c1) = pair
        if not np.array_equal(g0["reference_risk0.4.return"], g1["reference_risk0.4.return"]):
            raise ValueError(f"seed {seed}: the two analyses did not play the same games.")
        if c0 == c1 and not np.array_equal(g0["learned_planner.return"], g1["learned_planner.return"]):
            raise ValueError(f"seed {seed}: both analyses read collection {c0} but their games differ (C-d voids the read).")
        o0, o1 = outcomes(g0), outcomes(g1)
        diff = o1["return"] - o0["return"]
        print(f"seed {seed}: collections {c0} -> {c1} | return {o0['return'].mean():.1f} -> {o1['return'].mean():.1f}, "
              f"paired {diff.mean():+.1f} +- {diff.std(ddof=1) / np.sqrt(len(diff)):.1f} | deaths "
              f"{100 * o0['death'].mean():.1f} -> {100 * o1['death'].mean():.1f}% | unfinished "
              f"{100 * o0['unfinished'].mean():.1f} -> {100 * o1['unfinished'].mean():.1f}%")
        for key in o0:
            rows.setdefault(f"{key}/{base}", []).append(o0[key])
            rows.setdefault(f"{key}/{arm}", []).append(o1[key])
        rows.setdefault(f"gap/{arm} - {base}", []).append(diff)
    for name, values in rows.items():
        stat = hierarchical_bootstrap(np.stack(values))
        print(f"{name}: {stat.mean:.3f} [{stat.ci95[0]:.3f}, {stat.ci95[1]:.3f}]")
    gap = hierarchical_bootstrap(np.stack(rows[f"gap/{arm} - {base}"]))
    verdict = "PASS" if gap.ci95[0] > 0 else "FAIL" if gap.ci95[1] < 0 else "UNDECIDED (extend to seeds 10-14)"
    print(f"C-d verdict: {verdict}")


if __name__ == "__main__":
    main()
