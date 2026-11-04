# ABSOLUTE PATH: reports/2026-10-19_eval_protocol/search_budget_readout.py
# ==============================================================================
# READOUT OF PROTOCOL.md C-g (SEARCH BUDGET OF THE LEARNED PLANNER), READ ONCE
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Usage: CUDA_VISIBLE_DEVICES= uv run reports/2026-10-19_eval_protocol/search_budget_readout.py
#    - Reads, for held-out training seeds 5-9 of runs/sweeps/wumpus_retrain_heldout, the recipe's own 50-simulation
#      games (analysis_heldout/games.npz, learned_planner.return: C-a) and the 10- and 100-simulation plays
#      (analysis_heldout/exact_play_learned_sims<N>.npz, written by exact_model_play.py --sims=<N>).
#    - The plays use the same simulator stream (stream_seed(20261019, 1)), so the rows are the same games; this is
#      checked by the C-a pairing check: the 50-simulation replay of exact_model_play.py reproduces report.json, and
#      the game outcome flags of every N are paired by row.
# 2. Output: per seed and pooled (hierarchical bootstrap over seeds, then games; experiments/aggregate.py section 6)
#    the return at each N, the paired differences N100 - N50 and N10 - N50, and win / death / unfinished rates; then
#    the registered reading (C-g): the upper end of the N100 - N50 interval against +84.
# ==============================================================================
from pathlib import Path

import numpy as np

from ipomdp.experiments import hierarchical_bootstrap

SWEEP, SEEDS, BAR = Path("runs/sweeps/wumpus_retrain_heldout"), (5, 6, 7, 8, 9), 84.0


def main() -> None:
    returns = {n: [] for n in (10, 50, 100)}
    for seed in SEEDS:
        out = SWEEP / f"seed{seed}" / "analysis_heldout"
        returns[50].append(np.load(out / "games.npz")["learned_planner.return"].astype(np.float64))
        line = f"seed {seed}: N50 {returns[50][-1].mean():6.1f}"
        for n in (10, 100):
            play = np.load(out / f"exact_play_learned_sims{n}.npz")
            returns[n].append(play["returns"].astype(np.float64))
            if returns[n][-1].shape != returns[50][-1].shape:
                raise ValueError(f"seed {seed} N{n}: {returns[n][-1].shape} games vs {returns[50][-1].shape}")
            won, died, unfinished = (float(play[k]) for k in ("win_rate", "death_rate", "unfinished_rate"))  # game_outcomes
            line += (f" | N{n} {returns[n][-1].mean():6.1f} (d {returns[n][-1].mean() - returns[50][-1].mean():+6.1f};"
                     f" won {won:.3f} died {died:.3f} unfinished {unfinished:.3f})")
        print(line)
    print("\nhierarchical bootstrap over 5 seeds, then games:")
    for name, per_seed in (("N10", returns[10]), ("N50", returns[50]), ("N100", returns[100]),
                           ("N100 - N50", [a - b for a, b in zip(returns[100], returns[50])]),
                           ("N10 - N50", [a - b for a, b in zip(returns[10], returns[50])])):
        stat = hierarchical_bootstrap(np.stack(per_seed))
        print(f"  {name:11s} {stat.mean:8.1f} [{stat.ci95[0]:7.1f}, {stat.ci95[1]:7.1f}]")
        if name == "N100 - N50":
            upper = stat.ci95[1]
    print(f"\nC-g reading: N100 - N50 upper end {upper:+.1f} vs bar +{BAR:.0f}: "
          + ("SEARCH IS NOT THE LEVER (predicted)" if upper < BAR else "SEARCH BUDGET IS A SUBSTANTIAL LEVER"))


if __name__ == "__main__":
    main()
