# ABSOLUTE PATH: reports/2026-11-14_submission_audit/w5b_fatal_deadly.py
"""
Submission audit W5(b): of the recipe agents' fatal FORWARD steps on their 512 development analysis games, the share
taken into a cell whose exact danger is at least 0.999 (PROTOCOL.md C-l's finding, for the agents the paper reports).

Usage:
  python <this> export <seed>   best.pt networks of runs/sweeps/wumpus_retrain/seed<seed> -> runs/wumpus/w5b/
                                seed<seed>_best.pt, with the stored per-game returns (analysis/learned_planner_games.npz,
                                learned_planner.return) as <...>.returns.npy, so risk_localise.py's replay check
                                proves that the replayed games are the reported ones; a seed that stored only the mean
                                gets <...>.reference_mean.json instead (risk_localise.py header 6).
  python <this> read 0 1 2 3 4  per-seed shares from runs/wumpus/w5b/seed<seed>_best.risk.npz (written by
                                risk_localise.py), printed and saved to runs/wumpus/w5b/shares.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from ipomdp.training.checkpointing import load_checkpoint

OUT, DEADLY = Path("runs/wumpus/w5b"), 0.999


def export(seed: int) -> None:
    run = Path(f"runs/sweeps/wumpus_retrain/seed{seed}")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(load_checkpoint(run / "checkpoints" / "best.pt")["networks"], OUT / f"seed{seed}_best.pt")
    per_game = run / "analysis" / "learned_planner_games.npz"
    if per_game.exists():
        with np.load(per_game) as games:
            np.save(OUT / f"seed{seed}_best.returns.npy", games["learned_planner.return"])
    else:   # seeds 3-4 stored only the mean: risk_localise.py header 6 checks the replay's mean against it
        mean = json.loads((run / "analysis" / "report.json").read_text())["returns"]["learned_planner"]["discounted_return"]
        (OUT / f"seed{seed}_best.reference_mean.json").write_text(json.dumps({"discounted_return": mean}))
    print(f"exported seed {seed}")


def read(seeds: list[int]) -> None:
    out = {}
    for seed in seeds:
        rows = np.load(OUT / f"seed{seed}_best.risk.npz")
        fatal, danger = rows["fatal"], rows["danger"]
        n = int(fatal.sum())
        k = int((fatal & (danger >= DEADLY)).sum())
        out[seed] = {"fatal_forward": n, "into_provably_deadly": k, "share": k / n if n else float("nan"),
                     "deaths": int(fatal.any(1).sum())}
        print(f"seed {seed}: fatal FORWARDs {n}, into exact danger >= {DEADLY}: {k} "
              f"({out[seed]['share']:.3f}); games lost to a death {out[seed]['deaths']}")
    shares = [v["share"] for v in out.values() if v["fatal_forward"]]
    pooled_n = sum(v["fatal_forward"] for v in out.values())
    pooled_k = sum(v["into_provably_deadly"] for v in out.values())
    print(f"per-seed share mean {np.mean(shares):.3f} (range {min(shares):.3f}-{max(shares):.3f}); "
          f"pooled {pooled_k}/{pooled_n} = {pooled_k / pooled_n:.3f}")
    (OUT / "shares.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(int(sys.argv[2]))
    else:
        read([int(s) for s in sys.argv[2:]])
