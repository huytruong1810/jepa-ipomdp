# ABSOLUTE PATH: reports/2026-10-19_eval_protocol/final_retrain_checkpoint.py
# ==============================================================================
# THE FINAL-RETRAIN SELECTION OF A RUN TRAINED BEFORE training.checkpoint_selection EXISTED
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question:
#    - Claim C-d (PROTOCOL.md) compares the networks of a run's final whole-buffer retrain with its
#      best.pt. A run trained with training.checkpoint_selection=final_retrain writes that file
#      itself (main.py, section 5). The held-out recipe runs predate the option. Their latest.pt
#      holds the same networks (nothing trains after the last collection's retrain and
#      evaluation), so this script writes them as checkpoints/final_retrain.pt in the payload of a
#      selected checkpoint (collection, eval_return, networks; experiments/runs.py,
#      load_trained_run).
#
# 2. Usage:
#      uv run reports/2026-10-19_eval_protocol/final_retrain_checkpoint.py <run dir> [<run dir> ...]
#    It refuses a run whose latest.pt is not at the last collection, a last collection that is
#    not a retrain, or a missing evaluation of it (the same conditions as
#    training/checkpointing.py, check_checkpoint_selection), and it never overwrites a file.
#    - eval_return is the collection's 128-game evaluation as main.log prints it (two decimals).
#      It is a label only: the analysis replays the networks on its own games.
# ==============================================================================

from pathlib import Path
import re
import sys

from omegaconf import OmegaConf

from ipomdp.training import CheckpointSelection, check_checkpoint_selection, load_checkpoint, save_checkpoint


def main() -> None:
    for run_dir in map(Path, sys.argv[1:]):
        target = run_dir / "checkpoints" / "final_retrain.pt"
        if target.exists():
            raise FileExistsError(f"{target} exists already.")
        cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
        t = cfg.training
        total = t.total_episodes // t.env_batch_size
        check_checkpoint_selection(CheckpointSelection.FINAL_RETRAIN, total, t.eval_every, t.retrain_every)
        latest = load_checkpoint(run_dir / "checkpoints" / "latest.pt")
        if latest["collection"] != total:
            raise ValueError(f"{run_dir}: latest.pt is at collection {latest['collection']}, not the last ({total}).")
        lines = re.findall(rf"collection {total}: greedy eval discounted return (-?[0-9.]+)",
                           (run_dir / "main.log").read_text())
        if len(lines) != 1:
            raise ValueError(f"{run_dir}: expected one evaluation line of collection {total} in main.log, found {len(lines)}.")
        save_checkpoint(target, {"collection": total, "eval_return": float(lines[0]), "networks": latest["networks"]})
        print(f"{target}: collection {total}, eval_return {float(lines[0])}")


if __name__ == "__main__":
    main()
