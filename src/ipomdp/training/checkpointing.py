# ABSOLUTE PATH: src/ipomdp/training/checkpointing.py
# ==============================================================================
# ATOMIC CHECKPOINT FILES
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. What Is Saved Is Decided by the Owner of the State:
#    - TrainingRun.state_dict() defines the complete run state (networks, optimiser, replay
#      buffer, simulators, planners, agents, all random generators). This module only writes
#      and reads such dictionaries; it has no knowledge of their contents.
#
# 2. Two Files per Run Directory:
#      latest.pt  full run state, written every save_every collections and on SIGINT; the file
#                 to pass as `resume=` to continue a run bit-for-bit.
#      best.pt    networks + collection index of the best GREEDY EVALUATION return so far.
#    - An earlier version selected "best" by training loss, which says nothing about the
#      quality of the policy in an RL loop.
#
# 2b. Which Networks best.pt Holds (CheckpointSelection; main.py, section 5):
#    - BEST_EVAL (the default, canonical Tiger): the networks of the highest greedy evaluation
#      so far (ties go to the later collection).
#    - FINAL_RETRAIN (whole-buffer retrain recipes only, training/run.py section 14): the
#      networks of the final retrain, i.e. the evaluation of the last collection. best_eval.pt
#      then holds what BEST_EVAL would have selected, so one run answers a paired comparison of
#      the two rules. best.pt keeps the same payload either way (collection, eval_return,
#      networks), so every reader of best.pt is unchanged.
#    - check_checkpoint_selection fails loudly when FINAL_RETRAIN cannot mean "the final
#      retrain": no retrain, a last collection that is not a retrain, or no evaluation of it.
#
# 3. Atomic Writes, Loud Reads:
#    - Files are written to a temporary name and renamed, so a crash never leaves a truncated
#      checkpoint. Loading a missing file raises; there is no silent "start from scratch".
#
# 4. Always Load on the CPU:
#    - Random-generator states are CPU ByteTensors and must stay there (torch.Generator and
#      torch.set_rng_state reject CUDA tensors). Every owner's load_state_dict moves its own
#      tensors to its device, so checkpoints are read with map_location="cpu".
# ==============================================================================

from enum import Enum
from pathlib import Path

import torch


class CheckpointSelection(str, Enum):
    """Which networks best.pt holds (section 2b)."""

    BEST_EVAL = "best_eval"
    FINAL_RETRAIN = "final_retrain"


def check_checkpoint_selection(selection: CheckpointSelection, total_collections: int, eval_every: int,
                               retrain_every: int) -> None:
    """Raises unless FINAL_RETRAIN selects an evaluated final retrain (section 2b)."""
    if selection is not CheckpointSelection.FINAL_RETRAIN:
        return
    if retrain_every <= 0 or total_collections % retrain_every != 0:
        raise ValueError(f"checkpoint_selection=final_retrain needs the last collection ({total_collections}) to be a "
                         f"whole-buffer retrain (retrain_every={retrain_every}; training/run.py section 14).")
    if total_collections % eval_every != 0:
        raise ValueError(f"checkpoint_selection=final_retrain needs an evaluation of the last collection "
                         f"({total_collections}; eval_every={eval_every}).")


def selects_best_eval(eval_return: float, best_eval_return: float) -> bool:
    """The BEST_EVAL rule: the evaluation ties or beats every earlier one (best_eval_return includes it)."""
    return eval_return >= best_eval_return


def selects_best(selection: CheckpointSelection, collection: int, total_collections: int, eval_return: float,
                 best_eval_return: float) -> bool:
    """Whether the networks just evaluated at `collection` become best.pt (section 2b)."""
    if selection is CheckpointSelection.BEST_EVAL:
        return selects_best_eval(eval_return, best_eval_return)
    return collection == total_collections


def save_checkpoint(path: Path, state: dict) -> None:
    """Atomically writes `state` to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict:
    """Reads a checkpoint written by save_checkpoint onto the CPU (raises if it does not exist)."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)
