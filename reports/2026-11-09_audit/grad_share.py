# ABSOLUTE PATH: reports/2026-11-09_audit/grad_share.py
"""
Audit finding F5: how much of the filter's gradient, and of the joint clipped norm, comes from each loss term?

DESIGN DECISIONS & THEORETICAL FOUNDATIONS
==============================================================================
# 1. Question (2026-11-10):
#    - training/trainer.py sums every loss and clips the gradient of all parameters (filter and every head) with ONE
#      norm (grad_clip_norm = 1). The value and reward heads read detached latents in the JEPA representation, but in
#      the decoder representation (the Wumpus recipe) the observation and multi-step terms reach the filter, and every
#      head's gradient enters the joint norm: when one head's gradient is large, the clip shrinks the filter's step.
#    - Measured on the recipe's final networks (runs/sweeps/wumpus_retrain/seed{0,1,2}, latest.pt, online networks),
#      on 16 batches of the recipe's batch size sampled from the stored buffer, per loss term L_k:
#        (a) ||grad_filter L_k|| and its share of ||grad_filter sum_k L_k||^2 components;
#        (b) ||grad_all L_k||, and the joint norm ||grad_all sum_k L_k|| against the clip (1.0): the factor
#            min(1, 1 / norm) by which the clip shrinks every update.
#    - The loss terms are captured without changing src/: the script replaces the builtin `sum` in the trainer's
#      module namespace, reads the losses dict through the dict_values view's .mapping when train_step sums it, and aborts
#      before backward, so no
#      optimiser step, EMA or target update happens.
# 2. Reading (descriptive, no screen hangs on it): the share tells the S0 screen and Stage B whether a per-group clip
#    would change the filter's step. A term holding > 50% of the joint norm while not reaching the filter
#    would be the coupling the audit suspected.
# 3. Cost: CPU only (~2-3 min per seed); float32 (no autocast on CPU), so magnitudes are those of the fp32 graph.
# 4. RESULT (2026-11-10, CPU, 16 batches of 16 episodes per seed; seeds 0 / 1 / 2): F5 CONFIRMED.
#    - Mean ||grad|| into the filter: observation 0.17 / 0.13 / 0.16, multi-step observation 0.24 / 0.21 / 0.26,
#      reward 0.030 / 0.026 / 0.039 (the reward head reads the undetached latent in the decoder recipe), value 0.
#    - Mean ||grad|| over all parameters: value 1.44 / 1.13 / 1.36 against a joint norm of 1.60 / 1.32 / 1.55. The value
#      heads' parameters are disjoint from the rest, so their squared norms add: the value term holds 81% / 72% / 77% of
#      the joint squared norm while sending nothing to the filter.
#    - The joint clip (1.0) is active on almost every batch (joint norm min 1.04 / 0.92 / 1.02) and scales every update,
#      the filter's included, by 0.65 / 0.77 / 0.73 on average: the value loss sets the filter's effective step size.
#    - Consequence: a separate clip for the value heads (as the detached planning head already has, trainer.py 3d)
#      would raise the filter's and observation heads' steps by ~1.3-1.5x. Added as a candidate S0 arm; it changes the
#      effective learning rate, so it is screened like any other arm, not adopted as a fix.
# ==============================================================================
"""

import builtins
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

import ipomdp.training.trainer as trainer_module
from ipomdp.experiments.runs import build_training_run
from ipomdp.training.checkpointing import load_checkpoint

BATCHES = 16


class Captured(Exception):
    """Raised by the patched sum once the losses dict has been seen."""

    def __init__(self, losses: dict):
        super().__init__("captured")
        self.losses = losses


def main(seeds: list[int]) -> None:
    torch.set_num_threads(4)
    dev = torch.device("cpu")
    for seed in seeds:
        rd = Path(f"runs/sweeps/wumpus_retrain/seed{seed}")
        cfg = OmegaConf.load(rd / ".hydra" / "config.yaml")
        setup, run = build_training_run(cfg, dev)
        st = load_checkpoint(rd / "checkpoints" / "latest.pt")
        for name, net in run.networks.items():
            net.load_state_dict(st["networks"][name])
        stored = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in st["buffer"].items()}
        stored["generator"] = run.buffer.state_dict()["generator"]      # the stored stream is a CUDA one
        run.buffer.load_state_dict(stored)
        trainer = run.trainer
        filter_params = list(trainer.belief_filter.parameters())
        all_params = [p for p in trainer.parameters if p.requires_grad]

        def capture_sum(values, *args):
            if type(values).__name__ == "dict_values":                 # sum(losses.values()) in train_step
                raise Captured(dict(values.mapping))
            return builtins.sum(values, *args)

        torch.manual_seed(seed)
        rows = []
        for _ in range(BATCHES):
            batch = run.buffer.sample(cfg.training.batch_size)
            trainer_module.sum = capture_sum
            try:
                trainer.train_step(batch)
                raise RuntimeError("train_step finished without summing its losses dict.")
            except Captured as c:
                losses = c.losses
            finally:
                del trainer_module.sum
            total = builtins.sum(losses.values())
            row = {}
            for name, loss in list(losses.items()) + [("TOTAL", total)]:
                gf = torch.autograd.grad(loss, filter_params, retain_graph=True, allow_unused=True)
                ga = torch.autograd.grad(loss, all_params, retain_graph=True, allow_unused=True)
                nf = torch.sqrt(builtins.sum((g.double() ** 2).sum() for g in gf if g is not None) + torch.zeros((), dtype=torch.float64))
                na = torch.sqrt(builtins.sum((g.double() ** 2).sum() for g in ga if g is not None) + torch.zeros((), dtype=torch.float64))
                row[name] = (float(nf), float(na))
            rows.append(row)
        names = list(rows[0])
        print(f"seed {seed} ({cfg.model.representation}, {cfg.model.value_estimator}; {BATCHES} batches of "
              f"{cfg.training.batch_size}): mean ||grad|| into the filter / into all parameters")
        for name in names:
            f = torch.tensor([r[name][0] for r in rows])
            a = torch.tensor([r[name][1] for r in rows])
            print(f"  {name:24s} filter {f.mean():9.4f}   all {a.mean():9.4f}")
        joint = torch.tensor([r["TOTAL"][1] for r in rows])
        print(f"  joint norm mean {joint.mean():.3f} (min {joint.min():.3f}, max {joint.max():.3f}); clip 1.0 scales "
              f"updates by {torch.clamp(1.0 / joint, max=1.0).mean():.3f} on average")


if __name__ == "__main__":
    main([int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2").split(",")])
