# ABSOLUTE PATH: reports/2026-11-09_audit/precision_check.py
"""
Audit finding F4: do the value targets, computed on bfloat16-unrolled latents, differ from what the planner computes
on float32-unrolled latents?

DESIGN DECISIONS & THEORETICAL FOUNDATIONS
==============================================================================
# 1. Question (2026-11-10):
#    - training/trainer.py unrolls the online filter under bfloat16 autocast on CUDA (section 6) and casts the latents
#      to float32 for the Bellman targets. The planner unrolls the same kind of filter in float32. The trainer header
#      says the targets use "exactly the latents the online agent computes"; that holds only up to bf16 rounding
#      accumulated over up to 50 recurrent steps.
#    - Measured here on the whole-buffer-retrain recipe's final networks (runs/sweeps/wumpus_retrain/seed{0,1,2},
#      latest.pt, the ONLINE filter and heads the trainer uses), on stored buffer episodes (the training data):
#        (a) |z_bf16 - z_fp32| per step (max, mean, relative to |z|), by time step;
#        (b) the one-step backup Q(z, a) = R + gamma sum_o P V-bar through the trainer's own target search model,
#            evaluated on both latents: mean |delta target| and the share of latents whose argmax action differs.
#    - Post-game latents (after GAME_OVER) are excluded, as in the audit's Appendix M.
# 2. Decision rule (written before running): F4 matters if the argmax share differs on > 1% of live latents or the
#    mean |delta target| exceeds 1 (a seventh of the idle-vs-step margin ~7). Otherwise F4 is closed as negligible
#    and no code changes.
# 3. Cost: needs CUDA for faithful bf16 numerics (CPU autocast rounds differently); < 1 GB VRAM, ~1 min per seed.
#    Run only through the week-queue gate conditions (no runs/week/PAUSE, GPU idle, no ipomcp R-final job).
# 4. RESULT (2026-11-10, RTX 5080, 512 buffer episodes per seed; seeds 0 / 1 / 2): F4 NEGLIGIBLE, CLOSED, no code change.
#    - |z_bf16 - z_fp32|: mean 0.033 / 0.034 / 0.035 (relative 0.42-0.44%), max 0.47 / 0.23 / 0.25; it grows along the
#      episode (mean 0.011 at t = 0, 0.024 at t = 10, 0.035 at t = 25, 0.055-0.058 at t = 50).
#    - Backup through the trainer's target model: mean |delta target| 0.15 / 0.11 / 0.14, p99 1.6 / 1.2 / 1.5, max 9.4 / 4.9 /
#      13.5; argmax action differs on 0.26% / 0.31% / 0.39% of live latents (bar 1%), against a median top-2 gap of
#      4.6 / 2.5 / 4.2. Both are under the registered bars; the rare large deviations sit at the episode's end.
# ==============================================================================
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ipomdp.experiments.runs import build_training_run
from ipomdp.training.checkpointing import load_checkpoint

GAME_OVER = 32
EPISODES = 512
CHUNK = 64          # episodes per unroll; the expansion is chunked by rows below
ROWS = 2048


def main(seeds: list[int]) -> None:
    dev = torch.device("cuda")
    for seed in seeds:
        rd = Path(f"runs/sweeps/wumpus_retrain/seed{seed}")
        cfg = OmegaConf.load(rd / ".hydra" / "config.yaml")
        setup, run = build_training_run(cfg, dev)
        st = load_checkpoint(rd / "checkpoints" / "latest.pt")
        for name, net in run.networks.items():
            net.load_state_dict(st["networks"][name])
        b = st["buffer"]
        n = b["size"]
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(n, generator=g)[:EPISODES]
        acts, obs, init = b["actions"][idx].to(dev), b["observations"][idx].to(dev), b["initial_observations"][idx].to(dev)
        post = torch.cat([init[:, None], obs[:, :-1]], 1) == GAME_OVER          # latent z_t follows GAME_OVER
        live = torch.cat([~post, (obs[:, -1:] != GAME_OVER)], 1)                 # (B, T + 1): z_0..z_T
        filt = run.belief_filter.eval()
        zs = {}
        with torch.no_grad():
            for prec in ("bf16", "fp32"):
                out = []
                for s in range(0, EPISODES, CHUNK):
                    A = F.one_hot(acts[s:s + CHUNK], setup.domain.num_actions).float()
                    O = F.one_hot(obs[s:s + CHUNK], setup.domain.num_observations).float()
                    I = F.one_hot(init[s:s + CHUNK], setup.domain.num_observations).float()
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=prec == "bf16"):
                        out.append(filt.unroll(A, O, I).float())
                zs[prec] = torch.cat(out)
        diff = (zs["bf16"] - zs["fp32"]).norm(dim=-1)                              # (B, T + 1)
        rel = diff / zs["fp32"].norm(dim=-1).clamp(min=1e-6)
        steps = diff.shape[1]
        by_t = [float(diff[:, t][live[:, t]].mean()) for t in (0, 10, 25, steps - 1)]
        print(f"seed {seed}: live latents {int(live.sum())}; |dz| mean {diff[live].mean():.4f} max {diff[live].max():.4f}; "
              f"relative mean {rel[live].mean():.5f} max {rel[live].max():.5f}; mean |dz| at t=0/10/25/{steps - 1}: "
              + " / ".join(f"{v:.4f}" for v in by_t))
        q = {}
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            for prec in ("bf16", "fp32"):
                z = zs[prec][live]
                parts = []
                for s in range(0, len(z), ROWS):
                    e = run.trainer.target_search.expand(z[s:s + ROWS])
                    parts.append(e.rewards + run.trainer.discount * (e.observation_probs * e.next_values).sum(-1))
                q[prec] = torch.cat(parts)
        dt = (q["bf16"].max(-1).values - q["fp32"].max(-1).values).abs()
        flips = (q["bf16"].argmax(-1) != q["fp32"].argmax(-1)).float()
        gap = q["fp32"].topk(2, -1).values
        print(f"  backup: mean |d target| {dt.mean():.4f} p99 {dt.quantile(.99):.3f} max {dt.max():.3f}; "
              f"argmax flips {flips.mean():.4%} ({int(flips.sum())}); median fp32 top-2 gap {(gap[:, 0] - gap[:, 1]).median():.3f}")


if __name__ == "__main__":
    main([int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2").split(",")])
