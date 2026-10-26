# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/go_value_diag.py
# ==============================================================================
# DO THE ANCHORED RETRAINS FIT V = 0 ON THE STORED POST-GAME LATENTS? (objective_ab.py 2n, MECHANISM follow-up)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-26): objective_ab.py 2n's MECHANISM clause read the learned V_hat of the imagined GAME_OVER
#    child (z after GRAB with GLITTER, GAME_OVER) through phantom_reward_diag.py --networks, and found +18 to +233 on
#    the strong-weight-decay anchored sets (ctlA: +0.1 to +0.2). Under model.absorbing_value the planner and the
#    targets never read that value, so the clause may have measured a quantity that does not reach decisions. This
#    script reads what the value loss actually fits: the latents of the training buffer's own games (the last 1024
#    stored games of the run), split into the first GAME_OVER step of a game, later GAME_OVER steps and live steps,
#    with V_hat, max_a r_hat, the anchored double-value target and |V_hat - target| on each.
#    - Fit on the stored post-game latents (|V_hat - target| small, V_hat ~ 0): the anchor works where it is
#      trained; the imagined child is off the data and unused under the anchor -> the MECHANISM clause measured the
#      wrong quantity; state the deviation and go to the in-agent screen.
#    - No fit there: the value does not learn the anchored targets on post-game states (e.g. weight decay 1 holds
#      the head away from them); diagnose further before any in-agent run.
#
# 2. Memory (the reason this file exists): an unchunked version expanded ~51,000 latents at once (8.4 M child latents
#    of dimension 128 plus one-hot inputs and GRU intermediates, twice) and crashed the WSL2 VM twice on 2026-10-26
#    (dxgkio_make_resident ENOMEM; the VM dies instead of raising a CUDA OOM). Every expansion here is chunked
#    (CHUNK latents at a time), and the buffer is read once on the CPU. The process is also capped at
#    MEMORY_FRACTION of the device through the caching allocator, so an overshoot raises a CUDA OOM before
#    WSL2 is asked for the memory, and the peak allocation is printed after each saved set.
#
# 3. Usage: uv run python <this> <run dir> <saved .pt> [<saved .pt> ...]
#    e.g. runs/sweeps/wumpus_retrain/seed1 runs/wumpus/absorbing_train/{ctlA_seed1,wdA_seed1_u12800,wdA_seed1_u25600}.pt
#
# 4. Result (2026-10-26; runs/wumpus/absorbing_train/go_value_seed{0,1,2}.log; peak allocation 1.50 GiB per set):
#    SPLIT, a case the binary reading of section 1 did not anticipate. Means over the last 1024 stored games:
#        seed  set           first GAME_OVER (n ~320)   later GAME_OVER (n ~9-10k)   live (n ~41k)
#                            V_hat  target  |gap|        V_hat  |gap|                 |gap|   V>999
#        0     ctlA            0.00   0.00    0.00        0.00   0.00                  6.89   0.000
#              wdA_u12800     37.03   0.14   36.89        3.71   3.71                  5.73   0.000
#              wdA_u25600     29.05   0.19   28.86        0.09   0.09                  5.30   0.003
#        1     ctlA            0.12   0.00    0.11        0.00   0.00                  5.16   0.008
#              wdA_u12800     96.62   0.64   95.98        0.43   0.42                  5.70   0.000
#              wdA_u25600    151.72   1.24  150.48        0.90   0.86                  5.79   0.003
#        2     ctlA            0.13   0.02    0.12        0.00   0.00                  6.10   0.012
#              wdA_u12800     60.98   0.20   60.77        1.43   1.42                  5.11   0.000
#              wdA_u25600     10.89   0.18   10.70        0.35   0.34                  4.99   0.000
#    - The control fits V = 0 on every post-game latent. The strong-weight-decay sets fit the later GAME_OVER latents
#      (|gap| <= 1.4) and the live latents (|gap| ~5 on values ~500) but NOT the first GAME_OVER latent of a game
#      (+11 to +152 against targets <= 1.3): the latent one filter step past a live state, i.e. the same quantity
#      the 2n MECHANISM clause read on the imagined GRAB child. Reading: under weight decay 1 the value head is too
#      smooth to separate that latent from its live predecessor (V ~ 500), and it is only ~0.6% of the samples.
#    - Decision relevance: with model.absorbing_value every value the recipe reads behind a GAME_OVER percept is
#      replaced by 0 (trainer.py 3f: _pair_value_targets_and_children and the target_search expansion; the acting
#      planners through AbsorbingValueModel); the only unanchored bootstrap, nstep_returns, is off in the recipe
#      (model.value_nstep). So the misfit is on states whose learned value no target or decision reads; without
#      the anchor it is exactly what collapsed objective_ab.py 2m seed 2.
# ==============================================================================

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ipomdp.domain.wumpus import GAME_OVER
from ipomdp.experiments.runs import build_training_run
from ipomdp.training import load_checkpoint

sys.path.insert(0, str(Path(__file__).parent))
from saved_networks import in_layout  # noqa: E402  (same diagnostics directory)

CHUNK = 1024          # latents per expansion: 1024 x 5 x 33 children of dimension 128, well under 1 GB
GAMES = 1024          # the last stored games of the run
MEMORY_FRACTION = 0.25  # of the 16 GB device: ~4 GB, a few times the expected ~1-2 GB peak

run_dir, dev = Path(sys.argv[1]), torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, 0)
cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
cfg.model.absorbing_value = True                                   # the anchored targets, as trained in 2n
setup, run = build_training_run(cfg, dev)
buf = load_checkpoint(run_dir / "checkpoints" / "latest.pt")["buffer"]
size = buf["size"]
actions, observations, initial = (buf[k][size - GAMES:size] for k in ("actions", "observations", "initial_observations"))
del buf
num_a, num_o = setup.domain.num_actions, setup.domain.num_observations
go = observations == GAME_OVER
first = go & ~torch.cat([torch.zeros_like(go[:, :1]), go[:, :-1]], 1)
masks = {"first GAME_OVER": first, "later GAME_OVER": go & ~first, "live": ~go}
print(f"== {run_dir.name}: last {GAMES} stored games; steps per split "
      f"{ {k: int(m.sum()) for k, m in masks.items()} }", flush=True)


@torch.no_grad()
def latents_of_games() -> torch.Tensor:
    """z_1..z_T of every stored game (CPU), unrolled CHUNK games at a time with the loaded filter."""
    out = []
    for i in range(0, GAMES, CHUNK):
        a = F.one_hot(actions[i:i + CHUNK].to(dev), num_a).float()
        o = F.one_hot(observations[i:i + CHUNK].to(dev), num_o).float()
        o0 = F.one_hot(initial[i:i + CHUNK].to(dev), num_o).float()
        out.append(run.trainer.belief_filter.unroll(a, o, o0)[:, 1:].float().cpu())
    return torch.cat(out)


@torch.no_grad()
def readout(z: torch.Tensor) -> dict[str, float]:
    """Means of V_hat, max_a r_hat, the anchored target and |V_hat - target| over z (N, D), CHUNK latents at a time."""
    sums = dict(v=0.0, r=0.0, t=0.0, gap=0.0, over=0.0)
    tr = run.trainer
    for i in range(0, len(z), CHUNK):
        zz = z[i:i + CHUNK].to(dev)
        v = tr.target_model.values(zz)
        t = tr.pair_value_targets(zz)[0]
        r = tr.target_model.expand(zz).rewards.max(-1).values
        for key, x in (("v", v), ("r", r), ("t", t), ("gap", (v - t).abs()), ("over", (v > 999).float())):
            sums[key] += float(x.sum())
        del zz, v, t, r
    return {k: s / len(z) for k, s in sums.items()}


for path in sys.argv[2:]:
    states = in_layout(torch.load(path, map_location=dev), run.networks)       # saved_networks.py
    for name, net in run.networks.items():
        net.load_state_dict(states[name])
    z = latents_of_games()
    for label, m in masks.items():
        if not bool(m.any()):
            print(f"  {Path(path).stem:22s} {label:16s} none", flush=True)
            continue
        s = readout(z[m])
        print(f"  {Path(path).stem:22s} {label:16s} n={int(m.sum()):6d} V_hat {s['v']:8.2f} max_a r_hat {s['r']:8.2f} "
              f"anchored target {s['t']:8.2f} |V - target| {s['gap']:7.2f} V>999 {s['over']:.3f}", flush=True)
    del z
    print(f"  peak device memory {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB", flush=True)
    torch.cuda.empty_cache()
print("DONE", flush=True)
