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
#    (CHUNK latents at a time), and the buffer is read once on the CPU.
#
# 3. Usage: uv run python <this> <run dir> <saved .pt> [<saved .pt> ...]
#    e.g. runs/sweeps/wumpus_retrain/seed1 runs/wumpus/absorbing_train/{ctlA_seed1,wdA_seed1_u12800,wdA_seed1_u25600}.pt
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

run_dir, dev = Path(sys.argv[1]), torch.device("cuda")
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
    torch.cuda.empty_cache()
print("DONE", flush=True)
