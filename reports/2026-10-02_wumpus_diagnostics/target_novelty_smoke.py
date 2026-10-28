# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/target_novelty_smoke.py
# ==============================================================================
# DOES A PSEUDO-COUNT OF THE TURNED LATENT TARGET THE DECLINED "TURN, THEN STEP" DECISIONS? (DOSE SMOKE, PASS 2)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-28, after target_rule_smoke.py failed its selection on seed 0):
#    - The Q-gap of the declined macro carries no targeting signal: its median is +31.9 at the declined states and
#      +35.9 elsewhere, and at 3 trials per game it reaches 0.35 declined states per game (bar 1.3).
#    - Step D (feature_localise.py) located the model error at the TURNED latents of the declined states, off the
#      buffer's support: a turn-specific support shift. Count-based exploration (Bellemare et al. 2016; SimHash
#      counts, Tang et al. 2017) aims at exactly such pairs: trial TURN_x + FORWARD where the pair (turned latent,
#      FORWARD) is rare in the agent's own buffer. Non-privileged: it reads only the agent's filter, model and buffer.
#
# 2. Method (CPU only; usage: CUDA_VISIBLE_DEVICES= uv run python <this> <run dir> <smoke .npz> [<bits>=16]):
#    - The rows of target_rule_smoke.py (seed 0: every live non-FORWARD decision of the first 512 control games,
#      with the macro turn x* and the privileged labels). Those games are the first 512 of
#      runs/wumpus/coverage/games_control_<seed>.pt (asserted there), so the root latents are recomputed by the
#      acting filter on the stored histories.
#    - Turned latent: model.expand(z_t).next_states[x*, o*], o* = argmax_o P(o | z_t, x*) (the agent's own most
#      likely percept after the turn).
#    - Counts: SimHash code sign(A (z - mean)) with A ~ N(0, 1)^{bits x D} (fixed seed 0), mean over the buffer
#      latents. n_sa = number of buffer latents with the same code at which the agent played FORWARD (primary,
#      the state-action count of the pair the trial adds); n_s = same code, any action (reported).
#      Buffer = latest.pt's slots up to best.pt's collection (feature_localise.py section 2's reconstruction).
#    - Rule: trial when n_sa <= c, over a grid of c.
#
# 3. Pre-registered selection (written 2026-10-28 before this pass runs):
#    - The control stream holds only 1.17 strictly declined states per game at first order, so cover's 1.3 cannot
#      be met by any selector there; the bar is amended (before this pass) to target_rule_smoke.py's fallback:
#      PASS if some c gives declined trials D >= 1.0 per game at trials T <= 3.0 per game; the selected c is the
#      one with the largest D under T <= 3.0. Bits fixed at 16 (no search over bits).
#    - PASS -> coverage_games.py arm "target" with the selected c (header 7, pre-registered criteria from the
#      plan: DOSE, MECHANISM, RISK, PLAY vs the coverage control). FAIL -> no rule built from the agent's own
#      quantities reaches the declined states at cover's dose; report, and the fallback (bank the 3-way decision-
#      aware ablation with the Step B bound) applies.
# ==============================================================================

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ipomdp.domain.wumpus import WumpusAction
from ipomdp.experiments.runs import load_trained_run
from ipomdp.training.checkpointing import load_checkpoint

torch.set_num_threads(2)
run_dir, smoke_path = Path(sys.argv[1]), Path(sys.argv[2])
BITS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
dev = torch.device("cpu")
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model, filt = run.eval_agent.model, run.acting_filter
NUM_A, NUM_O, FWD, CHUNK = world.num_actions, world.num_observations, int(WumpusAction.FORWARD), 2048
rows = dict(np.load(smoke_path))
GAMES = 512


def unroll(actions, observations, initial, chunk: int = 512) -> torch.Tensor:
    """Acting-filter latents z_0..z_T (B, T + 1, D), float32, no autocast."""
    out = []
    with torch.no_grad():
        for i in range(0, actions.shape[0], chunk):
            out.append(filt.unroll(F.one_hot(actions[i:i + chunk], NUM_A).float(),
                                   F.one_hot(observations[i:i + chunk], NUM_O).float(),
                                   F.one_hot(initial[i:i + chunk], NUM_O).float()).float())
    return torch.cat(out)


# --- buffer latents and FORWARD mask (feature_localise.py section 2) ---
buf = load_checkpoint(run_dir / "checkpoints" / "latest.pt")["buffer"]
size, t_cfg, c = buf["size"], cfg.training, trained.checkpoint_collection
if size != buf["actions"].shape[0] and buf["next"] != size:
    raise RuntimeError("the buffer wrapped around; insertion-order reconstruction does not apply.")
end = c * t_cfg.env_batch_size + (c // t_cfg.eval_every) * t_cfg.eval_episodes
b_act, b_obs, b_init = (buf[k][:end] for k in ("actions", "observations", "initial_observations"))
del buf
z_buf = unroll(b_act, b_obs, b_init)[:, :-1]                                     # z_t at which a_t was played
mean = z_buf.reshape(-1, z_buf.shape[-1]).mean(0)
A = torch.randn(BITS, z_buf.shape[-1], generator=torch.Generator().manual_seed(0))
weights = 2 ** torch.arange(BITS)


def code(z: torch.Tensor) -> torch.Tensor:
    return (((z - mean) @ A.T) > 0).long() @ weights


codes = code(z_buf.reshape(-1, z_buf.shape[-1]))
forward = (b_act.reshape(-1) == FWD)
n_s = torch.bincount(codes, minlength=2 ** BITS)
n_sa = torch.bincount(codes[forward], minlength=2 ** BITS)
print(f"== {run_dir.name}: buffer {b_act.shape[0]} games to collection {c}, {codes.numel()} latents, "
      f"{int((n_s > 0).sum())} occupied codes of {2 ** BITS}", flush=True)

# --- turned latents of the smoke rows ---
games = torch.load(Path("runs/wumpus/coverage") / f"games_control_{run_dir.name}.pt", weights_only=False)
z_hist = unroll(*(games[k][:GAMES] for k in ("actions", "observations", "initial_observations")))
z_root = z_hist[torch.as_tensor(rows["game"]), torch.as_tensor(rows["t"])]
turn = torch.as_tensor(rows["turn"])
z_turn = []
with torch.no_grad():
    for i in range(0, len(z_root), CHUNK):
        e = model.expand(z_root[i:i + CHUNK])
        idx = torch.arange(e.next_states.shape[0])
        o_star = e.observation_probs[idx, turn[i:i + CHUNK]].argmax(-1)
        z_turn.append(e.next_states[idx, turn[i:i + CHUNK], o_star].float())
h = code(torch.cat(z_turn))
cnt_sa, cnt_s = n_sa[h].numpy(), n_s[h].numpy()
declined, unvisited, risky, wall = (rows[k].astype(bool) for k in ("declined", "unvisited", "risky", "wall"))
for name, mask in (("declined", declined), ("other", ~declined)):
    print(f"  {name}: n_sa median {np.median(cnt_sa[mask]):.0f} (zero {np.mean(cnt_sa[mask] == 0):.3f}), "
          f"n_s median {np.median(cnt_s[mask]):.0f}", flush=True)
best = None
for cmax in (0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000):
    on = cnt_sa <= cmax
    t_rate, d_rate = on.sum() / GAMES, (on & declined).sum() / GAMES
    print(f"  n_sa <= {cmax:>4}: trials {t_rate:5.2f}/game, declined {d_rate:4.2f}/game (precision "
          f"{(on & declined).sum() / max(on.sum(), 1):.3f}), unvisited {(on & unvisited).sum() / GAMES:4.2f}, "
          f"risky {(on & risky).sum() / GAMES:4.2f}, wall {(on & wall).sum() / GAMES:4.2f}", flush=True)
    if t_rate <= 3.0 and (best is None or d_rate > best[1]):
        best = (cmax, d_rate)
print(f"  {'PASS' if best and best[1] >= 1.0 else 'FAIL'}: best c = {best[0] if best else None}, declined "
      f"{best[1] if best else 0:.2f}/game at T <= 3.0 (bar 1.0)", flush=True)
np.savez(smoke_path.with_name(smoke_path.stem + "_novelty.npz"), n_sa=cnt_sa, n_s=cnt_s)
print("DONE", flush=True)
