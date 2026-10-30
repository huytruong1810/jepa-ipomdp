# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/turn_map_check.py
# ==============================================================================
# DOES A LINEAR TURN MAP FIT ON THE BUFFER CARRY THE GLITTER PREDICTION TO THE DECLINED TURNED LATENTS?
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-30; the restore-and-branch design notes' "pick #2", heading symmetry):
#    - Step D (feature_localise.py header 4) found the learned observation model under-rates glitter ~2x at the
#      planner's declined turned latents z_turn = filter(z_root, turn, o_turn), while the same factors decode on
#      buffer states: a support SHIFT specific to the turn step. A turn changes only the hunter's heading, a cyclic
#      group action (four headings); if the filter's turn transition were equivariant (a fixed map of the latent,
#      the same at every state), the declined turns would not be off-support at all: the map learned on the
#      buffer's ordinary turns would apply. An equivariance regulariser (e.g. Park et al. 2022, "Learning symmetric
#      embeddings for equivariant world models"; van der Pol et al. 2020, MDP homomorphic networks) is the remedy
#      this check prices before any training.
#    - The check: fit one affine map per turn direction, z_{t+1} ~ A_turn z_t + b_turn, on the buffer's own turn
#      transitions (ridge), then ask whether the planner's observation head at the MAPPED latent A z_root + b
#      predicts glitter after FORWARD better than at the filter's own turned latent z_turn.
#
# 2. Method (usage: uv run python <this> <run dir> [threads=4]; CPU only; then <this> summary <run dirs...>):
#    - Networks: the run's best.pt (load_trained_run), the states of feature_localise.py's declined_states.pt
#      (base mode, same networks). Consistency asserted: expand(z_root).next_states[turn, o_turn] reproduces the
#      saved z_turn (max abs error <= 1e-4).
#    - Buffer: feature_localise.py's slots (latest.pt buffer up to best.pt's collection's greedy block, copied),
#      acting-filter unroll in float32, no autocast. Pairs (z_t, z_{t+1}) at live steps (no GAME_OVER observation
#      before t) whose action a_t is TURN_LEFT or TURN_RIGHT, one map per direction. 10% of the games held out
#      (split seed 0); ridge lambda from {1e-4, 1e-3, 1e-2, 1e-1, 1, 10} by held-out MSE; R^2 pooled over the
#      latent dimensions (1 - SSE / SST around the held-out mean).
#    - At the declined rows: the joint observation head's P(o' | ., FORWARD) at z_turn and at z_map = A z_root + b;
#      glitter ratio = mean predicted P(glitter) / mean exact P(glitter) (exact from declined_states.pt's
#      p_fwd_turn), and the median total variation to the exact P(o' | FORWARD).
#    - Deviation from the design notes (recorded before any run): the notes named "the Step-D probes on A z". The
#      probes' weights were never saved and refitting them needs the layout-enumeration factors of the whole buffer
#      (GPU, ~10 min per seed; the GPU is held by the branch screen). The planner reads the observation head, not
#      the probes, and Step D found the probe and the head under-rate glitter by the same factor (0.51-0.61 vs
#      0.49-0.58), so the head's glitter ratio is the decision-relevant measure.
#
# 3. Pre-registered readout (written 2026-10-30 before any run; runs/sweeps/wumpus_retrain seeds 0-4):
#    - FIT: held-out R^2 >= 0.9 for both turn directions.
#    - CLOSES: glitter ratio at z_map - ratio at z_turn >= half of (1 - ratio at z_turn), and the TV median at z_map
#      not above the one at z_turn.
#    - PASS: FIT and CLOSES on >= 4 of 5 seeds -> an equivariance regulariser on the filter's turn transition becomes
#      a candidate method arm (retrain screen, objective_ab.py). FIT fails -> the latent's turn is not linear on the
#      buffer, the check cannot speak (report R^2, drop). FIT passes, CLOSES fails -> the shift is not a turn-map
#      failure: the mapped latent under-rates glitter too (the deficit is in z_root's state or in the head), drop.
# ==============================================================================

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

if sys.argv[1] == "summary":
    passes = 0
    for run in map(Path, sys.argv[2:]):
        s = json.loads((run / "analysis" / "turn_map_check.json").read_text())
        passes += s["pass"]
        print(f"{run.name}: R^2 left {s['fit']['left']['r2']:.3f} right {s['fit']['right']['r2']:.3f}; glitter ratio "
              f"z_turn {s['glitter_ratio_turn']:.2f} -> z_map {s['glitter_ratio_map']:.2f}; TV median "
              f"{s['tv_median_turn']:.3f} -> {s['tv_median_map']:.3f}; FIT {s['fit_ok']} CLOSES {s['closes']}")
    need = 4 if len(sys.argv[2:]) == 5 else len(sys.argv[2:])
    print(f"pass on {passes} of {len(sys.argv[2:])} seeds; readout {'PASS' if passes >= need else 'FAIL'}")
    sys.exit(0)

from ipomdp.domain.wumpus import GAME_OVER, Percept, WumpusAction  # noqa: E402
from ipomdp.experiments.runs import load_trained_run  # noqa: E402
from ipomdp.training.checkpointing import load_checkpoint  # noqa: E402

run_dir = Path(sys.argv[1])
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
dev = torch.device("cpu")
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
FWD = int(WumpusAction.FORWARD)
TURNS = {"left": int(WumpusAction.TURN_LEFT), "right": int(WumpusAction.TURN_RIGHT)}
NUM_A, NUM_O = world.num_actions, world.num_observations
GLITTER = torch.tensor([bool(o & Percept.GLITTER) and o != GAME_OVER for o in range(NUM_O)])

# --- Declined states (feature_localise.py, base mode) and the consistency check ---
d = torch.load(run_dir / "analysis" / "declined_states.pt", weights_only=False)
z_root, z_turn, o_turn = d["z_root"].float(), d["z_turn"].float(), d["o_turn"]
turn = torch.tensor([m["turn"] for m in d["meta"]])
p_exact = d["p_fwd_turn"].double()
with torch.no_grad():
    nxt = torch.cat([model.expand(z_root[i:i + 256]).next_states for i in range(0, len(z_root), 256)])
err = (nxt[torch.arange(len(turn)), turn, o_turn] - z_turn).abs().max().item()
if err > 1e-4:
    raise RuntimeError(f"best.pt does not reproduce declined_states.pt's turned latents (max abs error {err:.2e}).")
print(f"{run_dir.name}: {len(turn)} declined states, turned latents reproduced (max abs error {err:.1e})", flush=True)

# --- Buffer turn pairs (feature_localise.py section 2's slots, copied) ---
buf = load_checkpoint(run_dir / "checkpoints" / "latest.pt")["buffer"]
size, t_cfg, c = buf["size"], cfg.training, trained.checkpoint_collection
if size != buf["actions"].shape[0] and buf["next"] != size:
    raise RuntimeError("the buffer wrapped around; insertion-order reconstruction does not apply.")
start = c * t_cfg.env_batch_size + (c // t_cfg.eval_every - 1) * t_cfg.eval_episodes
end = start + t_cfg.eval_episodes
if "greedy" in buf and (not bool(buf["greedy"][start:end].all()) or bool(buf["greedy"][start - 1])):
    raise RuntimeError(f"slot {start}..{end} is not collection {c}'s greedy block.")
actions, observations, initial = (buf[k][:end] for k in ("actions", "observations", "initial_observations"))
del buf
games = actions.shape[0]
held_game = torch.zeros(games, dtype=torch.bool)
held_game[torch.randperm(games, generator=torch.Generator().manual_seed(0))[: games // 10]] = True
live = torch.ones_like(actions, dtype=torch.bool)
live[:, 1:] = ~(observations[:, :-1] == GAME_OVER).cumsum(1).bool()
pairs = {k: {"train": ([], []), "held": ([], [])} for k in TURNS}
with torch.no_grad():
    for i in range(0, games, 512):
        a, o, z0 = (F.one_hot(x[i:i + 512], n).float() for x, n in
                    ((actions, NUM_A), (observations, NUM_O), (initial, NUM_O)))
        z = run.acting_filter.unroll(a, o, z0).float()
        for name, act in TURNS.items():
            sel = (actions[i:i + 512] == act) & live[i:i + 512]
            g, t = sel.nonzero(as_tuple=True)
            for split, mask in (("train", ~held_game[i:i + 512][g]), ("held", held_game[i:i + 512][g])):
                pairs[name][split][0].append(z[g[mask], t[mask]])
                pairs[name][split][1].append(z[g[mask], t[mask] + 1])
print(f"  buffer: {games} games at collection {c}", flush=True)


def ridge(x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    """Affine ridge solution W (D + 1, D) for y ~ [x, 1] W, float64; the bias is not penalised."""
    xa = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1)
    reg = lam * torch.eye(xa.shape[1], dtype=x.dtype)
    reg[-1, -1] = 0.0
    return torch.linalg.solve(xa.T @ xa + reg, xa.T @ y)


def apply(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return x @ w[:-1] + w[-1]


fit, maps = {}, {}
for name in TURNS:
    xt, yt, xh, yh = (torch.cat(pairs[name][s][k]).double() for s in ("train", "held") for k in (0, 1))
    mse = {lam: float(((apply(ridge(xt, yt, lam), xh) - yh) ** 2).mean()) for lam in LAMBDAS}
    lam = min(mse, key=mse.get)
    maps[name] = ridge(xt, yt, lam)
    r2 = 1.0 - mse[lam] * yh.numel() / float(((yh - yh.mean(0)) ** 2).sum())
    fit[name] = {"pairs_train": len(xt), "pairs_held": len(xh), "lambda": lam, "r2": r2}
    print(f"  turn {name}: {len(xt)} train / {len(xh)} held-out pairs, lambda {lam:g}, held-out R^2 {r2:.4f}", flush=True)

# --- The observation head at z_turn and at z_map ---
z_map = torch.empty_like(z_root)
for name, act in TURNS.items():
    rows = turn == act
    z_map[rows] = apply(maps[name], z_root[rows].double()).float()


def head_fwd(z: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([model.expand(z[i:i + 256]).observation_probs[:, FWD] for i in range(0, len(z), 256)]).double()


exact_glitter = float(p_exact[:, GLITTER].sum(1).mean())
out = {"run": str(run_dir), "states": len(turn), "fit": fit, "exact_glitter_mean": exact_glitter,
       "map_to_turn_distance_median": float((z_map - z_turn).norm(dim=1).median())}
for key, z in (("turn", z_turn), ("map", z_map)):
    p = head_fwd(z)
    out[f"glitter_ratio_{key}"] = float(p[:, GLITTER].sum(1).mean()) / exact_glitter
    out[f"tv_median_{key}"] = float((0.5 * (p - p_exact).abs().sum(1)).median())
out["fit_ok"] = all(f["r2"] >= 0.9 for f in fit.values())
out["closes"] = (out["glitter_ratio_map"] - out["glitter_ratio_turn"] >= 0.5 * (1.0 - out["glitter_ratio_turn"])
                 and out["tv_median_map"] <= out["tv_median_turn"])
out["pass"] = out["fit_ok"] and out["closes"]
print(f"  glitter ratio (head / exact) at z_turn {out['glitter_ratio_turn']:.3f}, at z_map {out['glitter_ratio_map']:.3f};"
      f" TV median {out['tv_median_turn']:.3f} -> {out['tv_median_map']:.3f}; |z_map - z_turn| median "
      f"{out['map_to_turn_distance_median']:.3f}; FIT {out['fit_ok']} CLOSES {out['closes']}", flush=True)
(run_dir / "analysis" / "turn_map_check.json").write_text(json.dumps(out, indent=2))
print("DONE", flush=True)
