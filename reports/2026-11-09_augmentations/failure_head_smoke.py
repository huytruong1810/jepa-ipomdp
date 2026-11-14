# ABSOLUTE PATH: reports/2026-11-09_augmentations/failure_head_smoke.py
"""
Smoke M-A (PROTOCOL.md C-k): can a dedicated failure head, trained by binary cross-entropy on a network set's frozen
latents, rate the risk of the FORWARDs the planner chose and died on at a usable fraction of the exact danger?

DESIGN DECISIONS & THEORETICAL FOUNDATIONS
==============================================================================
# 1. Question (2026-11-13, written before any run):
#    - Augmentation A (chance-constrained search; ConstrainedZero / BetaZero style) would restrict the planner to
#      actions whose predicted failure probability f(z, a) stays under a budget Delta. It can only work if f is not
#      itself under-rated where the planner picks: risk_localise.py found that at every fatal FORWARD the observation
#      head's P(GAME_OVER | z, FORWARD) (and the reward head's implied risk) was 0.02-0.05 of the exact danger of the
#      cell ahead (the optimizer's curse: the argmax picks the under-estimates).
#    - A separate head, trained by BCE on the death label d_t = 1[r_t < -500] with its own initialisation, gives a
#      second estimate whose errors are not the ones the planner selected on. The question is whether that estimate
#      sees the risk at the fatal steps; the latent itself may not encode it (then no head on it can).
#    - Banked evidence against: head swaps were exploited (R1a); the coverage ensemble's spread measured novelty
#      (E1); bootstrapped whole-model ensembles shared the buffer's holes (C1).
# 2. Method (usage: uv run python <this> <coverage|union|random> <seed>; GPU for the replay and latents):
#    - Sets and their training games (as objective_ab.py 2j / 2q trained them): control = run buffer + coverage
#      control games; union = buffer + cover + own games; random = buffer + control games + random-restore games
#      (loss-masked: only the branch steps are live transitions).
#    - Latents: the set's acting filter on its training games, frozen. Transitions: live steps (the step's input
#      observation is not GAME_OVER) and, for masked games, only loss-mask-live steps. Label d_t = 1[r_t < -500].
#    - Head: an MLP on [z, one-hot a] (the run's hidden width and block count, as ObservationHead), one logit; Adam lr
#      1e-3, 6000 steps of 2048 transitions, seed 0; plain BCE (no class reweighting: calibration is the quantity).
#    - Read: the set's 512 analysis games are reproduced by replaying its stored actions (risk_localise.py's
#      <set>.risk.npz) through the seeded simulator on CUDA (exact: checked 2026-11-13, returns equal to 3e-4); the
#      head is applied to the acting filter's root latents at every step with FORWARD.
#    - Reported: median and mean of head / exact danger at the fatal chosen FORWARDs (C-k's quantity), the same for
#      the stored max(P_O, P_R); calibration mean(head) / mean(danger) over risky live states (danger > 0) and over
#      risky states with the cell ahead unvisited; phantom risk mean(head) at live states with a provably safe
#      unvisited cell ahead; AUC of the head for fatal vs survived among chosen risky FORWARDs.
# 3. Pre-registered rule (PROTOCOL.md C-k): on the union set, median head / danger at fatal chosen FORWARDs >= 0.25 on
#    2 of 3 seeds (0-2), and mean(head) / mean(danger) over risky live states <= 1.5 on every seed. Control and random
#    are reported only. Pass -> build A and register CC-play; fail -> A closed.
# 4. RESULT (2026-11-13; runs/wumpus/failure_head/, PROTOCOL.md "C-k result"): FAIL, A closed. Union: median head /
#    danger at the fatal chosen FORWARDs 0.000 / 0.000 / 0.000 (84 / 27 / 80 fatal steps; stored heads 0.026 / 0.033 /
#    0.014); calibration over risky states 0.86 / 0.71 / 0.64. Control 0.000 on 3 of 3, random 0.002 / 0.009 (seed 0 had
#    no fatal step). The exact danger at those steps has median 1 on 5 of 8 set-seeds: provably deadly cells the latent
#    does not mark, so no head on it (and no constraint on that head) can see them.
# ==============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ipomdp.domain import WumpusSimulator
from ipomdp.domain.wumpus import WumpusAction
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.models.layers import build_residual_stack
from ipomdp.training import stream_seed
from ipomdp.training.checkpointing import load_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-02_wumpus_diagnostics"))
from saved_networks import in_layout  # noqa: E402

STEPS, BATCH, LR, CHUNK, GAME_OVER, DEATH = 6000, 2048, 1e-3, 512, 32, -500.0
C, B = Path("runs/wumpus/coverage"), Path("runs/wumpus/branch")
SETS = {   # name -> (saved networks, extra training game files, risk rows)
    "control": (C / "control_seed{s}.pt", [C / "games_control_seed{s}.pt"], C / "control_seed{s}.risk.npz"),
    "union": (C / "union_seed{s}.pt", [C / "games_cover_seed{s}.pt", C / "games_own_seed{s}.pt"], C / "union_seed{s}.risk.npz"),
    "random": (B / "random_seed{s}.pt", [C / "games_control_seed{s}.pt", B / "games_random_seed{s}.pt"],
               B / "random_seed{s}.risk.npz"),
}


class FailureHead(nn.Module):
    """logit of P(death at this step | z, a)."""

    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int, num_blocks: int):
        super().__init__()
        self.net = build_residual_stack(latent_dim + num_actions, hidden_dim, 1, num_blocks)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, action], dim=-1)).squeeze(-1)


def unroll(filt, actions, observations, initial, num_a: int, num_o: int) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for i in range(0, actions.shape[0], CHUNK):
            out.append(filt.unroll(F.one_hot(actions[i:i + CHUNK], num_a).float(),
                                   F.one_hot(observations[i:i + CHUNK], num_o).float(),
                                   F.one_hot(initial[i:i + CHUNK], num_o).float()).float())
    return torch.cat(out)


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    name, seed = sys.argv[1], int(sys.argv[2])
    torch.set_num_threads(2)
    dev = torch.device("cuda")
    saved, extras, risk_path = SETS[name]
    saved, risk_path = Path(str(saved).format(s=seed)), Path(str(risk_path).format(s=seed))
    extras = [Path(str(p).format(s=seed)) for p in extras]
    run_dir = Path(f"runs/sweeps/wumpus_retrain/seed{seed}")
    trained = load_trained_run(run_dir, dev)
    run, cfg, world = trained.run, trained.cfg, trained.setup.domain
    states = in_layout(torch.load(saved, map_location=dev), run.networks)
    for key, net in run.networks.items():
        net.load_state_dict(states[key])
    filt, num_a, num_o, fwd = run.acting_filter, world.num_actions, world.num_observations, int(WumpusAction.FORWARD)

    # --- training games: buffer + the set's extra games (objective_ab.py 2j / 2q) ---
    buf = load_checkpoint(run_dir / "checkpoints" / "latest.pt")["buffer"]
    n = buf["size"]
    acts, obs, rew, init = (buf[k][:n].to(dev) for k in ("actions", "observations", "rewards", "initial_observations"))
    mask = torch.ones_like(acts, dtype=torch.bool)
    del buf
    for path in extras:
        extra = torch.load(path, map_location=dev)
        acts, obs = torch.cat([acts, extra["actions"]]), torch.cat([obs, extra["observations"]])
        rew, init = torch.cat([rew, extra["rewards"]]), torch.cat([init, extra["initial_observations"]])
        mask = torch.cat([mask, extra["loss_mask"].bool() if "loss_mask" in extra else torch.ones_like(extra["actions"], dtype=torch.bool)])
        del extra
    z = unroll(filt, acts, obs, init, num_a, num_o)[:, :-1]
    live = (torch.cat([init[:, None], obs[:, :-1]], 1) != GAME_OVER) & mask
    z_tr, a_tr, d_tr = z[live], acts[live], (rew[live] < DEATH).float()
    del z
    print(f"== {name} seed {seed}: {saved}; {acts.shape[0]} games, {len(z_tr)} transitions, death rate {d_tr.mean():.4f}",
          flush=True)

    # --- the failure head ---
    torch.manual_seed(0)
    head = FailureHead(cfg.model.latent_dim, num_a, cfg.model.hidden_dim, cfg.model.num_blocks).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    g = torch.Generator(device=dev).manual_seed(1)
    for step in range(STEPS):
        idx = torch.randint(0, len(z_tr), (BATCH,), device=dev, generator=g)
        loss = F.binary_cross_entropy_with_logits(head(z_tr[idx], F.one_hot(a_tr[idx], num_a).float()), d_tr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 2000 == 0:
            print(f"  step {step + 1}: batch BCE {loss.item():.5f}", flush=True)
    head.eval()
    del z_tr, a_tr, d_tr

    # --- the set's 512 analysis games, replayed from its stored actions (exact on CUDA) ---
    rows = dict(np.load(risk_path))
    played = torch.as_tensor(rows["action"], device=dev)
    settings = WumpusAnalysisSettings()
    sim = WumpusSimulator(world, played.shape[0], cfg.env.max_steps, stream_seed(settings.seed, 1), dev)
    o0 = sim.reset()
    steps = [sim.step(played[:, t]) for t in range(played.shape[1])]
    g_obs = torch.stack([s.observation for s in steps], 1)
    g_rew = torch.stack([s.reward for s in steps], 1)
    replay_deaths = (g_rew < DEATH).any(1).cpu().numpy()
    if not (replay_deaths == rows["fatal"].any(1)).all():
        raise RuntimeError(f"{risk_path}: the replay does not reproduce the stored games' deaths.")
    z_games = unroll(filt, played, g_obs, o0, num_a, num_o)[:, :-1]                     # root latent at each decision
    with torch.no_grad():
        p = torch.sigmoid(head(z_games, F.one_hot(torch.full(played.shape, fwd, device=dev), num_a).float())).double().cpu().numpy()

    fatal, live_r, danger = rows["fatal"], rows["live"], rows["danger"]
    stored = np.maximum(rows["p_o"], rows["p_r"])
    risky = live_r & (danger > 1e-9)
    chosen = risky & (rows["action"] == fwd)
    safe_new = live_r & (danger <= 1e-9) & rows["unvisited"]
    ratio = lambda q, m: float(q[m].mean() / danger[m].mean()) if m.any() else float("nan")  # noqa: E731
    med_head = float(np.median(p[fatal] / danger[fatal])) if fatal.any() else float("nan")
    med_stored = float(np.median(stored[fatal] / danger[fatal])) if fatal.any() else float("nan")
    print(f"  fatal chosen FORWARDs {int(fatal.sum())}; exact danger median {np.median(danger[fatal]) if fatal.any() else float('nan'):.3f}")
    print(f"  head / danger at fatal: median {med_head:.3f}, mean {float((p[fatal] / danger[fatal]).mean()) if fatal.any() else float('nan'):.3f}; "
          f"stored max(P_O, P_R): median {med_stored:.3f}")
    print(f"  calibration mean(head)/mean(danger): risky live {ratio(p, risky):.3f}, risky unvisited-ahead "
          f"{ratio(p, risky & rows['unvisited']):.3f} (stored {ratio(stored, risky):.3f}, {ratio(stored, risky & rows['unvisited']):.3f}); "
          f"phantom at safe new cells {p[safe_new].mean():.4f} (stored {stored[safe_new].mean():.4f})")
    print(f"  AUC fatal vs survived among chosen risky FORWARDs: head {auc(p[chosen], fatal[chosen]):.3f}, "
          f"stored {auc(stored[chosen], fatal[chosen]):.3f}")
    np.savez(Path(f"runs/wumpus/failure_head/{name}_seed{seed}.npz"), head=p)
    print(f"READ {name} seed {seed}: median {med_head:.3f} (bar 0.25), inflation {ratio(p, risky):.3f} (bar <= 1.5)", flush=True)


if __name__ == "__main__":
    main()
