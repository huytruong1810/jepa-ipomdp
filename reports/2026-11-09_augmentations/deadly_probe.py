# ABSOLUTE PATH: reports/2026-11-09_augmentations/deadly_probe.py
"""
Deadly-cell encoding probe (PROTOCOL.md C-l): when the agent faces a cell it has proven deadly and the observation
head misses it, is "deadly ahead" in the latent?

DESIGN DECISIONS & THEORETICAL FOUNDATIONS
==============================================================================
# 1. Question (2026-11-13, written before any run; follows C-k):
#    - C-k: the agent dies mostly by stepping into cells whose exact danger is 1. The stored risk rows show the
#      mechanism: at provably deadly states the head flags the danger (max(P_O, P_R) >= 0.5) 51-83% of the time and the
#      planner then avoids the cell; it steps in only where the head missed (0% of the steps into a deadly cell had a
#      head risk >= 0.5). Misses are no more frequent after a turn than after a FORWARD (pooled 43-54% vs 47%).
#    - Two readings. IN-LATENT: the latent holds "deadly ahead" and the head fails to read it; plausible, because the
#      head learns P(GAME_OVER | z, FORWARD) only where the agent stepped forward, and stepping into a provably deadly
#      cell is rare in the data (an outcome-only supervision gap). NOT-IN-LATENT: the latent itself lacks it.
#    - A probe with privileged targets (the exact danger, diagnostic only) separates them: it is trained on every
#      buffer state, not only where FORWARD was taken.
# 2. Method (usage: uv run python <this> <control|union> <seed>; GPU):
#    - Latents: the set's acting filter (runs/wumpus/coverage/<set>_seed<s>.pt) on the run's own buffer
#      (runs/sweeps/wumpus_retrain/seed<s>, latest.pt) and on the 512 analysis games, which are replayed exactly from
#      the set's stored actions (<set>_seed<s>.risk.npz) with the seeded simulator on CUDA (failure_head_smoke.py).
#    - Target: exact danger of the cell ahead, P(pit) + P(live wumpus) (0 for a wall), from the exact layout-
#      enumeration tracker, as risk_localise.py. Buffer states with the step's input observation GAME_OVER excluded.
#    - Probe: MLP (latent -> hidden -> hidden -> 1, SiLU), BCE to the soft target, Adam 1e-3, batches of 4096, early
#      stopping on 10% held-out games (patience 6 x 250 steps, at most 12000). A probe on a re-initialised filter's
#      latents is the floor (as feature_localise.py).
#    - Read at the analysis games' live states with danger >= 0.999, split by the stored head (max(P_O, P_R) >= 0.5:
#      flagged, else missed), and at the fatal ones (stepped into). Recall = share with probe >= 0.5.
#    - Reported: the same recalls of the random-filter probe; at missed vs flagged states, the share whose current
#      percept (the last observation) carries the relevant warning (breeze for a pit ahead, stench for a wumpus) and
#      the share where the danger is a pit vs a wumpus.
# 3. Pre-registered rule (PROTOCOL.md C-l; control set, seeds 0-2): recall at head-missed states >= 0.7 on 2 of 3 ->
#    IN-LATENT; <= 0.4 on 2 of 3 -> NOT-IN-LATENT; otherwise MIXED. Union reported.
# ==============================================================================
"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ipomdp.domain import WumpusSimulator
from ipomdp.domain.wumpus import GRID, WumpusBeliefTracker
from ipomdp.experiments.runs import load_trained_run
from ipomdp.experiments.wumpus import WumpusAnalysisSettings
from ipomdp.training import stream_seed
from ipomdp.training.checkpointing import load_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-02_wumpus_diagnostics"))
from saved_networks import in_layout  # noqa: E402

GAME_OVER, CHUNK, TRACK_CHUNK, CERTAIN = 32, 512, 128, 0.999
BREEZE, STENCH = 2, 1


def unroll(filt, actions, observations, initial, num_a: int, num_o: int) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for i in range(0, actions.shape[0], CHUNK):
            out.append(filt.unroll(F.one_hot(actions[i:i + CHUNK], num_a).float(),
                                   F.one_hot(observations[i:i + CHUNK], num_o).float(),
                                   F.one_hot(initial[i:i + CHUNK], num_o).float()).float())
    return torch.cat(out)


def exact_danger(world, actions, observations, initial) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact danger ahead (G, T), its pit part (G, T), and live (G, T) at every decision, by layout enumeration."""
    tracker, dangers, pits, lives = WumpusBeliefTracker(world), [], [], []
    for i in range(0, actions.shape[0], TRACK_CHUNK):
        consistent, own = tracker.start(initial[i:i + TRACK_CHUNK])
        d_t, p_t, l_t = [], [], []
        for t in range(actions.shape[1]):
            m = tracker.marginals(consistent, own)
            x = m.cell % GRID + world._delta[m.heading, 0]
            y = m.cell // GRID + world._delta[m.heading, 1]
            inside = (x >= 0) & (x < GRID) & (y >= 0) & (y < GRID)
            ahead = (x + GRID * y).clamp(0, GRID * GRID - 1).unsqueeze(1)
            pit = torch.where(inside, m.pit.gather(1, ahead).squeeze(1), 0.0)
            wum = torch.where(inside, m.wumpus_alive.double() * m.wumpus.gather(1, ahead).squeeze(1), 0.0)
            d_t.append(pit + wum)
            p_t.append(pit)
            l_t.append(~m.done)
            consistent, own = tracker.update(consistent, own, actions[i:i + TRACK_CHUNK, t], observations[i:i + TRACK_CHUNK, t])
        dangers.append(torch.stack(d_t, 1))
        pits.append(torch.stack(p_t, 1))
        lives.append(torch.stack(l_t, 1))
    return torch.cat(dangers), torch.cat(pits), torch.cat(lives)


def fit_probe(z_tr, y_tr, z_he, y_he, hidden: int, dev) -> nn.Module:
    torch.manual_seed(0)
    probe = nn.Sequential(nn.Linear(z_tr.shape[-1], hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                          nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    g = torch.Generator(device=dev).manual_seed(0)
    best, best_state, bad = float("inf"), None, 0
    for step in range(1, 12001):
        idx = torch.randint(0, len(z_tr), (4096,), device=dev, generator=g)
        loss = F.binary_cross_entropy_with_logits(probe(z_tr[idx]).squeeze(-1), y_tr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0:
            with torch.no_grad():
                held = float(F.binary_cross_entropy_with_logits(probe(z_he).squeeze(-1), y_he))
            if held < best - 1e-5:
                best, best_state, bad = held, copy.deepcopy(probe.state_dict()), 0
            else:
                bad += 1
                if bad >= 6:
                    break
    probe.load_state_dict(best_state)
    return probe.eval().requires_grad_(False)


def main() -> None:
    name, seed = sys.argv[1], int(sys.argv[2])
    torch.set_num_threads(2)
    dev = torch.device("cuda")
    run_dir = Path(f"runs/sweeps/wumpus_retrain/seed{seed}")
    saved, risk_path = Path(f"runs/wumpus/coverage/{name}_seed{seed}.pt"), Path(f"runs/wumpus/coverage/{name}_seed{seed}.risk.npz")
    trained = load_trained_run(run_dir, dev)
    run, cfg, world = trained.run, trained.cfg, trained.setup.domain
    states = in_layout(torch.load(saved, map_location=dev), run.networks)
    for key, net in run.networks.items():
        net.load_state_dict(states[key])
    num_a, num_o = world.num_actions, world.num_observations

    # --- buffer: latents, exact danger, train/held split by game ---
    buf = load_checkpoint(run_dir / "checkpoints" / "latest.pt")["buffer"]
    n = buf["size"]
    acts, obs, init = (buf[k][:n].to(dev) for k in ("actions", "observations", "initial_observations"))
    del buf
    danger, _, live = exact_danger(world, acts, obs, init)
    live &= torch.cat([init[:, None], obs[:, :-1]], 1) != GAME_OVER
    held_games = torch.zeros(n, dtype=torch.bool, device=dev)
    held_games[torch.randperm(n, generator=torch.Generator().manual_seed(0))[: n // 10].to(dev)] = True
    tr, he = live & ~held_games[:, None], live & held_games[:, None]
    y = danger.float()
    random_filter = copy.deepcopy(run.acting_filter)
    torch.manual_seed(0)
    for module in random_filter.modules():
        if module is not random_filter and hasattr(module, "reset_parameters"):
            module.reset_parameters()
    probes = {}
    for label, filt in (("latent", run.acting_filter), ("random", random_filter)):
        z = unroll(filt, acts, obs, init, num_a, num_o)[:, :-1]
        probes[label] = fit_probe(z[tr], y[tr], z[he], y[he], cfg.model.hidden_dim, dev)
        with torch.no_grad():
            p_he = torch.sigmoid(probes[label](z[he]).squeeze(-1))
        cert_he = y[he] >= CERTAIN
        print(f"  {label} probe, held-out buffer states: MAE {float((p_he - y[he]).abs().mean()):.4f}; recall at "
              f"certain death {float((p_he[cert_he] >= 0.5).float().mean()):.3f} ({int(cert_he.sum())} states)", flush=True)
        del z
    print(f"== {name} seed {seed}: {n} buffer games, {int(tr.sum())} train / {int(he.sum())} held-out live states", flush=True)

    # --- analysis games, replayed from the stored actions ---
    rows = dict(np.load(risk_path))
    played = torch.as_tensor(rows["action"], device=dev)
    settings = WumpusAnalysisSettings()
    sim = WumpusSimulator(world, played.shape[0], cfg.env.max_steps, stream_seed(settings.seed, 1), dev)
    o0 = sim.reset()
    steps = [sim.step(played[:, t]) for t in range(played.shape[1])]
    g_obs, g_rew = torch.stack([s.observation for s in steps], 1), torch.stack([s.reward for s in steps], 1)
    if not ((g_rew < -500).any(1).cpu().numpy() == rows["fatal"].any(1)).all():
        raise RuntimeError(f"{risk_path}: the replay does not reproduce the stored games' deaths.")
    g_danger, g_pit, _ = exact_danger(world, played, g_obs, o0)
    if np.abs(g_danger.cpu().numpy() - rows["danger"])[rows["live"]].max() > 1e-6:
        raise RuntimeError("Recomputed exact danger differs from the stored risk rows.")
    last_obs = torch.cat([o0[:, None], g_obs[:, :-1]], 1).cpu().numpy()             # percept the decision sees
    head = np.maximum(rows["p_o"], rows["p_r"])
    cert = rows["live"] & (rows["danger"] >= CERTAIN)
    missed, flagged, fatal = cert & (head < 0.5), cert & (head >= 0.5), cert & rows["fatal"]
    is_pit = g_pit.cpu().numpy() >= CERTAIN
    warned = np.where(is_pit, (last_obs & BREEZE) > 0, (last_obs & STENCH) > 0) & (last_obs != GAME_OVER)
    print(f"  provably deadly live states {int(cert.sum())}: head flagged {int(flagged.sum())}, missed {int(missed.sum())}, "
          f"stepped into {int(fatal.sum())}")
    recalls = {}
    for label in ("latent", "random"):
        z_g = unroll(run.acting_filter if label == "latent" else random_filter, played, g_obs, o0, num_a, num_o)[:, :-1]
        with torch.no_grad():
            pr = torch.sigmoid(probes[label](z_g).squeeze(-1)).double().cpu().numpy()
        rec = {k: float((pr[m] >= 0.5).mean()) if m.any() else float("nan")
               for k, m in (("missed", missed), ("flagged", flagged), ("fatal", fatal))}
        recalls[label] = rec
        print(f"  {label} probe recall (>= 0.5): head-missed {rec['missed']:.3f}, head-flagged {rec['flagged']:.3f}, "
              f"stepped into {rec['fatal']:.3f}; median probe at missed {np.median(pr[missed]) if missed.any() else float('nan'):.3f}",
              flush=True)
    for label, m in (("missed", missed), ("flagged", flagged), ("fatal", fatal)):
        if m.any():
            print(f"  {label}: current percept warns {warned[m].mean():.3f}; danger is a pit {is_pit[m].mean():.3f}")
    r = recalls["latent"]["missed"]
    verdict = "IN-LATENT" if r >= 0.7 else "NOT-IN-LATENT" if r <= 0.4 else "MIXED"
    print(f"READ {name} seed {seed}: probe recall at head-missed provably deadly states {r:.3f} -> {verdict} "
          f"(>= 0.7 in, <= 0.4 not; rule: 2 of 3 control seeds)", flush=True)


if __name__ == "__main__":
    main()
