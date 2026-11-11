# ABSOLUTE PATH: reports/2026-11-09_augmentations/sc_smoke.py
"""
Smoke S-C (PROTOCOL.md C-i): does an ensemble's information-gain score, weighted by the value at stake, rank the
declined "turn, then step" decisions above the agent's other non-FORWARD decisions?

DESIGN DECISIONS & THEORETICAL FOUNDATIONS
==============================================================================
# 1. Question (2026-11-10, written before any run):
#    - Augmentation C (ensemble information-directed collection; Russo & Van Roy 2018) would add to the collection
#      planner's reward a bonus span_o V-bar(z'_o) * JSD(P_1..P_K)(z, a): the disagreement of K observation heads about
#      the next percept (the epistemic part of the predictive entropy), weighted by how much the backed-up value depends
#      on which percept arrives. It is worth building only if the score points at the decisions whose model error
#      costs the most: the declined turn-then-step decisions, where the turned latent under-rates what the new cell
#      reveals (feature_localise.py, Step D).
#    - Banked evidence against: a head ensemble on one latent measured novelty, not error (E1); SimHash counts found
#      the declined pairs COMMON in the buffer (target_novelty_smoke.py). The value weight is what is new here.
# 2. Method (CPU; usage: CUDA_VISIBLE_DEVICES= uv run python <this> <run dir> <rows .npz>):
#    - Rows: target_rule_smoke.py's rows of the run (every live non-FORWARD decision of the first 512 control games
#      of runs/wumpus/coverage/games_control_<seed>.pt, with the macro turn x* and the privileged label "declined").
#      Root latents: the best.pt acting filter on the stored histories. Turned latent: the model's child after x* under
#      its most likely percept (as target_novelty_smoke.py).
#    - Ensemble: K = 5 ObservationHeads (the run's architecture) trained on FROZEN acting-filter latents of the run's
#      buffer up to best.pt's collection (live transitions z_t, a_t -> o_{t+1}), independent initialisations (seeds
#      0-4) and bootstrap weights Poisson(1) per (head, stored game); Adam lr 1e-3, 6000 steps of 2048 rows.
#    - Score at (turned latent, FORWARD): JSD = H(mean_k P_k) - mean_k H(P_k), times span = max - min of V-bar over
#      the FORWARD children the planner keeps (run model P >= 1e-3). Reported also: JSD alone and span alone.
#    - AUC (Mann-Whitney) of the score for declined rows against all other rows.
# 3. Pre-registered rule (manifest Part 4): AUC >= 0.65 on 2 of 3 seeds (0-2) -> build Augmentation C; otherwise C is
#    closed and not built. Seeds 1-2 need their rows from target_rule_smoke.py (a GPU play, queued after the C-h
#    screen); seed 0's rows exist (runs/wumpus/target/smoke_seed0.npz).
# 4. RESULT: filled after the runs.
# ==============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ipomdp.domain.wumpus import WumpusAction
from ipomdp.experiments.runs import load_trained_run
from ipomdp.models.heads import ObservationHead
from ipomdp.training.checkpointing import load_checkpoint

K, STEPS, BATCH, LR, PRUNE, CHUNK, GAMES, GAME_OVER = 5, 6000, 2048, 1e-3, 1e-3, 2048, 512, 32


def unroll(filt, actions, observations, initial, num_a, num_o, chunk: int = 512) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for i in range(0, actions.shape[0], chunk):
            out.append(filt.unroll(F.one_hot(actions[i:i + chunk], num_a).float(), F.one_hot(observations[i:i + chunk], num_o).float(),
                                   F.one_hot(initial[i:i + chunk], num_o).float()).float())
    return torch.cat(out)


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    ranks = np.empty(len(scores))
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    torch.set_num_threads(4)
    run_dir, rows_path = Path(sys.argv[1]), Path(sys.argv[2])
    dev = torch.device("cpu")
    trained = load_trained_run(run_dir, dev)
    world, run, cfg = trained.setup.domain, trained.run, trained.cfg
    num_a, num_o, fwd = world.num_actions, world.num_observations, int(WumpusAction.FORWARD)
    model, filt = run.eval_agent.model, run.acting_filter

    # --- frozen buffer latents up to best.pt's collection (feature_localise.py section 2) ---
    buf = load_checkpoint(run_dir / "checkpoints" / "latest.pt")["buffer"]
    c, t_cfg = trained.checkpoint_collection, cfg.training
    end = c * t_cfg.env_batch_size + (c // t_cfg.eval_every) * t_cfg.eval_episodes
    b_act, b_obs, b_init = (buf[k][:end].cpu() for k in ("actions", "observations", "initial_observations"))
    del buf
    z = unroll(filt, b_act, b_obs, b_init, num_a, num_o)[:, :-1]                          # z_t at which a_t was played
    live = torch.cat([b_init[:, None], b_obs[:, :-1]], 1) != GAME_OVER
    game = torch.arange(end)[:, None].expand_as(live)[live]
    z_tr, a_tr, o_tr = z[live], b_act[live], b_obs[live]
    print(f"== {run_dir.name}: best.pt collection {c}, {end} games, {len(z_tr)} live transitions", flush=True)

    # --- K bootstrapped heads on frozen latents ---
    heads = []
    for k in range(K):
        torch.manual_seed(k)
        head = ObservationHead(cfg.model.latent_dim, num_a, num_o, cfg.model.hidden_dim, cfg.model.num_blocks)
        weights = torch.poisson(torch.ones(end), generator=torch.Generator().manual_seed(100 + k))[game]
        opt = torch.optim.Adam(head.parameters(), lr=LR)
        g = torch.Generator().manual_seed(200 + k)
        for step in range(STEPS):
            idx = torch.randint(0, len(z_tr), (BATCH,), generator=g)
            w = weights[idx]
            loss = (w * F.cross_entropy(head(z_tr[idx], F.one_hot(a_tr[idx], num_a).float()), o_tr[idx],
                                        reduction="none")).sum() / w.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        heads.append(head)
        print(f"  head {k}: final batch NLL {loss.item():.4f}", flush=True)

    # --- turned latents of the rows and the score at (z_turn, FORWARD) ---
    rows = dict(np.load(rows_path))
    games = torch.load(Path("runs/wumpus/coverage") / f"games_control_{run_dir.name}.pt", weights_only=False)
    z_hist = unroll(filt, *(games[k][:GAMES] for k in ("actions", "observations", "initial_observations")), num_a, num_o)
    z_root = z_hist[torch.as_tensor(rows["game"]), torch.as_tensor(rows["t"])]
    turn = torch.as_tensor(rows["turn"])
    jsd, span = [], []
    with torch.no_grad():
        for i in range(0, len(z_root), CHUNK):
            e = model.expand(z_root[i:i + CHUNK])
            idx = torch.arange(e.next_states.shape[0])
            o_star = e.observation_probs[idx, turn[i:i + CHUNK]].argmax(-1)
            z_turn = e.next_states[idx, turn[i:i + CHUNK], o_star].float()
            fa = F.one_hot(torch.full((len(z_turn),), fwd), num_a).float()
            p = torch.stack([F.softmax(h(z_turn, fa), -1) for h in heads])               # (K, N, O)
            ent = lambda q: -(q * q.clamp(min=1e-12).log()).sum(-1)                      # noqa: E731
            jsd.append(ent(p.mean(0)) - ent(p).mean(0))
            f = model.expand(z_turn)
            kept = f.observation_probs[:, fwd] >= PRUNE
            v = f.next_values[:, fwd]
            span.append(torch.where(kept, v, -torch.inf).max(-1).values - torch.where(kept, v, torch.inf).min(-1).values)
    jsd, span = torch.cat(jsd).numpy(), torch.cat(span).numpy()
    score = jsd * span
    declined = rows["declined"].astype(bool)
    print(f"  rows {len(declined)}, declined {declined.sum()} ({declined.mean():.3f})")
    for name, s in (("span x JSD", score), ("JSD", jsd), ("span", span)):
        print(f"  {name:11s}: AUC {auc(s, declined):.3f}; median declined {np.median(s[declined]):.4g}, "
              f"other {np.median(s[~declined]):.4g}", flush=True)
    verdict = "PASS" if auc(score, declined) >= 0.65 else "FAIL"
    print(f"VERDICT {run_dir.name}: span x JSD AUC {auc(score, declined):.3f} -> {verdict} (bar 0.65; rule: 2 of 3 seeds)")


if __name__ == "__main__":
    main()
