# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/target_rule_smoke.py
# ==============================================================================
# CAN THE AGENT'S OWN TREE TARGET THE DECLINED "TURN, THEN STEP INTO THE NEW CELL" DECISIONS? (DOSE SMOKE)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-28, after the decision-aware objectives 2i / 2o / 2p all failed):
#    - Only data at the declined turn-then-step pairs ever repaired the turned-latent model (coverage_games.py cover
#      arm, glitter ratio +0.14 to +0.32 on 4 of 5), but that arm was privileged and safety-filtered (deaths +28
#      points). The non-privileged own arm (eps 0.2 at every non-FORWARD decision) kept the risk calibrated but
#      reached the declined states ~0.4 times per game against cover's ~1.3, and repaired nothing.
#    - Candidate rule (model-based DAgger on the planner's own near-misses; Ross & Bagnell 2012): trial the macro
#      TURN_x + FORWARD (x not the chosen action) where the agent's own tree rates it close to the chosen action. q_mac(x) = r_turn + gamma *
#      sum_o P(o) Q_o(FORWARD), read from the planner's root tree (macro_reach_diag.py section 2). Trial when
#      q_chosen - max_{x != chosen} q_mac(x) <= delta. Nothing privileged chooses; the exact belief only labels the states.
#    - This smoke asks, before any arm is built or retrained, whether some delta gives the dose cover had without
#      flooding the data with untargeted trials.
#
# 2. Method:
#    - Usage: uv run python <this> <run dir> <out .npz> [<games>=512]. best.pt greedy planner (Dirichlet off,
#      temperature 0) on coverage_games.py's control stream k = 0 (simulator stream_seed(COVERAGE_BASE, 0), search
#      100, agent 200), so the games equal the first 512 of runs/wumpus/coverage/games_control_seed<s>.pt (asserted
#      on the actions when that file exists and games = 512). The shadow exact-belief heuristic (risk 0.4) labels.
#    - At every live non-FORWARD decision one row: q_chosen, q_mac and root Q of each turn, and labels for the
#      macro x* = argmax q_mac over the turns other than the chosen action (a trial must override the planner): declined (coverage_games.py's count_dose definition: the heuristic plays x*, the
#      planner does not, no known gold underfoot, target distance 2), target unvisited, target risky (exact danger
#      > 0). Unexpanded turn children get one chunked model.expand (<= 2048 latents), as macro_reach_diag.py.
#    - Asserted: q_mac(x) <= root Q(x) + 1e-6 wherever every child of the turn edge is expanded (expectimax).
#    - First-order caveat: a trial changes the rest of its game, so the counted dose is that of the first
#      trials only; the arm reports the real dose.
#
# 3. Pre-registered selection (written 2026-10-28 before the run; seed 0 of runs/sweeps/wumpus_retrain):
#    - For delta on a grid: trials per game T(delta) and declined trials per game D(delta).
#    - SELECT the smallest delta with D(delta) >= 1.3 (cover's dose) and T(delta) <= 3.0 (declined precision of at
#      least ~40%; own spent ~2.7 trials per game for 0.4 declined, ~15%).
#    - No delta meets both -> the gap alone is not precise enough: a second pass adds a SimHash pseudo-count of the
#      turned latent against the buffer (Tang et al. 2017) to the score, with the same bars. If D(delta) never
#      reaches 1.0 at T <= 3.0 with either score, targeting from the agent's own quantities is not achievable at
#      this dose: report, and the fallback (bank the 3-way decision-aware ablation and the Step B bound) applies.
# ==============================================================================

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from ipomdp.agents import PlanningAgent, WumpusReferenceAgent
from ipomdp.agents.wumpus_reference import _ZERO, _ahead, _first_actions
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.domain.wumpus import NUM_CELLS, WumpusAction
from ipomdp.experiments.runs import load_trained_run
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import play_episodes, stream_seed

COVERAGE_BASE = 20261020
run_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 512
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(0.5, 0)
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
FWD, TURNS = int(WumpusAction.FORWARD), (int(WumpusAction.TURN_LEFT), int(WumpusAction.TURN_RIGHT))
PRUNE, GAMMA, CHUNK = cfg.mcts.observation_prune, model.discount, 2048
planner = PlanningAgent(
    model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                            dirichlet_epsilon=0.0, seed=stream_seed(COVERAGE_BASE, 100),
                            observation_prune=cfg.mcts.observation_prune),
    GAMES, temperature=0.0, seed=stream_seed(COVERAGE_BASE, 200), device=dev)


def target_distance(cell: int, heading: int, safe: list[bool], is_target) -> int:
    """Fewest actions (FORWARD through safe cells, turns) until the hunter stands on a target cell (override_diag.py)."""
    if is_target(cell):
        return 0
    queue, seen = deque([(cell, heading, 0)]), {(cell, heading)}
    while queue:
        c, h, d = queue.popleft()
        ahead = _ahead(c, h)
        for nxt in ((ahead, h) if ahead is not None and safe[ahead] else None, (c, (h + 1) % 4), (c, (h + 3) % 4)):
            if nxt is None or nxt in seen:
                continue
            if nxt[0] != c and is_target(nxt[0]):
                return d + 1
            seen.add(nxt)
            queue.append((*nxt, d + 1))
    return 10 ** 6


def pruned(probs: np.ndarray) -> np.ndarray:
    """The search's branch weights over all |O| (planning/mcts.py section 6): keep P >= prune, renormalise."""
    if PRUNE == 0.0:
        return probs
    kept = np.where(probs >= PRUNE, probs, 0.0)
    return kept / kept.sum()


def forward_q(states: list[torch.Tensor]) -> np.ndarray:
    """Q(s, FORWARD) of unexpanded nodes by one expansion each, chunked; (N,)."""
    out = []
    for i in range(0, len(states), CHUNK):
        e = model.expand(torch.stack(states[i:i + CHUNK]))
        r = e.rewards[:, FWD].double().cpu().numpy()
        p = np.stack([pruned(row) for row in e.observation_probs[:, FWD].double().cpu().numpy()])
        out.append(r + GAMMA * (p * e.next_values[:, FWD].double().cpu().numpy()).sum(-1))
    return np.concatenate(out) if out else np.zeros(0)


class Recorder:
    """Drives the planner and a shadow exact-belief heuristic; one row per live non-FORWARD decision."""

    def __init__(self):
        self.batch_size = GAMES
        self.shadow = WumpusReferenceAgent(world, GAMES, 0.4)
        self.rows: list[dict] = []
        self.t, self.start = 0, time.time()

    def reset(self, initial_observation):
        planner.reset(initial_observation)
        self.shadow.reset(initial_observation)

    def update(self, action, observation):
        planner.update(action, observation)
        self.shadow.update(action, observation)

    def act(self):
        s = self.shadow
        m = s._tracker.marginals(s._consistent, s._own)
        heuristic = s.act().tolist()
        action = planner.act()
        chosen = action.tolist()
        pit, wumpus, gold = m.pit.cpu().tolist(), m.wumpus.cpu().tolist(), m.gold.cpu().tolist()
        alive, done = m.wumpus_alive.tolist(), m.done.tolist()
        pending, pending_states, first_new = [], [], len(self.rows)
        for b in range(GAMES):
            if done[b] or chosen[b] == FWD:
                continue
            root = planner.planner.roots[b]
            row = {"game": b, "t": self.t, "chosen": chosen[b], "heuristic": heuristic[b],
                   "q_chosen": root.edges[chosen[b]].q, "cell": int(m.cell[b]), "heading": int(m.heading[b]),
                   "pit": pit[b], "wumpus": wumpus[b], "gold": gold[b], "alive": alive[b],
                   "visited": list(s._visited[b])}
            for x in TURNS:
                edge = root.edges[x]
                row[f"q_turn{x}"] = edge.q
                row[f"full{x}"] = all(c.edges is not None for c in edge.children)
                row[f"probs{x}"], row[f"q_o{x}"] = edge.observation_probs, [None] * len(edge.children)
                for k, child in enumerate(edge.children):
                    if child.edges is not None:
                        row[f"q_o{x}"][k] = child.edges[FWD].q
                    else:
                        pending.append((row, x, k))
                        pending_states.append(child.state)
            self.rows.append(row)
        for (row, x, k), qk in zip(pending, forward_q(pending_states)):
            row[f"q_o{x}"][k] = float(qk)
        for row in self.rows[first_new:]:
            root = planner.planner.roots[row["game"]]
            for x in TURNS:
                row[f"q_mac{x}"] = root.edges[x].reward + GAMMA * float(np.dot(row[f"probs{x}"], row[f"q_o{x}"]))
                if row[f"full{x}"] and row[f"q_mac{x}"] > row[f"q_turn{x}"] + 1e-6:
                    raise RuntimeError(f"Expectimax identity violated: q_mac {row[f'q_mac{x}']} > q_turn "
                                       f"{row[f'q_turn{x}']}.")
                del row[f"probs{x}"], row[f"q_o{x}"], row[f"full{x}"]
            label(row)
        self.t += 1
        if self.t % 10 == 0:
            print(f"   [{time.strftime('%H:%M:%S')}] decision {self.t}/{cfg.env.max_steps}, {len(self.rows)} rows, "
                  f"{time.time() - self.start:.0f}s", flush=True)
        return action


def label(row: dict) -> None:
    """Privileged labels of the macro x* = argmax q_mac (coverage_games.py count_dose); reporting only."""
    candidates = [x for x in TURNS if x != row["chosen"]]               # a trial overrides the chosen action
    turn = max(candidates, key=lambda x: row[f"q_mac{x}"])
    cell, heading, pit, wumpus, gold, alive = (row.pop(k) for k in ("cell", "heading", "pit", "wumpus", "gold", "alive"))
    visited = row.pop("visited")
    row["turn"], row["q_mac"] = turn, row[f"q_mac{turn}"]
    target = _ahead(cell, (heading + 1) % 4 if turn == int(WumpusAction.TURN_LEFT) else (heading + 3) % 4)
    row["wall"] = target is None
    row["unvisited"] = target is not None and not visited[target]
    row["risky"] = target is not None and pit[target] + (wumpus[target] if alive else 0.0) >= _ZERO
    row["declined"] = False
    if target is not None and row["heuristic"] == turn and gold[cell] <= 1.0 - _ZERO:
        danger = [pit[c] + (wumpus[c] if alive else 0.0) for c in range(NUM_CELLS)]
        safe = [d < _ZERO for d in danger]
        reachable = {c for c, _ in _first_actions(cell, heading, safe)} | {cell}
        known = [c for c in reachable if gold[c] > 1.0 - _ZERO]
        goal = (lambda c: c in known) if known else (lambda c: safe[c] and not visited[c] and c in reachable)
        row["declined"] = target_distance(cell, heading, safe, goal) == 2


print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), {GAMES} games, control stream k = 0",
      flush=True)
rec = Recorder()
episodes, _ = play_episodes(WumpusSimulator(world, GAMES, cfg.env.max_steps, stream_seed(COVERAGE_BASE, 0), dev), rec)
print("  outcomes " + str({k: round(float(v), 4) for k, v in game_outcomes(episodes.rewards).items()}), flush=True)
reference = Path("runs/wumpus/coverage") / f"games_control_{run_dir.name}.pt"
if GAMES == 512 and reference.exists():
    stored = torch.load(reference, weights_only=False)["actions"][:GAMES]
    if not torch.equal(stored, episodes.actions.cpu()):
        raise RuntimeError(f"Replay is not faithful to {reference}.")
    print(f"  replay identical to {reference} batch 0", flush=True)

cols = {k: np.array([r[k] for r in rec.rows]) for k in rec.rows[0]}
out_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(out_path, **cols)
gap = cols["q_chosen"] - cols["q_mac"]
print(f"  {len(gap)} non-FORWARD decisions ({len(gap) / GAMES:.1f}/game); declined {cols['declined'].sum() / GAMES:.2f}/game "
      f"(unvisited {cols['unvisited'].sum() / GAMES:.2f}, risky {cols['risky'].sum() / GAMES:.2f}, wall "
      f"{cols['wall'].mean():.3f}); gap median declined {np.median(gap[cols['declined']]):+.1f}, others "
      f"{np.median(gap[~cols['declined']]):+.1f}", flush=True)
selected = None
for delta in (0, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40, 50, 75, 100):
    on = gap <= delta
    t_rate, d_rate = on.sum() / GAMES, (on & cols["declined"]).sum() / GAMES
    print(f"  delta {delta:>5}: trials {t_rate:5.2f}/game, declined {d_rate:4.2f}/game (precision "
          f"{(on & cols['declined']).sum() / max(on.sum(), 1):.3f}), unvisited {(on & cols['unvisited']).sum() / GAMES:4.2f}, "
          f"risky {(on & cols['risky']).sum() / GAMES:4.2f}, wall {(on & cols['wall']).sum() / GAMES:4.2f}", flush=True)
    if selected is None and d_rate >= 1.3 and t_rate <= 3.0:
        selected = delta
print(f"  SELECTED delta = {selected}" if selected is not None else "  NO delta meets D >= 1.3 at T <= 3.0 (section 3)",
      flush=True)
print("DONE", flush=True)
