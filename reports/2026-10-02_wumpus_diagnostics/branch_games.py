# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/branch_games.py
# ==============================================================================
# RESTORE-AND-BRANCH GAMES: COUNTERFACTUAL "TURN, THEN STEP" DATA AT THE AGENT'S OWN DECISIONS, EVERY OUTCOME KEPT
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (2026-10-29, after every collection rule failed to reach the declined decisions; coverage_games.py 4-7):
#    - Only data at the declined "turn towards the unvisited safe cell, then step in" decisions ever repaired the
#      turned-latent observation model (coverage_games.py cover arm: glitter ratio +0.14 to +0.32). Those data were
#      privileged and safety-filtered, and the model learned "new cells are safe" (deaths +28 points). The agent's own
#      unfiltered trials kept the risk calibrated but their dose saturates: a risky trial ends the game before the
#      next declined decision (section 7 of coverage_games.py: 0.82 declined per game at trial probability 1.0, 85%
#      of the games lost).
#    - Restore-and-branch decouples the dose from the deaths. Each control game is played to its end; then the
#      simulator is RESTORED to a stored decision of that game and a disposable branch plays the counterfactual,
#      while the game itself goes on as played. The branch point and the branch action depend only on the agent's
#      own history and its own root Q; every outcome, deaths included, is kept. Theory: restart / reset
#      distributions (Kakade & Langford 2002, conservative policy iteration's mu-reset; Bagnell et al. 2003, PSDP),
#      local simulator access (Mhammedi, Foster & Rakhlin 2024: resetting to previously visited states makes
#      otherwise intractable exploration tractable), "return, then explore" (Ecoffet et al. 2021, Go-Explore), and
#      model-based DAgger (Ross & Bagnell 2012: the model must be trained on the planner's counterfactual choices).
#    - Access model (stated in the paper and in any protocol registration): restoring a state the agent has itself
#      visited is extra ENVIRONMENT access (local simulator access), not privileged CHOICE. The branch steps are the
#      environment steps it costs; the prefix is restored, not re-executed.
#    - Novelty check (2026-10-29 search): reset access is studied for policy learning (Mhammedi et al. 2024; the
#      agnostic-RL access study, arXiv 2504.05405; Single- vs Repeated-Reset rollouts in POMDPs, arXiv 2511.20870),
#      Go-Explore variants pair resets with world models for goal reaching (PEG, arXiv 2303.13002; MUN, arXiv
#      2411.02446), and coverage gaps of world models drive curiosity (arXiv 2606.27326). None found restores to the
#      agent's own declined decisions to repair a learned observation model's value of information in a POMDP.
#    - Selection-bias claim (a paper result if it holds): choosing the branch points from the history alone leaves
#      P(o' | h, a) unbiased at the branched pairs; choosing them on the hidden state (the cover arm's safety filter)
#      biases it. BRANCH vs cover tests it on the risk calibration.
#
# 2. Method (usage: uv run python <this> <run dir> <branch|random> <out .pt> [<batches>=4] [<games>=512]):
#    - Control replay: for each batch k the run's best.pt greedy planner plays coverage_games.py's control stream
#      (simulator stream_seed(COVERAGE_BASE, k), search 100 + k, agent 200 + k), so the games equal batch k of
#      runs/wumpus/coverage/games_control_seed<s>.pt (asserted on the actions when that file exists). play_episodes
#      also returns the hidden states s_0..s_T. A shadow exact-belief heuristic (risk 0.4) LABELS the decisions
#      (coverage_games.py count_dose's definitions: declined, unvisited, risky, wall); it never chooses.
#    - Branch points: live decisions (no GAME_OVER percept yet) with t <= T - 2.
#        branch: the greedy action is not FORWARD and the decision is among the first RUN_CAP = 4 of a run of
#                consecutive non-FORWARD decisions (later decisions of a run mostly repeat an idle loop). The CPU
#                estimate on the stored labels of target_rule_smoke.py (seed 0, batch 0): 13.4 branches and 0.98
#                declined per game; every non-FORWARD decision would give 23.0 and 1.17.
#        random: the same number of branch points per game, uniformly among the game's live decisions
#                (generator stream_seed(COVERAGE_BASE, 400 + k)); the attribution control at equal volume.
#    - Branch: the turn whose root Q in the control replay is higher (the agent's own estimate at that decision),
#      then FORWARD, then BRANCH_TAIL = 3 greedy planner steps, truncated at the window. Branch points are grouped
#      by their time t_b (<= 512 rows per group): a WumpusSimulator is restored with load_state_dict (s_{t_b},
#      elapsed t_b), a fresh greedy PlanningAgent (search 500 + k, agent 600 + k) is warmed by filtering the stored
#      prefix (no search), and the branch is played.
#    - Output (EpisodeBatch-shaped dict, full window T): the stored prefix, the branch steps, then padding (the last
#      branch action and percept repeated, reward 0), with loss_mask True on the branch steps only (trainer.py 2j,
#      objective_ab.py 2q). Plus per-branch rows (game, t_b, turn, labels, died, won) and the counts.
#    - Self-check per batch: 16 restored rows replay their stored next action and must reproduce the stored percept.
#
# 3. Pre-registered readout (written 2026-10-29 before any run; recipe runs runs/sweeps/wumpus_retrain seeds 0-2,
#    against the existing control games and control retrains of coverage_games.py, runs/wumpus/coverage/):
#    - DOSE (smoke: seed 0, 1 x 512 games, branch arm): >= 0.9 declined branches per control game. The handoff's bar
#      was 1.3 (cover's dose), which the stream cannot reach: it holds only 1.17 declined decisions per game. FAIL
#      -> report, no screen.
#    - Screen per seed s and arm (branch, random): 4 x 512 games -> objective_ab.py 2j/2q retrain (--extra-games
#      this file) -> feature_localise.py --networks -> retrain_play.py <run> control <arm> -> risk_localise.py.
#        MECHANISM: glitter ratio at the declined turned latents >= control + 0.15 on >= 2 of 3 seeds.
#        RISK: predicted risk at the chosen risky FORWARDs >= 50% of exact (control 2-5%) and deaths <= control + 2
#              points on every seed.
#        PLAY: branch - control >= +30 per game, paired on the analysis games, on >= 2 of 3 seeds (then seeds 3-4).
#        ATTRIBUTION: branch beats random on MECHANISM (glitter ratio) on >= 2 of 3 seeds.
#      All four PASS -> an in-agent default-off option (agent.branch_restore; Tiger fingerprint), then a protocol
#      registration before held-out seeds. MECHANISM PASS + RISK FAIL -> the risk error survives unbiased branch
#      data (optimizer's curse not a data artefact). MECHANISM FAIL at full dose -> data at the agent's own
#      decisions do not repair the turned-latent model: the cover arm's repair came from its privileged choice.
#    - Reported alongside: live branch transitions vs buffer transitions, environment steps per arm, branch deaths.
# ==============================================================================

import sys
import time
from collections import deque
from pathlib import Path

import torch

from ipomdp.agents import PlanningAgent, WumpusReferenceAgent
from ipomdp.agents.wumpus_reference import _ZERO, _ahead, _first_actions
from ipomdp.domain import WumpusSimulator, game_outcomes
from ipomdp.domain.wumpus import GAME_OVER, NUM_CELLS, WumpusAction
from ipomdp.experiments.runs import load_trained_run
from ipomdp.planning import BeliefTreeSearch
from ipomdp.training import play_episodes, stream_seed

COVERAGE_BASE = 20261020                     # coverage_games.py: the control stream
RUN_CAP, BRANCH_TAIL, GROUP = 4, 3, 512
run_dir, arm, out_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
batches = int(sys.argv[4]) if len(sys.argv) > 4 else 4
GAMES = int(sys.argv[5]) if len(sys.argv) > 5 else 512
if arm not in ("branch", "random"):
    raise ValueError(f"arm must be branch or random, not {arm}.")
dev = torch.device("cuda")
trained = load_trained_run(run_dir, dev)
world, run, cfg = trained.setup.domain, trained.run, trained.cfg
model = run.eval_agent.model
T = cfg.env.max_steps
FWD, TURNS = int(WumpusAction.FORWARD), (int(WumpusAction.TURN_LEFT), int(WumpusAction.TURN_RIGHT))
control_path = Path("runs/wumpus/coverage") / f"games_control_{run_dir.name}.pt"


def target_distance(cell: int, heading: int, safe: list[bool], is_target) -> int:
    """Fewest actions (FORWARD through safe cells, turns) until the hunter stands on a target cell (coverage_games.py)."""
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


def planner_for(rows: int, search_seed: int, agent_seed: int) -> PlanningAgent:
    return PlanningAgent(
        model, BeliefTreeSearch(model, cfg.mcts.num_simulations, cfg.mcts.c_puct, cfg.mcts.dirichlet_alpha,
                                dirichlet_epsilon=0.0, seed=search_seed, observation_prune=cfg.mcts.observation_prune),
        rows, temperature=0.0, seed=agent_seed, device=dev)


class LabelledReplay:
    """The greedy control game; records the root Q and the shadow heuristic's labels of every decision (counts only)."""

    def __init__(self, planner: PlanningAgent):
        self.planner, self.batch_size = planner, GAMES
        self.shadow = WumpusReferenceAgent(world, GAMES, 0.4)
        self.q, self.labels, self.t, self.start = [], [], 0, time.time()

    def reset(self, initial_observation):
        self.planner.reset(initial_observation)
        self.shadow.reset(initial_observation)

    def update(self, action, observation):
        self.planner.update(action, observation)
        self.shadow.update(action, observation)

    def act(self):
        s = self.shadow
        m = s._tracker.marginals(s._consistent, s._own)
        heuristic = s.act().tolist()
        chosen = self.planner.act()
        q = torch.from_numpy(self.planner.planner.statistics.q_values).float()
        self.q.append(q)
        pit, wumpus, gold = m.pit.cpu().tolist(), m.wumpus.cpu().tolist(), m.gold.cpu().tolist()
        alive = m.wumpus_alive.tolist()
        row = []
        for b in range(GAMES):
            turn = TURNS[0] if q[b, TURNS[0]] >= q[b, TURNS[1]] else TURNS[1]
            row.append(self.label(b, turn, heuristic[b], pit[b], wumpus[b], gold[b], alive[b], int(m.cell[b]),
                                  int(m.heading[b])))
        self.labels.append(row)
        self.t += 1
        if self.t % 10 == 0:
            print(f"   [{time.strftime('%H:%M:%S')}] replay decision {self.t}/{T}, {time.time() - self.start:.0f}s",
                  flush=True)
        return chosen

    def label(self, b, turn, heuristic, pit, wumpus, gold, alive, cell, heading):
        """(wall, unvisited, risky, declined) of "turn, then FORWARD" here (coverage_games.py count_dose)."""
        target = _ahead(cell, (heading + 1) % 4 if turn == int(WumpusAction.TURN_LEFT) else (heading + 3) % 4)
        if target is None:
            return (True, False, False, False)
        unvisited = not self.shadow._visited[b][target]
        risky = pit[target] + (wumpus[target] if alive else 0.0) >= _ZERO
        declined = False
        if heuristic == turn and gold[cell] <= 1.0 - _ZERO:
            danger = [pit[c] + (wumpus[c] if alive else 0.0) for c in range(NUM_CELLS)]
            safe = [d < _ZERO for d in danger]
            reachable = {c for c, _ in _first_actions(cell, heading, safe)} | {cell}
            known = [c for c in reachable if gold[c] > 1.0 - _ZERO]
            goal = (lambda c: c in known) if known else (lambda c: safe[c] and not self.shadow._visited[b][c]
                                                          and c in reachable)
            declined = target_distance(cell, heading, safe, goal) == 2
        return (False, unvisited, risky, declined)


def branch_points(episodes, k: int) -> list[tuple[int, int]]:
    """(game, t_b) pairs of this arm; liveness from the agent's own percepts (no GAME_OVER among o_0..o_t)."""
    actions = episodes.actions.cpu()
    seen = torch.cat([episodes.initial_observations.cpu().unsqueeze(1), episodes.observations.cpu()], 1) == GAME_OVER
    live = ~torch.cumsum(seen.long(), 1).bool()[:, :T]                       # decision t live iff o_0..o_t not GAME_OVER
    live[:, T - 1:] = False                                                  # turn and FORWARD must fit in the window
    own = []
    for g in range(GAMES):
        run_len = 0
        for t in range(T):
            run_len = run_len + 1 if actions[g, t] != FWD else 0
            if live[g, t] and actions[g, t] != FWD and run_len <= RUN_CAP:
                own.append((g, t))
    if arm == "branch":
        return own
    counts = torch.bincount(torch.tensor([g for g, _ in own], dtype=torch.int64), minlength=GAMES)
    generator = torch.Generator().manual_seed(stream_seed(COVERAGE_BASE, 400 + k))
    points = []
    for g in range(GAMES):
        candidates = live[g].nonzero().squeeze(1)
        pick = candidates[torch.randperm(len(candidates), generator=generator)[:int(counts[g])]]
        points += [(g, int(t)) for t in pick.sort().values]
    return points


def play_branches(episodes, states, q, points, k: int) -> dict:
    """Restores each (game, t_b), plays turn, FORWARD and BRANCH_TAIL greedy steps; returns the masked episodes."""
    acts, obs, rews = episodes.actions.cpu(), episodes.observations.cpu(), episodes.rewards.cpu()
    out = {"actions": [], "observations": [], "rewards": [], "initial_observations": [], "loss_mask": [], "rows": []}
    by_time: dict[int, list[int]] = {}
    for g, t in points:
        by_time.setdefault(t, []).append(g)
    for t_b in sorted(by_time):
        for start in range(0, len(by_time[t_b]), GROUP):
            games = torch.tensor(by_time[t_b][start:start + GROUP], dtype=torch.int64)
            n, steps = len(games), min(2 + BRANCH_TAIL, T - t_b)
            sim = WumpusSimulator(world, n, T, 0, dev)
            sim.load_state_dict({"generator": sim.state_dict()["generator"], "state": states[games.to(dev), t_b],
                                 "elapsed": torch.full((n,), t_b, dtype=torch.int64, device=dev)})
            agent = planner_for(n, stream_seed(COVERAGE_BASE, 500 + k), stream_seed(COVERAGE_BASE, 600 + k))
            agent.reset(episodes.initial_observations[games.to(dev)])
            for t in range(t_b):
                agent.update(acts[games, t].to(dev), obs[games, t].to(dev))
            qb = q[t_b][games]
            turn = torch.where(qb[:, TURNS[0]] >= qb[:, TURNS[1]], TURNS[0], TURNS[1]).to(dev)
            a_seq, o_seq, r_seq = [], [], []
            for j in range(steps):
                action = turn if j == 0 else torch.full_like(turn, FWD) if j == 1 else agent.act()
                step = sim.step(action)
                agent.update(action, step.observation)
                a_seq.append(action.cpu()), o_seq.append(step.observation.cpu()), r_seq.append(step.reward.cpu())
            a_b, o_b, r_b = torch.stack(a_seq, 1), torch.stack(o_seq, 1), torch.stack(r_seq, 1)
            pad = T - t_b - steps
            out["actions"].append(torch.cat([acts[games, :t_b], a_b, a_b[:, -1:].expand(n, pad)], 1))
            out["observations"].append(torch.cat([obs[games, :t_b], o_b, o_b[:, -1:].expand(n, pad)], 1))
            out["rewards"].append(torch.cat([rews[games, :t_b], r_b, torch.zeros(n, pad)], 1))
            out["initial_observations"].append(episodes.initial_observations.cpu()[games])
            mask = torch.zeros(n, T, dtype=torch.bool)
            mask[:, t_b:t_b + steps] = True
            out["loss_mask"].append(mask)
            died, won = (r_b < -500).any(1), (r_b > 500).any(1)
            out["rows"] += [(k, int(g), t_b, int(turn[i]), bool(died[i]), bool(won[i])) for i, g in enumerate(games)]
            del sim, agent
    return out


def self_check(episodes, states) -> None:
    """16 rows restored at mid-game decisions replay their stored action and must reproduce the stored percept."""
    t_b, games = T // 2, torch.arange(16)
    sim = WumpusSimulator(world, 16, T, 0, dev)
    sim.load_state_dict({"generator": sim.state_dict()["generator"], "state": states[games.to(dev), t_b],
                         "elapsed": torch.full((16,), t_b, dtype=torch.int64, device=dev)})
    step = sim.step(episodes.actions[games.to(dev), t_b])
    if not torch.equal(step.observation, episodes.observations[games.to(dev), t_b]):
        raise RuntimeError("restore self-check failed: a restored state does not reproduce the stored percept.")


print(f"== {run_dir.name} best.pt (collection {trained.checkpoint_collection}), arm {arm}, {batches} x {GAMES} games",
      flush=True)
control = torch.load(control_path) if control_path.exists() and GAMES == 512 else None
parts, rows, labels_all, env_steps = [], [], [], 0
for k in range(batches):
    replay = LabelledReplay(planner_for(GAMES, stream_seed(COVERAGE_BASE, 100 + k), stream_seed(COVERAGE_BASE, 200 + k)))
    episodes, states = play_episodes(WumpusSimulator(world, GAMES, T, stream_seed(COVERAGE_BASE, k), dev), replay)
    if control is not None and not torch.equal(episodes.actions.cpu(), control["actions"][k * GAMES:(k + 1) * GAMES]):
        raise RuntimeError(f"batch {k}: the replay differs from {control_path}.")
    self_check(episodes, states)
    points = branch_points(episodes, k)
    branches = play_branches(episodes, states, replay.q, points, k)
    lab = [replay.labels[t][g] for g, t in points]
    labels_all += lab
    steps_live = int(torch.cat(branches["loss_mask"]).sum())
    env_steps += steps_live
    died = sum(r[4] for r in branches["rows"])
    print(f"  batch {k}: {len(points)} branches ({len(points) / GAMES:.2f} per game); per game: declined "
          f"{sum(x[3] for x in lab) / GAMES:.2f}, unvisited {sum(x[1] for x in lab) / GAMES:.2f}, risky "
          f"{sum(x[2] for x in lab) / GAMES:.2f}, wall {sum(x[0] for x in lab) / GAMES:.2f}; branch deaths "
          f"{died} ({died / max(len(points), 1):.1%}); {steps_live} live steps; control outcomes "
          + str({key: round(float(v), 4) for key, v in game_outcomes(episodes.rewards).items()}), flush=True)
    parts.append(branches)
    rows += branches["rows"]
    del replay, episodes, states
    torch.cuda.empty_cache()
data = {key: torch.cat([torch.cat(p[key]) for p in parts]) for key in
        ("actions", "observations", "rewards", "initial_observations", "loss_mask")}
dose = {name: sum(x[i] for x in labels_all) for i, name in enumerate(("wall", "unvisited", "risky", "declined"))}
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(data | {"arm": arm, "overrides": len(rows), "dose": dose, "env_steps": env_steps, "control_games": batches * GAMES,
                   "branch_rows": torch.tensor([r[:4] for r in rows], dtype=torch.int64),
                   "branch_outcomes": torch.tensor([r[4:] for r in rows], dtype=torch.bool),
                   "branch_labels": torch.tensor(labels_all, dtype=torch.bool), "run": str(run_dir)}, out_path)
print(f"  saved {out_path}: {len(rows)} branches over {batches * GAMES} control games; declined per game "
      f"{dose['declined'] / (batches * GAMES):.3f}; dose {dose}; {env_steps} environment steps", flush=True)
print("DONE", flush=True)
