# ABSOLUTE PATH: tests/test_wumpus_agents.py
"""
The exact-belief Wumpus reference agent at the paper's risk threshold (agents/wumpus_reference.py, section 2).

The reference row of the paper plays at risk 0.4 (reports/2026-10-19_eval_protocol/PROTOCOL.md, "Frozen choices"),
while tests/test_wumpus.py (TestReferenceAgent) covers risk 0 only. The gamble rule (d) is checked here against an
INDEPENDENT exact filter (a second WumpusBeliefTracker fed the same actions and percepts) and an independent
cell-ahead computation:
    1. every step into an unvisited cell that is not provably safe has an exact death probability of at most the
       threshold at the moment of the decision, and never a certain one;
    2. the rule is reached (gambles happen), and on the same layouts it wins more games than risk 0, which never
       gambles and never dies.
Gambles are rare (6 in 64 games of 50 steps at seed 7; the largest exact danger taken is 1/3, no game ends in a death, and
risk 0.4 wins 68.8% of the games against 59.4% at risk 0), so the check needs whole games and is marked slow (one to
three minutes on one CPU thread: the agent's search and the exact filter run per game in Python).
"""

import pytest
import torch

from ipomdp.agents import WumpusReferenceAgent
from ipomdp.domain.wumpus import (GRID, START_CELL, Heading, WumpusAction, WumpusBeliefTracker, WumpusSimulator,
                                  WumpusWorld, game_outcomes)

CPU = torch.device("cpu")
GAMES, STEPS, SEED, RISK = 64, 50, 7, 0.4
STEP = {Heading.EAST: (1, 0), Heading.NORTH: (0, 1), Heading.WEST: (-1, 0), Heading.SOUTH: (0, -1)}


def _cell_ahead(cell: int, heading: int) -> int | None:
    dx, dy = STEP[Heading(heading)]
    x, y = cell % GRID + dx, cell // GRID + dy
    return x + GRID * y if 0 <= x < GRID and 0 <= y < GRID else None


def _play(world: WumpusWorld, risk: float) -> tuple[torch.Tensor, list[float]]:
    """Plays GAMES games; returns the rewards (B, T) and the exact death probability of every gamble taken."""
    simulator = WumpusSimulator(world, GAMES, STEPS, SEED, CPU)
    initial = simulator.reset()
    agent, tracker = WumpusReferenceAgent(world, GAMES, risk), WumpusBeliefTracker(world)
    agent.reset(initial)
    consistent, own = tracker.start(initial)
    visited = [{START_CELL} for _ in range(GAMES)]
    rewards, gambles = [], []
    for _ in range(STEPS):
        m = tracker.marginals(consistent, own)
        action = agent.act()
        for b in range(GAMES):
            if bool(m.done[b]) or int(action[b]) != WumpusAction.FORWARD:
                continue
            ahead = _cell_ahead(int(m.cell[b]), int(m.heading[b]))
            if ahead is None or ahead in visited[b]:
                continue
            danger = float(m.pit[b, ahead]) + (float(m.wumpus[b, ahead]) if bool(m.wumpus_alive[b]) else 0.0)
            if danger > 1e-12:
                gambles.append(danger)
        out = simulator.step(action)
        agent.update(action, out.observation)
        consistent, own = tracker.update(consistent, own, action, out.observation)
        for b, cell in enumerate(own.cell.tolist()):
            visited[b].add(cell)
        rewards.append(out.reward)
    return torch.stack(rewards, dim=1), gambles


@pytest.mark.slow
def test_gambles_only_within_the_threshold_and_wins_more_than_never_gambling():
    world = WumpusWorld(CPU)
    rewards, gambles = _play(world, RISK)
    safe_rewards, safe_gambles = _play(world, 0.0)
    assert gambles and max(gambles) <= RISK + 1e-9 and max(gambles) < 1.0 - 1e-12
    assert not safe_gambles
    outcomes, safe = game_outcomes(rewards), game_outcomes(safe_rewards)
    assert safe["death_rate"] == 0.0
    assert outcomes["win_rate"] > safe["win_rate"]
