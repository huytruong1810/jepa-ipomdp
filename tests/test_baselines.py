# ABSOLUTE PATH: tests/test_baselines.py
# ==============================================================================
# TESTS OF THE MODEL-FREE RECURRENT BASELINE (src/ipomdp/baselines/recurrent_dqn.py)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. What Is Checked:
#    - The value rescaling h and its closed-form inverse round-trip over the Wumpus return range.
#    - n-step targets equal a brute-force per-step sum, including the shortened horizon and the
#      bootstrap at h_T near the window end (section 3).
#    - The acting agent's recurrent state equals the training unroll's on the same episodes, so
#      the network is trained on exactly the histories it acts on.
#    - A tiny training run is deterministic given its seed.
# ==============================================================================

import pytest
import torch

from ipomdp.baselines import (RecurrentDQNAgent, RecurrentDQNConfig, RecurrentQNetwork, apex_epsilons, nstep_targets,
                              train_recurrent_dqn, value_rescale, value_rescale_inverse)
from ipomdp.domain import WumpusSimulator, WumpusWorld
from ipomdp.training import play_episodes

CPU = torch.device("cpu")


class TestRecurrentDQN:
    def test_value_rescale_round_trips(self):
        x = torch.linspace(-1e5, 1e5, 2001, dtype=torch.float64)
        assert torch.allclose(value_rescale_inverse(value_rescale(x)), x, rtol=1e-9, atol=1e-6)

    def test_nstep_targets_match_brute_force(self):
        generator = torch.Generator().manual_seed(0)
        rewards = torch.randn(3, 7, generator=generator, dtype=torch.float64)
        bootstrap = torch.randn(3, 8, generator=generator, dtype=torch.float64)
        gamma, n = 0.9, 3
        expected = torch.zeros_like(rewards)
        for t in range(7):
            horizon = min(n, 7 - t)
            expected[:, t] = sum(gamma ** k * rewards[:, t + k] for k in range(horizon)) + gamma ** horizon * bootstrap[:, t + horizon]
        assert torch.allclose(nstep_targets(rewards, bootstrap, gamma, n), expected)

    def test_apex_ladder_endpoints(self):
        eps = apex_epsilons(64)
        assert eps[0].item() == 0.4 and abs(eps[-1].item() - 0.4 ** 8) < 1e-12 and bool((eps.diff() < 0).all())

    def test_acting_state_equals_training_unroll(self):
        world = WumpusWorld(CPU)
        torch.manual_seed(0)
        network = RecurrentQNetwork(world.num_observations, world.num_actions, 32)
        agent = RecurrentDQNAgent(network, torch.full((4,), 0.5), seed=1, device=CPU)
        q_acting = []

        class Recorder:
            batch_size = 4

            def reset(self, initial_observation):
                agent.reset(initial_observation)

            def act(self):
                q_acting.append(network.q_values(agent._hidden))
                return agent.act()

            def update(self, action, observation):
                agent.update(action, observation)

        episodes, _ = play_episodes(WumpusSimulator(world, 4, 10, 3, CPU), Recorder())
        with torch.no_grad():
            q_unroll = network.unroll(episodes)
        assert torch.allclose(torch.stack(q_acting, 1), q_unroll[:, :-1], atol=1e-6)

    def test_annealed_exploration_follows_the_planner_schedule(self):
        # recurrent_dqn.py section 5: one epsilon for all rows, 0.25 in the first trained collection, x0.97 after each
        # trained collection, floor 0.1; the warm-up collections do not decay it.
        from ipomdp.baselines import recurrent_dqn as module
        seen = []
        original = module.RecurrentDQNAgent.act

        def recording_act(agent):
            if agent.batch_size == 8:
                seen.append(agent.epsilon.clone())
            return original(agent)

        module.RecurrentDQNAgent.act = recording_act
        try:
            cfg = RecurrentDQNConfig(hidden_dim=16, batch_size=4, env_batch_size=8, total_episodes=48, warmup_episodes=16,
                                     updates_per_collection=1, eval_every=100, eval_episodes=4, exploration="anneal",
                                     epsilon_start=0.25, epsilon_decay=0.5, epsilon_min=0.1)
            train_recurrent_dqn(cfg, WumpusWorld(CPU), 3, seed=0, device=CPU, log=lambda _: None)
        finally:
            module.RecurrentDQNAgent.act = original
        per_collection = [float(seen[i][0]) for i in range(0, len(seen), 3)]          # 3 decisions per game window
        assert all(bool((e == e[0]).all()) for e in seen)
        assert per_collection == pytest.approx([0.25, 0.125, 0.1, 0.1])

    def test_linear_value_scale_decodes_the_mean(self):
        # recurrent_dqn.py section 5: the linear map is unbiased for stochastic returns; h is not.
        from ipomdp.baselines.recurrent_dqn import RecurrentDQNTrainer
        world = WumpusWorld(CPU)
        network = RecurrentQNetwork(world.num_observations, world.num_actions, 8)
        returns = torch.tensor([999.0] * 3 + [-1.0] * 97, dtype=torch.float64)
        linear = RecurrentDQNTrainer(network, RecurrentDQNConfig(hidden_dim=8, value_scale="linear"), world.discount)
        rescale = RecurrentDQNTrainer(network, RecurrentDQNConfig(hidden_dim=8), world.discount)
        assert float(linear.decode(linear.encode(returns).mean())) == pytest.approx(float(returns.mean()))
        assert float(rescale.decode(rescale.encode(returns).mean())) < 0.5 * float(returns.mean())
        with pytest.raises(ValueError):
            RecurrentDQNConfig(value_scale="twohot")

    def test_tiny_training_run_is_deterministic(self):
        world = WumpusWorld(CPU)
        cfg = RecurrentDQNConfig(hidden_dim=16, batch_size=4, env_batch_size=8, total_episodes=32, warmup_episodes=8,
                                 updates_per_collection=2, eval_every=2, eval_episodes=4)
        runs = [train_recurrent_dqn(cfg, world, 10, seed=0, device=CPU, log=lambda _: None) for _ in range(2)]
        assert runs[0].evaluations == runs[1].evaluations and len(runs[0].evaluations) == 2
        assert all(torch.equal(runs[0].final_state[k], runs[1].final_state[k]) for k in runs[0].final_state)
