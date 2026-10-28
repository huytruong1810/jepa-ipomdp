# ABSOLUTE PATH: tests/test_training.py
"""Unit tests for the whole-episode replay buffer and the world-model trainer."""

import pytest
import torch

from ipomdp.domain import BatchedPOMDPEnv, build_tiger_pomdp
from ipomdp.models import (BeliefFilter, HorizonValueHead, LatentPredictor, ObservationHead, ObservationRewardHead, RecurrentJEPA,
                           RewardHead, TwoHotSymlog, ValueHead)
from ipomdp.training import EpisodeBatch, EpisodeBuffer, Representation, TrainerConfig, WorldModelTrainer

CPU = torch.device("cpu")
A, O, D, H, T = 3, 2, 16, 32, 6
CONFIG = TrainerConfig(learning_rate=3e-4, weight_decay=1e-4, grad_clip_norm=1.0, value_target_momentum=0.99)


def _episodes(batch: int, seed: int, device: torch.device = CPU) -> EpisodeBatch:
    pomdp = build_tiger_pomdp()
    env = BatchedPOMDPEnv(pomdp, batch, T, seed, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    actions, observations, rewards = [], [], []
    for _ in range(T):
        action = torch.randint(0, A, (batch,), generator=generator, device=device)
        out = env.step(action)
        actions.append(action)
        observations.append(out.observation)
        rewards.append(out.reward)
    return EpisodeBatch(torch.stack(actions, 1), torch.stack(observations, 1), torch.stack(rewards, 1), None)


def _trainer(device: torch.device = CPU, representation: Representation = Representation.JEPA) -> WorldModelTrainer:
    torch.manual_seed(0)
    belief_filter = BeliefFilter(A, O, D, H, 1).to(device)
    jepa = (RecurrentJEPA(belief_filter, LatentPredictor(D, A, H, 1), 0.99).to(device)
            if representation is Representation.JEPA else None)
    return WorldModelTrainer(belief_filter, jepa, ValueHead(D, H, 1, 255).to(device),
                             RewardHead(D, A, H, 1, 255).to(device), ObservationHead(D, A, O, H, 1).to(device),
                             TwoHotSymlog(255, 2000.0).to(device), A, O, discount=0.95, config=CONFIG, device=device)


class TestEpisodeBuffer:

    def test_add_sample_and_wraparound(self):
        buffer = EpisodeBuffer(capacity=5, episode_length=T, initial_observation=False, device=CPU, seed=0)
        with pytest.raises(RuntimeError):
            buffer.sample(1)
        first, second = _episodes(3, 0), _episodes(3, 1)
        buffer.add(first, greedy=False)
        assert buffer.size == 3
        buffer.add(second, greedy=True)  # 6 episodes into capacity 5: the oldest is overwritten
        assert buffer.size == 5
        assert torch.equal(buffer._actions[0], second.actions[2])
        assert torch.equal(buffer._actions[1], first.actions[1])
        assert buffer._greedy.tolist() == [True, False, False, True, True]     # slot 0 was overwritten by `second`
        sample = buffer.sample(8)
        assert sample.actions.shape == (8, T) and sample.rewards.dtype == torch.float32
        assert sample.greedy.dtype == torch.bool and sample.greedy.shape == (8,)

    def test_rejects_wrong_episode_length(self):
        buffer = EpisodeBuffer(capacity=5, episode_length=T + 1, initial_observation=False, device=CPU, seed=0)
        with pytest.raises(ValueError):
            buffer.add(_episodes(2, 0), greedy=False)

    def test_sampling_is_seeded(self):
        def draw(seed):
            buffer = EpisodeBuffer(capacity=16, episode_length=T, initial_observation=False, device=CPU, seed=seed)
            buffer.add(_episodes(16, 0), greedy=False)
            return buffer.sample(8).actions
        assert torch.equal(draw(3), draw(3))


class TestWorldModelTrainer:

    def test_value_targets_are_bellman_optimality_backups(self):
        # V_target(z) = max_a [R(z,a) + gamma sum_o P(o|z,a) V-bar(tau(z,a,o))], computed by hand.
        trainer = _trainer()
        latents = torch.randn(5, D)
        codec, filt = trainer.twohot, trainer.belief_filter
        expected = []
        with torch.no_grad():
            for a in range(A):
                action = torch.nn.functional.one_hot(torch.full((5,), a), A).float()
                q = codec.mean(trainer.reward_head(latents, action))
                probs = torch.softmax(trainer.observation_head(latents, action), -1)
                for o in range(O):
                    child = filt.step(latents, action, torch.nn.functional.one_hot(torch.full((5,), o), O).float())
                    q = q + 0.95 * probs[:, o] * codec.mean(trainer.target_value_head(child))
                expected.append(q)
        assert torch.allclose(trainer.value_targets(latents), torch.stack(expected, -1).max(-1).values, atol=1e-4)

    def test_finite_horizon_targets_end_the_window(self):
        # trainer.py, section 3b: at k = 1 the target is max_a R(z, a) (no bootstrap); train_step runs on whole windows.
        torch.manual_seed(0)
        belief_filter = BeliefFilter(A, O, D, H, 1)
        reward_head = RewardHead(D, A, H, 1, 255)
        torch.nn.init.normal_(reward_head.net[-1].weight)
        trainer = WorldModelTrainer(belief_filter, None, HorizonValueHead(D, H, 1, 255, T), reward_head,
                                    ObservationHead(D, A, O, H, 1), TwoHotSymlog(255, 2000.0), A, O, discount=0.95,
                                    config=CONFIG, device=CPU, horizon=T)
        latents = torch.randn(5, D)
        with torch.no_grad():
            rewards = torch.stack([trainer.twohot.mean(reward_head(latents, torch.nn.functional.one_hot(
                torch.full((5,), a), A).float())) for a in range(A)], -1)
        last = torch.cat([latents, torch.ones(5, 1)], -1)
        assert torch.allclose(trainer.value_targets(last), rewards.max(-1).values, atol=1e-4)
        metrics = trainer.train_step(_episodes(4, 0))
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        short = _episodes(4, 0)
        with pytest.raises(ValueError):                              # only whole windows (T = H)
            trainer.train_step(EpisodeBatch(short.actions[:, :-1], short.observations[:, :-1], short.rewards[:, :-1], None))

    def test_nstep_greedy_returns(self):
        # trainer.py, section 3c: G_t = sum_{i<m} gamma^i r_{t+i} + gamma^m V-bar(z_{t+m}), m = min(n, T - t).
        torch.manual_seed(0)
        belief_filter = BeliefFilter(A, O, D, H, 1)
        trainer = WorldModelTrainer(belief_filter, None, ValueHead(D, H, 1, 255), RewardHead(D, A, H, 1, 255),
                                    ObservationHead(D, A, O, H, 1), TwoHotSymlog(255, 2000.0), A, O, discount=0.95,
                                    config=CONFIG, device=CPU, nstep=2, nstep_mix=0.5)
        torch.nn.init.normal_(trainer.target_value_head.net[-1].weight)
        states, rewards = torch.randn(3, T + 1, D), torch.randn(3, T)
        with torch.no_grad():
            boot = trainer.twohot.mean(trainer.target_value_head(states.flatten(0, 1))).view(3, T + 1)
        expected = torch.zeros(3, T)
        for t in range(T):
            m = min(2, T - t)
            expected[:, t] = sum(0.95 ** i * rewards[:, t + i] for i in range(m)) + 0.95 ** m * boot[:, t + m]
        assert torch.allclose(trainer.nstep_returns(states, rewards), expected, atol=1e-4)
        episodes = _episodes(4, 0)
        with pytest.raises(ValueError):                              # needs the buffer's greedy flags
            trainer.train_step(episodes)
        greedy = EpisodeBatch(episodes.actions, episodes.observations, episodes.rewards, None,
                              torch.tensor([True, False, True, False]))
        metrics = trainer.train_step(greedy)
        assert metrics["value_nstep_rows"] == 2 * T
        with pytest.raises(ValueError):
            WorldModelTrainer(belief_filter, None, ValueHead(D, H, 1, 255), RewardHead(D, A, H, 1, 255),
                              ObservationHead(D, A, O, H, 1), TwoHotSymlog(255, 2000.0), A, O, discount=0.95,
                              config=CONFIG, device=CPU, nstep=2, nstep_mix=0.0)

    def test_target_value_head_tracks_online_head_by_ema(self):
        trainer = _trainer()
        before = [p.clone() for p in trainer.target_value_head.parameters()]
        trainer.train_step(_episodes(4, 0))
        for p, q, q0 in zip(trainer.value_head.parameters(), trainer.target_value_head.parameters(), before):
            assert torch.allclose(q, 0.99 * q0 + 0.01 * p, atol=1e-6)

    def test_train_step_updates_parameters_and_target(self):
        trainer = _trainer()
        filt, target = trainer.belief_filter, trainer.jepa.target_filter
        before = [p.clone() for p in filt.parameters()]
        target_before = [p.clone() for p in target.parameters()]
        metrics = trainer.train_step(_episodes(8, 0))
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        assert any(not torch.equal(a, b) for a, b in zip(before, filt.parameters()))
        assert any(not torch.equal(a, b) for a, b in zip(target_before, target.parameters()))

    def test_non_finite_loss_raises(self):
        trainer = _trainer()
        batch = _episodes(4, 0)
        broken = EpisodeBatch(batch.actions, batch.observations, batch.rewards * float("nan"), None)
        with pytest.raises(FloatingPointError):
            trainer.train_step(broken)

    def test_observation_head_does_not_shape_the_representation(self):
        # The planning model is trained on detached latents: its loss alone must leave the
        # belief filter's gradients at zero.
        trainer = _trainer()
        batch = _episodes(4, 0)
        actions = torch.nn.functional.one_hot(batch.actions, A).float()
        observations = torch.nn.functional.one_hot(batch.observations, O).float()
        latents = trainer.belief_filter.unroll(actions, observations, None)[:, :-1].reshape(-1, D)
        logits = trainer.observation_head(latents.detach(), actions.reshape(-1, A))
        torch.nn.functional.cross_entropy(logits, batch.observations.reshape(-1)).backward()
        assert all(p.grad is None for p in trainer.belief_filter.parameters())

    def test_jepa_must_wrap_the_trained_filter(self):
        belief_filter = BeliefFilter(A, O, D, H, 1)
        foreign = RecurrentJEPA(BeliefFilter(A, O, D, H, 1), LatentPredictor(D, A, H, 1), 0.99)
        with pytest.raises(ValueError):
            WorldModelTrainer(belief_filter, foreign, ValueHead(D, H, 1, 255), RewardHead(D, A, H, 1, 255),
                              ObservationHead(D, A, O, H, 1), TwoHotSymlog(255, 2000.0), A, O, 0.95, CONFIG, CPU)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_train_step_on_cuda_with_bfloat16_autocast(self):
        trainer = _trainer(torch.device("cuda"))
        metrics = trainer.train_step(_episodes(8, 0, torch.device("cuda")))
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


class TestDecoderBaseline:
    """Section 2b of training/trainer.py: the observation head shapes the filter; no JEPA parts."""

    def test_no_jepa_parts_and_no_prediction_loss(self):
        trainer = _trainer(representation=Representation.DECODER)
        assert trainer.jepa is None and trainer.representation is Representation.DECODER
        metrics = trainer.train_step(_episodes(8, 0))
        assert "loss_prediction" not in metrics
        assert {"loss_reward", "loss_value", "loss_observation"} <= metrics.keys()
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())

    def test_observation_likelihood_reaches_the_filter(self):
        # With zero rewards the reward term cannot move the filter on the first step (its final
        # layer is zero-initialised, so no gradient flows back into the latent). Without a JEPA
        # term, any filter gradient must then come from the observation cross-entropy.
        trainer = _trainer(representation=Representation.DECODER)
        batch = _episodes(4, 0)
        trainer.optimizer.step = lambda: None  # inspect the gradients, keep the weights
        trainer.train_step(EpisodeBatch(batch.actions, batch.observations, torch.zeros_like(batch.rewards), None))
        assert sum(float(p.grad.abs().sum()) for p in trainer.belief_filter.parameters()) > 0.0


class TestValueEquivalentTerm:
    """Section 2f of training/trainer.py: (sum_o P(o) V-bar(tau(z, a, o)) - V-bar(z')) / max|R|, squared."""

    @staticmethod
    def _decoder(weight: float, scale: float = 100.0) -> WorldModelTrainer:
        torch.manual_seed(0)
        belief_filter = BeliefFilter(A, O, D, H, 1)
        return WorldModelTrainer(belief_filter, None, ValueHead(D, H, 1, 255), RewardHead(D, A, H, 1, 255),
                                 ObservationHead(D, A, O, H, 1), TwoHotSymlog(255, 2000.0), A, O, discount=0.95,
                                 config=CONFIG, device=CPU, value_equivalent_weight=weight, value_equivalent_scale=scale)

    def test_off_by_default(self):
        assert "loss_value_equivalent" not in _trainer(representation=Representation.DECODER).train_step(_episodes(4, 0))

    def test_needs_the_decoder_and_a_valid_weight(self):
        torch.manual_seed(0)
        belief_filter = BeliefFilter(A, O, D, H, 1)
        with pytest.raises(ValueError):
            WorldModelTrainer(belief_filter, RecurrentJEPA(belief_filter, LatentPredictor(D, A, H, 1), 0.99),
                              ValueHead(D, H, 1, 255), RewardHead(D, A, H, 1, 255), ObservationHead(D, A, O, H, 1),
                              TwoHotSymlog(255, 2000.0), A, O, discount=0.95, config=CONFIG, device=CPU,
                              value_equivalent_weight=1.0)
        with pytest.raises(ValueError):
            self._decoder(-1.0)

    def test_matches_the_value_equivalence_residual_by_hand(self):
        trainer = self._decoder(2.0)
        with torch.no_grad():                         # a non-trivial V-bar (the projection is zero-initialised)
            for p in trainer.target_value_head.parameters():
                p.normal_(0.0, 0.5)
        batch = _episodes(4, 0)
        with torch.no_grad():
            a1 = torch.nn.functional.one_hot(batch.actions, A).float()
            latents = trainer.belief_filter.unroll(a1, torch.nn.functional.one_hot(batch.observations, O).float(), None)
            z = latents[:, :-1].reshape(-1, D)
            children = trainer.target_model.expand(z).next_values                   # (BT, A, O)
            branch = children[torch.arange(len(z)), batch.actions.reshape(-1)]      # (BT, O)
            realised = branch[torch.arange(len(z)), batch.observations.reshape(-1)]
            probs = torch.softmax(trainer.observation_head(z, a1.reshape(-1, A)), -1)
            expected = 2.0 * ((((probs * branch).sum(-1) - realised) / 100.0) ** 2).mean()
        assert float(expected) > 0.0
        assert trainer.train_step(batch)["loss_value_equivalent"] == pytest.approx(float(expected), rel=1e-4)


class TestDecisionAwareWeighting:
    """Section 2h of training/trainer.py: CE weighted by (1 - beta) + beta * span^2 / mean(span^2)."""

    @staticmethod
    def _decoder(beta: float, **kwargs) -> WorldModelTrainer:
        torch.manual_seed(0)
        belief_filter = BeliefFilter(A, O, D, H, 1)
        return WorldModelTrainer(belief_filter, None, ValueHead(D, H, 1, 255), RewardHead(D, A, H, 1, 255),
                                 ObservationHead(D, A, O, H, 1), TwoHotSymlog(255, 2000.0), A, O, discount=0.95,
                                 config=CONFIG, device=CPU, decision_aware_weight=beta, **kwargs)

    def test_invalid_weight_and_horizon_raise(self):
        for beta in (-0.1, 1.5):
            with pytest.raises(ValueError):
                self._decoder(beta)
        with pytest.raises(ValueError):
            self._decoder(0.5, horizon=6)

    def test_zero_spans_give_the_unweighted_loss(self):
        batch = _episodes(4, 0)          # V-bar is constant at initialisation (zero-initialised projection)
        assert (self._decoder(0.5).train_step(batch)["loss_observation"]
                == pytest.approx(self._decoder(0.0).train_step(batch)["loss_observation"], rel=1e-6))

    def test_matches_the_weighted_cross_entropy_by_hand(self):
        trainer = self._decoder(0.75)
        with torch.no_grad():                         # a non-trivial V-bar
            for p in trainer.target_value_head.parameters():
                p.normal_(0.0, 0.5)
        batch = _episodes(4, 0)
        with torch.no_grad():
            a1 = torch.nn.functional.one_hot(batch.actions, A).float()
            latents = trainer.belief_filter.unroll(a1, torch.nn.functional.one_hot(batch.observations, O).float(), None)
            z = latents[:, :-1].reshape(-1, D)
            rows, observed = torch.arange(len(z)), batch.observations.reshape(-1)
            branch = trainer.target_model.expand(z).next_values[rows, batch.actions.reshape(-1)]     # (BT, O)
            logits = trainer.observation_head(z, a1.reshape(-1, A))
            support = torch.softmax(logits, -1) >= 1e-3
            support[rows, observed] = True
            span = torch.stack([branch[i][support[i]].max() - branch[i][support[i]].min() for i in range(len(z))])
            weights = 0.25 + 0.75 * span ** 2 / (span ** 2).mean()
            expected = (weights * torch.nn.functional.cross_entropy(logits, observed, reduction="none")).mean()
        assert float(span.max()) > 0.0 and float(weights.std()) > 0.0
        assert trainer.train_step(batch)["loss_observation"] == pytest.approx(float(expected), rel=1e-4)


class TestQEquivalentTerm:
    """Section 2i of training/trainer.py: (sum_o P(o) [r(z, a, o) + gamma V-bar(tau)] - (r + gamma V-bar(z'))) / max|R|, squared."""

    @staticmethod
    def _decoder(weight: float, scale: float = 100.0, observation_reward: bool = True, **kwargs) -> WorldModelTrainer:
        torch.manual_seed(0)
        belief_filter = BeliefFilter(A, O, D, H, 1)
        reward_head = ObservationRewardHead(D, A, O, H, 1, 255) if observation_reward else RewardHead(D, A, H, 1, 255)
        return WorldModelTrainer(belief_filter, None, ValueHead(D, H, 1, 255), reward_head,
                                 ObservationHead(D, A, O, H, 1), TwoHotSymlog(255, 2000.0), A, O, discount=0.95,
                                 config=CONFIG, device=CPU, value_equivalent_scale=scale, q_equivalent_weight=weight,
                                 **kwargs)

    def test_off_by_default_and_invalid_combinations_raise(self):
        assert "loss_q_equivalent" not in self._decoder(0.0).train_step(_episodes(4, 0))
        with pytest.raises(ValueError):
            self._decoder(-1.0)
        with pytest.raises(ValueError):                        # needs the observation-conditioned reward
            self._decoder(1.0, observation_reward=False)
        with pytest.raises(ValueError):                        # one value-equivalence term at a time
            self._decoder(1.0, value_equivalent_weight=1.0)
        torch.manual_seed(0)
        belief_filter = BeliefFilter(A, O, D, H, 1)
        with pytest.raises(ValueError):                        # decoder only
            WorldModelTrainer(belief_filter, RecurrentJEPA(belief_filter, LatentPredictor(D, A, H, 1), 0.99),
                              ValueHead(D, H, 1, 255), ObservationRewardHead(D, A, O, H, 1, 255),
                              ObservationHead(D, A, O, H, 1), TwoHotSymlog(255, 2000.0), A, O, discount=0.95,
                              config=CONFIG, device=CPU, q_equivalent_weight=1.0)

    def test_reward_head_is_trained_on_the_realised_observation(self):
        trainer = self._decoder(0.0)
        with torch.no_grad():                                  # a non-trivial reward head
            torch.nn.init.normal_(trainer.reward_head.net[-1].weight)
        batch = _episodes(4, 0)
        with torch.no_grad():
            a1 = torch.nn.functional.one_hot(batch.actions, A).float()
            o1 = torch.nn.functional.one_hot(batch.observations, O).float()
            z = trainer.belief_filter.unroll(a1, o1, None)[:, :-1].reshape(-1, D)
            expected = trainer.twohot.loss(trainer.reward_head(z, a1.reshape(-1, A), o1.reshape(-1, O)),
                                           batch.rewards.reshape(-1)).mean()
        assert trainer.train_step(batch)["loss_reward"] == pytest.approx(float(expected), rel=1e-5)

    def test_matches_the_q_equivalence_residual_by_hand(self):
        trainer = self._decoder(2.0)
        with torch.no_grad():                                  # non-trivial V-bar and r (zero-initialised projections)
            for p in trainer.target_value_head.parameters():
                p.normal_(0.0, 0.5)
            torch.nn.init.normal_(trainer.reward_head.net[-1].weight)
        batch = _episodes(4, 0)
        with torch.no_grad():
            a1 = torch.nn.functional.one_hot(batch.actions, A).float()
            latents = trainer.belief_filter.unroll(a1, torch.nn.functional.one_hot(batch.observations, O).float(), None)
            z = latents[:, :-1].reshape(-1, D)
            rows, acts, observed = torch.arange(len(z)), batch.actions.reshape(-1), batch.observations.reshape(-1)
            expansion = trainer.target_model.expand(z)
            values = expansion.next_values[rows, acts]                                 # (BT, O)
            q = expansion.branch_rewards[rows, acts] + 0.95 * values
            realised = batch.rewards.reshape(-1) + 0.95 * values[rows, observed]
            probs = torch.softmax(trainer.observation_head(z, a1.reshape(-1, A)), -1)
            expected = 2.0 * ((((probs * q).sum(-1) - realised) / 100.0) ** 2).mean()
        assert float(expected) > 0.0
        assert trainer.train_step(batch)["loss_q_equivalent"] == pytest.approx(float(expected), rel=1e-4)

    def test_gradient_reaches_only_the_observation_head_and_the_filter(self):
        trainer = self._decoder(1.0)
        with torch.no_grad():
            for p in trainer.target_value_head.parameters():
                p.normal_(0.0, 0.5)
            torch.nn.init.normal_(trainer.reward_head.net[-1].weight)
        batch = _episodes(4, 0)
        a1 = torch.nn.functional.one_hot(batch.actions, A).float()
        latents = trainer.belief_filter.unroll(a1, torch.nn.functional.one_hot(batch.observations, O).float(), None)
        z = latents[:, :-1].reshape(-1, D)
        rows, acts, observed = torch.arange(len(z)), batch.actions.reshape(-1), batch.observations.reshape(-1)
        with torch.no_grad():
            expansion = trainer.target_model.expand(z.detach())
            values = expansion.next_values[rows, acts]
            q = expansion.branch_rewards[rows, acts] + 0.95 * values
            realised = batch.rewards.reshape(-1) + 0.95 * values[rows, observed]
        probs = torch.softmax(trainer.observation_head(z, a1.reshape(-1, A)), -1)
        (((probs * q).sum(-1) - realised) ** 2).mean().backward()
        grads = lambda net: [p.grad for p in net.parameters()]  # noqa: E731
        assert all(g is not None and g.abs().sum() > 0 for g in grads(trainer.observation_head)[-1:])
        assert any(g is not None and g.abs().sum() > 0 for g in grads(trainer.belief_filter))
        assert all(g is None for g in grads(trainer.reward_head) + grads(trainer.value_head)
                   + grads(trainer.target_value_head))
