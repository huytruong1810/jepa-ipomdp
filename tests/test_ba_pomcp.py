# ABSOLUTE PATH: tests/test_ba_pomcp.py
"""
BA-POMCP (src/ipomdp/baselines/ba_pomcp.py) on the canonical Tiger.

What is verified:
    1. The prior is validated, and the expected dynamics are the count ratios (paper, Eq. 1).
    2. Sampling never returns a structural zero, also at the top of the unit interval.
    3. The belief update is exact Bayesian bookkeeping: every kept particle carries exactly one new transition count at
       (a, its previous state, its new state) and one new observation count at (a, its new state, the observation
       received); the share of particles on each side matches the prior predictive; an observation the counts make
       impossible raises.
    4. With point-mass priors on the true tables the agent is POMCP on the true model. With the EXPECTED_MDP rollout it
       chooses Tiger's optimal action at every belief the optimal policy reaches. Its PLAY is not optimal at 1,000
       simulations, and no test claims it is: on the 256 paired episodes of seed 11 it loses 6.22 +- 3.27 per episode
       to the optimal agent, with a wrong door in 7% of the episodes
       (reports/2026-10-19_eval_protocol/ba_pomcp_point_mass_check.log). An earlier slow test that asserted the optimal
       return on 32 episodes passed only because its 5-standard-error tolerance was wider than that deficit, and it was
       removed. The uniform rollout's failure on the continuing Tiger is recorded in the module header, section 4b.
    5. Under the paper's formalism (rewards are not observations) a symmetric, uninformed growl prior cannot learn
       which growl means which side: the belief over the tiger stays at 1/2 whatever is heard.
    6. Runs are reproducible from the seed.
"""

import math

import numpy as np
import pytest
import torch

from ipomdp.baselines.ba_pomcp import BAPOMCPAgent, DirichletPrior, Rollout, _draw, _draw_rows
from ipomdp.domain import BatchedPOMDPEnv, TigerAction, build_tiger_pomdp

LISTEN, OPEN_LEFT, OPEN_RIGHT = int(TigerAction.LISTEN), int(TigerAction.OPEN_LEFT), int(TigerAction.OPEN_RIGHT)
GROWL_LEFT, GROWL_RIGHT = 0, 1


@pytest.fixture(scope="module")
def tiger():
    return build_tiger_pomdp()


def _paper_prior(tiger) -> DirichletPrior:
    """The paper's Tiger prior (section 5): T known with high confidence, LISTEN's growl 5 counts correct, 3 wrong."""
    return DirichletPrior.from_model(tiger, 1e6, 1e6).with_observation_counts(LISTEN, np.array([[5.0, 3.0], [3.0, 5.0]]))


def _agent(tiger, prior, particles=1000, sims=100, rollout=Rollout.EXPECTED_MDP, seed=0):
    return BAPOMCPAgent(tiger, prior, particles, sims, horizon=20, rollout=rollout, seed=seed, rejection_cap=1000)


class TestPrior:

    def test_rejects_invalid_counts(self, tiger):
        good = DirichletPrior.from_model(tiger, 10.0, 10.0)
        with pytest.raises(ValueError):
            DirichletPrior(-good.transition, good.observation)                       # negative counts
        with pytest.raises(ValueError):
            DirichletPrior(good.transition.astype(np.float32), good.observation)     # not float64
        with pytest.raises(ValueError):
            good.with_observation_counts(LISTEN, np.zeros((2, 2)))                    # a zero-sum row
        with pytest.raises(ValueError):
            DirichletPrior(good.transition, good.observation[:, :1])                 # shapes disagree on S
        with pytest.raises(ValueError):
            _agent(tiger, DirichletPrior(np.ones((2, 2, 2)), np.ones((2, 2, 2))))   # wrong number of actions

    def test_expected_dynamics_are_the_count_ratios(self, tiger):
        transition, observation = _agent(tiger, _paper_prior(tiger), particles=50).expected_dynamics()
        assert np.allclose(transition, tiger.transition.numpy())
        assert np.allclose(observation[LISTEN], [[0.625, 0.375], [0.375, 0.625]])
        assert np.allclose(observation[OPEN_LEFT], tiger.observation.numpy()[OPEN_LEFT])


class TestSampling:

    def test_never_returns_a_structural_zero(self):
        row = np.array([0.0, 2.0, 0.0, 2.0, 0.0])
        top = math.nextafter(1.0, 0.0)                                    # the largest u a generator returns
        assert [_draw(row, u) for u in (0.0, 0.49, 0.5, top)] == [1, 1, 3, 3]
        rows = np.stack([row, row, np.array([1.0, 0.0, 0.0, 0.0, 0.0])])
        assert _draw_rows(rows, np.array([0.0, top, top])).tolist() == [1, 3, 0]

    def test_frequencies_follow_the_counts(self):
        rng = np.random.default_rng(0)
        row = np.array([1.0, 0.0, 3.0])
        draws = np.array([_draw(row, rng.random()) for _ in range(20000)])
        share = float((draws == 2).mean())
        assert abs(share - 0.75) < 5 * math.sqrt(0.75 * 0.25 / 20000) and not (draws == 1).any()


class TestBeliefUpdate:

    def test_adds_exactly_the_counts_each_particle_used(self, tiger):
        prior = _paper_prior(tiger)
        agent = _agent(tiger, prior, particles=2000)
        agent.update(LISTEN, GROWL_LEFT)
        for k in range(agent.num_particles):
            dt = agent.transition_counts[k] - prior.transition
            do = agent.observation_counts[k] - prior.observation
            (a, s, s2), = np.argwhere(dt != 0)
            assert dt.sum() == 1.0 and a == LISTEN and s == s2 == agent.states[k]   # LISTEN never moves the tiger
            assert do.sum() == 1.0 and do[LISTEN, agent.states[k], GROWL_LEFT] == 1.0
        # The prior predictive P(tiger left | growl left) is 0.625 (b0 uniform, 5 / 3 counts).
        left = float((agent.states == 0).mean())
        assert abs(left - 0.625) < 5 * math.sqrt(0.625 * 0.375 / agent.num_particles)

    def test_an_impossible_observation_raises(self, tiger):
        certain = DirichletPrior.from_model(tiger, 10.0, 10.0).with_observation_counts(
            LISTEN, np.array([[1.0, 0.0], [1.0, 0.0]]))                              # a right growl cannot happen
        agent = BAPOMCPAgent(tiger, certain, 50, 10, horizon=20, rollout=Rollout.UNIFORM, seed=0, rejection_cap=3)
        with pytest.raises(RuntimeError):
            agent.update(LISTEN, GROWL_RIGHT)

    def test_a_symmetric_uninformed_growl_prior_never_locates_the_tiger(self, tiger):
        # Rewards are not observations in the BA-POMDP (paper, section 2.2), and an uninformed prior is symmetric under
        # swapping the meaning of the two growls, so no sequence of growls can move the belief off 1/2.
        uninformed = DirichletPrior.from_model(tiger, 1e6, 1e6).with_observation_counts(LISTEN, np.ones((2, 2)))
        agent = _agent(tiger, uninformed, particles=4000)
        for _ in range(8):
            agent.update(LISTEN, GROWL_LEFT)
        left = float((agent.states == 0).mean())
        # Nine independent resamplings of the particle set (the initial draw and eight updates) each add binomial noise.
        assert abs(left - 0.5) < 5 * math.sqrt(9 * 0.25 / agent.num_particles)
        # The paper's 5 / 3 prior moves the same belief to 11 / 12 = 0.917: the Dirichlet-multinomial likelihood of eight
        # left growls is (5 6 7) / (13 14 15) under "tiger left" against (3 4 5 6 7) / (11 12 13 14 15) under "right".
        informed = _agent(tiger, _paper_prior(tiger), particles=4000)
        for _ in range(8):
            informed.update(LISTEN, GROWL_LEFT)
        assert float((informed.states == 0).mean()) > 0.8


class TestPlanning:

    @pytest.mark.parametrize("p_left, optimal", [(0.5, LISTEN), (0.85, LISTEN), (0.15, LISTEN), (0.97, OPEN_RIGHT),
                                                 (0.03, OPEN_LEFT), (0.9999, OPEN_RIGHT)])
    def test_point_mass_prior_chooses_the_optimal_action(self, tiger, p_left, optimal):
        # The beliefs Tiger's optimal policy reaches from b0 (0.5 -> 0.85 -> 0.97 and the mirror), plus near certainty;
        # the optimal actions are the solver's (tests/test_planner_and_agent.py, test_greedy_action_is_optimal).
        point_mass = DirichletPrior.from_model(tiger, 1e9, 1e9)
        for seed in range(3):
            agent = _agent(tiger, point_mass, sims=1000, seed=seed)
            agent.redraw_states(np.array([p_left, 1 - p_left]))
            assert agent.act(100) == optimal

    def test_is_reproducible_from_the_seed(self, tiger):
        def run():
            agent = _agent(tiger, _paper_prior(tiger), particles=200, sims=50, seed=7)
            actions = []
            for o in (GROWL_LEFT, GROWL_LEFT, GROWL_RIGHT):
                a = agent.act(10)
                actions.append(a)
                agent.update(a, o)
            return actions, agent.states.copy(), agent.root_values
        first, again = run(), run()
        assert first[0] == again[0] and np.array_equal(first[1], again[1]) and first[2] == again[2]


def test_simulator_rows_do_not_depend_on_other_rows_actions(tiger):
    # The C-r evaluation plays the 256 paired episodes in chunks of live agents, the other rows taking a fixed action
    # (reports/2026-10-19_eval_protocol/ba_pomcp_baseline.py, section 4). That is exact only if a row's trajectory is a
    # function of its own actions: BatchedPOMDPEnv draws its random numbers independently of the actions.
    def trajectories(mixed: bool):
        env = BatchedPOMDPEnv(tiger, 64, 30, 11, torch.device("cpu"))
        env.reset()
        draws = torch.Generator().manual_seed(5)
        observations, states = [], []
        for _ in range(30):
            action = torch.randint(0, 3, (64,), generator=draws)
            if mixed:
                action[:16] = LISTEN                                # rows outside the chunk act differently
                action[48:] = OPEN_LEFT
            out = env.step(action)
            observations.append(out.observation)
            states.append(env.state.clone())
        return torch.stack(observations, 1), torch.stack(states, 1)
    (full_o, full_s), (mixed_o, mixed_s) = trajectories(False), trajectories(True)
    assert torch.equal(full_o[16:48], mixed_o[16:48]) and torch.equal(full_s[16:48], mixed_s[16:48])
    assert not torch.equal(full_o[:16], mixed_o[:16])               # the other rows did change
