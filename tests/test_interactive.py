# ABSOLUTE PATH: tests/test_interactive.py
"""Contribution (2), the I-POMDP core: the two-agent tiger tables, policy graphs, and the fold into a POMDP."""
# ==============================================================================
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS
# ==============================================================================
# 1. What Must Hold Before Any Learned Component Touches the Interactive Problem:
#    - The joint tables are Table 1 of Gmytrasiewicz & Doshi (2005), entry by entry.
#    - The fold (domain/interactive.py) is the exact decision problem of an agent facing the other agent executing
#      its controller: degenerate controllers reproduce problems whose values are known in closed form or from the
#      canonical Tiger; the folded simulator matches a direct two-agent simulation; the folded Bayes filter matches
#      brute-force enumeration over the other agent's hidden nodes, actions and observations.
#    - Policy graphs (domain/policy_graph.py) reproduce the plans they were extracted from and evaluate exactly.
# 2. Speed:
#    - Everything here is CPU and seconds: finite horizons on small folds. The certified infinite-horizon checks
#      and the published level-1 values live in the slow suite.
# ==============================================================================

import itertools

import pytest
import torch

from ipomdp.domain import (BatchedPOMDPEnv, FinitePOMDP, TigerAction, TigerState, action_value_functions, belief_update,
                           build_tiger_pomdp, solve_finite_horizon)
from ipomdp.domain.bounds import _HSVI, hsvi, hsvi_budget, revealed_upper_bound
from ipomdp.domain.interactive import (TIGER_LEVEL_ZERO_NOISE, fold, nested_problem, noise_model,
                                       relaxation_reveal)
from ipomdp.domain.multiagent_tiger import (CREAK_ACCURACY, CREAK_ERROR, OBSERVATIONS, Agent, Creak, Growl,
                                            build_multiagent_tiger)
from ipomdp.domain.policy_graph import (constant_policy, envelope, evaluate, node_law_on_segment, plan_graph,
                                        segment_breakpoints, stationary_graph)
from ipomdp.domain.solver import AlphaVectorSet
from ipomdp.domain.reduction import expand_observations, lump_observations

L, OL, OR = int(TigerAction.LISTEN), int(TigerAction.OPEN_LEFT), int(TigerAction.OPEN_RIGHT)
TL, TR = int(TigerState.TIGER_LEFT), int(TigerState.TIGER_RIGHT)
F64 = torch.float64


def onehot(i: int, n: int) -> torch.Tensor:
    v = torch.zeros(n, dtype=F64)
    v[i] = 1.0
    return v


def hsvi_corners(model: FinitePOMDP) -> torch.Tensor:
    """The fast informed bound's corner values (bounds.py, section 3)."""
    return torch.from_numpy(_HSVI(model).corner)


def value_at_b0(model: FinitePOMDP, horizon: int) -> list[float]:
    return [float(f.value(model.initial_belief[None])) for f in solve_finite_horizon(model, horizon)]


class TestTableOne:
    """Gmytrasiewicz & Doshi (2005), Table 1, entry by entry."""

    def test_transition_resets_unless_both_listen(self):
        phys = build_multiagent_tiger(0.95)
        for a_i, a_j in itertools.product(range(3), repeat=2):
            expected = torch.eye(2, dtype=F64) if (a_i, a_j) == (L, L) else torch.full((2, 2), 0.5, dtype=F64)
            assert torch.equal(phys.transition[a_i, a_j], expected)

    def test_rewards_depend_on_the_own_action_and_pre_transition_state_only(self):
        phys = build_multiagent_tiger(0.95)
        own = {L: (-1.0, -1.0), OL: (-100.0, 10.0), OR: (10.0, -100.0)}
        for a_i, a_j in itertools.product(range(3), repeat=2):
            assert phys.reward[Agent.I, a_i, a_j].tolist() == list(own[a_i])
            assert phys.reward[Agent.J, a_i, a_j].tolist() == list(own[a_j])

    def test_observations_of_a_listener_and_an_opener(self):
        phys = build_multiagent_tiger(0.95)
        names = {(g, c): k for k, (g, c) in enumerate(OBSERVATIONS)}
        # Table 1 row <L, OL>, TL for agent i: growl left 0.85, creak left 0.9.
        assert phys.observation[Agent.I, L, OL, TL, names[(Growl.GROWL_LEFT, Creak.CREAK_LEFT)]] == 0.85 * 0.9
        assert phys.observation[Agent.I, L, OL, TL, names[(Growl.GROWL_RIGHT, Creak.SILENCE)]] == 0.15 * 0.05
        # Row <OR, L> for agent j (j listened, i opened right), tiger right: growl right 0.85, creak right 0.9.
        assert phys.observation[Agent.J, OR, L, TR, names[(Growl.GROWL_RIGHT, Creak.CREAK_RIGHT)]] == 0.85 * 0.9
        # An opener hears all six pairs with probability 1/6, whatever the other did.
        for other in range(3):
            assert torch.equal(phys.observation[Agent.I, OL, other], torch.full((2, 6), 1 / 6, dtype=F64))
            assert torch.equal(phys.observation[Agent.J, other, OR], torch.full((2, 6), 1 / 6, dtype=F64))
        assert (CREAK_ACCURACY, CREAK_ERROR) == (0.9, 0.05)

    def test_views_put_the_own_action_first(self):
        phys = build_multiagent_tiger(0.95)
        vj = phys.view(Agent.J)
        for a_i, a_j in itertools.product(range(3), repeat=2):
            assert torch.equal(vj.transition[a_j, a_i], phys.transition[a_i, a_j])
            assert torch.equal(vj.own_observation[a_j, a_i], phys.observation[Agent.J, a_i, a_j])
            assert torch.equal(vj.other_observation[a_j, a_i], phys.observation[Agent.I, a_i, a_j])
            assert torch.equal(vj.own_reward[a_j, a_i], phys.reward[Agent.J, a_i, a_j])


class TestDegenerateFolds:
    """Controllers whose folded problem has a known solution (interactive.py, section 1)."""

    def test_a_listening_other_agent_leaves_the_canonical_tiger(self):
        # j always listens: no resets by j, and i's creak is independent of the state, so i's problem is the
        # canonical Tiger with an uninformative extra channel; V_h(b0) must equal the canonical values.
        folded = fold(build_multiagent_tiger(0.95).view(Agent.I), constant_policy(onehot(L, 3), 6, "listen"), L)
        canonical = build_tiger_pomdp()
        assert folded.pomdp.num_states == 2
        assert value_at_b0(folded.pomdp, 6) == pytest.approx(value_at_b0(canonical, 6), abs=1e-9)
        # With the creak channel merged, the folded tables are the canonical tables (to round-off: 0.85 is split
        # into 0.85 * 0.9 + 0.85 * 0.05 + 0.85 * 0.05 and summed back).
        merged = lump_observations(folded.pomdp).model
        assert torch.equal(merged.reward, canonical.reward)
        assert torch.allclose(merged.transition, canonical.transition, atol=1e-15, rtol=0.0)
        assert torch.allclose(merged.observation, canonical.observation, atol=1e-15, rtol=0.0)

    def test_an_opening_other_agent_makes_listening_forever_optimal(self):
        # j opens the left door every step: the tiger is re-placed every step, so one growl (about the new
        # position) gives belief 0.85, where opening is worth 0.85 * 10 - 0.15 * 100 = -6.5 < -1. V_h = -sum gamma^t.
        gamma = 0.95
        folded = fold(build_multiagent_tiger(gamma).view(Agent.I), constant_policy(onehot(OL, 3), 6, "open"), L)
        expected = [-(1 - gamma ** h) / (1 - gamma) for h in range(1, 13)]
        assert value_at_b0(folded.pomdp, 12) == pytest.approx(expected, abs=1e-9)

    def test_uniformly_random_other_agent_two_step_values(self):
        # Q_2(b0): listen -1 + gamma * (-1) (a growl reports the post-reset state: belief 0.85, listening optimal),
        # either door -45 + gamma * (-1).
        gamma = 0.95
        folded = fold(build_multiagent_tiger(gamma).view(Agent.I),
                      constant_policy(torch.full((3,), 1 / 3, dtype=F64), 6, "uniform"), L)
        q = action_value_functions(folded.pomdp, solve_finite_horizon(folded.pomdp, 1)[0], 1e-9)
        values = [float(x.value(folded.pomdp.initial_belief[None])) for x in q]
        assert values == pytest.approx([-1 - gamma, -45 - gamma, -45 - gamma], abs=1e-12)


def _level_zero_noise(gamma: float) -> FinitePOMDP:
    """j's level-0 problem as Gmytrasiewicz & Doshi define it: i's actions folded into T, O and R as fixed noise."""
    return noise_model(build_multiagent_tiger(gamma).view(Agent.J), TIGER_LEVEL_ZERO_NOISE)


class TestPolicyGraphs:

    def test_plan_graph_reproduces_every_plan_value(self):
        model = _level_zero_noise(0.9)
        value_functions = [envelope(g) for g in solve_finite_horizon(model, 3)]
        graph = plan_graph(model, value_functions)
        values = evaluate(model, graph)
        top = len(value_functions[-1].vectors)
        # The plan of node h3.k, followed for three steps and then the sink, is worth exactly Gamma_3[k] plus the
        # discounted sink tail, which is LISTEN forever from step 3 on: -gamma^3 / (1 - gamma) on every state.
        tail = -(0.9 ** 3) / (1 - 0.9)
        assert torch.allclose(values[:top], value_functions[-1].vectors + tail, atol=1e-9)

    def test_controller_value_of_a_constant_policy(self):
        model = build_tiger_pomdp()
        values = evaluate(model, constant_policy(onehot(L, 3), 2, "listen"))
        assert torch.allclose(values, torch.full((1, 2), -1 / (1 - 0.95), dtype=F64), atol=1e-12)

    def test_segment_law_of_a_uniform_prior(self):
        # Horizon-1 level-0 noise agent with a uniform prior over p_j(TL): it opens right above 0.9, left below 0.1
        # (110 p - 100 = -1), so its first action is OL 0.1, L 0.8, OR 0.1, the noise the paper folds into level 0.
        model = _level_zero_noise(0.9)
        gamma_1 = solve_finite_horizon(model, 1)[0]
        law = node_law_on_segment(gamma_1.vectors, onehot(TR, 2), onehot(TL, 2), 0, len(gamma_1.vectors))
        by_action = {int(a): float(w) for a, w in zip(gamma_1.actions, law)}
        assert by_action == pytest.approx({L: 0.8, OL: 0.1, OR: 0.1}, abs=1e-12)

    def test_level_zero_policy_matches_the_published_noise_pomdp_thresholds(self):
        # Gmytrasiewicz & Doshi (2005), Fig. 4: horizon-3 policy graph of the POMDP with noise (undiscounted):
        # region boundaries 0.045, 0.135, 0.175, 0.825, 0.865, 0.955 over p(TL). Our level-0 construction, with its
        # uninformative creaks merged, is that POMDP; the paper prints three decimals, four of ours agree within
        # 0.001 and two (0.138, 0.862) within 0.003.
        model = lump_observations(_level_zero_noise(0.5)).model
        assert model.observation_names == ("GROWL_LEFT|CREAK_LEFT+GROWL_LEFT|CREAK_RIGHT+GROWL_LEFT|SILENCE",
                                           "GROWL_RIGHT|CREAK_LEFT+GROWL_RIGHT|CREAK_RIGHT+GROWL_RIGHT|SILENCE")
        object.__setattr__(model, "discount", 1.0)        # the paper's finite-horizon examples are undiscounted
        top = solve_finite_horizon(model, 3)[-1]
        cuts = [p for p, _ in segment_breakpoints(top.vectors, onehot(TR, 2), onehot(TL, 2))[:-1]]
        assert cuts == pytest.approx([0.045, 0.135, 0.175, 0.825, 0.865, 0.955], abs=0.003)

    def test_lumping_merges_the_uninformative_creaks_without_changing_values(self):
        model = _level_zero_noise(0.95)
        lumped = lump_observations(model)
        assert lumped.group.tolist() == [0, 0, 0, 1, 1, 1]
        beliefs = torch.rand(64, 2, dtype=F64, generator=torch.Generator().manual_seed(0))
        beliefs = beliefs / beliefs.sum(-1, keepdim=True)
        for full, merged in zip(solve_finite_horizon(model, 4), solve_finite_horizon(lumped.model, 4)):
            assert torch.allclose(full.value(beliefs), merged.value(beliefs), atol=1e-12)
        # The canonical Tiger has nothing to merge.
        assert lump_observations(build_tiger_pomdp()).group.tolist() == [0, 1]
        graph = expand_observations(plan_graph(lumped.model, [envelope(g) for g in solve_finite_horizon(lumped.model, 2)]),
                                    lumped.group)
        assert graph.num_observations == 6


class TestFold:

    @staticmethod
    def _level_one(gamma: float = 0.9, horizon: int = 2):
        model = _level_zero_noise(gamma)
        value_functions = [envelope(g) for g in solve_finite_horizon(model, horizon)]
        graph = plan_graph(model, value_functions)
        graph = graph.with_initial_law(node_law_on_segment(value_functions[-1].vectors, onehot(TR, 2), onehot(TL, 2),
                                                           0, graph.num_nodes))
        phys = build_multiagent_tiger(gamma)
        return phys, graph, fold(phys.view(Agent.I), graph, L)

    def test_folded_problem_is_a_valid_pomdp_over_reachable_states(self):
        _, graph, folded = self._level_one()
        # Every kept state is reachable and the indices map back onto the grid.
        assert folded.pomdp.num_states < 2 * graph.num_nodes * 3
        assert folded.physical_marginal(folded.pomdp.initial_belief[None])[0].tolist() == [0.5, 0.5]
        assert torch.allclose(folded.other_node_marginal(folded.pomdp.initial_belief[None])[0], graph.initial_law)

    def test_folded_bayes_filter_matches_brute_force_enumeration(self):
        phys, graph, folded = self._level_one()
        histories = [((L, 2), (L, 0)), ((L, 5), (OL, 1)), ((L, 2), (L, 2))]       # (i's action, i's observation)
        for history in histories:
            belief = folded.pomdp.initial_belief[None]
            for action, obs in history:
                belief = belief_update(folded.pomdp, belief, torch.tensor([action]), torch.tensor([obs]))
            # Brute force: sum over s_0, j's first node and every (j action, j observation, s') path.
            joint: dict[tuple[int, int, int], float] = {}
            paths = [(s, n, L, 0.5 * float(graph.initial_law[n])) for s in range(2) for n in range(graph.num_nodes)
                     if graph.initial_law[n] > 0]
            for action, obs in history:
                nxt = []
                for s, n, _, w in paths:
                    for a_j in range(3):
                        pa = float(graph.action_law[n, a_j])
                        if pa == 0:
                            continue
                        for s2 in range(2):
                            for o_j in range(6):
                                p = (pa * float(phys.transition[action, a_j, s, s2])
                                     * float(phys.observation[Agent.I, action, a_j, s2, obs])
                                     * float(phys.observation[Agent.J, action, a_j, s2, o_j]))
                                for n2 in range(graph.num_nodes):
                                    q = float(graph.successor_law[n, a_j, o_j, n2])
                                    if p * q > 0:
                                        nxt.append((s2, n2, a_j, w * p * q))
                paths = nxt
            for s, n, a_j, w in paths:
                joint[(s, n, a_j)] = joint.get((s, n, a_j), 0.0) + w
            total = sum(joint.values())
            index = {(s, n, a): k for k, (s, n, a) in enumerate(zip(folded.physical_state.tolist(),
                                                                    folded.other_node.tolist(),
                                                                    folded.other_previous_action.tolist()))}
            expected = torch.zeros(folded.pomdp.num_states, dtype=F64)
            for key, w in joint.items():
                expected[index[key]] = w / total
            assert torch.allclose(belief[0], expected, atol=1e-12)

    def test_folded_simulator_matches_a_direct_two_agent_simulation(self):
        # Frequencies of i's observation and reward after one listening step, folded simulator against a direct
        # simulation of the physics with j executing its controller, within 5 standard errors.
        phys, graph, folded = self._level_one()
        n = 40_000
        env = BatchedPOMDPEnv(folded.pomdp, n, 4, seed=11, device=torch.device("cpu"))
        env.reset()
        step = env.step(torch.full((n,), L))
        folded_obs, folded_reward = step.observation, step.reward
        gen = torch.Generator().manual_seed(12)
        s = torch.multinomial(phys.initial_state, n, replacement=True, generator=gen)
        node = torch.multinomial(graph.initial_law, n, replacement=True, generator=gen)
        a_j = torch.multinomial(graph.action_law[node], 1, generator=gen).squeeze(1)
        s2 = torch.multinomial(phys.transition[L, a_j, s], 1, generator=gen).squeeze(1)
        direct_obs = torch.multinomial(phys.observation[Agent.I, L, a_j, s2], 1, generator=gen).squeeze(1)
        direct_reward = phys.reward[Agent.I, L, a_j, s]
        for o in range(6):
            p1, p2 = (folded_obs == o).double().mean(), (direct_obs == o).double().mean()
            se = ((p1 * (1 - p1) + p2 * (1 - p2)) / n).sqrt()
            assert abs(float(p1 - p2)) < 5 * float(se) + 1e-12
        assert float(folded_reward.double().mean()) == pytest.approx(float(direct_reward.mean()), abs=1e-6)   # float32 rewards


class TestBracket:
    """domain/bounds.py: HSVI's interval contains the certified canonical optimum; its lower end is attained."""

    def test_bracket_contains_the_certified_tiger_optimum_and_its_controller_attains_the_lower_bound(self):
        model = build_tiger_pomdp()
        bracket = hsvi(model, 0.5)
        assert bracket.upper - bracket.lower <= 0.5
        assert bracket.lower <= 19.3713 <= bracket.upper            # V*(b0), certified to 1e-4 (test_domain.py)
        values = evaluate(model, bracket.controller)
        attained = float(bracket.controller.initial_law @ (values @ model.initial_belief))
        assert attained >= bracket.lower - 1e-9
        # Every node's exact value dominates its lower-bound vector (bounds.py, section 2).
        assert (values >= bracket.lower_vectors - 1e-9).all()

    def test_compact_stationary_controller_of_the_tiger_is_certified_by_the_upper_bound(self):
        model = build_tiger_pomdp()
        bracket = hsvi(model, 0.5)
        compact = stationary_graph(model, AlphaVectorSet(bracket.lower_vectors, bracket.controller.action_law.argmax(-1)), 0.05)
        value = float(compact.initial_law @ (evaluate(model, compact) @ model.initial_belief))
        assert compact.num_nodes < bracket.controller.num_nodes
        assert value <= 19.3713 + 1e-4 and bracket.upper - value < 1.0


class TestHierarchy:

    def test_level_one_problem_folds_the_certified_level_zero_controller(self):
        problem = nested_problem(build_multiagent_tiger(0.95), Agent.I, 1, TIGER_LEVEL_ZERO_NOISE, 2.0, 2.0, L)
        (zero,) = problem.below
        assert zero.agent == Agent.J and zero.level == 0
        assert zero.problem.num_observations == 2 and zero.controller.num_observations == 6    # merged, then expanded
        assert problem.interactive.other is zero.controller and problem.pomdp.num_observations == 6
        assert zero.bracket.lower <= zero.bracket.upper <= zero.bracket.lower + 2.0
        assert zero.controller_value <= zero.bracket.upper + 1e-9


class TestRelaxedBound:
    """bounds.revealed_upper_bound and interactive.relaxation_reveal (bounds.py and interactive.py, section 5)."""

    def test_an_empty_revelation_gives_the_certified_tiger_optimum(self):
        # A constant symbol carries no information, so the relaxed problem is the canonical Tiger, whose beliefs
        # (growl-count differences, merged once a coordinate rounds to zero) form a finite graph.
        bound = revealed_upper_bound(build_tiger_pomdp(), lambda action, history: 0, 1)
        assert 19.3713 - 1e-4 <= bound.upper <= 19.3713 + 1e-4 + bound.merge_slack   # certified V*(b0) to 1e-4
        assert bound.merge_slack < 1e-5

    def test_revealing_the_state_gives_the_fully_observable_value(self):
        # Knowing the arrival state, the agent listens once and then opens the treasure door every step:
        # V(b0) = -1 + gamma * 10 / (1 - gamma) = 189.
        bound = revealed_upper_bound(build_tiger_pomdp(), lambda action, history: history[-1], 1)
        assert bound.upper == pytest.approx(189.0, abs=1e-6)

    def test_a_listening_other_agent_reveals_nothing(self):
        # j always listens: its previous action is always LISTEN and its node never changes, so the relaxation of
        # section 5 adds no information to the canonical Tiger with an uninformative creak.
        folded = fold(build_multiagent_tiger(0.95).view(Agent.I), constant_policy(onehot(L, 3), 6, "listen"), L)
        for delay in (0, 1, 2):
            reveal, memory = relaxation_reveal(folded, delay, (OL, OR))
            bound = revealed_upper_bound(folded.pomdp, reveal, memory)
            assert 19.3713 - 1e-4 <= bound.upper <= 19.3713 + 1e-4 + bound.merge_slack

    def test_seeded_budget_bracket_starts_from_the_relaxed_values(self):
        # An empty revelation's values are V* itself, so seeded with them the bracket's upper end is V*(b0) after a
        # single trial, the corners come from the relaxed corners, and the lower end stays below it.
        model = build_tiger_pomdp()
        relaxed = revealed_upper_bound(model, lambda action, history: 0, 1, extra_roots=torch.eye(2, dtype=F64))
        bracket = hsvi_budget(model, 0.5, 1, corners=relaxed.root_values[1:], points=(relaxed.beliefs, relaxed.values))
        assert bracket.trials == 1
        assert bracket.lower <= 19.3713 <= bracket.upper <= relaxed.upper + 1e-9
        # Unseeded, it reaches the requested width on the canonical Tiger and the bracket still contains V*(b0).
        unseeded = hsvi_budget(model, 0.5, 100_000)
        assert unseeded.upper - unseeded.lower <= 0.5 and unseeded.lower <= 19.3713 <= unseeded.upper

    @pytest.mark.slow
    def test_level_one_relaxation_tightens_with_delay_and_dominates_a_feasible_controller(self):
        # E1's level-1 problem (conf/env/itiger.yaml: j's 11-node controller extracted at 0.5, certified at 0.05;
        # 26 states; about 1.5 min to build and 2.5 min for the two bounds).
        problem = nested_problem(build_multiagent_tiger(0.95), Agent.I, 1, TIGER_LEVEL_ZERO_NOISE, 0.05, 0.5, L)
        (zero,) = problem.below
        assert problem.pomdp.num_states == 26 and zero.bracket.upper - zero.controller_value <= 0.051
        bounds = []
        for delay in (0, 1):
            reveal, memory = relaxation_reveal(problem.interactive, delay, (OL, OR))
            bounds.append(revealed_upper_bound(problem.pomdp, reveal, memory))
        # Delay 0 reveals a superset of what delay 1 reveals, so its bound cannot be lower.
        assert bounds[1].upper <= bounds[0].upper + bounds[0].merge_slack + bounds[1].merge_slack
        # Below the fast informed bound, above the exact value of a feasible controller: the canonical Tiger's
        # compact controller, deaf to creaks (growl-major columns, multiagent_tiger.OBSERVATIONS).
        fib = float(problem.pomdp.initial_belief @ hsvi_corners(problem.pomdp))
        canonical = build_tiger_pomdp()
        bracket = hsvi(canonical, 0.5)
        compact = stationary_graph(canonical, AlphaVectorSet(bracket.lower_vectors,
                                                             bracket.controller.action_law.argmax(-1)), 0.05)
        deaf = expand_observations(compact, torch.tensor([0, 0, 0, 1, 1, 1]))
        feasible = float(deaf.initial_law @ (evaluate(problem.pomdp, deaf) @ problem.pomdp.initial_belief))
        assert feasible <= bounds[1].upper <= bounds[0].upper < fib
        # The returned points are upper bounds at their marginals: b0's marginal is b0 itself.
        assert torch.equal(bounds[1].beliefs[0], problem.pomdp.initial_belief)
        assert float(bounds[1].values[0]) == bounds[1].upper
