# ABSOLUTE PATH: src/ipomdp/interpretability/error_bounds.py
# ==============================================================================
# FROM BELIEF-DECODING ERROR TO GUARANTEES ON VALUES AND DECISIONS
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. The Lemma (span-Hoelder inequality):
#    - For any vector alpha in R^S and beliefs b, b' in the simplex, sum_s (b - b')_s = 0, hence
#         |alpha . (b - b')| <= span(alpha) / 2 * ||b - b'||_1,   span(alpha) = max_s alpha_s - min_s alpha_s.
#      (Shift alpha by its mid-range c: alpha . (b - b') = (alpha - c) . (b - b') and
#       ||alpha - c||_inf = span(alpha) / 2.)
#    - A maximum of such linear functions (a PWLC value function V(b) = max_k alpha_k . b) is
#      Lipschitz in ||.||_1 with constant L = max_k span(alpha_k) / 2.
#
# 2. The Guarantees (psi = belief probe of belief_probe.py, eps = ||b* - psi(z)||_1):
#    (a) Value error:           |V*(b*) - V*(psi(z))| <= L_V eps
#                                with L_V from the alpha vectors of V*.
#    (b) One-step decision regret of acting greedily on the decoded belief,
#                                a^ = argmax_a Q*(psi(z), a):
#          Q*(b*, a*) - Q*(b*, a^) <= 2 L_Q eps,
#                                with L_Q = max_a Lipschitz(Q*(., a)) from the per-action vector
#                                sets (domain/solver.py action_value_functions). Proof:
#          Q*(b*, a*) - Q*(b*, a^) <= [Q*(psi, a*) + L_Q eps] - [Q*(psi, a^) - L_Q eps] <= 2 L_Q eps.
#    (c) Discounted loss of the policy "act greedily on psi(z_t)" forever, if eps_t <= eps_max:
#          V*(b0) - V^psi(b0) <= 2 L_Q eps_max / (1 - gamma)
#        (performance-difference lemma: the loss is the discounted sum of per-step regrets).
#    - The solver's certified error delta = ||V* - V_n||_inf adds 2 delta (a, b) and
#      2 delta / (1 - gamma) (c) to these bounds when V_n is used in place of V*.
#    - Expected-error form: (a) and (b) hold pointwise, so taking expectations gives
#         E[|V*(b*) - V*(psi)|] <= L_V E[eps],   E[regret] <= 2 L_Q E[eps],
#      and (c) with the per-step expectation under the policy's state distribution,
#         V*(b0) - V^psi(b0) <= 2 L_Q sum_t gamma^t E[eps_t] <= 2 L_Q eps-bar / (1 - gamma).
#      Phase-6 measurement (canonical Tiger, trained agent): the worst-case eps is set by a few
#      rare latents (max L1 0.3-0.9) and makes the worst-case bounds 50-100x loose, while the
#      expected-error bounds are within a small factor of the measurements. Both are reported;
#      eps-bar in (c) is measured on the evaluation data, not on the psi-policy's own
#      distribution, so that form is an estimate rather than a certificate.
#
# 3. Bounds Next to Measurements:
#    - Every bound is reported together with its empirical counterpart on the same held-out
#      beliefs (value errors and one-step regrets computed exactly with V*, Q*), and the policy
#      bound together with the measured return of the decoded-belief policy in the simulator
#      (analysis.py). A bound that is far above its measurement is honest but loose; a
#      measurement above its bound would indicate a bug.
#
# 4. With a Lower Bound in Place of V* and Q* (the folded interactive problems):
#    - A folded interactive problem has no exact solution (experiments/interactive_analysis.py). The vector sets
#      passed in are then a certified lower bound, L(b) = max_k alpha_k . b <= V*(b), and per action the vectors of
#      that action, Q_L(b, a) <= Q*(b, a). What the three guarantees say in that case:
#        (a) is a statement about the function L: |L(b*) - L(psi(z))| <= L_V eps. Nothing about V*.
#        (b) is a regret in Q_L: Q_L(b*, a_L) - Q_L(b*, a^) <= 2 L_Q eps, with a_L the action of the best vector.
#            The proof of section 2 uses only that each Q_L(., a) is Lipschitz.
#        (c) bounds the loss against the LOWER BOUND at b0, not against the optimum and not against the value of
#            the agent that plays a_L (which is at least L(b0)):
#               L(b0) - V^psi(b0) <= 2 L_Q sum_t gamma^t E[eps_t].
#    - (c) needs one property of the set: every vector is the value of its own pointer plan within the set,
#      alpha_k = R(., a_k) + gamma sum_o M_{a_k,o} alpha_{succ(k,o)}, as the vectors of a point-based backup are
#      (domain/bounds.py). Then, for every belief and action,
#           Q_L(b, a) <= R(b, a) + gamma sum_o P(o | b, a) L(tau(b, a, o)).                                    (*)
#      Proof of (c) from (*): along the psi-policy, L(b_t) = Q_L(b_t, a^_t) + regret_t
#      <= r(b_t, a^_t) + gamma E[L(b_{t+1})] + regret_t; unrolling, with L bounded, gives
#      L(b0) <= V^psi(b0) + sum_t gamma^t E[regret_t], and regret_t <= 2 L_Q eps_t by (b).
#    - With regret 0 the same argument shows that the agent playing a_L has value at least L(b0).
#    - (*) is not assumed for the two benchmark files of the interactive experiments: every one of their vectors
#      equals its pointer backup to 3.4e-13 (reports/2026-11-16_code_audit/benchmark_improvability.py and .log).
#    - Consequently V*(b0) - V^psi(b0) <= (U - L)(b0) + the bound of (c), with U the certified upper bound.
#
# 5. The Local Form: the Spans of the Active Vectors (2026-11-23):
#    - The constants of section 2 are the largest half-span of the whole set. One wide vector sets them even where
#      it is never the maximiser: on the folded interactive problems L_V = L_Q is about 57, from the few vectors that
#      open a door (span about 110), while the vector active at a visited belief has a half-span of 17 to 19 on
#      average. The lemma of section 1 gives more than that when it is applied to the right vector.
#    - Lemma (local form). Let V(b) = max_k alpha_k . b, and let alpha_b be a maximiser at b. For beliefs b, b':
#           V(b) - V(b') <= alpha_b . (b - b') <= span(alpha_b) / 2 * ||b - b'||_1
#      (V(b') >= alpha_b . b', then section 1), and the same with b and b' exchanged. Hence
#           |V(b) - V(b')| <= max(span(alpha_b), span(alpha_b')) / 2 * ||b - b'||_1.
#      Any maximiser may be taken at a tie. The global bound of section 2 is the case where the largest span of the
#      set replaces the two.
#    - (a) local: |V*(b*) - V*(psi)| <= max(span(alpha_{b*}), span(alpha_psi)) / 2 * eps.
#    - (b) local: with a* = argmax_a Q*(b*, a), a^ = argmax_a Q*(psi, a), alpha the vector active in Q*(., a*) at b*
#      and beta the vector active in Q*(., a^) at psi,
#           Q*(b*, a*) - Q*(b*, a^) = [Q*(b*, a*) - Q*(psi, a*)] + [Q*(psi, a*) - Q*(psi, a^)] + [Q*(psi, a^) - Q*(b*, a^)]
#                                   <= (span(alpha) + span(beta)) / 2 * eps,
#      because the middle bracket is at most 0 and each outer one is a one-sided difference of the lemma.
#    - (c) local: the proofs of (c) in sections 2 and 4 sum per-step regrets, so the per-step local bound may be used:
#           loss <= sum_t gamma^t E[(span(alpha_t) + span(beta_t)) / 2 * eps_t].
#      As in section 2, the reported number divides the mean of that product on the evaluation data by 1 - gamma: an
#      estimate, since the data are not the psi-policy's own distribution.
#    - What it is and is not. The coefficient depends on the pair (b*, psi) through the vectors active at the two
#      beliefs, so this is a bound per pair and, averaged, a bound on the expected error; it is not a single constant
#      known before the beliefs are. It needs nothing beyond the vector sets, holds for a lower-bound set exactly as
#      for V* (section 4), and is never above the global form, pair by pair. Both forms are reported.
# ==============================================================================

from dataclasses import dataclass

import torch
from torch import Tensor

from ..domain import AlphaVectorSet


def lipschitz_constant(value_function: AlphaVectorSet) -> float:
    """L = max_k span(alpha_k) / 2: the ||.||_1-Lipschitz constant of V(b) = max_k alpha_k . b (section 1)."""
    vectors = value_function.vectors
    return float((vectors.max(dim=1).values - vectors.min(dim=1).values).max()) / 2.0


def q_values(action_values: list[AlphaVectorSet], beliefs: Tensor) -> Tensor:
    """Q(b, a) for beliefs (N, S) from per-action vector sets; shape (N, |A|), float64."""
    return torch.stack([q.value(beliefs) for q in action_values], dim=-1)


def half_spans(value_function: AlphaVectorSet) -> Tensor:
    """span(alpha_k) / 2 of every vector, float64 (K,): the ||.||_1-Lipschitz constant of b -> alpha_k . b."""
    vectors = value_function.vectors
    return (vectors.max(dim=1).values - vectors.min(dim=1).values) / 2.0


def local_bounds(optimal_value: AlphaVectorSet, optimal_action_values: list[AlphaVectorSet], true_beliefs: Tensor,
                 decoded_beliefs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    The local form of guarantees (a) and (b), pair by pair (section 5).

    Args:
        optimal_value, optimal_action_values: as error_bounds.
        true_beliefs, decoded_beliefs: b* and psi(z), float64 (N, S) on the CPU.

    Returns:
        value_bound: max(span(alpha_{b*}), span(alpha_psi)) / 2 * eps, shape (N,).
        regret_bound: (span(alpha) + span(beta)) / 2 * eps with alpha active in Q(., a*) at b* and beta active in
            Q(., a^) at psi, shape (N,).
        active: span(alpha_{b*}) / 2, the half-span of the vector of the value set active at the exact belief, (N,).
    """
    eps = (true_beliefs - decoded_beliefs).abs().sum(-1)
    spans = half_spans(optimal_value)
    active = spans[optimal_value.maximiser(true_beliefs)]
    value_bound = torch.maximum(active, spans[optimal_value.maximiser(decoded_beliefs)]) * eps
    best = q_values(optimal_action_values, true_beliefs).argmax(-1)
    chosen = q_values(optimal_action_values, decoded_beliefs).argmax(-1)
    at_true = torch.stack([half_spans(q)[q.maximiser(true_beliefs)] for q in optimal_action_values], dim=-1)
    at_decoded = torch.stack([half_spans(q)[q.maximiser(decoded_beliefs)] for q in optimal_action_values], dim=-1)
    regret_bound = (at_true.gather(1, best.unsqueeze(1)).squeeze(1)
                    + at_decoded.gather(1, chosen.unsqueeze(1)).squeeze(1)) * eps
    return value_bound, regret_bound, active


@dataclass(frozen=True)
class ErrorBoundReport:
    """
    Guarantees and measurements for one probe on one set of held-out beliefs (module header).

    Attributes:
        lipschitz_value, lipschitz_q: L_V and L_Q.
        belief_l1_mean, belief_l1_max: Measured eps.
        value_error_bound_max, value_error_max, value_error_mean: bound (a) at eps_max vs measured.
        value_error_bound_mean: Expected-error form of (a), L_V E[eps].
        regret_bound_max, regret_max, regret_mean: bound (b) at eps_max vs measured one-step regret.
        regret_bound_mean: Expected-error form of (b), 2 L_Q E[eps].
        suboptimal_decision_rate: Fraction of beliefs where the decoded-belief action is not optimal.
        policy_loss_bound: Bound (c) with eps_max.
        policy_loss_bound_mean: Expected-error form of (c) with the measured E[eps] (an estimate).
        active_half_span_mean: Mean over the exact beliefs of span(alpha_{b*}) / 2 (section 5), to set beside L_V.
        value_error_bound_local_mean, value_error_bound_local_max: The local form of (a), mean and largest over the
            pairs. The mean bounds value_error_mean and the largest bounds value_error_max.
        regret_bound_local_mean, regret_bound_local_max: The local form of (b), likewise for the regret.
        policy_loss_bound_local_mean: The local form of (c), regret_bound_local_mean / (1 - gamma) (an estimate).
    """

    lipschitz_value: float
    lipschitz_q: float
    belief_l1_mean: float
    belief_l1_max: float
    value_error_bound_max: float
    value_error_max: float
    value_error_mean: float
    value_error_bound_mean: float
    regret_bound_max: float
    regret_max: float
    regret_mean: float
    regret_bound_mean: float
    suboptimal_decision_rate: float
    policy_loss_bound: float
    policy_loss_bound_mean: float
    active_half_span_mean: float
    value_error_bound_local_mean: float
    value_error_bound_local_max: float
    regret_bound_local_mean: float
    regret_bound_local_max: float
    policy_loss_bound_local_mean: float


def error_bounds(optimal_value: AlphaVectorSet, optimal_action_values: list[AlphaVectorSet], discount: float,
                 true_beliefs: Tensor, decoded_beliefs: Tensor) -> ErrorBoundReport:
    """
    Evaluates guarantees (a)-(c) and their empirical counterparts.

    Args:
        optimal_value: Alpha vectors of V* (or a certified approximation).
        optimal_action_values: Per-action alpha vectors of Q*.
        discount: gamma.
        true_beliefs: Exact posteriors b*, shape (N, S), float64.
        decoded_beliefs: Probe outputs psi(z), shape (N, S), float64.
    """
    true_beliefs, decoded_beliefs = true_beliefs.double().cpu(), decoded_beliefs.double().cpu()
    l_v, l_q = lipschitz_constant(optimal_value), max(lipschitz_constant(q) for q in optimal_action_values)
    eps = (true_beliefs - decoded_beliefs).abs().sum(-1)
    value_error = (optimal_value.value(true_beliefs) - optimal_value.value(decoded_beliefs)).abs()
    q_true = q_values(optimal_action_values, true_beliefs)
    chosen = q_values(optimal_action_values, decoded_beliefs).argmax(-1)
    regret = q_true.max(-1).values - q_true.gather(1, chosen.unsqueeze(1)).squeeze(1)
    eps_max, eps_mean = float(eps.max()), float(eps.mean())
    local_value, local_regret, active = local_bounds(optimal_value, optimal_action_values, true_beliefs, decoded_beliefs)
    return ErrorBoundReport(
        lipschitz_value=l_v, lipschitz_q=l_q,
        belief_l1_mean=eps_mean, belief_l1_max=eps_max,
        value_error_bound_max=l_v * eps_max, value_error_max=float(value_error.max()),
        value_error_mean=float(value_error.mean()), value_error_bound_mean=l_v * eps_mean,
        regret_bound_max=2 * l_q * eps_max, regret_max=float(regret.max()), regret_mean=float(regret.mean()),
        regret_bound_mean=2 * l_q * eps_mean,
        suboptimal_decision_rate=float((regret > 1e-9).double().mean()),
        policy_loss_bound=2 * l_q * eps_max / (1.0 - discount),
        policy_loss_bound_mean=2 * l_q * eps_mean / (1.0 - discount),
        active_half_span_mean=float(active.mean()),
        value_error_bound_local_mean=float(local_value.mean()), value_error_bound_local_max=float(local_value.max()),
        regret_bound_local_mean=float(local_regret.mean()), regret_bound_local_max=float(local_regret.max()),
        policy_loss_bound_local_mean=float(local_regret.mean()) / (1.0 - discount),
    )
