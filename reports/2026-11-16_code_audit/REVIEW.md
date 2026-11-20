<!-- ABSOLUTE PATH: reports/2026-11-16_code_audit/REVIEW.md -->

# Code audit, 2026-11-16: per-file review record

Bottom-up review of every tracked source, test, configuration and script file, one phase per layer. Each row records what was found, what changed, and the check that proved the change (fingerprints: `tools/fingerprint/`; load gate: `tools/fingerprint/load_digests.py`; golden files: `tests/golden/fingerprints/`). Reference commit for every identity check: `5bea8c3`.

## Gates

- **Numbers:** `reports/2026-11-14_submission_audit/ledger.py` exits 0; `sha256sum -c reports/2026-11-14_submission_audit/MANIFEST.sha256` passes.
- **Bits:** `uv run pytest -m golden` (CPU fingerprints of Tiger, the baseline and every remedy option); `tools/fingerprint/compare.sh 5bea8c3 options --device cuda` and the `wumpus_gpu` case for code under CUDA autocast; anything that is not identical is reverted unless it is a recorded bug fix.
- **Loads:** `tools/fingerprint/load_digests.py check` against the digests of every stored run at `5bea8c3`.
- **Tests:** `uv run pytest` (fast); `uv run pytest -m slow` once on the final commit.

## Baseline at 5bea8c3 (2026-11-17)

- Fast suite: 261 passed, 2 skipped (CUDA-only), 7 deselected (slow), 306 s at nice 19.
- Ledger: 254 claims, 254 OK, 0 not OK (495 values); manifest: 271 files verify.
- Golden fingerprints: 27 CPU cases recorded (`tests/golden/fingerprints/`, 51 golden tests pass in 31 s). The Tiger case reproduces the earlier Tiger fingerprint's output byte for byte when run on that tool's two CPU threads; the golden files pin one thread, because one CPU reduction rounds differently with two.
- Load gate: 140 stored runs (105 training runs, 35 baseline runs) reload strictly; none fails.

## Phase 1 (domain layer and the I-POMDP core): in progress, 2026-11-18

**New modules** (`src/ipomdp/domain/`):

| Module | What it holds |
|---|---|
| `multiagent_tiger.py` | The joint tables of the two-agent tiger game (Gmytrasiewicz & Doshi 2005, Table 1) |
| `policy_graph.py` | Finite-state controllers: exact plan trees, stationary extraction, exact evaluation |
| `interactive.py` | The fold of the other agent's controller into a FinitePOMDP over S x N_j x A_j; the paper's level-0 noise model; the nested recursion |
| `reduction.py` | Exact lumping of observations with proportional likelihoods |
| `bounds.py` | HSVI2 brackets whose lower bound is an executable controller |

The solver's witness LP becomes public; nothing numeric changes.

**Checks** (`tests/test_interactive.py`, 17 tests, CPU, seconds):

| Check | Result |
|---|---|
| Table 1 entries and per-agent views | Equal entry by entry |
| Other agent always listens | The fold equals the canonical Tiger (tables to 1e-15, V_h to 1e-9) |
| Other agent always opens | V_h = -sum gamma^t exactly |
| Uniformly random other agent | Q_2(b0) = -1.95 / -45.95 |
| Folded Bayes filter against brute-force enumeration over the other agent's nodes, actions and observations | Equal to 1e-12 |
| Folded simulator against a direct two-agent simulation | Within 5 s.e. |
| Plan graphs | Reproduce their plans' values |
| Uniform prior over the level-0 agent's belief | Gives the paper's noise 0.1 / 0.8 / 0.1 exactly |
| The paper's Fig. 4 thresholds (level-0 noise POMDP, horizon 3, undiscounted) | Within 0.003 |
| Observation lumping | Exact (1e-12) |
| HSVI on the canonical Tiger | Its bracket contains the certified V*, and its controller attains the lower bound |
| Level-1 problem | Folds the certified level-0 controller |

**Cross-check against an independent implementation** of the same game (`ipomcp_crosscheck.py`, log beside it):
- tables exact;
- Q_h to 2.8e-14 for three opponent laws, four beliefs and four horizons;
- PASSED.

**Not reproduced:** the level-1 action probabilities of the paper's Section 7.3 (L 0.96591). Ours are 0.87-0.90, and an independent brute force over a grid of the other agent's beliefs agrees with our fold to 1e-6.

**Open:**
- a tight certified bracket for the folded level-1 problem. The sawtooth upper bound stalls on the 26-state fold, so the options are being evaluated;
- the Fig. 11 thresholds (a slow test);
- deduplication of the seeded generators and of the Wumpus heading logic;
- a test for `belief.predict_state`.

## Phases 2-4 (belief filter, world model, planner): in progress, 2026-11-19

**Rules during the user's GPU hold (2026-11-19).** Every code change the review finds is on the Wumpus recipe path, and its proof of bit-identity needs `wumpus_gpu` and the CUDA option fingerprints. So only tests, findings and comment or docstring edits are committed now. Each edit is shown code-identical by an AST comparison against 3deb4d7 with docstrings stripped, plus golden (51 identical). The code changes are listed per file below as "Deferred", to be applied in one batch per phase once the GPU is back.

**Does the search get better with more simulations?** This was checked on the exact canonical Tiger model, so no learned component is involved (`planner_budget.py`, log beside it; CPU).
- **Guaranteed, and verified:**
  - Let eps = ||leaf - V*||_inf. Then every root edge satisfies |Q_tree - Q*| <= gamma^{d_a} eps, where d_a = 1 + the shallowest positive-probability frontier depth below the edge.
  - Checked at 6 beliefs, 5 leaf qualities (zero, V1, V3, V8, V12) and budgets 1-4096. The bound is never exceeded; the largest error minus bound is -0.015.
  - The tree after N simulations is a prefix of the tree after N' > N, so d_a never decreases and the bound is non-increasing in N.
  - A from-scratch expectimax of the final tree reproduces every cached value exactly, with and without pruning.
- **Not guaranteed: the decision.**
  - PUCT deepens the promising action first, so with biased leaves the actions are compared on subtrees of unequal depth (the horizon effect).
  - With zero leaves, the greedy agent plays the optimal actions on all 256 episodes at N = 1. It is worse at every larger budget, comes back to -0.65 at N = 64 and falls again to -4.26 at N = 128: the return oscillates with the budget, it does not converge monotonically.
  - V3 leaves dip at N = 4 (-4.13) and approach the optimum from 16 on. V8 leaves are optimal except at N = 8 (-0.75 +- 1.10).
  - Table: paired per episode with the optimal agent (22.375 on these episodes), mean +- 1.96 s.e. over the 256 episodes.
  - Part A agrees at fixed beliefs: with V8 leaves the root decision is optimal at every budget at all six beliefs, while zero and V1 leaves flip it back and forth.
  - The search deepens slowly: d_a is 2-3 at every root after 4096 simulations.

  | Leaves | N = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
  |---|---|---|---|---|---|---|---|---|
  | zero | +0.00 | -5.44 +- 2.88 | -5.93 +- 3.06 | -3.67 +- 2.92 | -2.89 +- 2.79 | -1.06 +- 1.99 | -0.65 +- 1.62 | -4.26 +- 2.81 |
  | V3 | +0.00 | +0.00 | -4.13 +- 2.88 | -1.88 +- 1.73 | -0.60 +- 1.17 | -0.42 +- 0.76 | -0.01 +- 0.81 | -0.23 +- 0.46 |
  | V8 | +0.00 | +0.00 | +0.00 | -0.75 +- 1.10 | -0.01 +- 0.68 | +0.00 | +0.00 | +0.00 |
- **The learned Tiger planner** (PROTOCOL C-q.3, read once; ten JEPA seeds on the same 256 episodes) is flat in the budget. Its mean gap to the optimal agent is between -0.07 and -0.28 per episode at every N from 1 to 128, and every Student-t interval over the ten seeds contains 0 (N = 128: -0.076 [-0.198, +0.046]). Its learned leaves behave like the accurate V8 leaves, not like zero leaves.
- **Reading:**
  - At practical budgets the leaf value and the model set the decision quality, not the number of simulations. With poor leaves the decision is not monotone in the budget even on an exact model.
  - More search converges to the optimal policy of the model it is given, so with a learned model it can exploit the model's errors. That fits the paper's more-search-more-deaths result on Wumpus (PROTOCOL C-g), which C-q.3's registered rule attributes to the Wumpus model.
  - With the exact Wumpus model the planner does improve with the budget: 335 / 424 / 476 per game at 50 / 200 / 1000 simulations (`wumpus_exact_model.py`, section 4).

**Reference agent at the paper's threshold (risk 0.4; `tests/test_wumpus_agents.py`, slow).** Against an independent exact filter, every gamble had an exact danger <= 0.4. Over 64 games, only 6 of its steps were gambles; the largest danger taken was 1/3, and no game ended in a death. It won 68.8% of the games against 59.4% at risk 0.

## Files

| Phase | File | Findings | Change | Check |
|---|---|---|---|---|
| 1 domain and I-POMDP core | `src/ipomdp/domain/__init__.py` | | | |
| 1 domain and I-POMDP core | `src/ipomdp/domain/belief.py` | | | |
| 1 domain and I-POMDP core | `src/ipomdp/domain/env.py` | | | |
| 1 domain and I-POMDP core | `src/ipomdp/domain/pomdp.py` | | | |
| 1 domain and I-POMDP core | `src/ipomdp/domain/solver.py` | | | |
| 1 domain and I-POMDP core | `src/ipomdp/domain/tiger.py` | | | |
| 1 domain and I-POMDP core | `src/ipomdp/domain/wumpus.py` | | | |
| 1 domain and I-POMDP core | `tests/test_domain.py` | | | |
| 1 domain and I-POMDP core | `tests/test_wumpus.py` | | | |
| 2 belief filter | `src/ipomdp/models/__init__.py` | Re-exports every model class except `CausalTransformerFilter`, which callers import from its module (default-off option). | None. | Imported by the fast suite. |
| 2 belief filter | `src/ipomdp/models/ema.py` | Correct: one in-place rule for the three uses (JEPA target, value target, acting copies). The filter and heads hold parameters only, so averaging parameters averages the whole state. | None. | `test_target_filter_is_frozen_copy_and_ema_updates`. |
| 2 belief filter | `src/ipomdp/models/history_filter.py` | Correct: causal mask, and the KV-cached append equals a full re-encode. `nan_to_num(0.0)` after the attention softmax (line 76) never acts on valid input: with every length >= 1 each query sees key 0, so no row is fully masked. It can only hide a NaN produced upstream, which the project's rules forbid. Header: dated narrative; no verdict line (C-j arm TR: FAIL on both rules). | DONE, comment only: verdict line (C-j arm TR). Deferred (code under autocast, GPU gate owed): `encode` raises on a length < 1 and the `nan_to_num` goes (bit-identical on valid input). | Existing: unroll is causal and equals `encode` at every prefix; the KV cache is invariant; token-state children equal update and re-encode. |
| 2 belief filter | `src/ipomdp/models/layers.py` | Correct: RMSNorm computes in float32 (eps 1e-6); a SwiGLU block is the identity at initialisation (zero `w3`). `build_residual_stack`'s default `num_blocks=2` is unused (every caller in `src/` passes it). `typing.List`. Section 3 of the header is history. | Deferred: drop the unused default (after a check of the report scripts); `list[...]`. | `test_swiglu_block_is_identity_at_initialisation`, `test_residual_stack_shapes`. |
| 2 belief filter | `src/ipomdp/models/world_model.py` | Correct: the filter has the exact filter's signature, and the hoisted input network keeps unroll equal to stepping. Header: section 7 dated. Section 2 still says slots return "when the interactive state gives them semantics", but the I-POMDP fold (`domain/interactive.py`) is one FinitePOMDP, so the interactive agent uses the same single latent. | Header to rewrite in single-version voice. | Added `test_unroll_from_an_initial_observation_equals_start_then_steps`: the Wumpus o_0 path had no test. |
| 2 belief filter | `tests/test_history_filter.py` | Covers the plan's unroll and KV-cache checks. | None. | Passes. |
| 2 belief filter | `tests/test_models.py` | Covered the codec, the EMA and unroll against stepping, but not the o_0 path. | One test added (`world_model.py` row). | 62 passed over this file, `test_history_filter.py` and `test_planner_and_agent.py`. |
| 3 world model, heads, losses | `src/ipomdp/models/distributions.py` | Correct: encode and decode are linear in real space, so the mean is exact; out-of-bound and non-finite targets raise. Header section 2 cites "the -100/+10 test below", which is in `tests/test_models.py`. | DONE, comment only: the pointer names `tests/test_models.py`. | `test_encoding_has_exact_mean`, `test_mean_is_unbiased_for_multimodal_targets`, `test_bound_targets_are_encoded_exactly`, `test_invalid_targets_raise`. |
| 3 world model, heads, losses | `src/ipomdp/models/heads.py` | Correct. The header was never closed, section 3b followed section 4, and the options had no verdict lines. | DONE, comment only: 3b moved before 4, header closed, and verdict lines for 2b (`training/run.py` section 9), 2c (PROTOCOL C-e arm 1), 2d (`objective_ab.py` 2p), 3b (C-j arm CH) and 4 (recipe). Open: the single-version rewrite. | AST-identical to 3deb4d7 (docstrings stripped); golden 51 identical. |
| 4 planner and agents | `src/ipomdp/agents/__init__.py` | Fine. | None. | |
| 4 planner and agents | `src/ipomdp/agents/planning_agent.py` | Correct. Section 4 gave the offline gain of the own trials but not their in-agent result. Constructor defaults (`epsilon`, `trial`, `trial_prob`). One of the seven generator copies (line 88; the Phase 1 helper). | DONE, comment only: verdict line (in-agent -11.2 per game, ledger `dev.remedy.own_margin_minus_recipe.seed0`; the registered bar is not met). Deferred: required keyword arguments; trials as an action-override wrapper that keeps the `state_dict` keys. | AST-identical; `TestPlanningAgent`, `TestOwnTrials`. |
| 4 planner and agents | `src/ipomdp/agents/uniform_agent.py` | Fine; a generator copy (line 27). | None (the Phase 1 helper). | `test_actions_are_uniform_and_seeded`. |
| 4 planner and agents | `src/ipomdp/agents/wumpus_reference.py` | Correct as read. `_DELTA` and `_ahead` duplicate the heading table of `domain/wumpus.py` (the Phase 1 heading owner). Header dated. | After the heading owner lands: use it. | New `tests/test_wumpus_agents.py` (slow): at risk 0.4 every gamble has an exact danger <= 0.4 against an independent filter, and it wins more games than risk 0 (section above). |
| 4 planner and agents | `src/ipomdp/planning/__init__.py` | Fine. | None. | |
| 4 planner and agents | `src/ipomdp/planning/mcts.py` | Verified: cached Q and V^ equal a from-scratch expectimax of the final tree, with and without pruning; the tree only grows with the budget; the convergence bound holds. Finding: the bound is monotone in the budget, the decision is not ("Phases 2-4" section above). Constructor default `observation_prune=0.0`; the analysis planner of `experiments/runs.py` relies on it (the copy the plan names). No code writes extra attributes on nodes, so `__dict__` in `DecisionNode.__slots__` serves nothing. Header sections 2 and 3 are history. | Header section 8 on what more simulations do and do not guarantee (comment only). Deferred: `observation_prune` required; `__dict__` out of the slots. | New: `test_cached_values_equal_the_expectimax_of_the_final_tree` (prune 0 and 0.2), `test_the_tree_only_grows_with_the_simulation_budget`; slow `test_root_error_is_bounded_by_the_frontier_depth` (passed, 474 s). |
| 4 planner and agents | `src/ipomdp/planning/search_model.py` | Correct: the wrappers' maths re-derived (the renormalised branches sum to 1; the reward shift follows the implied risk). Section 6 gave only the offline +33 of the risk margin, which the in-agent screen does not reproduce. `TokenSearchModel.horizon = None` is the duck-typing shim (Phase 7). Constructor defaults (`horizon`, `chunk`). | DONE, comment only: section 6 verdict (in-agent -34.9 per game, Student-t [-123.9, +54.0], ledger `dev.remedy.margin_minus_recipe.*`), and verdict lines for sections 5 (`run.py` 9), 7 (`objective_ab.py` 2n and `run.py` 14), 8 (`objective_ab.py` 2p), 9 (C-h arm CBP) and 10 (C-j arm TR). Deferred: required arguments. | AST-identical; existing tests. |
| 4 planner and agents | `src/ipomdp/planning/wumpus_exact_model.py` | Correct as read. With the exact model the planner improves with the budget on development games: 335 / 424 / 476 per game at 50 / 200 / 1000 simulations, 1000 minus 200 paired +51 +- 14 (header section 4). `_exact` is read by report scripts (it becomes public in Phase 7). A generator copy (line 178). | None now. | `tests/test_wumpus.py` (per-node reference). |
| 4 planner and agents | `tests/test_planner_and_agent.py` | Covered the exact one-step lookahead, the expectimax backup and greedy optimality with exact leaves, but nothing with inexact leaves or on growth with the budget. | Three fast tests and one slow test added; the two slow tests share one certified solve (`certified` fixture). | Fast: passes. Slow: the new test passes. |
| 5 training loop, remedies, baseline | `src/ipomdp/baselines/__init__.py` | | | |
| 5 training loop, remedies, baseline | `src/ipomdp/baselines/ba_pomcp.py` | NEW 2026-11-20 (the user's ruling of 2026-11-19: BA-POMCP is the competitor on Tiger and I-Tiger). Written to Katt, Oliehoek & Amato (2017; arXiv 1806.05631, read at the source): factored Dirichlet counts, 1000 particles updated by rejection sampling, UCB with c = H (max R - min R), the expected-model step. Findings: (1) with uniform random rollouts (the paper names none) the agent on the TRUE model listens where the optimum opens. At P(left) = 0.97 it listens at 1,000 and 10,000 simulations, and at 0.9999 at 10,000. The cause is the continuing Tiger: a random continuation is worth about -380, and opening reaches it one step sooner. (2) An EXPECTED_MDP rollout (the optimal policy of the agent's own posterior-mean MDP) removes this: it picks the optimal action at all six beliefs the optimal policy reaches, at 1,000 simulations and 0.08 s per decision. (3) Rewards are not observations in the BA-POMDP, so a symmetric uninformed growl prior can never tell which growl means which side. | New module, imported only by its tests; nothing in the training path reads it. Not exported from `baselines/__init__.py`. | `tests/test_ba_pomcp.py`. |
| 5 training loop, remedies, baseline | `tests/test_ba_pomcp.py` | NEW 2026-11-20. | 14 fast tests (2 s): prior validation, count ratios, sampling never returns a structural zero, exact count bookkeeping in the update, an impossible observation raises, the symmetric uninformed prior stays at 1/2, the point-mass prior chooses the optimal action at the six reachable beliefs, reproducibility. 1 slow test: the point-mass prior plays the optimal return. | Fast: 14 passed (2 s). Slow: passed (274 s; 32 paired episodes of 100 steps at 1,000 simulations). |
| 5 training loop, remedies, baseline | `src/ipomdp/baselines/recurrent_dqn.py` | | | |
| 5 training loop, remedies, baseline | `src/ipomdp/training/__init__.py` | | | |
| 5 training loop, remedies, baseline | `src/ipomdp/training/checkpointing.py` | | | |
| 5 training loop, remedies, baseline | `src/ipomdp/training/episode_buffer.py` | | | |
| 5 training loop, remedies, baseline | `src/ipomdp/training/privileged.py` | | | |
| 5 training loop, remedies, baseline | `src/ipomdp/training/rollout.py` | | | |
| 5 training loop, remedies, baseline | `src/ipomdp/training/run.py` | | | |
| 5 training loop, remedies, baseline | `src/ipomdp/training/seeding.py` | | | |
| 5 training loop, remedies, baseline | `src/ipomdp/training/trainer.py` | | | |
| 5 training loop, remedies, baseline | `tests/test_baselines.py` | | | |
| 5 training loop, remedies, baseline | `tests/test_training.py` | | | |
| 5 training loop, remedies, baseline | `tests/test_training_run.py` | | | |
| 5 training loop, remedies, baseline | `tests/test_world_model_acceptance.py` | | | |
| 6 interpretability and error bounds | `src/ipomdp/interpretability/__init__.py` | | | |
| 6 interpretability and error bounds | `src/ipomdp/interpretability/analysis.py` | | | |
| 6 interpretability and error bounds | `src/ipomdp/interpretability/belief_probe.py` | | | |
| 6 interpretability and error bounds | `src/ipomdp/interpretability/error_bounds.py` | | | |
| 6 interpretability and error bounds | `src/ipomdp/interpretability/wumpus_latents.py` | | | |
| 6 interpretability and error bounds | `tests/test_interpretability.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `README.md` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `analyze.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `conf/config.yaml` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `conf/env/tiger.yaml` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `conf/env/wumpus.yaml` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `main.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `pyproject.toml` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `recreate_venv.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-09-30_reviewer_page/README.md` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-09-30_reviewer_page/scripts/depth.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-09-30_reviewer_page/scripts/diagnose_heads.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-09-30_reviewer_page/scripts/extract.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-09-30_reviewer_page/scripts/extract2.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-09-30_reviewer_page/scripts/head_noise.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-09-30_reviewer_page/scripts/policy_check.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/REMOVED_DIAGNOSTICS.md` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/branch_games.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/counterfactual_diag.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/coverage_games.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/disagreement_screen.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/ensemble_disagreement.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/exact_model_play.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/feature_localise.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/full_info_vi.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/fvi_screen.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/g0_readout.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/go_value_diag.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/grounding_screen.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/head_shift.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/head_swap_screen.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/macro_reach_diag.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/noop_drift_cost.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/objective_ab.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/obs_fit_vs_latent.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/override_diag.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/pessimism_play.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/phantom_reward_diag.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/profile_recipe.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/q_head_screen.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/q_localise_diag.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/retrain_play.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/reward_diag.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/risk_localise.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/saved_networks.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/self_loop_diag.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/supervised_filter.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/target_novelty_smoke.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/target_rule_smoke.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/teacher_plugin_diag.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/teacher_robustness.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/tiger_acceptance_seed.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/turn_map_check.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/unfinished_diag.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/value_chain_diag.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-02_wumpus_diagnostics/value_diag.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-07_wumpus_reviewer_page/scripts/dump_latents.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-07_wumpus_reviewer_page/scripts/export.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-07_wumpus_reviewer_page/scripts/umap_latents.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-19_eval_protocol/PROTOCOL.md` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-19_eval_protocol/ablation_readout.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-19_eval_protocol/exact_planner_play.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-19_eval_protocol/exact_planner_smoke.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-19_eval_protocol/final_retrain_checkpoint.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-19_eval_protocol/recurrent_dqn_baseline.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-19_eval_protocol/search_budget_readout.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-19_eval_protocol/selection_readout.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-19_eval_protocol/untrained_probe_control.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-10-22_option_a/paired_readout.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-03_paper_figures/ladder.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-03_paper_figures/localisation.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-03_paper_figures/tiger_geometry.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-03_paper_figures/tiger_intervention.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-09_audit/grad_share.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-09_audit/precision_check.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-09_audit/s0_phantom.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-09_augmentations/deadly_probe.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-09_augmentations/failure_head_smoke.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-09_augmentations/sc_smoke.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-14_submission_audit/AUDIT_LOG.md` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-14_submission_audit/WORK_ORDERS.md` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-14_submission_audit/claims.yaml` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-14_submission_audit/ledger.md` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-14_submission_audit/ledger.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-14_submission_audit/repro_check.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-14_submission_audit/statistics.md` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-14_submission_audit/w4_accepted_states.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-14_submission_audit/w5b_fatal_deadly.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-14_submission_audit/w7_median_constant.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `reports/2026-11-15_interpretability/tiger_intervention.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/STAGE_TEMPLATE.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/cancelled/17_branch_datactl.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/queue.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/10_branch_seeds12.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/15_branch_seeds34.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/20_abl_max.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/30_mix_play.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/40_abl_noretrain.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/60_abl_jepa.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/70_tiger10.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/80_r2d2_tune.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/90_search_budget.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/a0_s0_screen.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/a1_sc_rows.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/a2_b_screen.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/a3_ego_screen.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/a4_w1_slow_suite.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/a5_w2_reproduction.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/a6_r2d2_c_n.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/a7_w2b_tiger_twice.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/a8_w5a_ve_glitter.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/a9_w4_accepted_states.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/b0_w5b_fatal_deadly.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/c0_w7_median_constant.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/d0_w8_tiger_reanalysis.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/d5_w11_guard_seed5.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/stages.d/e0_g0_oracle_localise.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/week/sweep_retry.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/wumpus/branch/branch_chain.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/wumpus/coverage_dose/smoke.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/wumpus/da_screen_chain.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/wumpus/gpu_window.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/wumpus/grok_chain.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/wumpus/grok_readout.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/wumpus/grok_seeds12.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/wumpus/heldout_resume_chain.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/wumpus/own_coverage_chain.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/wumpus/own_margin_resume.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/wumpus/qe_screen_chain.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/wumpus/risk_localise_chain.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `runs/wumpus/sp_screen_chain.sh` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `src/ipomdp/__init__.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `src/ipomdp/experiments/__init__.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `src/ipomdp/experiments/aggregate.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `src/ipomdp/experiments/runs.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `src/ipomdp/experiments/wumpus.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `src/ipomdp/telemetry/__init__.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `src/ipomdp/telemetry/guardrails.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `src/ipomdp/telemetry/metrics.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `src/ipomdp/telemetry/profiler.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `src/ipomdp/telemetry/system_monitor.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `src/ipomdp/telemetry/visualizers.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `sweep.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `tests/__init__.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `tests/conftest.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `tests/test_experiments.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `tests/test_integration_e2e.py` | | | |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `tests/test_system_monitoring.py` | | | |
