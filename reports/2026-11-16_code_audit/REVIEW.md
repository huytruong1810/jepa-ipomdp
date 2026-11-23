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

**The level-1 benchmark (2026-11-19 to 2026-11-21).**
- The certified bracket of V*(b0) on the 26-state fold is [16.27, 19.67] (`itiger_benchmark.log`): the upper end is the information relaxation at delay 2 (`bounds.py`, section 5), the lower end is HSVI's after 40 seeded trials. Its width is above 0.5, so by the rule fixed beforehand it is reported as a bracket.
- E1's benchmark file needs the lower-bound vectors of a longer search (1,000 trials). A first run lost about 17 hours at its last step, when it built the dense controller of 49,909 vectors (359 GB; `itiger_benchmark_artifact.log`). `hsvi_vector_bracket` returns the vectors without the controller, and since `0a42eac` it checkpoints the search and continues from the checkpoint; a continued run equals the uninterrupted one bit for bit (two tests on the canonical Tiger; checked on the level-1 fold at 2 + 2 against 4 trials). The rerun's log is `itiger_benchmark_artifact_rerun.log`.
- None of this is on the path of a stored run: `hsvi()`, which extracts the controllers that define the interactive problems, is unchanged, and golden is identical (51).

**Review of the files that existed before the audit (2026-11-21; rows under "Files").**
- Read in full: `pomdp.py`, `belief.py`, `env.py`, `tiger.py`, `solver.py`, `wumpus.py`, `__init__.py`, `tests/test_domain.py` and the domain classes of `tests/test_wumpus.py`.
- No finding changes a stored number.
- Two tests were added where a wrong implementation would have passed: `belief.predict_state` on an asymmetric chain (`d04ccf8`), and the arrow's direction along a row (`fc7c793`).
- Three code changes are deferred under the rule of the Phases 2-4 section (recipe path, GPU gate owed): the simulators clone the state they restore; `WumpusSimulator.step` raises before the first `reset`; one owner of the Wumpus heading table.

**Open:**
- DONE 2026-11-23: E1's level-1 benchmark file, bracket [16.5510, 19.6692] with 49,909 lower-bound vectors (`itiger_benchmark_artifact_rerun.log`);
- the level-2 benchmark (bracket of the 14-state fold; `itiger_benchmark.py --level 2`);
- the Fig. 11 thresholds (a slow test). `solver.prune` compares all pairs of vectors in two K x K x S boolean arrays, 4.4 GB each at K = 10,000 on a 44-state fold. That is a candidate cause of the horizon-3 level-1 solve that was killed for memory; it has not been measured;
- the deferred code changes above, and one seeded-generator helper for the seven copies (`env.py:81`, `wumpus.py:356`, `episode_buffer.py:94`, `planning_agent.py:92`, `uniform_agent.py:27`, `wumpus_exact_model.py:178`, `recurrent_dqn.py:268`);
- from the rows of the five new modules (read on 2026-11-21 by a session that did not write them): `envelope` raises on duplicates. DONE 2026-11-23: `Bracket` keeps its successors as an index array.

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
| 1 domain and I-POMDP core | `src/ipomdp/domain/__init__.py` | Correct: the 52 imported names and the 52 names of `__all__` are the same set (50 before `growl_counter` and `tiger_growl_counters`). `__all__` is not fully alphabetical (cosmetic). `bounds` exports `Bracket` and `hsvi` only; the budgeted and relaxed bounds are imported from the module by the benchmark script and the tests. | None. | Imported by the fast suite. |
| 1 domain and I-POMDP core | `src/ipomdp/domain/belief.py` | Correct: three stateless functions; the prediction sums over the departure state; an observation of probability zero raises. `predict_state` had no test of its own, and the canonical Tiger's transition matrices are symmetric (identity, uniform), so a sum over the wrong index would have passed every filter test. | DONE (`d04ccf8`): a test on an asymmetric three-state chain, where the correct first entry is 0.43 and the transposed sum gives 0.21. | `TestBeliefFilter`, 5 tests: brute-force path enumeration on all 216 histories of length 3 (1e-12), the closed-form listen update, the reset after a door, the raise, the chain. |
| 1 domain and I-POMDP core | `src/ipomdp/domain/env.py` | Correct: s0 from b0, s' from T, o from O at the arrival state, r = R[a, s] in float32, all read from the model's tensors; truncation without automatic reset; a private generator. Finding: `load_state_dict` keeps the caller's tensors when they already sit on the simulator's device (`Tensor.to` then returns the same object), and `_elapsed += 1` and `reset_rows` write in place. Measured on the CPU: one step of a restored simulator changes the saved `elapsed` from 1 to 2, so a second restore from the same dict would start one step late. No stored result is affected: the one caller (`TrainingRun.load_state_dict`) restores once per process from a checkpoint read from disk. | Deferred (every run's resume path; GPU gate owed): clone on load, with a test that restores twice from one dict. | Existing: frequencies of b0, T, O and the reward on 200,000 rows within 5 standard errors; the truncation protocol; invalid actions raise; seed reproducibility. |
| 1 domain and I-POMDP core | `src/ipomdp/domain/pomdp.py` | Correct: shapes, float64, finite entries, non-negativity, rows summing to 1 within 1e-9, a discount inside (0, 1) and unique names are checked at construction and raise; nothing is renormalised. `value_bound` is max abs R / (1 - gamma). Header section 4 states the fold of an I-POMDP. The validation test covers six of the invariants; non-finite entries and a non-tensor field are not exercised. | None. | `TestFinitePOMDPValidation` (6 malformed fields). |
| 1 domain and I-POMDP core | `src/ipomdp/domain/solver.py` | Correct as read, three points re-derived: (1) Lark's filter tests a candidate against the kept set only, so the epsilon guarantee cannot chain; (2) a backup runs abs O projection prunes, abs O - 1 cross-sum prunes and one union prune, which gives the stated 2 abs O epsilon; (3) the infinite-horizon bound (gamma times the last change plus the backup error, over 1 - gamma) follows from the triangle inequality and the contraction. Finding 1: the header calls the bound a proof, but every LP is solved by HiGHS at its default feasibility tolerances (1e-7, scipy 1.17.1), which is looser than the exact-pruning epsilon of 1e-9. The tests bound the effect on the Tiger: V_h equals an independent expectimax to 1e-9 for h = 1..5 at eleven beliefs, and the prune guarantee holds to 1e-12 on 300 random vectors. If the LP optima were accurate only to the tolerance, the 1e-4 bound on V*(b0) would grow by about 1e-5, which is (2 abs O + gamma) x 1e-7 / (1 - gamma); the measured accuracy is far better, so no number changes. Finding 2: memory, see "Open" above. | Open, comment only: one header sentence that the certificate holds up to the LP solver's tolerance. The file's code must stay bit-identical. | `TestExactSolver` (reference values for h = 1, 2, 3, 4, 5, 10; expectimax; optimal actions; the prune guarantee at three epsilons; the exact sup-norm against a 20,001-point grid); the Monte-Carlo return of the optimal 5-step policy; slow: V* = 19.3713 certified to 1e-4. |
| 1 domain and I-POMDP core | `src/ipomdp/domain/tiger.py` | Correct: `tiger.95.POMDP` entry for entry, with 0.85 and 0.15 both written out so that the tensor equals the reference file bit for bit. Header section 3 points to the two-agent tables. | None. | `TestTigerSpecification` (`torch.equal` on every table). |
| 1 domain and I-POMDP core | `src/ipomdp/domain/wumpus.py` | Correct as read: 105 x 13 x 12 = 16,380 layouts; breeze and stench tables; turns; the arrow kills a wumpus strictly ahead on the hunter's line; a death or a win pays its action cost; a shot without the arrow costs 1; the end is absorbing; the bounds 100,100 and 1,110 re-derived; every consistent layout shares the hunter's own state, so the tracker's column read is exact. Finding 1: the heading table exists three times in this file (`WumpusWorld._delta`, a second literal in `ego_map_targets`, and the tracker reads the private `world._delta`), once more in `agents/wumpus_reference.py`, and in six report scripts (two read `world._delta`, four carry their own literal). Finding 2: `WumpusSimulator.load_state_dict` has the aliasing of `env.py` for `_elapsed`. Finding 3: unlike `BatchedPOMDPEnv`, the simulator does not reset in its constructor, and `step` before `reset` plays layout 0 in every row without an error. The rollout resets first (`training/rollout.py:81`). Finding 4: `ego_map_targets` checks the shape of `initial_observations` after its loop. Finding 5: no rule test exercised the sign of "ahead" or a shot along a row. | DONE (`fc7c793`): the arrow test. Deferred (recipe path; GPU gate owed): one module-level owner of the heading table and of the cell ahead; clone on load; `step` raises before the first `reset`; the shape check moves to the top. | `TestLayouts`, `TestRules` (7 hand-built scenarios, rewards to the unit), `TestSimulator`, `TestExactFilter` (the true layout stays consistent, the own state is exact, the marginals are calibrated within 5 standard errors); the supervision and dead-reckoning target tests. |
| 1 domain and I-POMDP core | `tests/test_domain.py` | The filter test shares no code with `belief.py` (path enumeration), the solver tests use an independent expectimax, and the statistical tests use fixed seeds with a 5-standard-error tolerance. Gaps: every model in the file was the Tiger, whose transitions are symmetric; no test restores a `BatchedPOMDPEnv` directly (the training-run resume tests do). | DONE (`d04ccf8`): the asymmetric chain. The restore test comes with the deferred fix of `env.py`. | 41 fast tests and 1 slow test. |
| 1 domain and I-POMDP core | `tests/test_wumpus.py` | Domain classes read (`TestLayouts`, `TestRules`, `TestSimulator`, `TestExactFilter`, and the two target tests): the rule tests are the independent check of `step_rules`, because the filter tests use the same rules function as the simulator by design. `test_state_dict_roundtrip` restores once. The classes from `TestLearningStack` on exercise the training stack and the remedy options; they are reviewed with Phase 5. | DONE (`fc7c793`): the arrow test. | 12 tests in `TestLayouts`, `TestRules` and `TestSimulator` after the change, and 2 in `TestExactFilter`. |
| 1 domain and I-POMDP core | `src/ipomdp/domain/multiagent_tiger.py` (new; read by a session that did not write it, 2026-11-21) | Correct against Table 1 as the header states it: the tiger is re-placed unless both listen; a listener's growl reads the arrival state with 0.85 and its creak names the other agent's action with 0.9 (0.05 for each other creak); an opener hears the six pairs uniformly; a reward depends on the own action and the pre-transition state. Both views are built from the joint tables. | None. | `TestTableOne` (4 tests; the listener's entries are checked at three cells). Every entry equals the independent implementation's tables (`ipomcp_crosscheck.log`). |
| 1 domain and I-POMDP core | `src/ipomdp/domain/policy_graph.py` (new) | Correct as read: `evaluate` solves the right linear system over (node, state); `plan_graph` picks, among tied continuations, the combination that reproduces the vector, and raises if none does; `stationary_graph` is never trusted alone (it is evaluated exactly and certified). Finding 1: `envelope` drops BOTH copies of an exact duplicate, because each has margin 0 against the other. Measured: on {(1,0), (1,0), (0,1)} it keeps (0,1) only and the value at (1,0) falls from 1 to 0. Every caller passes a pruned set, and `prune` removes duplicates first, so no result is affected; the precondition is neither stated nor checked. Finding 2: `node_law_on_segment` and `segment_breakpoints` compute the same cut points in two copies. | Open (no stored run uses these functions): `envelope` raises on duplicates; one helper for the cuts. | `TestPolicyGraphs` (5 tests). |
| 1 domain and I-POMDP core | `src/ipomdp/domain/interactive.py` (new) | Correct as read. The fold's index algebra was checked against section 1: j acts from its node, the tiger moves, j's node moves on j's own observation at the arrival state, and i's observation reads the arrival state and j's action; transition, observation, reward and the initial belief use one grid order (s, n, a_j); the kept states are closed under the transition's support. `noise_model` averages T, O and R separately, the paper's level 0. Point for the paper's wording: in `nested_problem`, i's prior over j's first node is the node of j's own initial belief (i knows that j starts at 0.5, the first case of the paper's Fig. 5). The "no information" prior (`node_law_on_segment`) is used in the tests only. | DONE 2026-11-23 (level 2; no stored run uses this path): above level 0 the modelled agent's controller is the best member of a declared family of growl counters by exact evaluation (section 7), because both extractions from the level-1 lower bound fail (`level2_controller.py`: `stationary_graph` gives -14.3 to +5.1; the pointer controller has 32,637 reachable nodes that do not merge). `Level.bracket` is None there. `nested_problem` takes the family as a required argument. The level-1 fold is unchanged (digest 957782ffdac6d103). | `TestDegenerateFolds` (3), `TestFold` (3: brute-force filter to 1e-12, the simulator within 5 standard errors), `TestHierarchy` (4: the threshold-2 counter that ignores creaks is worth the canonical optimum 19.3713; the level-2 problem folds the best declared controller); slow: the level-1 relaxation, and the level-2 problem at the experiment's settings (controller value 15.3567, 14 states). |
| 1 domain and I-POMDP core | `src/ipomdp/domain/reduction.py` (new) | Correct: two observations merge only with the same support over the actions and equal normalised columns (1e-12); the merged column is their sum; `replace` validates the reduced model again; the expanded controller moves on an original observation as on its group. | None. | The lumping test (values equal to 1e-12 on 64 beliefs for horizons 1..4; the canonical Tiger merges nothing). |
| 1 domain and I-POMDP core | `src/ipomdp/domain/bounds.py` (new) | Correct as read. HSVI: each lower-bound vector is built from stored vectors and records them as successors; a point is stored only if it lowers the sawtooth; the corner backup applies all corners together. `revealed_upper_bound`, three steps re-derived: (1) the iteration starts above its own Bellman image, so every iterate bounds the merged graph's fixed point from above; (2) a merge at L1 distance d changes a continuation value by at most span / 2 x d, so the fixed point plus gamma x span / 2 x d / (1 - gamma) dominates the relaxed optimum on the graph, and the code adds the larger span / 2 x d / (1 - gamma); (3) a relaxed value at a belief over histories bounds V* at its marginal, because the past enters nothing ahead. Finding: `_bracket` builds the controller's successor law dense, (K, A, O, K). That lost the 17-hour run (359 GB at K = 49,909) and costs about 2 GB while any Interactive Tiger problem is built; the benchmark script held it for its whole run until `d10b21b` (2.3 GB, then 0.66 GB). A training process drops it when `itiger_setup` returns. | DONE: `hsvi_vector_bracket` without the controller (`bb1ef7a`); checkpoints and the corrected vector count (`0a42eac`). DONE 2026-11-23: `Bracket` keeps actions and successors as index arrays and builds the dense controller on request, refusing it above 4 GB; an Interactive Tiger process now peaks at 0.8 GB instead of 2.9 GB while it builds its problem, with the same fold digest. | `TestBracket` (4, two of them the checkpoint tests), `TestHierarchy` (the dense form equals the index arrays and is refused when too large), `TestRelaxedBound` (4 fast closed forms, 1 slow on E1's problem). |
| 1 domain and I-POMDP core | `tests/test_interactive.py` (new) | 27 fast tests and 2 slow tests (24 and 1 before the level-2 tests of 2026-11-23). The filter and simulator checks share no code with `fold` (path enumeration; a direct two-agent simulation). Two published checks are loose and say so: the Fig. 4 thresholds agree within 0.003 (0.1380 and 0.8620 against the printed 0.135 and 0.865; the other four within 0.001), and the Section 7.3 action probabilities are not reproduced (above). The Fig. 4 test sets the frozen model's discount to 1 with `object.__setattr__`, which bypasses the validation on purpose for an undiscounted finite-horizon solve. | DONE (`0a42eac`): the two checkpoint tests. | `uv run pytest tests/test_interactive.py`: 24 passed. |
| 2 belief filter | `src/ipomdp/models/__init__.py` | Re-exports every model class except `CausalTransformerFilter`, which callers import from its module (default-off option). | None. | Imported by the fast suite. |
| 2 belief filter | `src/ipomdp/models/ema.py` | Correct: one in-place rule for the three uses (JEPA target, value target, acting copies). The filter and heads hold parameters only, so averaging parameters averages the whole state. | None. | `test_target_filter_is_frozen_copy_and_ema_updates`. |
| 2 belief filter | `src/ipomdp/models/history_filter.py` | Correct: causal mask, and the KV-cached append equals a full re-encode. `nan_to_num(0.0)` after the attention softmax (line 76) never acts on valid input: with every length >= 1 each query sees key 0, so no row is fully masked. It can only hide a NaN produced upstream, which the project's rules forbid. Header: dated narrative; no verdict line (C-j arm TR: FAIL on both rules). | DONE, comment only: verdict line (C-j arm TR); 2026-11-21: section 1's date and stage label out. Deferred (code under autocast, GPU gate owed): `encode` raises on a length < 1 and the `nan_to_num` goes (bit-identical on valid input). | Existing: unroll is causal and equals `encode` at every prefix; the KV cache is invariant; token-state children equal update and re-encode. |
| 2 belief filter | `src/ipomdp/models/layers.py` | Correct: RMSNorm computes in float32 (eps 1e-6); a SwiGLU block is the identity at initialisation (zero `w3`). `build_residual_stack`'s default `num_blocks=2` is unused (every caller in `src/` passes it). `typing.List`. Section 3 of the header is history. | DONE, comment only (2026-11-21): section 3 states the scope instead of the removed design. Deferred: drop the unused default (after a check of the report scripts); `list[...]`. | `test_swiglu_block_is_identity_at_initialisation`, `test_residual_stack_shapes`. |
| 2 belief filter | `src/ipomdp/models/world_model.py` | Correct: the filter has the exact filter's signature, and the hoisted input network keeps unroll equal to stepping. Header: section 7 dated. Section 2 still says slots return "when the interactive state gives them semantics", but the I-POMDP fold (`domain/interactive.py`) is one FinitePOMDP, so the interactive agent uses the same single latent. | DONE, comment only (2026-11-21, reviewer seat): single-version header. Section 2 states why the fold needs no slots and where the other agent is read out (the marginal probes); the dated study labels of sections 1, 3 and 5 and the date of section 7 are gone, with every measured number kept. AST-identical to HEAD with docstrings stripped; golden 51 passed. | Added `test_unroll_from_an_initial_observation_equals_start_then_steps`: the Wumpus o_0 path had no test. |
| 2 belief filter | `tests/test_history_filter.py` | Covers the plan's unroll and KV-cache checks. | None. | Passes. |
| 2 belief filter | `tests/test_models.py` | Covered the codec, the EMA and unroll against stepping, but not the o_0 path. | One test added (`world_model.py` row). | 62 passed over this file, `test_history_filter.py` and `test_planner_and_agent.py`. |
| 3 world model, heads, losses | `src/ipomdp/models/distributions.py` | Correct: encode and decode are linear in real space, so the mean is exact; out-of-bound and non-finite targets raise. Header section 2 cites "the -100/+10 test below", which is in `tests/test_models.py`. | DONE, comment only: the pointer names `tests/test_models.py`. | `test_encoding_has_exact_mean`, `test_mean_is_unbiased_for_multimodal_targets`, `test_bound_targets_are_encoded_exactly`, `test_invalid_targets_raise`. |
| 3 world model, heads, losses | `src/ipomdp/models/heads.py` | Correct. The header was never closed, section 3b followed section 4, and the options had no verdict lines. | DONE, comment only: 3b moved before 4, header closed, and verdict lines for 2b (`training/run.py` section 9), 2c (PROTOCOL C-e arm 1), 2d (`objective_ab.py` 2p), 3b (C-j arm CH) and 4 (recipe). DONE, comment only (2026-11-21): the single-version rewrite (dates, stage names and the earlier-design note out; section 1 cites the fold and says why no opponent input; section 4 cites `supervised_filter.py` for the privileged-supervision result). | AST-identical to 3deb4d7 (docstrings stripped); golden 51 identical. |
| 4 planner and agents | `src/ipomdp/agents/__init__.py` | Fine. | None. | |
| 4 planner and agents | `src/ipomdp/agents/planning_agent.py` | Correct. Section 4 gave the offline gain of the own trials but not their in-agent result. Constructor defaults (`epsilon`, `trial`, `trial_prob`). One of the seven generator copies (line 88; the Phase 1 helper). | DONE, comment only: verdict line (in-agent -11.2 per game, ledger `dev.remedy.own_margin_minus_recipe.seed0`; the registered bar is not met). Deferred: required keyword arguments; trials as an action-override wrapper that keeps the `state_dict` keys. | AST-identical; `TestPlanningAgent`, `TestOwnTrials`. |
| 4 planner and agents | `src/ipomdp/agents/uniform_agent.py` | Fine; a generator copy (line 27). | None (the Phase 1 helper). | `test_actions_are_uniform_and_seeded`. |
| 4 planner and agents | `src/ipomdp/agents/wumpus_reference.py` | Correct as read. `_DELTA` and `_ahead` duplicate the heading table of `domain/wumpus.py` (the Phase 1 heading owner). Header dated. | DONE, comment only (2026-11-21): the date is out, and the citation of an untracked file (CLAUDE.md) is replaced by the reason itself (a non_blocking copy read before synchronisation is stale). After the heading owner lands: use it. | New `tests/test_wumpus_agents.py` (slow): at risk 0.4 every gamble has an exact danger <= 0.4 against an independent filter, and it wins more games than risk 0 (section above). |
| 4 planner and agents | `src/ipomdp/planning/__init__.py` | Fine. | None. | |
| 4 planner and agents | `src/ipomdp/planning/mcts.py` | Verified: cached Q and V^ equal a from-scratch expectimax of the final tree, with and without pruning; the tree only grows with the budget; the convergence bound holds. Finding: the bound is monotone in the budget, the decision is not ("Phases 2-4" section above). Constructor default `observation_prune=0.0`; the analysis planner of `experiments/runs.py` relies on it (the copy the plan names). No code writes extra attributes on nodes, so `__dict__` in `DecisionNode.__slots__` serves nothing. Header sections 2 and 3 are history. | Header section 8 on what more simulations do and do not guarantee (comment only). DONE, comment only (2026-11-21): single-version rewrite of sections 2-4 (the mean-backup evidence as the reason for expectimax; the old bug list restated as the invariants it left: reward-inclusive scores, no random-action fallback, no global generator) and of the dated headings and "earlier code" wording of sections 6-8. AST-identical; golden 51 passed. Deferred: `observation_prune` required; `__dict__` out of the slots. | New: `test_cached_values_equal_the_expectimax_of_the_final_tree` (prune 0 and 0.2), `test_the_tree_only_grows_with_the_simulation_budget`; slow `test_root_error_is_bounded_by_the_frontier_depth` (passed, 474 s). |
| 4 planner and agents | `src/ipomdp/planning/search_model.py` | Correct: the wrappers' maths re-derived (the renormalised branches sum to 1; the reward shift follows the implied risk). Section 6 gave only the offline +33 of the risk margin, which the in-agent screen does not reproduce. `TokenSearchModel.horizon = None` is the duck-typing shim (Phase 7). Constructor defaults (`horizon`, `chunk`). | DONE, comment only: section 6 verdict (in-agent -34.9 per game, Student-t [-123.9, +54.0], ledger `dev.remedy.margin_minus_recipe.*`), and verdict lines for sections 5 (`run.py` 9), 7 (`objective_ab.py` 2n and `run.py` 14), 8 (`objective_ab.py` 2p), 9 (C-h arm CBP) and 10 (C-j arm TR). DONE, comment only (2026-11-21): the dates of sections 5-10 are out, and section 1 cites the test instead of a phase label. Deferred: required arguments. | AST-identical; existing tests. |
| 4 planner and agents | `src/ipomdp/planning/wumpus_exact_model.py` | Correct as read. With the exact model the planner improves with the budget on development games: 335 / 424 / 476 per game at 50 / 200 / 1000 simulations, 1000 minus 200 paired +51 +- 14 (header section 4). `_exact` is read by report scripts (it becomes public in Phase 7). A generator copy (line 178). | None now. | `tests/test_wumpus.py` (per-node reference). |
| 4 planner and agents | `tests/test_planner_and_agent.py` | Covered the exact one-step lookahead, the expectimax backup and greedy optimality with exact leaves, but nothing with inexact leaves or on growth with the budget. | Three fast tests and one slow test added; the two slow tests share one certified solve (`certified` fixture). | Fast: passes. Slow: the new test passes. |
| 5 training loop, remedies, baseline | `src/ipomdp/baselines/__init__.py` | | | |
| 5 training loop, remedies, baseline | `src/ipomdp/baselines/ba_pomcp.py` | NEW 2026-11-20 (the user's ruling of 2026-11-19: BA-POMCP is the competitor on Tiger and I-Tiger). Written to Katt, Oliehoek & Amato (2017; arXiv 1806.05631, read at the source): factored Dirichlet counts, 1000 particles updated by rejection sampling, UCB with c = H (max R - min R), the expected-model step. Findings: (1) with uniform random rollouts (the paper names none) the agent on the TRUE model listens where the optimum opens. At P(left) = 0.97 it listens at 1,000 and 10,000 simulations, and at 0.9999 at 10,000. The cause is the continuing Tiger: a random continuation is worth about -380, and opening reaches it one step sooner. (2) An EXPECTED_MDP rollout (the optimal policy of the agent's own posterior-mean MDP) removes this: it picks the optimal action at all six beliefs the optimal policy reaches, at 1,000 simulations and 0.08 s per decision. (3) Rewards are not observations in the BA-POMDP, so a symmetric uninformed growl prior can never tell which growl means which side. (4) CORRECTED 2026-11-21: with the true model and EXPECTED_MDP the root decisions are optimal at those six beliefs, but play is not optimal at 1,000 simulations. On the 256 paired episodes it loses 6.22 +- 3.27 per episode, with a wrong door in 7% of them (`reports/2026-10-19_eval_protocol/ba_pomcp_point_mass_check.log`, a post hoc diagnostic of C-r). I had stated optimal play on the strength of a 32-episode slow test whose tolerance was wider than that deficit. | New module, imported only by its tests; nothing in the training path reads it. Not exported from `baselines/__init__.py`. Header 4b states the deficit (comment only, AST-identical). | `tests/test_ba_pomcp.py`. |
| 5 training loop, remedies, baseline | `tests/test_ba_pomcp.py` | NEW 2026-11-20. 2026-11-21: the slow test "the point-mass prior plays the optimal return" was wrong in substance (it passed only because 5 standard errors over 32 episodes exceeded the real 6.2 deficit) and is removed; its header names the measured deficit. | 15 fast tests (5 s): prior validation, count ratios, sampling never returns a structural zero, exact count bookkeeping in the update, an impossible observation raises, the symmetric uninformed prior stays at 1/2, the point-mass prior chooses the optimal action at the six reachable beliefs, reproducibility, and simulator rows independent of the other rows' actions. | 15 passed. |
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
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `analyze.py` | E1 plumbing (2026-11-20): an itiger run went to analyze_run and would have started an exact solve of the fold, which is out of reach. | itiger runs go to `analyze_interactive_run` and need `--benchmark`; the flag is refused for other domains. Phase 7 review still open. | `--help`; the slow end-to-end test in `tests/test_interactive_analysis.py`. |
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
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `src/ipomdp/experiments/interactive_analysis.py` | NEW 2026-11-20 (E1 plumbing). It analyses an itiger run against its fold's certified bracket, read from a benchmark file tied to the fold by a digest. The reference agent is the exact filter plus the best lower-bound vector's action (value >= L). The analysis reports the KL on the interactive state and on both marginals (physical, other node). ADDED 2026-11-21 (reviewer seat): E1's probe control (module section 5). The same probes run on the run's untrained acting filter, a TrainingRun built from the run's own config and seed over the loaded domain, on the trained probes' own histories: analyze_beliefs' random streams replayed, and section 3's episodes kept. There is no agent-history state probe, because the planner's draws cannot be replayed. | New module and new test file; the Tiger analysis path is not touched (analyze_run gained only a guard). The control adds `analyze_interactive(initial_filter=...)`, `initial_acting_filter`, `ProbeControl` and report.json["untrained"]. | `tests/test_interactive_analysis.py`: 7 fast tests (11 s) and 1 slow end to end (104 s). New: with the trained filter in its place, the control reproduces every trained MLP report exactly; a mutation that replays other streams breaks that equality (checked once by hand). End to end, the control's filter equals the run's filter before its first update, not best.pt. Golden 51 passed; `test_interactive.py` and `test_experiments.py`: 40 passed. |
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `src/ipomdp/experiments/runs.py` | E1 plumbing (2026-11-20): `analyze_run` would have run the exact solver on an itiger fold. Phase 7 review still open. | Guard: it raises for an itiger run before loading it. Tiger is unchanged: the guard only reads the run's config and compares the domain name. | The slow test checks the TypeError; `tests/test_experiments.py` passes. |
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
