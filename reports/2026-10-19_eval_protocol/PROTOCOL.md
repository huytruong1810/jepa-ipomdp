# ABSOLUTE PATH: reports/2026-10-19_eval_protocol/PROTOCOL.md

# Frozen evaluation protocol (registered 2026-10-19, before any held-out number was read)

## Why
Every Wumpus screen, threshold and design choice up to 2026-10-18 was read on training seeds 0–4 and on the same 512 analysis games (analysis seed 12345). That reuses the test set: pre-registration protects each readout, not the selection made across about 30 readouts (gap 2). The heuristic's risk threshold of 0.4 was also chosen on those games. From now on, headline numbers come only from the held-out set below.

## Sets
| Set | Training seeds | Analysis seed | Output directory | Status |
|---|---|---|---|---|
| Development | 0–4 | 12345 | `analysis/` | Everything read so far. **Exploratory only.** |
| Held-out | 5–9 | 20261019 | `analysis_heldout/` | Confirmatory. Each claim is read **once**. |

- If a confirmatory interval is too wide to decide a claim, extend the held-out training seeds to 10–14. Never fall back to 0–4.
- Re-analysing development runs (seeds 0–4) on analysis seed 20261019 is allowed and labelled exploratory. It measures how much the 12345 games flattered the development numbers. It does not count as confirmatory, because those training seeds took part in selecting the recipe.
- Before 2026-10-19, no Wumpus run used a training seed ≥ 5 and no script used seed 20261019 (checked by grep and directory listing).

## Frozen choices
- **Agent recipe:** whole-buffer retrain + double value, `runs/sweeps/wumpus_retrain/sweep.json`:
  - `env=wumpus model.representation=decoder model.observation_horizon=5 training.updates_per_collection=64 training.total_episodes=6400 training.store_eval_episodes=true agent.epsilon=0.25 agent.epsilon_min=0.1 model.value_estimator=double training.retrain_every=50 training.retrain_updates=6400`
  - Checkpoint: `best.pt`, as every development number used. `latest.pt` is reported beside it.
- **References:**
  - Uniform random play.
  - The exact-belief heuristic at risk 0.4, the threshold frozen from the development set.
  - The exact-belief online planner (`ExactWumpusSearchModel`). Its simulation count and rollout count are fixed on **development** games, in its module header, before any held-out play. Fixed 2026-10-19 04:20, before any held-out play, by the smoke's pre-written choice rule: full_info leaves, 1000 simulations, no rollouts (476 ± 41 on 128 development games, against 424 ± 41 at 200 and 544 for the heuristic on the same games). It plays once per analysis seed (`exact_planner_play.py`) and enters the aggregate through `sweep.py --exact-planner-games`.
- **Games:** 512 per agent, one simulator stream per analysis seed, so every agent plays the same layouts (paired).
- **Statistics** (`src/ipomdp/experiments/aggregate.py`, sections 2, 3 and 6):
  1. The mean over seeds with a Student-t 95% interval.
  2. The interquartile mean over seeds with a percentile-bootstrap 95% interval (Agarwal et al. 2021).
  3. The hierarchical bootstrap (seeds, then games) 95% interval. For gaps it is computed on per-game paired differences. **This is the interval cited for confirmatory claims**, because it alone contains both training and evaluation noise.

## Confirmatory claims (read once each on the held-out set)
- **C-a (performance).** The recipe's mean discounted return per game, with its paired gaps to random, the heuristic (risk 0.4) and the exact-belief planner. Win, death and unfinished rates are reported beside it. There is no pass threshold: these are the paper's headline numbers.
- **C-b (representation).** MLP-probe cell and heading accuracy and pit hazard error, on random and on agent histories, from the same held-out analysis.
- **C-c (model-free baseline; registered 2026-10-19 ~05:00, before any baseline run on either set).** The recipe against R2D2-lite (`src/ipomdp/baselines/recurrent_dqn.py`: recurrent double DQN, n-step, value rescaling, Ape-X epsilon ladder) at the matched budget (448,000 environment steps, 19,200 updates, the same data schedule including stored greedy evaluations, best checkpoint by the same evaluation). Quantity: the paired gap recipe minus baseline per (training seed, game), hierarchical-bootstrap 95% interval; the baseline's return, win, death and unfinished rates beside it. No pass threshold: it is reported either way.
  - **Development screen** (training seeds 0-2, analysis seed 12345, `--output analysis`; `reports/2026-10-19_eval_protocol/recurrent_dqn_baseline.py`): a 2 x 2 grid of learning rate {1e-4 (R2D2), 3e-4} x n-step {5 (R2D2), 10}, every other field at its default. Choice rule: the highest mean best-network return over the 3 seeds; if the best arm is within one standard error (over seeds) of the R2D2 defaults (1e-4, 5), keep the defaults. The choice is recorded in the module header 4 before the held-out runs.
  - **Held-out run:** the chosen setting on training seeds 5-9, analysis seed 20261019, `--output analysis_heldout`, then paired with the recipe's held-out games.
- **C-d (checkpoint selection; registered 2026-10-27, before any held-out play of the final-retrain networks).** The recipe acting with the networks of its final whole-buffer retrain (`training.checkpoint_selection=final_retrain`, `main.py` section 5) against the same runs' `best.pt` (the best of the 128-game evaluations, the selection behind C-a).
  - **Why the existing held-out runs answer it:** the option changes only which networks are saved, never training. The held-out recipe runs (`runs/sweeps/wumpus_retrain_heldout`, training seeds 5-9) are therefore runs of the recipe under either rule. Their `latest.pt` holds the collection-100 networks, i.e. the final retrain, and nothing has played them on analysis seed 20261019.
  - **Development evidence (exploratory):** +60.4 [+1.7, +119.1] per game over 5 development seeds (seeds 0-2 found after the fact, seeds 3-4 a pre-registered confirmation; `training/run.py` section 14).
  - **Known before the read:** the C-a numbers of the `best.pt` arm, and the `best.pt` collections: 100 / 100 / 95 / 90 / 90 on seeds 5-9. On seeds 5 and 6 both rules select the same networks, so their per-game differences are 0 by construction (their re-analysis must reproduce `analysis_heldout/games.npz` exactly; a mismatch voids the read). Only seeds 7-9 carry the effect.
  - **Procedure:** `reports/2026-10-19_eval_protocol/final_retrain_checkpoint.py` writes each run's `checkpoints/final_retrain.pt` from `latest.pt` (the same networks; it asserts collection 100 is a retrain). Then `sweep.py wumpus_retrain_heldout_final --seeds 5,6,7,8,9 --analysis-seed 20261019 --analysis-output analysis_heldout_final --checkpoint final_retrain.pt <recipe overrides>` runs over the same seed directories (symlinked). The readout is `reports/2026-10-19_eval_protocol/selection_readout.py`, on the CPU from the two `games.npz` per seed.
  - **Quantity:** the per-game paired difference final retrain minus `best.pt`, with its hierarchical-bootstrap 95% interval over the 5 seeds (seeds, then games). Death and unfinished rates of both arms are reported beside it.
  - **Decision rule:**
    - Interval above 0: **PASS**. `final_retrain` becomes the recipe's selection, and later headline numbers use it.
    - Interval contains 0: **UNDECIDED**. Extend to training seeds 10-14 (trained with `final_retrain`, both `best.pt` and `best_eval.pt` analysed), registered here before launch.
    - Interval below 0: **FAIL**. `best.pt` stays.
- **C-e (leave-one-out ablation; registered 2026-10-31, before any ablation run on any held-out seed).** Each arm removes one design decision from the frozen recipe and is trained on the held-out training seeds 5-9 with the recipe's command otherwise unchanged, then analysed on analysis seed 20261019 (`--analysis-output analysis_heldout`, `best.pt`, as C-a). No arm was ever selected on these seeds: the recipe was chosen on development seeds, and the arms are fixed here.
  - **Arms** (sweeps `abl_max`, `abl_noretrain`, `abl_jepa`; recipe overrides as in "Frozen choices" with one change each):
    1. `abl_max`: `model.value_estimator=max` (the plain maximum backup instead of the double value).
    2. `abl_noretrain`: without `training.retrain_every=50 training.retrain_updates=6400` (no whole-buffer retraining).
    3. `abl_jepa`: `model.representation=jepa` (the JEPA filter instead of the decoder).
  - **Quantity:** per arm, the per-game paired difference recipe minus arm (`runs/sweeps/wumpus_retrain_heldout/seed<k>/analysis_heldout/games.npz` against the arm's), hierarchical-bootstrap 95% interval over the 5 seeds, then games. Win, death and unfinished rates and the C-b probes of each arm beside it.
  - **Decision rule, per arm:** interval above 0: the removed component **contributes**. Interval contains 0: **no detectable contribution** (reported as such). Interval below 0: the component **hurts**. Three arms are read; no correction is applied, and the paper states that three intervals were read.
  - **Order and stopping:** arms run in the order 1, 2, 3 in one serial queue. An arm that has not finished all 5 seeds by the end of the week is reported with the seeds it has, labelled incomplete, and not given a verdict.
- **C-f (second tuning round of the model-free baseline; registered 2026-11-03 21:10, before any of its runs).** The C-c screen varied only lr and n-step. C-f varies the two matched-budget levers C-c left at default, so that the baseline in the paper is not under-tuned. The budget is unchanged: 448,000 environment steps, 19,200 updates and the same data schedule.
  - **Development screen** (training seeds 0-2, analysis seed 12345, `--output analysis`, sweep names `r2d2_f1`..`r2d2_f3`; runner `runs/week/stages.d/80_r2d2_tune.sh`). It has three arms against the C-c defaults (`r2d2_lr1e-4_n5`, already run on these seeds):
    - F1: `batch_size=64` (twice the replayed sequences per update).
    - F2: `hidden_dim=512`.
    - F3: `nstep=10 batch_size=64` (C-c's best arm plus F1).
  - **Choice rule.** The arm with the highest mean best-network return over seeds 0-2 is a candidate. It is CHOSEN only if its paired per-seed difference from the defaults has a mean greater than one standard error of that difference. Otherwise the defaults stay, C-f stops, and the paper says that a second screen found no better setting. The choice is recorded in `src/ipomdp/baselines/recurrent_dqn.py` header 4.
  - **Held-out read** (only if an arm is chosen): the chosen arm on held-out training seeds 5-9, analysis seed 20261019, `--output analysis_heldout`, blind. Quantity: recipe minus tuned baseline, per (training seed, game), with a hierarchical-bootstrap 95% interval, plus the tuned baseline's return and its win, death and unfinished rates. There is no threshold and it is reported either way. The paper's model-free row becomes the tuned baseline, and the text states that both settings were run and gives the C-c number.
- **C-g (search budget of the learned planner; registered 2026-11-04 07:30, before any held-out play at another simulation count).** This is descriptive: it fills the search-ablation gap and tests one prediction. The recipe networks (`runs/sweeps/wumpus_retrain_heldout/seed<k>`, best.pt, k = 5..9) play the held-out games (analysis seed 20261019) with their own learned model at N = 10 and N = 100 simulations per decision. N = 50 is the recipe's own count, C-a's `learned_planner.return`.
  - Runner: `reports/2026-10-02_wumpus_diagnostics/exact_model_play.py <run> learned --heldout --sims=<N>`, which writes `analysis_heldout/exact_play_learned_sims<N>.npz` (section 7 of that script). Stage `runs/week/stages.d/90_search_budget.sh`; readout `reports/2026-10-19_eval_protocol/search_budget_readout.py`.
  - Quantities: the return at each N, and the paired per-game differences N100 - N50 and N10 - N50, with hierarchical-bootstrap 95% intervals over seeds and then games. Win, death and unfinished rates are reported beside them.
  - Prediction (from the paper's account): the shortfall is a model error that more search over the same model cannot remove. So N100 - N50 is small next to the exact-model gain of +168 [+139, +195] (held-out). Concretely, the interval's upper end is below +84, half of that gain. If the upper end is at or above +84, the paper reports that search budget is a substantial lever. Either outcome is reported.
- **C-h (Stage-0 estimation fixes; registered 2026-11-10, before any run of its arms).** Development screen only; no held-out claim yet. Four default-off options, each fixing one measured defect of the value estimate (`model.codec_bound=absorbing`: two-hot range of the absorbing game; `model.target_prune=true`: targets pruned like the planner; `model.double_partition=slot`: each stored episode trains one head of the value pair; `trainer.separate_value_clip=true`: the value heads clipped on their own norm). Offline retrain screen on development seeds 0-2 (arms CB, CBP, SLOT, VCLIP against the existing control retrains), defined in `reports/2026-10-02_wumpus_diagnostics/objective_ab.py` header 2r.
  - **Decision rule, per arm:** PLAY passes if arm minus control is positive on 3 of 3 seeds with a mean of at least +20 per game, and deaths rise by no more than 2 points on any seed. For the codec arms, PHANTOM also has to pass: the share of the planner's imagined children valued above 999 must be under 1% on 3 of 3 seeds and below the control's. Four arms are read and no correction is applied.
  - **Consequence:** arms that pass enter the in-agent run R2 (registered here before launch), which gets its own held-out claim. If none passes, the defaults stay.
- **C-i (smoke S-C, the gate of ensemble information-directed collection; registered 2026-11-10, before any run).** On development seeds 0-2, five bootstrapped observation heads are trained on frozen latents. The score span x JSD at the turned latent under FORWARD must rank the declined turn-then-step decisions above the other non-FORWARD decisions with AUC >= 0.65 on 2 of 3 seeds; otherwise augmentation C is closed and not built. Defined in `reports/2026-11-09_augmentations/sc_smoke.py`.
- **C-j (Stage B architecture screen; registered 2026-11-11, before any run of its arms).** Development screen on seeds 0-2. Arms: TR (`model.filter=transformer`), CH (`model.observation_head=chain`) and TRCE (both, plus `model.kinematic_weight=0.1`), each against the existing control retrains. Defined in `objective_ab.py` header 2s.
  - **Decision rule, per arm:** MECHANISM passes if the glitter ratio at the declined turned latents is at least the control's + 0.15 and the "cell ahead unvisited" accuracy there is at least the control's + 0.10, on 2 of 3 seeds. PLAY passes if arm minus control is positive on 2 of 3 seeds and deaths rise by no more than 2 points on any seed. An arm passes only if both pass. Three arms are read and no correction is applied.
  - **Consequence:** passing arms enter R2. If none passes, Stage B is closed and reported as a negative result.
- **C-k (smoke M-A, the gate of chance-constrained search; registered 2026-11-13, before any run).** Development smoke on seeds 0-2. For each stored network set, a dedicated failure head p(death | z, a) is trained by binary cross-entropy on that set's frozen acting latents and its own training games. The sets are the coverage control (`runs/wumpus/coverage/control_seed<s>.pt`), the union of cover and own games (`union_seed<s>.pt`) and random restore (`runs/wumpus/branch/random_seed<s>.pt`). The head is read at the set's own fatal chosen FORWARDs on the 512 analysis games, against the exact danger of the cell ahead. Defined in `reports/2026-11-09_augmentations/failure_head_smoke.py`.
  - **Decision rule (union set):** CALIBRATION passes if, on 2 of 3 seeds, the median of head / exact danger over the fatal chosen FORWARDs is at least 0.25. The observation head's own P(GAME_OVER) reaches 0.02-0.05 there. The head must also not be inflated: on every seed, its mean over risky live states divided by the mean exact danger there must be at most 1.5. Control and random restore are reported only.
  - **Consequence:** if CALIBRATION passes, the constrained search is built and CC-play on the union networks is registered here before it runs (deaths at most control + 2 points, and at least half of the union arm's idling cure kept). Otherwise augmentation A is closed and not built.
- **C-l (deadly-cell encoding probe; registered 2026-11-13, before any run).** Diagnostic on development seeds 0-2. Question: when the agent faces a cell it has proven deadly (exact danger of the cell ahead at least 0.999) and the observation head misses it (max(P_O, P_R) < 0.5, the only provably deadly states it steps into), is "deadly ahead" in the latent?
  - **Method:** an MLP probe is trained on the frozen acting latents of the run's buffer states, with the exact danger ahead as the target (privileged, diagnostic only). It is read at the analysis-game states with a provably deadly cell ahead, split by whether the head flagged or missed them. Network sets: coverage control (decision) and union (reported). Defined in `reports/2026-11-09_augmentations/deadly_probe.py`.
  - **Decision rule (control set):**
    - IN-LATENT if the probe's recall (prediction at least 0.5) at the head-missed states is at least 0.7 on 2 of 3 seeds. The information is there and the head fails to use it, so the next fix targets the head's training signal.
    - NOT-IN-LATENT if that recall is at most 0.4 on 2 of 3 seeds. The next fix targets the representation.
    - MIXED otherwise.
- **C-m (dead-reckoned ego map; registered 2026-11-13, before any run of its arms).** Development screen on seeds 0-2. Arms: EGO1 (`model.ego_map_weight=1.0`) and EGO01 (`model.ego_map_weight=0.1`). Each is compared with the existing control retrains (`runs/wumpus/ve_screen/control_seed<s>.pt`). The term trains a readout of the hunter's own cell, heading and visited cells into the filter. The domain computes these targets from the agent's own actions and BUMP bits (no layout or belief). Defined in `objective_ab.py` header 2t.
  - **MISS RATE** is the share of live analysis-game states with a provably deadly cell ahead (exact danger at least 0.999) at which the observation head's max(P_O, P_R) is below 0.5. It comes from `risk_localise.py` on the arm and on the control, in the same stage.
  - **Decision rule, per arm:** MECHANISM passes if the arm's MISS RATE is at most the control's minus 0.10 on 2 of 3 seeds. PLAY passes if arm minus control is positive on 2 of 3 seeds and deaths rise by no more than 2 points on any seed. An arm passes only if both pass. Two arms are read and no correction is applied.
  - **Consequence:** a passing arm enters R2. If none passes, the ego map is closed.
- **C-n (third tuning round of the model-free baseline; registered 2026-11-14, before any of its runs).** C-c and C-f never screened two of the baseline's choices. First, its exploration: the Ape-X ladder has a fixed mean epsilon of about 0.065 and never anneals, while the planner explores with 0.25 decaying to 0.1. Second, its value rescaling h, which fits E[h(G)] and decodes a biased value for stochastic returns. Both are now default-off options in `src/ipomdp/baselines/recurrent_dqn.py` header 5. The budget is unchanged: 448,000 environment steps, 18,432 updates and the same data schedule.
  - **Development screen** (training seeds 0-2, analysis seed 12345, `--output analysis`; runner `runs/week/stages.d/a6_r2d2_c_n.sh`). Three arms against the C-c defaults (`r2d2_lr1e-4_n5`, already run on these seeds):
    - N1 (`r2d2_n1`): `exploration=anneal`, the planner's schedule: one epsilon for all rows, 0.25, times 0.97 per collection, floor 0.1.
    - N2 (`r2d2_n2`): `value_scale=linear`, targets and Q-values divided by 1000. This is a linear map, so the regression decodes E[G] without bias.
    - N3 (`r2d2_n3`): both.
  - **Choice rule (as C-f).** The arm with the highest mean best-network return over seeds 0-2 is a candidate. It is CHOSEN only if its paired per-seed difference from the defaults has a mean greater than one standard error of that difference. The readout prints the per-seed values, the Student-t 95% interval on the per-seed differences and the standard error.
  - **Held-out read (only if an arm is chosen):** the chosen arm on held-out training seeds 5-9, analysis seed 20261019, `--output analysis_heldout`, blind. It is reported as in C-f, and the paper's model-free row becomes that arm. Either way, the screen is reported.
- Later method claims (for example, a new objective against this recipe) must be added to this file **before** their held-out runs start. Each states its comparison, its interval and its decision rule.
- **C-o** is reserved for the ten-seed held-out headline (seeds 10-14; ordered 2026-11-15, deferred until the method is settled; nothing is registered under it yet).
- **C-p (training-time belief supervision as a remedy for the one-step model error; development screen, registered 2026-11-15 before any of its runs).** The localisation puts the error in the observation head at the decisions the planner declines, and no head on the frozen latent recovers the exact probability there. The literature offers one remedy that needs neither a calibrated error signal nor data at the declined actions: supervise the filter in training with the hidden state revealed in hindsight (Lee et al. 2023; Lambrechts et al. 2024), so that the latent carries the posterior the head needs. The exact-belief targets of `model.privileged_belief_loss` (cell, heading, per-cell pit / wumpus / gold posteriors, danger ahead; `domain/wumpus.py` section 6) are the zero-variance form of that signal: the cross-entropy minimiser of realised-state targets is the same posterior. They use the known model in training only; the acting agent sees observations only. A claim about hindsight state would need the realised-state implementation, which is built only if G2 passes.
  - **G0 (measurement on stored networks; no training).** Networks: `best.pt` of the three stored runs `runs/wumpus/u64_store_eval_eps_decoder_k5_privileged_double_seed{0,1,2}` (the 2026-10-15 recipe with `model.privileged_belief_loss=true` and `model.value_estimator=double`, no whole-buffer retrain; best.pt from collections 85 / 70 / 65; 216 / 231 / 223 per game with 68-70% of games unfinished on the development analysis games). On analysis seed 12345, per seed: `q_localise_diag.py <run> 512`, `feature_localise.py <run> cuda` (512 games), and `exact_model_play.py <run> learned` and `EPR` on the same 512 games (amended before any run: the first text said 256 games, but the script stores per-game returns and checks the learned replay against `report.json` only for the full 512). Reported per seed: the joint glitter ratio at the declined turned latents (`joint_glitter_ratio_c`; the recipe's is 0.22-0.44), the MLP probe's gold calibration and "cell ahead unvisited" accuracy at those latents, the flip shares under the learned model and under the exact one-step model, and EPR minus learned paired per game with deaths and unfinished games. Readout script `reports/2026-10-02_wumpus_diagnostics/g0_readout.py`.
  - **Readings.** HEAD REPAIRED: joint glitter ratio at least 0.8 on at least 2 of 3 seeds. MODEL STILL BINDS: EPR minus learned at least +50 per game on at least 2 of 3 seeds.
  - **Consequences.** HEAD REPAIRED: G1 arm A is run (the never-run combination: `model.privileged_belief_loss` with the current recipe, whole-buffer retrain and double value, as an offline retrain against the control retrain). Head not repaired and MODEL STILL BINDS: belief supervision does not close a coverage hole in (latent, action) space; only G1 arm B is built (a state-factored observation model in the backup: the belief readout combined with an observation model learned on state-action pairs). Head not repaired and the model does not bind: the screen is closed; on a belief-supervised latent the remaining idling is not a one-step-model error, and that is reported.
  - **G1 and G2 (bars fixed now; arm definitions go into `objective_ab.py`'s header before the first G1 run).** G1, offline retrain on seeds 0-2: MECHANISM joint glitter ratio at least 0.8 on 2 of 3; PLAY paired gain over the control retrain at least +30 per game on 2 of 3; deaths at most the control's plus 2 points on every seed; an arm passes only if all three hold. G2, in-agent on seeds 0-4, only after a G1 pass: paired gain over the recipe at least +50 per game with the Student-t interval above zero and deaths at most the recipe's plus 2 points. A failed gate closes the arm; it is reported as a remedy that failed.
- **C-q (descriptive analyses of stored artifacts, CPU only; each item is registered here before its script first runs on the data).**
  - **C-q.1 (Tiger equivalence; registered 2026-11-15, post hoc: the per-seed gaps it uses were read on 2026-11-03).** "The learned planner matches the optimal agent" is an equivalence claim, and an interval that contains zero does not support it by itself. For the ten per-seed paired gaps (learned planner minus optimal agent on the common analysis games), report the two one-sided tests (TOST) in two forms. First, the smallest margin at which equivalence holds at alpha = 0.05, which is the larger absolute bound of the 90% Student-t interval; no margin is chosen. Second, the TOST p-value at a reference margin of 1% of V*(b0). The same two numbers are reported for the decoder representation's gap and for decoder minus JEPA. Computed in `reports/2026-11-14_submission_audit/ledger.py` (`equivalence`; ids `tiger.*.gap_equivalence_margin`, `tiger.*.gap_tost_p_1pct`). No pass rule: the numbers are reported.
  - **C-q.2 (Tiger write-in intervention; registered 2026-11-15 before its first run).** Probes show that the belief is decodable from the latent; this tests whether the planner uses it. On each Tiger run (JEPA sweep `polyak`, seeds 0-9, best.pt, CPU), a belief-to-latent map mu is fitted on one set of random-policy histories; on a second set, 2048 latents z with exact belief b are moved towards the belief b' of another row: write-in z - mu(b) + mu(b'), replace mu(b'), a random direction of the same length, and no change. The greedy 50-simulation planner's action at the moved latent is compared with the exact optimal action at b' and at b. Reported per seed and as mean with Student-t 95% over seeds: on the rows whose optimal action differs between b and b' (the flip set), the share whose action becomes the optimal action of b' (flip rate) and the share that stays; agreement with the optimal action of b' over all rows; the forward probe's L1 error against b'. Analysis seed 20261115. No pass rule; the expectation stated in advance is that write-in and replace flip the action on most of the flip set and the random direction does not. Defined in `reports/2026-11-15_interpretability/tiger_intervention.py`. Amendment after a smoke on seed 0, before the ten-seed run: the random direction reached a pooled flip rate of 0.48 (write-in 0.83, replace 1.00) because a perturbed latent falls back to LISTEN, which many written beliefs ask for; the flip rate is therefore also reported on the subsets where the written belief asks for LISTEN, for a door, and for the other door than the source's (script header 2b). The door subsets are the discriminating ones.
  - **C-q.3 (search budget of the learned Tiger planner; registered 2026-11-19 before its first run).** On the exact Tiger model the search is consistent (the root error is at most gamma^{d_a} times the leaf error, and d_a only grows with the budget), but its decisions are not monotone in the budget when the leaves are biased (`reports/2026-11-16_code_audit/planner_budget.py`). This item measures the learned planner. On each canonical Tiger JEPA run (`runs/sweeps/polyak`, seeds 0-9, best.pt acting copies, CPU), the greedy learned planner (dirichlet_epsilon 0, the run's c_puct and observation_prune) plays 256 episodes x 100 steps on simulator seed 11 at N in {1, 2, 4, 8, 16, 32, 50, 64, 128} simulations, paired per episode with the exact optimal agent (argmax Q* of the certified solution). Quantity: per seed and N, the mean paired gap; per N, the mean over seeds with a Student-t 95% interval on the per-seed gaps, with the per-seed values beside it. Comparison, not a test: the exact model with zero, V3 and V8 leaves on the same episodes (`planner_budget.py`, part B). No pass rule; it is reported either way. Reading: if the interval contains 0 at every N >= 16, the learned value is accurate enough that the budget does not change Tiger play, and the Wumpus search-budget result (C-g) is attributed to the Wumpus model. Otherwise the learned planner shows the budget effect itself, and the paper says so. Defined in `reports/2026-11-16_code_audit/learned_planner_budget.py`.
  - **C-q.4 (the local form of the span bounds on the canonical Tiger; registered 2026-11-23 before its script first runs on the stored runs; post hoc: the form was derived on 2026-11-23, after every Tiger number had been read).** The error bounds reported for the canonical Tiger use one constant per value function, the largest half-span of its vectors. The local form (`src/ipomdp/interpretability/error_bounds.py`, section 5, with the proof) applies the same lemma to the vectors that are active at the exact and at the decoded belief: |V(b) - V(b')| <= max(span(alpha_b), span(alpha_b')) / 2 x ||b - b'||_1, and for the one-step regret (span(alpha) + span(beta)) / 2 x eps, with alpha active in Q(., a*) at the exact belief and beta active in Q(., a^) at the decoded one. It holds pair by pair and is never above the global form (tested). This item measures how much tighter it is on the canonical Tiger. On each JEPA run (`runs/sweeps/polyak`, seeds 0-9, best.pt acting copies), `analyze_beliefs` is called as the stored analysis called it (the certified solution at tolerance 0.01, the greedy planner at the run's budget, 512 episodes, 3,000 probe steps, analysis seed 12345), on the CPU. The stored reports were made on the GPU, where the same seed draws other episodes, so the global values of this item are a replication on other episodes and are printed beside the stored ones; nothing is written into a run directory. Quantities, per seed and as mean with Student-t 95% over the ten seeds, for the probes fitted and scored on the agent's own histories (MLP and linear): the mean L1 decoding error; the mean half-span of the active vector against L_V; the measured value error with its local and its global bound; the same for the one-step regret; the two policy forms (estimates, as before); the ratios local / global and local / measured. No pass rule: that the local form is not above the global one is a theorem, so the open quantity is the size of the gain. These numbers may be cited only as post hoc. Defined in `reports/2026-11-15_interpretability/tiger_local_bounds.py`. On the Interactive Tiger the same quantities will be registered with that experiment, before any of its runs.
  - **C-q.5 (is the MLP probe's fit stable?; registered 2026-11-23 before its first run at the registered size; post hoc: prompted by the read of C-q.4).** C-q.4 replicated the stored Tiger analysis on the CPU, where the same analysis seed draws other episodes. The linear probe's mean L1 replicated (0.0160 against a stored 0.0161). The MLP probe's replicated on eight of the ten runs (within 0.001) and not on two: seed 1 gave 0.0176 against a stored 0.0049, seed 5 gave 0.0267 against 0.0051, and on both the largest error was below the stored one, so the excess is spread over many histories. The MLP probe is fitted by 3,000 full-batch Adam steps at a constant rate of 3e-3 and the last iterate is taken (`fit_mlp_probe`); such a fit is not a descent method, and a last iterate inside a late rise of the loss would give this picture. This item tests that. On each JEPA run (`runs/sweeps/polyak`, seeds 0-9, best.pt acting copies, CPU) the two data sets of the agent's own histories are rebuilt as the analysis builds them (streams 3 and 4 of analysis seed 12345, 512 episodes) and stored. Faithfulness is checked first and a failure raises: `fit_mlp_probe` itself must reproduce the mean L1 that C-q.4 stored for the run (to 1e-9), and the script's traced loop must end at bit-identical parameters. Then, for fit seeds 0 (the analysis's) to 4, the same network, optimizer and step count are run under two schedules, constant (the code as it is) and cosine (the rate decays to zero over the 3,000 steps), with the training loss and the evaluation L1 traced every 25 steps. Quantities: per run and schedule, the last iterate's mean L1 for the five fit seeds; per run at fit seed 0 and the constant rate, the last iterate against the minimum, median and maximum of the trace over the last 1,000 steps; the number of the 50 fits per schedule above twice the median of the 50; over the ten runs, mean with Student-t 95% of the last iterate's L1 (fit seed 0, and the mean of the five fit seeds) under each schedule and of the paired difference cosine minus constant. No pass rule. Reading stated in advance: if the outlying fits are late rises (last iterate far above the trailing median) and the decayed rate removes them at every fit seed, the instability is the constant-rate last iterate; a change of the fit is then a decision about the instrument, because the stored reports were made with the constant rate, and this item changes nothing in `src/`. The chain was rehearsed on seed 1 at 8 episodes (808 rows, another data set; written outside the registered directory), which says nothing about the fit at the registered size. Defined in `reports/2026-11-15_interpretability/mlp_probe_fit_stability.py`.
  - **C-q.6 (does the lowest-training-loss iterate remove the MLP probe's late rises?; registered 2026-11-24 before its first run at the registered size; post hoc: prompted by the read of C-q.5 and by a look at its stored traces).** C-q.5 showed that the constant-rate last iterate of the MLP probe's fit lands on a late rise about once in 25 fits, and that a cosine rate removes the rises at a fifth more typical error. On C-q.5's traces (points every 25 steps) the point with the lowest training loss of the last 1,000 steps had no rise and a median error of 0.0042 against the last iterate's 0.0045. That rule was read off those traces, so this item tests it on fits that have not been run. On C-q.5's stored data sets of the ten JEPA runs, the fit's own loop (constant rate, `fit_mlp_probe`'s statements in their order) is run with fit seeds 5 to 9, keeping a copy of the iterate with the lowest training loss among the 1,001 iterates theta_2000 to theta_3000, each paired with its own loss; no evaluation history enters the choice. Fit seed 0 is run as well: its last iterate must equal C-q.5's stored value exactly (a failure raises), and its two known rises are an illustration, not a test. Quantities: per run, the mean L1 of the last and of the selected iterate at the five fresh fit seeds, and the selected step; the number of the 50 fresh fits above twice the median of the 50 fresh last iterates, under each rule; over the ten runs, mean with Student-t 95% of the per-run mean under each rule and of the paired difference selected minus last. No pass rule. Reading stated in advance: the rule is supported if no selected iterate is above twice that median and the paired difference is not above zero beyond its interval; if the 50 fresh last iterates hold no rise (probability about 0.12 at the rate C-q.5 saw), the item cannot show the removal on fresh fits and says so. It covers the Tiger probe on the agent's own histories only, and it changes nothing in `src/`: adopting a rule is a decision about the instrument. The chain was rehearsed on seed 1 at 60 steps (written outside the registered directory; the loss still falls there, so it says nothing about the rule), and both branches of the selection were exercised on synthetic data. Defined in `reports/2026-11-15_interpretability/mlp_probe_fit_selection.py`.
- **C-r (BA-POMCP, the Bayes-adaptive competitor, on Tiger and Interactive Tiger; registered 2026-11-20 before any run on data, on the user's rulings of 2026-11-19 and 2026-11-20).** BA-POMCP (Katt, Oliehoek & Amato 2017; `src/ipomdp/baselines/ba_pomcp.py`) is told the states, actions, observations and rewards, and starts from a Dirichlet prior over the dynamics. Our agent is told none of these, and every reported row says so. Settings fixed for every arm: expected-model steps, 1000 particles updated by rejection sampling, UCB with c = 20 (max R - min R), and the paper's planning horizon of 20, capped by the steps left in the window. Six arms, 10 independent runs each:
  - **Tiger `main`:** the paper's prior (T known at high confidence; LISTEN's growl 5 / 3; door observations uniform at high confidence), rollouts by the optimal policy of its posterior-mean MDP (EXPECTED_MDP), 1000 simulations.
  - **Tiger `matched`:** the same at 50 simulations, our planner's budget.
  - **Tiger `uniform`:** the paper's prior with uniformly random rollouts, 1000 simulations. This is the faithful default (POMCP's; the paper names none). On the continuing Tiger it listens where the optimum opens, even with the true model (module header 4b).
  - **Tiger `weak`:** LISTEN's growl at 1.25 / 0.75, the same 62.5% mean at a quarter of the confidence; EXPECTED_MDP, 1000. It replaces the uninformed arm first proposed: a symmetric growl prior can never learn which growl is which, because the BA-POMDP does not observe rewards (`tests/test_ba_pomcp.py`).
  - **I-Tiger `main`:** the 26-state fold of `conf/env/itiger.yaml`. BA-POMCP is told the interactive state space, which reveals the size and structure of j's controller. T is uninformed (Dirichlet 1 on every successor). i's own O is oriented: 8 (w O + (1 - w) / Z) counts per row with w = 5 / 14, which is exactly the paper's 5 / 3 on Tiger's LISTEN row. EXPECTED_MDP, 1000.
  - **I-Tiger `known_t`:** T known at high confidence as well, the same O, EXPECTED_MDP, 1000. This is an upper reference.
  - **One run:**
    - a learning period of 200 episodes x 100 steps on the run's own simulator stream: counts carry over, states reset to b0, and each episode's discounted return is a point of the learning curve;
    - then the 256 paired episodes of simulator seed 11 (the episodes of C-q.3). Each starts from a copy of the post-learning posterior and keeps updating within its episode;
    - they are played in chunks of 32. This is exact, because a simulator row's random numbers do not depend on the other rows' actions (tested).
  - **Quantities:**
    - per arm, the per-run paired gap, with Student-t 95% over the 10 runs and the per-run values beside it. On Tiger it is BA-POMCP minus the exact optimal agent. On I-Tiger it is BA-POMCP minus the reference agent of `runs/benchmarks/itiger_level1.pt` (exact filter plus the best lower-bound vector's action), with the arm's mean return set against the certified bracket [L, U];
    - on Tiger, BA-POMCP minus ours by Welch's two-sample t on the per-run against the per-seed gaps. Ours are C-q.3's per-seed gaps at N = 50: the same episodes and the same optimal agent;
    - learning curves against environment steps, beside our greedy evaluations during training (6,400 steps per collection);
    - on I-Tiger no learned agent exists until E1, so BA-POMCP stands against the bracket.
  - **No pass rule.** Each arm is read once and reported either way.
  - **Planning horizon:** an earlier note proposed planning to the end of the window (up to 100 steps). The protocol follows the paper's 20 instead, which also makes each rollout five times cheaper.
  - **Cost:** from a timing smoke on one CPU thread, the Tiger arms take 0.20 / 0.03 / 0.28 / 0.18 s per decision (main / matched / uniform / weak) and the I-Tiger arms 0.25 s. That is about 150 CPU-hours in all: Tiger on at most 8 workers, I-Tiger on 4 (about 1.3 GB each) at nice 19.
  - Defined in `reports/2026-10-19_eval_protocol/ba_pomcp_baseline.py`; outputs in `runs/baselines/ba_pomcp/`.

## Confirmatory results (read once, 2026-10-20 22:20)
Recipe: whole-buffer retrain + double value, training seeds 5-9, analysis seed 20261019, 512 games each (`runs/sweeps/wumpus_retrain_heldout/aggregate_analysis_heldout.json`, log `runs/wumpus/heldout_confirmatory_readout.log`). Intervals are hierarchical-bootstrap 95% (seeds, then games); gaps are per-game paired.

- **C-a.** Return **195.6 [155.3, 234.8]** per game (seeds: mean 195.6, t95 ± 58.1). Wins 28.6% [24.5, 32.6], deaths 2.0% [1.0, 3.1], unfinished 69.3% [65.5, 73.4].
  - Gap to random **+615.8 [570.6, 659.3]**.
  - Gap to the heuristic (risk 0.4; 579.9) **-384.2 [-426.5, -345.0]**.
  - Gap to the exact-belief planner (493.9 [476.3, 511.0]; wins 62.3%, deaths 1.6%, unfinished 36.1%) **-298.2 [-340.4, -259.3]**.
  - The exact planner itself is -86.0 [-97.6, -74.4] below the heuristic.
- **C-b.** MLP probes, mean over seeds (t95):
  - on random histories: cell 0.760 (± 0.057), heading 0.833 (± 0.076), pit hazard error 0.059 (± 0.005);
  - on agent histories: cell 0.882 (± 0.043), heading 0.917 (± 0.021).
- **C-c.** R2D2-lite 60.4 [36.4, 90.0] (wins 11.4%, deaths 0.0%, unfinished 88.6%). Recipe minus R2D2-lite, paired: **+135.2 [78.4, 190.7]**.
- The development numbers replicate: development 200 ± 44 vs held-out 196 ± 58. The R2D2-lite development figure was ~90.

## C-d result (read once, 2026-10-27 ~16:00)
Source: `runs/wumpus/heldout_cd_readout.log`, from `selection_readout.py` on `runs/sweeps/wumpus_retrain_heldout`, `analysis_heldout` against `analysis_heldout_final`. The sweep aggregate is `runs/sweeps/wumpus_retrain_heldout_final/aggregate_analysis_heldout_final.json`.

**Verdict: UNDECIDED.** Final retrain minus best.pt is **+7.5 [-21.6, +40.2]** per game (hierarchical bootstrap).
- Seeds 5 and 6 read the same networks and reproduced their games exactly (pairing check passed).
- Per seed, paired difference per game:

  | Seed | Final retrain − best.pt | Deaths (best.pt → final) | Unfinished (best.pt → final) |
  |---|---|---|---|
  | 5 | 0 | — | — |
  | 6 | 0 | — | — |
  | 7 | −35.7 ± 19.4 | 1.8 → 4.3% | — |
  | 8 | +61.8 ± 21.6 | 2.1 → 0.8% | — |
  | 9 | +11.4 ± 21.8 | 2.9 → 8.8% | 71.3 → 59.2% |

- Arms (hierarchical bootstrap):

  | Arm | Return | Deaths | Unfinished |
  |---|---|---|---|
  | best.pt | 195.6 [155.3, 234.8] | 2.0% | 69.3% |
  | Final retrain | 203.2 [171.7, 234.8] | 3.4% | 66.4% |

- On held-out seeds the effect is far smaller than on development seeds (+60.4), and deaths do not fall. The registered next step is seeds 10-14, to be registered here before launch.

## C-e results (each arm read once; `reports/2026-10-19_eval_protocol/ablation_readout.py`, logs `runs/week/readout_<arm>.log`)

### Arm 1, `abl_max` (read 2026-11-01 13:50): NO DETECTABLE CONTRIBUTION

| Training seed | Recipe | Max backup | Recipe - arm (per game, +- s.e.) |
|---|---|---|---|
| 5 | 246.5 | 120.9 | +125.6 +- 21.1 |
| 6 | 229.8 | 137.6 | +92.1 +- 19.3 |
| 7 | 204.6 | 203.6 | +1.0 +- 18.3 |
| 8 | 132.7 | 123.1 | +9.6 +- 16.7 |
| 9 | 164.6 | 225.3 | -60.7 +- 16.7 |

- Hierarchical bootstrap: recipe - arm +33.5 [-25.5, +94.4] per game; recipe 195.6 [155.3, 234.8] (reproduces C-a), arm 162.1 [122.6, 205.1]. Deaths 2.0% vs 2.1%; unfinished 69.3% vs 72.6%.
- Probes: the arm's latent matches the recipe's (linear, agent histories: cell 0.861 vs 0.861, heading 0.897 vs 0.877; random histories: cell 0.682 vs 0.699).
- Reading: the double value helps on two seeds by ~+100, is neutral on two and hurts on one; over five held-out seeds its contribution is not detectable. The paper reports it as such.

### Arm 2, `abl_noretrain` (read 2026-11-02 16:25): CONTRIBUTES

| Training seed | Recipe | No whole-buffer retrain | Recipe - arm (per game, +- s.e.) |
|---|---|---|---|
| 5 | 246.5 | 117.9 | +128.6 +- 22.6 |
| 6 | 229.8 | 211.2 | +18.6 +- 18.7 |
| 7 | 204.6 | 172.2 | +32.4 +- 13.6 |
| 8 | 132.7 | 136.4 | -3.7 +- 24.2 |
| 9 | 164.6 | 140.3 | +24.3 +- 17.9 |

- Hierarchical bootstrap: recipe - arm +40.0 [+2.6, +87.7] per game; arm 155.6 [123.9, 189.8]. Deaths 2.0% vs 2.8%; unfinished 69.3% vs 72.7%.
- Probes (random histories, MLP / linear): cell 0.759 / 0.699 vs 0.574 / 0.542, heading 0.833 / 0.781 vs 0.574 / 0.538; on the agent's own histories the gap is small (linear cell 0.861 vs 0.851, heading 0.877 vs 0.835).
- Reading: whole-buffer retraining contributes on held-out seeds, in return (positive on 4 of 5 seeds) and, most clearly, in the latent: about +18 points of cell and +25 points of heading accuracy on histories the agent did not choose.

### Arm 3, `abl_jepa` (read 2026-11-03 13:40): NO DETECTABLE CONTRIBUTION

| Training seed | Recipe | JEPA representation | Recipe - arm (per game, +- s.e.) |
|---|---|---|---|
| 5 | 246.5 | 68.1 | +178.5 +- 22.0 |
| 6 | 229.8 | 198.1 | +31.7 +- 19.4 |
| 7 | 204.6 | 148.5 | +56.1 +- 15.4 |
| 8 | 132.7 | 105.2 | +27.5 +- 19.5 |
| 9 | 164.6 | 192.5 | -27.9 +- 18.9 |

- Hierarchical bootstrap: recipe - arm +53.2 [-1.0, +120.0] per game; arm 142.5 [95.1, 187.2]. Deaths 2.0% vs 1.9%; unfinished 69.3% vs 76.2%.
- Probes (random histories, MLP / linear): cell 0.759 / 0.699 vs 0.595 / 0.564, heading 0.833 / 0.781 vs 0.641 / 0.616; on the agent's own histories the JEPA latent decodes slightly better (MLP cell 0.906 vs 0.882, heading 0.936 vs 0.917).
- Reading: the decoder is better on 4 of 5 seeds, but the interval just includes 0, so by the registered rule the choice of the decoder over JEPA has no detectable contribution to return. The JEPA latent is more specialised to the agent's own histories and generalises worse to histories it did not choose (~16 / ~19 points lower cell / heading accuracy). Three intervals were read (arms 1-3), no correction.

## C-f result (read 2026-11-03 22:55): DEFAULTS KEPT

Best-network return on the development games (training seeds 0 / 1 / 2, mean): defaults 65 / 70 / 126 (87.2); F1 batch 64 70 / 93 / 78 (80.3); F2 width 512 58 / 46 / 72 (58.6); F3 n 10 + batch 64 62 / 73 / 74 (69.8). The best arm is below the defaults, so none is chosen and C-f stops without a held-out run. The held-out model-free number stays C-c's 60.4 [36.4, 90.0]. Across the two screens, seven settings were tried and the R2D2 defaults remain the best on development.

## C-g result (read 2026-11-04 08:50; `reports/2026-10-19_eval_protocol/search_budget_readout.py`, log `runs/week/readout_search_budget.log`): SEARCH IS NOT THE LEVER (as predicted)

| Training seed | N = 10 | N = 50 (C-a) | N = 100 | deaths N50 -> N100 | unfinished N50 -> N100 |
|---|---|---|---|---|---|
| 5 | 227.5 | 246.5 | 235.0 | 0.2 -> 0.6% | 66.6 -> 67.6% |
| 6 | 206.5 | 229.8 | 203.3 | 3.1 -> 5.7% | 64.3 -> 62.3% |
| 7 | 200.0 | 204.6 | 198.8 | 1.8 -> 4.5% | 68.8 -> 63.9% |
| 8 | 122.0 | 132.7 | 138.0 | 2.1 -> 2.9% | 75.8 -> 74.2% |
| 9 | 110.1 | 164.6 | 162.6 | 2.9 -> 3.7% | 71.3 -> 70.5% |

- Hierarchical bootstrap: N100 - N50 -8.1 [-23.2, +6.2] per game (upper end below the +84 bar); N10 - N50 -22.4 [-43.2, -3.4]; N10 173.2, N100 187.5.
- Descriptive, not registered: at 100 simulations the deaths rise on 5 of 5 seeds (mean 2.0% -> 3.5%) while wins rise and unfinished games fall slightly. Deeper search over the learned model finds more of the steps whose risk the model under-rates, which matches the optimizer's-curse reading of the remedies.
- Reading: the planner's budget is not what limits it. Doubling the search does not recover any of the +168 that the exact one-step model gives at 50 simulations.

## C-h result (development screen, read 2026-11-11 13:30; logs `runs/wumpus/s0_screen/`, phantom `phantom_readout.log`): NO ARM PASSES, DEFAULTS KEPT

Arm minus control per game, paired on the 512 analysis games of training seeds 0 / 1 / 2 (± one standard error); control plays 270.0 / 214.7 / 194.3 with deaths 1.0 / 1.8 / 7.2%.

| Arm | Seed 0 | Seed 1 | Seed 2 | Mean | Deaths (seeds 0 / 1 / 2) | PHANTOM (0 / 1 / 2) | Verdict |
|---|---|---|---|---|---|---|---|
| CB | −115.5 ± 18.8 | +19.5 ± 14.1 | +42.1 ± 17.5 | −17.9 | 17.6 / 2.3 / 0.0% | 18.7 / 19.8 / 18.7% | FAIL (play, deaths, phantom) |
| CBP | +5.7 ± 14.0 | +41.9 ± 15.2 | +9.8 ± 18.1 | +19.2 | 1.8 / 2.5 / 0.6% | 18.9 / 20.6 / 20.1% | FAIL (mean below +20, phantom) |
| SLOT | +6.3 ± 12.9 | +15.8 ± 16.1 | +22.8 ± 18.4 | +15.0 | 1.8 / 3.5 / 2.3% | — | FAIL (mean below +20) |
| VCLIP | −0.6 ± 13.5 | −133.3 ± 17.1 | +47.7 ± 16.6 | −28.7 | 1.8 / 1.6 / 1.6% | — | FAIL (play) |

- The control's PHANTOM share is 18.4 / 20.3 / 19.9%. The bounded codec (range ±1110 instead of ±100100) leaves it unchanged: the imagined children valued above 999 lie inside the reachable range, so the share measures over-estimation of reachable values, not leakage onto unreachable bins.
- SLOT does not decorrelate the value pair: corr(V_A, V_B) is 0.9996 / 0.9999 / 0.9994 (control 0.9999 / 0.9999 / 0.9998).
- Reported, no verdict: the joint glitter ratio at the declined turned latents (control 0.45 / 0.39 / 0.33) is 0.38 / 0.33 / 0.38 (CB), 0.37 / 0.33 / 0.40 (CBP), 0.43 / 0.31 / 0.33 (SLOT) and 0.43 / 0.25 / 0.38 (VCLIP). The MLP "cell ahead unvisited" accuracy there (control 0.693 / 0.672 / 0.539) rises on 11 of the 12 arm-seeds (0.52–0.84).
- Consequence: no Stage-0 fix enters R2. CBP and SLOT are positive on 3 of 3 seeds but below the registered mean; this is recorded and not re-tested.

## C-i result (development smoke, read 2026-11-11 14:50; logs `runs/wumpus/sc_smoke/seed<s>.log`): PASS BY THE REGISTERED RULE, WITH A CAVEAT

| Seed | Rows (declined) | span x JSD AUC | JSD AUC | span AUC | Within non-wall rows: span x JSD / JSD / span | Within unvisited-target rows: span x JSD / JSD / span |
|---|---|---|---|---|---|---|
| 0 | 11752 (599) | **0.865** | 0.876 | 0.806 | 0.637 / 0.703 / 0.448 | 0.554 / 0.680 / 0.282 |
| 1 | 17566 (993) | **0.664** | 0.610 | 0.593 | 0.538 / 0.583 / 0.436 | 0.471 / 0.589 / 0.330 |
| 2 | 13492 (714) | **0.816** | 0.786 | 0.801 | 0.538 / 0.558 / 0.471 | 0.385 / 0.491 / 0.289 |

- Registered verdict: AUC >= 0.65 on 3 of 3 seeds, so C-i PASSES.
- Caveat (the within-row lines were added after seed 0's first read and are reported, not registered): most of the separation is "the turn faces a wall or not". A wall-facing turn makes FORWARD's outcome certain, so its span and JSD are 0. Among the non-wall rows the score is near chance (0.54-0.64). Among rows whose target cell is unvisited it is at or below chance (0.39-0.55): the value span there points the wrong way (0.28-0.33), and JSD alone stays weakly informative (0.49-0.68).
- Consequence: augmentation C is no longer gated out. Its build is deferred until C-j is read, and the caveat goes with any claim made about it.

## C-j result (development screen, read 2026-11-12 01:00; logs `runs/wumpus/b_screen/`): NO ARM PASSES, STAGE B CLOSED

Arm minus control per game, paired on the 512 analysis games of training seeds 0 / 1 / 2 (± one standard error). The control plays 270.0 / 214.7 / 194.3, with deaths 1.0 / 1.8 / 7.2%, a joint glitter ratio at the declined turned latents of 0.45 / 0.39 / 0.33, and an MLP "cell ahead unvisited" accuracy there of 0.693 / 0.672 / 0.539.

| Arm | Play: seed 0 / 1 / 2 | Deaths | Glitter ratio | Unvisited acc. | Verdict |
|---|---|---|---|---|---|
| TR | −487.6 ± 30.7 / −408.1 ± 28.1 / −425.0 ± 32.3 | 55.1 / 53.9 / 56.1% | 0.20 / 0.23 / 0.25 | 0.279 / 0.174 / 0.216 | FAIL (both) |
| CH | −22.8 ± 15.8 / +41.0 ± 14.4 / −28.9 ± 20.3 | 3.5 / 1.8 / 18.4% | 0.30 / 0.42 / 0.46 | 0.714 / 0.714 / 0.692 | FAIL (both) |
| TRCE | −548.1 ± 33.3 / −392.8 ± 28.5 / −454.2 ± 31.0 | 61.7 / 53.3 / 59.6% | 0.21 / 0.27 / 0.29 | 0.322 / 0.398 / 0.269 | FAIL (both) |

- CH meets the unvisited bar only on seed 2, never meets the glitter bar, and raises deaths by 11 points on seed 2.
- The Transformer arms do not fail through the search path. Token-state latents equal the filter's unroll to 1e-6 on stored games, and the acting and online copies agree. They fail because, with the recipe's optimiser, the history filter learns a weaker latent in 6400 updates:
  - MLP cell / heading 0.69 / 0.65–0.69, against the control's 0.88 / 0.93, 0.87 / 0.91 and 0.82 / 0.86;
  - the head flags certain death 0.16–0.34 of the time (control 0.39–0.68), so the planner steps into risks it does not see (deaths above 50%).
  - Training also takes about 1.4 times as long (about 1800 s vs 1356 s for CH on seed 0).
- Consequence: no Stage-B arm enters R2. This is an architecture A/B negative under the recipe's optimisation settings. The Transformer received no tuning of its own (learning rate, warm-up, depth), and that limits the claim.

## C-k result (development smoke, read 2026-11-13 23:00; logs `runs/wumpus/failure_head/<set>_seed<s>.log`): FAIL, AUGMENTATION A CLOSED

Failure head / exact danger at the set's fatal chosen FORWARDs on the 512 analysis games. The stored column is the observation and reward heads' max(P_O, P_R), as in `risk_localise.py`. Calibration is mean(head) / mean(danger) over risky live states.

| Set | Seed | Fatal FORWARDs | Exact danger (median) | Head / danger: median (mean) | Stored: median | Calibration |
|---|---|---|---|---|---|---|
| union | 0 | 84 | 0.556 | **0.000** (0.022) | 0.026 | 0.861 |
| union | 1 | 27 | 1.000 | **0.000** (0.000) | 0.033 | 0.709 |
| union | 2 | 80 | 1.000 | **0.000** (0.013) | 0.014 | 0.638 |
| control | 0 / 1 / 2 | 23 / 16 / 28 | 0.522 / 1.000 / 0.950 | 0.000 / 0.000 / 0.000 | 0.122 / 0.012 / 0.020 | 0.980 / 1.184 / 0.942 |
| random restore | 0 / 1 / 2 | 0 / 22 / 22 | — / 0.458 / 1.000 | — / 0.002 / 0.009 | — / 0.034 / 0.023 | 0.802 / 0.924 / 0.470 |

- Verdict: the median is below 0.25 on 3 of 3 union seeds, so CALIBRATION fails. The inflation guard holds: at most 1.5 on every seed.
- Reading: averaged over risky states the dedicated head is about calibrated (0.64-1.18), but at the steps the planner dies on it rates the risk at about 0. On several seeds the exact danger there is 1: the cell ahead is provably deadly, and the latent does not carry that. The dedicated head is no better than the stored heads. The AUC for fatal vs survived among chosen risky FORWARDs is 0.07-0.78 (stored 0.09-0.81).
- Consequence: a constraint on any head read from this latent cannot remove these deaths, so the chance-constrained search is not built and CC-play is not run.

## C-l result (diagnostic, read 2026-11-14 00:30; logs `runs/wumpus/deadly_probe/<set>_seed<s>.log`): MIXED

Probe recall (prediction at least 0.5) at live analysis-game states with a provably deadly cell ahead. "Missed" means the observation head's max(P_O, P_R) was below 0.5; "stepped" means the agent moved FORWARD into the cell and died. The floor is a probe on a re-initialised filter.

| Set | Seed | Deadly states (flagged / missed / stepped) | Recall: missed | Recall: flagged | Recall: stepped | Floor: missed |
|---|---|---|---|---|---|---|
| control | 0 | 353 / 113 / 9 | **0.717** | 0.963 | 0.778 | 0.106 |
| control | 1 | 492 / 486 / 13 | **0.344** | 0.904 | 0.308 | 0.101 |
| control | 2 | 129 / 259 / 14 | **0.517** | 0.791 | 0.500 | 0.344 |
| union | 0 / 1 / 2 | 940 / 345 / 38, 226 / 220 / 20, 710 / 298 / 50 | 0.470 / 0.709 / 0.574 | 0.864 / 0.947 / 0.979 | 0.289 / 0.000 / 0.380 | 0.122 / 0.050 / 0.087 |

- Verdict: MIXED. On the control set seed 0 reads IN-LATENT, seed 1 NOT-IN-LATENT and seed 2 MIXED, so no reading holds on 2 of 3 seeds.
- On held-out buffer states the probe's recall at provably deadly cells is 0.72-0.88 (floor 0.24-0.30). The latent carries "deadly ahead" on most of its own states, partly at the states the head misses, and least at the states the agent steps into: recall there is 0.00-0.78, and at most 0.38 on 4 of 6 set-seeds.
- At every provably deadly state the head missed, on all six set-seeds, the hunter's current percept already warned: a breeze for a pit ahead, a stench for a wumpus. The certainty comes from combining that warning with which neighbours are known safe (visited). The deaths therefore sit on the visited-tracking deficit found by the earlier localisation (Step D), not on missing percept information.
- Consequence (no rule-bound action, as the reading is MIXED): both levers remain. One is the head's training signal: P(GAME_OVER | z, FORWARD) is supervised only where FORWARD was taken, which is rare at provably deadly cells. The other is the representation's tracking of visited cells.

## C-m result (development screen, read 2026-11-14 20:15; logs `runs/wumpus/ego_screen/`, readout `readout_cm.log`): NO ARM PASSES, EGO MAP CLOSED

Arm minus control per game, paired on the 512 analysis games of training seeds 0 / 1 / 2 (± one standard error). MISS RATE is the share of live states with a provably deadly cell ahead (exact danger at least 0.999) where max(P_O, P_R) < 0.5. The control plays 270.0 / 214.7 / 194.3, with deaths 1.0 / 1.8 / 7.2% and MISS RATE 0.248 / 0.398 / 0.185.

| Arm | Play: seed 0 / 1 / 2 | Mean, Student-t 95% (nested; crossed) | Deaths | MISS RATE | Steps into deadly cells (control 3 / 4 / 1) | Verdict |
|---|---|---|---|---|---|---|
| EGO1 | −45.0 ± 19.1 / −4.2 ± 14.8 / +14.2 ± 19.8 | −11.7 [−87.0, +63.6] ([−47.7, +22.2]; [−51.0, +26.5]) | 2.7 / 3.3 / 3.5% | 0.168 / 0.057 / 0.180 | 10 / 16 / 18 | FAIL (MECHANISM 1 of 3, PLAY 1 of 3) |
| EGO01 | −102.7 ± 19.4 / +37.4 ± 16.6 / +34.8 ± 19.3 | −10.2 [−209.2, +188.9] ([−99.0, +50.7]; [−94.2, +54.0]) | 2.5 / 2.7 / 2.3% | 0.283 / 0.230 / 0.321 | 8 / 7 / 10 | FAIL (MECHANISM 1 of 3; PLAY 2 of 3) |

- Neither arm passes MECHANISM, so neither passes; EGO01 passes PLAY alone. Deaths stay within control + 2 points on every seed.
- The ego-map readout helps the head recognise provably deadly cells strongly on one seed (EGO1 seed 1: MISS RATE 0.398 to 0.057) and not on the others.
- Both arms face many more provably deadly states than the control (EGO1 980 / 1052 / 1078 against 572 / 432 / 730) and step into them more often (EGO1 10 / 16 / 18 against 3 / 4 / 1). The latent that tracks its own position makes the agent go forward more, and the head's remaining misses then cost more.
- Consequence: the ego map does not enter R2 and is closed.

## C-n result (development screen, read 2026-11-15 06:10; `runs/baselines/readout_c_n.log`): DEFAULTS KEPT

Best-network return on the 512 development games (analysis seed 12345). Paired difference from the defaults (`r2d2_lr1e-4_n5`: 65.4 / 69.8 / 126.4 on seeds 0 / 1 / 2).

| Arm | Return (seeds 0 / 1 / 2, mean) | Arm − defaults per seed | Mean (se); Student-t 95% | Nested; crossed |
|---|---|---|---|---|
| N1 `exploration=anneal` | 64.1 / 72.8 / 92.5 (76.5) | −1.3 / +3.0 / −33.9 | −10.7 (11.6); [−60.8, +39.3] | [−34.6, +10.6]; [−37.9, +14.2] |
| N2 `value_scale=linear` | 72.6 / 53.7 / 46.1 (57.5) | +7.2 / −16.1 / −80.3 | −29.7 (26.2); [−142.4, +82.9] | [−78.5, +7.0]; [−83.4, +6.7] |
| N3 both | 65.0 / 71.6 / 60.6 (65.7) | −0.4 / +1.7 / −65.8 | −21.5 (22.2); [−116.8, +73.8] | [−62.6, +1.6]; [−63.0, +1.6] |

- Every arm has 0% deaths and 10-14% wins. The candidate is N1 (highest mean). Its paired difference, −10.7, is not above one standard error (11.6), so it is NOT CHOSEN. The defaults stay and no held-out run follows.
- Across the three screens (C-c, C-f, C-n), ten settings have been tried; the R2D2 defaults remain the best on development.
- Operational note: r2d2_n3 seed 1 died twice at collections 15-20 with a WSL GPU driver fault ("CUDA error: unknown error"). The registered run is the identical-condition rerun with CUDA_LAUNCH_BLOCKING=1, which uses the same kernels and numerics (`runs/baselines/screen_chain.log`).

## C-p result, G0 (development, analysis seed 12345, read once 2026-11-16 08:05; `runs/wumpus/g0/readout.log`, `summary.json`): HEAD NOT REPAIRED, MODEL STILL BINDS
Networks: best.pt of the three stored oracle + double runs (exact-belief targets in training, no whole-buffer retrain). Per seed 0 / 1 / 2, 512 games:
- Joint glitter ratio at the declined turned latents: 0.48 / 0.43 / 0.36 (the recipe's: 0.22-0.44; the registered bar for HEAD REPAIRED was 0.8 on 2 of 3). Probe gold calibration 0.82 / 0.66 / 0.71; "cell ahead unvisited" accuracy 0.89 / 0.84 / 0.86.
- Flip shares (the turn-then-step macro beats the chosen action): learned model 0.10 / 0.11 / 0.08; exact P 0.51 / 0.37 / 0.57; exact P and R 0.66 / 0.60 / 0.81.
- Play, learned planner: 215.7 / 231.4 / 223.1 per game, deaths 2.3 / 1.6 / 1.0%, unfinished 68.2 / 67.8 / 69.5%.
- Play, exact one-step model (EPR): 413.8 / 459.8 / 399.5, deaths 0.0%, unfinished 51.0 / 45.9 / 51.6%. EPR minus learned, paired per game: +198.0 / +228.4 / +176.4 (game-level s.e. 18-20).
- Readings: HEAD REPAIRED NO; MODEL STILL BINDS YES (all three seeds above +50). Registered consequence: only G1 arm B is built (a state-factored observation model in the backup); arm A (belief supervision with the full recipe) is not run.
- What it says: a latent supervised with the exact belief (cell, heading, per-cell pit, wumpus and gold posteriors, danger ahead) leaves the glitter under-prediction at declined turn-then-step states in the same range as the unsupervised recipe, and the exact one-step model is still worth about +200 per game on it. Belief-sufficiency of the latent, as trained here, does not close the policy-induced coverage hole in (latent, action) space. This is a negative result for belief supervision as a remedy and is to be reported as such; the 3 seeds are development seeds and all numbers are exploratory.

## C-q results (descriptive, development artifacts)
- **C-q.1 (Tiger equivalence, computed 2026-11-15).** The learned planner is equivalent to the optimal agent within 0.063 per episode at alpha = 0.05 (0.33% of V*(b0)); TOST p = 3.3e-5 at a margin of 1% of V*(b0). Decoder: margin 0.061, p = 1.1e-4. Decoder minus JEPA: margin 0.104, p = 2.3e-3. Ledger ids `tiger.*.gap_equivalence_margin`, `tiger.*.gap_tost_p_1pct`.
- **C-q.2 (Tiger write-in intervention, read 2026-11-15 19:10; log `runs/interpretability/tiger_intervention/polyak.log`, ten seeds, 2048 rows each, flip set 281 rows: 138 where the written belief asks for LISTEN, 143 for a door).** Mean over seeds with Student-t 95% [min, max]:
  - replace (the belief manifold alone): the planner takes the optimal action of the written belief on 100.0% of all rows on every seed; flip rate 100.0%.
  - write-in (belief component moved, residual kept): flip rate 87.4 +- 2.9% [82.9, 94.7]; to a door 75.9 +- 5.7% [66.4, 90.2]; to LISTEN 99.4%; agreement with the written belief's optimal action 97.9 +- 0.4%.
  - random direction of the same length: to a door 0.07 +- 0.16% [0.0, 0.7]; to LISTEN 94.3% (a perturbed latent falls back to LISTEN); pooled 46.3%.
  - no change: flip rate 0.5%.
  - Reading: the planner's action is determined by the belief the latent encodes (replace 100%); a random move of the same size never opens a door. The residual of the latent beyond the belief matters: keeping it leaves about a quarter of the door flips undone. Ledger ids `tiger.intervention.*`.
- **C-q.3 (search budget of the learned Tiger planner, read once 2026-11-20 00:30; log `reports/2026-11-16_code_audit/learned_planner_budget.log`, run at edb2483; ten JEPA seeds of `runs/sweeps/polyak`, 256 paired episodes each; the optimal agent scores 22.375 on these episodes).** Mean paired gap (learned minus optimal) per episode over the ten seeds, Student-t 95% on the per-seed gaps:

  | N | 1 | 2 | 4 | 8 | 16 | 32 | 50 | 64 | 128 |
  |---|---|---|---|---|---|---|---|---|---|
  | mean | -0.127 | -0.147 | -0.275 | -0.182 | -0.168 | -0.119 | -0.088 | -0.067 | -0.076 |
  | lower | -0.327 | -0.306 | -0.720 | -0.425 | -0.377 | -0.249 | -0.237 | -0.183 | -0.198 |
  | upper | +0.073 | +0.012 | +0.169 | +0.061 | +0.042 | +0.010 | +0.062 | +0.049 | +0.046 |

  - Per-seed values are in the log. Seeds 1 and 6 carry most of the gap: -0.33 and -0.32 at N = 128. Seed 9 is +0.198 at N = 1 and at N >= 50.
  - Reading, by the registered rule: the interval contains 0 at every N >= 16, in fact at every N. The learned value is accurate enough that the search budget does not change Tiger play, and the Wumpus search-budget result (C-g) is attributed to the Wumpus model.
  - Comparison, on the exact model with the same paired episodes (`reports/2026-11-16_code_audit/planner_budget.log`, minus optimal per episode, +- s.e.):
    - zero leaves: -5.44 at N = 2 and -5.93 at N = 4, back to -0.65 at N = 64, then -4.26 +- 2.81 at N = 128;
    - V3 leaves: optimal at N <= 2, -4.13 at N = 4, -0.01 at N = 64, -0.23 at N = 128;
    - V8 leaves: optimal at every N except N = 8 (-0.75 +- 1.10).
    - So with biased leaves the return oscillates with the budget (the horizon effect) instead of improving monotonically. The learned leaves behave like accurate leaves, not like zero leaves.
- **C-q.4 (the local form of the span bounds on the canonical Tiger, read once 2026-11-23 20:22; POST HOC; log `reports/2026-11-15_interpretability/tiger_local_bounds.log`, analysed at 440adad; ten JEPA seeds of `runs/sweeps/polyak`, CPU, 512 episodes, probes fitted and scored on the agent's own histories).** Mean over the ten seeds, Student-t 95%; L_V = L_Q = 55.00, the active vector's half-span is 18.72 on average:

  | quantity | MLP probe | linear probe |
  |---|---|---|
  | mean L1 decoding error | 0.0078 [0.0022, 0.0135] | 0.0160 [0.0147, 0.0172] |
  | value error, measured | 0.0559 [0.0372, 0.0745] | 0.1422 [0.1293, 0.1551] |
  | value error, local bound | 0.0572 [0.0382, 0.0761] | 0.1504 [0.1370, 0.1638] |
  | value error, global bound | 0.4305 [0.1210, 0.7401] | 0.8782 [0.8097, 0.9466] |
  | value error, local / global | 0.173 [0.139, 0.207] | 0.172 [0.161, 0.182] |
  | value error, local bound / measured | 1.02 [1.02, 1.03] | 1.06 [1.05, 1.07] |
  | one-step regret, measured | 0.0012 [0.0008, 0.0015] | 0.0085 [0.0070, 0.0099] |
  | one-step regret, local bound | 0.1113 [0.0741, 0.1484] | 0.2805 [0.2550, 0.3059] |
  | one-step regret, global bound | 0.8611 [0.2420, 1.4802] | 1.7564 [1.6195, 1.8933] |
  | one-step regret, local / global | 0.168 [0.135, 0.201] | 0.160 [0.150, 0.170] |
  | one-step regret, local bound / measured | 96 [83, 110] | 34 [29, 40] |
  | policy form, local (estimate) | 2.23 [1.48, 2.97] | 5.61 [5.10, 6.12] |
  | policy form, global (estimate) | 17.22 [4.84, 29.60] | 35.13 [32.39, 37.87] |

  - The size of the gain: the local form is about a sixth of the global one, for both probes and both quantities (per seed between 0.08 and 0.22). The policy form falls from 17.2 to 2.2 with the MLP probe and from 35.1 to 5.6 with the linear probe, against V*(b0) = 19.37: the global form with the linear probe is above the value itself.
  - The local value bound is within 2% (MLP) and 6% (linear) of the measured error. **This is a property of a two-state problem and is not evidence that the form is tight in general:** with two states |alpha.(b - b')| equals span(alpha)/2 x ||b - b'||_1 exactly whenever one vector is active at both beliefs, so the bound is slack only where the active vector changes. Nothing of the kind holds on the 14- and 26-state folds.
  - The one-step regret bound stays loose: 96 times (MLP) and 34 times (linear) the measured regret. The regret is zero wherever the decoded belief does not change the action (the decoded belief asks for a suboptimal action on 0.11 to 0.34% of the steps with the MLP probe and on 0.8 to 1.8% with the linear one), and a norm bound cannot see that.
  - The largest measured value error and regret are under the largest local bounds on 10 of 10 seeds for both probes (largest value error 1.6 to 3.8, largest local bound 3.4 to 12.3, largest global bound 11.6 to 33.3).
  - **Replication of the stored global form (the registered side comparison).** The linear probe replicates: mean L1 0.0160 against a stored 0.0161, global value bound 0.878 against 0.887. The MLP probe replicates on eight seeds (mean L1 within 0.001) and not on seeds 1 and 5: 0.0176 and 0.0267 against stored 0.0049 and 0.0051. Over the ten seeds the MLP probe's mean L1 is therefore 0.0078 [0.0022, 0.0135] here against a stored 0.0044 [0.0040, 0.0048], and its global value bound 0.43 [0.12, 0.74] against a stored 0.24 [0.22, 0.26]. On both seeds the largest L1 error is below the stored one, so the excess is spread over many histories. Whether the cause is the fit (a last iterate of full-batch Adam at a constant rate) is C-q.5's question; until it is read, the MLP rows of this table carry that caveat and the linear rows do not.
  - Cost: 41.9 min per seed on one CPU thread, beside other jobs.
- **C-q.5 (the stability of the MLP probe's fit, read once 2026-11-24 05:25; POST HOC; log `reports/2026-11-15_interpretability/mlp_probe_fit_stability.log`, played at 28c1df1 to 030d617; the ten JEPA runs of `runs/sweeps/polyak`, the agent's own histories of C-q.4, fit seeds 0-4 under two schedules, 100 fits; no pass rule).** THE CONSTANT-RATE LAST ITERATE IS UNSTABLE; THE READING STATED IN ADVANCE HOLDS, AND THE DECAYED RATE HAS A COST THAT WAS NOT ANTICIPATED.
  - Faithfulness: on every run `fit_mlp_probe` itself reproduced C-q.4's mean L1, and the traced loop ended at its parameters.
  - **Constant rate (the code as it is).** The median of the 50 fits is 0.0045. Two of the 50 are above twice the median: 0.0176 (run 1) and 0.0267 (run 5), both at fit seed 0, the analysis's own. These are the two values that C-q.4 did not replicate. The next largest are 0.0077, 0.0072 and 0.0068.
  - **They are late rises.** At fit seed 0 the last iterate is 3.55 (run 1) and 5.19 (run 5) times the median of its own last 1,000 steps and equals that tail's maximum; on the other eight runs it is 0.89 to 1.13 times the median. Every run's tail holds such rises: its maximum is 1.6 to 5.3 times its median. The training loss varies by 0.0001 to 0.0007 within a tail (of 0.445).
  - **Cosine rate.** Median 0.0054, none of the 50 above twice the median, largest 0.0069. The standard deviation over the five fit seeds, averaged over the runs, falls from 0.0022 to 0.0004.
  - Over the ten runs, mean [Student-t 95%]:

  | mean L1 of the last iterate | constant | cosine | cosine minus constant (paired per run) |
  |---|---|---|---|
  | fit seed 0 (the analysis's) | 0.0078 [0.0022, 0.0135] | 0.0053 [0.0048, 0.0059] | -0.0025 [-0.0080, 0.0030] |
  | mean of the five fit seeds | 0.0053 [0.0041, 0.0064] | 0.0053 [0.0050, 0.0056] | 0.0000 [-0.0011, 0.0011] |

  - **Reading.**
    - The instability is the constant-rate last iterate, as stated in advance: both outlying fits are late rises, and the decayed rate removes them at every fit seed.
    - Not anticipated: the decayed rate is not free. Its typical error is a fifth higher (median 0.0054 against 0.0045), because it trains less in the same 3,000 steps. The two schedules have the same mean; the cosine has the narrower interval and the worse typical fit.
    - The stored reports (mean L1 0.0044 [0.0040, 0.0048], behind the ledger's `tiger.bounds.*`) sit at the constant rate's typical value: none of their ten fits landed on a rise. They are typical fits of an instrument with a heavy right tail, about one fit in 25 here.
  - **Post hoc exploration of the stored traces (2026-11-24; no new fit; not registered).** Within a fit's last 1,000 steps the training loss and the evaluation error rise together: Spearman 0.93 at the median of the 50 constant-rate fits, at least 0.79. Of the 2,000 traced points 84 (4.2%) are above twice their fit's tail median. The traced point with the lowest TRAINING loss of the last 1,000 steps has a median of 0.0042 over the 50 fits and a largest value of 0.0053; at fit seed 0 over the ten runs it gives 0.0042 [0.0039, 0.0046]. So an iterate chosen by the training loss alone, without evaluation data, would remove the rises without the cosine's cost. The rule was read off these traces (points every 25 steps, the loss one step before the evaluated parameters), so it is a candidate to test on other fits, not a result.
  - **Not covered:** the probe fitted on random histories (the decoded-belief agent's), the Wumpus MLP probe (`fit_mlp_wumpus_probe`, the same recipe), and the probes of the Interactive Tiger analysis (26 and 14 states).
  - Nothing in `src/` is changed by this item. Whether the fit changes is a decision about the instrument.
  - Cost: 65.9 min per run on one CPU thread, of which 15.9 min for the two data sets, beside other jobs.

## C-r result, Tiger (read once 2026-11-21 10:01; `reports/2026-10-19_eval_protocol/ba_pomcp_tiger_readout.log`; campaign at 5609947, log beside it): OURS AHEAD OF EVERY BA-POMCP ARM

On the 256 paired episodes of seed 11 the optimal agent scores 22.375, as in C-q.3. Gap to the optimal agent per episode, Student-t 95% over 10 runs (ours: the ten JEPA seeds at N = 50, from C-q.3):

| Arm | Return | Gap to optimal | BA-POMCP minus ours (Welch) |
|---|---|---|---|
| ours | 22.29 | -0.087 [-0.237, +0.062] | |
| main (paper prior, EXPECTED_MDP, 1000 simulations) | 15.62 | -6.756 [-8.408, -5.104] | -6.669 [-8.323, -5.014] |
| matched (50 simulations) | -18.61 | -40.987 [-42.598, -39.375] | -40.899 [-42.513, -39.285] |
| uniform rollouts (1000) | 8.48 | -13.898 [-14.929, -12.866] | -13.810 [-14.846, -12.774] |
| weak prior (1.25 / 0.75) | -38.84 | -61.215 [-183.484, +61.054] | -61.128 [-183.397, +61.141] |

- Per-run values are in the log. In the weak arm, nine runs lie at -4.4 to -11.0 and run 9 at -547.6.
- Learning curves (mean over runs, per 2,000 environment steps):
  - main: flat, between 5.4 and 18.6 from the first 2,000 steps on (13.14 in the first block, 13.32 in the last);
  - ours, the greedy evaluations during training: -19.88 at 32,000 steps, 17.52 at 96,000, 20.27 at 160,000.
- Reading:
  - Our agent ends ahead of every BA-POMCP arm. Told the states, the rewards and the transitions, BA-POMCP plays its final policy from its first 20 learning episodes on; our agent needs about 100,000 steps to learn a latent model from scratch.
  - BA-POMCP needs its full budget of 1000 simulations: at our 50 it loses 41 per episode, while ours is within 0.1 of the optimum.
- **Post hoc diagnostics** (written after the readout, not registered):
  1. **The models are learned.** In 39 of the 40 runs the post-learning posterior mean of the listening accuracy is 0.84 to 0.86 (truth 0.85), identical across particles. Weak run 9 converged to the MIRRORED model (0.148, every particle): it believes each growl points to the wrong side and opens the wrong door again and again. With rewards unobserved, a weakly oriented prior can tip into the mirror image and lock in; our agent observes rewards. Read from `runs/baselines/ba_pomcp/tiger/<arm>/run<r>_agent.pkl`.
  2. **The deficit is the search, not the learning.** BA-POMCP with the TRUE model at point-mass confidence, everything else as main, loses -6.224 +- 3.267 per episode on the same 256 episodes (1.96 s.e. over episodes; one planner), as much as main with its learned model (-6.756). It opens a wrong door in 7.0% of episodes (gap < -50), and 1.2% of its episodes match the optimal agent's return (`reports/2026-10-19_eval_protocol/ba_pomcp_point_mass_check.py` and `.log`).
     - Its root decisions are optimal at each of the six beliefs the optimal policy reaches. The loss comes from search noise over whole episodes: a UCB constant of 20 (max R - min R) explores almost uniformly, and a Monte Carlo average sometimes opens a door on the wrong side.
     - Our planner (expectimax over exact observation branching with a learned value leaf) is near the optimum at 50 simulations (C-q.3).

## C-r result, I-Tiger (read once 2026-11-23 00:16; `reports/2026-10-19_eval_protocol/ba_pomcp_itiger_readout.log`; campaign at 5609947, log beside it): BOTH BA-POMCP ARMS ABOUT 30 PER EPISODE BELOW THE REFERENCE AGENT

The benchmark of the 26-state fold (`runs/benchmarks/itiger_level1.pt`, fold digest 957782ffdac6d103): the certified bracket of V*(b0) is [16.5510, 19.6692]. Its lower end is HSVI's after 1,000 trials (49,909 lower-bound vectors; `reports/2026-11-16_code_audit/itiger_benchmark_artifact_rerun.log`), its upper end the information relaxation at delay 2. The reference agent (exact filter, the best lower-bound vector's action) scores 16.306 on the 256 paired episodes of seed 11. Gap to the reference agent per episode, Student-t 95% over 10 runs:

| Arm | Return | Gap to the reference agent |
|---|---|---|
| main (T uninformed, O oriented, EXPECTED_MDP, 1000 simulations) | -14.10 | -30.411 [-33.433, -27.388] |
| known_t (T known as well; the upper reference) | -18.50 | -34.809 [-39.712, -29.906] |

- Per-run values are in the log. Both arms' mean returns lie about 30 below the lower end of the bracket.
- Learning curves (mean over runs, per 2,000 environment steps): main goes from -18.66 in the first block to -12.81 in the last; known_t goes from -5.06 to -13.83.
- No learned agent of ours exists on this problem yet, so by the registration BA-POMCP stands against the bracket. The comparison with our agent follows with the first interactive experiment.
- All 180 jobs of the campaign finished (2 arms x 10 runs x one learning period and 8 evaluation chunks); the campaign log has no error.
- **Two numbers that need an explanation** (the registered readout cannot give one; post hoc diagnostics follow below when they are run, labelled as such):
  1. The upper reference is not above the main arm. Post hoc Welch on the per-run gaps, known_t minus main: -4.399 [-9.826, +1.029] (15.0 df), so the difference is within the noise of ten runs, but its sign is the opposite of what more knowledge should give.
  2. known_t's return falls during its own learning period (-5.06 to -13.83), while main's rises.
- **Post hoc diagnostics** (written after the readout, not registered; `reports/2026-10-19_eval_protocol/ba_pomcp_itiger_diagnostics.py` and `.log`):
  - **A scale for both arms:** always listening is worth -19.88 over the 100-step window. Main (-14.10) and known_t (-18.50) sit between "never open a door" and a few points above it.
  - **1. What each run learned.** Posterior means after the learning period, averaged over the 1,000 particles, with rows weighted by the evidence they received:
    - No run converged to the mirrored model (growl accuracy under 0.5), which the Tiger readout had found once.
    - known_t learned the observation model nearly exactly: growl accuracy 0.835 to 0.857 over the ten runs (truth 0.85; the prior's mean is 0.625), creak accuracy 0.833 to 0.869 (truth 0.9; prior 0.536). So its low return is not a learning failure.
    - main learned it partly: growl accuracy 0.715 to 0.822, creak accuracy 0.673 to 0.777.
    - main's transition rows cannot be scored against the truth row by row. With T uninformed, fold states that share an observation row can swap labels freely, so the total-variation distance the script prints (0.88 to 0.93) measures the relabelling, not what was learned.
  - **2. The deficit is the search, not the learning.** BA-POMCP with the TRUE model at point-mass confidence, everything else as the main arm, returns -19.11 on the same 256 episodes: a gap of -35.419 +- 6.538 to the reference agent (1.96 s.e. over episodes; one planner). That is as much as the arms that learned their models (-30.411 and -34.809).
    - It opens doors in every episode, and in 36.7% of them its gap is under -50 (a wrong door at least once); 31.6% of its episodes are at or above the reference agent's return. On the canonical Tiger the same check found a wrong door in 7.0% of episodes.
    - So 1,000 simulations to a planning horizon of 20 do not plan this 26-state problem, whatever model they are given.
  - **What this says about the two numbers.**
    - known_t is not above main because both sit at the level the search allows. With a nearly exact model, known_t lands on the true-model level (-34.8 against -35.4), and main (-30.4) lies within the noise of it.
    - known_t's return falls during learning because it moves towards that level as its model becomes right. Its early, wrong model (growl accuracy 0.625) made it open less often. The script does not measure how often each arm opened, so this last sentence is a reading, not a measurement.

## Change log
Any change after the first held-out read is logged here with its date and reason. A change makes the claims it affects exploratory.

- No change to any claim. Operational note: held-out training seed 7 was aborted at collection 19 on 2026-10-19 by the VRAM guardrail (a diagnostic job beside it, not a property of the recipe). Before any held-out read it was moved to `runs/wumpus/heldout_seed7_aborted_20261019` and retrained from scratch with the identical command (training is deterministic per seed).
- Operational note (2026-11-02): C-e arm 2 (`abl_noretrain`) training seed 5 was aborted at collection 20 on 2026-11-01 22:31 by the VRAM guardrail (an unrelated process on the GPU took ~12 GB during the run; not a property of the arm). Before any read of the arm it was moved to `runs/wumpus/abl_noretrain_aborted_20261101` and the arm restarts from scratch with the identical command (training is deterministic per seed).
- Correction (2026-11-14, no claim affected): the model-free baseline takes 96 x 192 = 18,432 optimiser updates, not 19,200 as C-c and C-f state. It does not train during the 4 warm-up collections; the recipe does. The baseline's header (`recurrent_dqn.py` section 2) is corrected. The paper already gives 18,432.

## Deviations and post hoc analyses (submission audit, 2026-11-14)
Found by the audit in `reports/2026-11-14_submission_audit/` (`AUDIT_LOG.md`; every number below is recomputed from the per-game files by `ledger.py`, table in `statistics.md`). Nothing above this section was edited.

- **The interval the paper cites (deviation from "Statistics", item 3).** The registered interval is the hierarchical bootstrap. Two defects make it too narrow, so the paper now gives the Student-t 95% interval on the per-seed means (for a gap: on the per-seed means of the paired per-game differences) as its main interval and the registered one beside it, marked as such.
  1. It is a percentile bootstrap over 5 seed clusters. Its seed part is narrower than the Student-t interval by about sqrt(4/5) x 1.96 / 2.78 = 0.63.
  2. Every seed is analysed on the same 512 games, a crossed design, but the games are redrawn independently for every drawn seed. For an agent played once per analysis seed (exact-belief planner, heuristic, random) the row is the same for every seed and the interval is sqrt(5) too narrow. `aggregate.py` section 7 adds `crossed_bootstrap` (seeds and one set of games per resample); `hierarchical_bootstrap` is unchanged, so every readout above reproduces.

  | Quantity | Registered (nested) | Student-t over seeds | Crossed bootstrap |
  |---|---|---|---|
  | C-a return | 195.6 [155.3, 234.8] | [137.5, 253.8] | [146.8, 244.0] |
  | C-a exact-belief planner | 493.9 [476.3, 511.0] | (one play) | [453.9, 532.7] |
  | C-a gap to the exact-belief planner | -298.2 [-340.4, -259.3] | [-356.4, -240.1] | [-349.0, -248.7] |
  | C-c recipe minus R2D2-lite | +135.2 [78.4, 190.7] | [48.2, 222.3] | [74.7, 198.6] |
  | C-e arm 1, recipe minus max backup | +33.5 [-25.5, 94.4] | [-59.4, 126.5] | [-28.3, 98.4] |
  | C-e arm 2, recipe minus no retrain | +40.0 [2.6, 87.7] | [-23.6, 103.7] | [-3.1, 91.9] |
  | C-e arm 3, recipe minus JEPA | +53.2 [-1.0, 120.0] | [-41.8, 148.1] | [-6.1, 124.0] |
  | C-g N100 minus N50 | -8.1 [-23.2, 6.2] | [-22.9, 6.8] | [-26.7, 9.5] |
  | C-g N10 minus N50 | -22.4 [-43.2, -3.4] | [-46.4, 1.6] | [-46.3, 0.4] |
  | Exact one-step model minus learned (post hoc) | +167.6 [139.3, 195.4] | [131.1, 204.1] | [131.6, 202.7] |
  | Even mixture minus learned (post hoc) | +88.7 [52.4, 124.2] | [35.3, 142.1] | [48.5, 128.0] |

  Consequences for verdicts. C-e arm 2 keeps its registered verdict (CONTRIBUTES, by the registered rule on the registered interval), but the other two intervals include zero, and the paper claims no effect of whole-buffer retraining on the return; its effect on the random-history probes holds on every seed (cell 56-59% against 68-79%, heading 54-60% against 72-87%). C-g: the N10 difference is negative on every seed, and only the registered interval excludes zero. Every other reading is the same under the three intervals.
- **What the C-e arms change (clarification).** Arm 3 (`abl_jepa`) keeps `model.observation_horizon=5`, and the multi-step observation term trains the filter in either representation, so the arm replaces only the one-step observation term by latent self-prediction. Arm 2 (`abl_noretrain`) also takes 6,400 optimiser updates where the recipe takes 19,200, so it does not separate re-initialisation from the additional updates.
- **Reads of the held-out set.** Training seeds 5-9 with analysis seed 20261019 have been read for C-a, C-b, C-c, C-d, the three C-e arms and C-g, and, without registration, for the exact one-step model and the even mixture (`exact_model_play.py` section 6). No correction for these reads is applied; the paper says which were registered.
- **The evaluation games are fresh draws, not unseen layouts.** The 512 games of an analysis seed are drawn from the same 16,380 layouts as training; an agent's buffer of 8,960 games covers about 42% of the layouts in expectation (1 - exp(-8960 / 16380)).
