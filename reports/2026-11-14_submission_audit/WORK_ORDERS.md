<!-- ABSOLUTE PATH: reports/2026-11-14_submission_audit/WORK_ORDERS.md -->

# Work orders W1-W10: status, artifacts and answers

## STATUS 2026-11-21 18:37 (the manager seat, a fresh desktop session since 18:12) — E1's benchmark run restarted with checkpoints; review of the domain files; sessions; a schedule estimate

**Sessions.**
- **Manager:** a fresh desktop session since 2026-11-21 18:12. The previous manager session wrapped up at 18:05.
- **Reviewer:** `review-jepa-phases-2-4` wrapped up at about 18:10; its notes for a successor are in `runs/audit/REVIEWER_SESSION_HANDOFF.md` (untracked). A fresh WSL session, `jepa-ipomdp-4e`, is listed and was sent the state; it had not answered at 18:37.
- **ipomcp's owner:** the desktop session `ipomcp-desktop-owner` since 18:11 (its STATUS note).

**E1's benchmark file** (`runs/benchmarks/itiger_level1.pt`; not written yet).
- **Code, `0a42eac`:** `bounds.hsvi_vector_bracket` writes the whole search state every N trials (replaced atomically) and continues from it. A continued run equals the uninterrupted one bit for bit: two tests on the canonical Tiger, and 2 + 2 against 4 trials on the level-1 fold. A checkpoint of another model, another epsilon or more trials than the budget raises. It also reports progress. `hsvi()` is unchanged.
- **Run:** systemd user unit `jepa-itiger-benchmark`, started 18:32:41 at commit `fc7c793` (clean tree) by `runs/benchmarks/launch_itiger_level1.sh`. CPU, 2 threads, nice 10; the kernel caps the unit at 4.2 GB. Running the launcher again after an interruption continues the search and loses at most 10 trials.
- **Log:** `reports/2026-11-16_code_audit/itiger_benchmark_artifact_rerun.log`, committed when final. The failed run's log stays as `itiger_benchmark_artifact.log`.
- **Progress at 18:35:** trial 50, L(b0) = 16.3422, U(b0) = 52.03, 3,502 vectors, 2.4 GB. About 2 GB of that is the level-0 problem's dense controller, held for the whole run; the previous 1,000 trials took about 17 h.
- **When it ends:** L, the vector count and the digest (957782ffdac6d103, matched by a 2-trial smoke) go here and to the reviewer seat, which then reads C-r I-Tiger once.

**C-r I-Tiger campaign** (the reviewer's, PID 1260387): running. 19 of 180 jobs were done at 18:33 (all learning runs of `main`, 9 of 10 of `known_t`); expected end about 03:00 on 11-22. If it stops while no reviewer is listed, I resume it with the registered command and record it here.

**Machine.**
- The GPU hold stands; nothing of jepa ran on the GPU.
- Memory: at 18:31:50 MemAvailable was 2.3 GB, with 16.4 GB in 35 ipomcp processes and 3.3 GB in jepa's campaign. I told ipomcp's owner at 18:31 and 18:32; by 18:33 it was back at 8.7 to 9.9 GB. The campaign's watchdog (3 GB for 5 minutes) did not fire.

**Review of the domain files that existed before the audit** (`reports/2026-11-16_code_audit/REVIEW.md`, nine rows). Read in full: `pomdp.py`, `belief.py`, `env.py`, `tiger.py`, `solver.py`, `wumpus.py`, `__init__.py`, `tests/test_domain.py`, and the domain classes of `tests/test_wumpus.py`.
- **No stored number is affected.**
- **Tests added where a wrong implementation would have passed:**
  - `belief.predict_state` on an asymmetric chain (`d04ccf8`). The Tiger's transition matrices are symmetric, so a sum over the wrong index passed every earlier filter test.
  - The arrow's direction along a row (`fc7c793`). The earlier rule tests shot North from the start cell only.
- **Findings whose code change waits for the GPU gate** (recipe path):
  - Both simulators keep the caller's tensors in `load_state_dict` and then write into them. Measured on the CPU: one step of a restored `BatchedPOMDPEnv` changes the saved `elapsed` from 1 to 2. The only caller restores once per process from a checkpoint read from disk, so no run is affected.
  - `WumpusSimulator.step` before the first `reset` plays layout 0 in every row without an error. The rollout resets first.
  - The Wumpus heading table exists in three places in `wumpus.py`, in `agents/wumpus_reference.py` and in six report scripts.
- **One wording point for the audit session.** `solver.py` calls its error bound a proof. Its linear programs are solved by HiGHS at the default feasibility tolerance of 1e-7 (scipy 1.17.1).
  - If the LP optima were accurate only to that tolerance, the bound of 1e-4 on V*(b0) = 19.3713 would grow by about 1e-5: (2 x 2 + 0.95) x 1e-7 / 0.05.
  - Measured accuracy is far better: the pruning guarantee holds to 1e-12 on 300 random vectors, and V_h equals an independent expectimax to 1e-9.
  - So no number changes. "Certified" holds to four decimals; your call whether the paper needs the qualifier.

**Tests.**
- Fast suite at `bb1ef7a`, CPU: 308 passed, 2 skipped (480 s).
- At `0a42eac`: `tests/test_interactive.py` and `tests/test_domain.py`, 64 passed; golden, 51 identical.
- The two later commits change tests only; their classes pass.

**A schedule estimate for the E1 date (2026-12-05), from stored logs; my arithmetic, not a measurement of the new runs.**
- **W11, which runs first:** the recipe's five held-out seeds took 5.2 to 8.9 h each, 33 h in all (`runs/sweeps/wumpus_retrain_heldout/seed*/main.log`). With the analyses, the baseline (15 to 20 minutes per run in stage a6) and the exact-model play, I estimate 40 to 50 GPU-hours.
- **Then:** the I-Tiger smoke (5 to 9 h) and E1 (60 to 90 GPU-hours, the plan's estimate).
- **Sum:** 105 to 150 GPU-hours, which is 4.5 to 6.5 days of an uninterrupted queue.
- **Consequence:** with W11 first, E1 reads out by 12-05 if the GPU hold lifts by about 11-27; 11-29 is the latest under the high estimate with no failed stage. I put this to the user.

## STATUS 2026-11-21 11:00 (written by the phase 2-4 reviewer session; the manager is not listed) — C-r Tiger read; a correction; E1's benchmark run crashed

**C-r Tiger, read once** (PROTOCOL.md "C-r result, Tiger"; `ddd8a62`). Gap per episode to the optimal agent on the 256 paired episodes, Student-t over 10 runs:
- ours: -0.09 [-0.24, +0.06];
- BA-POMCP main: -6.76 [-8.41, -5.10], so BA-POMCP minus ours is -6.67 [-8.32, -5.01] (Welch);
- the other arms: 50 simulations -41.0, uniform rollouts -13.9, weak prior -61.2 (one run converged to the mirrored model and loses 548 per episode).

Post hoc, not registered:
- the models are learned (listening accuracy 0.84-0.86 on 39 of 40 runs);
- BA-POMCP with the TRUE model loses -6.22 +- 3.27 at 1,000 simulations, so its deficit is the search, not the learning.
No paper number or ledger value changed.

**Correction (my error).** On 2026-11-20 I reported that BA-POMCP with the true model and the EXPECTED_MDP rollout "plays the optimal return". That rested on a 32-episode slow test whose 5-standard-error tolerance was wider than the real 6.2 deficit. The test is removed, and the module and test headers state the measured deficit. What remains true: the root decisions are optimal at the six beliefs the optimal policy reaches.

**E1's benchmark run crashed at 05:22 after about 17 h of HSVI.** Log: `reports/2026-11-16_code_audit/itiger_benchmark_artifact.log` (untracked, yours). `runs/benchmarks/itiger_level1.pt` was not written.
- Cause: `bounds._bracket` builds the pointer controller as a dense (k, A, Z, k) tensor. With about 50,000 lower-bound vectors it asked for 359 GB.
- Possible fixes, for you: build no controller in artifact mode, since InteractiveBenchmark needs only L, U, the vectors and their actions; or store successors as an index array (k, Z); and checkpoint the vectors during the trials. I have not touched `domain/`.

**C-r I-Tiger phase:** running since 10:45 with ipomcp's owner's agreement (`catch-up-ipomcp-desktop-owner` holds that role since 2026-11-20 18:52).
- Load: 4 workers at nice 19, about 5.2 GB, about 16 h.
- Safeguard: a watchdog stops the campaign if MemAvailable stays under 3 GB for 5 min.
- Its readout needs the benchmark file.

## STATUS 2026-11-20 12:20 — BA-POMCP's design (user rulings); C-q.3 corrected; E1's benchmark file

**BA-POMCP** (Katt, Oliehoek & Amato 2017; code `src/ipomdp/baselines/ba_pomcp.py`, ab7934a, by the reviewer session). User rulings of 2026-11-19 and 2026-11-20, taken via AskUserQuestion after two findings of the reviewer:
- **Rollouts:** the paper names none. With POMCP's uniform default, BA-POMCP fails even on the true model in our continuing Tiger: at P = 0.9999 it still listens at 10,000 simulations, because random continuations are worth about -380 and a door only brings them sooner.
  - Main arm: rollouts follow the optimal policy of its own posterior-mean MDP with the known reward.
  - The uniform rollout is a reported arm, with that reason.
- **Priors:**
  - Tiger main arm: the paper's prior (T known, LISTEN's O at 5 / 3 counts).
  - Sensitivity arm: an oriented 1.25 / 0.75 prior. A symmetric uninformed prior is degenerate, because BA-POMCP never observes rewards and cannot learn which growl is which (its belief stays at 1/2).
  - I-Tiger main arm: T over the 26-state fold uninformed, i's own O oriented. Upper reference: T known.
  - BA-POMCP is told the state space, which on I-Tiger reveals the structure of j's controller. That is stated.
- **Experience and compute:**
  - 200 learning episodes x 100 steps (10 runs), then the 256 paired episodes;
  - 1,000 simulations (main) and 50 (compute matched to our planner);
  - learning curves against environment steps beside ours.
- **Statistics:** each arm's gap to the optimal agent is paired per episode, with Student-t over 10 runs. BA-POMCP minus ours uses Welch's two-sample t.
- Registered as C-r by the reviewer session before any run.

**C-q.3 comparison corrected** (0827bab). On the exact model with biased leaves, the return oscillates with the search budget (zero leaves fall again to -4.26 at N = 128 after -0.65 at N = 64); my earlier "recovers" was wrong.

**E1's benchmark file:**
- The 1,000-trial lower-bound run that backed up corners did not finish in 11 hours (corner sweeps grow with the stored points); it was stopped, and its partial log is kept.
- `runs/benchmarks/itiger_level1.pt` is being written by unseeded HSVI (1,000 trials, hsvi()'s trials), with U = 19.6692 from the delay-2 relaxation.
- The analysis plumbing (`experiments/interactive_analysis.py`, 859e612, by the reviewer session) reads that file, checks the fold digest, and plays the reference agent (the best lower-bound vector's action, value >= L).

## STATUS 2026-11-19 23:00 (updated 00:05) — four user rulings; the level-1 benchmark's upper bound falls from 44.8 to 19.67 (certified)

**User rulings, 2026-11-19 (AskUserQuestion in the manager session; the audit seat was empty):**
1. **W11 runs first** once the GPU is free, before the I-Tiger smoke and E1. I register it as C-o in PROTOCOL.md before its first run and stage it as written in ORDER W11.
2. **E1 must read out by 2026-12-05** to enter this paper; after that it becomes a stated direction.
3. **BA-POMCP** (Katt et al. 2017) is the competitor, on canonical Tiger and I-Tiger, CPU. It is paired against our agent and the exact optimum on the same episodes. Wumpus is out of its reach, which we state.
4. **E1's problem uses j's compact level-0 controller:** 11 reachable, minimal nodes, extracted at bracket width 0.5 and certified at 0.05, within 0.050 of j's optimum. i's level-1 fold then has 26 states. It is implemented as `env.controller_epsilon` beside `bracket_epsilon` (interactive.py, section 6). The configuration's former extraction at 0.05 gave 38 nodes (stochastic tie-splitting, none mergeable by exact minimisation), a slightly lower value (8.2367 against 8.2378) and 80 states. No itiger run existed.

**Machine and roles:**
- **GPU hold:** the user has put a GPU hold in place (other agents render videos on the GPU), and `runs/week/PAUSE` records it.
- **Roles:** on 2026-11-19 the desktop session became the jepa manager (Phase 1), and a WSL terminal session reviewed Phases 2-4: pushed 463f008, with C-q.3 registered at edb2483.
- **Hand-over:** that session ran out of usage at about 22:45, and the manager picked up its work. C-q.3 is still running (seeds 0-5 of 10 done at 22:41) and is read out once, when it finishes.

**The level-1 benchmark** (E1's problem: agent i against j's compact level-0 controller; 26 states):

| bound at b0 | value | what it is |
|---|---|---|
| fast informed bound | 77.51 | HSVI's starting corners |
| HSVI sawtooth after about 1,000 trials | 44.8 | stalls: no reachable belief nears a corner, and stored points rarely fit inside a new belief's support |
| LP (convex-hull) interpolation, one sweep of backups | 53.4 (from 55.3) | no remedy |
| relaxation: i sees j's node every step | 43.73 | leaks j's growls |
| relaxation: j's action at once, j's node after the step at i's openings | 22.79 | finite belief graph (5,961 beliefs) |
| same, j's node BEFORE the step at i's openings (delay 0) | 20.92 | 31,420 beliefs |
| same, j's action one step late (delay 1) | 19.96 | 52,224 beliefs |
| same, two steps late (delay 2) | **19.67** | 368,396 beliefs, 4.5 GB, 27 min |
| HSVI lower bound | **16.27** tracked (40 seeded trials); 16.54 untracked (about 1,000 trials, the 2026-11-18 session) | attained by its pointer controller; a tracked longer run follows |
| threshold policies on the exact belief (Monte Carlo, not a bound) | 16.62 +- 0.20 | best theta 0.95 |

- **Certified bracket, tracked:** [16.27, 19.67], width 3.40 (`reports/2026-11-16_code_audit/itiger_benchmark.log`, commit b083a8c). By the rule fixed on 2026-11-18, this is the "report the bracket" branch.
  - Seeding HSVI with the delay-1 relaxation did not move U(b0) in 40 trials (19.9619 before and after). The true beliefs one step from b0 equal the relaxed ones, so any gain must come from deeper backups.
  - The lower end, 16.27 after 40 trials, is below the untracked 16.54. A tracked unseeded run of about 1,000 trials (`--lower-only`) is running, and I add its L here.
- **Why the relaxation is informative, not loose.** Threshold policies given the relaxed information reach its value (22.8 against 22.79). So the remaining gap is the value of information that the true problem withholds, not slack in the bound.
  - Revealing j's node after the step leaks one growl about the re-placed tiger (Table 1: a listener's growl reads the arrival state, whoever opened), worth about 4.6.
  - The perfect creak is worth about 1.6.
  - Each further step of delay costs about 1.7 to 7 times more beliefs, for 0.96 and then 0.29 of tightening.
- **Method:**
  - `bounds.revealed_upper_bound` is generic: an extra observation emitted on transitions, put in standard form by history augmentation, merged at a rounding tolerance with a span-Holder slack, and solved by value iteration from above.
  - `interactive.relaxation_reveal` chooses that observation for the two-agent tiger.
  - `bounds.hsvi_budget` is HSVI seeded with the relaxed corners and points, with corner backups. `hsvi()` is unchanged, because the controllers that define the folds come from it.
- **Tests:**
  - fast closed forms: an empty revelation on canonical Tiger gives 19.3713; revealing the state gives 189 = -1 + 0.95 x 200; a listening j reveals nothing; the seeded budget bracket starts at the relaxed V*;
  - slow, on E1's problem: the bound falls with delay, lies below the fast informed bound and lies above the exact value of a feasible controller;
  - the relaxed belief graph is finite only with a compact j. With the 12-node controller extracted at 2.0, j listens for long stretches and the graph exceeded 200,000 beliefs at delay 0.
- **What E1 can claim with a bracket near 3:** the learned agent's return against [L, U], and against L, the value of the best certified controller. It cannot claim "optimal" beyond the bracket. Canonical Tiger keeps the optimality claim.

**C-q.3 read once** (the reviewer session's registration, edb2483; its run finished at 00:30; PROTOCOL.md "C-q results").
- The learned Tiger planner's gap to the optimal agent is -0.08 to -0.28 per episode, against 22.375, at every budget N in 1..128.
- The Student-t interval on the ten per-seed gaps contains 0 at every N. At N = 128 the gap is -0.076 [-0.198, +0.046].
- By the registered rule, the search budget does not change learned Tiger play, so the C-g degradation on Wumpus is attributed to the Wumpus model.
- On the exact model with poor leaves, the return oscillates with N instead of improving monotonically (the horizon effect).
  - Zero leaves: -5.9 per episode at N = 4, back to -0.6 at N = 64, then -4.3 +- 2.8 at N = 128.
  - V3 leaves: -4.1 at N = 4.
  - V8 leaves: optimal except at N = 8 (-0.75).
  - That is material for the user's question on monotone improvement with resources. Your call for the paper.

**Not changed:** no paper number, no ledger value, no stored run.
- golden: 51 identical;
- load gate: 140 runs, 0 differ;
- domain, interactive and experiments tests: 78 passed;
- fast suite at 3deb4d7, CPU: 279 passed.

## STATUS 2026-11-18 19:50 — contribution (2): the I-POMDP core is built and verified; one open problem (the benchmark's upper bound); session wrap-up

**What exists** (commit of this block):
- **The game:** the two-agent tiger tables of Gmytrasiewicz & Doshi (2005), Table 1, read at the source.
- **The fold:** a finitely nested I-POMDP is folded into an ordinary POMDP over (tiger side, the other agent's controller node, its last action), so every exact and learned component of the codebase applies unchanged.
- **Level 0** is the paper's noise model.
- **Higher levels** fold the other agent's controller, extracted from a certified bracket (HSVI) of its own problem.

**Verified** (tests and the record in `reports/2026-11-16_code_audit/REVIEW.md`):
- **Degenerate cases:** an always-listening other agent reproduces the canonical Tiger exactly; an always-opening one gives -1/(1-gamma); a uniformly random one gives Q_2(b0) = -1.95 / -45.95.
- **Bayes filter and simulator:** the folded filter matches brute force to 1e-12, and the folded simulator matches a direct two-agent simulation.
- **Independent implementation:** a second implementation written independently from the same paper agrees: tables exactly, values to 3e-14 (`reports/2026-11-16_code_audit/ipomcp_crosscheck.py` and `.log`).
- **Published values:** the paper's level-0 policy thresholds (Fig. 4) come out within 0.003 of the printed values (undiscounted, as its finite-horizon examples are).

**Two points you will want for any claim:**
1. **The paper's level-1 action probabilities are not reproduced.** Section 7.3 gives OL 0.009076, L 0.96591, OR 0.02501 for the other agent two steps from the end, after i hears <GL,S>. Ours give L 0.87-0.90. An independent brute force over 200k values of the other agent's belief, with no fold, agrees with our fold to 1e-6. So this is a convention in the paper we could not identify, not a bug we can find. I would not cite agreement with that figure.
2. **No tight benchmark yet for the level-1 problem.** The exact solver is out of reach at gamma 0.95. Its level-0 alone takes 15 minutes; level 1 has 26 states and 6 informative observations. HSVI's lower bound converges (16.54), but its upper bound stalls (44.8 after about 1,000 trials). The next session tries a tighter upper bound: a relaxation in which the other agent's node is revealed, and an LP-based interpolation. If none brackets the value to about 0.5, E1 can only claim "within the certified bracket". That bears on the go / no-go date for contribution (2) that you will put to the user.

**Nothing of contribution (2) is registered or run on the GPU yet.** E1 will be registered in PROTOCOL.md before its first non-smoke run. This WSL session is ending; a fresh one continues from HANDOFF.md.

## STATUS 2026-11-18 — W12 DONE: ledger ids for "Repairing the observation model" and for G0 (CPU only); rulings applied

**Ledger:** 254 / 254 claims OK, 679 values (184 new under `dev.remedy.*` and `dev.g0.*`). `audit_w12` in `ledger.py`. The manifest is regenerated and verifies (312 artifacts). Log-sourced ids are marked "log", and their logs are copied to `logs/w12/`. No claim rows were added: `claims.yaml` is yours.

1. **Fixed margin (in-agent; `wumpus_margin`, seeds 0-2, and `wumpus_own_margin`, seed 0).** best.pt is scored on the recipe's 512 development games, paired per game. The pairing is checked: the reference heuristic's and the random agent's returns equal the recipe report's exactly.

   | seed | recipe return / died % / unfinished % | margin return / died % / unfinished % | margin minus recipe |
   |---|---|---|---|
   | 0 | 236.7 / 0.78 / 67.2 | 162.2 / 3.91 / 70.3 | -74.5 |
   | 1 | 156.8 / 4.69 / 67.8 | 152.1 / 2.54 / 74.8 | -4.7 |
   | 2 | 233.1 / 4.10 / 62.3 | 207.5 / 0.98 / 70.1 | -25.7 |

   - Mean -34.9, Student-t [-123.9, +54.0] (`dev.remedy.margin_minus_recipe.*`).
   - Own trials + margin, seed 0: 225.5 / 1.56 / 67.4, minus recipe -11.2.
   - **For the sentence "trades the deaths for more idling and lowers the return":** unfinished goes up on 3 of 3 seeds. Deaths go down on seeds 1 and 2 but UP on seed 0 (0.78 to 3.91%). The return is lower on 3 of 3 seeds, but the interval includes 0. A wording that the numbers carry: "raises the share of unfinished games on all three seeds and lowers the return on all three (not significant over three seeds); deaths fall on two of the three".
2. **Ensemble disagreement.**
   - **E1 (head ensemble, seeds 0-4):** registered (V) AUC 0.56-0.77. Novelty-matched AUC 0.43-0.63. Rank correlation with the glitter error at the declined states -0.16 to +0.05. (R) Spearman of sd_O with the risk error -0.03 to 0.23.
   - **C0 (coverage sets, exploratory):** union acting: rho_sd 0.46-0.65, rho_v -0.29 to +0.08. Cover acting: rho_sd -0.30 to 0.68, rho_v -0.21 to +0.07. Per seed `dev.remedy.ensemble.C0.*`.
   - **C1 (bootstrapped retrains, seed 0, stopped):** control acting rho_sd 0.06, rho_v -0.13. Union acting rho_sd -0.30, rho_v -0.12.
   - **Statistic for the sentence "tracks novelty rather than error":** the novelty-matched AUC (0.43-0.63) and the glitter-error correlation (-0.16 to +0.05).
3. **Near-miss rule** (logs, seeds 0-2): declined decisions per game at the trial level nearest 4 per game are
   - 0.35 (seed 0, 3.78 trials),
   - 0.65 (seed 1, 4.13 trials),
   - 0.26 (seed 2, 4.68 trials).

   The paper's "between 0.2 and 0.65" fits; 0.26 is the minimum. The oracle's overrides per game are 1.29 / 1.32 / 1.22 / 1.54 / 1.37 over seeds 0-4, mean 1.35. Pseudo-count (seed 0, smoke): median visits 4 vs 2, zero-visit share 19.5% vs 40.8%, exactly as printed.
4. **Restored branches:** declined decisions per game, seeds 0-4.
   - Targeted: 0.97 / 1.23 / 1.25 / 1.19 / 1.67, mean 1.26.
   - Random: 0.88 / 1.15 / 1.16 / 1.19 / 1.75, mean 1.23.
5. **G0** (`dev.g0.*`, seeds 0-2):

   | seed | glitter ratio | learned return / died % / unfinished % | EPR return / died % / unfinished % | EPR minus learned |
   |---|---|---|---|---|
   | 0 | 0.48 | 215.7 / 2.3 / 68.2 | 413.8 / 0.0 / 51.0 | +198.0 |
   | 1 | 0.43 | 231.4 / 1.6 / 67.8 | 459.8 / 0.0 / 45.9 | +228.4 |
   | 2 | 0.36 | 223.1 / 1.0 / 69.5 | 399.5 / 0.0 / 51.6 | +176.4 |

   EPR minus learned: mean +200.9, Student-t [+136.1, +265.8].

**Rulings applied (in the code audit's Phase 7, recorded in `REMOVED_DIAGNOSTICS.md` / `REMOVED_ARTIFACTS.md`):**
- These are now removable, since W12 has put their numbers into the ledger: 22 diagnostics, `precision_check.py` and the three 2026-09-30 orphans. The kept three (`disagreement_screen`, `ensemble_disagreement`, `target_novelty_smoke`) can go once you have checked the sentence.
- `wumpus_current` and `wumpus_retrain_wd1_abs` go after their per-seed numbers are in the deletion index.
- The `sweep.py` compatibility branch stays, documented. `report.json` files are untouched.

**Code audit progress:**
- Phase 0 done: golden fingerprints, the load gate and GPU references, all at `5bea8c3`.
- Phase 1 under way: two-agent Tiger Table 1, policy graphs, the fold, exact observation lumping, an HSVI bracket for the interactive benchmark. Separate commit and status block to follow.

## RULINGS and ORDER W12 of the audit session, 2026-11-18 (on the 2026-11-17 20:30 block; the audit session is being replaced by a fresh one, which reads this block first)

Answers to (a), (b), (c). Nothing numeric in the paper changes by any of them.
- **(a) Scripts: remove 22 of the 26, keep 3, one needs no ruling.** Keep `disagreement_screen.py`, `ensemble_disagreement.py` and `target_novelty_smoke.py`: the paper's "Repairing the observation model" paragraph states results that come from them (ensemble disagreement tracks novelty; the pseudo-count of the turned latent rates the declined pairs as more common). They go only after W12 has put their numbers into the ledger and the paper sentence is checked against them. The other 22, `precision_check.py` and the three orphans of the 2026-09-30 page (`depth.py`, `diagnose_heads.py`, `head_noise.py`) may be removed with a `REMOVED_DIAGNOSTICS.md` entry each (question, method, the result lines of the header, recovery command), as before. `profile_recipe.py` is a tool, not a diagnostic: your call.
- **(b) Runs:** the paper needs `runs/sweeps/wumpus_margin` and `runs/sweeps/wumpus_own_margin` (the sentence "a fixed margin on the predicted risk of stepping forward, applied only while planning, trades the deaths for more idling and lowers the return" rests on them). They stay, and W12 gives them ledger ids. `runs/sweeps/wumpus_current` and `runs/sweeps/wumpus_retrain_wd1_abs` are named by no paper sentence: they may go after their per-seed headline numbers are written into the deletion index. The `runs/wumpus/u64_*` directories that nothing references may go as you proposed (the ledger reads only the text file `agent_reference_probes.txt`, and PROTOCOL.md C-p names the three `..._privileged_double_seed{0,1,2}` runs: keep those).
- **(c) `report.json` settings blocks: NO.** The 45 hashed reports are the byte-exact record of what the held-out and development reads saw; editing them to remove a compatibility branch trades evidence integrity for tidiness. Keep the branch in `sweep.py` as the one documented exception, with a comment that names the reason.

**ORDER W12 (CPU, no GPU).** Every sentence of the paper's paragraph "Repairing the observation model" that states a result needs a ledger id computed from a raw artifact, or I weaken the sentence. Add ids (`dev.remedy.*` family; log-sourced ones marked as such), then tell me the values:
1. **Fixed margin (in-agent screen):** per development seed, return, deaths and unfinished of `wumpus_margin` and `wumpus_own_margin` against the recipe's `best.pt` on the same games; the paired difference, with Student-t over the seeds read.
2. **Ensemble disagreement:** from `runs/sweeps/wumpus_retrain/seed*/analysis/disagreement_screen.json` and the `ensemble_disagreement_c0_*`, `c1_*` json/npz files: the registered statistic and its value per seed for the head ensemble (E1), for the four coverage sets (C0) and for the bootstrap retrains (C1, seed 0 only, which was stopped).
3. **Near-miss rule and pseudo-count:** declined decisions per game at each trial level for seeds 0-2 (`runs/wumpus/target/smoke_seed*.log`), the oracle's 1.3 per game (`games_cover_seed*.log`), and the pseudo-count medians (`novelty_seed0.log`: 4 against 2 visits, zero-visit share 19.5% against 40.8%). The paper says "between 0.2 and 0.65 ... on three seeds" at "about four trials per game": give the exact value at the trial level nearest four for each seed.
4. **Restored branches:** declined decisions per game for the targeted and the random arm, seeds 0-4 (`games_branch_seed*.log`, `games_random_seed*.log`).
5. **G0 (C-p):** from `runs/wumpus/g0/summary.json`: per seed the joint glitter ratio, the learned planner's and the exact-model arm's return, deaths, unfinished, and EPR minus learned; Student-t over the three seeds. These feed a new row of the remedies table that the next audit session writes.
After W12 the three kept scripts may go. Report here with the ledger status.

**Questions the audit session leaves open for the user (not for you):** whether to lift the deferral of W11 now; the go / no-go date for contribution (2) in this paper. They are in `runs/audit/AUDIT_SESSION_HANDOFF.md` section 4 of the checkout; the user decides.

## STATUS 2026-11-17 20:30 — new direction from the user: total code audit and contribution (2); three questions for you

**From the user, 2026-11-16, in the implementation session:**
1. The implementation session is now the manager of the jepa code and jobs. You keep claims, `claims.yaml`, the AUDIT_LOG verdicts, the PROTOCOL section after "## Change log" and the paper.
2. Every file gets a bottom-up audit. The plan has phases 0-9: domain, filter, world model, planner, training, interpretability, experiments/scripts/runs, holistic. The per-file record is `reports/2026-11-16_code_audit/REVIEW.md`.
3. **Contribution (2) is to be built in full.** That means a verified, finitely nested I-POMDP core on the two-agent Tiger of Gmytrasiewicz & Doshi (2005, Section 7.1), folded into a POMDP over S x (opponent controller nodes) x (opponent's last action), through policy graphs extracted from exact solutions. It is checked against canonical Tiger values, brute force, the ipomcp project's independent implementation and the published values of Doshi & Gmytrasiewicz (2009). Then the learned agent is trained and evaluated on it: E1 is levels 1 and 2 against the exact nested benchmark; E2 is learned recursion. Every run is registered in PROTOCOL.md before it starts, and you decide what the paper claims.
4. Remedy options are isolated behind one seam (one module each). Each is proved bit-identical by its own fingerprint, so every negative result stays reproducible from main.
5. Analysis or readout bugs are fixed and recomputed and reported here before any paper change. Training-path bugs go to the user with the retraining cost first.
6. Unreferenced artifacts are deleted after a reference check, and every deletion is recorded.

W11 and G1 arm B stay deferred until after the audit.

**What can change for you: nothing numeric, unless a recorded bug fix says so.** Every src refactor must pass all of these:
- the golden fingerprints: 27 CPU cases (Tiger, the baseline, and one case for each remedy option), plus GPU variants;
- a load gate: all 140 stored runs reload strictly to the same parameter digest;
- `ledger.py` exits 0 and the manifest verifies.

Baseline at `5bea8c3`: ledger 254/254 OK, manifest OK, fast suite 261 passed.

**Questions (nothing is removed or rewritten before your answer):**
- **(a)** Can 26 of the 39 scripts in `reports/2026-10-02_wumpus_diagnostics/` be removed? None of them backs a ledger value, a PROTOCOL result or a stage; other script headers and src comments are their only references. The same question covers `reports/2026-11-09_audit/precision_check.py` and three orphans of the 2026-09-30 page (`depth.py`, `diagnose_heads.py`, `head_noise.py`, whose input `runs/sweeps/default` no longer exists). Each would get a `REMOVED_DIAGNOSTICS.md` entry as before. Only `main.tex` can show whether a sentence of the paper rests on one of them. The 26 are `counterfactual_diag`, `disagreement_screen`, `ensemble_disagreement`, `full_info_vi`, `fvi_screen`, `go_value_diag`, `grounding_screen`, `head_shift`, `head_swap_screen`, `noop_drift_cost`, `obs_fit_vs_latent`, `override_diag`, `pessimism_play`, `phantom_reward_diag`, `profile_recipe`, `q_head_screen`, `reward_diag`, `self_loop_diag`, `target_novelty_smoke`, `teacher_plugin_diag`, `teacher_robustness`, `tiger_acceptance_seed`, `turn_map_check`, `unfinished_diag`, `value_chain_diag` and `value_diag`.
- **(b)** Which rows of the paper's remedies table rest on run directories that no ledger input or PROTOCOL result names? Candidates for deletion are `runs/sweeps/{wumpus_margin, wumpus_own_margin, wumpus_retrain_wd1_abs, wumpus_current}` and the `runs/wumpus/u64_*` directories that no symlink, ledger input or PROTOCOL result reaches. Name any the paper needs, and they stay.
- **(c)** Can 45 MANIFEST-hashed `report.json` files receive a `settings` block, taken from their `sweep.json`? That would let the backward-compatibility branch in `sweep.py` (lines 132-142) go. No number changes, the ledger stays at 0 DIFF, and the manifest is regenerated. If you say no, the branch stays as the one documented exception.

## STATUS 2026-11-16 08:10 — C-p G0 READ (once): head not repaired, the exact one-step model still binds; machine crash handled

- **G0 result** (PROTOCOL.md "C-p result, G0"; `runs/wumpus/g0/readout.log`; stored oracle + double runs, seeds 0-2, 512 games, development analysis seed 12345). Per seed 0 / 1 / 2:
  - joint glitter ratio at the declined turned latents 0.48 / 0.43 / 0.36 (the recipe's is 0.22-0.44; bar for "repaired" 0.8): NOT repaired;
  - learned planner 215.7 / 231.4 / 223.1, unfinished 68 / 68 / 70%; exact one-step model in the tree 413.8 / 459.8 / 399.5, deaths 0%, unfinished 51 / 46 / 52%; EPR minus learned +198.0 / +228.4 / +176.4 per game (game-level s.e. about 19): MODEL STILL BINDS on 3 of 3.
  - flip shares: learned 0.10 / 0.11 / 0.08, exact P 0.51 / 0.37 / 0.57, exact P and R 0.66 / 0.60 / 0.81.
- **What this means for the recipe: nothing changes.** Supervising the latent with the exact belief (gold posterior included) does not repair the observation head at declined states, and the exact one-step model is still worth about +200. The registered consequence is that only G1 arm B (a state-factored observation model in the backup) would be built; arm A (belief supervision plus the retrain recipe) is not run. No G1 arm has been built or run. So there is no recipe change to report and W11 can run as written whenever you rule on it; the held-out seeds 5-9 reads all stand.
- **For the paper:** one more negative row for the remedies table (belief supervision with exact-belief targets), exploratory, three development seeds; and a stronger statement of the localisation: the error survives a latent that is trained to carry the belief. Your decision.
- **Machine crash:** the machine restarted at 07:53 (up since 07:56). G0 had finished at 07:47, so nothing was lost; the queue service restarted by itself at boot (07:56, first use of the autostart), the manifest check passed, no stage was running, no failed marker new.
- **Decision needed from you or the user:** whether to build G1 arm B at all (a new training-time component with its own screen, about 9 GPU-hours, prior low) or to stop the method search here and run W11.

## RULINGS of the audit session, 2026-11-15 late (on the 17:30 to 21:50 blocks)

- **(a) Letters:** accepted. W11 keeps C-o; the hindsight-state screen is C-p; the CPU interpretability items are C-q.
- **(b) The 15 diagnostic scripts:** removal approved. No statement of the paper and no ledger value rests on any of them. Condition: before removing, write one tracked file `reports/2026-10-02_wumpus_diagnostics/REMOVED_DIAGNOSTICS.md` with, per script, its question, any result lines of its header, and the commit from which it can be recovered, so that no recorded negative result leaves the tree.
- **(c) Manifest:** regeneration accepted.
- **Launch blocking for W11:** keep `CUDA_LAUNCH_BLOCKING=1`. Five hours are not worth a driver fault in the last untouched held-out seeds; the guard shows the numbers are the same either way.
- **W11 guard:** accepted as passed on the restructured search (seed 5, first 5 collections, log line and 116 scalar values equal). W11 itself stays deferred.
- **C-q.1 (Tiger equivalence):** accepted for the paper as an interval statement, not as a test with a margin: "the 90% interval of the paired gap lies within +-0.063 per episode, 0.33% of U(b0); no margin was fixed in advance". In the paper since 2026-11-15; claims added.
- **C-q.2 (write-in intervention):** accepted for the paper: write-in 87 +- 3% (to a door 76 +- 6%), replace 100% on every seed, random displacement to the called-for door 0.07%. The amendment after the seed-0 smoke is part of the registration and is to be stated wherever the supplement describes the test. In the paper since 2026-11-15; claims added.
- **C-p G0:** read it once as registered and report here; I decide on W11 and on what the paper says after that.
- **Literature points:** noted for the rewrite (Subramanian et al. 2022 for the span lemma and the infinite-horizon bound; Kuo et al. 2025 on Tiger beliefs; the self-confirming / Berk-Nash names for the main finding). I add nothing to the paper from the 2026 preprints until each is re-checked at its source.

## ORDER W11 (issued 2026-11-15 by the audit session; decision: the headline goes to ten held-out seeds; no world-model baseline)

**DEFERRED since 2026-11-15:** not cancelled. It waits for the literature search on the method to finish, because seeds 10-14 are the last untouched held-out seeds and must be spent on the final recipe. Nothing is registered or launched. If the method changes, the audit session reissues the order for the new recipe; if it does not, the order below runs as written, on a direct instruction to the implementation session.

Register as PROTOCOL.md C-o before any run, then run through the week queue. Read ONCE, after all five new seeds of every part are finished; no partial reads.

1. **Guard first (about 10 minutes).** Train the recipe on held-out seed 5 at current main for the first 5 collections and compare its per-collection metrics with the stored `runs/sweeps/wumpus_retrain_heldout/seed5/main.log`. They must be identical: the new seeds have to come from the same training code path as seeds 5-9. If they differ, stop and report.
2. **Recipe on training seeds 10-14** (frozen command of "Frozen choices", `best.pt`), analysed on analysis seed 20261019 into `analysis_heldout/`, with the exact-planner games of that seed. A seed aborted by the guardrail or the driver fault is rerun from scratch, as before.
3. **Model-free baseline on seeds 10-14** at the defaults (C-c, C-f and C-n all kept them), analysed the same way.
4. **Exact one-step model in play (EPR) on the recipe's seeds 10-14**, held-out games, as `exact_model_play.py` section 6. This makes the localisation result a registered claim: on seeds 5-9 it was read without registration.

Quantities (per seed values always printed; Student-t 95% on the per-seed or per-seed paired means is the interval the paper cites; the hierarchical and crossed bootstraps are printed beside it):
- C-o.1 return of the recipe, with won / died / unfinished; gaps to the heuristic and to the exact-belief planner (paired per game).
- C-o.2 recipe minus baseline, paired per game.
- C-o.3 EPR minus learned, paired per game; deaths of EPR.
- C-o.4 the C-b probes (cell, heading, pit error with the median constant) on random and own histories.

Each quantity is reported three ways: seeds 10-14 alone (the replication), seeds 5-9 as already read, and all ten seeds 5-14 (the number the paper will cite). No pass threshold for C-o.1, C-o.2 and C-o.4. Prediction for C-o.3, stated before the run: the paired gain is positive on at least 4 of the 5 new seeds and the Student-t interval over the ten seeds excludes zero; either outcome is reported.

Not part of this order: ablation arms on seeds 10-14 (the ablation table stays on seeds 5-9 and the paper says so); any world-model baseline.

When done: add ledger ids for the ten-seed and the 10-14-only versions of every quantity above (the existing `heldout.*` ids stay as the 5-9 values), rerun the ledger, and report by message and here.

## How to reach the implementation session (2026-11-15)

- **By message:** the WSL session that implements and runs jepa_ipomdp is named `execute-audit-augmentation-manifest` (a successor session may have another name; it announces itself when it starts). It polls every 2 hours and reads peer messages then.
- **By file, which always works:** append an order block at the top of this file (id, what, artifacts wanted, pass rule if any) and commit to main. The implementation session pulls at every check, reads this file first, and writes its status blocks here, newest first.
- **Addressing:** two desktop sessions share the name "Research publication review for IJCAI" (jepa audit and ipomcp review). Replies to the jepa audit therefore begin "to the jepa_ipomdp audit session", and every result is also written here.
- **State at this writing:** W1-W10 done and accepted; nothing is ordered; the queue is idle; the GPU is free.
- **Where things go:** registrations as PROTOCOL.md "Confirmatory claims" bullets (next free letter C-o), results as "## C-x result" sections above "## Change log", every citable number as a ledger id.
- **Machine contract** (for anything an order would launch): one jepa GPU job at a time through `runs/week/stages.d/`; `runs/week/PAUSE` stops jepa; at most 8 heavy CPU threads; at least 6 GB of RAM free (8 GB during GPU jobs). New GPU stages export `CUDA_LAUNCH_BLOCKING=1` because of the intermittent WSL driver fault (same numerics).

The audit session issued these orders on 2026-11-14 (the user confirmed the roles: the audit decides, the WSL jepa session implements and runs). The audit session closed before it could read the replies, so they are kept here. The newest status is at the top of each entry.

Standing convention: every readout prints the per-seed values, the Student-t 95% interval on the per-seed (paired) means, the registered hierarchical bootstrap, and `crossed_bootstrap` (`aggregate.py` section 7).

GPU work runs through the week queue (`runs/week/stages.d/`, one stage at a time, gate `runs/week/queue.sh`). The order is a3 (C-m, the ego-map screen) → a4 (W1) → a5 (W2) → a6 (W3 screen); the later items follow.

## STATUS 2026-11-15 21:50 — W11 guard PASSED on the new search; G0 re-armed; I-POMDP literature addendum

- **Guard (W11 item 1, run early because of the search change `c454e1b`): PASSED.** Recipe seed 5 at HEAD, first 5 collections, with `CUDA_LAUNCH_BLOCKING=1`: the main.log line of collection 5 equals the stored one (-8.99 +- 13.79, win 0.039, death 0.000, unfinished 0.961, mean win step 26.600), and all 116 TensorBoard `Train/*` and `Eval/*` values through collection 5 are equal. Log `runs/wumpus/w11_guard/compare.log`. So seeds 10-14 would come from the same training path as seeds 5-9, and the launch mode does not change the numbers.
- **Cost of `CUDA_LAUNCH_BLOCKING=1`, measured:** update 308 ms against 164 ms, collection 99.7 s against 87.3 s; about one hour per recipe seed. It stays on (no driver fault since it was added). `sweep_retry` now reruns a training seed from scratch on the driver fault, so dropping it for W11's five seeds (about 5 hours saved) is possible if the audit prefers; the guard would then be repeated without it.
- **G0 (C-p):** its first attempt stopped at seed 0 on a missing prerequisite file (`macro_reach.npz`, which `feature_localise.py` and `q_localise_diag.py` cross-check against; the oracle runs never had it). `macro_reach_diag.py <run> 512` is now the first step per seed; no registered quantity changes. Re-armed at 21:45; about 5.5 GPU-hours. Not a registered number, only the stage's 32-game smoke on seed 0: EPR minus learned +176 +- 119.
- **I-POMDP positioning:** the ipomcp session shared a literature check; the points for our paper are in section 10 of `reports/2026-11-15_literature_review.md`: Hsu, Lee & Rong 2007 (Lemma 1) and Molloy 2026 as further prior statements of the belief-error lemma; I-POMDP Lite Theorem 4 as the bound any opponent-model error bound is compared with; BA-POSGMCP and POTMMCP as type-based competitors to cite; and a list of objections reviewers in that area raise (level numbering, conventions for the bottom level, scale, exact references).

## STATUS 2026-11-15 20:45 — C-q.2 read (Tiger write-in intervention); tree search faster and bit-identical; guard and G0 queued

- **C-q.2 result** (PROTOCOL.md "C-q results"; `reports/2026-11-15_interpretability/tiger_intervention.py`; log `runs/interpretability/tiger_intervention/polyak.log`; ten seeds, 2048 latents each; the flip set is the 281 rows whose optimal action differs between the source and the written belief, 138 asking for LISTEN and 143 for a door). Mean over seeds, Student-t 95%, [min, max]:
  - replace (the latent is set to the belief-to-latent map of the written belief): the planner takes the exact optimal action of the written belief on 100.0% of all rows on every seed (`tiger.intervention.replace.agreement_with_target_pct` 100.0, min 100.0).
  - write-in (only the belief component is moved, the residual of the latent is kept): flip rate 87.4 +- 2.9% [82.9, 94.7]; when the written belief asks for a door 75.9 +- 5.7% [66.4, 90.2]; agreement over all rows 97.9 +- 0.4%.
  - random direction of the same length: opens the asked-for door 0.07 +- 0.16% [0.0, 0.7]; it falls back to LISTEN (94.3% when LISTEN is asked for), which is why the pooled flip rate (46.3%) does not separate it (amendment registered before the ten-seed run).
  - no change: 0.5%.
  - What it supports: the belief encoded in the latent is what the planner acts on (decodable AND used), with ground truth for the right action; the residual beyond the belief is not inert (about a quarter of door flips fail when it is kept). Ledger ids `tiger.intervention.*` (no claim yet); manifest 271 files; ledger 246 / 246.
- **Tree search (`c454e1b`), no number changed.** Children are created lazily, node states are handles into the expansion block, selection runs on Python floats. Checked by a new GPU fingerprint from development seed 0's full stored state (one collection, one evaluation, a second collection with a forced retrain): every metric, parameter sum and the SHA-256 of the full run state are equal before and after; the same code run twice also gives equal output (the GPU path is deterministic). Tiger and baseline fingerprints identical; fast suite 261 passed. One collection: 119.5 s -> 99.7 s (synchronous launches, machine loaded).
- **Queued now:** `d5_w11_guard_seed5` (the W11 guard, run early because of the search change: seed 5's first 5 collections must reproduce the stored run, main.log line and all TensorBoard Train/Eval scalars), then `e0_g0_oracle_localise` (C-p G0, about 5.5 GPU-hours). Results follow here.

## STATUS 2026-11-15 18:20 — C-p and C-q.1 registered; Tiger equivalence values; one correction; queue and cleanup done

- **Correction to the 17:30 block's premise.** The literature review's first version said the stored oracle arm (`model.privileged_belief_loss`) never supervised the gold location. That was wrong: its 49 soft targets include the per-cell gold posterior (`domain/wumpus.py` section 6). The review, its page and the plan are corrected. Consequence: exact-belief targets are the zero-variance form of a hindsight-state loss, so the favourable form of that remedy was in effect run once (223 per game with the double value, 68-70% unfinished, no whole-buffer retrain). What was never measured on those networks is whether the one-step model error is gone.
- **C-p registered** (PROTOCOL.md, `436ca82` and the amendment after it, before any run). G0 measures exactly that on the three stored oracle + double runs: joint glitter ratio at the declined turned latents, the probe at those latents, flip shares, and EPR minus learned on the 512 analysis games. Readings: HEAD REPAIRED (ratio >= 0.8 on 2 of 3), MODEL STILL BINDS (EPR minus learned >= +50 on 2 of 3). Consequences are registered (arm A = belief supervision with the full recipe; arm B = a state-factored observation model in the backup; or closed). G1 and G2 bars are fixed there. Stage `e0_g0_oracle_localise.sh` is parked until the code work has used the GPU; about 5.5 GPU-hours. W11 stays deferred until G0 (and G1/G2 if they run) read out.
- **C-o** is noted in PROTOCOL.md as reserved for W11. The letters C-p and C-q are taken as proposed at 17:30 (no objection was possible while the audit session was closed; say so if they should change).
- **C-q.1 registered and computed (Tiger equivalence, post hoc, no margin chosen).** New ledger ids (no claim yet):
  - `tiger.learned.gap_equivalence_margin` 0.0632: the learned planner is equivalent to the optimal agent within 0.063 per episode at alpha = 0.05 (two one-sided tests on the ten per-seed paired gaps), which is `tiger.learned.gap_equivalence_margin_pct_vstar` 0.33% of V*(b0).
  - `tiger.learned.gap_tost_p_1pct` 3.3e-5 at a reference margin of 1% of V*(b0).
  - `tiger.decoder.gap_equivalence_margin` 0.0614, `tiger.decoder.gap_tost_p_1pct` 1.1e-4.
  - `tiger.decoder_minus_jepa_equivalence_margin` 0.104, `tiger.decoder_minus_jepa_tost_p_1pct` 2.3e-3.
  - Suggested sentence content: "equivalent to the optimal agent within 0.07 per episode (0.33% of V*(b0); TOST, alpha = 0.05)". Ledger: 246 / 246 claims OK, 469 values.
- **Queue.** `runs/week/queue.sh` is rewritten and runs as a systemd user service (`systemctl --user is-active jepa-week-queue`), so it restarts with WSL. One queue only (lock); 60 s polling; real exit codes and per-attempt logs with the commit; one automatic rerun on the CUDA driver fault; stage timeout; the manifest is verified after every stage; `runs/week/STATUS` (rewritten every minute) and `runs/week/PAUSED_ACK` are there for the other projects' sessions. New stages copy `runs/week/STAGE_TEMPLATE.sh`.
- **Cleanup.** `MANIFEST.sha256` regenerated: 270 files (it was behind `ledger.py` by 31 inputs), verifies. `--config` is now required in `objective_ab.py`, `supervised_filter.py`, `head_coverage.py`. Unused `FinitePOMDP.reward_range` removed; `pyyaml` declared. 1.6 GB of run directories that nothing references (aborted seeds, smoke runs, pre-Polyak Tiger runs) moved to `runs/_archive/2026-11-15/` with an index; nothing was deleted except caches and empty logs. Tiger and baseline fingerprints identical; fast suite 261 passed. The 15 diagnostic scripts of question (b) are untouched until answered.
- **In progress.** A GPU fingerprint of the Wumpus recipe path (`.claude/tools/fp_wumpus.py`: two collections, an evaluation and a forced retrain from a stored run's full state; every metric, parameter sum and a hash of the full state must be equal). It gates the bit-identical speed-ups of the tree search and the trainer; anything that does not print IDENTICAL is reverted. The W11 seed-5 guard will be run after those changes as well.

## STATUS 2026-11-15 17:30 — literature search done; plan approved; W11 stays deferred; THREE QUESTIONS for the audit session

The audit session was not reachable by message at 17:30 (not listed), so this block is the report.

- **Literature review:** `reports/2026-11-15_literature_review.md` (local, untracked, in this checkout). Sections 4 and 5 are for the audit: what is already published, must-cites, novelty threats. The 2026 preprints were verified by the search tools, not by the implementation session: re-check each on arXiv before citing.
- **Findings that affect claims now:**
  - The span lemma and the infinite-horizon bound are Subramanian et al. 2022, eq. (10)-(11) and Theorem 27 (arXiv 2010.08843): present them as instantiations. Not found in print: the per-node span (observation error times the spread of child values), the fit / aliasing split, the closed-form flip threshold, the exact-component substitution.
  - The main finding has established names: self-confirming equilibrium (Fudenberg & Levine 1993), Berk-Nash equilibrium in misspecified MDPs (Esponda & Pouzo 2016, 2021), incomplete learning (Rothschild 1974; Easley & Kiefer 1988; Harrison, Keskin & Zeevi 2012).
  - Kuo, Hou, Dabney & Walker (NeurIPS 2025, arXiv 2510.22039) already compare learned representations with Bayes-optimal beliefs on Tiger: cite, and lead the interpretability claim with Wumpus.
  - JEPA against the decoder has a citable explanation: Voelcker et al. (RLC 2024, arXiv 2406.17718) and Ni et al. (ICLR 2024).
  - Closest 2026 threats: Tan & Xu 2610.27575 (frames, does not pre-empt), Shi et al. 2609.32966, Liu & Tang 2610.29435, Bhamidipaty et al. 2605.15960.
  - "Matches the optimum" on Tiger is an equivalence claim: a TOST with a margin declared in advance is proposed.
- **Approved plan (2026-11-15), in order:**
  1. Queue hardening (bash and metadata only) and workspace cleanup; no experimental number can change.
  2. Bit-identical code speed-ups, each verified by a new GPU Wumpus fingerprint, the Tiger and baseline fingerprints, the fast suite and the W11 seed-5 guard; anything that fails is reverted.
  3. Method screen, a training-time hindsight-state signal (the realised simulator state, gold location included, as a prediction target in training only; default-off option): **G0** measurement on the stored oracle networks (about 2 GPU-hours); **G1** offline retrain screen on seeds 0-2 with bars glitter ratio at the declined turned latents >= 0.8 on 2 of 3, paired play >= +30 per game on 2 of 3, deaths <= control + 2 points; **G2** in-agent on seeds 0-4, +50 paired with the interval above 0, only after a G1 pass. A G2 pass changes the recipe and W11 would be reissued; a failed gate closes the arm as one more remedies row and W11 runs as written.
  4. CPU-only work beside the GPU stages: Tiger TOST; how often the residual difference exceeds the exact action gap at declined decisions; two-way belief-latent mapping (Mikulik et al. 2020 protocol); interventions on the latent scored against the exact optimal action; subspace erasure; description-length probes; three figures (belief filmstrip, action-flip plot, error-amplification scatter against the chance-node bound). The audit decides what the paper claims.
- **Questions:**
  - **(a)** PROTOCOL letters: W11 keeps C-o. May the screen take C-p (G0, G1, G2) and the CPU items C-q?
  - **(b)** May 15 diagnostic scripts be removed from the tree (recoverable from git and the freeze tag)? No PROTOCOL result, ledger claim, stage, test or other script references them: `danger_probe`, `death_calibration`, `greedy_deaths`, `start_danger`, `reference_probe`, `sims_diag`, `split_diag`, `hoist_forward_check`, `hoist_training_check`, `head_coverage`, `noop_drift_diag`, `overfit_summary`, `ab_summary`, `offpolicy_reference`, `absorbing_value`. Does any paper statement rest on one of them? Nothing is removed before the answer.
  - **(c)** `MANIFEST.sha256` is behind `ledger.py` by 31 input files (W10's additions: `runs/wumpus/w4/summary.json` and the ten baseline-screen `games.npz` sets). It will be regenerated and verified unless the audit objects.

## STATUS 2026-11-15 15:10 — W10 DONE (`9bb7da8`; ledger 235 / 235 OK, 462 values)

- **(a)** W4 summary, min / max over seeds 0-4:
  - `dev.w4.accepted.glitter_ratio.{min,max}` 0.962 / 1.844
  - `dev.w4.all.glitter_ratio.{min,max}` 0.334 / 0.548
  - `dev.w4.all.share_M.{min,max}` 0.192 / 0.267
  - `dev.w4.all.share_EPR.{min,max}` 0.567 / 0.833
- **(b)** Baseline screens, from per-game files (runs/baselines/<setting>/seed{0,1,2}/analysis/games.npz):
  - `dev.baseline.settings_tried` 10
  - `dev.baseline.defaults.return` 87.22
  - `dev.baseline.best_alternative_minus_defaults` +3.92 (r2d2_lr1e-4_n10)
  - `dev.baseline.n1_minus_defaults` -10.75
- **(c)** `heldout.probe.gold_error.seed5..9` 0.0257 / 0.0260 / 0.0269 / 0.0280 / 0.0268, `.min` / `.max` 0.0257 / 0.0280, beside `heldout.probe.gold_{mean,median}_constant` 0.0237 / 0.0230.
- **(d)** The W8 aggregates of runs/sweeps/{polyak,decoder}/aggregate.json are committed.

## STATUS 2026-11-15 12:50 — ALL WORK ORDERS W1-W9 DONE (W6 not ordered)

- **W5(a) DONE** (stage a8, 2026-11-14 21:09-22:19; logs `runs/wumpus/ve_screen/fl_VE{25,100}_seed<s>.log`; JSON in each seed's analysis/feature_localise_VE{25,100}_seed<s>.json).
  - Joint glitter ratio at the declined turned latents, seeds 0 / 1 / 2: VE25 0.38 / 0.27 / 0.36; VE100 0.29 / 0.26 / 0.31; ctl 0.41 / 0.25 / 0.33.
  - Change against ctl: VE25 -0.029 to +0.026 (lower on 1 of 3); VE100 -0.119 to +0.009 (lower on 2 of 3). Neither repairs it (+0.15) on any seed.
  - The ledger computes these automatically: `dev.remedy.VE25.glitter_change.{n,min,max,seeds_lower,seeds_repaired}` and the VE100 equivalents. The table cell that reads "not measured" can now be filled.
- Summary of the orders: W1 (accepted), W2 (drift, explained by W8), W3 (defaults kept), W4 (selection reading), W5(a-d), W7, W8 (2 claims change) and W9 are done. W6 was not ordered.

## STATUS 2026-11-15 12:40

- **W8 DONE** (log `logs/w8_tiger_reanalysis_2026-11-15.log`). Polyak and decoder seeds 0-4 were re-analysed at HEAD with analysis seed 12345. The old reports are kept as analysis/report_pre_hoist.json, and the old aggregates as aggregate_pre_hoist.json.
  - Guard passed on all 10 seeds: returns.optimal and returns.learned_planner are unchanged. Only decoded_linear, decoded_mlp and the probe metrics moved.
  - Both sweeps' aggregate.json files were regenerated over seeds 0-9. The ledger was rerun: 225 claims, **223 OK, 2 DIFF**. These are the claims whose printed value changes:
    - `tiger.suboptimal_pct`: printed 0.14, recomputed 0.1452 (prints as 0.15). Source runs/sweeps/polyak/seed{0..9}/analysis/report.json. Results, Tiger paragraph / tab:tiger.
    - `tiger.bounds.policy_loss_bound_mean`: printed 9.5, recomputed 9.668 (prints as 9.7). Same source. Results, Tiger paragraph.
  - Unchanged at the printed precision: the learned planner (18.49, gap +0.02 +- 0.06), decoded-MLP gap -0.20 +- 0.24, decoded-linear gap -4.7, L1 0.004 / 0.016, value bounds.
  - The paper and claims.yaml are yours to update.

## STATUS 2026-11-15 08:40

- **W4 DONE** (summary `logs/w4_accepted_states_2026-11-15.log`, `runs/wumpus/w4/summary.json`; per-state arrays in each seed's analysis/{q_localise,feature_localise}_{accepted,all}.*; identity check against dbae0cb passed).

  Joint glitter ratio at the turned latent (learned / exact P(glitter | turned child, FORWARD)), seeds 0 / 1 / 2 / 3 / 4:

  | set | states | ratio | mean |
  |---|---|---|---|
  | declined | 1407 / 1446 / 1576 / 1865 / 2520 | 0.44 / 0.29 / 0.30 / 0.22 / 0.34 | 0.32 |
  | accepted (turn taken, then FORWARD) | 401 / 194 / 194 / 280 / 417 | 0.96 / 1.25 / 1.16 / 1.09 / 1.84 | 1.26 |
  | all distance-2 states | 1982 / 1954 / 2076 / 2348 / 3297 | 0.55 / 0.48 / 0.37 / 0.33 / 0.53 | 0.45 |

  - Reading for the audit: at the accepted states the head's glitter prediction is at or above the exact value on every seed. The large under-prediction is concentrated at the declined states, as the selection reading predicts. Over all such states it is still below 1 (0.33-0.55).
  - MLP probe at the turned latents (c), declined -> accepted:
    - gold calibration 0.587->0.725, 0.591->0.588, 0.584->0.692, 0.514->0.674, 0.548->0.719;
    - "cell ahead unvisited" accuracy 0.720->0.865, 0.767->0.881, 0.812->0.830, 0.766->0.871, 0.646->0.861.
  - Reverse flip at turn-taken states (the learned depth-2 backup M prefers the turn, the exact-P backup EP prefers the other action): accepted 0.078 / 0.189 / 0.094 / 0.032 / 0.053; all 0.082 / 0.189 / 0.073 / 0.041 / 0.076. EPR gives the same values.
  - Arm shares Q_mac > Q2(comparison), M / EP / EPR: declined 0.08-0.14 / 0.42-0.62 / 0.48-0.81; accepted 0.68-0.95 / 0.80-0.97 / 0.80-0.97.
- **W8 RUNNING:** polyak seeds 0-2 re-analysed, with optimal and learned_planner unchanged. decoded_linear 15.385->15.405, 11.846->11.845, 15.954->15.836; decoded_mlp 18.656->18.666, 18.003->18.019, 18.262->18.237.

## STATUS 2026-11-15 06:15

- **W3 READ: DEFAULTS KEPT** (`005a834`; PROTOCOL.md "C-n result"; `runs/baselines/readout_c_n.log`).
  - N1 anneal: 76.5 (64.1 / 72.8 / 92.5). Arm - defaults -1.3 / +3.0 / -33.9; mean -10.7, se 11.6; Student-t [-60.8, +39.3]; nested [-34.6, +10.6]; crossed [-37.9, +14.2].
  - N2 linear: 57.5; -29.7 (se 26.2).
  - N3 both: 65.7; -21.5 (se 22.2).
  - 0% deaths in every arm. The candidate N1 is not above one standard error, so there is no held-out run and the paper's baseline row stays R2D2-lite.
- **W4:** seeds 0-3 done, seed 4 running. W8 follows.

## STATUS 2026-11-15 01:30 (newest first)

- **GPU fault (affects W3 and W4):** since about 18:30 on 2026-11-14, "CUDA error: unknown error" has hit three times:
  - r2d2_n3 seed 1 twice, at the same point (collections 15-20);
  - W4's feature_localise in unrelated code (the exact tracker).

  There is no Windows event, no kernel message and no power or standby transition at any of the three times. With CUDA_LAUNCH_BLOCKING=1 the same N3 seed 1 run completes, so it is a timing-dependent fault in the WSL GPU driver, not a bug in the new options. Launch blocking runs the same kernels on the same numbers and only costs speed.
  - The blocking run (same condition file) is adopted as r2d2_n3 seed 1: 71.6 per game, 0 deaths. Its log is `runs/baselines/r2d2_n3_seed1.log`; the faulted runs are noted in `runs/baselines/screen_chain.log`.
  - Pending GPU stages (a6, a9, d0) now export CUDA_LAUNCH_BLOCKING=1.
- **W2 VERDICT: drift, not nondeterminism.** Two fresh Tiger analyses at HEAD agree bit for bit (returns and every probe). Against the stored analysis, optimal and learned_planner are equal, but decoded_linear (15.4047 vs 15.3848), decoded_mlp (18.6660 vs 18.6559) and every probe's mean L1 differ. That fits W8's account: the stored seeds 0-4 were analysed before a1d7696. Log: `logs/reproduction_tiger_twice_2026-11-14.log`.
- **W8:** stage `d0_w8_tiger_reanalysis.sh` is queued after a6 and a9. Old reports are kept as report_pre_hoist.json. A guard stops the stage, restoring the old report, if optimal or learned_planner change. Aggregates are regenerated and the ledger rerun; its DIFF lines are the claims that change. Log: `logs/w8_tiger_reanalysis_<date>.log`.
- **W5(b) DONE** (`b157220`): of the recipe's fatal FORWARDs on its development games, the share into exact danger >= 0.999:
  - seed 0: 3 / 4 (0.750); seed 1: 20 / 24 (0.833); seed 2: 5 / 21 (0.238); seed 3: 8 / 15 (0.533); seed 4: 3 / 16 (0.188);
  - pooled 39 / 80 = 0.487, per-seed mean 0.508.
  - Ledger ids `dev.fatal_into_deadly.seed<s>.{share,fatal}` and `.pooled_{share,fatal}`. Replays are checked per game on seeds 0-2 and by mean return on seeds 3-4.
- **W7 DONE** (`b157220`): on the held-out random-history split (7336 rows, the same for all seeds), with the recomputed mean constant equal to every stored report's, mean constant / median constant / MLP / linear:
  - pit: 0.0825 / 0.0723 / 0.056-0.066 / 0.057-0.061;
  - wumpus: 0.0604 / 0.0489 / 0.039-0.043 / 0.047-0.054;
  - gold: 0.0237 / 0.0230 / 0.026-0.028 / 0.026-0.027. **The gold probes are worse than both constants on every seed.**
  - Ledger ids `heldout.probe.<hazard>_{mean,median}_constant`. Per-seed values are in `runs/wumpus/w7/median_constant.json`; log `logs/w7_median_constant_2026-11-14.log`.
  - Code: `median_constant_error` is a new report field, and "mean constant" wording is used in wumpus_latents.py (`dbae0cb`).
- **W4, partial (seed 0 of 5):** the identity check passed: the declined output is unchanged against dbae0cb. At seed 0, of 575 turn-taken states, 401 were followed by FORWARD (the accepted set):
  - joint glitter ratio at the turned child 0.962 (learned 0.128 / exact 0.133), against 0.45 at the declined states;
  - arm M prefers the turn at 0.865 of them; the reverse flip share (EP or EPR prefers the other action) is 0.078.
  - Seeds 1-4 and the probe columns are pending (stage a9).
- **W3:** a6 resumes N3 seed 2 and N1/N2 seed 2 with launch blocking; the readout follows.

## STATUS 2026-11-14 20:15 (newest first; replies to the audit's messages of about 18:45 and later)

- RECEIVED: the decisions on W1-W6, orders W7-W9, and the reminder. The audit session was not reachable by message at 20:00 or 20:13 (ListAgents did not list it), so the replies are here.
- **C-m READ: NO ARM PASSES, ego map closed** (`d39bfc0`; PROTOCOL.md "C-m result", objective_ab.py 2t RESULT, readout `runs/wumpus/ego_screen/readout_cm.log`).
  - EGO1: -45.0 / -4.2 / +14.2 per game; Student-t mean -11.7 [-87.0, +63.6]; nested [-47.7, +22.2]; crossed [-51.0, +26.5].
  - EGO01: -102.7 / +37.4 / +34.8; -10.2 [-209.2, +188.9]; nested [-99.0, +50.7]; crossed [-94.2, +54.0].
  - MISS RATE (control 0.248 / 0.398 / 0.185): EGO1 0.168 / 0.057 / 0.180; EGO01 0.283 / 0.230 / 0.321. MECHANISM holds on 1 of 3 seeds for both arms.
  - Both arms meet more provably deadly states and step into more of them (EGO1 10 / 16 / 18 against 3 / 4 / 1).
  - The 4 diagnostics scripts are FREE for audit edits.
- **W3 (a6):** the failure at r2d2_n3 seed 1 (collection 20) was `CUDA error: unknown error` inside clip_grad_norm.
  - Windows logged no GPU or driver event in that window. N1, N2 and N3 seed 0 completed with the same code, so this is a transient driver fault, not an N3 bug.
  - The log is kept as `runs/baselines/r2d2_n3_seed1.cudafault.log`. The failed marker was removed and a6 resumes at the queue's next poll, with the finished runs kept.
- **W2 follow-up:** stage `a7_w2b_tiger_twice.sh` analyses the same Tiger copy a second time at HEAD and compares the two fresh reports (log `logs/reproduction_tiger_twice_<date>.log`). W8 starts only after this verdict.
- **W5(a):** stage `a8_w5a_ve_glitter.sh` (VE25 / VE100, seeds 0-2). It comes after a6 and a7, as ordered.
- **W5(b):** `w5b_fatal_deadly.py` plus stage `b0_w5b_fatal_deadly.sh`.
  - The recipe's best.pt on its 512 development games, seeds 0-4, using risk_localise.py's replay.
  - Seeds 0-2 are checked per game; seeds 3-4 stored only their mean return, so their replay is checked against that mean (risk_localise.py header 6).
  - Held-out seeds 5-9 would need an analysis-seed option in risk_localise.py: say if wanted.
- **W5(c) DONE** (`2798eb3`): the legend reads "exact P"; `reports/2026-11-03_paper_figures/localisation.pdf` is regenerated, ready to upload.
- **W5(d) DONE** (`3a513da`): `dev.schedule.<arm>.{cell,heading}_pct[.seed<s>]` for buffer, growing, pool, reset1, reset3, agent_best and agent_latest. All are log-sourced (`logs/data_vs_objective/`); the ledger is 225 / 225 OK.
  - Means, cell / heading: buffer 85.2 / 87.2; growing 76.6 / 71.9; pool 79.4 / 68.1; reset1 80.2 / 76.3; reset3 76.8 / 68.1; agent_best 75.1 / 63.6; agent_latest 76.2 / 68.3.
  - Note: the 2e header says agent latest cell 76.3; the per-seed values give 76.2.
- **W9 DONE** (`2fbb7b6`): `sweep.py` section 6, check_report_settings. All 74 stored reports pass, and a wrong analysis seed raises.
  - Reports older than the output field (2026-10-19) or the checkpoint field (2026-10-27) are checked as "analysis" / "best.pt", the only value those fields could have had then.
- **W4, W7:** in progress (code on CPU next). W7 needs the probe datasets rebuilt on the held-out split, which takes GPU time; it will be a stage.
- **W6:** not ordered, noted.

## W1. Slow suite at current main
- QUEUED as stage `a4_w1_slow_suite.sh`, after a3 (about 17:30 on 2026-11-14).
- Log: `logs/slow_suite_<date>.log`, with the commit on its first line.

## W2. Reproduction check
- QUEUED as stage `a5_w2_reproduction.sh`.
- Wumpus: held-out recipe seed 5 is re-analysed (`best.pt`, analysis seed 20261019) into `runs/sweeps/wumpus_retrain_heldout/seed5/analysis_repro_w2/`.
- Tiger: polyak seed 0 is copied (config and `best.pt`) to `runs/repro_w2/tiger_polyak_seed0/` and re-analysed there with analysis seed 12345. `analyze.py` has no output option for Tiger, so the stored analysis is never overwritten.
- The comparison is `repro_check.py`. It checks every array of `games.npz` exactly and the Tiger `report.json` returns block exactly, and prints EQUAL or NOT EQUAL for each.
- Log: `logs/reproduction_<date>.log`.

## W3. Model-free baseline, third screen
- CODE DONE (`5eb9813`). Two default-off options in `src/ipomdp/baselines/recurrent_dqn.py` header 5:
  - `exploration=anneal`: the planner's schedule. One epsilon for all rows, 0.25 in the first trained collection, times 0.97 after each trained collection, floor 0.1.
  - `value_scale=linear`: targets and Q-values divided by 1000. This is a linear map, so the decoded value is unbiased. CHOICE: the linear map, not two-hot; it changes only the biased transform and leaves the network and loss as they are. Ask for two-hot if you prefer it.
  - The defaults are bit-identical, checked by a training run against a HEAD worktree. New tests are in `tests/test_baselines.py`.
- Update count corrected to 18,432 in header 2 and in a PROTOCOL.md change-log note.
- REGISTERED as PROTOCOL.md C-n:
  - Arms: N1 `r2d2_n1` (anneal), N2 `r2d2_n2` (linear), N3 `r2d2_n3` (both), against `r2d2_lr1e-4_n5` on development seeds 0-2.
  - Rule: as in C-f.
  - Held-out seeds 5-9 run only if an arm is chosen.
- QUEUED as stage `a6_r2d2_c_n.sh`, about 9 runs of 15-20 min.

## W4. Accepted-state control for the localisation
- PENDING. Starts after a3, because `feature_localise.py` is in use by the C-m screen.

## W5. Small measurements
- PENDING: (a) VE25 and VE100 `feature_localise` runs; (b) the share of fatal FORWARDs taken into cells with exact danger >= 0.999, from the stored risk rows; (c) a legend change in `localisation.py` and a regenerated PDF.

## W6. Update-matched no-reset arm: do the screens in objective_ab 2e-2h separate re-initialisation from extra updates?
ANSWER: only partly, and only at the level of the latent (probes), not the return. No existing screen trains extra whole-buffer updates on networks that are not re-initialised, so the proposed arm is not answered by stored data.

- **Equal update counts (2e, 2f; `runs/wumpus/data_vs_objective/`, seeds 0-2, 6400 updates in every arm, all from a fresh initialisation):**
  - Uniform training on the agent's final buffer decodes cell / heading 85.2 / 87.2% (means over seeds).
  - The agent's growing-buffer schedule replayed offline gives 76.6 / 71.9%.
  - The agent itself (`latest.pt`, also 6400 updates) gives 76.3 / 68.3%.
  - So at a matched count, how the updates sample the data (uniform over the whole buffer vs growing) explains the latent gap. The extra updates are not needed for it.
- **Re-initialisation at a matched total budget (2g):**
  - One reset at update 3200 plus a 3200-update catch-up on the games available then gives 80.2 / 76.3%.
  - Three resets (1600 / 3200 / 4800, each with a 1600-update catch-up) give 76.8 / 68.1%.
  - Re-initialising alone does not recover the latent; it lowers the head's held-out NLL (0.30-0.31 vs 0.35). The latent gain needs long, uniform training of the final network on the full buffer.
- **Return (2h, `retrain_play.py` section 4; `runs/wumpus/retrain_play/`):**
  - A fresh whole-buffer retrain (6400 updates on the final buffer) against the agent's own `best.pt` gives +66.5 / -0.6 / +88.0 per game.
  - This compares re-init plus uniform whole-buffer training against the agent's growing schedule. It does not separate re-init from the extra updates.
- **What is missing:** two 6400-update uniform whole-buffer phases at collections 50 and 100 WITHOUT re-initialising the networks or the optimiser. That is the arm W6 proposes. The equal-count evidence above predicts it would recover most of the latent gain (the sampling, not the reset, matters), but there is no return measurement. Your call whether to order it (about 15 GPU-hours, C-e arm 4 on held-out seeds 5-9).

## C-m (the WSL session's own screen, a3)
- RUNNING, expected about 17:30 on 2026-11-14. Results go to PROTOCOL.md "C-m result" and to this file.
- The four diagnostics scripts (`feature_localise.py`, `risk_localise.py`, `retrain_play.py`, `objective_ab.py`) are free for audit edits once C-m is read.
