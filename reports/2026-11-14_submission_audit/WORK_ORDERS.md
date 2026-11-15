<!-- ABSOLUTE PATH: reports/2026-11-14_submission_audit/WORK_ORDERS.md -->

# Work orders W1-W10: status, artifacts and answers

## ORDER W11 (issued 2026-11-15 by the audit session; decision: the headline goes to ten held-out seeds; no world-model baseline)

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
