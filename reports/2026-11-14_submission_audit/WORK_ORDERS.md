<!-- ABSOLUTE PATH: reports/2026-11-14_submission_audit/WORK_ORDERS.md -->

# Work orders W1-W6: status, artifacts and answers

The audit session issued these orders on 2026-11-14 (the user confirmed the roles: the audit decides, the WSL jepa session implements and runs). The audit session closed before it could read the replies, so they are kept here. The newest status is at the top of each entry.

Standing convention: every readout prints the per-seed values, the Student-t 95% interval on the per-seed (paired) means, the registered hierarchical bootstrap, and `crossed_bootstrap` (`aggregate.py` section 7).

GPU work runs through the week queue (`runs/week/stages.d/`, one stage at a time, gate `runs/week/queue.sh`). The order is a3 (C-m, the ego-map screen) → a4 (W1) → a5 (W2) → a6 (W3 screen); the later items follow.

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
