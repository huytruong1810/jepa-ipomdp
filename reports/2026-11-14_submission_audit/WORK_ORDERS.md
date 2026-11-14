<!-- ABSOLUTE PATH: reports/2026-11-14_submission_audit/WORK_ORDERS.md -->

# Work orders W1-W6: status, artifacts and answers

The audit session issued these orders on 2026-11-14 (the user confirmed the roles: the audit decides, the WSL jepa session implements and runs). The audit session closed before it could read the replies, so they are kept here. The newest status is at the top of each entry.

Standing convention: every readout prints the per-seed values, the Student-t 95% interval on the per-seed (paired) means, the registered hierarchical bootstrap, and `crossed_bootstrap` (`aggregate.py` section 7).

GPU work runs through the week queue (`runs/week/stages.d/`, one stage at a time, gate `runs/week/queue.sh`). The order is a3 (C-m, the ego-map screen) → a4 (W1) → a5 (W2) → a6 (W3 screen); the later items follow.

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
