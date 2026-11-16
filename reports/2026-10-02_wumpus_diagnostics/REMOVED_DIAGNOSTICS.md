<!-- ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/REMOVED_DIAGNOSTICS.md -->
# Diagnostic scripts removed on 2026-11-15, with what each recorded

These fifteen scripts were one-off diagnostics of October 2026. No paper statement, ledger value, PROTOCOL.md result, queue stage, test or other script uses them. They were removed to keep this folder to the scripts that back reported results. Their questions and recorded results are kept here. Each can be recovered with the command given.

How to read the entries:

- "Recorded result" holds only what the script's own header comment stated. Numbers are copied as written. Where a header stated no result, the entry says so.
- Every script is held by commit `bc95524` and by the tag `audit-freeze-2026-11-14` (both checked with `git cat-file -e` on 2026-11-15). The tag form of the recovery command is `git show audit-freeze-2026-11-14:reports/2026-10-02_wumpus_diagnostics/<name>.py`.
- Run directories under `runs/` are not tracked in git. Those named below existed on the working machine on 2026-11-15.

## danger_probe.py
- **Question:** (2026-10-02) Is the danger ahead decodable from the latent? The observation head's P(GAME_OVER | z, FORWARD) barely separates provably safe steps from certain death (premise taken from `death_calibration.py`). If a fresh MLP probe decodes the exact danger ahead from the same latents, the head (its training signal) is the bottleneck. If it cannot, the latent is.
- **Method:** Reference-agent (risk 0.4) and uniform-random games are pooled. An MLP probe is fitted on one set of games (seed 4321) and scored on another (seed 8765) against the exact danger ahead from the enumeration filter: mean absolute error and binned calibration, next to the observation head's prediction and a constant baseline. Usage: `uv run python <this> <run dir> <games> <checkpoint name>`.
- **Recorded result:** No result recorded in the header.
- **Runs / logs it names:** None. The run directory and checkpoint are command-line arguments.
- **Recover:** `git show bc95524:reports/2026-10-02_wumpus_diagnostics/danger_probe.py`

## death_calibration.py
- **Question:** (2026-10-02, after the off-policy diagnostic made the planner die more) Does the observation head predict death correctly? The exact probability that FORWARD kills is P(pit ahead) + P(live wumpus ahead) from the enumeration filter (0 at a wall). The model's prediction is P_hat(GAME_OVER | z, FORWARD) from the acting observation head.
- **Method:** States come from reference-agent (risk 0.4) and random games on the analysis seed, alive states only. Per exact-danger bin it reports the mean exact danger, the mean predicted P(GAME_OVER), and the learned one-step Q(FORWARD) - Q(TURN_LEFT) (positive means the model prefers moving). Usage: `uv run python <this> <run dir> <games> <checkpoint name, e.g. best.pt>`.
- **Recorded result:** No result recorded in the header. (The header of `danger_probe.py` cites this script for the statement that the head's P(GAME_OVER | z, FORWARD) "barely separates provably safe steps from certain death"; no numbers are given in either header.)
- **Runs / logs it names:** None. The run directory and checkpoint are command-line arguments.
- **Recover:** `git show bc95524:reports/2026-10-02_wumpus_diagnostics/death_calibration.py`

## greedy_deaths.py
- **Question:** (2026-10-05, combined-fix run `eps_decoder_k5_u256_seed0` dying in ~50% of games) `death_calibration.py` scores the head on reference and random histories. How does the head score on the greedy planner's own histories, only at the FORWARD steps it actually takes?
- **Method:** The run's evaluation agent plays `eval_episodes` games on a fixed seed. At each FORWARD step taken, the exact danger (enumeration filter) is compared with the acting observation head's P(GAME_OVER | z, FORWARD), by exact-danger bin. Usage: `uv run python <this> <run dir> <checkpoint name>`.
- **Recorded result:** Result 2026-10-05 (latest.pt, ~collection 40): 28 FORWARDs at exact danger >= 0.6 and 55 at 0.4-0.6, all predicted at P(death) <= 0.01; death rate 0.47. The planner steps exactly where the head wrongly says "safe" (optimizer's curse on a miscalibrated head). Older runs show the same at a smaller scale (`north50_decoder_seed0` 34, `north50_seed0` 12 such steps).
- **Runs / logs it names:** `runs/wumpus/eps_decoder_k5_u256_seed0` (latest.pt), `runs/wumpus/north50_decoder_seed0`, `runs/wumpus/north50_seed0`. The header gives the run names only; the `runs/wumpus/` prefix is where they are stored.
- **Recover:** `git show bc95524:reports/2026-10-02_wumpus_diagnostics/greedy_deaths.py`

## start_danger.py
- **Question:** (2026-10-05) Does the head use o_0 for the first step's danger?
- **Method:** 4096 fresh layouts; FORWARD from the start cell. By start percept it compares the empirical death rate with P(GAME_OVER | z_0(o_0), FORWARD), and prints V_hat(z_0). Usage: `uv run python <this> <run dir>:<checkpoint> [...]`.
- **Recorded result:** Result 2026-10-05: yes, in every run (`eps_decoder_k5_u256_seed0`: breeze 0.53 -> 0.71, stench 0.50 -> 0.73, both 1.00 -> 1.00, quiet 0 -> 0.001). The fatal misses are mid-game. V_hat(z_0) is 1490-1670 in that run, above the largest achievable return (999).
- **Runs / logs it names:** `runs/wumpus/eps_decoder_k5_u256_seed0` (run name only in the header). The other runs covered by "every run" are not named.
- **Recover:** `git show bc95524:reports/2026-10-02_wumpus_diagnostics/start_danger.py`

## reference_probe.py
- **Question:** (2026-10-02) The agent-history probes (`analyze.py`) look good, but the planner's own games rarely travel far. Does the latent still know the hunter's cell and heading, and the danger ahead, on histories that do travel (the reference agent's)?
- **Method:** Reference-agent (risk 0.4) games are replayed through the acting filter. An MLP probe is fitted on one half of the games and evaluated on the other. Hazard errors are printed next to a constant baseline, and cell and heading accuracies next to their majority baselines. Usage: `uv run python <this> <run dir> <games> <checkpoint name>`.
- **Recorded result:** No result recorded in the header.
- **Runs / logs it names:** None. The run directory and checkpoint are command-line arguments.
- **Recover:** `git show bc95524:reports/2026-10-02_wumpus_diagnostics/reference_probe.py`

## sims_diag.py
- **Question:** (scratch diagnostic of 2026-10-02, kept so the reported numbers are reproducible) How does the greedy evaluation change with a different number of simulations?
- **Method:** The run's greedy planner is rebuilt with the given simulation count and plays on the same analysis simulator stream as `analyze.py`. It prints the mean discounted return with its standard error, the game outcomes and the action fractions as JSON. Usage: `uv run python <this> <run dir> <games> <simulations>`.
- **Recorded result:** No numbers recorded in the header. The header states only that this script is the source of the statement "search depth is not the bottleneck".
- **Runs / logs it names:** None. The run directory is a command-line argument.
- **Recover:** `git show bc95524:reports/2026-10-02_wumpus_diagnostics/sims_diag.py`

## split_diag.py
- **Question:** (scratch diagnostic of 2026-10-02, kept so the reported numbers are reproducible) How do the planner and the reference agent compare when games are split by the start percept? A quiet start (o_0 = 0) makes the first step provably safe, so the split separates idling under real danger from failing to explore.
- **Method:** The greedy planner and the reference agent (risk 0.4) play the same games. For quiet starts and starts with a percept it prints the return and the win, death and unfinished rates. For the planner it also prints the share of games with no FORWARD at all and the action fractions. Usage: `uv run python <this> <run dir> <games>`.
- **Recorded result:** No result recorded in the header.
- **Runs / logs it names:** None. The run directory is a command-line argument.
- **Recover:** `git show bc95524:reports/2026-10-02_wumpus_diagnostics/split_diag.py`

## hoist_forward_check.py
- **Question:** (diagnostic of 2026-10-04/05, kept so the reported numbers are reproducible) Is `BeliefFilter.unroll` with `input_net` hoisted out of the per-step loop equivalent in the forward pass to the per-step loop, and how much faster is it?
- **Method:** Both variants are run on Tiger and Wumpus sizes, with and without bf16 autocast. It checks bit-identity, prints the largest difference and times 20 calls of each.
- **Recorded result:** Bit-identical under autocast; 4.5-7.6x faster. The header adds that the script was written before the hoist was committed, that today's `unroll()` IS the hoisted version, and that reproducing needs a rerun "at commit 62deb68".
- **Runs / logs it names:** None. It names commit `62deb68`. That hash does not resolve in this repository (checked 2026-11-15). The commit that introduced the hoist here is `a1d7696` ("perf(model): hoist the filter's input network out of the unroll loop (4.5-7.6x faster unroll)"), so the per-step loop is in its parent, `a1d7696~1`.
- **Recover:** `git show bc95524:reports/2026-10-02_wumpus_diagnostics/hoist_forward_check.py`

## hoist_training_check.py
- **Question:** (diagnostic of 2026-10-04/05, kept so the reported numbers are reproducible) Does training give the same fingerprint with the per-step unroll and with the hoisted unroll?
- **Method:** Two collections on Tiger and on Wumpus with each unroll variant (monkeypatched; `training.warmup_episodes=128`, `training.updates_per_collection=8`). It compares the metrics and the parameter sums.
- **Recorded result:** Result (2026-10-04): not bit-identical (parameter sums differ by 0.004 on Tiger and 0.025 on Wumpus after 16 updates; gradient summation order).
- **Runs / logs it names:** None.
- **Recover:** `git show bc95524:reports/2026-10-02_wumpus_diagnostics/hoist_training_check.py`

## head_coverage.py
- **Question:** (2026-10-05) Is the observation head's death prediction limited by data coverage? With longer training the latent nearly reaches the supervised bound, but the head's P(GAME_OVER | z, FORWARD) stays badly calibrated (`objective_ab_long`). Hypothesis: the head learns it only from sampled outcomes, and dangerous FORWARD steps are rare in the data (the reference agent avoids them, random hunters die early).
- **Method:** The latent is held fixed: the agent's `BeliefFilter` architecture is trained with privileged supervision (`supervised_filter.py` targets, `--filter-steps` updates), then frozen. A fresh `ObservationHead` is fitted with cross-entropy on detached latents in two arms with the same steps and minibatch size: (a) NORMAL, every (z_t, a_t, o_{t+1}) transition of the training pool; (b) AUGMENTED, (a) plus one forced FORWARD transition from every alive state of the pool's reference games, its outcome taken from the true rules (extra sampled experience, no privileged label). Score on held-out games: P_hat(GAME_OVER | z, FORWARD) against the exact danger ahead (mean error, and the means in the safe and certain-death bins).
- **Recorded result:** No result recorded in the header.
- **Runs / logs it names:** `runs/wumpus/supervised_filter/pools.pt` (the pool cache in the usage line); `objective_ab_long` (named as the source of the premise). The script prints its numbers as `RESULT normal_head_*` and `RESULT augmented_head_*` lines; the header names no log file that holds them.
- **Recover:** `git show bc95524:reports/2026-10-02_wumpus_diagnostics/head_coverage.py`

## noop_drift_diag.py
- **Question:** (2026-10-15, conjecture C4, representation plan R0/R1) The exact belief is unchanged by a GRAB without GLITTER or a SHOOT without the arrow (both decidable from the history alone). Does the acting filter's latent stay put there, and does it differ between the current and the oracle recipe?
- **Method:** The 128 greedy games of best.pt stored in the buffer are replayed through the acting filter (CPU, ~7 s). It measures the relative drift |z_{t+1} - z_t| / |z_t| on live no-op steps against other live steps, and the acting head's P(GAME_OVER | z, FORWARD) < 0.02 bin with its deaths. These are the head's own training games, so that bin is expected to be calibrated (`head_shift.py` item 3). Usage: `uv run python <this> <run dir>`.
- **Recorded result:** Result 2026-10-15 (the six double-value runs): drift on no-ops mean 0.217 / 0.117 / 0.098 (current seeds 0-2) vs 0.030 / 0.023 / 0.021 (oracle), against 0.47-0.81 on other actions in both; 0-1 confident "safe" deaths per network. Whether the drift costs information: `noop_drift_cost.py` (it does not).
- **Runs / logs it names:** "The six double-value runs" (current recipe seeds 0-2 and oracle recipe seeds 0-2); the header gives no directory names. It refers to `noop_drift_cost.py`, `head_shift.py` and `phantom_reward_diag.py`, which stay in this folder.
- **Recover:** `git show bc95524:reports/2026-10-02_wumpus_diagnostics/noop_drift_diag.py`

## overfit_summary.py
- **Question:** None of its own. It summarises the overfitting A/B: `objective_ab.py` logs with learning curves.
- **Method:** It reads logs named `obj_<arm>_seed<n>.log` holding `RESULT@<updates> name value` lines and final `RESULT name value` lines. Per arm and update count it prints the mean over seeds with a Student-t 95% half-width, and each arm's paired difference against the baseline arm (the first in sorted order) at the same update count and seed, with its own t interval. Metrics: head_safe_miss, head_danger_error, head_safe, head_certain_death, head_nll_train, head_nll_held, probe_danger_error, probe_cell_acc, probe_heading_acc. Usage: `uv run python <this> <final update count> runs/wumpus/overfit_ab/obj_*.log`.
- **Recorded result:** No result recorded in the header.
- **Runs / logs it names:** `runs/wumpus/overfit_ab/obj_*.log`.
- **Recover:** `git show bc95524:reports/2026-10-02_wumpus_diagnostics/overfit_summary.py`

## ab_summary.py
- **Question:** None of its own. It summarises the A/B of `supervised_filter.py` logs (statistics rule of 2026-10-03: A/B with good statistics).
- **Method:** One log per (arch, seed). Seeds differ only in initialisation and minibatch order and the data pools are identical, so the arms are compared per seed (paired difference B - A with its own interval) as well as by arm means with Student-t 95% intervals. It also reads `objective_ab.py` logs named `obj_{arm}_seed{n}.log` through their `RESULT name value` lines. With more than two arms, every arm is paired against the first in sorted order.
- **Recorded result:** No result recorded in the header.
- **Runs / logs it names:** `runs/wumpus/supervised_filter/ab_{arch}_seed{n}.log`; `obj_{arm}_seed{n}.log` logs of `objective_ab.py`.
- **Recover:** `git show bc95524:reports/2026-10-02_wumpus_diagnostics/ab_summary.py`

## offpolicy_reference.py
- **Question:** (2026-10-02) The baseline planner (`runs/wumpus/north50_seed0`) has a good latent but wanders instead of exploring new safe cells. Is the bottleneck the data its own exploration produces, or the learning (value through the learned model)? If the greedy planner trained on directed-exploration data approaches the reference agent, the data/exploration is the bottleneck; if it stays near the baseline, value learning is.
- **Method:** The unchanged stack is trained with the baseline's resolved config and seed, for 100 collections, where each collection is half reference-agent games (risk 0.4) and half uniformly random games; the planner never collects. Greedy evaluation and best.pt selection are as in the baseline. It is a diagnostic only: the reference agent uses the privileged exact belief, so this is never a training method for the agent.
- **Recorded result:** No result recorded in the header. (The header of `death_calibration.py` says its question was asked "after the off-policy diagnostic made the planner die more"; no numbers are given in either header.)
- **Runs / logs it names:** Baseline `runs/wumpus/north50_seed0`; output run directory `runs/wumpus/offpolicy_reference_seed0` (with `.hydra/` copied from the baseline, `checkpoints/best.pt`, `checkpoints/final.pt` and `main.log`, which holds the evaluation lines).
- **Recover:** `git show bc95524:reports/2026-10-02_wumpus_diagnostics/offpolicy_reference.py`

## absorbing_value.py
- **Question:** (2026-10-05, combined-fix run `eps_decoder_k5_u256_seed0`) The greedy agent died in 40-60% of evaluation games through collection 40 and the mean Bellman value target rose to 800-1550, above the largest achievable return (999). Does the value head also rate the absorbing GAME_OVER state highly? If so, dying is worth -1001 + gamma * V_hat(dead) to the planner, and that can beat playing on. The exact value of any state after the game ended is 0 (absorbing, reward 0).
- **Method:** Uniform-random games on a fixed seed, latents from the acting filter; V_hat is the codec mean of the acting value head. It reports V_hat by status (alive / after death / after a win) and the one-step learned Q(a) at alive states. Usage: `uv run python <this> <run dir> <games> <checkpoint name>`.
- **Recorded result:** Result 2026-10-05: the absorbing state is NOT the problem (V_hat after death 114, after a win 114; exact 0). Live states are overestimated and the overestimate grows with updates: mean V_hat alive 1565 (`eps_decoder_k5_u256_seed0`, 256 updates/collection, ~collection 40) vs 868 (`north50_decoder_seed0`, 64/collection, collection 100) and 559 (`north50_seed0` best.pt); the largest achievable return is 999 and the reference agent earns 573.
- **Runs / logs it names:** `runs/wumpus/eps_decoder_k5_u256_seed0`, `runs/wumpus/north50_decoder_seed0`, `runs/wumpus/north50_seed0` (best.pt). The header gives the run names only; the `runs/wumpus/` prefix is where they are stored.
- **Recover:** `git show bc95524:reports/2026-10-02_wumpus_diagnostics/absorbing_value.py`
