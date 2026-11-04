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
- Later method claims (for example, a new objective against this recipe) must be added to this file **before** their held-out runs start. Each states its comparison, its interval and its decision rule.

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

## Change log
Any change after the first held-out read is logged here with its date and reason. A change makes the claims it affects exploratory.

- No change to any claim. Operational note: held-out training seed 7 was aborted at collection 19 on 2026-10-19 by the VRAM guardrail (a diagnostic job beside it, not a property of the recipe). Before any held-out read it was moved to `runs/wumpus/heldout_seed7_aborted_20261019` and retrained from scratch with the identical command (training is deterministic per seed).
- Operational note (2026-11-02): C-e arm 2 (`abl_noretrain`) training seed 5 was aborted at collection 20 on 2026-11-01 22:31 by the VRAM guardrail (an unrelated process on the GPU took ~12 GB during the run; not a property of the arm). Before any read of the arm it was moved to `runs/wumpus/abl_noretrain_aborted_20261101` and the arm restarts from scratch with the identical command (training is deterministic per seed).
