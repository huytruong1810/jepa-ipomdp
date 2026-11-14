<!-- ABSOLUTE PATH: reports/2026-11-14_submission_audit/AUDIT_LOG.md -->

# Submission audit log

Frozen starting point: tag `audit-freeze-2026-11-14`. One entry per finding. Status is OPEN (found, not yet changed), FIXED (code or paper changed; the entry says where) or CLOSED (checked, nothing to change). `ledger.py` recomputes every number named here from the raw artifacts listed in `MANIFEST.sha256`; `statistics.md` holds the per-seed values and the three intervals of every held-out contrast.

## A. Numbers

| # | Status | Finding | Evidence | Change |
|---|---|---|---|---|
| A1 | FIXED | The interval of an agent played once per analysis seed is sqrt(5) too narrow: the nested bootstrap redraws the same 512 games independently for each of the 5 seed slots. Exact-belief planner: paper 494 [476, 511]; game-level [454, 533]. Random and heuristic rows are affected the same way (not printed in the paper). | `statistics.md`, rows `heldout.{exact_planner,heuristic,random}.return` | `aggregate.py` section 7: `crossed_bootstrap`, `game_metric_rows`; three tests in `tests/test_experiments.py`. `hierarchical_bootstrap` is unchanged so registered readouts reproduce. Paper: `tab:wumpus` gives game-level intervals for the three untrained agents; `ladder.py` and the figure follow. |
| A2 | FIXED | "the model predicts glitter in the new cell between 0.33 and 0.60 times as often as the exact belief": that range belongs to the control retrains of the data screen. The agents the paragraph localises give 0.22 to 0.44 (0.44, 0.29, 0.30, 0.22, 0.34). The Discussion repeats it as "between a third and three fifths". | ledger `dev.declined.glitter_ratio.{min,max}` | paper (2026-11-14): 0.22 to 0.44; Discussion "between a fifth and less than a half" |
| A3 | FIXED | Development exact-model gain "+208 +- 48 (95% interval over 5 seeds)": the Student-t half-width of the five paired means is 50.6. | ledger `dev.exact_model_minus_learned_ci` | paper (2026-11-14): +208 +- 51 (Student-t) |
| A4 | FIXED | `tab:remedies`, union row: deaths "+8.4" points; the arm minus its control is +6.7 (13.7% against 7.0%). | ledger `dev.remedy.union.deaths_mean` | paper (2026-11-14): +6.7 |
| A5 | FIXED | `tab:remedies`, spread-weighted likelihood: glitter "lower on 2 of 3". Against the control its returns are paired with, it is lower on 1 of 3 (-0.10, +0.02, +0.07); the "2 of 3" compared it with the data screen's control, a different network. | ledger `dev.remedy.DA50.glitter_change.seeds_lower` | paper (2026-11-14): "lower on 1 of 3"; the text gives the range -0.10 to +0.09 against the matching control |
| A6 | FIXED | `tab:remedies`, value-equivalence row: "unchanged" glitter ratio has no stored measurement for either weight; the row reports the weight 25 only (weight 100: -64 / +31 / +12). | no `feature_localise_VE*` file; ledger `dev.remedy.VE100.seed*` | paper (2026-11-14): cell reads "not measured"; the text says so and gives the larger weight's returns |
| A7 | FIXED | `tab:remedies`, agent's own trials: glitter change "+0.00 to +0.24" is +0.01 to +0.24. | ledger `dev.remedy.own.glitter_change.min` | paper (2026-11-14): +0.01 |
| A8 | FIXED | Tiger geometry (figure and text) is computed on held-out random-policy histories, not "the agent's own histories". | `interpretability/analysis.py`, `geometry=latent_geometry(data["random"][1])` | paper (2026-11-14): caption and text say "histories of a uniformly random policy"; script header corrected |
| A9 | FIXED | Tiger probe sentence joins two probes. The L1 error 0.004 and the 0.14% of suboptimal decisions belong to the probe fitted on the planner's histories and read on the planner's held-out histories. The decoded-belief agent, whose return is -0.20 +- 0.24 below the optimum, acts on the probe fitted on random histories; the 0.14% is not a rate of that agent's decisions. Same for the linear probe (0.016 and "loses 4.7"). | `interpretability/analysis.py`, `bounds` and `returns["decoded_*"]` | paper (2026-11-14): the sentence is split: probe accuracy and the 0.14% on the planner's histories; the decoded-belief agent with its random-history probe |
| A10 | FIXED | "19.36" (caption: the solver's value at b0, tolerance 0.01, certified error 0.0096) and "19.37" (setup text: the published value) appear under the same description. | ledger `tiger.optimal.value_b0`; `tests/test_domain.py` reference constant | paper (2026-11-14): caption says the solver's vectors give 19.36, within their certified 0.01 of U(b0); it also states that all seeds share the 512 games |
| A11 | FIXED | "the worst-case forms, 24 and 48" are means over seeds of each seed's worst case (largest: 43 and 86). Part (iii) of the decoding proposition is promised "as an indicative figure" and never given (mean 9.5). | ledger `tiger.bounds.*` | paper (2026-11-14): "24 and 48 on average over seeds, at most 43 and 86"; form (iii) reported as 9.5 and called uninformative |
| A12 | FIXED | Wumpus rewards: the step cost is charged on the last step too, so a win pays +999 and a death -1001; the two-hot range uses 1001. The text gives +1000 and -1000 and "rewards range from -1000 to +1000". "|Omega| = 33 observations together with a terminal observation" can be read as 34. | `domain/wumpus.py` reward table and `value_bound` | paper (2026-11-14): +999 / -1001 stated; "from -1001 to +999"; 32 percept combinations plus a terminal observation |

All other Tiger, held-out, ablation and localisation numbers (170 of 176 ledger claims at the start of the audit) recompute to the printed values. After the paper changes of 2026-11-14 the ledger holds 207 claims and all recompute.

## B. Method text against code

| # | Status | Finding | Evidence | Change |
|---|---|---|---|---|
| B1 | FIXED | The Wumpus "self-predictive (JEPA) representation" ablation keeps the 5-step open-loop observation predictor, whose loss trains the filter in either representation. The arm replaces only the one-step observation gradient by latent self-prediction; it is not a latent "trained to predict its own future latent" alone. Tiger's self-predictive runs have no observation term in the filter. | `runs/sweeps/abl_jepa/sweep.json` (`model.observation_horizon=5`); `training/trainer.py` header 2c and the multi-step term | paper (2026-11-14): section 4.1 states what the Wumpus arm keeps; the ablation row is "Latent self-prediction as the one-step term" |
| B2 | FIXED (text); a matched-update control is still a candidate run | The no-retraining ablation also removes 12,800 of the recipe's 19,200 optimiser updates (two retrains of 6,400). | hyperparameter table; `baselines/recurrent_dqn.py` budget note | paper (2026-11-14): the Ablation paragraph states 6,400 against 19,200 updates and that the arm does not separate the two |
| B3 | OPEN | Not stated in the method: the evaluated networks are those of the best in-training 128-game evaluation; the heuristic's risk threshold is 0.4; the uniform-action probability decays by 0.97 per collection; the value is the infinite-horizon value while games are scored on a 50-step window; the model-free baseline replays 32 episodes per update against 16. | saved run configs; `PROTOCOL.md` frozen choices | paper text |
| B4 | OPEN | The offline flip shares (right panel of the localisation figure) value exact game-over children at 0; the played exact-model arm (left panel) keeps the learned value there. The text describes one substitution. | headers of `q_localise_diag.py` and `exact_model_play.py` (3a) | paper text |

## C. Statistics

| # | Status | Finding | Evidence | Change |
|---|---|---|---|---|
| C1 | FIXED | The registered interval (percentile bootstrap over 5 seed clusters) is narrower than the Student-t interval on the five per-seed means, by the expected factor of about 0.63 to 0.7. Two statements depend on it: "no whole-buffer retraining: +40 [+3, +88], contributes" (t: [-24, +104]; crossed: [-3, +92]) and "10 simulations: -22 [-43, -3]" (t: [-46, +2]; crossed: [-46, 0]). The exact one-step-model gain, the even mixture and the gap to the model-free baseline exclude zero under all three. | `statistics.md` | paper (2026-11-14): Evaluation Protocol paragraph, `tab:wumpus`, `tab:ablation`, Ablation and Localisation paragraphs use the Student-t interval with the registered one marked "boot."; no effect of retraining on the return is claimed; `PROTOCOL.md` has a deviations section |
| C2 | FIXED | All seeds are analysed on the same 512 games; the nested bootstrap treats games as nested in seeds (see A1). | `aggregate.py` section 7 | as A1 |
| C3 | FIXED | The evaluation protocol paragraph promises the interquartile mean over seeds; no table or sentence gives one. The `tab:wumpus` caption says the untrained agents have a single value while one of them has an interval. | paper | paper (2026-11-14): the interquartile-mean promise is removed; the caption describes both kinds of interval |

## E. Hygiene

| # | Status | Finding | Change |
|---|---|---|---|
| E1 | OPEN | The slow acceptance suite last ran on 2026-10-19. | rerun at the audit's final commit |
| E2 | OPEN | `README.md` describes the September state (Tiger only). | rewrite |
| E3 | FIXED | Several paper numbers existed only in logs that git does not track. | copied to `logs/`; `ledger.py` recomputes the rest from raw artifacts |
| E4 | FIXED | `reports/2026-11-03_paper_figures/localisation.py` hard-codes its 35 values. | `localisation.py` and `ladder.py` read the raw per-game files and logs; figures regenerated |
