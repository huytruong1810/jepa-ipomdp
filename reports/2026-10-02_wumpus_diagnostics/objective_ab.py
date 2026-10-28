# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/objective_ab.py
# ==============================================================================
# OFFLINE OBJECTIVE A/B: WHICH REAL (NON-PRIVILEGED) OBJECTIVE MAKES THE LATENT A BELIEF?
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Question (design decision, 2026-10-03):
#    - The same GRU filter learns cell, heading and the danger ahead almost perfectly from
#      privileged supervision (supervised_filter.py), but not from the agent's objectives. Which
#      objective closes the gap? Arms: representation (jepa | decoder) x observation_horizon
#      (0 | k), the multi-step observation term of training/trainer.py, section 2c.
#
# 2. Protocol (fast, no planning, identical data for every arm):
#    - The REAL TrainingRun and WorldModelTrainer (value targets, Polyak acting copies and all)
#      built from the baseline's config with the arm's overrides and --seed.
#    - Training data: the cached pool of supervised_filter.py (reference-agent risk 0.4 and
#      random games, 50 steps), uniformly sampled minibatches of the config's batch_size; the
#      update count defaults to the full agent run's (100 collections x 64 updates).
#    - Scoring on the held-out pool, from the ACTING filter (what the planner uses):
#      a. probes on frozen latents (MLP, fitted on 1024 training games): hazard marginals,
#         cell, heading and the danger ahead (exactly the supervised_filter.py targets);
#      b. the planning model's death prediction: P_hat(GAME_OVER | z, FORWARD) from the acting
#         observation head against the exact danger ahead, overall and in the safe and
#         certain-death bins. This is what the planner's step decisions depend on.
#    - Output: "RESULT name value" lines after training (parsed by ab_summary.py), plus
#      "RESULT@<updates> name value" lines at the --score-at update counts (learning curves).
#
# 2b. Overfitting A/B (design decision, 2026-10-06; head_shift.py found the jointly trained filter and
#    observation head calibrated in-sample but overconfident on fresh games):
#    - Arms: --weight-decay (AdamW, every trained network) and --train-games (the first n games of
#      each half of the training pool, so 2n games), plus the --score-at learning curve as the
#      early-stopping lever.
#    - Extra scores: the observation head's next-observation NLL on 1024 training games vs the
#      held-out pool (alive steps, every action; the generalisation gap), and head_safe_miss: the
#      mean exact danger at held-out states where the head predicts P(death | FORWARD) < 0.02 (the
#      confident "safe" errors the greedy planner exploits; 0 for a perfect head).
#
# 2c. Horizon A/B and saved networks (design decision, 2026-10-09: latent work after every value-side
#    remedy for the unfinished games failed):
#    - Arms: --horizon 5 (the current agent's recipe) against 10 and 20, decoder, 3 seeds each, with
#      the current agent's config (--config runs/wumpus/u64_store_eval_eps_decoder_k5_seed0/...).
#      Over longer action sequences the future BUMPs and percepts depend on position and heading, so
#      the multi-step term is the natural non-privileged pressure towards them.
#    - --save writes every network's state (run.networks, online and acting copies) after training,
#      so later diagnostics need no retraining.
#    - Result 2026-10-10 (runs/wumpus/horizon_ab/, summary.txt): LONGER HORIZONS DO NOT HELP, k = 20 HURTS.
#      Held-out MLP probes, cell / heading per seed 1-3: k5 77/63, 81/69, 81/84%; k10 77/60, 80/70, 78/62%;
#      k20 76/60, 76/56, 76/60%. k20 - k5: cell -3.8 +- 5.2 points (lower on 3/3 seeds), heading -13 +- 24
#      (lower on 3/3); wumpus error +0.0008 +- 0.0004; head held-out NLL +0.014 +- 0.019. Heading is still
#      rising at 6400 updates for k5 (seed 3: 55 -> 58 -> 69 -> 84% at 1600/3200/4800/6400).
#
# 2d. Decoupled planning head (design decision, 2026-10-10, after 2c: the latent's heading keeps
#    improving with more updates, but the jointly trained observation head overfits, section 2b):
#    - --planning-head: at every scoring point, a FRESH ObservationHead (the agent's architecture) is
#      fitted on the frozen acting latents of the training games, i.e. a head on detached latents,
#      and early-stopped on next-observation NLL over 10% of the training games (the held-out pool
#      stays untouched for scoring). Its scores are printed as "RESULT plan_<name>" next to the
#      joint head's, so one long run scores 6400 / 12800 / 25600 updates for both heads.
#    - A post-hoc fit is the screen; in the agent, the decoupled head would train alongside on
#      detached latents with its own stopping rule.
#    - Result 2026-10-10 (runs/wumpus/planning_head_ab/, seeds 1-3, 25600 updates; per seed at
#      6400 -> 12800 -> 25600 updates):
#      * The LATENT keeps improving with more updates: cell 77/81/81 -> 80/84/86 -> 79/85/87%, heading
#        63/69/84 -> 68/77/91 -> 69/77/92%, probe danger error 0.100/0.101/0.057 -> 0.091/0.090/0.030.
#        Most of the gain is in by 12800.
#      * The JOINT head overfits as before: held-out NLL 0.242/0.234/0.230 -> 0.386/0.364/0.319, and
#        safe-miss 0.019/0.020/0.020 -> 0.080/0.036/0.029.
#      * The DECOUPLED head (early-stopped at 2250-2750 steps) does not overfit: held-out NLL
#        0.248/0.239/0.221 at 25600. But against the joint head at the agent's 6400 it is no better on
#        danger error (12800: 0.115/0.111/0.090 vs 0.113/0.111/0.098) and WORSE on the confident "safe"
#        misses the planner exploits, on 3/3 seeds: safe-miss 0.023/0.072/0.038 vs 0.019/0.020/0.020
#        (paired +0.025 +- 0.062). Even on the same 6400 latent the fresh head misses more
#        (0.021/0.065/0.050): NLL early stopping does not make the head conservative where it matters.
#
# 2e. Data vs objective for dead reckoning (design decision, 2026-10-15 night, R1 of the representation plan):
#    - The agent's best.pt filters decode cell / heading 59-61% / 58-66% on the held-out random pool, whereas
#      this script's offline decoder k5 at the same 6400 updates reaches 77-81% / 63-84% (2c). Is the agent's
#      own data (idle-heavy greedy and exploring games) what limits dead reckoning, or its in-agent training?
#    - --train-buffer <run>: train on that run's replay buffer (latest.pt, every game, uniformly sampled) instead
#      of the pool. The probes are still fitted on the pool's training games and scored on the held-out pool,
#      so every arm is scored identically. Caveat: the agent trained on a GROWING buffer; uniform sampling of
#      the final buffer gives later games more weight than the agent saw.
#    - --agent-reference <run>: no training; the run's best.pt networks are loaded and scored the same way.
#    - Arms (3 seeds, paired by seed with u64_store_eval_eps_decoder_k5_seed{0,1,2}, config of seed 0, decoder,
#      k = 5, 6400 updates): pool (rerun, same code), buffer (that seed's run), agent reference.
#    - Pre-registered readout (before any run), on held-out probe cell and heading accuracy, means over seeds:
#      * DATA-LIMITED: the buffer arm is within 5 points of the agent reference AND >= 8 points below the pool
#        arm, on both cell and heading -> exploration / data arms next (e.g. a share of random-action games).
#      * TRAINING-LIMITED: the buffer arm is within 5 points of the pool arm AND >= 8 points above the agent
#        reference, on both -> the in-agent training limits it (growing buffer, Polyak, value-loss coupling);
#        ablate those next.
#      * Otherwise MIXED: report the share of the pool - agent gap that the data explain,
#        (pool - buffer) / (pool - agent), per probe.
#    - AMENDED before any training run (2026-10-15 night), after scoring the agent references with this
#      script (runs/wumpus/agent_reference_probes.txt): the 59-61% premise came from noop_drift_cost.py's harder
#      random-only half. On the full held-out pool the agents decode cell / heading 74-76% / 62-65% (current
#      recipe, 5 seeds; double runs 73-75% / 60-64%), against 77-81% / 63-84% for the pool arm (2c) and
#      97-100% / 98-100% for the oracle-recipe networks. The pool - agent gap (~5 / ~8 points) is too small for
#      the 8-point thresholds above, so the readout becomes: per probe, the share of the pool - agent gap the
#      data explain, (pool - buffer) / (pool - agent), means over seeds, plus the buffer arm against the oracle.
#      Whatever the share, the data can move the latent by at most the pool - agent gap, a fraction of the
#      25-35 points to the oracle latent; that bound is the main reading.
#    - Result 2026-10-15 night (runs/wumpus/data_vs_objective/, seeds 0 / 1 / 2; 6400 updates each):
#                      cell                  heading               probe danger error
#        buffer        0.845 / 0.838 / 0.874  0.917 / 0.781 / 0.919  0.031 / 0.088 / 0.036
#        pool          0.802 / 0.773 / 0.808  0.719 / 0.635 / 0.689  0.098 / 0.100 / 0.101
#        agent best    0.761 / 0.751 / 0.740  0.646 / 0.645 / 0.616  0.098 / 0.101 / 0.104  (collections 85/90/80)
#        agent latest  0.770 / 0.758 / 0.760  0.746 / 0.654 / 0.649  0.074 / 0.100 / 0.100  (6400 updates, as here)
#        oracle+double 0.999 / 0.995 / 0.965  1.000 / 1.000 / 0.982  0.023 / 0.025 / 0.034  (best.pt)
#      (pool seed 1 reproduces 2c's k5 seed 1 exactly.) The amended share is NEGATIVE: the agent's own buffer
#      is BETTER data than the pool. By the original criteria this is TRAINING-LIMITED: the buffer arm is above
#      the pool arm and 7-27 points above the agent at the same 6400 updates, with the same code (TrainingRun,
#      train_step, _update_acting) and the same games. Heading on seeds 0 and 2 (0.92) approaches the oracle's.
#      The remaining difference is the agent's data SCHEDULE (2f).
#
# 2f. Growing-buffer schedule (2026-10-15 night, after 2e): --buffer-schedule growing samples update u from the
#    games the agent had then: collection c = (u - 1) // 64 + 1 sees c * 64 + ((c - 1) // 5) * 128 games
#    (run.py: collect 64, 64 updates, then every 5th collection 128 stored greedy games), uniformly, as
#    EpisodeBuffer.sample does. Everything else is 2e's buffer arm.
#    - Pre-registered readout (before any run), cell and heading, means over seeds 0-2:
#      * SCHEDULE (primacy bias, Nikishin et al. 2022: early updates on few games leave a latent that later
#        data does not repair): the growing arm is within 5 points of agent latest AND >= 5 below the buffer
#        arm on both -> next an offline reset arm (growing schedule with network resets), then a default-off
#        reset option in the agent.
#      * NOT THE SCHEDULE: the growing arm is within 5 points of the buffer arm on both -> the difference is
#        elsewhere in the agent loop; diff TrainingRun.collect_and_train against this script.
#      * Otherwise PARTIAL: report the share of the buffer - agent gap the schedule reproduces.
#    - Result 2026-10-16 (runs/wumpus/data_vs_objective/growing_seed*.log): SCHEDULE. Cell 0.750 / 0.770 / 0.778
#      (mean 76.6 vs agent latest 76.3, buffer 85.2), heading 0.777 / 0.658 / 0.722 (mean 71.9 vs 68.3 and 87.2):
#      within 5 points of the agent and 8.6 / 15.3 below the buffer arm. Held-out NLL 0.34-0.35 vs 0.26-0.29
#      for the buffer arm: the early, repeatedly replayed games are overfitted and later data do not repair it.
#
# 2g. Resets with a catch-up (2026-10-16, after 2f; the primacy-bias remedy of Nikishin et al. 2022 adapted to
#    a fixed update budget): --reset-at u1 u2 ... re-initialises EVERY network and the optimizer (a fresh
#    TrainingRun from the same config, seed + 1000 k) after schedule update u_k, then runs --catch-up U updates
#    sampled uniformly from the games available at that point, then resumes the growing schedule. Arms
#    (3 seeds, growing schedule, 6400 schedule updates):
#      (i)  one reset at 3200 (collection 50), catch-up 3200 (9600 updates in all);
#      (ii) resets at 1600 / 3200 / 4800, catch-up 1600 each (11200 in all).
#    - Pre-registered readout (before any run), cell and heading means over seeds 0-2:
#      * PASS: within 5 points of the buffer arm (85.2 / 87.2) on both -> a default-off reset option in the
#        agent (Tiger fingerprint), then 3 seeds with model.value_estimator=double paired with the double runs.
#      * PARTIAL: >= +5 over the growing arm (76.6 / 71.9) on both but not a pass -> the better arm in the
#        agent anyway if it is >= +8 on heading (the probe furthest from the oracle), else report.
#      * FAIL: < +5 over the growing arm on either -> a reset does not undo the primacy effect offline; report.
#    - Result 2026-10-16 (runs/wumpus/data_vs_objective/reset{1,3}_seed*.log; the first launch at 00:17 died with
#      the machine before any reset fired, cause unknown (not memory: a rebuilt TrainingRun costs ~0.4 GB RAM,
#      40 MB VRAM); relaunched unchanged at 13:10): FAIL on both arms.
#                      cell                          heading                        held-out head NLL
#        growing       0.750 / 0.770 / 0.778 (76.6)  0.777 / 0.658 / 0.722 (71.9)   0.355 / 0.352 / 0.336
#        (i) reset1    0.806 / 0.797 / 0.804 (80.2)  0.746 / 0.747 / 0.796 (76.3)   0.295 / 0.310 / 0.308
#        (ii) reset3   0.767 / 0.769 / 0.767 (76.8)  0.712 / 0.644 / 0.687 (68.1)   0.319 / 0.308 / 0.305
#        buffer (2e)   0.874 / 0.845 / 0.838 (85.2)  0.781 / 0.917 / 0.919 (87.2)   0.288 / 0.265 / 0.287
#      (i) gains +3.6 / +4.4 and (ii) +0.2 / -3.8 over the growing arm, below the +5 needed on both. The resets
#      do cut the held-out NLL (0.30-0.31 vs 0.35), so they remove the head's overfitting, but not the
#      cell/heading deficit. Reading: matching the TOTAL budget (pre-registered) left the FINAL network with
#      less whole-buffer training than the buffer arm: (i)'s last network saw 3200 updates on the first 4352
#      games and 3200 growing ones; (ii)'s only 1600 + 1600, which is why more resets did worse. A reset helps
#      only to the extent the final network is trained long and uniformly on the full buffer, i.e. it
#      approaches the buffer arm itself. A reset schedule is NOT added to the agent; next is 2h.
#
# 2i. Value-equivalent observation term, retrain screen (2026-10-18; the latent-side
#     decision-aware objective):
#    - Question: does model.value_equivalent_weight (trainer.py 2f: the observation head's predicted next value
#      must match the realised one, gradients into the filter) make the retrain recipe's networks play better,
#      on the same games, than the same retrain without it? Theory: value equivalence (Grimm et al. 2020/2021),
#      decision-aware model learning (VaGraM, Voelcker et al. 2022). Motivation: exact_model_play.py (the exact
#      one-step model is worth +208 per game) and obs_fit_vs_latent.py (the missing information is not
#      decodable from the latent).
#    - Why a retrain screen first ("screen the whole fix on frozen data"): the recipe's
#      final networks ARE a whole-buffer retrain (training.retrain_every), so retraining offline on a run's
#      final buffer is the in-agent condition minus its later data; 2h predicted the in-agent result.
#    - --set (OmegaConf dotlist) applies config overrides to the run's own config before build_training_run.
#    - Arms (runs/sweeps/wumpus_retrain seeds 0-2, each run's own .hydra config, --representation decoder --horizon
#      5 --seed <s> --train-buffer <run>, 6400 updates, --save): control (w = 0), VE25 (w = 25), VE100 (w = 100).
#      Doses from a smoke measurement on seed 0's best.pt networks and buffer (6 batches): at w = 1 the term's
#      gradient into the filter is 1-2% of the other terms' (RMS residual ~32 value units), so w = 25 makes it
#      ~25-50% and w = 100 about equal. Played with retrain_play.py (section 5) on the 512 analysis games,
#      paired per game with the control arm; the probe lines score each retrained latent (cell / heading).
#    - Pre-registered readout (written before any run):
#      * PASS for an arm: paired gain over control >= +50 on >= 2 of 3 seeds, with the unfinished rate down and
#        the death rate up by at most 2 points on those seeds. If both pass, the larger mean gain is taken.
#        -> train the retrain + double recipe with that weight in the agent, 3 seeds paired with
#        runs/sweeps/wumpus_retrain (pre-registered there), then 5 seeds.
#      * FAIL: neither passes -> report per seed (with the probe accuracies, to see whether the term hurt the
#        latent) and take the next variant (multi-step value equivalence, a reward term) before further runs.
#    - Result 2026-10-18 (runs/wumpus/ve_screen/, chain ve_screen_chain.sh at 712dbe3; ~77 min per retrain, 3 in
#      parallel; every best.pt replay = report.json). **FAIL: no arm reaches +50 over control on any seed.**
#                 control (vs best.pt)    VE25 vs control   VE100 vs control   deaths c / 25 / 100     unfinished c / 25 / 100
#      seed 0     270.0 (+33.3 +- 30.5)   -57.0 +- 36.9     -63.8 +- 36.0      1.0 / 4.7 / 4.3%        64.3 / 63.5 / 64.3%
#      seed 1     214.7 (+57.9 +- 36.2)    +2.0 +- 26.4     +30.7 +- 26.0      1.8 / 1.4 / 0.8%        69.1 / 69.1 / 67.8%
#      seed 2     194.3 (-38.8 +- 40.4)   +27.3 +- 34.8     +12.4 +- 43.7      7.2 / 0.4 / 5.7%        60.2 / 69.9 / 62.3%
#      Probes (held-out pool), cell / heading / gold error / head danger error / head held-out NLL:
#        control  88 / 93 / .023 / .056 / .269;  87 / 91 / .024 / .070 / .299;  82 / 86 / .023 / .086 / .342
#        VE25     88 / 93 / .023 / .056 / .273;  86 / 91 / .024 / .080 / .316;  81 / 82 / .024 / .079 / .349
#        VE100    79 / 70 / .024 / .112 / .320;  81 / 73 / .024 / .115 / .367;  82 / 84 / .023 / .090 / .357
#    - Reading: at w = 25 the term changes neither the latent (probes within 1-3 points, gold error identical) nor
#      the unfinished games; the return differences are within the retrain-to-retrain noise (the control alone
#      moves -39 to +58 against best.pt). At w = 100 it damages the latent (heading -23 / -17 points on seeds 0-1,
#      danger error doubled, held-out NLL up) without putting gold information into it (gold error unchanged).
#      The value-of-information gap is not closed by a next-VALUE equivalence term on this latent.
#    - Mechanism hypothesis (untested): the term matches only V-bar(z_{t+1}); on Wumpus the death penalty and the
#      gold reward are in the REWARD (r_t), not in the next value, and the learned V-bar at game-over latents is
#      co-adapted (exact_model_play.py 3a), so the term is blind to the outcomes that matter most and trades
#      death/danger calibration for value matching (danger error up with w). A Q-equivalent variant would need
#      observation-conditioned rewards, sum_o P(o)[r(o) + gamma V-bar(tau)] against the realised r + gamma V-bar.
#    - Also learned: n = 3 retrain screens have retrain-to-retrain noise of ~+-50 per seed (control vs best.pt),
#      so a +50 bar on 2 of 3 is underpowered for effects below ~+50 (gap 2). Per the readout: report
#      and choose the next variant before further runs.
#
# 2j. Coverage screen (2026-10-19, coverage_games.py): --extra-games <file> appends the
#    games saved by coverage_games.py (an EpisodeBatch-shaped dict) to the --train-buffer data, so the uniform
#    retrain sees the run's whole buffer plus the arm's extra games. Question, arms, pre-registered readout and
#    result: coverage_games.py. Several files may be given (2026-10-21): the union arm appends the
#    cover AND own games of a seed (4096 extra games), the data upper bound of the two coverage deficits together.
#
# 2k. Bootstrapped whole-model retrains (2026-10-21; question, readout and result in
#    ensemble_disagreement.py): --bootstrap <j> resamples the uniform retrain's games (buffer plus --extra-games)
#    with replacement, as many as there are, from a generator seeded with j that is separate from the update
#    sampler's. K such retrains with different --seed and --bootstrap form a bootstrapped ensemble of whole models,
#    filter included (Osband et al. 2016; Lakshminarayanan et al. 2017), whose disagreement on a history is the
#    candidate epistemic-uncertainty signal. Without the flag nothing changes (the resample draws from its own
#    generator only when given).
#
# 2l. Self-predictive latent, retrain screen (2026-10-22):
#    - Question: does adding latent self-prediction (the ZP condition, Ni et al. 2024; trainer.py 2g) make the
#      retrain recipe's networks play better on the same games than the same retrain without it? And does JEPA (ZP
#      with the one-step observation term detached; contribution 1) on the retrain recipe match or beat the decoder?
#    - Arms (runs/sweeps/wumpus_retrain seeds 0-2, each run's own .hydra config, --horizon 5 --seed <s> --train-buffer
#      <run>, 6400 updates, --save into runs/wumpus/sp_screen/):
#        ctl   the existing 2i control retrains (runs/wumpus/ve_screen/control_seed<s>.pt, linked as ctl_seed<s>.pt;
#              same data, config and seed; their returns are re-played here as the pairing's first set)
#        zp    --representation decoder --set model.self_predictive_weight=1.0   (OP + ZP: the hybrid)
#        jepa  --representation jepa                                              (ZP, OP detached)
#      w = 1 is JEPA's own weight, so ctl / zp / jepa form a factorial on identical data. One dose only (compute:
#      the in-agent option (a) sweeps hold two GPU slots).
#    - Played with retrain_play.py <run> ctl zp jepa (paired per game with ctl on the 512 analysis games; JEPA-layout
#      sets play through saved_networks.py) and scored with feature_localise.py --networks (glitter ratio and decision
#      share M at the declined turned latents) plus this script's probe lines (cell / heading / gold / danger / NLL).
#    - Pre-registered readout (written before any run):
#      * PASS for an arm: paired gain over ctl >= +50 on >= 2 of 3 seeds, with unfinished down and deaths up by at
#        most 2 points on those seeds. If both pass, the larger mean gain is taken.
#        -> zp: the in-agent option (model.self_predictive_weight=1 on the retrain recipe, stacked on option (a) if
#        (a) passes), 3 seeds paired with runs/sweeps/wumpus_retrain (or the (a) sweep). jepa: the same with
#        model.representation=jepa, and contribution 1 is rescued on Wumpus.
#      * MECHANISM (reported for both, decides the reading of a FAIL): glitter ratio at the declined turned latents up
#        by >= 0.1 over ctl on >= 2 of 3 seeds.
#      * FAIL with MECHANISM pass: the latent gained the missing information but play did not follow (planner
#        exploitation again) -> stack with option (a)'s margin offline (pessimism_play.py on the saved sets).
#      * FAIL without MECHANISM: self-prediction does not put the declined-state information into the latent at this
#        data -> the latent objective is not the lever; next is the grokking screen (long fixed-buffer retrains with
#        weight decay) and decision-aware weighting (1b candidate 2).
#    - Power caveat stated in advance: retrain-to-retrain noise is ~+-50 per seed (2i), so the bar detects only large
#      effects; probes and the mechanism readout carry the reading of a null.
#    - Result (2026-10-25, seeds 0-2; runs/wumpus/sp_screen/play_seed<s>.log, fl_<arm>_seed<s>.log): FAIL for both
#      arms, without MECHANISM.
#        seed  ctl return (deaths)   zp vs ctl (deaths, unfinished)       jepa vs ctl (deaths)    glitter ctl / zp / jepa
#        0     270.0 (1.0%)          -61.1 +- 31.1 (3.7%, 65.8%)           -20.9 +- 27.8 (1.2%)    0.41 / 0.50 / 0.25
#        1     214.7 (1.8%)           +0.5 +- 36.5 (11.5%, 50.2%)          -24.3 +- 38.6 (10.2%)   0.25 / 0.30 / 0.23
#        2     194.3 (7.2%)          +78.9 +- 35.6 (0.0%, 66.0%)           +69.6 +- 34.5 (0.6%)    0.33 / 0.50 / 0.27
#      zp clears +50 on 1 of 3 seeds and the glitter bar (+0.10) on 1 of 3 (+0.09 / +0.05 / +0.17); jepa clears +50
#      on 1 of 3 and lowers the glitter ratio on every seed. Seed 2's gains are measured against a control that plays
#      38.8 per game below that seed's best.pt (vs best.pt: zp +40.1, jepa +30.8). Where zp cures idling (seed 1,
#      unfinished 69 -> 50%) the freed steps die (deaths 1.8 -> 11.5%). Reading, per the registered branch: latent
#      self-prediction does not reliably put the declined-state information into the latent at this data; JEPA on
#      the retrain recipe is worse than the decoder on latent, mechanism and play. Next: the grokking screen (2m).
#
# 2m. Grokking screen: long fixed-buffer retrain with strong weight decay (2026-10-22; queued after
#    Step 2, written before the run):
#    - Question: trained far past the overfitting point (~6400 updates, 2b) on the FIXED whole buffer with strong
#      decoupled weight decay, does the model generalise late (grokking: Power et al. 2022; Omnigrok, Liu et al. 2023:
#      weight decay drives the weight norm to the generalising solution on non-algorithmic data too; Nanda et al.
#      2023), in particular OFF the data's support, i.e. the glitter prediction at the declined turned latents?
#      Wumpus has crisp rules (glitter iff gold in the cell entered), the setting where grokking is reported.
#    - Why it is not a repeat: 2b tried weight decay <= 0.1 up to ~25600 updates on the pool data and saw held-out NLL
#      rise; grokking needs decay ~1 and training many times past the rise. In-agent, the growing buffer is
#      non-stationary (no fixed dataset to grok), so the screen is offline on the whole-buffer retrain.
#    - Arm: runs/sweeps/wumpus_retrain/seed0's config and buffer, --representation decoder --horizon 5 --seed 0,
#      --weight-decay 1.0, --updates 51200, --score-at 6400 12800 25600 51200 with --save (checkpoints
#      runs/wumpus/grok_screen/wd1_seed0_u<k>.pt). Reference: the 2i control (weight decay 1e-4, 6400 updates;
#      held-out NLL 0.269, glitter ratio 0.41 at the declined turned latents, sp_screen fl_ctl_seed0.log).
#    - Measures per checkpoint: the probe and held-out NLL lines (RESULT@k), feature_localise.py --networks (glitter
#      ratio, TV, decision share M at the declined turned latents).
#    - Pre-registered readout:
#      * GROK: the held-out NLL falls by >= 0.02 below its 6400-update value at a later checkpoint after not falling
#        between 6400 and 12800, OR the glitter ratio at a late checkpoint (>= 25600) is >= ctl + 0.1. -> play the best
#        late checkpoint vs ctl (retrain_play.py); >= +50 paired -> seeds 1-2, then an in-agent schedule question.
#      * NO GROK: neither -> training length and weight decay are not the lever at this data scale; recorded as a
#        negative (the paper can say the off-support model error is not an optimisation-length artefact).
#      * Collapse check: if probes (cell / heading) fall by > 10 points at 6400 against ctl, weight decay 1 is too
#        strong for this network; report and rerun at 0.3 once.
#    - Result, seed 0 (2026-10-24; the run died past 40000 updates in the 2026-10-22 reboot, so 51200 is missing; every
#      best.pt / ctl replay reproduces report.json and the 2i returns). Strict grokking clause FAIL (NLL falls already
#      by 12800, no rise first); glitter clause PASS (u25600 0.55 vs ctl 0.41).
#                 held-out NLL  cell / heading   glitter ratio (c)  TV (c)   share M   return vs ctl (95%)   deaths   unfinished
#      ctl        0.269         88 / 93          0.41               0.355    0.25      270.0                 1.0%     64.3%
#      u6400      0.276         84 / 86          0.46               0.455    0.27      -
#      u12800     0.234         92 / 93          0.58               0.286    0.31      +5.9 +- 24.1          0.2%     66.2%
#      u25600     0.232         93 / 93          0.55               0.257    0.38      +35.5 +- 25.6         0.0%     63.5%
#      Play is below the registered +50 single-seed bar. DEVIATION, stated openly: a single seed is underpowered
#      (retrain noise ~+-50 per seed, 2i) and the mechanism passed, so seeds 1-2 are run under a 3-seed criterion
#      written here before they run:
#    - Extension (seeds 1, 2; weight decay 1.0, 25600 updates, checkpoint 12800; paired with the 2i ctl's stored
#      returns runs/wumpus/ve_screen/control_seed<s>.returns.npy, whose replay was identical on seed 0 twice):
#      * PASS: u25600 - ctl > 0 on 3 of 3 seeds, mean >= +30, deaths <= ctl + 2 points on each seed, and the glitter
#        ratio >= ctl + 0.1 on >= 2 of 3 -> the agent option question: an in-agent schedule whose final whole-buffer
#        retrain uses weight decay 1 and 25600 updates (training.retrain_updates plus a retrain weight-decay key),
#        3 seeds paired with runs/sweeps/wumpus_retrain.
#      * FAIL: otherwise; seed 0 reported as a single-seed positive.
#    - Result of the extension (2026-10-25; runs/wumpus/grok_screen/{train_wd1,fl,play}_seed<s>*.log): FAIL. The
#      mechanism holds on every seed; play fails on seed 2 by a collapse of the u25600 set.
#        seed  set     held-out NLL  cell / heading  glitter (c)  TV (c)  return   vs ctl (95%)       deaths  unfinished
#        1     ctl     0.299         87 / 91         0.25         0.317   214.7    -                  1.8%    69.1%
#        1     u12800  0.241         93 / 93         0.50         0.210   260.0    +45.4 +- 31.7      0.2%    68.4%
#        1     u25600  0.237         94 / 94         0.45         0.214   231.2    +16.5 +- 34.4      0.0%    70.5%
#        2     ctl     0.342         82 / 86         0.33         0.570   194.3    -                  7.2%    60.2%
#        2     u12800  0.243         91 / 92         0.49         0.444   262.5    +68.2 +- 35.0      0.0%    68.0%
#        2     u25600  0.236         93 / 92         0.56         0.403   -41.0    -235.2 +- 45.6     0.0%    99.4%
#      u25600 > ctl on 2 of 3 seeds (+35.5 / +16.5 / -235.2); glitter >= ctl + 0.1 on 3 of 3 (+0.14 / +0.20 / +0.23).
#      Seed 0 stands as a single-seed positive. u12800 beats ctl on 3 of 3 (+5.9 / +45.4 / +68.2, mean +39.8), but
#      that checkpoint was not the registered one and is read here only as a lead for a new pre-registration.
#    - Diagnosis of the seed-2 collapse (phantom_reward_diag.py --networks, best.pt's 128 stored greedy games): the
#      heads are exact (r_hat(GRAB | GLITTER) 998.4, no-op rewards -1.0), but on the 51 GLITTER steps the one-step
#      Q(GRAB) = r_hat + gamma V_hat(GAME_OVER child) is 948.4 against 1048.0 for the best other action, so GRAB is
#      the argmax on 3.9% of them (best.pt 100%, u12800 100%). V_hat of the absorbing GAME_OVER latent, whose true
#      value is 0, reads about -50 (u25600), +184 (u12800) and +70 (best.pt), while the other actions at the gold
#      exceed the 999 that any game can pay. The absorbing state's value is fixed only by its own Bellman residual,
#      V(go) - gamma V(go) = (1 - gamma) V(go), a gradient 100x weaker than an ordinary target at gamma = 0.99, so a
#      long, strongly regularised retrain can leave it anywhere (wumpus.py section 3 expected the targets to reach
#      the fixed point by themselves). Whether the agent grabs the gold rests on that unanchored value, which also
#      adds seed-to-seed noise to every retrain comparison.
#
# 2n. Anchored absorbing value in training, retrain screen (2026-10-25; model.absorbing_value, trainer.py 3f, run.py 19;
#    written before any run):
#    - Question: anchoring only the planner on networks trained without it fails (retrain_play.py 6: seed 0 best.pt
#      -169, u12800 -320, u25600 -349 per game; ctl -2), because the learned value carries a common offset that
#      the anchor removes from the GRAB side only. Does anchoring the TARGETS (V = 0 behind GAME_OVER) remove the
#      offset itself, and does the strong-weight-decay retrain then pay, with the planner anchored too?
#    - Arms (runs/sweeps/wumpus_retrain seeds 0-2, each run's own .hydra config, --representation decoder --horizon 5
#      --seed <s> --train-buffer <run> --set model.absorbing_value=true, --save into runs/wumpus/absorbing_train/):
#        ctlA  6400 updates at the config's weight decay (1e-4): the 2i control plus the anchor
#        wdA   --weight-decay 1.0 --updates 25600 --score-at 12800 25600: the 2m arm plus the anchor
#      Played with retrain_play.py <run> ctlA wdA_u12800 wdA_u25600 --absorbing-value (planner anchored as in the
#      agent option; 512 analysis games, the games of every earlier play), paired per game with the unanchored 2i
#      control's stored returns (runs/wumpus/ve_screen/control_seed<s>.returns.npy) and with ctlA.
#      phantom_reward_diag.py --networks on each set (unanchored readout of the learned values: V_hat(GAME_OVER
#      child) = (Q(GRAB) - r_hat(GRAB)) / gamma at the GLITTER steps, GRAB's argmax share there, V_hat > 999 share).
#    - Pre-registered readout:
#      * MECHANISM: on every anchored set, the learned V_hat behind GAME_OVER at the GLITTER steps is within +-10 of 0,
#        and GRAB is the one-step argmax on >= 95% of the GLITTER steps.
#      * PASS-A (the anchor on the plain recipe): ctlA - ctl >= 0 on the mean of the 3 seeds, no seed below -25, and
#        deaths <= ctl + 2 points on each seed.
#      * PASS-W (the strong-weight-decay retrain once anchored, both sides anchored): wdA_u25600 - ctlA > 0 on 3 of 3
#        seeds, mean >= +30, deaths <= ctlA + 2 points on each seed (u12800 reported, not decided on).
#      -> PASS-W (with MECHANISM): G2 in-agent, the recipe with training.retrain_updates=25600
#         training.retrain_weight_decay=1.0 model.absorbing_value=true (run.py 14 pre-registration with the anchor
#         added), seeds 0-2 paired with runs/sweeps/wumpus_retrain.
#      -> PASS-A only: the in-agent screen tests model.absorbing_value=true on the plain recipe.
#      -> MECHANISM fails: the anchor does not reach the value's offset; diagnose before any in-agent run.
#      -> otherwise: report; the anchor stays a default-off option.
#    - Result (2026-10-26; runs/wumpus/absorbing_train/{train,play,phantom}_*): PASS-W PASS, PASS-A FAIL, MECHANISM
#      FAIL as registered. Anchored plays (planner anchored), 512 games, paired per game:
#        seed  ctl (unanch.)    ctlA: vs ctl, deaths     wdA_u12800: vs ctlA     wdA_u25600: vs ctlA, vs ctl, deaths
#        0     270.0, 1.0%      225.5: -44.5, 5.5%       298.9: +73.4            271.1: +45.6 +- 32.9, +1.1, 1.4%
#        1     214.7, 1.8%      226.2: +11.6, 1.0%       268.1: +41.9            300.7: +74.5 +- 32.0, +86.1, 0.0%
#        2     194.3, 7.2%      200.0: +5.7, 8.2%        255.3: +55.3            264.1: +64.1 +- 29.8, +69.8, 0.0%
#      PASS-W: u25600 > ctlA on 3 of 3, mean +61.4, deaths below ctlA on every seed. u25600 averages 278.6 against
#      the plain retrain recipe's best.pt 208.9 on these seeds (not paired) and 226.3 for the unanchored ctl.
#      PASS-A: ctlA - ctl mean -9.1 (seed 0 -44.5, deaths +4.5 points): the anchor alone does not help the recipe.
#      MECHANISM (phantom_reward_diag.py --networks, unanchored readout of the learned values): V_hat > 999 on
#      <= 0.3% of live steps on the wdA sets (0 / 2.2 / 1.1% on ctlA); GRAB the one-step argmax at GLITTER on 100%
#      except ctlA seed 2 (0%); but the learned V_hat of the imagined GAME_OVER child reads +28 / +36, +233 / +18,
#      +57 / +29 on the wdA sets (ctlA +0.1 / +0.1 / +0.2), outside the +-10 clause. The anchored planner never reads
#      that value; whether the stored post-game latents fit their anchored targets is go_value_diag.py's question,
#      which the registered branch ("diagnose before any in-agent run") requires before the in-agent screen.
#    - Follow-up (go_value_diag.py header 4, 2026-10-26): on the stored games the wdA sets fit V = 0 on the later
#      GAME_OVER latents and the live targets, but not on the first GAME_OVER latent of a game (+11 to +152; ctlA
#      <= 0.13), the stored counterpart of the imagined child above. No target or decision of the anchored recipe
#      reads that value, so the clause's failure is real but off the decision path; it is a deviation from the
#      registered MECHANISM reading, stated here rather than re-registered.
#
# 2o. Decision-aware weighting of the observation term, retrain screen (2026-10-27, written before any run;
#     model.decision_aware_weight, trainer.py 2h, run.py 20):
#    - Question: does weighting each transition's observation cross-entropy by the squared span of its branch values
#      (a bound on the backup's squared value-equivalence error; VaGraM for discrete observations) make the retrain
#      recipe's networks repair the value of information at the declined states and play better than the same
#      retrain without it? Unlike 2i the loss stays a proper scoring rule per transition and reads the values of
#      every CHILD branch, where the gold's value lives.
#    - Smoke (seed 0, the 2i control networks on the run's buffer, 32,000 transitions; weights at beta = 1): 92.5%
#      of transitions have zero span, the top 5% carry 97.7% of the weight (ESS 3.3%); mean ratio 7.2 on FORWARD,
#      0.02 on the other actions, 5.9 on realised GLITTER. So beta is the share of the CE mass moved onto the ~5%
#      of transitions whose outcome changes a value; beta = 0.5 is fixed a priori (one dose: half the representation
#      term stays uniform, since in the decoder representation the CE also shapes the latent).
#    - Arm (runs/sweeps/wumpus_retrain seeds 0-2, each run's own .hydra config, --representation decoder --horizon 5
#      --seed <s> --train-buffer <run> --pool-cache runs/wumpus/supervised_filter/pools.pt, 6400 updates, --set
#      model.decision_aware_weight=0.5, --save runs/wumpus/da_screen/DA50_seed<s>.pt): the 2i control
#      (runs/wumpus/ve_screen/control_seed<s>.pt: same seed, init and batches, beta = 0) is the paired control, i.e.
#      a final-retrain baseline, never best.pt. Played with retrain_play.py <run> control DA50 (512 analysis games,
#      paired per game with the control), then feature_localise.py --networks DA50 for the mechanism, compared with
#      feature_localise_control_seed<s>.json. Chain: runs/wumpus/da_screen_chain.sh, one GPU job at a time.
#    - Pre-registered readout:
#      * PLAY: DA50 - control > 0 on 3 of 3 seeds, mean of the three >= +30, deaths <= control + 2 points each.
#      * MECHANISM: joint glitter ratio at the declined turned latents (c) >= control + 0.10 on >= 2 of 3 seeds.
#      * Latent guard (reported): probe cell / heading not more than 5 points below the control's (2i probes).
#      -> PLAY and MECHANISM: the in-agent screen (recipe + model.decision_aware_weight=0.5 + checkpoint_selection=
#         final_retrain), seeds 0-2, pre-registered in run.py 20 before launch, paired with the recipe's
#         final-retrain networks (runs/wumpus/absorbing_train/recipe_seed<s>_c100.returns.npy).
#      -> PLAY only: the gain does not come through the value of information; diagnose (action mix, risk) first.
#      -> MECHANISM only: the model repairs but the planner does not profit; report, then one higher dose (0.8).
#      -> neither: FAIL; the next candidate is the Q-equivalent observation term (reward-carried outcomes).
#    - Result 2026-10-27 (runs/wumpus/da_screen/; ~1 h per seed; control replays reproduce the 2i plays and every
#      best.pt replay reproduces report.json): FAIL (neither PLAY nor MECHANISM).
#                 control   DA50    vs control       deaths c / DA     unfinished c / DA   joint glitter c / DA   cell / heading c / DA
#      seed 0     270.0     208.7   -61.2 +- 34.1    1.0 / 5.9%        64.3 / 62.7%        0.45 / 0.31            88 / 93 -> 84 / 79
#      seed 1     214.7     292.8   +78.2 +- 27.8    1.8 / 0.4%        69.1 / 62.5%        0.39 / 0.27            87 / 91 -> 84 / 80
#      seed 2     194.3     244.7   +50.4 +- 40.7    7.2 / 0.8%        60.2 / 67.2%        0.33 / 0.40            82 / 86 -> 86 / 90
#      Mean +22.5; PLAY fails on seed 0 (negative, deaths +4.9 points); MECHANISM fails (glitter ratio DOWN on seeds
#      0-1, +0.07 on seed 2); the latent guard fails on seeds 0-1 (heading -14 / -11 points).
#    - Reading: moving half the CE mass onto the ~5% of high-span transitions starves the uniform representation
#      pressure that tracks heading, and the observation head does not become better calibrated where it matters (the
#      glitter under-prediction gets worse on 2 of 3 seeds). The two positive seeds come with fewer deaths, not fewer
#      unfinished games on seed 2, so the gains are not the value-of-information mechanism. The span weights are set by
#      the model's OWN value at its own imagined children, which under-rates exactly the branches it under-predicts
#      (the feedback the theory assumes away: V is the true value in the bound). Per the registered branch: the next
#      candidate is the Q-equivalent observation term.
#
# 2p. Observation-conditioned reward and the Q-equivalent observation term, retrain screen (2026-10-27, written before
#     any run; model.observation_reward, models/heads.py 2d, planning/search_model.py 8; model.q_equivalent_weight,
#     trainer.py 2i; run.py 21):
#    - Question: does matching the model's whole one-step backup, sum_o P(o | z, a) [r(z, a, o) + gamma V-bar(tau)],
#      to the realised r_t + gamma V-bar(z_{t+1}) (proper value equivalence, Grimm et al. 2020/2021; MuZero's principle)
#      repair the value of information at the declined states and make the retrain recipe's networks play better
#      than the same retrain without it? It needs a reward that depends on the branch, r(z, a, o'): on Wumpus the
#      reward is a function of (alive, a, o') (GAME_OVER after FORWARD is a death, after GRAB a win), so the joint
#      model p(o', r | z, a) = P(o') p(r | o') holds every outcome's probability in P, and the term sees the death and
#      the gold that 2i (V only) could not. Third arm of the decision-aware ablation: 2i (V-equivalence), 2o (span^2
#      CE weights), 2p (Q-equivalence).
#    - Arms (runs/sweeps/wumpus_retrain seeds 0-2, each run's own .hydra config, --representation decoder --horizon 5
#      --seed <s> --train-buffer <run> --pool-cache runs/wumpus/supervised_filter/pools.pt, 6400 updates, --save into
#      runs/wumpus/qe_screen/):
#        OR       --set model.observation_reward=true                                      (the structure alone)
#        QE<w>    --set model.observation_reward=true --set model.q_equivalent_weight=<w>  (structure + term)
#      The 2i control (runs/wumpus/ve_screen/control_seed<s>.pt: same seed, R(z, a), no term; the wider reward head
#      shifts the initial draws of the networks built after it, so the pairing is per seed and game, not per weight)
#      is the paired control, a final-retrain baseline, never best.pt. OR isolates the
#      restructured reward, so QE - OR is the term's own effect.
#    - Dose rule (fixed now, applied once on seed 0 before the other runs): on the OR seed-0 networks and its run's
#      buffer, 6 training batches, measure r = ||grad_filter QE(w = 1)|| / ||grad_filter (all other terms)||; w is the
#      value of {10, 25, 50, 100, 250} (or 5 / 2.5 below that) closest in log ratio to w r = 0.375, i.e. the term
#      carries ~25-50% of the other terms' filter gradient (2i's method). One dose, recorded here before the chain.
#      Smoke 2026-10-27 (runs/wumpus/qe_screen/dose_smoke_seed0.log; OR seed 0: cell / heading 87 / 92, held-out NLL
#      0.269, as the control): r = 0.082 (0.023-0.166 over the 6 batches) -> **w = 5** (w r = 0.41). The residual lives
#      on FORWARD (RMS 300, mean +41: the model's backup is optimistic there on average; SHOOT 15, the others < 0.4),
#      and the backed-up FORWARD reward equals -1 - 1000 P(GAME_OVER) to 0.15 on live latents (the factorisation holds).
#    - Plays: retrain_play.py <run> OR_seed<s>.pt QE<w>_seed<s>.pt --set model.observation_reward=true (512 analysis
#      games; header 7: best.pt replays first with its own config and must reproduce report.json; QE paired in-script
#      with OR), then feature_localise.py <run> cuda --networks <arm> --set model.observation_reward=true for each arm.
#      Pairing with the control from the stored per-game arrays (control_seed<s>.returns.npy; plays 270.0 / 214.7 /
#      194.3; glitter ratio 0.45 / 0.39 / 0.33; probes cell / heading 88 / 93, 87 / 91, 82 / 86) on the CPU.
#    - Pre-registered readout:
#      * PLAY (QE): QE - control > 0 on 3 of 3 seeds, mean of the three >= +30, deaths <= control + 2 points each.
#      * MECHANISM (QE): joint glitter ratio at the declined turned latents (c) >= control + 0.10 on >= 2 of 3 seeds.
#      * Latent guard (reported): probe cell / heading not more than 5 points below the control's.
#      * Attribution (reported): OR - control and QE - OR per seed, with deaths and unfinished rates.
#      -> QE passes PLAY and MECHANISM: the in-agent screen (recipe + both options + checkpoint_selection=final_retrain),
#         seeds 0-2, pre-registered in run.py 21 before launch, paired with the recipe's final-retrain networks
#         (runs/wumpus/absorbing_train/recipe_seed<s>_c100.returns.npy).
#      -> OR meets PLAY's criterion on its own (whatever QE does): the restructured reward carries the effect; the
#         in-agent candidate is OR (with QE only if QE - OR > 0 on 3 of 3).
#      -> QE passes PLAY only: the gain does not come through the value of information; diagnose (action mix, deaths).
#      -> QE passes MECHANISM only: the model repairs but the planner does not profit; report.
#      -> neither (and OR does not pass PLAY): FAIL; report the 2i / 2o / 2p ablation, no further objective variant.
#    - Result 2026-10-28 (runs/wumpus/qe_screen/, chain qe_screen_chain.sh at 57bb994, w = 5; ~45 min per arm per seed;
#      every best.pt replay reproduces report.json): FAIL (neither PLAY nor MECHANISM; OR does not pass PLAY either).
#                 control   OR      QE5     QE - control     OR - control     QE - OR          deaths c / OR / QE   unfinished c / OR / QE
#      seed 0     270.0     224.1   179.0   -91.0 +- 40.9    -45.9 +- 31.8    -45.1 +- 41.6    1.0 / 3.1 / 7.4%     64.3 / 65.0 / 61.9%
#      seed 1     214.7     235.2   256.4   +41.7 +- 30.1    +20.6 +- 25.8    +21.2 +- 25.2    1.8 / 0.6 / 2.5%     69.1 / 68.4 / 62.3%
#      seed 2     194.3     151.5   234.3   +40.1 +- 38.9    -42.8 +- 39.1    +82.8 +- 33.0    7.2 / 3.3 / 1.2%     60.2 / 70.1 / 68.2%
#      QE - control mean -3.1 (PLAY: > 0 on 2 of 3, deaths +6.4 points on seed 0); OR - control mean -22.7; QE - OR mean
#      +19.6. Joint glitter ratio at (c), control / OR / QE: 0.45 / 0.32 / 0.44, 0.39 / 0.30 / 0.34, 0.33 / 0.48 / 0.34
#      (MECHANISM: QE - control -0.00 / -0.05 / +0.02, 0 of 3). Latent guard holds: cell / heading QE 84 / 91, 86 / 91,
#      87 / 91, OR 87 / 92, 87 / 91, 87 / 91 (control 88 / 93, 87 / 91, 82 / 86); held-out NLL QE 0.285 / 0.316 / 0.301,
#      OR 0.269 / 0.297 / 0.286 (control 0.269 / 0.299 / 0.342).
#    - Reading: the Q-equivalent term does not put the value of information into the model: the glitter under-prediction
#      at the declined turned latents is unchanged (it is the same 2-3x as the control's), and the unfinished games move
#      within the retrain-to-retrain noise. Its residual lives on FORWARD (smoke: RMS 300, mean +41) and is dominated by
#      the death branch, so the term mostly re-prices P(GAME_OVER), which moves deaths in either direction (seed 0 7.4%,
#      seed 2 1.2%). The restructured reward alone is not neutral either: moving the death risk from R(z, a) into P makes
#      the planner rely on one head for both the risk and the branching, and it lost on 2 of 3 seeds. Across the three
#      decision-aware objectives (2i V-equivalence, 2o span^2 CE weights, 2p Q-equivalence) none moves the glitter ratio
#      at the declined states: with the agent's own value and its own data, the objective cannot supply information that
#      the data do not show (the declined turn-then-step outcomes are rare in the buffer; C10 (V), coverage_games.py).
#
# 3. Usage:
#    uv run python reports/2026-10-02_wumpus_diagnostics/objective_ab.py --representation jepa \
#        --horizon 5 --seed 1 --pool-cache runs/wumpus/supervised_filter/pools.pt
# ==============================================================================

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from ipomdp.domain.wumpus import GAME_OVER, NUM_CELLS
from ipomdp.experiments.runs import build_training_run
from ipomdp.models import ObservationHead
from ipomdp.training import EpisodeBatch, load_checkpoint

sys.path.insert(0, str(Path(__file__).parent))
from supervised_filter import pool  # noqa: E402  (same diagnostics directory)


def latents_of(belief_filter, data, rows, num_a, num_o):
    with torch.no_grad():
        z = belief_filter.unroll(F.one_hot(data["actions"][rows], num_a).float(),
                                 F.one_hot(data["observations"][rows], num_o).float(),
                                 F.one_hot(data["initial"][rows], num_o).float())
    alive = data["alive"][rows]
    return z[alive].float(), {k: data[k][rows][alive] for k in ("hazards", "cell", "heading", "danger")}


def observation_nll(run, data, rows, num_a: int, num_o: int) -> torch.Tensor:
    """Acting head's next-observation NLL at alive steps of data[rows] (every action taken)."""
    with torch.no_grad():
        actions = F.one_hot(data["actions"][rows], num_a).float()
        z = run.acting_filter.unroll(actions, F.one_hot(data["observations"][rows], num_o).float(),
                                     F.one_hot(data["initial"][rows], num_o).float())[:, :-1]
        logits = run.acting_observation_head(z, actions).float()
        nll = F.cross_entropy(logits.flatten(0, 1), data["observations"][rows].flatten(), reduction="none")
    return nll[data["alive"][rows][:, :-1].flatten()].mean()


def transitions(belief_filter, data, rows, num_a: int, num_o: int):
    """(z_t, one-hot a_t, o_{t+1}) at the alive steps of data[rows], frozen, in chunks of 1024 games."""
    zs, acts, obs = [], [], []
    for i in range(0, len(rows), 1024):
        r = rows[i:i + 1024]
        with torch.no_grad():
            actions = F.one_hot(data["actions"][r], num_a).float()
            z = belief_filter.unroll(actions, F.one_hot(data["observations"][r], num_o).float(),
                                     F.one_hot(data["initial"][r], num_o).float())[:, :-1]
        alive = data["alive"][r][:, :-1]
        zs.append(z[alive].float()); acts.append(actions[alive]); obs.append(data["observations"][r][alive])
    return torch.cat(zs), torch.cat(acts), torch.cat(obs)


def fit_planning_head(run, train, fit_rows, num_a: int, num_o: int, cfg, device: torch.device) -> nn.Module:
    """Section 2d: a fresh ObservationHead on frozen acting latents, early-stopped on 10% of the games."""
    val_rows, head_rows = fit_rows[::10], fit_rows[torch.arange(len(fit_rows), device=device) % 10 != 0]
    z, a, o = transitions(run.acting_filter, train, head_rows, num_a, num_o)
    vz, va, vo = transitions(run.acting_filter, train, val_rows, num_a, num_o)
    torch.manual_seed(0)
    head = ObservationHead(cfg.model.latent_dim, num_a, num_o, cfg.model.hidden_dim, cfg.model.num_blocks).to(device)
    optimizer = torch.optim.Adam(head.parameters(), 3e-4)
    generator = torch.Generator(device=device).manual_seed(0)
    best, best_state, patience = float("inf"), None, 0
    for step in range(1, 40001):
        rows = torch.randint(0, len(z), (4096,), device=device, generator=generator)
        loss = F.cross_entropy(head(z[rows], a[rows]), o[rows])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 250 == 0:
            with torch.no_grad():
                val = sum(F.cross_entropy(head(vz[i:i + 65536], va[i:i + 65536]), vo[i:i + 65536], reduction="sum").item()
                          for i in range(0, len(vz), 65536)) / len(vz)
            if val < best - 1e-4:
                best, best_state, patience = val, {k: v.clone() for k, v in head.state_dict().items()}, 0
            else:
                patience += 1
                if patience == 6:
                    break
    head.load_state_dict(best_state)
    print(f"planning head: stopped at step {step}, validation NLL {best:.4f}", flush=True)
    return head


def score_both(run, train, held, fit_rows, num_a, num_o, cfg, device, planning_head: bool, prefix: str) -> None:
    """Prints the joint head's scores and, with planning_head, the decoupled head's (section 2d)."""
    for name, value in score(run, train, held, num_a, num_o, device).items():
        print(f"{prefix} {name} {value:.4f}", flush=True)
    if planning_head:
        joint = run.acting_observation_head
        run.acting_observation_head = fit_planning_head(run, train, fit_rows, num_a, num_o, cfg, device)
        for name, value in score(run, train, held, num_a, num_o, device).items():
            if name.startswith("head_"):
                print(f"{prefix} plan_{name} {value:.4f}", flush=True)
        run.acting_observation_head = joint


def score(run, train, held, num_a: int, num_o: int, device: torch.device) -> dict[str, float]:
    """Section 2's scores of the run's acting filter and observation head on the held-out pool."""
    # (a) Probes on frozen acting latents.
    acting = run.acting_filter
    acting.eval()
    fit_z, fit_t = latents_of(acting, train, slice(0, 1024), num_a, num_o)
    fit_z2, fit_t2 = latents_of(acting, train, slice(4096, 5120), num_a, num_o)          # the random half
    fit_z = torch.cat([fit_z, fit_z2])
    fit_t = {k: torch.cat([fit_t[k], fit_t2[k]]) for k in fit_t}
    held_z, held_t = latents_of(acting, held, slice(None), num_a, num_o)
    mean, std = fit_z.mean(0), fit_z.std(0) + 1e-6
    torch.manual_seed(0)
    probe = nn.Sequential(nn.Linear(fit_z.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                          nn.Linear(256, 3 * NUM_CELLS + NUM_CELLS + 4 + 1)).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), 1e-3)
    probe_generator = torch.Generator(device=device).manual_seed(0)
    for _ in range(3000):
        rows = torch.randint(0, len(fit_z), (512,), device=device, generator=probe_generator)
        out = probe((fit_z[rows] - mean) / std)
        loss = (F.binary_cross_entropy_with_logits(out[:, :48], fit_t["hazards"][rows])
                + F.cross_entropy(out[:, 48:64], fit_t["cell"][rows]) + F.cross_entropy(out[:, 64:68], fit_t["heading"][rows])
                + F.binary_cross_entropy_with_logits(out[:, 68], fit_t["danger"][rows]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        out = probe((held_z - mean) / std)
        hazards, danger = torch.sigmoid(out[:, :48]), torch.sigmoid(out[:, 68])
        forward = F.one_hot(torch.zeros(len(held_z), dtype=torch.int64, device=device), num_a).float()
        head = F.softmax(run.acting_observation_head(held_z, forward).float(), -1)[:, GAME_OVER]
    exact = held_t["danger"]
    safe, certain = exact < 1e-9, exact > 0.6
    results = {
        "probe_pit_error": (hazards[:, :16] - held_t["hazards"][:, :16]).abs().mean(),
        "probe_wumpus_error": (hazards[:, 16:32] - held_t["hazards"][:, 16:32]).abs().mean(),
        "probe_gold_error": (hazards[:, 32:48] - held_t["hazards"][:, 32:48]).abs().mean(),
        "probe_cell_acc": (out[:, 48:64].argmax(1) == held_t["cell"]).float().mean(),
        "probe_heading_acc": (out[:, 64:68].argmax(1) == held_t["heading"]).float().mean(),
        "probe_danger_error": (danger - exact).abs().mean(),
        "probe_certain_death": danger[certain].mean(),
        "head_danger_error": (head - exact).abs().mean(),
        "head_safe": head[safe].mean(),
        "head_certain_death": head[certain].mean(),
        "head_safe_miss": exact[head < 0.02].mean(),
        "head_nll_train": observation_nll(run, train, slice(0, 1024), num_a, num_o),
        "head_nll_held": observation_nll(run, held, slice(None), num_a, num_o),
    }
    acting.train()
    return {name: float(value) for name, value in results.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="runs/wumpus/north50_seed0/.hydra/config.yaml")
    parser.add_argument("--representation", choices=("jepa", "decoder"), required=True)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=6400)
    parser.add_argument("--pool-cache", type=Path, required=True)
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay (default: the config's).")
    parser.add_argument("--train-games", type=int, default=None,
                        help="Train on the first n games of each half of the pool (2n games; default: all).")
    parser.add_argument("--planning-head", action="store_true", help="Also score a decoupled planning head (section 2d).")
    parser.add_argument("--train-buffer", type=Path, default=None, help="Train on this run's replay buffer (2e).")
    parser.add_argument("--buffer-schedule", choices=("uniform", "growing"), default="uniform",
                        help="With --train-buffer: sample the whole buffer (2e) or the agent's growing buffer (2f).")
    parser.add_argument("--reset-at", type=int, nargs="*", default=[],
                        help="Re-initialise every network after these schedule updates (2g).")
    parser.add_argument("--catch-up", type=int, default=0, help="Uniform updates right after each reset (2g).")
    parser.add_argument("--agent-reference", type=Path, default=None,
                        help="Score this run's networks (--agent-checkpoint) without training (2e).")
    parser.add_argument("--agent-checkpoint", default="best.pt", help="Checkpoint of --agent-reference (2e).")
    parser.add_argument("--save", type=Path, default=None, help="Write run.networks' states here after training.")
    parser.add_argument("--extra-games", type=Path, nargs="+", default=None,
                        help="With --train-buffer: append these coverage_games.py game files to the buffer (2j; "
                             "several files = the union arm E0).")
    parser.add_argument("--bootstrap", type=int, default=None,
                        help="With a uniform --train-buffer: resample its games with replacement, seed j (2k).")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Config overrides (OmegaConf dotlist), e.g. model.value_equivalent_weight=1.0 (2i).")
    parser.add_argument("--score-at", type=int, nargs="*", default=[],
                        help="Also score at these update counts (lines 'RESULT@<updates> name value'); with --save, "
                             "also save the networks there as <save stem>_u<updates>.pt (section 2m).")
    args = parser.parse_args()

    device = torch.device("cuda")
    cfg = OmegaConf.load(args.config)
    cfg.model.representation, cfg.model.observation_horizon, cfg.seed = args.representation, args.horizon, args.seed
    if args.weight_decay is not None:
        cfg.trainer.weight_decay = args.weight_decay
    for item in args.set:
        key = item.split("=", 1)[0]
        if OmegaConf.select(cfg, key) is None:
            raise KeyError(f"--set {item}: {key} is not in the config.")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.set))
    setup, run = build_training_run(cfg, device)
    world = setup.domain
    num_a, num_o = world.num_actions, world.num_observations
    length = cfg.env.max_steps
    if args.pool_cache.exists():
        cached = torch.load(args.pool_cache, map_location=device)
        if cached["length"] != length or "rewards" not in cached["train"]:
            raise ValueError(f"{args.pool_cache} does not match ({cached['length']} steps, keys {list(cached['train'])}).")
        train, held = cached["train"], cached["held"]
    else:
        train, held = pool(world, 4096, length, 2026, device), pool(world, 512, length, 2027, device)
        torch.save({"games": 4096, "length": length, "train": train, "held": held}, args.pool_cache)
    half = train["actions"].shape[0] // 2
    fit_rows = (torch.arange(train["actions"].shape[0], device=device) if args.train_games is None else
                torch.cat([torch.arange(args.train_games, device=device), half + torch.arange(args.train_games, device=device)]))
    data = train
    if args.train_buffer is not None:                                      # section 2e
        if args.train_games is not None:
            raise ValueError("--train-games selects pool games; it does not apply to --train-buffer.")
        buf = load_checkpoint(args.train_buffer / "checkpoints" / "latest.pt")["buffer"]
        size = buf["size"]
        data = {"actions": buf["actions"][:size].to(device), "observations": buf["observations"][:size].to(device),
                "rewards": buf["rewards"][:size].to(device), "initial": buf["initial_observations"][:size].to(device)}
        if data["actions"].shape[1] != length:
            raise ValueError(f"buffer episodes have {data['actions'].shape[1]} steps, the pool {length}.")
        del buf
        for extra_path in args.extra_games or []:                         # section 2j
            extra = torch.load(extra_path, map_location=device)
            if extra["actions"].shape[1] != length:
                raise ValueError(f"{extra_path} has {extra['actions'].shape[1]}-step games, the buffer {length}.")
            for key, source in (("actions", "actions"), ("observations", "observations"), ("rewards", "rewards"),
                                ("initial", "initial_observations")):
                data[key] = torch.cat([data[key], extra[source].to(device)])
            print(f"extra games: {extra['actions'].shape[0]} from {extra_path} (arm {extra['arm']}, "
                  f"{extra['overrides']} overrides)", flush=True)
            del extra
        fit_rows = torch.arange(data["actions"].shape[0], device=device)
    if args.extra_games is not None and (args.train_buffer is None or args.buffer_schedule != "uniform"):
        raise ValueError("--extra-games extends a uniform --train-buffer retrain (2j).")
    if args.bootstrap is not None:                                         # section 2k
        if args.train_buffer is None or args.buffer_schedule != "uniform":
            raise ValueError("--bootstrap resamples a uniform --train-buffer retrain (2k).")
        resample = torch.Generator(device=device).manual_seed(args.bootstrap)
        fit_rows = fit_rows[torch.randint(0, len(fit_rows), (len(fit_rows),), device=device, generator=resample)]
        print(f"bootstrap {args.bootstrap}: {len(fit_rows)} games resampled with replacement, "
              f"{len(torch.unique(fit_rows))} distinct", flush=True)
    if args.reset_at and args.buffer_schedule != "growing":
        raise ValueError("--reset-at emulates the agent's schedule; it needs --buffer-schedule growing.")
    if args.buffer_schedule == "growing" and (args.train_buffer is None or not cfg.training.store_eval_episodes):
        raise ValueError("--buffer-schedule growing needs --train-buffer of a run that stored its greedy games.")
    if args.agent_reference is not None:                                   # section 2e
        if args.train_buffer is not None:
            raise ValueError("--agent-reference scores a trained run; it takes no training data.")
        networks = load_checkpoint(args.agent_reference / "checkpoints" / args.agent_checkpoint)["networks"]
        for name, net in run.networks.items():
            net.load_state_dict(networks[name])
        args.updates = 0
    print(f"arm representation={args.representation} horizon={args.horizon} seed={args.seed} "
          f"weight_decay={cfg.trainer.weight_decay}; {len(fit_rows)} training games"
          + (f" from the buffer of {args.train_buffer}" if args.train_buffer is not None else "")
          + (f"; agent reference {args.agent_reference} {args.agent_checkpoint}" if args.agent_reference is not None else ""),
          flush=True)

    t0 = time.time()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for update in range(1, args.updates + 1):
        available = len(fit_rows)
        if args.buffer_schedule == "growing":                              # section 2f
            c = (update - 1) // cfg.training.updates_per_collection + 1
            available = c * cfg.training.env_batch_size + ((c - 1) // cfg.training.eval_every) * cfg.training.eval_episodes
            if available > len(fit_rows):
                raise ValueError(f"update {update} needs {available} games, the buffer holds {len(fit_rows)}.")
        rows = fit_rows[torch.randint(0, available, (cfg.training.batch_size,), device=device, generator=generator)]
        metrics = run.trainer.train_step(EpisodeBatch(data["actions"][rows], data["observations"][rows],
                                                      data["rewards"][rows], data["initial"][rows]))
        run._update_acting()
        if update in args.reset_at:                                        # section 2g
            reset_cfg = cfg.copy()
            reset_cfg.seed = args.seed + 1000 * (args.reset_at.index(update) + 1)
            del run
            torch.cuda.empty_cache()
            _, run = build_training_run(reset_cfg, device)
            for _ in range(args.catch_up):
                rows = fit_rows[torch.randint(0, available, (cfg.training.batch_size,), device=device,
                                              generator=generator)]
                run.trainer.train_step(EpisodeBatch(data["actions"][rows], data["observations"][rows],
                                                    data["rewards"][rows], data["initial"][rows]))
                run._update_acting()
            print(f"reset after update {update}: {args.catch_up} catch-up updates on {available} games", flush=True)
        if update in args.score_at:
            score_both(run, train, held, fit_rows, num_a, num_o, cfg, device, args.planning_head, f"RESULT@{update}")
            if args.save is not None:                                       # section 2m: checkpoints along one run
                path = args.save.with_name(f"{args.save.stem}_u{update}{args.save.suffix}")
                torch.save({name: net.state_dict() for name, net in run.networks.items()}, path)
        if update % 1600 == 0:
            print(f"update {update}: " + " ".join(f"{k} {v:.4f}" for k, v in metrics.items() if k.startswith("loss")),
                  flush=True)
    print(f"training took {time.time() - t0:.0f}s", flush=True)
    if args.save is not None:
        torch.save({name: net.state_dict() for name, net in run.networks.items()}, args.save)

    score_both(run, train, held, fit_rows, num_a, num_o, cfg, device, args.planning_head, "RESULT")

if __name__ == "__main__":
    main()
