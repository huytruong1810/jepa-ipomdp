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

## Files

| Phase | File | Findings | Change | Check |
|---|---|---|---|---|
| 1 domain and I-POMDP core | `src/ipomdp/domain/__init__.py` | | | |
| 1 domain and I-POMDP core | `src/ipomdp/domain/belief.py` | | | |
| 1 domain and I-POMDP core | `src/ipomdp/domain/env.py` | | | |
| 1 domain and I-POMDP core | `src/ipomdp/domain/pomdp.py` | | | |
| 1 domain and I-POMDP core | `src/ipomdp/domain/solver.py` | | | |
| 1 domain and I-POMDP core | `src/ipomdp/domain/tiger.py` | | | |
| 1 domain and I-POMDP core | `src/ipomdp/domain/wumpus.py` | | | |
| 1 domain and I-POMDP core | `tests/test_domain.py` | | | |
| 1 domain and I-POMDP core | `tests/test_wumpus.py` | | | |
| 2 belief filter | `src/ipomdp/models/__init__.py` | | | |
| 2 belief filter | `src/ipomdp/models/ema.py` | | | |
| 2 belief filter | `src/ipomdp/models/history_filter.py` | | | |
| 2 belief filter | `src/ipomdp/models/layers.py` | | | |
| 2 belief filter | `src/ipomdp/models/world_model.py` | | | |
| 2 belief filter | `tests/test_history_filter.py` | | | |
| 2 belief filter | `tests/test_models.py` | | | |
| 3 world model, heads, losses | `src/ipomdp/models/distributions.py` | | | |
| 3 world model, heads, losses | `src/ipomdp/models/heads.py` | | | |
| 4 planner and agents | `src/ipomdp/agents/__init__.py` | | | |
| 4 planner and agents | `src/ipomdp/agents/planning_agent.py` | | | |
| 4 planner and agents | `src/ipomdp/agents/uniform_agent.py` | | | |
| 4 planner and agents | `src/ipomdp/agents/wumpus_reference.py` | | | |
| 4 planner and agents | `src/ipomdp/planning/__init__.py` | | | |
| 4 planner and agents | `src/ipomdp/planning/mcts.py` | | | |
| 4 planner and agents | `src/ipomdp/planning/search_model.py` | | | |
| 4 planner and agents | `src/ipomdp/planning/wumpus_exact_model.py` | | | |
| 4 planner and agents | `tests/test_planner_and_agent.py` | | | |
| 5 training loop, remedies, baseline | `src/ipomdp/baselines/__init__.py` | | | |
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
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `analyze.py` | | | |
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
| 7 experiments, entry points, telemetry, reports, runs, queue, packaging | `src/ipomdp/experiments/runs.py` | | | |
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
