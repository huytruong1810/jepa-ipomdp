# ABSOLUTE PATH: main.py
# ==============================================================================
# TRAINING ENTRY POINT
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Thin Shell Over TrainingRun:
#    - All experiment state and logic live in ipomdp.training.TrainingRun (build, collect,
#      train, evaluate, state_dict). This file only reads the Hydra config, wires telemetry,
#      schedules evaluation / checkpointing / visualisation, and handles SIGINT.
#
# 2. One Directory per Run:
#    - Every artifact of a run (checkpoints, TensorBoard events, plots, main.log) is written
#      under Hydra's run output directory, so runs never overwrite each other. A run is
#      continued bit-for-bit with `resume=<run dir>/checkpoints/latest.pt` (which restores the
#      replay buffer and every random generator, see training/run.py).
#
# 3. Single Source of Truth for the Domain:
#    - The FinitePOMDP supplies |A|, |O|, action/observation names, the discount and the
#      value bound; nothing domain-specific is configured here. The config -> run mapping
#      (domain builders, typed RunConfig) lives in ipomdp.experiments, shared with analyze.py
#      and sweep.py.
#
# 4. Failure Is Loud, Shutdown Is Graceful:
#    - Errors (including non-finite losses) propagate and stop the run. SIGINT writes
#      latest.pt, closes the TensorBoard writer and exits with status 130; a guardrail abort
#      writes latest.pt (the guardrail itself writes emergency.pt) and exits with status 1.
#      Only a run that completed every collection exits with 0, which is what sweep.py relies
#      on before analysing a run.
#
# 5. Checkpoint Selection (training.checkpoint_selection; training/checkpointing.py, 2b):
#    - best_eval (the default; canonical Tiger and every run before 2026-10-27): best.pt holds the
#      networks of the highest greedy evaluation so far.
#    - final_retrain (whole-buffer retrain recipes only): best.pt holds the networks of the final
#      retrain (the evaluation of the last collection); best_eval.pt holds what best_eval would
#      have selected, so both rules are read from one run. The evaluation and its logging are the
#      same under both rules: the score informs, it no longer selects.
#    - Evidence (Wumpus retrain recipe, runs/sweeps/wumpus_retrain, 5 development seeds, 512 paired
#      games each; reports/2026-10-02_wumpus_diagnostics/retrain_play.py on each run's latest.pt;
#      training/run.py section 14 has the registrations): the final-retrain networks beat best.pt
#      by +0.0 / +115.0 / +27.9 / +63.6 / +95.6 per game (seeds 0-2 found after the fact, seeds
#      3-4 a pre-registered confirmation that passed), mean +60.4 [+1.7, +119.1] (Student-t over
#      seeds), 260.6 +- 30.4 against 200.2 per game, deaths down on every seed that died.
#    - Mechanism: the maximum of noisy 128-game evaluations is a winner's curse (optimizer's curse,
#      Smith & Winkler 2006). Between retrains the networks re-degrade (collections 55-95 had deaths
#      up to 25% on one seed; training/run.py section 14), so the evaluations differ mostly by luck, and the argmax picks a lucky mid-run
#      checkpoint over the final retrain, which trains longest and uniformly on the whole buffer. The
#      same selection effect makes the planner pick the risks its model under-rates.
#    - best_eval stays the default: Tiger has no retrain, and its gate (Phase 8) was read with it.
# ==============================================================================

import logging
from pathlib import Path
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from ipomdp.experiments import build_training_run, latent_figure
from ipomdp.telemetry import (
    ExecutionGuardrail,
    MCTSGraphVisualizer,
    MetricsLogger,
    PipelineProfiler,
    RewardTrajectoryVisualizer,
    SystemTelemetryMonitor,
)
from ipomdp.training import (CheckpointSelection, TrainingRun, check_checkpoint_selection, load_checkpoint, play_episodes,
                             save_checkpoint, selects_best, selects_best_eval)

torch.set_float32_matmul_precision('high')

def visualize(run: TrainingRun, collection: int, plots: Path, metrics_logger: MetricsLogger) -> None:
    """Latent geometry, search tree and cumulative rewards of greedy evaluation episodes."""
    domain = run.domain
    episodes, _ = play_episodes(run.eval_env, run.eval_agent)
    figure = latent_figure(domain, run.acting_filter, episodes, plots / "geometry", f"geometry_c{collection}")
    metrics_logger.log_figure("Visuals/belief_geometry", figure, collection)
    plt.close(figure)
    MCTSGraphVisualizer(save_dir=str(plots / "trees")).visualize(
        run.eval_agent.planner.roots[0], domain.discount, domain.action_names, domain.observation_names,
        filename=f"mcts_tree_c{collection}")
    action_map = dict(enumerate(domain.action_names))
    trajectories = [{"actions": acts, "rewards": rews, "cum_rewards": np.concatenate([[0.0], np.cumsum(rews)]).tolist()}
                    for acts, rews in zip(episodes.actions[:16].tolist(), episodes.rewards[:16].tolist())]
    figure = RewardTrajectoryVisualizer(save_dir=str(plots)).plot_cumulative_rewards(
        trajectories=trajectories, filename=f"eval_cumulative_rewards_c{collection}",
        title_suffix=f"Collection {collection:,}", action_map=action_map)
    metrics_logger.log_figure("Visuals/eval_cumulative_reward", figure, collection)
    plt.close(figure)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = logging.getLogger("ipomdp")  # Hydra writes it to the console and <run dir>/main.log
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, run = build_training_run(cfg, device)
    if cfg.resume is not None:
        run.load_state_dict(load_checkpoint(Path(cfg.resume)))
        logger.info(f"Resumed from {cfg.resume} at collection {run.collection}")
    logger.info(f"Run directory: {run_dir} | device: {device} | seed: {cfg.seed}")

    checkpoints = run_dir / "checkpoints"
    metrics_logger = MetricsLogger(log_dir=str(run_dir / "tensorboard"))
    system_monitor = SystemTelemetryMonitor(thermal_threshold_c=cfg.guardrail.thermal_trip_c)
    profiler = PipelineProfiler(window_size=100)
    guardrail = ExecutionGuardrail(
        logger=logger, system_monitor=system_monitor, thermal_trip_c=cfg.guardrail.thermal_trip_c,
        thermal_recovery_c=cfg.guardrail.thermal_recovery_c, vram_trip_mb=cfg.guardrail.vram_trip_mb,
        rss_trip_mb=cfg.guardrail.rss_trip_mb,
        emergency_save_fn=lambda: save_checkpoint(checkpoints / "emergency.pt", run.state_dict()))

    t = cfg.training
    total_collections = t.total_episodes // t.env_batch_size
    transitions_per_collection = t.env_batch_size * cfg.env.max_steps
    selection = CheckpointSelection(t.checkpoint_selection)                                     # section 5
    check_checkpoint_selection(selection, total_collections, t.eval_every, t.retrain_every)
    pbar = tqdm(total=total_collections, initial=run.collection, desc="Collections")
    try:
        while run.collection < total_collections:
            with profiler.profile("collect_and_train"):
                metrics = run.collect_and_train()
            profiler.record_step(transitions=transitions_per_collection, train_steps=t.updates_per_collection)
            collection = run.collection
            metrics_logger.log_metrics(metrics, collection, prefix="Train")

            if collection % t.eval_every == 0:
                with profiler.profile("evaluate"):
                    evaluation = run.evaluate()
                metrics_logger.log_metrics(evaluation, collection, prefix="Eval")
                outcomes = "".join(f", {k.removeprefix('eval_')} {v:.3f}" for k, v in evaluation.items()
                                   if k.removeprefix("eval_") in run.domain_metric_names)
                logger.info(f"collection {collection}: greedy eval discounted return "
                            f"{evaluation['eval_discounted_return']:.2f} +- {evaluation['eval_discounted_return_stderr']:.2f} "
                            f"(best {run.best_eval_return:.2f}){outcomes}")
                eval_return = evaluation["eval_discounted_return"]
                selected = {"collection": collection, "eval_return": eval_return,
                            "networks": {name: net.state_dict() for name, net in run.networks.items()}}
                if selects_best(selection, collection, total_collections, eval_return, run.best_eval_return):
                    save_checkpoint(checkpoints / "best.pt", selected)
                if selection is CheckpointSelection.FINAL_RETRAIN and selects_best_eval(eval_return, run.best_eval_return):
                    save_checkpoint(checkpoints / "best_eval.pt", selected)                     # section 5
                pbar.set_postfix({"eval": f"{evaluation['eval_discounted_return']:.2f}",
                                  "best": f"{run.best_eval_return:.2f}", "loss": f"{metrics['loss_total']:.3f}"})
            if collection % t.viz_every == 0:
                visualize(run, collection, run_dir / "plots", metrics_logger)
            if collection % t.save_every == 0:
                save_checkpoint(checkpoints / "latest.pt", run.state_dict())

            metrics_logger.log_metrics(system_monitor.get_metrics(), collection, prefix="System")
            metrics_logger.log_metrics(profiler.get_all_metrics(), collection, prefix="Profiler")
            metrics_logger.log_metrics(guardrail.get_incident_metrics(), collection, prefix="Guardrail")
            if not guardrail.check_system_health(collection):
                logger.critical("[!] ExecutionGuardrail triggered emergency abort. Halting training.")
                save_checkpoint(checkpoints / "latest.pt", run.state_dict())
                metrics_logger.close()
                sys.exit(1)
            pbar.update(1)
        save_checkpoint(checkpoints / "latest.pt", run.state_dict())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: saving latest.pt and shutting down.")
        save_checkpoint(checkpoints / "latest.pt", run.state_dict())
        metrics_logger.close()
        sys.exit(130)
    metrics_logger.close()


if __name__ == "__main__":
    main()
