# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/saved_networks.py
# ==============================================================================
# SAVED NETWORK SETS IN ANOTHER RUN'S LAYOUT (objective_ab.py 2l)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Why:
#    - retrain_play.py and feature_localise.py swap saved network sets (objective_ab.py --save) into a run's
#      planner, and require the saved keys to equal run.networks. A set trained with JEPA parts (JEPA, or the
#      decoder with the self-predictive term, training/run.py section 18) holds "world_model" (the filter, the
#      predictor and the EMA target filter) where a decoder run holds "belief_filter".
#    - The planner reads only the acting copies and the heads, which have the same keys and shapes in both
#      layouts; the predictor and target filter are training-only. So such a set plays in a decoder run once
#      "world_model" is replaced by its belief_filter part. Nothing else is renamed or dropped, and any other
#      key mismatch still raises.
#
# 2. A host run with config overrides (2026-10-27, objective_ab.py 2p; --set in retrain_play.py and feature_localise.py):
#    - A set trained with an option that changes a network's shape (model.observation_reward: the reward head reads
#      o') does not load into the run's own networks. host_run rebuilds the run from its config with the OmegaConf
#      dotlist overrides applied and returns that fresh TrainingRun; the saved set is loaded into it, and its
#      evaluation planner plays with the option's model. The best.pt replay keeps the run's own config.
# ==============================================================================

PREFIX = "belief_filter."


def host_run(trained, overrides: list[str], device):
    """trained.run, or a fresh TrainingRun of the run's config with the dotlist overrides applied (header 2)."""
    if not overrides:
        return trained.run
    from omegaconf import OmegaConf
    from ipomdp.experiments.runs import build_training_run
    cfg = OmegaConf.merge(trained.cfg, OmegaConf.from_dotlist(overrides))
    _, run = build_training_run(cfg, device)
    run.belief_filter.eval()
    return run


def in_layout(states: dict, run_keys) -> dict:
    """The saved set with "world_model" mapped to "belief_filter" when the run needs that (header 1)."""
    run_keys = set(run_keys)
    if "world_model" in states and "belief_filter" in run_keys and "world_model" not in run_keys:
        states = dict(states)
        world = states.pop("world_model")
        states["belief_filter"] = {k[len(PREFIX):]: v for k, v in world.items() if k.startswith(PREFIX)}
    if set(states) != run_keys:
        raise ValueError(f"saved networks {sorted(states)} do not match the run's {sorted(run_keys)}.")
    return states
