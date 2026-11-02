#!/bin/bash
# PROTOCOL.md C-e arm 3: recipe with the JEPA representation, held-out seeds 5-9 (CPU smoke 2026-10-31: JEPA runs
# through two whole-buffer retrains without error).
cd /home/andyj1810/projects/jepa_ipomdp
source runs/week/sweep_retry.sh
sweep_retry abl_jepa --seeds 5,6,7,8,9 --analysis-seed 20261019 --analysis-output analysis_heldout env=wumpus model.representation=jepa model.observation_horizon=5 training.updates_per_collection=64 training.total_episodes=6400 training.store_eval_episodes=true agent.epsilon=0.25 agent.epsilon_min=0.1 model.value_estimator=double training.retrain_every=50 training.retrain_updates=6400
