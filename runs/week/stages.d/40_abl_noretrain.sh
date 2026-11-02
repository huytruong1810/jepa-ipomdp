#!/bin/bash
# PROTOCOL.md C-e arm 2: recipe without whole-buffer retraining, held-out seeds 5-9.
cd /home/andyj1810/projects/jepa_ipomdp
source runs/week/sweep_retry.sh
sweep_retry abl_noretrain --seeds 5,6,7,8,9 --analysis-seed 20261019 --analysis-output analysis_heldout env=wumpus model.representation=decoder model.observation_horizon=5 training.updates_per_collection=64 training.total_episodes=6400 training.store_eval_episodes=true agent.epsilon=0.25 agent.epsilon_min=0.1 model.value_estimator=double
