#!/bin/bash
# PROTOCOL.md C-e arm 1: recipe with the plain max backup, held-out seeds 5-9 (sweep.py resumes per seed).
cd /home/andyj1810/projects/jepa_ipomdp
nice -n 10 .venv/bin/python sweep.py abl_max --seeds 5,6,7,8,9 --analysis-seed 20261019 --analysis-output analysis_heldout env=wumpus model.representation=decoder model.observation_horizon=5 training.updates_per_collection=64 training.total_episodes=6400 training.store_eval_episodes=true agent.epsilon=0.25 agent.epsilon_min=0.1 model.value_estimator=max training.retrain_every=50 training.retrain_updates=6400
