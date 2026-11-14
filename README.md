<!-- ABSOLUTE PATH: README.md -->

# Belief-tree search over learned latent filters

Research code for an agent that plans in partially observable domains over a belief state it learns. A recurrent latent filter takes the place of the Bayes filter, learned reward, observation and value heads take the place of the model, and a belief-tree search that branches exactly over observations plans over them. The code also holds the exact counterparts (Bayes filter, solver, exact-belief planner), which are used to measure where the learned planner goes wrong.

Two domains:

- **Tiger** (Cassandra's `tiger.95.POMDP`, γ = 0.95), where the exact optimum is known.
- **4×4 Wumpus World** (one hunter, 16,380 layouts, 33 observations, 50-step games, γ = 0.99), where the exact belief can be computed by enumerating layouts but no exact solution is available.

## Method

**Exact reference layer (`src/ipomdp/domain/`).** `FinitePOMDP` holds float64 tensors `T[a,s,s']`, `O[a,s',o]`, `R[a,s]`, `b0`, `γ` and is the single source of truth for the batched simulator (`env.py`), the exact Bayes filter (`belief.py`) and exact value iteration by incremental pruning with a certified error bound (`solver.py`). `wumpus.py` holds the Wumpus rules, simulator and the layout-enumeration belief tracker.

**Filter and one-step model (`src/ipomdp/models/`, `src/ipomdp/training/trainer.py`).**

```
z_0 = learned,      z_{t+1} = GRU([a_t, o_{t+1}], z_t)          (counterpart of the belief update)
```

Whole episodes are unrolled and trained with:

| Term | Role |
|---|---|
| representation term: next-observation cross-entropy into the filter (`model.representation=decoder`), or self-prediction of an EMA target filter's latent (`jepa`) | shapes the latent |
| optional k-step open-loop observation prediction (`model.observation_horizon=k`) | shapes the latent |
| two-hot reward cross-entropy | shapes the latent |
| observation head `P(o' \| z, a)` | chance-node probabilities for the search |
| two-hot value towards `max_a [R + γ Σ_o P(o) V̄(τ(z,a,o))]` through the learned model, slow target head, detached latents | leaf values for the search |

The two-hot codec places bins at symlog-spaced values but encodes and decodes linearly in the value, so decoded means are unbiased.

**Planning (`src/ipomdp/planning/`, `src/ipomdp/agents/`).** `BeliefTreeSearch` is PUCT at decision nodes, exact branching over observations at chance nodes, and expectimax backups. It runs over a `SearchModel`: `ExactSearchModel` (exact beliefs), `LearnedSearchModel` (filter latents and heads; a child is the filter's own update `filter.step(z, a, o')`), or `ExactWumpusSearchModel` (the exact-belief reference planner). The agent acts with Polyak-averaged copies of its networks.

**Diagnostics (`src/ipomdp/interpretability/`, `reports/`).** Probes map latents to exact beliefs; span bounds turn the decoding error into bounds on value and regret. Exact-component localisation substitutes the exact reward or observation probabilities into the learned backup at the planner's own decision points and measures what changes.

## Results

Every number below is recomputed from stored per-game files by `reports/2026-11-14_submission_audit/ledger.py`. Intervals are 95%: Student-t over training seeds for trained agents, a bootstrap over the evaluation games for agents that are not trained.

**Tiger, 10 training seeds, the same 512 games for every agent and seed** (`uv run sweep.py polyak --seeds 0,...,9`). The optimal agent earns 18.48 on these games (the solver's value at b₀ is 19.36).

| Agent | Gap to the exact optimal agent |
|---|---|
| Learned planner | +0.02 ± 0.06 |
| Decoded-belief agent, MLP probe | −0.20 ± 0.24 |
| Decoded-belief agent, linear probe | −4.7 |

With next-observation prediction as the representation term instead of latent self-prediction the gap is 0.00 ± 0.07.

**Wumpus World, 5 held-out training seeds, 512 held-out games each** (`reports/2026-10-19_eval_protocol/PROTOCOL.md`).

| Agent | Return per game | Won | Died | Unfinished |
|---|---|---|---|---|
| Random | −420 [−471, −369] | 12.3% | 58.4% | 29.3% |
| Recurrent model-free baseline (R2D2 style, same budget) | 60 [21, 100] | 11.4% | 0.0% | 88.6% |
| Learned planner | 196 [138, 254] | 28.6% | 2.0% | 69.3% |
| Exact-belief planner | 494 [454, 533] | 62.3% | 1.6% | 36.1% |
| Exact-belief heuristic (hand-written) | 580 [540, 618] | 71.3% | 2.0% | 26.8% |

- The learned planner is +135 per game [48, 222] above the model-free baseline and −298 [−356, −240] below the exact-belief planner. Its shortfall is in unfinished games.
- With the exact one-step model (reward and observation probabilities) at every node of its tree and its own filter and value kept, it earns 363 per game: +168 [131, 204], positive on every seed, no deaths.
- Leave-one-out ablations on the held-out seeds (maximum backup instead of the double value, no whole-buffer retraining, latent self-prediction as the one-step representation term) change the return by 34 to 53 per game, none distinguishable from zero over five seeds. Whole-buffer retraining improves how well position and heading are decoded from the latent on every seed.
- Decision-aware objectives for the observation model and several ways of adding data did not repair the observation-model error reliably; the scripts and their registered tests are in `reports/2026-10-02_wumpus_diagnostics/`.

## Usage

The project uses [uv](https://docs.astral.sh/uv/) with Python 3.12 and CUDA 12.8 torch. `./recreate_venv.sh` rebuilds `.venv` from `uv.lock`.

```bash
uv run main.py                                   # train on Tiger (Hydra); writes runs/<env>/<timestamp>_seed<seed>/
uv run main.py env=wumpus                        # train on Wumpus World
uv run main.py seed=1 training.total_episodes=1280 mcts.num_simulations=20   # any config key can be overridden
uv run main.py resume=runs/tiger/<run>/checkpoints/latest.pt                 # continue a run bit-for-bit

uv run analyze.py runs/tiger/<run>               # probes, bounds, returns, geometry -> <run>/analysis/
uv run sweep.py <name> --seeds 0,1,2,3,4 [overrides ...]   # train and analyse each seed -> runs/sweeps/<name>/

uv run pytest                                    # fast suite
uv run pytest -m slow                            # certified solve, exact planner vs optimal, world-model acceptance (GPU)
uv run python reports/2026-11-14_submission_audit/ledger.py   # recompute every reported number from stored results
```

The Wumpus agent of the table is

```bash
uv run sweep.py wumpus_retrain_heldout --seeds 5,6,7,8,9 --analysis-seed 20261019 --analysis-output analysis_heldout \
  env=wumpus model.representation=decoder model.observation_horizon=5 training.updates_per_collection=64 \
  training.total_episodes=6400 training.store_eval_episodes=true agent.epsilon=0.25 agent.epsilon_min=0.1 \
  model.value_estimator=double training.retrain_every=50 training.retrain_updates=6400
```

One training run takes about an hour on Tiger and two and a half to three hours on Wumpus World on one GPU (RTX 5080).

## Layout

```
main.py  analyze.py  sweep.py       thin entry points (train / analyse a run / multi-seed experiment)
conf/                               Hydra config (config.yaml, env/tiger.yaml, env/wumpus.yaml)
src/ipomdp/
  domain/            FinitePOMDP, Tiger, Wumpus World, simulators, exact Bayes filters, exact solver
  models/            latent filter, predictor, value/reward/observation heads, two-hot codec
  planning/          belief-tree search; exact, learned and exact-Wumpus search models
  agents/            planning agent, uniform random agent, exact-belief Wumpus heuristic
  baselines/         recurrent model-free baseline (R2D2 style)
  training/          episode buffer, trainer, rollouts, checkpointable TrainingRun
  interpretability/  belief probes, error bounds, belief analysis
  experiments/       config -> run, run-directory reload and analysis, seed aggregation and intervals
  telemetry/         TensorBoard, visualizers, system monitor, profiler, guardrails
reports/
  2026-10-19_eval_protocol/        frozen evaluation protocol, its registered claims and their readouts
  2026-10-02_wumpus_diagnostics/   localisation and remedy screens (each script's header states its registered test)
  2026-11-03_paper_figures/        figure scripts
  2026-11-14_submission_audit/     number ledger, per-seed statistics, audit log, artifact manifest
tests/
```

Every source file opens with a `DESIGN DECISIONS & THEORETICAL FOUNDATIONS` block that records why the module is built the way it is, including approaches that were tried and failed. Options that belong to remedies which failed their screens are kept, default off, so that those screens can be rerun.
