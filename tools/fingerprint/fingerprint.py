# ABSOLUTE PATH: tools/fingerprint/fingerprint.py
"""
Bit-identity fingerprints of the training paths: one small, fully seeded run per case, printed so that two code
versions are equivalent on a path exactly when their outputs are byte-equal.

    python tools/fingerprint/fingerprint.py --list
    python tools/fingerprint/fingerprint.py --case tiger [--code-root <checkout>] [--device cpu|cuda] [--out <dir>]
    python tools/fingerprint/fingerprint.py --case cpu --out tests/golden/fingerprints     # every CPU case
    python tools/fingerprint/fingerprint.py --case wumpus_gpu --device cuda --run-dir <stored recipe run>
"""
# ==============================================================================
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS
# ==============================================================================
# 1. What a Fingerprint Proves:
#    - A case trains a tiny run (collections, greedy evaluations, and where the case asks for it a whole-buffer
#      retrain) and prints every metric with repr (full float precision), the parameter sum of every network, and a
#      SHA-256 over every tensor of the run's complete state (networks, optimizer moments, replay buffer, every
#      generator state). Equal output means the two code versions took the same random draws, summed in the same
#      order and produced the same bits on that path. It is the gate for every refactor that must leave cited runs
#      reproducible (efficiency rule: bit-identical changes are free, anything that changes rounding needs approval).
#    - A metric-only comparison is too weak (an earlier check "passed" on two empty files, and a change to an unused
#      buffer field would not show in the metrics); the state hash catches what the metrics do not.
#
# 2. Cases Are Hydra Overrides on the Real conf/:
#    - Every case is composed exactly as main.py composes a run (conf/config.yaml, the env group, then overrides) and
#      built by experiments.build_training_run. The case therefore survives refactors of the Python-side config
#      (RunConfig grouping, remedy modules) as long as the YAML keys stay byte-stable, which is itself a rule
#      (stored runs reload from their saved .hydra/config.yaml).
#    - --code-root selects the checkout whose conf/ is composed; the caller puts the same checkout's src/ first on
#      PYTHONPATH (compare.sh does both), so a base commit and the working tree are fingerprinted by one harness.
#
# 3. Every Remedy Option Has Its Own Case, and Each Case Must Reach Its Code:
#    - The default-off options back reported negative results and stay reproducible from main; the remedy seam
#      moves their code, so each one needs its own proof. A case runs warm-up (one collection), a greedy evaluation
#      (so n-step targets see greedy rows when the case stores evaluation games), two planner collections (the second
#      one retrains when retrain_every = 2) and a final evaluation.
#    - tests/test_fingerprints.py also requires each option case's output to differ from its base case's: an option
#      whose case prints the base output never ran, and its fingerprint would prove nothing.
#
# 4. Determinism Settings:
#    - torch.set_num_threads(1) and torch.manual_seed(0) at the start of every case: CPU reductions can change with
#      the thread count, and the golden files must not depend on how many cores the machine had free.
#    - On CUDA the harness sets the TF32 matmul precision main.py uses; the queue exports CUDA_LAUNCH_BLOCKING=1
#      (same kernels and numerics). GPU outputs are only comparable on the same device and torch build.
# ==============================================================================

import argparse
import hashlib
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig

REPO = Path(__file__).resolve().parents[2]

# Small Tiger run (the values of tests/test_training_run.py::_config, written as overrides of conf/config.yaml).
TIGER_TINY = ("seed=7", "env.max_steps=8", "training.env_batch_size=4", "training.warmup_episodes=4",
              "training.buffer_capacity=64", "training.batch_size=4", "training.updates_per_collection=2",
              "training.eval_episodes=4", "model.latent_dim=8", "model.hidden_dim=16", "model.acting_ema_momentum=0.9",
              "trainer.value_target_momentum=0.99", "mcts.num_simulations=4", "agent.temperature_decay=0.5")

# Small Wumpus run (the values of tests/test_wumpus.py::TestLearningStack._config; observation_prune and the
# discount come from conf/env/wumpus.yaml as in every Wumpus run).
WUMPUS_TINY = ("seed=3", "env.max_steps=12", "training.env_batch_size=4", "training.warmup_episodes=4",
               "training.buffer_capacity=32", "training.batch_size=4", "training.updates_per_collection=2",
               "training.eval_episodes=4", "model.latent_dim=16", "model.hidden_dim=32",
               "model.acting_ema_momentum=0.9", "trainer.value_target_momentum=0.99", "mcts.num_simulations=4",
               "agent.temperature_decay=0.5")

DECODER = ("model.representation=decoder",)
DOUBLE = ("model.value_estimator=double",)
RETRAIN = ("training.retrain_every=2", "training.retrain_updates=3")

# name -> (base case, extra overrides). Each option case extends one base case and must differ from it (section 3).
# The bases are the plain JEPA run, the Wumpus recipe, and w_prune: target_prune only acts on branches the planner
# prunes, and the tiny untrained head never puts a branch below the recipe's 1e-3, so its base prunes at 0.025
# (the threshold must stay below 1/|O| = 1/33). Every option case names the companions its validation requires
# (training/run.py) and nothing else.
OPTION_CASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "w_recipe": ("w_base", DECODER + DOUBLE + RETRAIN + ("model.observation_horizon=3", "agent.epsilon=0.25",
                                                         "agent.epsilon_min=0.1", "training.store_eval_episodes=true")),
    "time_aware_value": ("w_base", ("model.time_aware_value=true",)),
    "value_nstep": ("w_base", ("model.value_nstep=3", "training.store_eval_episodes=true")),
    "detached_planning_head": ("w_base", DECODER + ("model.detached_planning_head=true",)),
    "privileged_belief_loss": ("w_base", ("model.privileged_belief_loss=true",)),
    "value_equivalent_weight": ("w_base", DECODER + ("model.value_equivalent_weight=0.5",)),
    "decision_aware_weight": ("w_base", DECODER + ("model.decision_aware_weight=0.5",)),
    "observation_reward": ("w_base", ("model.observation_reward=true",)),
    "q_equivalent_weight": ("w_base", DECODER + ("model.observation_reward=true", "model.q_equivalent_weight=0.5")),
    "self_predictive_weight": ("w_base", DECODER + ("model.self_predictive_weight=0.5",)),
    "absorbing_value": ("w_base", ("model.absorbing_value=true",)),
    "trial_prob": ("w_base", ("agent.trial_prob=0.5",)),
    "risk_margin": ("w_base", ("mcts.risk_margin=2.0",)),
    "codec_bound": ("w_base", DECODER + DOUBLE + ("model.codec_bound=absorbing",)),
    "w_prune": ("w_base", ("mcts.observation_prune=0.025",)),
    "target_prune": ("w_prune", ("model.target_prune=true",)),
    "clipped_double": ("w_base", DECODER + ("model.value_estimator=clipped_double",)),
    "double_partition": ("w_base", DECODER + DOUBLE + ("model.double_partition=slot",)),
    "separate_value_clip": ("w_base", ("trainer.separate_value_clip=true",)),
    "observation_head_chain": ("w_base", ("model.observation_head=chain",)),
    "kinematic_weight": ("w_base", ("model.kinematic_weight=0.5",)),
    "filter_transformer": ("w_base", DECODER + ("model.filter=transformer",)),
    "ego_map_weight": ("w_base", ("model.ego_map_weight=0.5",)),
    "retrain_weight_decay": ("w_recipe", ("training.retrain_weight_decay=0.1",)),
}
BASE_CASES = {"w_base": ("wumpus", WUMPUS_TINY)}
CPU_CASES = ("tiger", "dqn", "w_base", *OPTION_CASES)


def compose(code_root: Path, env: str, overrides: tuple[str, ...]) -> DictConfig:
    """conf/config.yaml + env group + overrides, composed by Hydra exactly as main.py's @hydra.main does."""
    from hydra import compose as hydra_compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(code_root / "conf"), version_base=None):
        return hydra_compose(config_name="config", overrides=[f"env={env}", *overrides])


def case_overrides(name: str) -> tuple[str, tuple[str, ...]]:
    """The env and the full override list of a Wumpus case, following its base chain."""
    if name in BASE_CASES:
        return BASE_CASES[name]
    base, extra = OPTION_CASES[name]
    env, inherited = case_overrides(base)
    return env, inherited + extra


def digest(obj, h) -> None:
    """Feeds every tensor (and scalar) of a nested state into the hash, in a fixed order."""
    if isinstance(obj, torch.Tensor):
        h.update(str(obj.dtype).encode()); h.update(str(tuple(obj.shape)).encode())
        flat = obj.detach().float() if obj.dtype == torch.bfloat16 else obj.detach()
        h.update(flat.cpu().contiguous().view(-1).numpy().tobytes())
    elif isinstance(obj, dict):
        for key in sorted(obj, key=str):
            h.update(str(key).encode()); digest(obj[key], h)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            digest(item, h)
    else:
        h.update(repr(obj).encode())


def state_lines(run) -> list[str]:
    """Parameter sums of every network and the SHA-256 of the run's complete state."""
    h = hashlib.sha256()
    digest(run.state_dict(), h)
    sums = [float(sum(p.detach().double().sum() for p in net.parameters())) for net in run.networks.values()]
    return [repr(sums), f"state sha256 {h.hexdigest()}"]


def _tiger(code_root: Path, device: torch.device) -> list[str]:
    """The three Tiger configurations of the original Tiger fingerprint, in one process: JEPA + max, decoder + double,
    decoder + double with retrain; four collections each, then an evaluation."""
    from ipomdp.experiments import build_training_run
    lines = []
    for extra in ((), DECODER + DOUBLE, DECODER + DOUBLE + RETRAIN):
        _, run = build_training_run(compose(code_root, "tiger", TIGER_TINY + extra), device)
        out = [run.collect_and_train() for _ in range(4)] + [run.evaluate()]
        lines += [repr(out), *state_lines(run)]
    return lines


def _dqn(device: torch.device) -> list[str]:
    """The R2D2-lite baseline at its defaults on a tiny budget (baselines/recurrent_dqn.py)."""
    from ipomdp.baselines import RecurrentDQNConfig, train_recurrent_dqn
    from ipomdp.domain import WumpusWorld
    cfg = RecurrentDQNConfig(hidden_dim=16, batch_size=4, env_batch_size=8, total_episodes=48, warmup_episodes=16,
                             updates_per_collection=3, eval_every=2, eval_episodes=4)
    result = train_recurrent_dqn(cfg, WumpusWorld(device), 12, seed=3, device=device, log=lambda _: None)
    return [repr(result.evaluations), repr([float(v.double().sum()) for v in result.final_state.values()])]


def _option(name: str, code_root: Path, device: torch.device) -> list[str]:
    """One Wumpus option case: warm-up, evaluation, two planner collections, evaluation (section 3)."""
    from ipomdp.experiments import build_training_run
    env, overrides = case_overrides(name)
    _, run = build_training_run(compose(code_root, env, overrides), device)
    lines = [" ".join(overrides)]
    for step in ("collect", "evaluate", "collect", "collect", "evaluate"):
        metrics = run.collect_and_train() if step == "collect" else run.evaluate()
        lines.append(f"{step} {run.collection} {repr(sorted(metrics.items()))}")
    return lines + state_lines(run)


def _wumpus_gpu(run_dir: Path) -> list[str]:
    """From the full stored state of a finished recipe run: one exploring collection with its updates, one greedy
    evaluation whose games enter the buffer, and a second collection that forces a whole-buffer retrain
    (retrain_every = 51 at collection 102, 128 updates). The search then runs in its usual regime (a trained
    observation head, pruned branches)."""
    from omegaconf import OmegaConf
    from ipomdp.experiments import build_training_run
    from ipomdp.training import load_checkpoint
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    cfg.training.retrain_every, cfg.training.retrain_updates = 51, 128
    _, run = build_training_run(cfg, torch.device("cuda"))
    run.load_state_dict(load_checkpoint(run_dir / "checkpoints" / "latest.pt"))
    lines = [f"start collection {run.collection}"]
    for step in ("collect", "evaluate", "collect"):
        metrics = run.collect_and_train() if step == "collect" else run.evaluate()
        lines.append(f"{step} {run.collection} {repr(sorted(metrics.items()))}")
        print(f"{step} done (collection {run.collection})", file=sys.stderr, flush=True)
    return lines + state_lines(run)


def run_case(name: str, code_root: Path, device: torch.device, run_dir: Path | None = None) -> str:
    """The fingerprint text of one case (section 4: one thread, seed 0, main.py's matmul precision on CUDA)."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    if name == "tiger":
        lines = _tiger(code_root, device)
    elif name == "dqn":
        lines = _dqn(device)
    elif name == "wumpus_gpu":
        if run_dir is None or device.type != "cuda":
            raise ValueError("wumpus_gpu needs --device cuda and --run-dir <stored recipe run>")
        lines = _wumpus_gpu(run_dir)
    elif name in BASE_CASES or name in OPTION_CASES:
        lines = _option(name, code_root, device)
    else:
        raise KeyError(f"unknown case {name!r}; --list prints the cases")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", help="a case name, or 'cpu' for every CPU case")
    parser.add_argument("--list", action="store_true", help="print the case names")
    parser.add_argument("--code-root", type=Path, default=REPO, help="checkout whose conf/ is composed")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-dir", type=Path, help="stored recipe run (wumpus_gpu only)")
    parser.add_argument("--out", type=Path, help="write <out>/<case>.txt instead of printing")
    args = parser.parse_args()
    if args.list:
        print("\n".join((*CPU_CASES, "wumpus_gpu")))
        return
    if args.case is None:
        parser.error("--case or --list is required")
    names = CPU_CASES if args.case == "cpu" else (args.case,)
    for name in names:
        text = run_case(name, args.code_root.resolve(), torch.device(args.device), args.run_dir)
        if args.out is None:
            sys.stdout.write(text)
        else:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{name}.txt").write_text(text)
            print(f"{name}: {len(text.splitlines())} lines", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
