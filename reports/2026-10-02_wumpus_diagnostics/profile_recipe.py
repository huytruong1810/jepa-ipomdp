# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/profile_recipe.py
# ==============================================================================
# WHERE THE WALL TIME OF ONE RECIPE COLLECTION GOES (SEARCH AND UPDATES)
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Measurement before optimisation (2026-11-15):
#    - A recipe seed takes about 2.5 h on a free machine; telemetry times play, updates and the
#      whole-buffer retrain together (Profiler/collect_and_train), so it cannot say how the time
#      splits between the Python tree search, the model's expand on the GPU and the updates.
#    - This script loads a finished recipe run (its config and latest.pt: networks, optimizer,
#      buffer, generator states), plays ONE exploring collection (env_batch_size games, the
#      training planner) under cProfile, then runs N updates under cProfile. Nothing is written
#      to the run directory.
#    - Run it with CUDA_LAUNCH_BLOCKING=1 (as every GPU stage is since 2026-11-14): kernel
#      launches are then synchronous, so cProfile attributes GPU time to the Python call that
#      launched it. Without it, GPU time piles up at the next host synchronisation
#      (.cpu(), .item()), which the report would misread as copy time.
#    - It replaces profile_train_step.py (2026-10-04), which timed one update of an older
#      configuration with six jobs sharing the GPU (950 ms per update; no longer representative).
# 2. Usage (GPU, about 3 minutes; through the queue contract):
#      bash tools/fingerprint/gpu_run.sh "profile recipe" .venv/bin/python \
#          reports/2026-10-02_wumpus_diagnostics/profile_recipe.py runs/sweeps/wumpus_retrain/seed0 [updates]
# ==============================================================================

import cProfile, io, pstats, sys, time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ipomdp.experiments import build_training_run
from ipomdp.training import load_checkpoint, play_episodes

torch.set_float32_matmul_precision("high")   # as main.py
run_dir, updates = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 32
device = torch.device("cuda")
_, run = build_training_run(OmegaConf.load(run_dir / ".hydra" / "config.yaml"), device)
run.load_state_dict(load_checkpoint(run_dir / "checkpoints" / "latest.pt"))


def profiled(label: str, fn) -> None:
    """Runs fn under cProfile and prints wall time and the 28 largest entries by internal time."""
    profile = cProfile.Profile()
    torch.cuda.synchronize(); start = time.perf_counter()
    profile.enable(); fn(); profile.disable()
    torch.cuda.synchronize(); wall = time.perf_counter() - start
    out = io.StringIO()
    pstats.Stats(profile, stream=out).sort_stats("tottime").print_stats(28)
    print(f"===== {label}: {wall:.1f} s wall (cProfile adds overhead; compare shares, not totals) =====")
    print("\n".join(line for line in out.getvalue().splitlines() if line.strip())[:6000])


for _ in range(2):                                         # warm the kernels and the allocator
    run.trainer.train_step(run.buffer.sample(run.cfg.batch_size)); run._update_acting()
start = time.perf_counter(); play_episodes(run.env, run.train_agent); torch.cuda.synchronize()
print(f"one exploring collection without the profiler: {time.perf_counter() - start:.1f} s")
profiled("one exploring collection (search + filter steps)", lambda: play_episodes(run.env, run.train_agent))


def train() -> None:
    for _ in range(updates):
        run.trainer.train_step(run.buffer.sample(run.cfg.batch_size)); run._update_acting()


start = time.perf_counter(); train(); torch.cuda.synchronize()
print(f"{updates} updates without the profiler: {(time.perf_counter() - start) / updates * 1000:.0f} ms per update")
profiled(f"{updates} updates (train_step + acting copies)", train)
print(f"peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
