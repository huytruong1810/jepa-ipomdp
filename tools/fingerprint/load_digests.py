# ABSOLUTE PATH: tools/fingerprint/load_digests.py
"""
Load gate: every stored run under runs/ reloads strictly through experiments.load_trained_run, and the SHA-256 of
its loaded networks equals the recorded digest.

    python tools/fingerprint/load_digests.py record  <out.json>     # digests of every stored run at this commit
    python tools/fingerprint/load_digests.py check   <ref.json>     # exit 1 on any missing, failing or changed run
"""
# ==============================================================================
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS
# ==============================================================================
# 1. What the Gate Protects:
#    - Every analysis, readout and ledger producer reloads networks from <run>/checkpoints/best.pt into the networks
#      that build_training_run constructs from the run's saved .hydra/config.yaml, with strict load_state_dict.
#      A refactor that renames a network, moves a module under a new parent or changes a parameter path breaks
#      that reload. A refactor that changes what the config builds (another head class, a reordered stack) can
#      still load and silently give other networks. The gate catches both: the load must succeed, and the hash of
#      the loaded tensors (names, dtypes, shapes, bytes, in sorted order) must equal the digest recorded before the
#      refactor.
#
# 2. Which Runs:
#    - Every directory below runs/ that holds .hydra/config.yaml and checkpoints/best.pt, symlinks resolved and each
#      real directory once, except runs/_archive (not evidence by construction). Runs that fail to load at the
#      reference commit are recorded with their error, so the check compares like with like: a run that failed
#      before must fail the same way, and a run that loaded must load to the same bits.
#    - The R2D2-lite baseline runs (runs/baselines/<name>/seed<s>/, written by
#      reports/2026-10-19_eval_protocol/recurrent_dqn_baseline.py) are reloaded exactly as that script reloads them:
#      a RecurrentQNetwork sized from config.json, strict load of best.pt and of final.pt["state"].
# ==============================================================================

import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"


def stored_runs() -> list[Path]:
    """Real directories of every stored run with a config and a best.pt (section 2)."""
    seen: dict[Path, None] = {}
    for best in sorted(RUNS.rglob("checkpoints/best.pt")):
        if "_archive" in best.relative_to(RUNS).parts:
            continue
        run_dir = best.parent.parent
        if (run_dir / ".hydra" / "config.yaml").exists():
            seen.setdefault(run_dir.resolve())
    return list(seen)


def network_digest(run_dir: Path) -> str:
    """SHA-256 over every tensor of the networks loaded from best.pt (strict), keys in sorted order."""
    from ipomdp.experiments import load_trained_run
    trained = load_trained_run(run_dir, torch.device("cpu"))
    h = hashlib.sha256()
    for name in sorted(trained.run.networks):
        state = trained.run.networks[name].state_dict()
        for key in sorted(state):
            t = state[key].detach()
            t = t.float() if t.dtype == torch.bfloat16 else t
            h.update(f"{name}.{key}:{t.dtype}:{tuple(t.shape)}".encode())
            h.update(t.cpu().contiguous().view(-1).numpy().tobytes())
    return h.hexdigest()


def baseline_runs() -> list[Path]:
    """Real directories of every stored R2D2-lite run (section 2)."""
    return sorted({p.parent.resolve() for p in (RUNS / "baselines").glob("*/seed*/config.json")
                   if (p.parent / "best.pt").exists()})


def baseline_digest(run_dir: Path) -> str:
    """SHA-256 over the baseline network loaded strictly from best.pt and from final.pt["state"] (section 2)."""
    from dataclasses import replace
    from ipomdp.baselines import RecurrentDQNConfig, RecurrentQNetwork
    from ipomdp.domain.wumpus import NUM_OBSERVATIONS, WumpusAction
    cfg = replace(RecurrentDQNConfig(), **json.loads((run_dir / "config.json").read_text())["config"])
    h = hashlib.sha256()
    for state in (torch.load(run_dir / "best.pt", map_location="cpu"),
                  torch.load(run_dir / "final.pt", map_location="cpu")["state"]):
        network = RecurrentQNetwork(NUM_OBSERVATIONS, len(WumpusAction), cfg.hidden_dim)
        network.load_state_dict(state)
        for key, t in sorted(network.state_dict().items()):
            h.update(f"{key}:{t.dtype}:{tuple(t.shape)}".encode())
            h.update(t.detach().contiguous().view(-1).numpy().tobytes())
    return h.hexdigest()


def digests() -> dict[str, str]:
    torch.set_num_threads(1)
    result = {}
    jobs = [(run_dir, network_digest) for run_dir in stored_runs()] + [(d, baseline_digest) for d in baseline_runs()]
    for run_dir, digest_of in jobs:
        key = str(run_dir.relative_to(RUNS.resolve()))
        try:
            result[key] = digest_of(run_dir)
        except Exception as error:          # recorded, not hidden: the check requires the same failure (section 2)
            result[key] = f"FAILED {type(error).__name__}: {str(error).splitlines()[0][:200]}"
        print(f"{key}: {result[key][:24]}", file=sys.stderr, flush=True)
    return result


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in ("record", "check"):
        raise SystemExit(__doc__)
    mode, path = sys.argv[1], Path(sys.argv[2])
    current = digests()
    if mode == "record":
        path.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        print(f"recorded {len(current)} runs ({sum(v.startswith('FAILED') for v in current.values())} failing)")
        return
    reference = json.loads(path.read_text())
    bad = sorted(k for k in reference.keys() | current.keys() if reference.get(k) != current.get(k))
    for key in bad:
        print(f"DIFF {key}: reference {reference.get(key)} / now {current.get(key)}")
    print(f"LOAD GATE {'PASSED' if not bad else 'FAILED'}: {len(current)} runs, {len(bad)} differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
