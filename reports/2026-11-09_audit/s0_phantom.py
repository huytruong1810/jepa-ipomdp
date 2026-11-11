# ABSOLUTE PATH: reports/2026-11-09_audit/s0_phantom.py
"""
PHANTOM readout of the Stage-0 retrain screen (objective_ab.py 2r, PROTOCOL.md C-h): impossible values in the planner's
imagined children, and the agreement of the value pair's heads, for saved network sets.

DESIGN DECISIONS & THEORETICAL FOUNDATIONS
==============================================================================
# 1. Quantities (registered in objective_ab.py 2r before any run):
#    - From the run's stored buffer (runs/sweeps/wumpus_retrain/seed<s>, latest.pt buffer), 2000 live latents of the
#      ACTING filter (the planner's), drawn with a fixed generator (seed 0) so every arm sees the same histories.
#    - The evaluation planner's model expands them; among the children with P(o | z, a) >= 1e-3 (the planner's prune),
#      the share whose decoded value exceeds 999 (no game pays more) is PHANTOM. Also reported: the share of the 2000
#      latents themselves above 999, and for a value pair corr(V_A, V_B) and mean |V_A - V_B| on those latents.
#    - Post-game latents (after GAME_OVER) are excluded, as in the audit's Appendix M.
# 2. Usage: uv run python reports/2026-11-09_audit/s0_phantom.py <run dir> <saved .pt> [<saved .pt> ...] [--set k=v ...]
#    Every saved set of one call shares the overrides (saved_networks.host_run), as in retrain_play.py. CPU only.
# ==============================================================================
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-10-02_wumpus_diagnostics"))
from saved_networks import host_run, in_layout  # noqa: E402

from ipomdp.experiments.runs import load_trained_run  # noqa: E402
from ipomdp.training.checkpointing import load_checkpoint  # noqa: E402

GAME_OVER, LATENTS, PRUNE, CEILING, CHUNK = 32, 2000, 1e-3, 999.0, 250


def main() -> None:
    args = sys.argv[1:]
    overrides = []
    while "--set" in args:
        i = args.index("--set")
        overrides.append(args[i + 1])
        del args[i:i + 2]
    run_dir, saved_sets = Path(args[0]), [Path(a) for a in args[1:]]
    torch.set_num_threads(4)
    dev = torch.device("cpu")
    trained = load_trained_run(run_dir, dev)
    run = host_run(trained, overrides, dev)
    buffer = load_checkpoint(run_dir / "checkpoints" / "latest.pt")["buffer"]
    n = buffer["size"]
    acts, obs, init = (buffer[k][:n].cpu() for k in ("actions", "observations", "initial_observations"))
    num_a, num_o = run.trainer.num_actions, run.trainer.num_observations
    episodes = torch.randperm(n, generator=torch.Generator().manual_seed(0))[:400]
    post = torch.cat([init[episodes, None], obs[episodes, :-1]], 1) == GAME_OVER
    print(f"{run_dir}: overrides {overrides}; codec range +-{run.codec.max_abs_value:.0f}")
    for saved in saved_sets:
        states = in_layout(torch.load(saved, map_location=dev), run.networks)
        for name, net in run.networks.items():
            net.load_state_dict(states[name])
        with torch.no_grad():
            z = run.acting_filter.unroll(F.one_hot(acts[episodes], num_a).float(), F.one_hot(obs[episodes], num_o).float(),
                                         F.one_hot(init[episodes], num_o).float())[:, :-1][~post]
            z = z[torch.randperm(len(z), generator=torch.Generator().manual_seed(1))[:LATENTS]]
            model = run.eval_agent.model
            over, kept = 0, 0
            for s in range(0, len(z), CHUNK):
                e = model.expand(z[s:s + CHUNK])
                mask = e.observation_probs >= PRUNE
                over += int((e.next_values[mask] > CEILING).sum())
                kept += int(mask.sum())
            line = (f"  {saved.stem:18s} PHANTOM {over / kept:.4%} of {kept} imagined children > {CEILING:.0f}; "
                    f"latents > {CEILING:.0f}: {(model.values(z) > CEILING).float().mean():.4%}")
            head = run.acting_value_head
            if hasattr(head, "head_values"):
                va, vb = head.head_values(z, run.codec)
                line += f"; corr(V_A, V_B) {torch.corrcoef(torch.stack([va, vb]))[0, 1]:.4f}, mean |A-B| {(va - vb).abs().mean():.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
