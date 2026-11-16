# ABSOLUTE PATH: reports/2026-10-02_wumpus_diagnostics/g0_readout.py
"""
PROTOCOL.md C-p, G0 readout (CPU): does a belief-supervised latent remove the one-step model error?

Reads, for the three stored oracle + double runs (best.pt; development analysis seed 12345), the outputs of
feature_localise.py, q_localise_diag.py (declined states) and exact_model_play.py (learned and EPR), all on
the same 512 games, and prints per seed:
- the joint glitter ratio at the declined turned latents (the recipe's is 0.22-0.44), from both scripts;
- the MLP probe at those latents: gold calibration and "cell ahead unvisited" accuracy;
- the flip shares (the macro beats the chosen action) under the learned model (M) and the exact one-step model;
- return, deaths and unfinished games of the learned and EPR plays, and EPR minus learned paired per game
  with its game-level standard error.
Then the two registered readings: HEAD REPAIRED (joint glitter ratio >= 0.8 on >= 2 of 3 seeds) and
MODEL STILL BINDS (EPR minus learned >= +50 per game on >= 2 of 3 seeds), and the registered consequence.
Writes runs/wumpus/g0/summary.json.
"""

import json
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)
RUN = "runs/wumpus/u64_store_eval_eps_decoder_k5_privileged_double_seed{}"


def outcomes(returns: np.ndarray) -> tuple[float, float, float]:
    """(mean return, % died, % unfinished) from per-game discounted returns (a win pays > 0, a death < -500)."""
    won, died = returns > 0, returns < -500
    return float(returns.mean()), 100 * float(died.mean()), 100 * float((~won & ~died).mean())


def main() -> None:
    out, glitter, gain = {}, [], []
    for s in SEEDS:
        a = Path(RUN.format(s)) / "analysis"
        fl = json.loads((a / "feature_localise.json").read_text())
        q = dict(np.load(a / "q_localise.npz", allow_pickle=True))
        learned = np.load(a / "exact_play_learned.npz")["returns"].astype(np.float64)
        epr = np.load(a / "exact_play_EPR.npz")["returns"].astype(np.float64)
        if learned.shape != epr.shape:
            raise ValueError(f"seed {s}: the learned and EPR plays cover different games.")
        diff = epr - learned
        row = {"declined_states": int(len(q["q_mac_M"])),
               "glitter_ratio_fl": fl["joint_glitter_ratio_c"],
               "glitter_ratio_q": float(q["p_glitter_fwd_model"].mean() / q["p_glitter_fwd_exact"].mean()),
               "gold_calibration_c": fl["mlp"]["c"]["gold"]["calibration"],
               "unvisited_acc_c": fl["mlp"]["c"]["unvisited"]["acc"],
               "share": {arm: float((q[f"q_mac_{arm}"] > q[f"q_cho_{arm}"]).mean()) for arm in ("M", "EP", "EPR")},
               "learned": outcomes(learned), "EPR": outcomes(epr), "games": int(len(diff)),
               "epr_minus_learned": float(diff.mean()), "epr_minus_learned_se": float(diff.std(ddof=1) / np.sqrt(len(diff)))}
        out[s] = row
        glitter.append(row["glitter_ratio_fl"]); gain.append(row["epr_minus_learned"])
        print(f"seed {s}: declined states {row['declined_states']}; joint glitter ratio {row['glitter_ratio_fl']:.3f} "
              f"(q_localise {row['glitter_ratio_q']:.3f}); probe gold calibration {row['gold_calibration_c']:.3f}, "
              f"unvisited accuracy {row['unvisited_acc_c']:.3f}")
        print(f"        flip shares M {row['share']['M']:.3f}, EP {row['share']['EP']:.3f}, EPR {row['share']['EPR']:.3f}")
        for arm in ("learned", "EPR"):
            mean, died, unfinished = row[arm]
            print(f"        {arm:>7}: return {mean:7.1f}, died {died:4.1f}%, unfinished {unfinished:4.1f}% ({row['games']} games)")
        print(f"        EPR - learned: {row['epr_minus_learned']:+.1f} +- {row['epr_minus_learned_se']:.1f} (game-level s.e.)")
    repaired = sum(g >= 0.8 for g in glitter) >= 2
    binds = sum(g >= 50.0 for g in gain) >= 2
    print(f"HEAD REPAIRED (joint glitter ratio >= 0.8 on >= 2 of 3): {'YES' if repaired else 'NO'} "
          f"({', '.join(f'{g:.2f}' for g in glitter)})")
    print(f"MODEL STILL BINDS (EPR - learned >= +50 on >= 2 of 3): {'YES' if binds else 'NO'} "
          f"({', '.join(f'{g:+.0f}' for g in gain)})")
    consequence = ("G1 arm A is run (belief supervision with the recipe: retrain + double)" if repaired else
                   "only G1 arm B is built (state-factored observation model in the backup)" if binds else
                   "the screen is closed: on a belief-supervised latent the remaining idling is not a one-step-model error")
    print(f"Registered consequence: {consequence}.")
    out["readings"] = {"head_repaired": repaired, "model_still_binds": binds, "consequence": consequence}
    Path("runs/wumpus/g0").mkdir(parents=True, exist_ok=True)
    Path("runs/wumpus/g0/summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
