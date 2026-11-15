# ABSOLUTE PATH: reports/2026-11-14_submission_audit/w4_accepted_states.py
"""
Submission audit W4 summary: the localisation's quantities at the declined, accepted and all distance-2 near-turn
states of development seeds 0-4 (q_localise_diag.py section 5, feature_localise.py section 7). Descriptive; no rule.

Per seed and state set:
- states, and how many had the turn taken;
- joint glitter ratio at the turned latent: feature_localise's joint_glitter_ratio_c, and q_localise's
  mean learned / mean exact P(glitter | turned child, FORWARD) (the two must agree on the same states);
- the MLP probe at the turned latents (c): gold calibration (mean predicted / mean exact P(gold ahead)) and
  "cell ahead unvisited" accuracy;
- arm shares: Q_mac > Q2(comparison action) under M, EP and EPR; at turn-taken states, the reverse flip share
  (EP or EPR prefers the other action where M prefers the turn).
Reads <run>/analysis/{q_localise,feature_localise}{,_accepted,_all}.{npz,json}; prints a table and writes
runs/wumpus/w4/summary.json.
"""

import json
from pathlib import Path

import numpy as np

SEEDS, SETS = range(5), ("declined", "accepted", "all")


def main() -> None:
    out = {}
    print(f"{'seed':>4} {'set':>9} {'states':>6} {'taken':>5} {'glit(fl)':>8} {'glit(q)':>8} {'gold cal':>8} "
          f"{'unvis acc':>9} {'M':>6} {'EP':>6} {'EPR':>6} {'rev EP':>7} {'rev EPR':>8}")
    for s in SEEDS:
        a = Path(f"runs/sweeps/wumpus_retrain/seed{s}/analysis")
        for st in SETS:
            suffix = "" if st == "declined" else f"_{st}"
            q = dict(np.load(a / f"q_localise{suffix}.npz", allow_pickle=True))
            fl = json.loads((a / f"feature_localise{suffix}.json").read_text())
            taken = q["chose_turn"].astype(bool) if "chose_turn" in q else np.zeros(len(q["q_mac_M"]), bool)
            share = {arm: float((q[f"q_mac_{arm}"] > q[f"q_cho_{arm}"]).mean()) for arm in ("M", "EP", "EPR")}
            rev = {}
            for arm in ("EP", "EPR"):
                m_turn = taken & (q["q_mac_M"] > q["q_cho_M"])
                rev[arm] = (float(((q[f"q_mac_{arm}"] <= q[f"q_cho_{arm}"]) & m_turn).sum() / m_turn.sum())
                            if m_turn.any() else float("nan"))
            row = {"states": int(len(q["q_mac_M"])), "turn_taken": int(taken.sum()),
                   "glitter_ratio_fl": fl["joint_glitter_ratio_c"],
                   "glitter_ratio_q": float(q["p_glitter_fwd_model"].mean() / q["p_glitter_fwd_exact"].mean()),
                   "gold_calibration_c": fl["mlp"]["c"]["gold"]["calibration"],
                   "unvisited_acc_c": fl["mlp"]["c"]["unvisited"]["acc"],
                   "share": share, "reverse_flip": rev}
            out.setdefault(s, {})[st] = row
            print(f"{s:>4} {st:>9} {row['states']:>6} {row['turn_taken']:>5} {row['glitter_ratio_fl']:>8.3f} "
                  f"{row['glitter_ratio_q']:>8.3f} {row['gold_calibration_c']:>8.3f} {row['unvisited_acc_c']:>9.3f} "
                  f"{share['M']:>6.3f} {share['EP']:>6.3f} {share['EPR']:>6.3f} {rev['EP']:>7.3f} {rev['EPR']:>8.3f}")
    for st in SETS:
        g = [out[s][st]["glitter_ratio_fl"] for s in SEEDS]
        print(f"{st:>9}: joint glitter ratio per seed {', '.join(f'{x:.2f}' for x in g)} (mean {np.mean(g):.2f})")
    Path("runs/wumpus/w4").mkdir(parents=True, exist_ok=True)
    Path("runs/wumpus/w4/summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
