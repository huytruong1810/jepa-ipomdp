# ABSOLUTE PATH: reports/2026-11-03_paper_figures/ladder.py
# ======================================================================================================================
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS
# ----------------------------------------------------------------------------------------------------------------------
# Paper figure: the Wumpus World held-out ladder (Table tab:wumpus as a picture), two panels sharing one agent axis.
# - Left: mean discounted return per game with the hierarchical-bootstrap 95% interval over seeds and games
#   (experiments/aggregate.py section 6), read from runs/sweeps/wumpus_retrain_heldout/aggregate_analysis_heldout.json,
#   the confirmatory held-out readout (PROTOCOL.md C-a/C-b). Random and the heuristic act on the same games but are not
#   trained, so their interval is over games only; we still draw it (it is narrow) to keep one mark per agent.
# - Right: the outcome of each game (won / died / unfinished within the window) as a 100% stacked bar. The point of the
#   figure is that the learned planner dies as rarely as the exact-belief planner and loses its return to unfinished
#   games; a stacked bar shows that composition directly. Segments carry direct percentage labels (the aqua slot is
#   below 3:1 contrast on white, so colour never carries the reading alone) and are separated by 2 pt white gaps.
# - Two panels instead of one chart with two y-scales (return and percentage are different measures).
# - Colours: the dataviz reference categorical slots 1-3 (blue, orange, aqua), validated with validate_palette.js in
#   light mode (CVD worst adjacent dE 9.2); the agent dots are neutral ink, ours in slot 1 blue.
# - Vector PDF for the paper (fonts embedded as Type 42), CPU only, no randomness: the numbers are read from the json.
# Run: uv run reports/2026-11-03_paper_figures/ladder.py  ->  reports/2026-11-03_paper_figures/ladder.pdf
# ======================================================================================================================
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
H = json.loads((ROOT / "runs/sweeps/wumpus_retrain_heldout/aggregate_analysis_heldout.json").read_text())["hierarchical"]

# (label, json key or None). Rows from the bottom up so the best agent is on top.
AGENTS = [("Random", "random"), ("Recurrent model-free", "recurrent_dqn"), ("Learned planner (ours)", "learned_planner"),
          ("Exact-belief planner", "exact_planner"), ("Exact-belief heuristic", "reference_risk0.4")]
# Outcome rates of random and the heuristic are not in the hierarchical block: they are untrained references that play
# the same held-out games for every training seed, so any seed's report holds them (seed 5's is read).
_REF = json.loads((ROOT / "runs/sweeps/wumpus_retrain_heldout/seed5/analysis_heldout/report.json").read_text())["returns"]
FIXED_RATES = {k: tuple(_REF[k][m] for m in ("win_rate", "death_rate", "unfinished_rate"))
               for k in ("random", "reference_risk0.4")}

WON, DIED, UNFIN = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def rates(key):
    if key in FIXED_RATES:
        return FIXED_RATES[key]
    return tuple(H[f"{m}/{key}"]["mean"] for m in ("win_rate", "death_rate", "unfinished_rate"))


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})
    fig, (ax_r, ax_o) = plt.subplots(1, 2, figsize=(5.5, 1.9), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    ys = range(len(AGENTS))
    for y, (label, key) in zip(ys, AGENTS):
        r = H[f"return/{key}"]
        lo, hi = r["ci95"]
        colour = WON if key == "learned_planner" else INK
        ax_r.plot([lo, hi], [y, y], color=colour, lw=2, solid_capstyle="round")
        ax_r.plot(r["mean"], y, "o", ms=5, color=colour, mec="white", mew=1.2, zorder=3)
        ax_r.text(hi + 25, y, f"{r['mean']:.0f}", va="center", ha="left", color=INK, fontsize=7)
        left = 0.0
        for value, colour_o in zip(rates(key), (WON, DIED, UNFIN)):
            ax_o.barh(y, value * 100, left=left, height=0.62, color=colour_o, edgecolor="white", linewidth=1.0)
            if value >= 0.08:
                ax_o.text(left + value * 50, y, f"{value * 100:.0f}", va="center", ha="center", color="white",
                          fontsize=6.5, fontweight="bold")
            left += value * 100
    ax_r.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax_r.set_yticks(list(ys), [a for a, _ in AGENTS])
    ax_r.set_xlim(-480, 700)
    ax_r.set_xlabel("Discounted return per game")
    ax_o.set_xlim(0, 100)
    ax_o.set_xlabel("Games (%)")
    for ax in (ax_r, ax_o):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (WON, DIED, UNFIN)]
    ax_o.legend(handles, ["Won", "Died", "Unfinished"], ncol=3, frameon=False, fontsize=7, loc="lower center",
                bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=1.0)
    fig.tight_layout(w_pad=0.8)
    out = Path(__file__).with_suffix(".pdf")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(out)


if __name__ == "__main__":
    main()
