# ABSOLUTE PATH: reports/2026-11-03_paper_figures/tiger_intervention.py
# ======================================================================================================================
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS
# ----------------------------------------------------------------------------------------------------------------------
# Paper figure: the write-in intervention on the Tiger latent (PROTOCOL.md C-q.2;
# reports/2026-11-15_interpretability/tiger_intervention.py), ten training seeds of runs/sweeps/polyak.
# - Question the figure answers: when another belief is written into the latent, does the planner take the optimal
#   action OF THE WRITTEN BELIEF? Only rows of the flip set are shown (the exact optimal action differs between the
#   latent's own belief and the written one), so "yes" always means the action had to change.
# - x: the written belief P(tiger left), grouped into Tiger's belief levels (the log-odds are multiples of
#   log(0.85 / 0.15); the two outer groups pool every level beyond two net growls). The optimal action of each group
#   is printed above the plot: a door in the outer groups, LISTEN in the three inner ones.
# - y: share of flip-set rows whose planner action equals the written belief's optimal action. One dot per seed, the
#   line joins the means over seeds.
# - Three interventions: replace (the latent is set to the belief-to-latent map of the written belief), write-in (only
#   the belief component is moved, the rest of the latent is kept), and a random direction of the same length (the
#   control). The control is drawn in neutral grey: it is a reference, not a series to compare hues on. It reaches
#   the LISTEN groups because a perturbed latent falls back to LISTEN, and stays at zero where a door is asked for,
#   which is why the door groups are the discriminating ones (the amendment registered before the ten-seed run).
# - Two hues (dataviz slots 1 and 2, blue and orange; validated: CVD separation 24.7, contrast >= 3:1 on the light
#   surface) plus the grey control; every series is labelled directly at its right end and in the legend.
# - Read from runs/interpretability/tiger_intervention/polyak_seed<s>.npz: no randomness, CPU only, vector PDF with
#   Type 42 fonts.
# Run: uv run reports/2026-11-03_paper_figures/tiger_intervention.py -> reports/2026-11-03_paper_figures/tiger_intervention.pdf
# ======================================================================================================================
import math
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEEDS = range(10)
STEP = math.log(0.85 / 0.15)                                   # one growl of evidence, in log-odds
GROUPS = [(-2, r"$\leq 0.03$"), (-1, "0.15"), (0, "0.5"), (1, "0.85"), (2, r"$\geq 0.97$")]
ARMS = [("replace", "replace", "#2a78d6"), ("write_in", "write-in", "#eb6834"), ("random", "random direction", "#8a8983")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
ACTIONS = {0: "listen", 1: "open left", 2: "open right"}


def shares(seed: int) -> tuple[dict[str, list[float]], list[int], list[int]]:
    """Per arm, the share of flip-set rows per belief group that take the written belief's optimal action."""
    d = np.load(ROOT / f"runs/interpretability/tiger_intervention/polyak_seed{seed}.npz")
    b = np.clip(d["belief_target"], 1e-9, 1 - 1e-9)
    level = np.clip(np.rint(np.log(b / (1 - b)) / STEP), -2, 2).astype(int)
    flip = d["optimal_source"] != d["optimal_target"]
    out, counts, optimal = {arm: [] for arm, _, _ in ARMS}, [], []
    for k, _ in GROUPS:
        rows = flip & (level == k)
        counts.append(int(rows.sum()))
        asked = np.unique(d["optimal_target"][rows])
        if len(asked) != 1:
            raise ValueError(f"seed {seed}, group {k}: the written beliefs ask for {len(asked)} different actions.")
        optimal.append(int(asked[0]))
        for arm, _, _ in ARMS:
            out[arm].append(float((d[f"action_{arm}"][rows] == d["optimal_target"][rows]).mean()))
    return out, counts, optimal


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    per_seed = [shares(s) for s in SEEDS]
    counts, optimal = per_seed[0][1], per_seed[0][2]
    if any(p[1] != counts or p[2] != optimal for p in per_seed):
        raise ValueError("The seeds do not share one set of intervened rows.")
    fig, ax = plt.subplots(figsize=(5.5, 2.1))
    xs = np.arange(len(GROUPS))
    for (arm, label, colour), dx in zip(ARMS, (-0.09, 0.0, 0.09)):
        values = 100 * np.array([p[0][arm] for p in per_seed])                 # (seeds, groups)
        ax.plot(np.repeat(xs + dx, len(SEEDS)), values.T.reshape(-1), "o", ms=2.6, color=colour, alpha=0.45, mec="none",
                zorder=2)
        ax.plot(xs + dx, values.mean(0), "-", lw=1.5, color=colour, zorder=3, label=label)
        ax.plot(xs + dx, values.mean(0), "o", ms=4.2, color=colour, mec="white", mew=0.8, zorder=4)
        print(f"{label:17s}", " ".join(f"{v:6.1f}" for v in values.mean(0)), " (mean % per group)")
    ax.set_xticks(xs, [label for _, label in GROUPS])
    ax.set_xlim(-0.45, len(GROUPS) - 0.55)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Written belief, P(tiger behind the left door)")
    ax.set_ylabel("Planner takes the written\nbelief's optimal action (%)")
    for x, action, n in zip(xs, optimal, counts):
        ax.text(x, 111, f"{ACTIONS[action]}\n(n = {n})", ha="center", va="bottom", fontsize=6.5, color=MUTED, linespacing=1.1)
    for x in (0.5, 3.5):                                                       # the optimal action changes here
        ax.axvline(x, color=MUTED, lw=0.6, ls=(0, (3, 3)), zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.4)
    fig.tight_layout()
    out = Path(__file__).with_suffix(".pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(out, "flip-set rows per group", counts, "optimal action per group", [ACTIONS[a] for a in optimal])


if __name__ == "__main__":
    main()
