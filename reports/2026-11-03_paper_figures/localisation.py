# ABSOLUTE PATH: reports/2026-11-03_paper_figures/localisation.py
# ======================================================================================================================
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS
# ----------------------------------------------------------------------------------------------------------------------
# Paper figure: exact-component localisation on Wumpus World (paragraph "Localisation"), two panels.
# - Left (play): per training seed, the return per game of the planner with its own learned one-step model and with
#   the exact one-step model (exact P(o' | b, a) and R(b, a) at every node; learned filter and value kept), drawn as a
#   dumbbell from learned to exact. Development seeds 0-4 (exact_model_play.py section 4) and held-out seeds 5-9
#   (exact_model_play.py section 6, the double-value recipe networks) are separated and labelled, because the paper
#   labels every number as development or held-out. Every seed moves right: that is the figure's message.
# - Right (decisions): at the declined decisions of development seeds 0-4, the share at which the backup prefers
#   "turn, then step in" to the chosen action, with the learned model (M), the exact observation probabilities
#   substituted after the turn (EP), and both exact components (EPR) (q_localise_diag.py section 4). The three
#   substitutions are ordered (more of the model made exact), so they take an ordinal one-hue ramp, light to dark.
# - Values are copied from the cited script headers (the readouts of record; the raw logs are not tracked), with the
#   row each comes from. No randomness; CPU only; vector PDF with Type 42 fonts.
# - Colours: learned = neutral grey, exact = dataviz slot-1 blue; the ramp is the blue ramp steps 250 / 450 / 650.
# Run: uv run reports/2026-11-03_paper_figures/localisation.py  ->  reports/2026-11-03_paper_figures/localisation.pdf
# ======================================================================================================================
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt

# exact_model_play.py section 4 (development, learned vs EPR) and section 6 (held-out, recipe networks, alpha 0 vs 1).
PLAY = [  # (seed, learned, exact one-step model)
    (0, 236.7, 460.4), (1, 156.8, 321.7), (2, 233.1, 400.3), (3, 174.7, 433.7), (4, 199.7, 423.8),
    (5, 246.5, 430.6), (6, 229.8, 433.0), (7, 204.6, 340.8), (8, 132.7, 271.0), (9, 164.6, 340.8),
]
# q_localise_diag.py section 4: shares of Q(turn, then step) > Q(chosen) at the declined decisions, M / EP / EPR.
SHARES = [(0, 10.0, 43.0, 69.4), (1, 14.4, 41.5, 47.9), (2, 10.9, 46.3, 64.1), (3, 11.2, 62.2, 80.9),
          (4, 7.6, 57.7, 68.3)]

BLUE, GREY = "#2a78d6", "#8d8c86"
RAMP = ("#86b6ef", "#2a78d6", "#104281")
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})
    fig, (ax_p, ax_d) = plt.subplots(1, 2, figsize=(5.5, 2.3), gridspec_kw={"width_ratios": [1.2, 1]})

    # Rows top to bottom: development 0-4, a gap, held-out 5-9.
    ys = {s: -(s + (1 if s >= 5 else 0)) for s, _, _ in PLAY}
    for s, learned, exact in PLAY:
        y = ys[s]
        ax_p.plot([learned, exact], [y, y], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)
        ax_p.plot(learned, y, "o", ms=5, color=GREY, mec="white", mew=1.0, zorder=2)
        ax_p.plot(exact, y, "o", ms=5, color=BLUE, mec="white", mew=1.0, zorder=3)
        ax_p.text(exact + 12, y, f"+{exact - learned:.0f}", va="center", ha="left", color=INK, fontsize=6.5)
    ax_p.set_yticks([ys[s] for s, _, _ in PLAY], [f"seed {s}" for s, _, _ in PLAY])
    for label, y in (("development", ys[0] + 0.75), ("held-out", ys[5] + 0.75)):
        ax_p.text(95, y, label, ha="left", va="center", color=MUTED, fontsize=7, style="italic")
    ax_p.set_xlim(90, 540)
    ax_p.set_ylim(ys[9] - 0.7, ys[0] + 1.3)
    ax_p.set_xlabel("Discounted return per game")
    ax_p.plot([], [], "o", color=GREY, label="learned model")
    ax_p.plot([], [], "o", color=BLUE, label="exact one-step model")
    ax_p.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2,
                columnspacing=1.0)

    width = 0.26
    names = ("learned model", "exact $P$ after the turn", "exact $P$ and $R$")
    for k, (name, colour) in enumerate(zip(names, RAMP)):
        xs = [s + (k - 1) * width for s, *_ in SHARES]
        ax_d.bar(xs, [row[k + 1] for row in SHARES], width=width - 0.03, color=colour, label=name)
    ax_d.set_xticks([s for s, *_ in SHARES], [f"{s}" for s, *_ in SHARES])
    ax_d.set_xlabel("Development seed")
    ax_d.set_ylabel("Declined decisions where\nthe backup prefers stepping in (%)")
    ax_d.set_ylim(0, 100)
    ax_d.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.0)

    for ax in (ax_p, ax_d):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2)
        ax.set_axisbelow(True)
    ax_p.grid(axis="x", color=GRID, lw=0.5)
    ax_d.grid(axis="y", color=GRID, lw=0.5)
    fig.tight_layout(w_pad=1.2)
    out = Path(__file__).with_suffix(".pdf")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(out)


if __name__ == "__main__":
    main()
