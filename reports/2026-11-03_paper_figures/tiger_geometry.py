# ABSOLUTE PATH: reports/2026-11-03_paper_figures/tiger_geometry.py
# ======================================================================================================================
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS
# ----------------------------------------------------------------------------------------------------------------------
# Paper figure: geometry of the learned Tiger latent over the 10 training seeds of runs/sweeps/polyak (the default
# configuration, the agent of Table tab:tiger), from each seed's analysis/report.json "geometry" block
# (interpretability/analysis.py section 2: PCA of the acting latents on the agent's own histories, |Spearman| of each
# top component with the exact posterior log-odds (the side) and with |log-odds| (the confidence), minimality ratio).
# - Left: explained-variance ratio of the first five components, one dot per seed plus the mean as a bar. Two
#   components carry almost all of the variance; a Bayes filter for Tiger needs one number (the log-odds), so the
#   second dimension is history the filter keeps beyond the belief (the minimality ratio, quoted in the caption).
# - Right: |Spearman| of PC1-PC3 with side and with confidence, one dot per seed, grouped by component. The pattern
#   (PC2 = side on every seed, PC1 = confidence) is the figure's message; the seed-to-seed agreement is shown by the
#   dots rather than an interval.
# - Two categories (side, confidence): dataviz slots 1 and 2 (blue, orange), validated in the ladder figure's set.
# - Read straight from the reports: no randomness, CPU only, vector PDF with Type 42 fonts.
# Run: uv run reports/2026-11-03_paper_figures/tiger_geometry.py  ->  reports/2026-11-03_paper_figures/tiger_geometry.pdf
# ======================================================================================================================
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SEEDS = range(10)
GEOMETRY = [json.loads((ROOT / f"runs/sweeps/polyak/seed{s}/analysis/report.json").read_text())["geometry"]
            for s in SEEDS]

SIDE, CONF = "#2a78d6", "#eb6834"
INK, MUTED, GRID, BAR = "#1f1f1e", "#6b6a64", "#e4e3de", "#c9c8c2"


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})
    fig, (ax_s, ax_r) = plt.subplots(1, 2, figsize=(5.5, 1.9))

    k_max = 5
    for k in range(k_max):
        values = [g["explained_variance_ratio"][k] * 100 for g in GEOMETRY]
        ax_s.bar(k + 1, statistics.mean(values), width=0.6, color=BAR, zorder=1)
        ax_s.plot([k + 1] * len(values), values, "o", ms=3, color=INK, alpha=0.7, mec="none", zorder=2)
    ax_s.set_xticks(range(1, k_max + 1), [f"PC{k}" for k in range(1, k_max + 1)])
    ax_s.set_ylabel("Variance explained (%)")
    ax_s.set_ylim(0, 100)

    offset = 0.17
    for k in range(3):
        for dx, key, colour in ((-offset, "log_odds_spearman", SIDE), (offset, "confidence_spearman", CONF)):
            values = [g[key][k] for g in GEOMETRY]
            ax_r.plot([k + 1 + dx] * len(values), values, "o", ms=3.5, color=colour, alpha=0.8, mec="none")
    ax_r.set_xticks([1, 2, 3], ["PC1", "PC2", "PC3"])
    ax_r.set_ylabel(r"$|$Spearman $\rho|$")
    ax_r.set_ylim(-0.03, 1.0)
    ax_r.plot([], [], "o", color=SIDE, label="side (log-odds)")
    ax_r.plot([], [], "o", color=CONF, label="confidence ($|$log-odds$|$)")
    ax_r.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2,
                columnspacing=1.0)

    for ax in (ax_s, ax_r):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.5)
    out = Path(__file__).with_suffix(".pdf")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    minimality = [g["minimality_ratio"] for g in GEOMETRY]
    top2 = [sum(g["explained_variance_ratio"][:2]) for g in GEOMETRY]
    print(out, f"minimality {min(minimality):.2f}-{max(minimality):.2f} mean {statistics.mean(minimality):.2f}; "
               f"PC1+PC2 {min(top2):.2f}-{max(top2):.2f}")


if __name__ == "__main__":
    main()
