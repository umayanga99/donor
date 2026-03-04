"""
Comprehensive matrix plots for claude_pd_baseline_20260223_001720.json
Produces four figure groups:
  1. Per-Round metrics (4 subplots)
  2. Per-Generation metrics (5 subplots)
  3. Overall summary metrics (4 subplots)
  4. Per-Agent metrics (4 subplots)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from scipy import stats

# ── Load data ────────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(
    os.path.dirname(__file__), "..",
    "latest_results", "claude_pd_baseline_20260223_001720.json"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "new_matrices")

with open(DATA_FILE) as f:
    raw = json.load(f)

hp      = raw["hyperparameters"]
records = raw["agents_data"]

NUM_GENS   = hp["num_generations"]          # 12
NUM_ROUNDS = hp["num_rounds_per_generation"] # 5
NUM_AGENTS = hp["num_agents"]               # 10

STRATEGIES = sorted(set(r["strategy"] for r in records))
STRATEGY_COLORS = {
    "Always Cooperate": "#2ecc71",
    "Always Defect":    "#e74c3c",
    "Tit-for-Tat":      "#3498db",
    "Generous TFT":     "#9b59b6",
    "Pavlov":           "#f39c12",
    "Random":           "#95a5a6",
}
CMAP = plt.cm.tab10

# ── Helper: cooperation flag from donated field ───────────────────────────────
# donated == 1.0  → cooperated this round
def cooperated(r):
    return float(r["donated"]) > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PER-ROUND aggregates  (keyed by absolute_round = (gen-1)*NUM_ROUNDS + round)
# ─────────────────────────────────────────────────────────────────────────────
# Group records by (gen, round)
by_gen_round = defaultdict(list)
for r in records:
    by_gen_round[(r["current_generation"], r["round_number"])].append(r)

abs_rounds = []
round_coop_rate   = []   # mean cooperation rate across all agents
round_mean_regret = []   # mean regret_level
round_rep_var     = []   # variance of reputation values
round_avg_rep_diff_pairs = []  # average |rep_i - rep_j| within paired agents

for g in range(1, NUM_GENS + 1):
    for rnd in range(1, NUM_ROUNDS + 1):
        grp = by_gen_round[(g, rnd)]
        abs_rounds.append((g - 1) * NUM_ROUNDS + rnd)

        # cooperation rate: use donated field (1.0 = cooperate)
        coop = np.mean([float(r["donated"]) for r in grp])
        round_coop_rate.append(coop)

        # mean regret
        round_mean_regret.append(np.mean([r["regret_level"] for r in grp]))

        # reputation variance
        reps = [r["reputation"] for r in grp]
        round_rep_var.append(np.var(reps))

        # avg rep diff within pairs
        # paired_with stores the partner agent_name
        paired = {}
        for rec in grp:
            paired[rec["agent_name"]] = rec
        diffs = []
        seen = set()
        for rec in grp:
            partner_name = rec["paired_with"]
            key = tuple(sorted([rec["agent_name"], partner_name]))
            if key not in seen and partner_name in paired:
                seen.add(key)
                diffs.append(abs(rec["reputation"] - paired[partner_name]["reputation"]))
        round_avg_rep_diff_pairs.append(np.mean(diffs) if diffs else 0.0)

abs_rounds          = np.array(abs_rounds)
round_coop_rate     = np.array(round_coop_rate)
round_mean_regret   = np.array(round_mean_regret)
round_rep_var       = np.array(round_rep_var)
round_avg_rep_diff  = np.array(round_avg_rep_diff_pairs)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PER-GENERATION aggregates
# ─────────────────────────────────────────────────────────────────────────────
by_gen = defaultdict(list)
for r in records:
    by_gen[r["current_generation"]].append(r)

gens = list(range(1, NUM_GENS + 1))

gen_strategy_dist   = []   # dict strategy→count (last round of gen for consistency)
gen_avg_payoff      = []   # dict strategy→mean resources at END of generation
gen_mean_global_rep = []   # mean reputation across all agents in gen
gen_mean_forgiveness= []   # mean forgiveness_given
gen_coop_stability  = []   # variance of per-agent cooperation_rate within generation

for g in gens:
    grp = by_gen[g]

    # strategy distribution (count unique agents → use last round per agent)
    last_round_recs = {}
    for rec in grp:
        a = rec["agent_name"]
        if a not in last_round_recs or rec["round_number"] > last_round_recs[a]["round_number"]:
            last_round_recs[a] = rec
    strat_counts = defaultdict(int)
    strat_resources = defaultdict(list)
    for rec in last_round_recs.values():
        strat_counts[rec["strategy"]] += 1
        strat_resources[rec["strategy"]].append(rec["resources"])
    gen_strategy_dist.append(dict(strat_counts))
    gen_avg_payoff.append({s: np.mean(v) for s, v in strat_resources.items()})

    # mean global reputation (all records in this gen)
    gen_mean_global_rep.append(np.mean([rec["reputation"] for rec in grp]))

    # mean forgiveness
    gen_mean_forgiveness.append(np.mean([rec["forgiveness_given"] for rec in grp]))

    # cooperation stability: variance of cooperation_rate across agents × rounds
    coop_rates = [rec["cooperation_rate"] for rec in grp]
    gen_coop_stability.append(np.var(coop_rates))

# ─────────────────────────────────────────────────────────────────────────────
# 3.  OVERALL metrics  (single values)
# ─────────────────────────────────────────────────────────────────────────────
# Use per-round cooperation rate array computed above
final_coop_rate   = round_coop_rate[-1]
overall_avg_coop  = float(np.mean(round_coop_rate))

# Trend slope via linear regression on absolute rounds
slope, intercept, r_val, p_val, std_err = stats.linregress(abs_rounds, round_coop_rate)
coop_trend_slope  = slope

# Reputation–cooperation correlation (per-round means)
per_round_mean_rep = []
for g in range(1, NUM_GENS + 1):
    for rnd in range(1, NUM_ROUNDS + 1):
        grp = by_gen_round[(g, rnd)]
        per_round_mean_rep.append(np.mean([rec["reputation"] for rec in grp]))
per_round_mean_rep = np.array(per_round_mean_rep)
# Correlation is undefined when reputation is constant (e.g. baseline with no reputation mechanism)
if np.std(per_round_mean_rep) == 0 or np.std(round_coop_rate) == 0:
    rep_coop_corr, rep_coop_p = np.nan, np.nan
else:
    rep_coop_corr, rep_coop_p = stats.pearsonr(per_round_mean_rep, round_coop_rate)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  PER-AGENT aggregates  (collapse all records for each LOGICAL agent id 1..10)
# ─────────────────────────────────────────────────────────────────────────────
# agent_name format: "{gen}_{id}"  – same logical agent id appears in each gen
logical_agents = list(range(1, NUM_AGENTS + 1))

agent_coop_ratio   = []
agent_mean_regret  = []
agent_mean_forg    = []
agent_lifetime_pay = []

for aid in logical_agents:
    recs_a = [rec for rec in records if rec["agent_name"].endswith(f"_{aid}")]
    agent_coop_ratio.append(np.mean([float(rec["donated"]) for rec in recs_a]))
    agent_mean_regret.append(np.mean([rec["regret_level"] for rec in recs_a]))
    agent_mean_forg.append(np.mean([rec["forgiveness_given"] for rec in recs_a]))
    # lifetime payoff = final resources in last gen's last round
    last_rec = max(recs_a, key=lambda r: (r["current_generation"], r["round_number"]))
    agent_lifetime_pay.append(last_rec["resources"])

agent_coop_ratio   = np.array(agent_coop_ratio)
agent_mean_regret  = np.array(agent_mean_regret)
agent_mean_forg    = np.array(agent_mean_forg)
agent_lifetime_pay = np.array(agent_lifetime_pay)

# ─────────────────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 – Per-Round Metrics  (2×2 grid)
# ════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9))
fig1.suptitle("Per-Round Metrics  ·  Baseline PD Simulation", fontsize=15, fontweight="bold", y=1.01)
fig1.patch.set_facecolor("#f8f9fa")

# Add generation dividers helper
def add_gen_dividers(ax, num_gens, num_rounds, alpha=0.15):
    for g in range(1, num_gens):
        ax.axvline(g * num_rounds + 0.5, color="grey", lw=0.8, alpha=alpha, ls="--")

# -- 1a: Cooperation Rate --
ax = axes1[0, 0]
ax.set_facecolor("#fdfdfd")
ax.plot(abs_rounds, round_coop_rate, color="#2980b9", lw=1.8, marker="o", ms=4, label="Coop rate")
trend_y = slope * abs_rounds + intercept
ax.plot(abs_rounds, trend_y, color="#e74c3c", lw=1.5, ls="--", label=f"Trend (slope={slope:.4f})")
add_gen_dividers(ax, NUM_GENS, NUM_ROUNDS)
ax.set_title("Cooperation Rate per Round", fontweight="bold")
ax.set_xlabel("Absolute Round"); ax.set_ylabel("Cooperation Rate")
ax.set_ylim(-0.05, 1.1); ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# -- 1b: Mean Regret --
ax = axes1[0, 1]
ax.set_facecolor("#fdfdfd")
ax.bar(abs_rounds, round_mean_regret, color="#9b59b6", alpha=0.75, width=0.7)
add_gen_dividers(ax, NUM_GENS, NUM_ROUNDS)
ax.set_title("Mean Regret per Round", fontweight="bold")
ax.set_xlabel("Absolute Round"); ax.set_ylabel("Mean Regret Level")
ax.grid(True, alpha=0.3, axis="y")

# -- 1c: Reputation Variance --
ax = axes1[1, 0]
ax.set_facecolor("#fdfdfd")
ax.fill_between(abs_rounds, round_rep_var, alpha=0.4, color="#e67e22")
ax.plot(abs_rounds, round_rep_var, color="#e67e22", lw=1.8, marker="s", ms=4)
add_gen_dividers(ax, NUM_GENS, NUM_ROUNDS)
ax.set_title("Reputation Variance per Round", fontweight="bold")
ax.set_xlabel("Absolute Round"); ax.set_ylabel("Variance of Reputation")
ax.grid(True, alpha=0.3)

# -- 1d: Avg Rep Diff Within Pairs --
ax = axes1[1, 1]
ax.set_facecolor("#fdfdfd")
ax.plot(abs_rounds, round_avg_rep_diff, color="#16a085", lw=1.8, marker="^", ms=4)
add_gen_dividers(ax, NUM_GENS, NUM_ROUNDS)
ax.set_title("Avg Reputation Difference Within Pairs", fontweight="bold")
ax.set_xlabel("Absolute Round"); ax.set_ylabel("|rep_i − rep_j|")
ax.grid(True, alpha=0.3)

# Generation label strip on last row
for col in [0, 1]:
    ax = axes1[1, col]
    for g in range(1, NUM_GENS + 1):
        mid = (g - 1) * NUM_ROUNDS + (NUM_ROUNDS + 1) / 2
        ax.text(mid, ax.get_ylim()[0] - 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"G{g}", ha="center", va="top", fontsize=6.5, color="grey")

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "plot_1_per_round_metrics.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig1)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 – Per-Generation Metrics  (3 rows: stacked bar | line×3)
# ─────────────────────────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(16, 14))
fig2.patch.set_facecolor("#f8f9fa")
fig2.suptitle("Per-Generation Metrics  ·  Baseline PD Simulation", fontsize=15, fontweight="bold")
gs2 = gridspec.GridSpec(3, 2, figure=fig2, hspace=0.45, wspace=0.35)

# -- 2a: Strategy Distribution (stacked bar) --
ax2a = fig2.add_subplot(gs2[0, :])
ax2a.set_facecolor("#fdfdfd")
bottom = np.zeros(NUM_GENS)
for s in STRATEGIES:
    counts = [gen_strategy_dist[g].get(s, 0) for g in range(NUM_GENS)]
    color = STRATEGY_COLORS.get(s, "#bdc3c7")
    ax2a.bar(gens, counts, bottom=bottom, label=s, color=color, alpha=0.85)
    bottom += np.array(counts)
ax2a.set_title("Strategy Distribution per Generation", fontweight="bold")
ax2a.set_xlabel("Generation"); ax2a.set_ylabel("Agent Count")
ax2a.set_xticks(gens)
ax2a.legend(loc="upper right", fontsize=8, ncol=3)
ax2a.grid(True, alpha=0.3, axis="y")

# -- 2b: Average Payoff per Strategy --
ax2b = fig2.add_subplot(gs2[1, :])
ax2b.set_facecolor("#fdfdfd")
for s in STRATEGIES:
    payoffs = [gen_avg_payoff[g - 1].get(s, np.nan) for g in gens]
    color = STRATEGY_COLORS.get(s, "#bdc3c7")
    ax2b.plot(gens, payoffs, label=s, color=color, lw=1.8, marker="o", ms=4)
ax2b.set_title("Average Payoff (Resources) per Strategy per Generation", fontweight="bold")
ax2b.set_xlabel("Generation"); ax2b.set_ylabel("Mean Resources")
ax2b.set_xticks(gens)
ax2b.legend(loc="upper right", fontsize=8, ncol=3)
ax2b.grid(True, alpha=0.3)

# -- 2c: Mean Global Reputation --
ax2c = fig2.add_subplot(gs2[2, 0])
ax2c.set_facecolor("#fdfdfd")
ax2c.plot(gens, gen_mean_global_rep, color="#2980b9", lw=2, marker="o", ms=5)
ax2c.fill_between(gens, gen_mean_global_rep, alpha=0.2, color="#2980b9")
ax2c.set_title("Mean Global Reputation per Generation", fontweight="bold")
ax2c.set_xlabel("Generation"); ax2c.set_ylabel("Mean Reputation")
ax2c.set_xticks(gens); ax2c.grid(True, alpha=0.3)

# -- 2d: Mean Forgiveness Level --
ax2d = fig2.add_subplot(gs2[2, 1])
ax2d.set_facecolor("#fdfdfd")
ax2d.bar(gens, gen_mean_forgiveness, color="#1abc9c", alpha=0.8)
ax2d.set_title("Mean Forgiveness Level per Generation", fontweight="bold")
ax2d.set_xlabel("Generation"); ax2d.set_ylabel("Mean Forgiveness Given")
ax2d.set_xticks(gens); ax2d.grid(True, alpha=0.3, axis="y")

# -- 2e: Cooperation Stability (variance within generation) --
# Overlay on a separate figure panel
fig2e, ax2e = plt.subplots(1, 1, figsize=(8, 3.5))
fig2e.patch.set_facecolor("#f8f9fa")
ax2e.set_facecolor("#fdfdfd")
ax2e.bar(gens, gen_coop_stability, color="#e74c3c", alpha=0.75)
ax2e.set_title("Cooperation Stability (Variance of Coop Rate Within Generation)", fontweight="bold")
ax2e.set_xlabel("Generation"); ax2e.set_ylabel("Variance of Cooperation Rate")
ax2e.set_xticks(gens); ax2e.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
out2e = os.path.join(OUT_DIR, "plot_2e_coop_stability.png")
fig2e.savefig(out2e, dpi=150, bbox_inches="tight")
print(f"Saved: {out2e}")
plt.close(fig2e)

out2 = os.path.join(OUT_DIR, "plot_2_per_generation_metrics.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 – Overall Summary Metrics  (2×2 dashboard)
# ─────────────────────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(13, 9))
fig3.patch.set_facecolor("#f8f9fa")
fig3.suptitle("Overall Summary Metrics  ·  Baseline PD Simulation", fontsize=15, fontweight="bold")

summary_labels  = ["Final Coop Rate", "Overall Avg Coop", "Coop Trend Slope", "Rep–Coop Corr"]
summary_values  = [final_coop_rate, overall_avg_coop, coop_trend_slope, rep_coop_corr]
summary_colors  = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad"]
summary_fmts    = [".3f", ".3f", ".5f", ".4f"]

for idx, (ax, lbl, val, col, fmt) in enumerate(
        zip(axes3.flat, summary_labels, summary_values, summary_colors, summary_fmts)):
    ax.set_facecolor("#fdfdfd")

    if idx == 0:
        # -- Final coop rate: gauge-style bar --
        ax.barh(["Final Round"], [val], color=col, alpha=0.8, height=0.5)
        ax.set_xlim(0, 1)
        ax.axvline(val, color="red", lw=1.5, ls="--")
        ax.text(val + 0.02, 0, f"{val:{fmt}}", va="center", fontsize=13, color=col, fontweight="bold")
        ax.set_title(lbl, fontweight="bold")
        ax.set_xlabel("Cooperation Rate")

    elif idx == 1:
        # -- Overall avg: full round-by-round with mean highlighted --
        ax.plot(abs_rounds, round_coop_rate, color="#95a5a6", lw=1, alpha=0.7)
        ax.axhline(val, color=col, lw=2, ls="--", label=f"Mean = {val:{fmt}}")
        ax.fill_between(abs_rounds, round_coop_rate, val, where=(round_coop_rate >= val),
                        alpha=0.2, color="#27ae60", interpolate=True)
        ax.fill_between(abs_rounds, round_coop_rate, val, where=(round_coop_rate < val),
                        alpha=0.2, color="#e74c3c", interpolate=True)
        ax.set_title(lbl, fontweight="bold")
        ax.set_xlabel("Absolute Round"); ax.set_ylabel("Cooperation Rate")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    elif idx == 2:
        # -- Trend slope: scatter + regression line --
        ax.scatter(abs_rounds, round_coop_rate, color="#95a5a6", s=25, alpha=0.7, label="Data")
        ax.plot(abs_rounds, trend_y, color=col, lw=2.2, label=f"Slope = {val:{fmt}}")
        ax.set_title(lbl, fontweight="bold")
        ax.set_xlabel("Absolute Round"); ax.set_ylabel("Cooperation Rate")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    elif idx == 3:
        # -- Rep–coop correlation: scatter mean rep vs coop rate per round --
        sc = ax.scatter(per_round_mean_rep, round_coop_rate, c=abs_rounds,
                        cmap="viridis", s=40, alpha=0.8, zorder=3)
        cb = fig3.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("Abs. Round", fontsize=8)
        # regression line (only when reputation varies)
        if np.std(per_round_mean_rep) > 0:
            m2, b2, _, _, _ = stats.linregress(per_round_mean_rep, round_coop_rate)
            x_range = np.linspace(per_round_mean_rep.min(), per_round_mean_rep.max(), 100)
            corr_lbl = f"r = {val:{fmt}}, p={rep_coop_p:.3f}" if not np.isnan(val) else "r = N/A (const. rep)"
            ax.plot(x_range, m2 * x_range + b2, color=col, lw=2, label=corr_lbl)
            ax.legend(fontsize=8)
        else:
            corr_lbl = "r = N/A (reputation constant)"
            ax.text(0.5, 0.5, corr_lbl, transform=ax.transAxes, ha="center",
                    va="center", fontsize=9, color=col)
        ax.set_title(lbl, fontweight="bold")
        ax.set_xlabel("Mean Reputation per Round")
        ax.set_ylabel("Cooperation Rate")
        ax.grid(True, alpha=0.3)

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "plot_3_overall_metrics.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close(fig3)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 – Per-Agent Metrics  (2×2 grid)
# ─────────────────────────────────────────────────────────────────────────────
fig4, axes4 = plt.subplots(2, 2, figsize=(13, 9))
fig4.patch.set_facecolor("#f8f9fa")
fig4.suptitle("Per-Agent Metrics  ·  Baseline PD Simulation", fontsize=15, fontweight="bold")

agent_ids  = [f"Agent {i}" for i in logical_agents]
x_pos      = np.arange(NUM_AGENTS)

# Determine per-agent dominant strategy (most common across all records)
agent_dominant_strategy = []
for aid in logical_agents:
    recs_a = [rec["strategy"] for rec in records if rec["agent_name"].endswith(f"_{aid}")]
    from collections import Counter
    dominant = Counter(recs_a).most_common(1)[0][0]
    agent_dominant_strategy.append(dominant)
bar_colors = [STRATEGY_COLORS.get(s, "#bdc3c7") for s in agent_dominant_strategy]

# -- 4a: Cooperation Ratio --
ax = axes4[0, 0]
ax.set_facecolor("#fdfdfd")
bars = ax.bar(x_pos, agent_coop_ratio, color=bar_colors, alpha=0.85, edgecolor="white")
ax.set_title("Cooperation Ratio per Agent", fontweight="bold")
ax.set_xticks(x_pos); ax.set_xticklabels(agent_ids, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Mean Cooperation Ratio")
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis="y")
# Legend for strategies
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=STRATEGY_COLORS.get(s, "#bdc3c7"), label=s) for s in STRATEGIES
                  if s in agent_dominant_strategy]
ax.legend(handles=legend_patches, fontsize=7, loc="upper right", ncol=2)

# -- 4b: Mean Regret --
ax = axes4[0, 1]
ax.set_facecolor("#fdfdfd")
ax.bar(x_pos, agent_mean_regret, color="#9b59b6", alpha=0.8, edgecolor="white")
ax.set_title("Mean Regret per Agent", fontweight="bold")
ax.set_xticks(x_pos); ax.set_xticklabels(agent_ids, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Mean Regret Level")
ax.grid(True, alpha=0.3, axis="y")

# -- 4c: Mean Forgiveness Level --
ax = axes4[1, 0]
ax.set_facecolor("#fdfdfd")
ax.bar(x_pos, agent_mean_forg, color="#1abc9c", alpha=0.8, edgecolor="white")
ax.set_title("Mean Forgiveness Level per Agent", fontweight="bold")
ax.set_xticks(x_pos); ax.set_xticklabels(agent_ids, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Mean Forgiveness Given")
ax.grid(True, alpha=0.3, axis="y")

# -- 4d: Lifetime Payoff --
ax = axes4[1, 1]
ax.set_facecolor("#fdfdfd")
bars = ax.bar(x_pos, agent_lifetime_pay, color=bar_colors, alpha=0.85, edgecolor="white")
ax.set_title("Lifetime Payoff (Resources at End) per Agent", fontweight="bold")
ax.set_xticks(x_pos); ax.set_xticklabels(agent_ids, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Final Resources")
ax.grid(True, alpha=0.3, axis="y")
# Annotate bars
for bar, val in zip(bars, agent_lifetime_pay):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.0f}", ha="center", va="bottom", fontsize=7.5)

plt.tight_layout()
out4 = os.path.join(OUT_DIR, "plot_4_per_agent_metrics.png")
fig4.savefig(out4, dpi=150, bbox_inches="tight")
print(f"Saved: {out4}")
plt.close(fig4)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 – Combined Summary Dashboard (all 4 groups in one figure)
# ─────────────────────────────────────────────────────────────────────────────
fig5 = plt.figure(figsize=(22, 24))
fig5.patch.set_facecolor("#f0f2f5")
fig5.suptitle(
    "Baseline PD Simulation  ·  Full Metrics Dashboard\n"
    "claude_pd_baseline_20260223_001720",
    fontsize=16, fontweight="bold", y=1.005
)

outer_gs = gridspec.GridSpec(4, 1, figure=fig5, hspace=0.55)

# ── Section titles
section_titles = [
    "① Per-Round Metrics",
    "② Per-Generation Metrics",
    "③ Overall Summary",
    "④ Per-Agent Metrics",
]

def section_label(fig, gs_row, title, y_offset=0.005):
    """Add a section header band above each sub-grid."""
    # We rely on suptitle + subplot titles; no action needed here.
    pass

# ----- Section 1: Per-Round (1 row × 4 cols) -----
inner1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_gs[0], wspace=0.38)
ax_pr = [fig5.add_subplot(inner1[0, c]) for c in range(4)]
fig5.text(0.01, ax_pr[0].get_position().y1 + 0.01, section_titles[0],
          fontsize=12, fontweight="bold", color="#2c3e50", va="bottom")

datasets_pr = [
    (round_coop_rate,   "Cooperation Rate",                 "Rate",        "#2980b9"),
    (round_mean_regret, "Mean Regret",                      "Regret Level","#9b59b6"),
    (round_rep_var,     "Reputation Variance",              "Variance",    "#e67e22"),
    (round_avg_rep_diff,"Avg Rep Diff Within Pairs",        "|Δrep|",      "#16a085"),
]
for ax, (arr, title, ylabel, col) in zip(ax_pr, datasets_pr):
    ax.set_facecolor("#fdfdfd")
    ax.plot(abs_rounds, arr, color=col, lw=1.6, marker="o", ms=3)
    add_gen_dividers(ax, NUM_GENS, NUM_ROUNDS)
    ax.set_title(title, fontweight="bold", fontsize=8.5)
    ax.set_xlabel("Round", fontsize=7); ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, alpha=0.25)

# ----- Section 2: Per-Generation (1 row × 4 cols + full-width stacked bar) -----
inner2 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[1], wspace=0.35, hspace=0.45)

ax_strat = fig5.add_subplot(inner2[0, :])   # full-width stacked bar
ax_strat.set_facecolor("#fdfdfd")
bottom2 = np.zeros(NUM_GENS)
for s in STRATEGIES:
    cnts = [gen_strategy_dist[g].get(s, 0) for g in range(NUM_GENS)]
    ax_strat.bar(gens, cnts, bottom=bottom2, label=s,
                 color=STRATEGY_COLORS.get(s, "#bdc3c7"), alpha=0.85)
    bottom2 += np.array(cnts)
ax_strat.set_title("Strategy Distribution per Generation", fontweight="bold", fontsize=9)
ax_strat.set_xlabel("Generation", fontsize=7); ax_strat.set_ylabel("Agents", fontsize=7)
ax_strat.set_xticks(gens); ax_strat.tick_params(labelsize=6.5)
ax_strat.legend(fontsize=6.5, ncol=3, loc="upper right")
ax_strat.grid(True, alpha=0.25, axis="y")

ax_pg = [fig5.add_subplot(inner2[1, c]) for c in range(2)]
pg_datasets = [
    (gens, gen_mean_global_rep, "Mean Global Reputation",  "Mean Rep",   "#2980b9", "o"),
    (gens, gen_mean_forgiveness,"Mean Forgiveness Level",  "Forgiveness","#1abc9c", "s"),
]
for ax, (xd, yd, title, ylabel, col, mk) in zip(ax_pg, pg_datasets):
    ax.set_facecolor("#fdfdfd")
    ax.plot(xd, yd, color=col, lw=1.8, marker=mk, ms=4)
    ax.fill_between(xd, yd, alpha=0.15, color=col)
    ax.set_title(title, fontweight="bold", fontsize=8.5)
    ax.set_xlabel("Generation", fontsize=7); ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xticks(gens); ax.tick_params(labelsize=6.5); ax.grid(True, alpha=0.25)

# ----- Section 3: Overall (1 row × 4 cols) -----
inner3 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_gs[2], wspace=0.4)
ax_ov = [fig5.add_subplot(inner3[0, c]) for c in range(4)]

# 3-0: Final cooperation rate gauge
ax = ax_ov[0]; ax.set_facecolor("#fdfdfd")
ax.barh(["Final\nRound"], [final_coop_rate], color="#2980b9", alpha=0.8, height=0.45)
ax.set_xlim(0, 1); ax.axvline(final_coop_rate, color="red", lw=1.5, ls="--")
ax.text(min(final_coop_rate + 0.04, 0.95), 0, f"{final_coop_rate:.3f}",
        va="center", fontsize=10, color="#2980b9", fontweight="bold")
ax.set_title("Final Coop Rate", fontweight="bold", fontsize=8.5)
ax.set_xlabel("Rate", fontsize=7); ax.tick_params(labelsize=6.5)

# 3-1: Overall avg coop with mean line
ax = ax_ov[1]; ax.set_facecolor("#fdfdfd")
ax.plot(abs_rounds, round_coop_rate, color="#95a5a6", lw=1, alpha=0.7)
ax.axhline(overall_avg_coop, color="#27ae60", lw=2, ls="--",
           label=f"Avg={overall_avg_coop:.3f}")
ax.set_title("Overall Avg Coop", fontweight="bold", fontsize=8.5)
ax.set_xlabel("Round", fontsize=7); ax.set_ylabel("Rate", fontsize=7)
ax.legend(fontsize=7); ax.tick_params(labelsize=6.5); ax.grid(True, alpha=0.25)

# 3-2: Trend slope
ax = ax_ov[2]; ax.set_facecolor("#fdfdfd")
ax.scatter(abs_rounds, round_coop_rate, color="#95a5a6", s=18, alpha=0.7)
ax.plot(abs_rounds, trend_y, color="#e67e22", lw=2,
        label=f"Slope={coop_trend_slope:.5f}")
ax.set_title("Coop Trend Slope", fontweight="bold", fontsize=8.5)
ax.set_xlabel("Round", fontsize=7); ax.set_ylabel("Rate", fontsize=7)
ax.legend(fontsize=7); ax.tick_params(labelsize=6.5); ax.grid(True, alpha=0.25)

# 3-3: Rep-coop correlation scatter
ax = ax_ov[3]; ax.set_facecolor("#fdfdfd")
sc = ax.scatter(per_round_mean_rep, round_coop_rate, c=abs_rounds,
                cmap="viridis", s=25, alpha=0.8)
if np.std(per_round_mean_rep) > 0:
    m2_d, b2_d, _, _, _ = stats.linregress(per_round_mean_rep, round_coop_rate)
    xr_d = np.linspace(per_round_mean_rep.min(), per_round_mean_rep.max(), 100)
    corr_label = f"r={rep_coop_corr:.3f}" if not np.isnan(rep_coop_corr) else "r=N/A"
    ax.plot(xr_d, m2_d * xr_d + b2_d, color="#8e44ad", lw=2, label=corr_label)
    ax.legend(fontsize=7)
else:
    ax.text(0.5, 0.5, "r = N/A\n(reputation constant)", transform=ax.transAxes,
            ha="center", va="center", fontsize=8, color="#8e44ad")
ax.set_title("Rep–Coop Correlation", fontweight="bold", fontsize=8.5)
ax.set_xlabel("Mean Reputation", fontsize=7); ax.set_ylabel("Coop Rate", fontsize=7)
ax.tick_params(labelsize=6.5); ax.grid(True, alpha=0.25)

# ----- Section 4: Per-Agent (1 row × 4 cols) -----
inner4 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_gs[3], wspace=0.4)
ax_ag = [fig5.add_subplot(inner4[0, c]) for c in range(4)]

pa_datasets = [
    (agent_coop_ratio,   "Cooperation Ratio",      "Ratio",      bar_colors),
    (agent_mean_regret,  "Mean Regret",             "Regret",     "#9b59b6"),
    (agent_mean_forg,    "Mean Forgiveness Level",  "Forgiveness","#1abc9c"),
    (agent_lifetime_pay, "Lifetime Payoff",         "Resources",  bar_colors),
]
for ax, (arr, title, ylabel, cols) in zip(ax_ag, pa_datasets):
    ax.set_facecolor("#fdfdfd")
    if isinstance(cols, list):
        ax.bar(x_pos, arr, color=cols, alpha=0.85, edgecolor="white")
    else:
        ax.bar(x_pos, arr, color=cols, alpha=0.8, edgecolor="white")
    ax.set_title(title, fontweight="bold", fontsize=8.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"A{i}" for i in logical_agents], fontsize=6, rotation=45)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6.5); ax.grid(True, alpha=0.25, axis="y")

out5 = os.path.join(OUT_DIR, "plot_5_full_dashboard.png")
fig5.savefig(out5, dpi=150, bbox_inches="tight")
print(f"Saved: {out5}")
plt.close(fig5)

print("\n✓ All plots generated successfully.")
print(f"  {out1}")
print(f"  {out2}")
print(f"  {out2e}")
print(f"  {out3}")
print(f"  {out4}")
print(f"  {out5}")

# ─────────────────────────────────────────────────────────────────────────────
# TXT REPORT
# ─────────────────────────────────────────────────────────────────────────────
txt_path = os.path.join(OUT_DIR, "plot_output.txt")
SEP  = "=" * 68
SEP2 = "-" * 68

lines = []
def L(s=""): lines.append(s)

L(SEP)
L(f"  SIMULATION REPORT  —  {os.path.basename(DATA_FILE)}")
L(f"  Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
L(SEP)

# ── Hyperparameters ──────────────────────────────────────────────────────────
L()
L("HYPERPARAMETERS")
L(SEP2)
L(f"  Mechanisms              : {hp['mechanisms']}")
L(f"  Num agents              : {hp['num_agents']}")
L(f"  Num generations         : {hp['num_generations']}")
L(f"  Rounds per generation   : {hp['num_rounds_per_generation']}")
L(f"  Initial endowment       : {hp['initial_endowment']}")
L(f"  Enable regret           : {hp['enable_regret']}")
L(f"  Enable gossip           : {hp['enable_gossip']}")
L(f"  Enable forgiveness      : {hp['enable_forgiveness']}")
L(f"  Seed                    : {hp['seed']}")
L(f"  Payoffs  (C,C)          : {hp['payoffs']['(C,C)']}")
L(f"           (D,C)          : {hp['payoffs']['(D,C)']}")
L(f"           (C,D)          : {hp['payoffs']['(C,D)']}")
L(f"           (D,D)          : {hp['payoffs']['(D,D)']}")
L(f"  Total records           : {len(records)}")

# ── Loading summary ──────────────────────────────────────────────────────────
L()
L(SEP)
L(f"Loading data from {DATA_FILE}...")
L(f"Total records: {len(records)}")
L(f"Generations: {NUM_GENS}")
L(f"Agents: {NUM_AGENTS}")
L()
for p in [out1, out2, out2e, out3, out4, out5]:
    L(f"Visualization saved to: {p}")

# ── Strategy Evolution ───────────────────────────────────────────────────────
L()
L(SEP)
L("--- Strategy Evolution ---")
L(SEP2)
for g in range(1, NUM_GENS + 1):
    dist = gen_strategy_dist[g - 1]
    L(f"Gen {g:>2}: {dict(sorted(dist.items(), key=lambda x: -x[1]))}")

# ── Per-Round Metrics ────────────────────────────────────────────────────────
L()
L(SEP)
L("--- Per-Round Metrics ---")
L(SEP2)
L(f"  {'Round':>5}  {'Gen':>3}  {'Rnd':>3}  {'CoopRate':>9}  {'MeanRegret':>11}  {'RepVar':>9}  {'AvgRepDiff':>11}")
L(f"  {'-'*5}  {'-'*3}  {'-'*3}  {'-'*9}  {'-'*11}  {'-'*9}  {'-'*11}")
for i, ar in enumerate(abs_rounds):
    g   = int((ar - 1) // NUM_ROUNDS) + 1
    rnd = int((ar - 1) %  NUM_ROUNDS) + 1
    L(f"  {ar:>5}  {g:>3}  {rnd:>3}  {round_coop_rate[i]:>9.4f}  "
      f"{round_mean_regret[i]:>11.4f}  {round_rep_var[i]:>9.4f}  {round_avg_rep_diff[i]:>11.4f}")

# ── Per-Generation Metrics ───────────────────────────────────────────────────
L()
L(SEP)
L("--- Per-Generation Metrics ---")
L(SEP2)
L(f"  {'Gen':>3}  {'MeanRep':>8}  {'MeanForg':>9}  {'CoopStab':>9}  {'DomStrategy':<20}  {'AvgPayoff':>10}")
L(f"  {'-'*3}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*20}  {'-'*10}")
for g in range(1, NUM_GENS + 1):
    idx = g - 1
    dom_strat = max(gen_strategy_dist[idx], key=gen_strategy_dist[idx].get)
    avg_pay   = np.mean(list(gen_avg_payoff[idx].values()))
    L(f"  {g:>3}  {gen_mean_global_rep[idx]:>8.4f}  {gen_mean_forgiveness[idx]:>9.4f}  "
      f"{gen_coop_stability[idx]:>9.4f}  {dom_strat:<20}  {avg_pay:>10.2f}")

# ── Per-Generation Strategy Payoffs ─────────────────────────────────────────
L()
L(SEP)
L("--- Average Payoff per Strategy per Generation ---")
L(SEP2)
all_strats = sorted(set(s for d in gen_avg_payoff for s in d))
header = f"  {'Gen':>3}  " + "  ".join(f"{s[:14]:>14}" for s in all_strats)
L(header)
L("  " + "-" * (len(header) - 2))
for g in range(1, NUM_GENS + 1):
    row = f"  {g:>3}  "
    for s in all_strats:
        v = gen_avg_payoff[g - 1].get(s, float("nan"))
        row += f"  {v:>14.2f}" if not np.isnan(v) else f"  {'—':>14}"
    L(row)

# ── Overall Metrics ──────────────────────────────────────────────────────────
L()
L(SEP)
L("--- Overall Summary Metrics ---")
L(SEP2)
L(f"  Final cooperation rate       : {final_coop_rate:.4f}")
L(f"  Overall average cooperation  : {overall_avg_coop:.4f}")
L(f"  Cooperation trend slope      : {coop_trend_slope:+.6f}  "
  f"({'rising' if coop_trend_slope > 0 else 'falling' if coop_trend_slope < 0 else 'flat'})")
if np.isnan(rep_coop_corr):
    L(f"  Reputation–coop correlation  : N/A  (reputation constant — no reputation mechanism)")
else:
    L(f"  Reputation–coop correlation  : {rep_coop_corr:.4f}  (p={rep_coop_p:.4f})")

# ── Per-Agent Metrics ────────────────────────────────────────────────────────
L()
L(SEP)
L("--- Per-Agent Metrics ---")
L(SEP2)
L(f"  {'Agent':>7}  {'DomStrategy':<20}  {'CoopRatio':>9}  {'MeanRegret':>11}  {'MeanForg':>9}  {'LifetimePay':>12}")
L(f"  {'-'*7}  {'-'*20}  {'-'*9}  {'-'*11}  {'-'*9}  {'-'*12}")
for i, aid in enumerate(logical_agents):
    recs_a = [rec for rec in records if rec['agent_name'].endswith(f"_{aid}")]
    from collections import Counter as _C
    dom = _C(rec['strategy'] for rec in recs_a).most_common(1)[0][0]
    L(f"  {f'Agent {aid}':>7}  {dom:<20}  {agent_coop_ratio[i]:>9.4f}  "
      f"{agent_mean_regret[i]:>11.4f}  {agent_mean_forg[i]:>9.4f}  {agent_lifetime_pay[i]:>12.1f}")

# ── Cooperation Rate Summary Stats ───────────────────────────────────────────
L()
L(SEP)
L("--- Cooperation Rate Statistics (across all rounds) ---")
L(SEP2)
L(f"  Min    : {round_coop_rate.min():.4f}  (round {abs_rounds[round_coop_rate.argmin()]})")
L(f"  Max    : {round_coop_rate.max():.4f}  (round {abs_rounds[round_coop_rate.argmax()]})")
L(f"  Mean   : {round_coop_rate.mean():.4f}")
L(f"  Median : {np.median(round_coop_rate):.4f}")
L(f"  Std    : {round_coop_rate.std():.4f}")

L()
L(SEP)
L("END OF REPORT")
L(SEP)

with open(txt_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\n✓ TXT report saved: {txt_path}")

