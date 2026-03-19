"""
single_number_metrics.py
─────────────────────────
For every JSON result file in latest_results/ compute ONE scalar per metric:

Per-Round metrics
  • cooperation_rate        – mean cooperation rate across all rounds
  • mean_regret             – mean regret level across all rounds
  • reputation_variance     – mean per-round reputation variance
  • avg_rep_diff_pairs      – mean per-round avg |rep_i − rep_j| within pairs

Per-Generation metrics
  • strategy_entropy        – mean Shannon entropy of strategy distribution per generation
  • avg_payoff              – mean payoff (resources) across all strategies & generations
  • mean_global_reputation  – mean reputation across all records
  • mean_forgiveness        – mean forgiveness_given across all records
  • coop_stability          – mean per-generation variance of cooperation_rate (lower = more stable)

Overall metrics
  • final_coop_rate         – cooperation rate in the very last round
  • overall_avg_coop        – mean cooperation rate across all rounds (same as above per-round mean)
  • coop_trend_slope        – slope of linear regression of coop rate over absolute rounds
  • rep_coop_corr           – Pearson r between per-round mean reputation & cooperation rate

Per-Agent metrics
  • agent_coop_ratio        – mean cooperation ratio across all agents
  • agent_mean_regret       – mean regret across all agents
  • agent_mean_forgiveness  – mean forgiveness across all agents
  • agent_lifetime_payoff   – mean lifetime payoff (final resources) across all agents

Output
  • plot/new_matrices/single_number_matrices.csv   – machine-readable table
  • plot/new_matrices/single_number_matrices.txt   – human-readable report
  • plot/new_matrices/single_number_matrices.png   – heatmap comparison chart
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "latest_results")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "new_matrices")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Discover all JSON result files ────────────────────────────────────────────
json_files = sorted(
    f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")
)
print(f"Found {len(json_files)} result files:")
for f in json_files:
    print(f"  {f}")

# ── Metric definitions (ordered) ──────────────────────────────────────────────
METRIC_GROUPS = {
    "Per-Round": [
        "cooperation_rate",
        "mean_regret",
        "reputation_variance",
        "avg_rep_diff_pairs",
    ],
    "Per-Generation": [
        "strategy_entropy",
        "avg_payoff",
        "mean_global_reputation",
        "mean_forgiveness",
        "coop_stability",
    ],
    "Overall": [
        "final_coop_rate",
        "overall_avg_coop",
        "coop_trend_slope",
        "rep_coop_corr",
    ],
    "Per-Agent": [
        "agent_coop_ratio",
        "agent_mean_regret",
        "agent_mean_forgiveness",
        "agent_lifetime_payoff",
    ],
}

METRIC_LABELS = {
    "cooperation_rate":       "Cooperation Rate (mean over all rounds)",
    "mean_regret":            "Mean Regret (mean over all rounds)",
    "reputation_variance":    "Reputation Variance (mean per-round variance)",
    "avg_rep_diff_pairs":     "Avg Rep Diff Within Pairs (mean over rounds)",
    "strategy_entropy":       "Strategy Entropy (mean Shannon entropy per gen)",
    "avg_payoff":             "Avg Payoff / Resources (mean over strats & gens)",
    "mean_global_reputation": "Mean Global Reputation (all records)",
    "mean_forgiveness":       "Mean Forgiveness Given (all records)",
    "coop_stability":         "Coop Stability – variance of coop_rate (lower=stable)",
    "final_coop_rate":        "Final Cooperation Rate (last round)",
    "overall_avg_coop":       "Overall Avg Cooperation (all rounds)",
    "coop_trend_slope":       "Cooperation Trend Slope (linear regression)",
    "rep_coop_corr":          "Reputation–Coop Correlation (Pearson r)",
    "agent_coop_ratio":       "Agent Cooperation Ratio (mean across agents)",
    "agent_mean_regret":      "Agent Mean Regret (mean across agents)",
    "agent_mean_forgiveness": "Agent Mean Forgiveness (mean across agents)",
    "agent_lifetime_payoff":  "Agent Lifetime Payoff (mean final resources)",
}

ALL_METRICS = [m for group in METRIC_GROUPS.values() for m in group]


def compute_metrics(filepath):
    """Return a dict of metric_name → scalar value for one result file."""
    with open(filepath) as f:
        raw = json.load(f)

    hp      = raw["hyperparameters"]
    records = raw["agents_data"]

    NUM_GENS   = hp["num_generations"]
    NUM_ROUNDS = hp["num_rounds_per_generation"]
    NUM_AGENTS = hp["num_agents"]

    # ── Group records ──────────────────────────────────────────────────────────
    by_gen_round = defaultdict(list)
    by_gen       = defaultdict(list)
    for r in records:
        by_gen_round[(r["current_generation"], r["round_number"])].append(r)
        by_gen[r["current_generation"]].append(r)

    # ── Per-Round aggregates ───────────────────────────────────────────────────
    abs_rounds          = []
    rnd_coop_rate       = []
    rnd_mean_regret     = []
    rnd_rep_var         = []
    rnd_avg_rep_diff    = []
    per_round_mean_rep  = []

    for g in range(1, NUM_GENS + 1):
        for rnd in range(1, NUM_ROUNDS + 1):
            grp = by_gen_round[(g, rnd)]
            abs_rounds.append((g - 1) * NUM_ROUNDS + rnd)

            rnd_coop_rate.append(np.mean([float(r["donated"]) for r in grp]))
            rnd_mean_regret.append(np.mean([r["regret_level"] for r in grp]))
            reps = [r["reputation"] for r in grp]
            rnd_rep_var.append(np.var(reps))
            per_round_mean_rep.append(np.mean(reps))

            # avg |rep_i - rep_j| within pairs
            paired = {rec["agent_name"]: rec for rec in grp}
            diffs, seen = [], set()
            for rec in grp:
                partner = rec["paired_with"]
                key = tuple(sorted([rec["agent_name"], partner]))
                if key not in seen and partner in paired:
                    seen.add(key)
                    diffs.append(abs(rec["reputation"] - paired[partner]["reputation"]))
            rnd_avg_rep_diff.append(np.mean(diffs) if diffs else 0.0)

    abs_rounds         = np.array(abs_rounds)
    rnd_coop_rate      = np.array(rnd_coop_rate)
    rnd_mean_regret    = np.array(rnd_mean_regret)
    rnd_rep_var        = np.array(rnd_rep_var)
    rnd_avg_rep_diff   = np.array(rnd_avg_rep_diff)
    per_round_mean_rep = np.array(per_round_mean_rep)

    # ── Per-Generation aggregates ──────────────────────────────────────────────
    gen_strategy_dist   = []
    gen_avg_payoff_vals = []
    gen_mean_rep        = []
    gen_mean_forg       = []
    gen_coop_stability  = []

    for g in range(1, NUM_GENS + 1):
        grp = by_gen[g]

        # last-round record per agent
        last_recs = {}
        for rec in grp:
            a = rec["agent_name"]
            if a not in last_recs or rec["round_number"] > last_recs[a]["round_number"]:
                last_recs[a] = rec

        strat_counts = defaultdict(int)
        strat_res    = defaultdict(list)
        for rec in last_recs.values():
            strat_counts[rec["strategy"]] += 1
            strat_res[rec["strategy"]].append(rec["resources"])
        gen_strategy_dist.append(dict(strat_counts))

        # mean payoff across strategies
        gen_avg_payoff_vals.append(
            np.mean([np.mean(v) for v in strat_res.values()])
        )

        gen_mean_rep.append(np.mean([rec["reputation"] for rec in grp]))
        gen_mean_forg.append(np.mean([rec["forgiveness_given"] for rec in grp]))
        gen_coop_stability.append(np.var([rec.get("cooperation_rate", float(rec["donated"])) for rec in grp]))

    # Shannon entropy of strategy distribution per generation
    def shannon_entropy(counts_dict):
        total = sum(counts_dict.values())
        if total == 0:
            return 0.0
        probs = np.array([v / total for v in counts_dict.values()])
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    strategy_entropies = [shannon_entropy(d) for d in gen_strategy_dist]

    # ── Overall metrics ────────────────────────────────────────────────────────
    final_coop_rate = rnd_coop_rate[-1]
    overall_avg_coop = float(np.mean(rnd_coop_rate))

    slope, intercept, *_ = stats.linregress(abs_rounds, rnd_coop_rate)
    coop_trend_slope = float(slope)

    if np.std(per_round_mean_rep) == 0 or np.std(rnd_coop_rate) == 0:
        rep_coop_corr = float("nan")
    else:
        rep_coop_corr, _ = stats.pearsonr(per_round_mean_rep, rnd_coop_rate)
        rep_coop_corr = float(rep_coop_corr)

    # ── Per-Agent aggregates ───────────────────────────────────────────────────
    logical_agents = list(range(1, NUM_AGENTS + 1))
    agent_coop     = []
    agent_regret   = []
    agent_forg     = []
    agent_pay      = []

    for aid in logical_agents:
        recs_a = [rec for rec in records if rec["agent_name"].endswith(f"_{aid}")]
        if not recs_a:
            continue
        agent_coop.append(np.mean([float(rec["donated"]) for rec in recs_a]))
        agent_regret.append(np.mean([rec["regret_level"] for rec in recs_a]))
        agent_forg.append(np.mean([rec["forgiveness_given"] for rec in recs_a]))
        last_rec = max(recs_a, key=lambda r: (r["current_generation"], r["round_number"]))
        agent_pay.append(last_rec["resources"])

    return {
        # Per-Round
        "cooperation_rate":       float(np.mean(rnd_coop_rate)),
        "mean_regret":            float(np.mean(rnd_mean_regret)),
        "reputation_variance":    float(np.mean(rnd_rep_var)),
        "avg_rep_diff_pairs":     float(np.mean(rnd_avg_rep_diff)),
        # Per-Generation
        "strategy_entropy":       float(np.mean(strategy_entropies)),
        "avg_payoff":             float(np.mean(gen_avg_payoff_vals)),
        "mean_global_reputation": float(np.mean(gen_mean_rep)),
        "mean_forgiveness":       float(np.mean(gen_mean_forg)),
        "coop_stability":         float(np.mean(gen_coop_stability)),
        # Overall
        "final_coop_rate":        float(final_coop_rate),
        "overall_avg_coop":       float(overall_avg_coop),
        "coop_trend_slope":       float(coop_trend_slope),
        "rep_coop_corr":          rep_coop_corr,
        # Per-Agent
        "agent_coop_ratio":       float(np.mean(agent_coop)),
        "agent_mean_regret":      float(np.mean(agent_regret)),
        "agent_mean_forgiveness": float(np.mean(agent_forg)),
        "agent_lifetime_payoff":  float(np.mean(agent_pay)),
    }


# ── Compute metrics for every file ────────────────────────────────────────────
results = {}   # filename → metrics dict
for fname in json_files:
    fpath = os.path.join(RESULTS_DIR, fname)
    print(f"\nProcessing: {fname}")
    metrics = compute_metrics(fpath)
    results[fname] = metrics
    for m, v in metrics.items():
        print(f"  {m:<30} {v:.6f}" if not (isinstance(v, float) and np.isnan(v)) else f"  {m:<30} N/A")

# ── Short labels for table display ────────────────────────────────────────────
def short_name(fname):
    """Strip .json, keep last 2 underscore-separated tokens as date/time."""
    base = fname.replace(".json", "")
    parts = base.split("_")
    # e.g. claude_pd_baseline_20260118_224004 → baseline_20260118_224004
    # Keep mechanism name + date + time
    mech_idx = 2  # after "claude_pd"
    return "_".join(parts[mech_idx:])

file_labels = [short_name(f) for f in json_files]

# ── Write CSV ─────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "single_number_matrices.csv")
with open(csv_path, "w") as f:
    # Header
    f.write("metric," + ",".join(file_labels) + "\n")
    for metric in ALL_METRICS:
        row = [metric]
        for fname in json_files:
            v = results[fname][metric]
            row.append("N/A" if (isinstance(v, float) and np.isnan(v)) else f"{v:.6f}")
        f.write(",".join(row) + "\n")
print(f"\nSaved CSV: {csv_path}")

# ── Write TXT ─────────────────────────────────────────────────────────────────
txt_path = os.path.join(OUT_DIR, "single_number_matrices.txt")
SEP  = "=" * 120
SEP2 = "-" * 120
COL_W = 22   # column width per file

lines = []
def L(s=""): lines.append(s)

L(SEP)
L("  SINGLE-NUMBER METRICS COMPARISON")
L(f"  Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
L(f"  Files compared: {len(json_files)}")
L(SEP)

# Table header
header = f"  {'Metric':<36}" + "".join(f"{lbl[:COL_W-1]:>{COL_W}}" for lbl in file_labels)
L(header)
L("  " + "-" * (36 + COL_W * len(json_files)))

for group_name, metrics in METRIC_GROUPS.items():
    L()
    L(f"  ── {group_name} ──")
    for metric in metrics:
        row = f"  {metric:<36}"
        for fname in json_files:
            v = results[fname][metric]
            cell = "N/A" if (isinstance(v, float) and np.isnan(v)) else f"{v:.5f}"
            row += f"{cell:>{COL_W}}"
        L(row)

L()
L(SEP)
L("METRIC DESCRIPTIONS")
L(SEP2)
for group_name, metrics in METRIC_GROUPS.items():
    L(f"\n  {group_name}:")
    for m in metrics:
        L(f"    {m:<36} {METRIC_LABELS[m]}")
L()
L(SEP)
L("END OF REPORT")
L(SEP)

with open(txt_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved TXT: {txt_path}")

# ── Plot: heatmap of all metrics × all files ───────────────────────────────────
matrix = np.array([
    [results[fname][m] for fname in json_files]
    for m in ALL_METRICS
], dtype=float)

# Normalise each row to [0,1] for colour (NaN → 0)
matrix_norm = np.copy(matrix)
for i in range(len(ALL_METRICS)):
    row = matrix_norm[i]
    finite = row[np.isfinite(row)]
    if len(finite) == 0 or finite.max() == finite.min():
        matrix_norm[i] = 0.5
    else:
        matrix_norm[i] = np.where(
            np.isfinite(row),
            (row - finite.min()) / (finite.max() - finite.min()),
            0.5
        )

# Short metric names for y-axis
short_metrics = [
    "coop_rate", "mean_regret", "rep_variance", "rep_diff_pairs",
    "strat_entropy", "avg_payoff", "mean_rep", "mean_forg", "coop_stability",
    "final_coop", "overall_coop", "trend_slope", "rep_coop_corr",
    "agent_coop", "agent_regret", "agent_forg", "agent_payoff",
]

# Group separator y-positions
group_sizes = [len(v) for v in METRIC_GROUPS.values()]
group_borders = np.cumsum(group_sizes)[:-1] - 0.5   # lines between groups

n_metrics = len(ALL_METRICS)
n_files   = len(json_files)

fig_h = max(8, n_metrics * 0.55 + 3)
fig_w = max(10, n_files * 1.8 + 4)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("#f8f9fa")

im = ax.imshow(matrix_norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

# Cell annotations (raw values)
for i in range(n_metrics):
    for j in range(n_files):
        v = matrix[i, j]
        txt = "N/A" if not np.isfinite(v) else (
            f"{v:.3f}" if abs(v) < 1000 else f"{v:.0f}"
        )
        brightness = matrix_norm[i, j]
        color = "black" if 0.25 < brightness < 0.75 else ("white" if brightness <= 0.25 else "black")
        ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                color=color, fontweight="bold")

# Axes labels
ax.set_xticks(range(n_files))
ax.set_xticklabels(file_labels, rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(n_metrics))
ax.set_yticklabels(short_metrics, fontsize=8.5)

# Group separator lines + labels
group_names = list(METRIC_GROUPS.keys())
group_label_positions = [0] + list(np.cumsum(group_sizes))
for border in group_borders:
    ax.axhline(border, color="white", lw=2.5)

# Right-side group labels
for idx, (gname, gsize) in enumerate(zip(group_names, group_sizes)):
    mid = group_label_positions[idx] + gsize / 2 - 0.5
    ax.text(n_files - 0.35, mid, gname, ha="left", va="center",
            fontsize=8, fontweight="bold", color="#2c3e50",
            transform=ax.get_yaxis_transform())

cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.12)
cbar.set_label("Normalised value (row-wise)", fontsize=8)

ax.set_title(
    "Single-Number Metrics per Test Run\n(colour = row-normalised; values shown are raw)",
    fontsize=12, fontweight="bold", pad=12
)
plt.tight_layout()

png_path = os.path.join(OUT_DIR, "single_number_matrices.png")
fig.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved PNG: {png_path}")

print("\n✓ All outputs written to:", OUT_DIR)
print(f"  {csv_path}")
print(f"  {txt_path}")
print(f"  {png_path}")

