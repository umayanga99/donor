import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load the JSON data
# Adjust filename if needed to match the user's specific file
file_path = '/home/indunil/personal/DONOR/latest_results/claude_pd_gossip_20260118_234717.json'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

print(f"Loading data from {file_path}...")
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract agents data
if 'agents_data' in data:
    agents_data = data['agents_data']
else:
    print("Error: 'agents_data' not found in JSON.")
    exit(1)

df = pd.DataFrame(agents_data)

# Ensure numeric columns
# Note: 'donated' in PD is typically 1 (Cooperate) or 0 (Defect)
numeric_cols = ['current_generation', 'round_number', 'resources', 'donated', 'received', 'regret_level', 'reputation']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Print basic stats to console
print(f"Total records: {len(df)}")
print(f"Generations: {len(df['current_generation'].unique())}")
print(f"Agents: {df['agent_name'].nunique()}")

# Setup plots
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Prisoner\'s Dilemma Simulation Analysis\n{os.path.basename(file_path)}', fontsize=16)

# ---------------------------------------------------------
# 1. Cooperation Rate over Generations
# ---------------------------------------------------------
# We average 'donated' (1=Coop, 0=Defect) per generation
coop_per_gen = df.groupby('current_generation')['donated'].mean().reset_index()

sns.lineplot(data=coop_per_gen, x='current_generation', y='donated', ax=axes[0, 0], marker='o', linewidth=3, color='dodgerblue')
axes[0, 0].set_title('Average Cooperation Rate per Generation')
axes[0, 0].set_ylabel('Cooperation Rate (0.0 - 1.0)')
axes[0, 0].set_xlabel('Generation')
axes[0, 0].set_ylim(-0.05, 1.05)
axes[0, 0].grid(True)

# ---------------------------------------------------------
# 2. Average Payoff per Round
# ---------------------------------------------------------
# 'received' is the points earned from interaction
# PD Payoffs: 5 (Temptation), 3 (Reward), 1 (Punishment), 0 (Sucker)
payoff_per_gen = df.groupby('current_generation')['received'].mean().reset_index()

sns.lineplot(data=payoff_per_gen, x='current_generation', y='received', ax=axes[0, 1], marker='s', color='forestgreen', linewidth=3)
axes[0, 1].set_title('Average Payoff (Score) per Interaction')
axes[0, 1].set_ylabel('Avg Points Received')
axes[0, 1].set_xlabel('Generation')
axes[0, 1].grid(True)
# Add reference lines for standard PD payoffs
axes[0, 1].axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Mutual Coop (3)')
axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.3, label='Mutual Defect (1)')
axes[0, 1].legend()

# ---------------------------------------------------------
# 3. Social Metrics: Regret & Reputation
# ---------------------------------------------------------
# Check which columns are available
metrics_to_plot = []
if 'regret_level' in df.columns:
    metrics_to_plot.append('regret_level')
if 'reputation' in df.columns:
    metrics_to_plot.append('reputation')

social_per_gen = df.groupby('current_generation')[metrics_to_plot].mean().reset_index()

if metrics_to_plot:
    # Melt for easier plotting with seaborn
    social_melted = social_per_gen.melt(id_vars=['current_generation'], value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
    sns.lineplot(data=social_melted, x='current_generation', y='Score', hue='Metric', ax=axes[1, 0], marker='^', linewidth=2.5)
    axes[1, 0].set_title('Evolution of Mechanism Metrics')
    axes[1, 0].set_ylabel('Score (Normalized)')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].grid(True)
else:
    axes[1, 0].text(0.5, 0.5, "No Regret/Reputation Data", ha='center')

# ---------------------------------------------------------
# 4. Strategy Analysis (Text) or Resource Accumulation
# ---------------------------------------------------------
# Let's look at average final resources per generation
# Finding the max resources for each agent in each generation
end_gen_resources = df.groupby(['current_generation', 'agent_name'])['resources'].max().reset_index()
avg_end_resources = end_gen_resources.groupby('current_generation')['resources'].mean().reset_index()

sns.barplot(data=avg_end_resources, x='current_generation', y='resources', ax=axes[1, 1], hue='current_generation', palette="viridis", legend=False)
axes[1, 1].set_title('Average Final Resources per Generation')
axes[1, 1].set_ylabel('Total Resources')
axes[1, 1].set_xlabel('Generation')

plt.tight_layout()

# Save plot
output_img = file_path + '.png'
plt.savefig(output_img)
print(f"Visualization saved to: {output_img}")

# Show unique strategies if any
if 'strategy' in df.columns:
    print("\n--- Strategy Evolution ---")
    # Show top strategy per generation
    for gen in sorted(df['current_generation'].unique()):
        gen_df = df[df['current_generation'] == gen]
        top_strategies = gen_df['strategy'].value_counts().head(3)
        print(f"Gen {gen}: {top_strategies.to_dict()}")

