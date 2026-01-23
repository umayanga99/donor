import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import os

# --- 1. Load the Data ---
# Automatically finds the most recent JSON result file
# list_of_files = glob.glob('latest_results/*.json')
latest_file = "../latest_results/claude_pd_forgiveness_20260119_114419.json"


with open(latest_file, 'r') as f:
    data = json.load(f)

# --- 2. Process Data into DataFrames ---
records = []
for agent_data in data['agents_data']:
    records.append({
        'Generation': agent_data['current_generation'],
        'Round': agent_data['round_number'],
        'Agent': agent_data['agent_name'],
        'Strategy': agent_data['strategy'],
        'Cooperated': 1 if agent_data['donated'] > 0 else 0, # 1=Coop, 0=Defect
        'Score': agent_data['resources'],
        'Forgiveness_Avg': agent_data.get('forgiveness_given', 0.5)
    })

df = pd.DataFrame(records)

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# --- VISUALIZATION 1: THE COOPERATION MATRIX (Heatmap) ---
# Shows the % of Cooperation for every Round in every Generation
# Goal: See the "cooling down" of trust.

pivot_coop = df.pivot_table(
    index='Generation',
    columns='Round',
    values='Cooperated',
    aggfunc='mean'
)

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_coop, annot=True, cmap="RdYlGn", vmin=0, vmax=1, fmt=".2f")
plt.title('The "Collapse of Trust" Matrix\n(Average Cooperation Rate)', fontsize=16)
plt.ylabel('Generation (Evolution)', fontsize=12)
plt.xlabel('Round Number', fontsize=12)
plt.tight_layout()
plt.show()

# --- VISUALIZATION 2: STRATEGY DOMINANCE (Stacked Chart) ---
# Shows which strategies survived.
# Goal: Visualize "Always Defect" taking over.

strategy_counts = df.groupby(['Generation', 'Strategy'])['Agent'].nunique().unstack(fill_value=0)
strategy_counts = strategy_counts.div(strategy_counts.sum(axis=1), axis=0) * 100 # Convert to %

ax = strategy_counts.plot(kind='area', stacked=True, figsize=(12, 6), colormap='tab20', alpha=0.8)
plt.title('Evolution of Strategies: The Victory of Defection', fontsize=16)
plt.ylabel('Population Share (%)', fontsize=12)
plt.xlabel('Generation', fontsize=12)
plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.margins(0, 0)
plt.tight_layout()
plt.show()

# --- VISUALIZATION 3: THE FORGIVENESS TRAP (Scatter) ---
# Did being nice pay off?
# Goal: Show that high Forgiveness = Low Survival (in this specific setup).

final_round_df = df[df['Round'] == 5] # Only look at end of generations

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=final_round_df,
    x='Forgiveness_Avg',
    y='Score',
    hue='Strategy',
    style='Generation',
    s=100,
    palette='deep',
    alpha=0.7
)
plt.title('The Forgiveness Trap: Trust vs. Survival Resources', fontsize=16)
plt.xlabel('Average Forgiveness Level (Trust in Others)', fontsize=12)
plt.ylabel('Final Resources (Score)', fontsize=12)
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
plt.text(0.6, final_round_df['Score'].max(), "High Trust Zone (Dangerous)", color='red')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()