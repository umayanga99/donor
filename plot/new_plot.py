import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ---------------------------------------------------------
# ⚙️ CONFIGURATION: PASTE YOUR FILENAME HERE
# ---------------------------------------------------------
TARGET_FILE = "../latest_results/claude_pd_baseline_20260222_151130.json"
# ---------------------------------------------------------

# Set plot style for academic/professional look
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")


def load_data(filename):
    """Loads the specific JSON simulation file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ The file '{filename}' was not found. Please check the path.")

    print(f"📂 Loading data from: {filename}")

    with open(filename, 'r') as f:
        data = json.load(f)

    # Flatten the data into a Pandas DataFrame
    df = pd.DataFrame(data['agents_data'])

    # Calculate explicit 'Choice' column for easier analysis
    df['choice'] = df['donated'].apply(lambda x: 'Cooperate' if x > 0 else 'Defect')

    # Normalize generations (if they started at 0 or 1, ensure 1-based for plotting)
    if df['current_generation'].min() == 0:
        df['current_generation'] += 1

    return df, data['hyperparameters']


def plot_evolutionary_dynamics(df):
    """
    Plot 1: The 'Battle Royale' View
    Shows which strategies survived and how global cooperation trended.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- 1A. Strategy Composition (Stacked Bar) ---
    # Filter for round 1 to count each agent once per generation (population census)
    gen_strategy = df[df['round_number'] == 1].groupby(['current_generation', 'strategy']).size().unstack(fill_value=0)

    # Normalize to percentage to see relative dominance
    gen_strategy_pct = gen_strategy.div(gen_strategy.sum(axis=1), axis=0) * 100

    gen_strategy_pct.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis', alpha=0.9, width=0.8)
    ax1.set_title('Evolution of Strategies (Population Share %)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Percentage of Population')
    ax1.legend(title='Strategy', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # --- 1B. Global Cooperation Rate ---
    # Calculate mean donation (0 or 1) per generation
    coop_rate = df.groupby('current_generation')['donated'].mean() * 100

    sns.lineplot(x=coop_rate.index, y=coop_rate.values, ax=ax2, marker='o', markersize=8, linewidth=3,
                 color='dodgerblue')

    # Add trend line
    if len(coop_rate) > 1:
        z = np.polyfit(coop_rate.index, coop_rate.values, 1)
        p = np.poly1d(z)
        ax2.plot(coop_rate.index, p(coop_rate.index), "r--", alpha=0.5, label='Trend')

    ax2.set_title('Global Cooperation Rate (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cooperation %')
    ax2.set_xlabel('Generation')
    ax2.set_ylim(0, 105)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('plot_1_evolution.png', dpi=300)
    print("✅ Generated: plot_1_evolution.png")


def plot_regret_matrix(df):
    """
    Plot 2: The 'Regret vs Outcome' Matrix
    Did agents feel regret when they were supposed to?
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Categorize outcomes:
    # Reward (3pts) = Coop/Coop
    # Sucker (0pts) = Coop/Defect
    # Temptation (5pts) = Defect/Coop
    # Punishment (1pts) = Defect/Defect

    def classify_outcome(row):
        # We need the partner's choice to know the outcome.
        # Since we don't have it directly in the row, we infer from score.
        # Score 3 -> CC, 0 -> CD, 5 -> DC, 1 -> DD
        score = row['received']
        if score == 3: return "Mutual Coop (R)"
        if score == 0: return "Sucker (S)"
        if score == 5: return "Temptation (T)"
        if score == 1: return "Mutual Defect (P)"
        return "Unknown"

    df['outcome_type'] = df.apply(classify_outcome, axis=1)

    # Boxplot of Regret levels by Outcome
    sns.boxplot(x='outcome_type', y='regret_level', data=df,
                order=["Mutual Coop (R)", "Temptation (T)", "Mutual Defect (P)", "Sucker (S)"], palette="Set3", ax=ax)
    sns.stripplot(x='outcome_type', y='regret_level', data=df, color='black', alpha=0.1, jitter=True,
                  order=["Mutual Coop (R)", "Temptation (T)", "Mutual Defect (P)", "Sucker (S)"])

    ax.set_title('Did Agents Learn? (Regret Level by Game Outcome)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Regret Level (0=None, 1=High)')
    ax.set_xlabel('Game Outcome')

    # Annotation
    text = "Expectation:\n- Sucker (S): Highest Regret\n- Mutual Coop (R): Lowest Regret"
    ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('plot_2_regret_outcomes.png', dpi=300)
    print("✅ Generated: plot_2_regret_outcomes.png")


def plot_interaction_matrix(df):
    """
    Plot 3: The 'Who Trusts Whom' Matrix
    A heatmap showing average cooperation rates between different strategies.
    This is the most useful matrix for analyzing strategy dominance.
    """
    plt.figure(figsize=(12, 10))

    # Create a mapping of agent name to strategy for every generation
    # Key: (Generation, Name), Value: Strategy
    strat_map = df.set_index(['current_generation', 'agent_name'])['strategy'].to_dict()

    def get_partner_strat(row):
        try:
            return strat_map.get((row['current_generation'], row['paired_with']), 'Unknown')
        except:
            return 'Unknown'

    df['partner_strategy'] = df.apply(get_partner_strat, axis=1)

    # Filter out 'Unknown' or None
    clean_df = df[df['partner_strategy'] != 'Unknown']

    # Calculate Cooperation Rate (0 to 1)
    # Pivot: Index=My Strategy, Columns=Opponent Strategy, Values=Donated (0 or 1)
    matrix = clean_df.pivot_table(
        index='strategy',
        columns='partner_strategy',
        values='donated',
        aggfunc='mean'
    )

    # Fill NaN with 0 (if strategies never met)
    matrix = matrix.fillna(0)

    # Plot Heatmap
    sns.heatmap(matrix, annot=True, cmap='RdYlGn', vmin=0, vmax=1.0, fmt='.2%', linewidths=.5,
                cbar_kws={'label': 'Cooperation Rate'})

    plt.title('Strategy Interaction Matrix: Who Cooperates with Whom?', fontsize=16, fontweight='bold')
    plt.ylabel('Agent Strategy (Actor)', fontsize=12)
    plt.xlabel('Partner Strategy (Recipient)', fontsize=12)

    plt.tight_layout()
    plt.savefig('plot_3_interaction_matrix.png', dpi=300)
    print("✅ Generated: plot_3_interaction_matrix.png")


def main():
    try:
        # Load Data
        df, params = load_data(TARGET_FILE)

        print(f"📊 Analyzing {len(df)} records from {params['num_generations']} generations...")

        # Generate Plots
        plot_evolutionary_dynamics(df)
        plot_regret_matrix(df)
        plot_interaction_matrix(df)

        print("\n✨ Analysis Complete! Check the .png files in this folder.")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()