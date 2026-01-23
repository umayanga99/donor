import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Define the directory containing the results
RESULTS_DIR = '/home/indunil/personal/DONOR/latest_results'
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'mechanism_comparison.png')

def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Determine label from filename
        filename = os.path.basename(file_path)
        # Assuming format claude_pd_<mechanism>_<date>.json
        parts = filename.split('_')
        # Extract mechanism part (everything between 'claude_pd_' and the date)
        # Example: claude_pd_regret_gossip_forgiveness_2025... -> regret_gossip_forgiveness
        # Example: claude_pd_baseline_2026... -> baseline

        # Simple heuristic: join parts from index 2 up to the one that starts with '20' (year)
        mechanism_parts = []
        for part in parts[2:]:
            if part.startswith('20') and len(part) == 8 and part.isdigit():
                break
            mechanism_parts.append(part)

        mechanism = "_".join(mechanism_parts)
        if not mechanism:
            mechanism = "Unknown"

        if 'agents_data' not in data:
            print(f"Skipping {filename}: 'agents_data' not found.")
            return None

        df = pd.DataFrame(data['agents_data'])
        df['Mechanism'] = mechanism

        # Convert numeric columns
        cols = ['current_generation', 'donated', 'received']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    print("Finding JSON files...")
    json_files = glob.glob(os.path.join(RESULTS_DIR, '*.json'))

    all_dfs = []

    for f in json_files:
        print(f"Processing {os.path.basename(f)}...")
        df = load_data(f)
        if df is not None and not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("No valid data found.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    print(f"Combined data: {len(combined_df)} records.")

    # Aggregating per generation per mechanism
    # Calculate Mean and CI (Seaborn does CI automatically, but aggregating makes it faster for lineplot)

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Comparison of Social Mechanisms in Prisoner\'s Dilemma', fontsize=18)

    # 1. Cooperation Rate
    sns.lineplot(
        data=combined_df,
        x='current_generation',
        y='donated',
        hue='Mechanism',
        ax=axes[0],
        marker='o',
        linewidth=2.5,
        palette='tab10'
    )
    axes[0].set_title('Cooperation Rate over Generations', fontsize=14)
    axes[0].set_ylabel('Cooperation Rate (1=Coop, 0=Defect)')
    axes[0].set_xlabel('Generation')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(title='Mechanism', loc='best')

    # 2. Average Points
    sns.lineplot(
        data=combined_df,
        x='current_generation',
        y='received',
        hue='Mechanism',
        ax=axes[1],
        marker='s',
        linewidth=2.5,
        palette='tab10'
    )
    axes[1].set_title('Average Score per Interaction', fontsize=14)
    axes[1].set_ylabel('Avg Points')
    axes[1].set_xlabel('Generation')

    # Add Payoff References
    axes[1].axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Mutual Coop (3)')
    axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.3, label='Mutual Defect (1)')

    # Fix legend duplications if any (seaborn handles hue well generally)
    # axes[1].legend(title='Mechanism')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"Comparison plot saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

