"""
Visualization: Social Incentive Normalization Fix (Matthew Effect Prevention)

This script demonstrates:
1. The PROBLEM: Percentage-based bonuses create wealth compounding
2. The SOLUTION: Fixed-point normalized bonuses maintain fairness
3. Real simulation data showing the impact
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_simulation_data(filepath):
    """Load JSON simulation results"""
    with open(filepath, 'r') as f:
        return json.load(f)


def simulate_matthew_effect(initial_scores, bonus_percent=0.20):
    """
    Simulate the OLD approach: percentage-based bonuses
    Shows how wealth compounds over generations
    """
    scores = np.array(initial_scores, dtype=float)
    generations = []
    generations.append(scores.copy())

    for gen in range(4):  # 4 more generations
        # Apply percentage bonus
        coop_ratio = np.random.uniform(0.4, 0.8, len(scores))  # Simulated cooperation
        bonus = np.where(coop_ratio > 0.7, scores * bonus_percent, 0)
        penalty = np.where(coop_ratio < 0.3, -scores * bonus_percent, 0)

        scores = scores + bonus + penalty
        scores = np.maximum(scores, 1.0)  # Can't go below 1
        generations.append(scores.copy())

    return np.array(generations)


def simulate_fixed_point_incentive(initial_scores, base_bonus=3.0, initial_endowment=10.0):
    """
    Simulate the NEW approach: fixed-point normalized bonuses
    Shows how fairness is maintained across generations
    """
    scores = np.array(initial_scores, dtype=float)
    generations = []
    generations.append(scores.copy())

    for gen in range(4):  # 4 more generations
        # Calculate generation average
        gen_avg = np.mean(scores)

        # Apply fixed-point bonus (normalized by generation average)
        normalized_bonus = base_bonus * (initial_endowment / max(1.0, gen_avg))

        # Simulated cooperation
        coop_ratio = np.random.uniform(0.4, 0.8, len(scores))
        bonus = np.where(coop_ratio > 0.7, normalized_bonus, 0)
        penalty = np.where(coop_ratio < 0.3, -normalized_bonus, 0)

        scores = scores + bonus + penalty
        scores = np.maximum(scores, 1.0)  # Can't go below 1
        generations.append(scores.copy())

    return np.array(generations)


def create_comparison_figure():
    """Create side-by-side comparison of old vs new approach"""
    fig = plt.figure(figsize=(16, 12))

    # Example scenario: 6 agents with varying initial scores
    initial_scores = np.array([8.0, 12.0, 10.0, 9.0, 11.0, 10.5])
    agent_labels = [f"Agent {i+1}" for i in range(len(initial_scores))]

    # Simulate both approaches
    old_generations = simulate_matthew_effect(initial_scores, bonus_percent=0.20)
    new_generations = simulate_fixed_point_incentive(initial_scores, base_bonus=3.0, initial_endowment=10.0)

    # ===== PANEL 1: Old Approach (Percentage-based) =====
    ax1 = plt.subplot(2, 3, 1)
    for i in range(len(agent_labels)):
        ax1.plot(range(5), old_generations[:, i], marker='o', label=agent_labels[i], linewidth=2)
    ax1.set_xlabel('Generation', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('OLD: Percentage-Based Bonus\n(±20% of current wealth)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best')

    # ===== PANEL 2: New Approach (Fixed-point normalized) =====
    ax2 = plt.subplot(2, 3, 2)
    for i in range(len(agent_labels)):
        ax2.plot(range(5), new_generations[:, i], marker='s', label=agent_labels[i], linewidth=2)
    ax2.set_xlabel('Generation', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('NEW: Fixed-Point Normalized Bonus\n(Fixed value, scaled by generation avg)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='best')

    # ===== PANEL 3: Wealth Inequality (Gini Coefficient) =====
    ax3 = plt.subplot(2, 3, 3)

    def gini_coefficient(scores):
        """Calculate Gini coefficient (0=perfect equality, 1=perfect inequality)"""
        sorted_scores = np.sort(scores)
        n = len(sorted_scores)
        cumsum = np.cumsum(sorted_scores)
        return (2 * np.sum(np.arange(1, n + 1) * sorted_scores)) / (n * np.sum(sorted_scores)) - (n + 1) / n

    old_gini = [gini_coefficient(old_generations[i]) for i in range(5)]
    new_gini = [gini_coefficient(new_generations[i]) for i in range(5)]

    x = np.arange(5)
    width = 0.35
    ax3.bar(x - width/2, old_gini, width, label='Old (%-based)', color='#ff6b6b', alpha=0.8)
    ax3.bar(x + width/2, new_gini, width, label='New (Fixed-point)', color='#51cf66', alpha=0.8)
    ax3.set_xlabel('Generation', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Gini Coefficient (Inequality)', fontsize=11, fontweight='bold')
    ax3.set_title('Wealth Inequality Over Time\n(Lower is more equal)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # ===== PANEL 4: Bonus Amount Comparison (Gen 1 vs Gen 4) =====
    ax4 = plt.subplot(2, 3, 4)

    # Simulate bonus amounts
    test_scores = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    old_bonus = test_scores * 0.20  # 20% bonus

    # New bonus (normalized)
    gen_avg = np.mean(test_scores)
    new_bonus = np.full_like(test_scores, 3.0 * (10.0 / gen_avg))

    x_pos = np.arange(len(test_scores))
    width = 0.35
    ax4.bar(x_pos - width/2, old_bonus, width, label='Old: % of wealth', color='#ff6b6b', alpha=0.8)
    ax4.bar(x_pos + width/2, new_bonus, width, label='New: Fixed-normalized', color='#51cf66', alpha=0.8)
    ax4.set_xlabel('Agent Initial Score', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Bonus Amount (pts)', fontsize=11, fontweight='bold')
    ax4.set_title('Bonus Comparison by Wealth Level\n(Same cooperation, different wealth)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{s:.0f}' for s in test_scores])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # ===== PANEL 5: Distribution Spread =====
    ax5 = plt.subplot(2, 3, 5)

    def score_std(generations):
        return [np.std(gen) for gen in generations]

    old_std = score_std(old_generations)
    new_std = score_std(new_generations)

    x = np.arange(5)
    ax5.plot(x, old_std, marker='o', linewidth=2.5, markersize=8, label='Old (%-based)', color='#ff6b6b')
    ax5.plot(x, new_std, marker='s', linewidth=2.5, markersize=8, label='New (Fixed-point)', color='#51cf66')
    ax5.fill_between(x, old_std, new_std, alpha=0.2, color='gray')
    ax5.set_xlabel('Generation', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Standard Deviation of Scores', fontsize=11, fontweight='bold')
    ax5.set_title('Score Variance Over Time\n(Lower = more equal distribution)', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ===== PANEL 6: Problem Explanation =====
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    explanation = """
MATTHEW EFFECT - THE PROBLEM:

Agent A (Score: 30):  +20% bonus = +6 pts  →  36
Agent B (Score: 10):  +20% bonus = +2 pts  →  12

Result: Wealth gap increases (30→24) despite same behavior!

WHY THIS IS BAD:
• Richer agents grow faster
• Compound interest effect
• Strategy quality matters less than starting wealth
• Selection pressure on luck, not cooperation

THE FIX - FIXED-POINT NORMALIZATION:

Both agents: +3 pts (fixed, normalized by generation avg)

Agent A: 30 + 3 = 33
Agent B: 10 + 3 = 13

Result: Same absolute bonus, fair comparison
        Evolutionary pressure stays on strategy!
    """

    ax6.text(0.05, 0.95, explanation, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(
        'Social Incentive Normalization: Preventing the Matthew Effect in Evolution',
        fontsize=14, fontweight='bold', y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def analyze_real_data(json_file=None):
    """Analyze real simulation data if available"""
    if json_file is None:
        # Try to find a recent result file
        latest_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/latest_results"
        if os.path.exists(latest_dir):
            files = sorted([f for f in os.listdir(latest_dir) if f.endswith('.json')])
            if files:
                json_file = os.path.join(latest_dir, files[-1])

    if json_file and os.path.exists(json_file):
        print(f"\n📊 Analyzing real data from: {os.path.basename(json_file)}")
        data = load_simulation_data(json_file)

        # Extract unique agents and their progression
        agents_by_gen = {}
        for round_data in data.get('agents_data', []):
            gen = round_data['current_generation']
            agent = round_data['agent_name']

            if gen not in agents_by_gen:
                agents_by_gen[gen] = {}
            if agent not in agents_by_gen[gen]:
                agents_by_gen[gen][agent] = []

            agents_by_gen[gen][agent].append(round_data)

        print("\n📈 Generation Summary:")
        print("-" * 70)
        for gen in sorted(agents_by_gen.keys()):
            agents = agents_by_gen[gen]
            final_scores = [max(rounds, key=lambda x: x['round_number'])['resources']
                          for rounds in agents.values()]

            print(f"Generation {gen}:")
            print(f"  Agents: {len(agents)}")
            print(f"  Avg Score: {np.mean(final_scores):.2f}")
            print(f"  Std Dev: {np.std(final_scores):.2f}")
            print(f"  Min/Max: {np.min(final_scores):.2f} / {np.max(final_scores):.2f}")
            print(f"  Gini: {gini_coefficient(np.array(final_scores)):.3f}")

        return agents_by_gen

    return None


def gini_coefficient(scores):
    """Calculate Gini coefficient"""
    sorted_scores = np.sort(scores)
    n = len(sorted_scores)
    if n == 0 or np.sum(sorted_scores) == 0:
        return 0
    cumsum = np.cumsum(sorted_scores)
    return (2 * np.sum(np.arange(1, n + 1) * sorted_scores)) / (n * np.sum(sorted_scores)) - (n + 1) / n


def main():
    print("\n" + "="*70)
    print("SOCIAL INCENTIVE NORMALIZATION FIX VISUALIZATION")
    print("="*70)

    # Create main comparison figure
    fig = create_comparison_figure()

    # Save figure
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'social_incentive_fix_comparison.png'
    )
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison figure saved: {output_file}")

    # Analyze real data if available
    analyze_real_data()

    plt.show()


if __name__ == '__main__':
    main()

