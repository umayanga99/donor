#!/usr/bin/env python3
"""
Baseline: All mechanisms OFF
"""
import os
from donor_game import DonorGameBase

def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    print("BASELINE: All mechanisms OFF")

    game = DonorGameBase(
        api_key=api_key,
        enable_regret=False,
        enable_gossip=False,
        enable_forgiveness=False
    )

    game.run_simulation(num_generations=3, num_agents=6)
    print("\nBaseline simulation complete!")

if __name__ == "__main__":
    main()

