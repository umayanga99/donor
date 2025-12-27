#!/usr/bin/env python3
"""
Only Gossip (Reputation) ON
"""
import os
from donor_game_old import DonorGameBase

def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    print("GOSSIP (REPUTATION) ONLY")

    game = DonorGameBase(
        api_key=api_key,
        enable_regret=False,
        enable_gossip=True,  # Reputation mechanism
        enable_forgiveness=False
    )

    game.run_simulation(num_generations=3, num_agents=12)
    print("\nGossip-only simulation complete!")

if __name__ == "__main__":
    main()

