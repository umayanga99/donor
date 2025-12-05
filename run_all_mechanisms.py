#!/usr/bin/env python3
"""
All mechanisms ON: Gossip (Reputation) + Forgiveness + Regret
"""
import os
from donor_game import DonorGameBase

def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    print("ALL MECHANISMS: GOSSIP (REPUTATION) + FORGIVENESS + REGRET")


    game = DonorGameBase(
        api_key=api_key,
        enable_regret=True,  # Regret mechanism
        enable_gossip=True,  # Reputation mechanism
        enable_forgiveness=True  # Forgiveness mechanism
    )

    game.run_simulation(num_generations=3, num_agents=12)
    print("\nAll mechanisms simulation complete!")

if __name__ == "__main__":
    main()

