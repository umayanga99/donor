#!/usr/bin/env python3
"""
Gossip (Reputation) + Forgiveness ON
"""
import os
from prisoners import PrisonersDilemmaBase
from dotenv import load_dotenv

def main():
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    print("="*80)
    print("GOSSIP (REPUTATION) + FORGIVENESS")
    print("="*80)

    game = PrisonersDilemmaBase(
        api_key=api_key,
        enable_regret=False,
        enable_gossip=True,
        enable_forgiveness=True
    )

    game.run_simulation(num_generations=3, num_agents=6)
    print("\nGossip + Forgiveness simulation complete!")

if __name__ == "__main__":
    main()
