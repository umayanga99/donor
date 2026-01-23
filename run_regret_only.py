#!/usr/bin/env python3
"""
Only Regret ON
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
    print("REGRET ONLY")
    print("="*80)

    game = PrisonersDilemmaBase(
        api_key=api_key,
        enable_regret=True,
        enable_gossip=False,
        enable_forgiveness=False
    )

    game.run_simulation(num_generations=12, num_agents=10)
    print("\nRegret-only simulation complete!")

if __name__ == "__main__":
    main()
