# Donor Game Experiment Scripts

This directory contains scripts to run the donor game with different combinations of social mechanisms.

## Mechanisms

1. **Gossip (Reputation)** - Agents share information about others' behavior, building reputation beliefs
2. **Forgiveness** - Agents track forgiveness levels toward others based on past interactions
3. **Regret** - Agents experience regret about past decisions that affect future choices

## Experiment Scripts

### Individual Scripts

Each script runs one specific combination:

1. **run_baseline.py** - All mechanisms OFF (baseline)
2. **run_gossip_only.py** - Only Gossip/Reputation (1)
3. **run_forgiveness_only.py** - Only Forgiveness (2)
4. **run_regret_only.py** - Only Regret (3)
5. **run_gossip_forgiveness.py** - Gossip + Forgiveness (1+2)
6. **run_forgiveness_regret.py** - Forgiveness + Regret (2+3)
7. **run_regret_gossip.py** - Regret + Gossip (1+3)
8. **run_all_mechanisms.py** - All mechanisms ON (1+2+3)

### Master Script

**run_all_experiments.py** - Runs all 8 experiments sequentially

## Usage

### Set up API key
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Run individual experiment
```bash
python3 run_baseline.py
python3 run_gossip_only.py
python3 run_forgiveness_only.py
# ... etc
```

### Run all experiments
```bash
python3 run_all_experiments.py
```

This will run all 8 combinations sequentially and provide a summary at the end.

## Output

Results are saved to the `results/` directory with filenames indicating:
- The mechanism combination (e.g., `gossip_forgiveness`)
- Timestamp

Example: `claude_donor_gossip_forgiveness_20231206_143022.json`

## Experiment Design

This setup allows you to compare:
- **Baseline** (no mechanisms) vs individual mechanisms
- **Single mechanisms** vs pairwise combinations
- **Pairwise combinations** vs all mechanisms together

Total: 8 experiments covering all possible combinations of 3 binary mechanisms.

## Parameters

All experiments use:
- 3 generations
- 12 agents per generation
- 12 rounds per generation
- Initial endowment: 10 units
- Cooperation gain: 2x multiplier
- Claude Opus 4.5 model

