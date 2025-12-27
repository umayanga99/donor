import anthropic
import random
import os
import numpy as np
from dataclasses import dataclass, field, asdict
import datetime
import json
import re
import time
import threading
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional

print_lock = threading.Lock()

STRATEGIES = ["Tit-for-Tat", "Always Cooperate", "Always Defect", "Random", "Generous TFT", "Pavlov"]


@dataclass
class RegretMemory:
    round_number: int
    decision: float
    outcome_score: float
    regret_level: float
    context: str


@dataclass
class ForgivenessRecord:
    agent_name: str
    offense_count: int
    forgiveness_level: float
    last_interaction_round: int


@dataclass
class GossipMessage:
    about_agent: str
    sentiment: float
    reliability: float
    round_received: int
    source: str


@dataclass
class Agent:
    name: str
    resources: float
    reputation: float = 0.5
    total_donated: float = 0.0
    total_received: float = 0.0
    history: list = field(default_factory=list)
    strategy: str = ""
    strategy_justification: str = ""
    traces: list = field(default_factory=list)
    total_final_score: float = 0.0
    regret_memories: List[RegretMemory] = field(default_factory=list)
    forgiveness_records: Dict[str, ForgivenessRecord] = field(default_factory=dict)
    gossip_received: List[GossipMessage] = field(default_factory=list)
    gossip_to_share: List[GossipMessage] = field(default_factory=list)
    reputation_beliefs: Dict[str, float] = field(default_factory=dict)
    current_round_gossip_influenced: bool = False
    optimism: float = 0.5
    current_partner: Optional['Agent'] = None
    last_donation: float = 0.0
    last_received: float = 0.0
    justification: str = ""
    regret_level: float = 0.0
    forgiveness_given: float = 0.0
    generation: int = 1


@dataclass
class SimulationData:
    hyperparameters: dict
    agents_data: list = field(default_factory=list)

    def to_dict(self):
        return {'hyperparameters': self.hyperparameters, 'agents_data': self.agents_data}


@dataclass
class AgentRoundData:
    agent_name: str
    round_number: int
    game_number: int
    paired_with: str
    current_generation: int
    resources: float
    donated: float
    received: float
    strategy: str
    strategy_justification: str
    traces: list
    history: list
    justification: str = ""
    regret_level: float = 0.0
    forgiveness_given: float = 0.0
    gossip_influenced: bool = False
    reputation: float = 0.5


class PrisonersDilemmaBase:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", enable_regret=False,
                 enable_gossip=False, enable_forgiveness=False, seed=42,
                 num_rounds_per_generation: int = 5):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.enable_regret = enable_regret
        self.enable_gossip = enable_gossip
        self.enable_forgiveness = enable_forgiveness
        self.initial_endowment = 10.0
        self.cooperation_gain = 2.0
        self.regret_decay = 0.9
        self.forgiveness_decay = 0.95
        self.forgiveness_increment = 0.1
        self.reputation_bias_strength = 0.4
        self.forgiveness_bias_strength = 0.4
        self.regret_bias_strength = 0.4
        self.gossip_spread_probability = 0.3
        self.mutation_rate = 0.1
        self.seed = seed
        self.num_rounds_per_generation = num_rounds_per_generation
        self.simulation_data = None

    def get_mechanism_name(self) -> str:
        mechs = []
        if self.enable_regret: mechs.append("regret")
        if self.enable_gossip: mechs.append("gossip")
        if self.enable_forgiveness: mechs.append("forgiveness")
        return "_".join(mechs) if mechs else "baseline"

    def prompt_claude(self, prompt, system_prompt):
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model, max_tokens=1024, temperature=1.0,
                    system=system_prompt, messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception as e:
                print(f"API error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
        return ""

    def update_global_reputations(self, agents):
        for agent in agents:
            beliefs = [oa.reputation_beliefs[agent.name] for oa in agents if
                       oa != agent and agent.name in oa.reputation_beliefs]
            agent.reputation = 0.5 + 0.5 * np.tanh(sum(beliefs) / len(beliefs)) if beliefs else 0.5

    def get_agent_decision(self, agent, partner, r_idx, gen, system_prompt):
        agent.current_round_gossip_influenced = False
        ctx = f"{self.get_gossip_context(agent, partner)}{self.get_regret_context(agent)}{self.get_forgiveness_context(agent, partner)}"

        history_text = ""
        if agent.history:
            history_text = f"Your recent history: {'; '.join(agent.history[-3:])}\n"

        prompt = (
            f"You are agent {agent.name} playing Prisoner's Dilemma.\n"
            f"Strategy: {agent.strategy}\n"
            f"Generation {gen}, Round {r_idx + 1}.\n"
            f"Your resources: {agent.resources:.1f}. Partner: {partner.name}.\n"
            f"{history_text}"
            f"{ctx}"
            f"Choose how much to contribute (0 to {agent.resources:.1f}).\n"
            f"You lose what you contribute, but gain {self.cooperation_gain}x what your partner contributes.\n"
            f"Explain your reasoning briefly, then end with: Answer: [number]"
        )

        for attempt in range(5):
            raw = self.prompt_claude(prompt, system_prompt)
            if not raw:
                continue

            # Add trace of LLM response
            agent.traces.append({
                "round": r_idx + 1,
                "generation": gen,
                "partner": partner.name,
                "prompt": prompt[:200] + "...",
                "response": raw[:500] + ("..." if len(raw) > 500 else ""),
                "attempt": attempt + 1
            })

            # Try multiple parsing strategies
            # Strategy 1: Look for "Answer:" followed by number
            if "Answer:" in raw:
                ans_part = raw.split("Answer:")[-1]
                numbers = re.findall(r'[\d.]+', ans_part[:50])
                if numbers:
                    try:
                        val = float(numbers[0])
                        return min(val, agent.resources), raw
                    except ValueError:
                        pass

            # Strategy 2: Look for any number at the end
            numbers = re.findall(r'[\d.]+', raw[-100:])
            if numbers:
                try:
                    val = float(numbers[-1])
                    if 0 <= val <= agent.resources:
                        return val, raw
                except ValueError:
                    pass

            # Strategy 3: Look for pattern like "I'll contribute X" or "donate X"
            match = re.search(r'(?:contribute|donate|give|offer)\s*:?\s*([\d.]+)', raw, re.IGNORECASE)
            if match:
                try:
                    val = float(match.group(1))
                    return min(val, agent.resources), raw
                except ValueError:
                    pass

        # Default based on strategy if parsing fails
        default_val = self._get_strategy_default(agent)
        default_justification = self._get_strategy_justification(agent, partner, default_val)
        agent.traces.append({
            "round": r_idx + 1,
            "generation": gen,
            "partner": partner.name,
            "note": "Parsing failed, using strategy default",
            "default_value": default_val,
            "justification": default_justification
        })
        return default_val, default_justification

    def _get_strategy_justification(self, agent, partner, contribution: float) -> str:
        """Generate a justification for the strategy-based default decision"""
        strategy_reasons = {
            "Always Cooperate": f"Following my cooperative strategy, I contribute {contribution:.1f} to encourage mutual benefit with {partner.name}.",
            "Always Defect": f"Following my defection strategy, I contribute {contribution:.1f} to maximize my personal gain.",
            "Random": f"Following my random strategy, I chose to contribute {contribution:.1f} unpredictably.",
            "Tit-for-Tat": f"Following Tit-for-Tat, I cooperate on the first move by contributing {contribution:.1f}.",
            "Generous TFT": f"Following Generous TFT, I start cooperatively by contributing {contribution:.1f}.",
            "Pavlov": f"Following Pavlov strategy, I contribute {contribution:.1f} based on past outcomes."
        }
        return strategy_reasons.get(agent.strategy, f"Applied {agent.strategy} strategy: contributing {contribution:.1f}")

    def _get_strategy_default(self, agent) -> float:
        """Get default contribution based on strategy when parsing fails"""
        if agent.strategy == "Always Cooperate":
            return agent.resources * 0.5
        elif agent.strategy == "Always Defect":
            return 0.0
        elif agent.strategy == "Random":
            return random.uniform(0, agent.resources)
        elif agent.strategy in ["Tit-for-Tat", "Generous TFT"]:
            # Cooperate by default (first move)
            return agent.resources * 0.4
        elif agent.strategy == "Pavlov":
            return agent.resources * 0.5
        else:
            return agent.resources * 0.3

    def apply_behavioral_biases(self, response, donor, recipient):
        biased = response
        if self.enable_gossip:
            biased *= (1 + self.reputation_bias_strength * (recipient.reputation - 0.5))
        if self.enable_forgiveness and recipient.name in donor.forgiveness_records:
            biased *= (1 + self.forgiveness_bias_strength * (
                        donor.forgiveness_records[recipient.name].forgiveness_level - 0.5))
        if self.enable_regret and donor.regret_level > 0:
            biased *= (1 - self.regret_bias_strength * donor.regret_level)
        return max(0.0, min(biased, donor.resources))

    def handle_pairing(self, a, b, r_idx, gen, sys_prompt, locks):
        with locks[a.name], locks[b.name]:
            a.current_partner, b.current_partner = b, a
            dec_a, just_a = self.get_agent_decision(a, b, r_idx, gen, sys_prompt)
            dec_b, just_b = self.get_agent_decision(b, a, r_idx, gen, sys_prompt)

            f_a = self.apply_behavioral_biases(dec_a, a, b)
            f_b = self.apply_behavioral_biases(dec_b, b, a)

            # Update resources
            old_a, old_b = a.resources, b.resources
            a.resources = (a.resources - f_a) + (self.cooperation_gain * f_b)
            b.resources = (b.resources - f_b) + (self.cooperation_gain * f_a)

            # Store donation info
            a.last_donation, a.last_received, a.justification = f_a, f_b, just_a
            b.last_donation, b.last_received, b.justification = f_b, f_a, just_b

            # Update totals
            a.total_donated += f_a
            a.total_received += f_b
            b.total_donated += f_b
            b.total_received += f_a

            # Update history
            a.history.append(f"R{r_idx+1}: Gave {f_a:.1f} to {b.name}, Got {f_b:.1f}, Resources: {old_a:.1f}→{a.resources:.1f}")
            b.history.append(f"R{r_idx+1}: Gave {f_b:.1f} to {a.name}, Got {f_a:.1f}, Resources: {old_b:.1f}→{b.resources:.1f}")

            # Update Social Mechanisms
            self.calculate_regret(a, f_a, a.resources, r_idx + 1, f"PD vs {b.name}")
            self.calculate_regret(b, f_b, b.resources, r_idx + 1, f"PD vs {a.name}")
            self.update_forgiveness(a, b.name, (f_b / self.initial_endowment) > 0.3, r_idx + 1)
            self.update_forgiveness(b, a.name, (f_a / self.initial_endowment) > 0.3, r_idx + 1)

            # Generate gossip
            self._generate_gossip(a, b, f_b, r_idx + 1)
            self._generate_gossip(b, a, f_a, r_idx + 1)

            return f"{a.name}({f_a:.1f}) <-> {b.name}({f_b:.1f})"

    def _generate_gossip(self, observer, observed, contribution, round_num):
        """Generate gossip about an observed agent"""
        if not self.enable_gossip or random.random() > self.gossip_spread_probability:
            return
        sentiment = 2 * (contribution / self.initial_endowment) - 1  # -1 to 1
        gossip = GossipMessage(
            about_agent=observed.name,
            sentiment=sentiment,
            reliability=0.6 + 0.4 * observer.reputation,
            round_received=round_num,
            source=observer.name
        )
        observer.gossip_to_share.append(gossip)

    def spread_gossip(self, agents: List[Agent]):
        """Spread gossip between agents"""
        if not self.enable_gossip:
            return
        for agent in agents:
            if agent.gossip_to_share:
                # Pick random recipients (excluding self)
                other_agents = [a for a in agents if a != agent]
                recipients = random.sample(other_agents, min(3, len(other_agents)))
                for recipient in recipients:
                    for g in agent.gossip_to_share:
                        # Reduce reliability when gossip spreads
                        recipient.gossip_received.append(
                            GossipMessage(g.about_agent, g.sentiment, g.reliability * 0.9,
                                         g.round_received, agent.name)
                        )
                agent.gossip_to_share.clear()

    # --- SOCIAL MECHANISM LOGIC ---
    def calculate_regret(self, agent, decision, outcome, round_num, context):
        if not self.enable_regret:
            return 0.0
        optimal = agent.resources * 0.5
        opp_regret = min(1.0, max(0.0, abs(decision - optimal) / max(optimal, 0.01)))
        out_regret = min(1.0, max(0.0, (self.initial_endowment - outcome) / self.initial_endowment))
        regret = (0.7 * out_regret + 0.3 * opp_regret) * (1.0 - 0.5 * agent.optimism)
        agent.regret_memories.append(RegretMemory(round_num, decision, outcome, regret, context))
        agent.regret_level = regret
        return regret

    def update_forgiveness(self, agent, other_name, was_pos, round_num):
        if not self.enable_forgiveness:
            return
        if other_name not in agent.forgiveness_records:
            agent.forgiveness_records[other_name] = ForgivenessRecord(other_name, 0, 1.0, round_num)
        rec = agent.forgiveness_records[other_name]

        # Apply decay based on time since last interaction
        time_factor = self.forgiveness_decay ** max(0, round_num - rec.last_interaction_round - 1)
        rec.offense_count *= time_factor

        if not was_pos:
            rec.offense_count += 1
            rec.forgiveness_level = max(0.0, rec.forgiveness_level - 0.1 * (1 + 0.1 * rec.offense_count))
        else:
            rec.forgiveness_level = min(1.0, rec.forgiveness_level + self.forgiveness_increment)

        rec.last_interaction_round = round_num

        # Update average forgiveness_given
        if agent.forgiveness_records:
            agent.forgiveness_given = sum(r.forgiveness_level for r in agent.forgiveness_records.values()) / len(
                agent.forgiveness_records)

    def get_gossip_context(self, a, b):
        if not self.enable_gossip:
            return ""
        gossip_about_b = [g for g in a.gossip_received if g.about_agent == b.name]
        if not gossip_about_b:
            return ""
        a.current_round_gossip_influenced = True
        avg_sentiment = sum(g.sentiment * g.reliability for g in gossip_about_b[-3:]) / len(gossip_about_b[-3:])
        if avg_sentiment > 0.3:
            return f"You've heard {b.name} is generally cooperative. "
        elif avg_sentiment < -0.3:
            return f"You've heard {b.name} tends to defect. "
        return f"You've heard mixed things about {b.name}. "

    def get_regret_context(self, a):
        if not self.enable_regret or not a.regret_memories:
            return ""
        recent = [r for r in a.regret_memories[-3:] if r.regret_level > 0.3]
        if not recent:
            return ""
        return f"Recent regret level: {a.regret_level:.2f}. Consider adjusting your strategy. "

    def get_forgiveness_context(self, a, b):
        if not self.enable_forgiveness:
            return ""
        if b.name not in a.forgiveness_records:
            return ""
        rec = a.forgiveness_records[b.name]
        if rec.forgiveness_level < 0.3:
            return f"You have low trust in {b.name} (forgiveness: {rec.forgiveness_level:.2f}). "
        elif rec.forgiveness_level > 0.7:
            return f"You trust {b.name} (forgiveness: {rec.forgiveness_level:.2f}). "
        return ""

    def initialize_agents(self, num_agents: int, generation: int) -> List[Agent]:
        """Initialize agents with diverse strategies"""
        strategy_descriptions = {
            "Tit-for-Tat": "Start by cooperating, then copy opponent's last move",
            "Always Cooperate": "Always contribute generously to build mutual trust",
            "Always Defect": "Never contribute to maximize personal gain",
            "Random": "Choose contributions randomly to be unpredictable",
            "Generous TFT": "Like Tit-for-Tat but occasionally forgive defections",
            "Pavlov": "Cooperate if both made same choice last round, otherwise defect"
        }
        agents = []
        for i in range(num_agents):
            strategy = STRATEGIES[i % len(STRATEGIES)]
            justification = strategy_descriptions.get(strategy, f"Following {strategy} strategy")
            agents.append(Agent(
                name=f"{generation}_{i + 1}",
                resources=self.initial_endowment,
                strategy=strategy,
                strategy_justification=justification,
                generation=generation
            ))
        return agents

    def round_robin_pairings(self, agents: List[Agent]) -> List[List[Tuple[Agent, Agent]]]:
        """Generate round-robin pairings so each agent plays each other agent"""
        n = len(agents)
        if n < 2:
            return []

        # Make a copy and add a dummy if odd number
        agent_list = agents[:]
        if n % 2 == 1:
            agent_list.append(None)  # Bye
            n += 1

        rounds = []
        for _ in range(n - 1):
            round_pairs = []
            for i in range(n // 2):
                a = agent_list[i]
                b = agent_list[n - 1 - i]
                if a is not None and b is not None:
                    round_pairs.append((a, b))
            rounds.append(round_pairs)
            # Rotate all except first
            agent_list = [agent_list[0]] + [agent_list[-1]] + agent_list[1:-1]

        return rounds

    def run_simulation(self, num_generations: int = 3, num_agents: int = 6):
        random.seed(self.seed)
        np.random.seed(self.seed)

        mech_name = self.get_mechanism_name()
        sys_prompt = (
            f"You are playing a simultaneous Prisoner's Dilemma game. "
            f"If you contribute X and your partner contributes Y, you lose X but gain {self.cooperation_gain}*Y. "
            f"Maximize your total resources over multiple rounds."
        )

        self.simulation_data = SimulationData({
            "mechanisms": mech_name,
            "num_agents": num_agents,
            "num_generations": num_generations,
            "num_rounds_per_generation": self.num_rounds_per_generation,
            "cooperation_gain": self.cooperation_gain,
            "initial_endowment": self.initial_endowment,
            "enable_regret": self.enable_regret,
            "enable_gossip": self.enable_gossip,
            "enable_forgiveness": self.enable_forgiveness,
            "seed": self.seed
        })

        agents = self.initialize_agents(num_agents, 1)

        for gen in range(1, num_generations + 1):
            print(f"\n=== GENERATION {gen} ({mech_name}) ===")

            # Reset resources at start of generation
            for a in agents:
                a.resources = self.initial_endowment

            # Get round-robin pairings
            all_rounds = self.round_robin_pairings(agents)

            # Run multiple rounds per generation
            for r_idx in range(min(self.num_rounds_per_generation, len(all_rounds))):
                pairings = all_rounds[r_idx % len(all_rounds)]
                print(f"  Round {r_idx + 1}: {len(pairings)} pairings")

                locks = {a.name: Lock() for a in agents}

                # Update reputations before round
                self.update_global_reputations(agents)

                # Process pairings
                with ThreadPoolExecutor(max_workers=min(10, len(pairings))) as exe:
                    results = list(exe.map(
                        lambda p: self.handle_pairing(p[0], p[1], r_idx, gen, sys_prompt, locks),
                        pairings
                    ))
                    for r in results:
                        print(f"    {r}")

                # Spread gossip after round
                self.spread_gossip(agents)

                # Collect data after each round
                for a in agents:
                    paired_with = a.current_partner.name if a.current_partner else "None"
                    round_data = AgentRoundData(
                        agent_name=a.name,
                        round_number=r_idx + 1,
                        game_number=gen,
                        paired_with=paired_with,
                        current_generation=gen,
                        resources=a.resources,
                        donated=a.last_donation,
                        received=a.last_received,
                        strategy=a.strategy,
                        strategy_justification=a.strategy_justification,
                        traces=list(a.traces),  # Copy current traces
                        history=list(a.history),  # Copy current history
                        justification=a.justification,
                        regret_level=a.regret_level,
                        forgiveness_given=a.forgiveness_given,
                        gossip_influenced=a.current_round_gossip_influenced,
                        reputation=a.reputation
                    )
                    self.simulation_data.agents_data.append(asdict(round_data))

            # Print generation summary
            print(f"\n  Generation {gen} Summary:")
            for a in sorted(agents, key=lambda x: x.resources, reverse=True):
                print(f"    {a.name} ({a.strategy}): {a.resources:.1f} resources")

            # Evolutionary Selection
            survivors = sorted(agents, key=lambda x: x.resources, reverse=True)[:max(2, len(agents) // 2)]

            if gen < num_generations:
                # Reset survivors and create offspring
                for s in survivors:
                    s.resources = self.initial_endowment
                    s.total_final_score = s.resources

                # Create new agents inheriting strategies from survivors
                strategy_descriptions = {
                    "Tit-for-Tat": "Start by cooperating, then copy opponent's last move",
                    "Always Cooperate": "Always contribute generously to build mutual trust",
                    "Always Defect": "Never contribute to maximize personal gain",
                    "Random": "Choose contributions randomly to be unpredictable",
                    "Generous TFT": "Like Tit-for-Tat but occasionally forgive defections",
                    "Pavlov": "Cooperate if both made same choice last round, otherwise defect"
                }
                new_agents = []
                for i, survivor in enumerate(survivors):
                    strategy = survivor.strategy
                    # Mutation
                    if random.random() < self.mutation_rate:
                        strategy = random.choice(STRATEGIES)
                    justification = strategy_descriptions.get(strategy, f"Following {strategy} strategy")
                    new_agents.append(Agent(
                        name=f"{gen + 1}_{i + 1}",
                        resources=self.initial_endowment,
                        strategy=strategy,
                        strategy_justification=justification,
                        generation=gen + 1
                    ))

                agents = survivors + new_agents
                random.shuffle(agents)

        # Save Data
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/claude_pd_{mech_name}_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(self.simulation_data.to_dict(), f, indent=2)

        print(f"\n✓ Results saved to: {filename}")
        return filename
