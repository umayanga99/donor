import anthropic
import random
import os
import numpy as np
from dataclasses import dataclass, field, asdict
import datetime
import json
import time
import threading
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional
import re

print_lock = threading.Lock()

STRATEGIES = ["Tit-for-Tat", "Always Cooperate", "Always Defect", "Random", "Generous TFT", "Pavlov"]


@dataclass
class RegretMemory:
    round_number: int
    decision: str
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
    interaction_history: Dict[str, List[dict]] = field(default_factory=dict)
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
    def __init__(self, api_key: str, model: str = "claude-opus-4-5-20251101",
                 enable_regret=True, enable_gossip=True, enable_forgiveness=True,
                 seed=42, num_rounds_per_generation=5):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.enable_regret = enable_regret
        self.enable_gossip = enable_gossip
        self.enable_forgiveness = enable_forgiveness
        self.initial_endowment = 10.0

        # Standard PD Matrix: T > R > P > S (5 > 3 > 1 > 0)
        self.PAYOFFS = {
            ('COOPERATE', 'COOPERATE'): (3, 3),  # R = 3 (Reward)
            ('DEFECT', 'COOPERATE'): (5, 0),  # T = 5 (Temptation), S = 0 (Sucker)
            ('COOPERATE', 'DEFECT'): (0, 5),  # S = 0 (Sucker), T = 5 (Temptation)
            ('DEFECT', 'DEFECT'): (1, 1)  # P = 1 (Punishment)
        }

        self.regret_decay = 0.9
        self.forgiveness_decay = 0.95
        self.forgiveness_increment = 0.1
        self.gossip_spread_probability = 0.3
        self.mutation_rate = 0.1
        self.seed = seed
        self.num_rounds_per_generation = num_rounds_per_generation
        self.simulation_data = None
        self.current_round = 0

    def get_mechanism_name(self) -> str:
        mechs = []
        if self.enable_regret: mechs.append("regret")
        if self.enable_gossip: mechs.append("gossip")
        if self.enable_forgiveness: mechs.append("forgiveness")
        return "_".join(mechs) if mechs else "baseline"

    def prompt_claude(self, prompt, system_prompt):
        """Call Claude API with retry logic"""
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=1.0,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception as e:
                print(f"API error (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
        return ""

    def get_agent_decision(self, agent, partner, r_idx, gen, system_prompt):
        """Get agent's decision via LLM with context from social mechanisms"""
        agent.current_round_gossip_influenced = False

        # Gather context from social mechanisms
        gossip_ctx = self.get_gossip_context(agent, partner)
        regret_ctx = self.get_regret_context(agent)
        forgiveness_ctx = self.get_forgiveness_context(agent, partner)

        ctx = f"{gossip_ctx}{regret_ctx}{forgiveness_ctx}"

        history_text = ""
        if agent.history:
            history_text = f"Your recent history: {'; '.join(agent.history[-3:])}\n"

        prompt = (
            f"You are agent {agent.name}. Strategy: {agent.strategy}.\n"
            f"Partner: {partner.name}. Round: {r_idx + 1}.\n"
            f"{history_text}"
            f"{ctx}\n"
            f"Payoff Matrix:\n"
            f"  Both COOPERATE: (3, 3) pts\n"
            f"  You DEFECT, Partner COOPERATES: (5, 0) pts\n"
            f"  You COOPERATE, Partner DEFECTS: (0, 5) pts\n"
            f"  Both DEFECT: (1, 1) pts\n"
            f"Choose: [COOPERATE] or [DEFECT].\n"
            f"Explain your reasoning briefly, then end with: Answer: [CHOICE]"
        )

        # Try to get LLM decision with multiple attempts
        for attempt in range(5):
            raw = self.prompt_claude(prompt, system_prompt)
            if not raw:
                continue

            agent.traces.append({
                "round": r_idx + 1,
                "generation": gen,
                "partner": partner.name,
                "prompt": prompt[:200] + "...",
                "response": raw[:500] + ("..." if len(raw) > 500 else ""),
                "attempt": attempt + 1
            })

            choice, parsed = self._parse_llm_decision(raw, agent, partner)

            if choice in ["COOPERATE", "DEFECT"]:
                return choice, raw

        # Fallback to strategy-based decision
        default_choice = self._get_strategy_default_choice(agent, partner)
        agent.traces.append({
            "round": r_idx + 1,
            "generation": gen,
            "partner": partner.name,
            "note": "Parsing failed, using strategy default",
            "default_choice": default_choice
        })
        return default_choice, "Defaulted to strategy choice"

    def _parse_llm_decision(self, raw: str, agent, partner) -> Tuple[str, str]:
        """Parse LLM response with robust regex pattern"""
        # Look for explicit Answer: [COOPERATE] or Answer: [DEFECT] pattern
        match = re.search(r'Answer:\s*\[?(COOPERATE|DEFECT)\]?', raw.upper())
        if match:
            return match.group(1), raw

        # Fallback: look for the words in context
        cooperate_pos = raw.upper().rfind('COOPERATE')
        defect_pos = raw.upper().rfind('DEFECT')

        if cooperate_pos != -1 and defect_pos != -1:
            # Both found, use the last one mentioned
            if cooperate_pos > defect_pos:
                return "COOPERATE", raw
            else:
                return "DEFECT", raw
        elif cooperate_pos != -1:
            return "COOPERATE", raw
        elif defect_pos != -1:
            return "DEFECT", raw

        # Ultimate fallback: use strategy-based decision
        return self._get_strategy_default_choice(agent, partner), raw

    def _get_strategy_default_choice(self, agent, partner) -> str:
        """Implement actual strategy logic for fallback decisions"""
        if agent.strategy == "Always Cooperate":
            return "COOPERATE"

        elif agent.strategy == "Always Defect":
            return "DEFECT"

        elif agent.strategy == "Tit-for-Tat":
            if partner and partner.name in agent.interaction_history:
                history = agent.interaction_history[partner.name]
                if history:
                    return history[-1]['partner_choice']
            return "COOPERATE"

        elif agent.strategy == "Generous TFT":
            if partner and partner.name in agent.interaction_history:
                history = agent.interaction_history[partner.name]
                if history:
                    last_partner_choice = history[-1]['partner_choice']
                    if last_partner_choice == "DEFECT":
                        # 10% chance to forgive and cooperate anyway
                        return "COOPERATE" if random.random() < 0.1 else "DEFECT"
                    return "COOPERATE"
            return "COOPERATE"

        elif agent.strategy == "Pavlov":
            # Win-stay, lose-shift
            if partner and partner.name in agent.interaction_history:
                history = agent.interaction_history[partner.name]
                if history:
                    last = history[-1]
                    # If got good payoff (3 or 5), repeat choice
                    if last['my_payoff'] >= 3:
                        return last['my_choice']
                    else:
                        # Switch if got bad payoff
                        return "DEFECT" if last['my_choice'] == "COOPERATE" else "COOPERATE"
            return "COOPERATE"

        elif agent.strategy == "Random":
            return "COOPERATE" if random.random() > 0.5 else "DEFECT"

        else:
            return "COOPERATE"

    def handle_pairing(self, a, b, r_idx, gen, sys_prompt, locks):
        """Handle a single pairing interaction between two agents"""
        with locks[a.name], locks[b.name]:
            a.current_partner, b.current_partner = b, a

            # 1. Get decisions from both agents
            choice_a, just_a = self.get_agent_decision(a, b, r_idx, gen, sys_prompt)
            choice_b, just_b = self.get_agent_decision(b, a, r_idx, gen, sys_prompt)

            # 2. Apply payoff matrix
            payoff_a, payoff_b = self.PAYOFFS[(choice_a, choice_b)]

            old_resources_a = a.resources
            old_resources_b = b.resources

            a.resources += payoff_a
            b.resources += payoff_b

            # 3. Record interaction history (crucial for strategy logic)
            if b.name not in a.interaction_history:
                a.interaction_history[b.name] = []
            if a.name not in b.interaction_history:
                b.interaction_history[a.name] = []

            a.interaction_history[b.name].append({
                'my_choice': choice_a,
                'partner_choice': choice_b,
                'my_payoff': payoff_a,
                'round': r_idx + 1
            })

            b.interaction_history[a.name].append({
                'my_choice': choice_b,
                'partner_choice': choice_a,
                'my_payoff': payoff_b,
                'round': r_idx + 1
            })

            # 4. Calculate and store regret
            regret_a = self._calculate_regret(a, choice_a, choice_b, payoff_a)
            regret_b = self._calculate_regret(b, choice_b, choice_a, payoff_b)

            if self.enable_regret:
                a.regret_memories.append(
                    RegretMemory(r_idx + 1, choice_a, payoff_a, regret_a, f"vs {b.name}")
                )
                a.regret_level = regret_a

                b.regret_memories.append(
                    RegretMemory(r_idx + 1, choice_b, payoff_b, regret_b, f"vs {a.name}")
                )
                b.regret_level = regret_b

            # 5. Update forgiveness/trust levels
            self.update_forgiveness(a, b.name, choice_b == "COOPERATE", r_idx + 1)
            self.update_forgiveness(b, a.name, choice_a == "COOPERATE", r_idx + 1)

            # 6. Generate gossip about the interaction
            self._generate_gossip(a, b, choice_b, r_idx + 1)
            self._generate_gossip(b, a, choice_a, r_idx + 1)

            # 7. Update tracking variables for agent A
            a.last_donation = 1.0 if choice_a == "COOPERATE" else 0.0
            a.last_received = payoff_a
            a.justification = just_a
            a.total_donated += (1.0 if choice_a == "COOPERATE" else 0.0)
            a.total_received += payoff_a
            a.history.append(
                f"R{r_idx + 1}: {choice_a} vs {b.name}({choice_b}) → +{payoff_a}pts "
                f"[{old_resources_a:.1f}→{a.resources:.1f}]"
            )

            # 8. Update tracking variables for agent B
            b.last_donation = 1.0 if choice_b == "COOPERATE" else 0.0
            b.last_received = payoff_b
            b.justification = just_b
            b.total_donated += (1.0 if choice_b == "COOPERATE" else 0.0)
            b.total_received += payoff_b
            b.history.append(
                f"R{r_idx + 1}: {choice_b} vs {a.name}({choice_a}) → +{payoff_b}pts "
                f"[{old_resources_b:.1f}→{b.resources:.1f}]"
            )

            return f"{a.name}({choice_a}) vs {b.name}({choice_b}) → {payoff_a}/{payoff_b} pts"

    def _calculate_regret(self, agent, choice_self, choice_partner, payoff_self):
        """Calculate regret as difference between best possible and actual payoff"""
        if not self.enable_regret:
            return 0.0

        # What was the best payoff I could have gotten given partner's choice?
        if choice_partner == "COOPERATE":
            best_payoff = 5  # Could have defected for 5
        else:  # partner defected
            best_payoff = 1  # Best was mutual defect for 1

        # Regret = missed opportunity, modulated by optimism
        raw_regret = max(0, best_payoff - payoff_self)
        regret = raw_regret * (1.0 - 0.5 * agent.optimism)

        return regret

    def _generate_gossip(self, observer, partner, partner_choice, round_num):
        """Generate gossip about partner's behavior"""
        if not self.enable_gossip or random.random() > self.gossip_spread_probability:
            return

        # Generate sentiment based on partner's choice
        sentiment = 1.0 if partner_choice == "COOPERATE" else -1.0

        observer.gossip_to_share.append(GossipMessage(
            about_agent=partner.name,
            sentiment=sentiment,
            reliability=0.6 + 0.4 * observer.reputation,
            round_received=round_num,
            source=observer.name
        ))

    def spread_gossip(self, agents):
        """Spread gossip between agents"""
        if not self.enable_gossip:
            return

        for agent in agents:
            if agent.gossip_to_share:
                # Share with random subset of other agents
                other_agents = [a for a in agents if a != agent]
                num_recipients = min(2, len(other_agents))

                if num_recipients > 0:
                    recipients = random.sample(other_agents, num_recipients)

                    for recipient in recipients:
                        for gossip in agent.gossip_to_share:
                            # Reduce reliability as gossip spreads
                            recipient.gossip_received.append(GossipMessage(
                                about_agent=gossip.about_agent,
                                sentiment=gossip.sentiment,
                                reliability=gossip.reliability * 0.9,
                                # This preserves when the event actually happened, not when it was shared
                                round_received=gossip.round_received,
                                source=agent.name
                            ))

                agent.gossip_to_share.clear()

    def get_gossip_context(self, agent, partner):
        """Get gossip context about partner for decision-making"""
        if not self.enable_gossip:
            return ""

        # Get recent gossip about partner
        relevant_gossip = [g for g in agent.gossip_received if g.about_agent == partner.name]
        if not relevant_gossip:
            return ""

        # Calculate weighted average sentiment from recent gossip
        recent = relevant_gossip[-3:]  # Last 3 gossip messages
        total_weight = sum(g.reliability for g in recent)

        if total_weight == 0:
            return ""

        avg_sentiment = sum(g.sentiment * g.reliability for g in recent) / total_weight

        agent.current_round_gossip_influenced = True

        if avg_sentiment > 0.5:
            return f"Gossip: {partner.name} is known to cooperate. "
        elif avg_sentiment < -0.5:
            return f"Gossip: {partner.name} is known to defect. "
        else:
            return f"Gossip: {partner.name} has mixed behavior. "

    def get_regret_context(self, agent):
        """Get regret context for decision-making"""
        if not self.enable_regret or not agent.regret_memories:
            return ""

        if agent.regret_level > 0.5:
            return "You feel significant regret from being exploited. Be cautious. "
        elif agent.regret_level > 0.2:
            return "You have some regret from recent interactions. "

        return ""

    def get_forgiveness_context(self, agent, partner):
        """Get forgiveness/trust context for decision-making"""
        if not self.enable_forgiveness:
            return ""

        if partner.name not in agent.forgiveness_records:
            return ""

        rec = agent.forgiveness_records[partner.name]

        if rec.forgiveness_level < 0.3:
            return f"Trust for {partner.name}: {rec.forgiveness_level:.2f} (LOW - they've betrayed you). "
        elif rec.forgiveness_level > 0.7:
            return f"Trust for {partner.name}: {rec.forgiveness_level:.2f} (HIGH - reliable partner). "
        else:
            return f"Trust for {partner.name}: {rec.forgiveness_level:.2f} (MODERATE). "

    def update_forgiveness(self, agent, partner_name, was_cooperative, round_num):
        """Update trust/forgiveness level based on interaction"""
        if not self.enable_forgiveness:
            return

        # Initialize record if doesn't exist
        if partner_name not in agent.forgiveness_records:
            agent.forgiveness_records[partner_name] = ForgivenessRecord(
                agent_name=partner_name,
                offense_count=0,
                forgiveness_level=0.5,
                last_interaction_round=0
            )

        rec = agent.forgiveness_records[partner_name]

        # Apply decay based on time since last interaction
        rounds_passed = round_num - rec.last_interaction_round
        if rounds_passed > 1:
            decay_factor = self.forgiveness_decay ** (rounds_passed - 1)
            rec.offense_count *= decay_factor

        # Update based on current interaction
        if was_cooperative:
            # Trust increases with cooperation
            rec.forgiveness_level = min(1.0, rec.forgiveness_level + self.forgiveness_increment)
            # Offense count decreases slightly
            rec.offense_count = max(0, rec.offense_count - 0.5)
        else:
            # Trust decreases with defection
            rec.offense_count += 1
            trust_loss = 0.1 * (1 + 0.1 * rec.offense_count)
            rec.forgiveness_level = max(0.0, rec.forgiveness_level - trust_loss)

        rec.last_interaction_round = round_num

        # Update agent's average forgiveness level
        if agent.forgiveness_records:
            agent.forgiveness_given = sum(
                r.forgiveness_level for r in agent.forgiveness_records.values()
            ) / len(agent.forgiveness_records)

    def update_global_reputations(self, agents):
        """
        Update reputation scores based on BOTH direct trust (forgiveness)
        and indirect social proof (gossip).
        """
        for target_agent in agents:
            individual_beliefs = []

            # Calculate what every OTHER agent thinks of the target_agent
            for observer in agents:
                if observer.name == target_agent.name:
                    continue

                # --- 1. Get Direct Trust (Forgiveness) ---
                direct_score = None
                if target_agent.name in observer.forgiveness_records:
                    direct_score = observer.forgiveness_records[target_agent.name].forgiveness_level

                # --- 2. Get Indirect Trust (Gossip) ---
                gossip_score = None

                # Filter gossip the observer has received about the target
                relevant_gossip = [
                    g for g in observer.gossip_received
                    if g.about_agent == target_agent.name
                ]

                if relevant_gossip:
                    # Calculate weighted average sentiment
                    total_reliability = sum(g.reliability for g in relevant_gossip)

                    if total_reliability > 0:
                        weighted_sentiment = sum(
                            g.sentiment * g.reliability for g in relevant_gossip
                        ) / total_reliability

                        # IMPORTANT: Normalize Sentiment (-1 to 1) to Trust Scale (0 to 1)
                        # -1 (Defect) -> 0.0
                        #  0 (Neutral)-> 0.5
                        # +1 (Coop)   -> 1.0
                        gossip_score = (weighted_sentiment + 1) / 2

                # --- 3. Combine Beliefs ---
                final_belief = 0.5  # Default neutral if nothing is known

                if direct_score is not None and gossip_score is not None:
                    # If we have both, weight direct experience higher (e.g., 70% direct, 30% gossip)
                    final_belief = (0.7 * direct_score) + (0.3 * gossip_score)

                elif direct_score is not None:
                    # Only have direct experience
                    final_belief = direct_score

                elif gossip_score is not None:
                    # Only have gossip (no direct interaction yet)
                    final_belief = gossip_score

                # Store what the observer thinks of the target
                observer.reputation_beliefs[target_agent.name] = final_belief
                individual_beliefs.append(final_belief)

            # --- 4. Update Global Public Reputation ---
            # This is the "average social standing" of the agent
            if individual_beliefs:
                target_agent.reputation = sum(individual_beliefs) / len(individual_beliefs)
            else:
                target_agent.reputation = 0.5

    def initialize_agents(self, num_agents, generation):
        """Initialize agents with diverse strategies"""
        strategy_descriptions = {
            "Tit-for-Tat": "Start by cooperating, then copy opponent's last move",
            "Always Cooperate": "Always cooperate to build mutual trust",
            "Always Defect": "Always defect to maximize personal gain",
            "Random": "Choose randomly to be unpredictable",
            "Generous TFT": "Like Tit-for-Tat but occasionally forgive defections",
            "Pavlov": "Win-stay, lose-shift strategy"
        }

        agents = []
        for i in range(num_agents):
            strategy = STRATEGIES[i % len(STRATEGIES)]
            agents.append(Agent(
                name=f"{generation}_{i + 1}",
                resources=self.initial_endowment,
                strategy=strategy,
                strategy_justification=strategy_descriptions.get(strategy, ""),
                generation=generation
            ))
        return agents

    def round_robin_pairings(self, agents):
        """Generate round-robin pairings so each agent plays each other"""
        n = len(agents)
        if n < 2:
            return []

        agent_list = agents[:]
        if n % 2 == 1:
            agent_list.append(None)  # Add bye for odd number

        rounds = []
        for _ in range(len(agent_list) - 1):
            round_pairs = []
            for i in range(len(agent_list) // 2):
                a = agent_list[i]
                b = agent_list[len(agent_list) - 1 - i]
                if a is not None and b is not None:
                    round_pairs.append((a, b))
            rounds.append(round_pairs)
            # Rotate all except first
            agent_list = [agent_list[0]] + [agent_list[-1]] + agent_list[1:-1]

        return rounds

    def run_simulation(self, num_generations=3, num_agents=6):
        """Run the complete simulation"""
        random.seed(self.seed)
        np.random.seed(self.seed)

        mech_name = self.get_mechanism_name()

        sys_prompt = (
            "You are playing a Prisoner's Dilemma game. "
            "Choose COOPERATE or DEFECT to maximize your total points. "
            "Payoffs: (C,C)=3 each, (D,C)=5 for defector/0 for cooperator, (D,D)=1 each."
        )

        self.simulation_data = SimulationData({
            "mechanisms": mech_name,
            "num_agents": num_agents,
            "num_generations": num_generations,
            "num_rounds_per_generation": self.num_rounds_per_generation,
            "payoffs": {
                "(C,C)": (3, 3),
                "(D,C)": (5, 0),
                "(C,D)": (0, 5),
                "(D,D)": (1, 1)
            },
            "initial_endowment": self.initial_endowment,
            "enable_regret": self.enable_regret,
            "enable_gossip": self.enable_gossip,
            "enable_forgiveness": self.enable_forgiveness,
            "seed": self.seed
        })

        agents = self.initialize_agents(num_agents, 1)

        for gen in range(1, num_generations + 1):
            print(f"\n{'=' * 60}")
            print(f"GENERATION {gen} ({mech_name})")
            print(f"{'=' * 60}")

            # Reset resources at start of generation
            for a in agents:
                a.resources = self.initial_endowment

            # Get round-robin pairings
            all_rounds = self.round_robin_pairings(agents)

            # Run rounds
            for r_idx in range(min(self.num_rounds_per_generation, len(all_rounds))):
                self.current_round = r_idx + 1
                pairings = all_rounds[r_idx % len(all_rounds)]

                print(f"\nRound {r_idx + 1}: {len(pairings)} pairings")

                locks = {a.name: Lock() for a in agents}

                # Update reputations before round
                self.update_global_reputations(agents)

                # Process pairings in parallel
                with ThreadPoolExecutor(max_workers=min(10, len(pairings))) as exe:
                    results = list(exe.map(
                        lambda p: self.handle_pairing(p[0], p[1], r_idx, gen, sys_prompt, locks),
                        pairings
                    ))
                    for r in results:
                        print(f"  {r}")

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
                        traces=list(a.traces),
                        history=list(a.history),
                        justification=a.justification,
                        regret_level=a.regret_level,
                        forgiveness_given=a.forgiveness_given,
                        gossip_influenced=a.current_round_gossip_influenced,
                        reputation=a.reputation
                    )
                    self.simulation_data.agents_data.append(asdict(round_data))

            # Print generation summary
            print(f"\n{'=' * 80}")
            print(f"GENERATION {gen} FINAL RESULTS")
            print(f"{'=' * 80}")
            print(f"{'Agent':<12} | {'Strategy':<18} | {'Score':<8} | {'Reputation':<11} | {'Avg Trust':<10}")
            print("-" * 80)

            sorted_agents = sorted(agents, key=lambda x: x.resources, reverse=True)
            for a in sorted_agents:
                if a.forgiveness_records:
                    avg_trust = sum(
                        rec.forgiveness_level for rec in a.forgiveness_records.values()
                    ) / len(a.forgiveness_records)
                else:
                    avg_trust = 0.5

                print(f"{a.name:<12} | {a.strategy:<18} | {a.resources:<8.1f} | "
                      f"{a.reputation:<11.2f} | {avg_trust:<10.2f}")
            print("-" * 80)

            # Evolutionary Selection
            if gen < num_generations:
                # Select top 50% as survivors
                survivors = sorted(agents, key=lambda x: x.resources, reverse=True)[
                    :max(2, num_agents // 2)
                ]

                print(f"\nSurvivors: {[s.name for s in survivors]}")

                # Create offspring inheriting strategies from survivors
                new_agents = []
                for i, survivor in enumerate(survivors):
                    strategy = survivor.strategy

                    # Mutation: small chance to switch strategy
                    if random.random() < self.mutation_rate:
                        strategy = random.choice(STRATEGIES)
                        print(f"  Mutation: {survivor.name}'s offspring mutated to {strategy}")

                    new_agents.append(Agent(
                        name=f"{gen + 1}_{len(survivors) + i + 1}",
                        resources=self.initial_endowment,
                        strategy=strategy,
                        generation=gen + 1
                    ))

                # Reset survivors for next generation
                for s in survivors:
                    s.resources = self.initial_endowment
                    s.regret_memories.clear()
                    s.gossip_received.clear()
                    s.gossip_to_share.clear()
                    s.interaction_history.clear()
                    s.history.clear()
                    s.traces.clear()
                    s.forgiveness_records.clear()
                    s.reputation_beliefs.clear()
                    s.generation = gen + 1
                    # Rename for next generation
                    old_name = s.name
                    s.name = f"{gen + 1}_{survivors.index(s) + 1}"

                # Combine survivors and offspring
                agents = survivors + new_agents
                random.shuffle(agents)

                print(f"Next generation: {len(agents)} agents")

        # Save results to JSON
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/claude_pd_{mech_name}_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(self.simulation_data.to_dict(), f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"✓ Simulation complete!")
        print(f"✓ Results saved to: {filename}")
        print(f"{'=' * 60}")

        return filename

