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
    actions: List[str] = field(default_factory=list)  # Added for chronologically windowed tracking
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
    current_social_adjustment: float = 0.0


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
    cooperation_rate: float = 0.0
    social_adjustment: float = 0.0


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

        self.PAYOFFS = {
            ('COOPERATE', 'COOPERATE'): (3, 3),
            ('DEFECT', 'COOPERATE'): (5, 0),
            ('COOPERATE', 'DEFECT'): (0, 5),
            ('DEFECT', 'DEFECT'): (1, 1)
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
        agent.current_round_gossip_influenced = False

        gossip_ctx = self.get_gossip_context(agent, partner)
        regret_ctx = self.get_regret_context(agent)
        forgiveness_ctx = self.get_forgiveness_context(agent, partner)

        ctx = f"{gossip_ctx}{regret_ctx}{forgiveness_ctx}"

        history_text = ""
        if agent.history:
            history_text = f"Your recent history: {'; '.join(agent.history[-3:])}\n\n"

        total_rounds = self.num_rounds_per_generation
        current_coop_rate = self.get_agent_cooperation_ratio(agent, window=5)

        prompt = (
            f"--- CURRENT STATE & TELEMETRY ---\n"
            f"Identity: You are agent {agent.name}.\n"
            f"Time Horizon: Generation {gen} | Round {r_idx + 1} out of {total_rounds}.\n"
            f"Wealth: Your current accumulated resources are {agent.resources:.1f}.\n\n"

            f"--- SOCIAL & MATCHING CONTEXT ---\n"
            f"Your Global Reputation: {agent.reputation:.2f} (Scale: 0.0 to 1.0, higher is more cooperative).\n"
            f"Current Partner: {partner.name}.\n"
            f"Partner's Reputation: {partner.reputation:.2f}.\n"
            f"Match Reason: You were paired for this round specifically because your reputation scores are similar.\n\n"

            f"--- INSTITUTIONAL TRACKING ---\n"
            f"Your Recent Cooperation Rate (last 5 rounds): {current_coop_rate * 100:.1f}%\n"
            f"Tax/Reward Rule Reminder: Immediately after this interaction, the institution will evaluate your Global Reputation.\n"
            f"If >0.70 Reputation: You will receive an instant +2.0 fixed point bonus to your wealth.\n"
            f"If <0.30 Reputation: You will suffer an instant -2.0 fixed point tax on your wealth.\n\n"

            f"--- RECENT HISTORY & MEMORY ---\n"
            f"{history_text}"
            f"{ctx}\n"

            f"--- DECISION CONSTRAINTS ---\n"
            f"Objective: Strictly maximize your own total accumulated resources to survive evolutionary selection.\n"
            f"Action: Choose [COOPERATE] or [DEFECT].\n\n"
            f"Provide a brief, mathematical and logical rationale calculating how your choice maximizes your long-term resources based on the payoff matrix, matching rules, time horizon, and immediate reputation tax thresholds. Then conclude exactly with: Answer: [CHOICE]"
        )

        for attempt in range(5):
            raw = self.prompt_claude(prompt, system_prompt)
            if not raw: continue

            agent.traces.append({
                "round": r_idx + 1, "generation": gen, "partner": partner.name,
                "prompt": prompt[:200] + "...", "response": raw[:500] + ("..." if len(raw) > 500 else ""),
                "attempt": attempt + 1
            })

            choice, parsed = self._parse_llm_decision(raw, agent, partner)
            if choice in ["COOPERATE", "DEFECT"]: return choice, raw

        default_choice = self._get_strategy_default_choice(agent, partner)
        agent.traces.append({"round": r_idx + 1, "generation": gen, "partner": partner.name,
                             "note": "Parsing failed, using strategy default", "default_choice": default_choice})
        return default_choice, "Defaulted to strategy choice"

    def _parse_llm_decision(self, raw: str, agent, partner) -> Tuple[str, str]:
        match = re.search(r'Answer:\s*\[?(COOPERATE|DEFECT)\]?', raw.upper())
        if match: return match.group(1), raw
        cooperate_pos, defect_pos = raw.upper().rfind('COOPERATE'), raw.upper().rfind('DEFECT')
        if cooperate_pos != -1 and defect_pos != -1:
            return self._get_strategy_default_choice(agent, partner), raw
        elif cooperate_pos != -1:
            return "COOPERATE", raw
        elif defect_pos != -1:
            return "DEFECT", raw
        return self._get_strategy_default_choice(agent, partner), raw

    def _get_strategy_default_choice(self, agent, partner) -> str:
        if agent.strategy == "Always Cooperate":
            return "COOPERATE"
        elif agent.strategy == "Always Defect":
            return "DEFECT"
        elif agent.strategy == "Tit-for-Tat":
            if partner and partner.name in agent.interaction_history and agent.interaction_history[partner.name]:
                return agent.interaction_history[partner.name][-1]['partner_choice']
            return "COOPERATE"
        elif agent.strategy == "Generous TFT":
            if partner and partner.name in agent.interaction_history and agent.interaction_history[partner.name]:
                if agent.interaction_history[partner.name][-1]['partner_choice'] == "DEFECT":
                    return "COOPERATE" if random.random() < 0.1 else "DEFECT"
                return "COOPERATE"
            return "COOPERATE"
        elif agent.strategy == "Pavlov":
            if partner and partner.name in agent.interaction_history and agent.interaction_history[partner.name]:
                last = agent.interaction_history[partner.name][-1]
                if last['my_payoff'] >= 3:
                    return last['my_choice']
                else:
                    return "DEFECT" if last['my_choice'] == "COOPERATE" else "COOPERATE"
            return "COOPERATE"
        elif agent.strategy == "Random":
            return "COOPERATE" if random.random() > 0.5 else "DEFECT"
        return "COOPERATE"

    def calculate_interaction_reputation_tax(self, agent):
        """Calculates a fixed penalty or reward per-interaction based on Global Reputation"""
        if agent.reputation > 0.7:
            return 2.0  # Fixed points bonus
        elif agent.reputation < 0.3:
            return -2.0  # Fixed 5-point penalty
        return 0.0

    def handle_pairing(self, a, b, r_idx, gen, sys_prompt, locks):
        with locks[a.name], locks[b.name]:
            a.current_partner, b.current_partner = b, a

            # 1. Get decisions
            choice_a, just_a = self.get_agent_decision(a, b, r_idx, gen, sys_prompt)
            choice_b, just_b = self.get_agent_decision(b, a, r_idx, gen, sys_prompt)

            # Store actions for windowed tracking
            a.actions.append(choice_a)
            b.actions.append(choice_b)

            # 2. Compute PD Payoffs
            payoff_a, payoff_b = self.PAYOFFS[(choice_a, choice_b)]
            old_resources_a, old_resources_b = a.resources, b.resources
            a.resources += payoff_a
            b.resources += payoff_b

            # 3. Apply Immediate Institutional Reputation Tax/Bonus (Layer 2)
            adj_a = self.calculate_interaction_reputation_tax(a)
            adj_b = self.calculate_interaction_reputation_tax(b)
            a.resources += adj_a
            b.resources += adj_b
            a.current_social_adjustment = adj_a
            b.current_social_adjustment = adj_b

            # 4. History tracking and Social Metrics (Regret, Forgiveness, Gossip)
            if b.name not in a.interaction_history: a.interaction_history[b.name] = []
            if a.name not in b.interaction_history: b.interaction_history[a.name] = []

            a.interaction_history[b.name].append(
                {'my_choice': choice_a, 'partner_choice': choice_b, 'my_payoff': payoff_a, 'round': r_idx + 1})
            b.interaction_history[a.name].append(
                {'my_choice': choice_b, 'partner_choice': choice_a, 'my_payoff': payoff_b, 'round': r_idx + 1})

            regret_a = self._calculate_regret(a, choice_a, choice_b, payoff_a)
            regret_b = self._calculate_regret(b, choice_b, choice_a, payoff_b)

            if self.enable_regret:
                a.regret_memories.append(RegretMemory(r_idx + 1, choice_a, payoff_a, regret_a, f"vs {b.name}"))
                a.regret_level = regret_a
                b.regret_memories.append(RegretMemory(r_idx + 1, choice_b, payoff_b, regret_b, f"vs {a.name}"))
                b.regret_level = regret_b

            self.update_forgiveness(a, b.name, choice_b == "COOPERATE", r_idx + 1)
            self.update_forgiveness(b, a.name, choice_a == "COOPERATE", r_idx + 1)

            self._generate_gossip(a, b, choice_b, r_idx + 1)
            self._generate_gossip(b, a, choice_a, r_idx + 1)

            # 5. Log final data
            a.last_donation, a.last_received, a.justification = (
                1.0 if choice_a == "COOPERATE" else 0.0), payoff_a, just_a
            a.total_donated += a.last_donation
            a.total_received += payoff_a
            a.history.append(
                f"R{r_idx + 1}: {choice_a} vs {b.name} → PD:{payoff_a} Inst:{adj_a:+.1f} [{old_resources_a:.1f}→{a.resources:.1f}]")

            b.last_donation, b.last_received, b.justification = (
                1.0 if choice_b == "COOPERATE" else 0.0), payoff_b, just_b
            b.total_donated += b.last_donation
            b.total_received += payoff_b
            b.history.append(
                f"R{r_idx + 1}: {choice_b} vs {a.name} → PD:{payoff_b} Inst:{adj_b:+.1f} [{old_resources_b:.1f}→{b.resources:.1f}]")

            return f"{a.name}({choice_a}) vs {b.name}({choice_b}) → PD: {payoff_a}/{payoff_b} | Inst: {adj_a:+.1f}/{adj_b:+.1f}"

    def _calculate_regret(self, agent, choice_self, choice_partner, payoff_self):
        if not self.enable_regret: return 0.0
        if choice_self == "COOPERATE" and choice_partner == "COOPERATE": return 0.0
        best_payoff = 5 if choice_partner == "COOPERATE" else 1
        raw_regret = max(0, best_payoff - payoff_self)
        return (raw_regret / 1.0) * (1.0 - 0.5 * agent.optimism)

    def _generate_gossip(self, observer, partner, partner_choice, round_num):
        if not self.enable_gossip or random.random() > self.gossip_spread_probability: return
        sentiment = 1.0 if partner_choice == "COOPERATE" else -1.0
        observer.gossip_to_share.append(GossipMessage(
            about_agent=partner.name, sentiment=sentiment, reliability=0.6 + 0.4 * observer.reputation,
            round_received=round_num, source=observer.name
        ))

    def spread_gossip(self, agents):
        if not self.enable_gossip: return
        for agent in agents:
            if agent.gossip_to_share:
                other_agents = [a for a in agents if a != agent]
                num_recipients = min(2, len(other_agents))
                if num_recipients > 0:
                    recipients = random.sample(other_agents, num_recipients)
                    for recipient in recipients:
                        for gossip in agent.gossip_to_share:
                            recipient.gossip_received.append(GossipMessage(
                                about_agent=gossip.about_agent, sentiment=gossip.sentiment,
                                reliability=gossip.reliability * 0.9,
                                round_received=gossip.round_received, source=agent.name
                            ))
                agent.gossip_to_share.clear()

    def get_gossip_context(self, agent, partner):
        if not self.enable_gossip: return ""
        relevant_gossip = [g for g in agent.gossip_received if g.about_agent == partner.name]
        if not relevant_gossip: return ""
        recent = relevant_gossip[-3:]
        total_weight = sum(g.reliability for g in recent)
        if total_weight == 0: return ""
        avg_sentiment = sum(g.sentiment * g.reliability for g in recent) / total_weight
        agent.current_round_gossip_influenced = True
        if avg_sentiment > 0.5:
            return f"- Gossip network reports: {partner.name} predominantly chooses COOPERATE.\n"
        elif avg_sentiment < -0.5:
            return f"- Gossip network reports: {partner.name} predominantly chooses DEFECT.\n"
        else:
            return f"- Gossip network reports: {partner.name} exhibits mixed behavior.\n"

    def get_regret_context(self, agent):
        if not self.enable_regret or not agent.regret_memories: return ""
        if agent.regret_level > 0.4:
            return "- Internal state: You recently experienced high regret after receiving a 0 payoff from being exploited.\n"
        elif agent.regret_level > 0.1:
            return "- Internal state: You recently experienced mild regret regarding your choices.\n"
        return ""

    def get_forgiveness_context(self, agent, partner):
        if not self.enable_forgiveness: return ""
        if partner.name not in agent.forgiveness_records: return ""
        rec = agent.forgiveness_records[partner.name]
        return f"- Direct experience metric: Your recorded trust level for {partner.name} is {rec.forgiveness_level:.2f} (Scale: 0.0 to 1.0).\n"

    def update_forgiveness(self, agent, partner_name, was_cooperative, round_num):
        if not self.enable_forgiveness: return
        if partner_name not in agent.forgiveness_records:
            agent.forgiveness_records[partner_name] = ForgivenessRecord(partner_name, 0, 0.5, 0)
        rec = agent.forgiveness_records[partner_name]
        rounds_passed = round_num - rec.last_interaction_round
        if rounds_passed > 1: rec.offense_count *= (self.forgiveness_decay ** (rounds_passed - 1))
        if was_cooperative:
            rec.forgiveness_level = min(1.0, rec.forgiveness_level + self.forgiveness_increment)
            rec.offense_count = max(0, rec.offense_count - 0.5)
        else:
            rec.offense_count += 1
            trust_loss = 0.1 * (1 + 0.1 * rec.offense_count)
            rec.forgiveness_level = max(0.0, rec.forgiveness_level - trust_loss)
        rec.last_interaction_round = round_num
        if agent.forgiveness_records:
            agent.forgiveness_given = sum(r.forgiveness_level for r in agent.forgiveness_records.values()) / len(
                agent.forgiveness_records)

    def update_global_reputations(self, agents):
        for target_agent in agents:
            individual_beliefs = []
            for observer in agents:
                if observer.name == target_agent.name: continue
                direct_score = observer.forgiveness_records[
                    target_agent.name].forgiveness_level if target_agent.name in observer.forgiveness_records else None
                gossip_score = None
                relevant_gossip = [g for g in observer.gossip_received if g.about_agent == target_agent.name]
                if relevant_gossip:
                    total_reliability = sum(g.reliability for g in relevant_gossip)
                    if total_reliability > 0:
                        weighted_sentiment = sum(
                            g.sentiment * g.reliability for g in relevant_gossip) / total_reliability
                        gossip_score = (weighted_sentiment + 1) / 2
                final_belief = 0.5
                if direct_score is not None and gossip_score is not None:
                    final_belief = (0.7 * direct_score) + (0.3 * gossip_score)
                elif direct_score is not None:
                    final_belief = direct_score
                elif gossip_score is not None:
                    final_belief = gossip_score
                observer.reputation_beliefs[target_agent.name] = final_belief
                individual_beliefs.append(final_belief)
            target_agent.reputation = sum(individual_beliefs) / len(individual_beliefs) if individual_beliefs else 0.5

    def assortative_reputation_pairings(self, agents):
        if len(agents) < 2: return []
        sorted_agents = sorted(agents, key=lambda a: a.reputation, reverse=True)
        pairings = []
        for i in range(0, len(sorted_agents) - 1, 2):
            pairings.append((sorted_agents[i], sorted_agents[i + 1]))
        return pairings

    def get_agent_cooperation_ratio(self, agent, window=None):
        """Calculates cooperation ratio, optionally using a rolling window of recent moves"""
        if window and agent.actions:
            recent_moves = agent.actions[-window:]
            return recent_moves.count("COOPERATE") / len(recent_moves) if recent_moves else 0.0

        total_moves = len(agent.history)
        return agent.total_donated / total_moves if total_moves > 0 else 0.0

    def initialize_agents(self, num_agents, generation):
        agents = []
        for i in range(num_agents):
            strategy = STRATEGIES[i % len(STRATEGIES)]
            agents.append(Agent(name=f"{generation}_{i + 1}", resources=self.initial_endowment, strategy=strategy,
                                generation=generation))
        return agents

    def run_simulation(self, num_generations=3, num_agents=6):
        random.seed(self.seed)
        np.random.seed(self.seed)
        mech_name = self.get_mechanism_name()

        # UNBIASED SYSTEM PROMPT Defining Rules, Payoffs, and Consequences
        sys_prompt = (
            "You are an autonomous agent in a multi-generation evolutionary simulation based on the Iterated Prisoner's Dilemma.\n"
            "Your sole objective is to maximize your own total accumulated resources. At the end of the generation, the agents with the highest resources survive; the bottom half are eliminated.\n\n"
            "1. Payoff Matrix (Per Round):\n"
            "   - Both choose COOPERATE: You gain 3, Partner gains 3.\n"
            "   - You choose DEFECT, Partner chooses COOPERATE: You gain 5, Partner gains 0.\n"
            "   - You choose COOPERATE, Partner chooses DEFECT: You gain 0, Partner gains 5.\n"
            "   - Both choose DEFECT: You gain 1, Partner gains 1.\n\n"
            "2. Social Consequences & Assortative Matching Rule:\n"
            "   - When you choose COOPERATE, your partner's trust increases and positive gossip about you spreads.\n"
            "   - When you choose DEFECT, your partner's trust decreases and negative gossip about you spreads.\n"
            "   - This gossip forms your 'Global Reputation'.\n"
            "   - IMPORTANT: Before every round, the environment sorts all agents by their Global Reputation. You are exclusively paired with an agent who has a similar reputation to yours.\n"
            "   - Therefore, accumulating positive reputation pairs you with other cooperators. Accumulating negative reputation traps you in pairings with other defectors.\n\n"
            "3. Institutional Tax & Reward (Per Interaction):\n"
            "   - Immediately after every interaction, agents with a Global Reputation above 0.70 receive a fixed +2.0 point bonus to their resources.\n"
            "   - Agents with a Global Reputation below 0.30 suffer a fixed -2.0 point penalty to their resources.\n\n"
            "Evaluate your state and make the choice that maximizes your personal survival fitness."
        )

        self.simulation_data = SimulationData({
            "mechanisms": mech_name,
            "num_agents": num_agents,
            "num_generations": num_generations,
            "num_rounds_per_generation": self.num_rounds_per_generation,
            "payoffs": {"(C,C)": (3, 3), "(D,C)": (5, 0), "(C,D)": (0, 5), "(D,D)": (1, 1)},
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

            for a in agents: a.resources = self.initial_endowment

            for r_idx in range(self.num_rounds_per_generation):
                self.current_round = r_idx + 1

                # Update global reputations BEFORE the interaction
                self.update_global_reputations(agents)
                pairings = self.assortative_reputation_pairings(agents)

                print(f"\nRound {r_idx + 1}: {len(pairings)} pairings")
                locks = {a.name: Lock() for a in agents}

                with ThreadPoolExecutor(max_workers=min(10, max(1, len(pairings)))) as exe:
                    results = list(
                        exe.map(lambda p: self.handle_pairing(p[0], p[1], r_idx, gen, sys_prompt, locks), pairings))
                    for r in results: print(f"  {r}")

                self.spread_gossip(agents)

                for a in agents:
                    paired_with = a.current_partner.name if a.current_partner else "None"
                    round_data = AgentRoundData(
                        agent_name=a.name, round_number=r_idx + 1, game_number=gen, paired_with=paired_with,
                        current_generation=gen, resources=a.resources, donated=a.last_donation,
                        received=a.last_received,
                        strategy=a.strategy, strategy_justification=a.strategy_justification, traces=list(a.traces),
                        history=list(a.history), justification=a.justification, regret_level=a.regret_level,
                        forgiveness_given=a.forgiveness_given, gossip_influenced=a.current_round_gossip_influenced,
                        reputation=a.reputation, cooperation_rate=self.get_agent_cooperation_ratio(a, window=5),
                        social_adjustment=a.current_social_adjustment
                    )
                    self.simulation_data.agents_data.append(asdict(round_data))

            print(f"\n{'=' * 90}")
            print(f"GENERATION {gen} FINAL RESULTS")
            print(f"{'=' * 90}")
            print(
                f"{'Agent':<12} | {'Strategy':<18} | {'Score':<8} | {'Coop Rate':<9} | {'Reputation':<11} | {'Avg Trust':<10}")
            print("-" * 90)

            # Evolutionary Selection purely by accumulated resources
            sorted_agents = sorted(agents, key=lambda x: x.resources, reverse=True)
            for a in sorted_agents:
                avg_trust = sum(rec.forgiveness_level for rec in a.forgiveness_records.values()) / len(
                    a.forgiveness_records) if a.forgiveness_records else 0.5
                coop_rate = self.get_agent_cooperation_ratio(a, window=5)
                print(
                    f"{a.name:<12} | {a.strategy:<18} | {a.resources:<8.1f} | {coop_rate:<9.2f} | {a.reputation:<11.2f} | {avg_trust:<10.2f}")
            print("-" * 90)

            if gen < num_generations:
                survivors = sorted_agents[:max(2, num_agents // 2)]
                print(f"\nSurvivors: {[s.name for s in survivors]}")

                new_agents = []
                for i, survivor in enumerate(survivors):
                    strategy = survivor.strategy
                    if random.random() < self.mutation_rate:
                        strategy = random.choice(STRATEGIES)
                        print(f"  Mutation: {survivor.name}'s offspring mutated to {strategy}")
                    new_agents.append(
                        Agent(name=f"{gen + 1}_{len(survivors) + i + 1}", resources=self.initial_endowment,
                              strategy=strategy, generation=gen + 1))

                for s in survivors:
                    s.resources = self.initial_endowment
                    s.regret_memories.clear()
                    s.gossip_received.clear()
                    s.gossip_to_share.clear()
                    s.interaction_history.clear()
                    s.history.clear()
                    s.actions.clear()  # Clear chronological actions for new generation
                    s.traces.clear()
                    s.forgiveness_records.clear()
                    s.reputation_beliefs.clear()
                    s.generation = gen + 1
                    s.name = f"{gen + 1}_{survivors.index(s) + 1}"

                agents = survivors + new_agents
                random.shuffle(agents)

        os.makedirs("latest_results", exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"latest_results/claude_pd_{mech_name}_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(self.simulation_data.to_dict(), f, indent=2)

        print(f"\n{'=' * 60}\n✓ Simulation complete!\n✓ Results saved to: {filename}\n{'=' * 60}")
        return filename