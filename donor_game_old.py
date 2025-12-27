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


@dataclass
class RegretMemory:
    """Stores regret information about past decisions"""
    round_number: int
    decision: float
    outcome_score: float
    regret_level: float
    context: str


@dataclass
class ForgivenessRecord:
    """Tracks forgiveness towards other agents"""
    agent_name: str
    offense_count: int
    forgiveness_level: float
    last_interaction_round: int


@dataclass
class GossipMessage:
    """Represents gossip about an agent"""
    about_agent: str
    sentiment: float  # Changed from message to sentiment [-1, 1]
    reliability: float
    round_received: int
    source: str


@dataclass
class Agent:
    name: str
    resources: float
    reputation: float = 0.0
    total_donated: float = 0.0
    total_received: float = 0.0
    history: list = field(default_factory=list)
    strategy: str = ""
    strategy_justification: str = ""
    traces: list = field(default_factory=list)
    total_final_score: float = 0.0
    old_traces: list = field(default_factory=list)

    # Mechanism fields
    regret_memories: List[RegretMemory] = field(default_factory=list)
    forgiveness_records: Dict[str, ForgivenessRecord] = field(default_factory=dict)
    gossip_received: List[GossipMessage] = field(default_factory=list)
    gossip_to_share: List[GossipMessage] = field(default_factory=list)
    reputation_beliefs: Dict[str, float] = field(default_factory=dict)  # Aggregated belief scores

    # Current round mechanism influence flags
    current_round_gossip_influenced: bool = False
    optimism: float = 0.5  # Optimism level for regret dampening (0=pessimistic, 1=very optimistic)
    current_partner: Optional['Agent'] = None
    last_donation: float = 0.0
    last_received: float = 0.0
    is_donor: bool = False
    justification: str = ""
    regret_level: float = 0.0
    forgiveness_given: float = 0.0
    generation: int = 1


@dataclass
class SimulationData:
    hyperparameters: dict
    agents_data: list = field(default_factory=list)

    def to_dict(self):
        return {
            'hyperparameters': self.hyperparameters,
            'agents_data': self.agents_data
        }


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
    is_donor: bool
    traces: list
    history: list
    justification: str = ""
    regret_level: float = 0.0
    forgiveness_given: float = 0.0
    gossip_influenced: bool = False
    reputation: float = 0.5  # Add reputation tracking


class DonorGameBase:
    """Base donor game with toggleable mechanisms"""

    def __init__(
            self,
            api_key: str,
            model: str = "claude-opus-4-5-20251101",
            enable_regret: bool = False,
            enable_gossip: bool = False,
            enable_forgiveness: bool = False
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Mechanism toggles
        self.enable_regret = enable_regret
        self.enable_gossip = enable_gossip
        self.enable_forgiveness = enable_forgiveness
        self.enable_reputation_bias = True

        # Game parameters
        self.cooperation_gain = 2
        self.initial_endowment = 10
        self.number_of_rounds = 12

        # Mechanism parameters
        self.regret_decay = 0.9
        self.forgiveness_threshold = 0.6
        self.gossip_spread_probability = 0.3
        self.forgiveness_increment = 0.1
        self.forgiveness_decay = 0.95

        # Optimized regret parameters
        self.optimal_ratio = 0.5
        self.regret_outcome_weight = 0.7
        self.regret_opportunity_weight = 0.3
        self.max_regret_memories = 10

        # Reputation bias parameters (always active)
        self.reputation_bias_strength = 0.4
        self.forgiveness_bias_strength = 0.4
        self.regret_bias_strength = 0.4

        # Tracking
        self.all_agents = []
        self.all_average_final_resources = []
        self.simulation_data = None

    def get_mechanism_name(self) -> str:
        """Get descriptive name for active mechanisms"""
        mechanisms = []
        if self.enable_regret:
            mechanisms.append("regret")
        if self.enable_gossip:
            mechanisms.append("gossip")
        if self.enable_forgiveness:
            mechanisms.append("forgiveness")
        return "_".join(mechanisms) if mechanisms else "baseline"

    def prompt_claude(self, prompt: str, system_prompt: str, max_retries: int = 3,
                      timeout: int = 30, temperature: float = 1.0, max_tokens: int = 1024) -> str:
        """Send prompt to Claude and get response"""
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=timeout,
                )
                return response.content[0].text
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                print(f"Error: {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        raise Exception("Failed to get response from Claude")

    # === MECHANISM 1: REGRET ===
    def calculate_regret(self, agent: Agent, decision: float, outcome: float,
                         round_number: int, context: str) -> float:
        """
        Full regret model:
        - outcome regret (loss regret)
        - opportunity regret (missed-optimal regret)
        - optimism dampening
        - exponential memory decay
        """
        if not self.enable_regret:
            return 0.0

        # ---- EXPECTATION MODEL ------------------------------------
        # expected outcome from optimal decision
        optimal = agent.resources * self.optimal_ratio  # e.g., 0.5 or learnable
        expected_outcome = self.initial_endowment  # baseline expectation

        # ---- REGRET COMPONENTS ------------------------------------
        # 1. Opportunity Regret: wrong choice, regardless of outcome
        opp_regret = abs(decision - optimal) / max(optimal, 0.01)
        opp_regret = min(1.0, max(0.0, opp_regret))

        # 2. Outcome Regret: result worse than expectation
        outcome_delta = expected_outcome - outcome
        out_regret = max(0.0, outcome_delta) / max(expected_outcome, 0.01)
        out_regret = min(1.0, max(0.0, out_regret))

        # ---- WEIGHTING --------------------------------------------
        alpha = self.regret_outcome_weight  # e.g. 0.7
        beta = self.regret_opportunity_weight  # e.g. 0.3

        combined = alpha * out_regret + beta * opp_regret

        # ---- OPTIMISTIC MODIFIER ----------------------------------
        # optimism: reduces negative emotion if the agent sees future potential
        # optimism value in [0,1], 0 pessimistic, 1 very optimistic
        optimism = agent.optimism
        optimism_factor = 1.0 - 0.5 * optimism  # up to −50% reduction
        # alpha * out_regret + beta * opp_regret * optimism_factor

        regret = combined * optimism_factor
        regret = min(1.0, max(0.0, regret))

        # ---- STORE IN MEMORY --------------------------------------
        memory = RegretMemory(
            round_number=round_number,
            decision=decision,
            outcome_score=outcome,
            regret_level=regret,
            context=context,
        )
        agent.regret_memories.append(memory)

        # ---- EXPONENTIAL DECAY ------------------------------------
        # decay depending on how many rounds passed
        for m in agent.regret_memories:
            rounds_passed = round_number - m.round_number
            m.regret_level *= self.regret_decay ** rounds_passed

        # you can prune old memories if needed (optional)
        if len(agent.regret_memories) > self.max_regret_memories:
            agent.regret_memories = agent.regret_memories[-self.max_regret_memories:]

        return regret

    def get_regret_context(self, agent: Agent) -> str:
        """Get regret context for prompt (only if enabled)"""
        if not self.enable_regret:
            return ""

        recent_regrets = [r for r in agent.regret_memories[-5:] if r.regret_level > 0.3]
        if not recent_regrets:
            return ""

        regret_text = "You have regrets about past decisions: "
        for regret in recent_regrets:
            regret_text += f"Round {regret.round_number}: donated {regret.decision:.1f}, regret level {regret.regret_level:.2f}. "
        return regret_text + "\n"

    # You have regrets about past decisions:
    #  donated 0.0 regret level 0.9
    # round 2 donated 9.0 regret 0.7

    # === MECHANISM 2: GOSSIP ===
    def generate_gossip(self, observer: Agent, observed: Agent, interaction_quality: float,
                        round_number: int) -> Optional[GossipMessage]:
        """Generate gossip with sentiment scoring (only if enabled)"""
        if not self.enable_gossip:
            return None

        if random.random() > self.gossip_spread_probability:
            return None

        # sentiment ∈ [-1, 1]
        sentiment = 2 * interaction_quality - 1

        # credibility based on observer's own reputation score
        credibility = getattr(observer, "reputation", 0.5)  # default neutral
        reliability = 0.6 + 0.4 * credibility  # 0.6→1.0 range

        return GossipMessage(
            about_agent=observed.name,
            sentiment=sentiment,
            reliability=reliability,
            round_received=round_number,
            source=observer.name
        )


    def absorb_gossip(self, agent: Agent, current_round: int):
        """Absorb gossip into reputation beliefs with time decay"""
        if not self.enable_gossip:
            return

        # create/update belief scores
        for gossip in agent.gossip_received:
            age = current_round - gossip.round_received
            time_decay = 0.95 ** age

            if gossip.about_agent not in agent.reputation_beliefs:
                agent.reputation_beliefs[gossip.about_agent] = 0.0

            agent.reputation_beliefs[gossip.about_agent] += (
                    gossip.sentiment * gossip.reliability * time_decay
            )

        # keep only recent 20 gossip entries
        agent.gossip_received = agent.gossip_received[-20:]

    def get_gossip_context(self, agent: Agent, about_agent: Agent) -> str:
        """Get gossip context with sentiment-based descriptions (only if enabled)"""
        if not self.enable_gossip:
            return ""

        gossip_list = [g for g in agent.gossip_received if g.about_agent == about_agent.name]
        if not gossip_list:
            return ""

        # Mark that gossip is influencing this decision
        agent.current_round_gossip_influenced = True

        # create display strings
        texts = []
        for g in gossip_list[-3:]:
            direction = "cooperative" if g.sentiment > 0 else "uncooperative"
            texts.append(f"You heard {about_agent.name} is {direction} (via {g.source})")

        return " ".join(texts) + "\n"

    def spread_gossip(self, agents: List[Agent]):
        """Spread gossip among agents (only if enabled)"""
        if not self.enable_gossip:
            return

        for agent in agents:
            if agent.gossip_to_share:
                num_to_share = random.randint(2, min(3, len(agents) - 1))
                recipients = random.sample([a for a in agents if a != agent], num_to_share)

                for recipient in recipients:
                    for gossip in agent.gossip_to_share:
                        new_gossip = GossipMessage(
                            about_agent=gossip.about_agent,
                            sentiment=gossip.sentiment,
                            reliability=gossip.reliability * 0.9,
                            round_received=gossip.round_received,
                            source=agent.name
                        )
                        # Removed recipient.gossip_belief gating; deliver gossip directly
                        recipient.gossip_received.append(new_gossip)

                agent.gossip_to_share.clear()

    # === MECHANISM 3: FORGIVENESS ===
    def update_forgiveness(self, agent: Agent, other_agent_name: str,
                           was_positive_interaction: bool, round_number: int,
                           severity: float = 1.0):
        """Improved forgiveness model with decay and proportional changes."""
        if not self.enable_forgiveness:
            return

        # Initialize record if not exists
        if other_agent_name not in agent.forgiveness_records:
            agent.forgiveness_records[other_agent_name] = ForgivenessRecord(
                agent_name=other_agent_name,
                offense_count=0,
                forgiveness_level=1.0,
                last_interaction_round=round_number
            )

        record = agent.forgiveness_records[other_agent_name]

        # Time-based natural forgiveness: exponential decay of offense memory
        rounds_passed = round_number - record.last_interaction_round
        record.offense_count *= self.forgiveness_decay ** rounds_passed

        if not was_positive_interaction:
            # Severity scales offense impact
            record.offense_count += severity
            # Forgiveness decreases very gradually - max drop is 0.15 at full trust
            drop = severity * (0.05 + 0.1 * record.forgiveness_level)
            record.forgiveness_level = max(0.0, record.forgiveness_level - drop)
        else:
            # Forgiveness builds slowly from low levels, faster from medium levels
            healing = self.forgiveness_increment * (record.forgiveness_level + 0.1)
            record.forgiveness_level = min(1.0, record.forgiveness_level + healing)

        record.last_interaction_round = round_number

    def get_forgiveness_context(self, agent: Agent, other_agent: Agent) -> str:
        """Get forgiveness context with improved emotional descriptions (only if enabled)"""
        if not self.enable_forgiveness:
            return ""

        record = agent.forgiveness_records.get(other_agent.name)
        if not record:
            return ""

        level = record.forgiveness_level
        offenses = record.offense_count

        if level < 0.2:
            return (f"You still feel deeply hurt by {other_agent.name}. "
                    f"Forgiveness is low ({level:.2f}), offenses: {offenses:.1f}.\n")

        elif level < 0.6:
            return (f"You remember past conflicts with {other_agent.name}. "
                    f"Forgiveness ({level:.2f}) is recovering, offenses: {offenses:.1f}.\n")

        elif offenses > 0.1:
            return (f"You have mostly forgiven {other_agent.name}. "
                    f"Past issues remain in memory ({level:.2f}).\n")

        return ""

    # === REPUTATION SYSTEM ===
    def update_global_reputations(self, agents: List[Agent]):
        """
        Update global reputation scores from aggregated beliefs.

        Key insight: When gossip/regret/forgiveness are OFF, no belief signals exist,
        so reputation defaults to neutral (0.5). When mechanisms are ON, belief signals
        accumulate and drive reputation changes.

        This allows us to demonstrate that social norms cause reputation dynamics.
        """
        for agent in agents:
            # Collect all beliefs about this agent from other agents
            beliefs_about_agent = []
            for other_agent in agents:
                if other_agent != agent and agent.name in other_agent.reputation_beliefs:
                    beliefs_about_agent.append(other_agent.reputation_beliefs[agent.name])

            if beliefs_about_agent:
                # Calculate mean belief and convert to reputation using tanh
                mean_belief = sum(beliefs_about_agent) / len(beliefs_about_agent)
                # reputation = 0.5 + 0.5 * tanh(mean_belief)
                # tanh bounds output to [-1, 1], so final reputation is in [0, 1]
                agent.reputation = 0.5 + 0.5 * np.tanh(mean_belief)
            else:
                # Default neutral reputation if no beliefs exist
                # This is what happens in baseline condition (no gossip → no beliefs)
                agent.reputation = 0.5

    def apply_behavioral_biases(self, response: float, donor: Agent, recipient: Agent) -> float:
        """Apply reputation, forgiveness, and regret biases to donation (always enabled)"""
        biased_response = response

        # Bias based on recipient's reputation (only if gossip is enabled)
        if self.enable_gossip:
            reputation_modifier = 1 + self.reputation_bias_strength * (recipient.reputation - 0.5)
            biased_response *= reputation_modifier
            # 10
            # 1 + 0.4 * (0-0.5)
            # 0.5  =N
            #0.4 = 1 + 0.4 (-0.1)=0.8

        # Bias based on forgiveness level toward recipient (only if forgiveness is enabled)
        if self.enable_forgiveness and recipient.name in donor.forgiveness_records:
            forgiveness_level = donor.forgiveness_records[recipient.name].forgiveness_level
            forgiveness_modifier = 1 + self.forgiveness_bias_strength * (forgiveness_level - 0.5)
            biased_response *= forgiveness_modifier

        # Bias based on donor's regret level (only if regret is enabled)
        if self.enable_regret and donor.regret_level > 0:
            regret_modifier = 1 - self.regret_bias_strength * donor.regret_level
            biased_response *= regret_modifier

        # Clamp to valid range
        biased_response = max(0.0, min(biased_response, donor.resources))

        return biased_response

    # === CORE GAME LOGIC ===
    def parse_strategy_output(self, output: str) -> Tuple[str, str]:
        """Parse strategy from LLM output"""
        parts = output.split("My strategy will be", 1)
        if len(parts) == 2:
            strategy_justification = parts[0].strip()
            strategy = "My strategy will be" + parts[1].strip()
        else:
            strategy_justification = ""
            strategy = output.strip()
        return strategy_justification, strategy

    def generate_strategy(self, agent_name: str, generation_number: int,
                          inherited_strategies: str, system_prompt: str) -> Tuple[str, str]:
        """Generate strategy for an agent"""
        mechanism_desc = []
        if self.enable_regret:
            mechanism_desc.append("your past regrets")
        if self.enable_gossip:
            mechanism_desc.append("gossip about other players")
        if self.enable_forgiveness:
            mechanism_desc.append("your forgiveness towards others")

        mechanism_text = ", ".join(mechanism_desc) if mechanism_desc else "your partner's recent behavior"

        if generation_number == 1:
            prompt = (
                f"Your name is {agent_name}. "
                f"Based on the game description, create a strategy. "
                f"You will have access to: {mechanism_text}. "
                "Think step by step about a successful strategy. "
                "Then describe your strategy in one sentence starting: My strategy will be."
            )
        else:
            prompt = (
                f"Your name is {agent_name}. "
                f"Here is advice from the best 50% of the previous generation:\n{inherited_strategies}\n\n"
                f"Create your own improved strategy considering: {mechanism_text}. "
                "Then describe your strategy in one sentence starting: My strategy will be."
            )

        strategy_output = self.prompt_claude(prompt, system_prompt)
        strategy_justification, strategy = self.parse_strategy_output(strategy_output)

        print(f"{agent_name}:\nStrategy: {strategy}\n")
        return strategy_justification, strategy

    def initialize_agents(self, num_agents: int, generation_number: int,
                          inherited_strategies: list, system_prompt: str) -> List[Agent]:
        """Initialize agents with strategies"""
        agents = []

        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(num_agents):
                name = f"{generation_number}_{i + 1}"
                inherited_str = "\n".join(inherited_strategies) if inherited_strategies else ""
                futures.append(executor.submit(
                    self.generate_strategy, name, generation_number, inherited_str, system_prompt
                ))

            for i, future in enumerate(futures):
                strategy_justification, strategy = future.result()
                name = f"{generation_number}_{i + 1}"
                agents.append(Agent(
                    name=name,
                    resources=float(self.initial_endowment),
                    strategy=strategy,
                    strategy_justification=strategy_justification,
                ))

        random.shuffle(agents)
        return agents

    def create_donor_prompt(self, donor: Agent, recipient: Agent, round_number: int,
                            generation: int) -> str:
        """Create prompt with active mechanisms"""
        strategy_text = f"Your strategy: {donor.strategy}" if donor.strategy else ""

        # Collect mechanism contexts
        regret_context = self.get_regret_context(donor)
        gossip_context = self.get_gossip_context(donor, recipient)
        forgiveness_context = self.get_forgiveness_context(donor, recipient)

        # Get recipient behavior
        recipient_behavior = ""
        if donor.traces and recipient.traces:
            last_trace = recipient.traces[-1]
            if isinstance(last_trace, list) and len(last_trace) > 0:
                recipient_behavior = " ".join(last_trace[-3:][::-1])

        prompt = (
            f"Your name is {donor.name}. {strategy_text}\n"
            f"Generation {generation}, Round {round_number}\n"
            f"Paired with {recipient.name} who has {recipient.resources:.1f} units.\n"
            f"You have {donor.resources:.1f} units.\n\n"
            f"Recent behavior: {recipient_behavior}\n"
            f"{gossip_context}"
            f"{regret_context}"
            f"{forgiveness_context}\n"
            "How many units do you give up? (0 to your resources)\n"
            "Briefly explain, then state: Answer: [number]"
        )

        return prompt

    def bipartite_round_robin(self, agents: List[Agent]) -> List[List[Tuple[Agent, Agent]]]:
        """Generate round-robin pairings"""
        num_agents = len(agents)
        assert num_agents % 2 == 0

        group_a = agents[:num_agents // 2]
        group_b = agents[num_agents // 2:]
        rounds = []
        toggle_roles = False

        for i in range(len(group_a)):
            rotated_group_b = group_b[-i:] + group_b[:-i]
            if toggle_roles:
                round_pairings = list(zip(rotated_group_b, group_a))
            else:
                round_pairings = list(zip(group_a, rotated_group_b))
            rounds.append(round_pairings)
            toggle_roles = not toggle_roles

        return rounds

    def handle_pairing(self, donor: Agent, recipient: Agent, round_index: int,
                       generation: int, game_number: int, agent_locks: Dict,
                       system_prompt: str) -> Tuple[str, AgentRoundData, AgentRoundData]:
        """Handle single pairing interaction"""
        with agent_locks[donor.name], agent_locks[recipient.name]:
            # Reset gossip influence flag for this round
            donor.current_round_gossip_influenced = False
            donor.current_partner = recipient
            donor.is_donor = True

            recipient.current_partner = donor
            recipient.is_donor = False

            if round_index > 0 and recipient.traces:
                donor.traces.append(recipient.traces[-1].copy())

            prompt = self.create_donor_prompt(donor, recipient, round_index + 1, generation)

            # Get decision
            valid_response = False
            response = 0
            justification = ""

            for attempt in range(5):
                try:
                    full_response = self.prompt_claude(prompt, system_prompt, timeout=30)

                    parts = full_response.split('Answer:', 1)
                    if len(parts) == 2:
                        justification = parts[0].strip()
                        answer_part = parts[1].strip()
                        match = re.search(r'(\d+(?:\.\d+)?)', answer_part)
                        if match:
                            response = float(match.group(1))
                            if 0 <= response <= donor.resources:
                                valid_response = True
                                break
                except Exception as e:
                    print(f"Error: {e}")

            if not valid_response:
                response = 0

            # Apply behavioral biases (always enabled)
            response = self.apply_behavioral_biases(response, donor, recipient)

        # Store justification and gossip influence status
        donor.justification = justification
        gossip_influenced = donor.current_round_gossip_influenced

        # Process donation
        percentage_donated = response / donor.resources if donor.resources > 0 else 0
        donor.resources -= response
        donor.total_donated += response
        donor.last_donation = response
        recipient.resources += self.cooperation_gain * response
        recipient.total_received += self.cooperation_gain * response
        recipient.last_received = self.cooperation_gain * response

        # Apply mechanisms
        outcome = donor.resources
        regret_level = self.calculate_regret(
            donor, response, outcome, round_index + 1,
            f"Donated {response:.1f} to {recipient.name}"
        )
        donor.regret_level = regret_level

        was_positive = percentage_donated > 0.2  # Lowered from 0.3 to 0.2
        severity = 1.0 if not was_positive else 0.5
        self.update_forgiveness(donor, recipient.name, was_positive, round_index + 1, severity)
        self.update_forgiveness(recipient, donor.name, was_positive, round_index + 1, severity)

        # Track forgiveness given (updated after update_forgiveness to ensure record exists)
        donor.forgiveness_given = donor.forgiveness_records[recipient.name].forgiveness_level if recipient.name in donor.forgiveness_records else 1.0
        recipient.forgiveness_given = recipient.forgiveness_records[donor.name].forgiveness_level if donor.name in recipient.forgiveness_records else 1.0

        interaction_quality = percentage_donated
        gossip = self.generate_gossip(donor, recipient, interaction_quality, round_index + 1)
        if gossip:
            donor.gossip_to_share.append(gossip)

        # Update traces
        new_trace = recipient.traces[-1].copy() if recipient.traces else []
        new_trace.append(
            f"In round {round_index + 1}, {donor.name} donated {percentage_donated * 100:.1f}% to {recipient.name}."
        )
        donor.traces.append(new_trace)

        action_info = f"{donor.name} → {recipient.name}: {response:.1f} ({percentage_donated:.1%})\n"

        # Update histories
        donor_history = f"Round {round_index + 1}: Donated {response:.1f} to {recipient.name}"
        recipient_history = f"Round {round_index + 1}: Received {self.cooperation_gain * response:.1f} from {donor.name}"
        donor.history.append(donor_history)
        recipient.history.append(recipient_history)

        # Create round data with reputation
        donor_data = AgentRoundData(
            agent_name=donor.name,
            round_number=round_index + 1,
            game_number=game_number,
            paired_with=recipient.name,
            current_generation=generation,
            resources=donor.resources,
            donated=response,
            received=0,
            strategy=donor.strategy,
            strategy_justification=donor.strategy_justification,
            is_donor=True,
            traces=donor.traces,
            history=donor.history,
            justification=justification,
            regret_level=regret_level,
            forgiveness_given=donor.forgiveness_given,
            gossip_influenced=gossip_influenced,
            reputation=donor.reputation
        )

        recipient_data = AgentRoundData(
            agent_name=recipient.name,
            round_number=round_index + 1,
            game_number=game_number,
            paired_with=donor.name,
            current_generation=generation,
            resources=recipient.resources,
            donated=0,
            received=self.cooperation_gain * response,
            strategy=recipient.strategy,
            strategy_justification=recipient.strategy_justification,
            is_donor=False,
            traces=recipient.traces,
            history=recipient.history,
            regret_level=0.0,
            forgiveness_given=recipient.forgiveness_given,
            gossip_influenced=False,
            reputation=recipient.reputation
        )

        return action_info, donor_data, recipient_data

    def play_donor_game(self, agents: List[Agent], rounds: List[Tuple[Agent, Agent]],
                        generation: int, system_prompt: str) -> List[str]:
        """Play the donor game"""
        full_history = []
        agent_locks = {agent.name: Lock() for agent in agents}

        for round_index, round_pairings in enumerate(rounds):
            if round_index == 0:
                for agent in agents:
                    agent.traces = [[f"{agent.name} had no previous interactions."]]

            # ALWAYS absorb gossip (no-op if gossip disabled)
            # This ensures reputation_beliefs dict is updated when gossip exists
            if self.enable_gossip:
                for agent in agents:
                    self.absorb_gossip(agent, round_index + 1)

            # ALWAYS update global reputations from aggregated beliefs
            # When gossip is OFF: no beliefs exist → all agents stay at 0.5
            # When gossip is ON: beliefs accumulate → reputations diverge
            # This is the KEY to showing that norms drive reputation dynamics
            self.update_global_reputations(agents)

            round_history = []

            with ThreadPoolExecutor(max_workers=min(len(round_pairings), 10)) as executor:
                futures = []
                for donor, recipient in round_pairings:
                    future = executor.submit(
                        self.handle_pairing,
                        donor, recipient, round_index, generation, 1,
                        agent_locks, system_prompt
                    )
                    futures.append(future)

                for future in futures:
                    action_info, donor_data, recipient_data = future.result()
                    round_history.append(action_info)
                    self.simulation_data.agents_data.append(asdict(donor_data))
                    self.simulation_data.agents_data.append(asdict(recipient_data))

            # Spread gossip (no-op if gossip disabled)
            self.spread_gossip(agents)

            full_history.append(f"\n=== Round {round_index + 1} ===\n")
            full_history.extend(round_history)

        for agent in agents:
            agent.total_final_score = agent.resources

        avg_resources = sum(agent.resources for agent in agents) / len(agents)
        avg_reputation = sum(agent.reputation for agent in agents) / len(agents)
        print(f"Average final resources: {avg_resources:.2f}, Average reputation: {avg_reputation:.4f}")
        self.all_average_final_resources.append(avg_resources)

        return full_history

    def select_top_agents(self, agents: List[Agent]) -> List[Agent]:
        """Select top 50% of agents"""
        return sorted(agents, key=lambda x: x.total_final_score, reverse=True)[:len(agents) // 2]

    def run_simulation(self, num_generations: int = 3, num_agents: int = 12):
        """Run full simulation"""
        mechanism_names = self.get_mechanism_name()

        system_prompt = (
            f"You are a game-playing agent in a multi-round donation game. "
            f"Initial endowment: {self.initial_endowment} units. "
            f"Donations are multiplied by {self.cooperation_gain}x. "
            f"Active mechanisms: {mechanism_names}. "
            "Goal: maximize resources."
        )

        self.simulation_data = SimulationData(hyperparameters={
            "num_generations": num_generations,
            "num_agents": num_agents,
            "initial_endowment": self.initial_endowment,
            "cooperation_gain": self.cooperation_gain,
            "model": self.model,
            "enable_regret": self.enable_regret,
            "enable_gossip": self.enable_gossip,
            "enable_forgiveness": self.enable_forgiveness,
            "enable_reputation_bias": True,  # Always enabled
            "mechanisms": mechanism_names
        })

        agents = self.initialize_agents(num_agents, 1, [], system_prompt)
        self.all_agents.extend(agents)

        for generation in range(1, num_generations + 1):
            print(f"\n{'=' * 60}")
            print(f"GENERATION {generation} ({mechanism_names})")
            print(f"{'=' * 60}\n")

            rounds = self.bipartite_round_robin(agents)
            history = self.play_donor_game(agents, rounds, generation, system_prompt)

            print("\nGeneration Summary:")
            for agent in sorted(agents, key=lambda x: x.total_final_score, reverse=True):
                print(f"  {agent.name}: {agent.total_final_score:.1f} resources, reputation: {agent.reputation:.2f}")

            if generation < num_generations:
                surviving_agents = self.select_top_agents(agents)
                surviving_strategies = [agent.strategy for agent in surviving_agents]

                for agent in surviving_agents:
                    agent.resources = float(self.initial_endowment)
                    agent.old_traces = agent.traces

                new_agents = self.initialize_agents(
                    num_agents // 2, generation + 1, surviving_strategies, system_prompt
                )

                agents = surviving_agents + new_agents
                self.all_agents.extend(new_agents)
                random.shuffle(agents)

        self.save_results()

    def save_results(self, folder_path: str = 'results'):
        """Save simulation results"""
        os.makedirs(folder_path, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mechanism_name = self.get_mechanism_name()
        filename = f"claude_donor_{mechanism_name}_{timestamp}.json"
        filepath = os.path.join(folder_path, filename)

        def make_serializable(obj):
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            elif hasattr(obj, '__dict__'):
                return make_serializable(obj.__dict__)
            else:
                return str(obj)

        data_dict = self.simulation_data.to_dict()
        serializable_data = make_serializable(data_dict)

        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)

        print(f"\nResults saved: {filepath}")

