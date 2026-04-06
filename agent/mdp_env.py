import math
import logging
from typing import Dict, Any, List, Optional
import sys
import os

# Add orchestrator to path to import models if needed, though here we'll keep it decoupled
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator.orchestrator import EvaluationResult

logger = logging.getLogger("MDP_Env")

class AutoResearchEnv:
    def __init__(self, sota_bpb: float = 1.0):
        """
        MDP Environment for AutoResearch-RL agent.

        Args:
            sota_bpb (float): The current State-of-the-Art BPB value.
        """
        self.sota_bpb = sota_bpb
        self.history = []  # H_t: Memory of past K experiments

        # Penalties defined by the specification
        self.p_syntax = 5.0
        self.p_waste = 2.0    # Penalty for OOM or capacity limit exceeded
        self.p_causality = 100.0

    def calculate_reward(self, result: EvaluationResult, is_novel: bool = True, causality_leak: bool = False, abort_step: int = 0, total_expected_steps: int = 1000) -> float:
        """
        Calculates the reward r_t for a given action (diff patch execution).

        Formula: r_t = Δbpb_t + r_novelty - p_syntax - p_waste - p_causality
        """
        reward = 0.0
        reward_components = {}

        # 1. Causality check
        if causality_leak:
            logger.error("Causality leak detected! Applying maximum penalty.")
            reward_components["causality"] = -self.p_causality
            return -self.p_causality, reward_components

        # 2. Syntax / Compilation failure
        if result.status == "ABORTED" and result.error_message == "SyntaxError":
            reward_components["syntax"] = -self.p_syntax
            return -self.p_syntax, reward_components

        # 3. Constraint waste (OOM, 16MB limit, etc.)
        if result.status == "ABORTED" and result.error_message == "CapacityLimitExceeded":
            reward_components["capacity"] = -self.p_waste
            return -self.p_waste, reward_components

        # 4. Improvement metric (Δbpb_t)
        # We want to minimize BPB, so a lower final_bpb means a positive delta.
        # Δbpb_t = (SOTA - final_bpb) * scaling_factor
        if result.status == "COMPLETED" and result.final_bpb is not None:
            # Scale to make small improvements meaningful
            delta_bpb = (self.sota_bpb - result.final_bpb) * 10.0
            reward += delta_bpb
            reward_components["delta_bpb"] = delta_bpb

            # Update SOTA if we beat it
            if result.final_bpb < self.sota_bpb:
                logger.info(f"New SOTA achieved! {result.final_bpb:.4f} < {self.sota_bpb:.4f}")
                self.sota_bpb = result.final_bpb
        elif result.status == "ABORTED" and result.error_message == "SPRT_EARLY_STOPPING":
            # Penalty for wasting compute. The later we abort, the heavier the penalty.
            waste_ratio = abort_step / max(1, total_expected_steps)
            sprt_penalty = -0.5 * (1 + waste_ratio * 2.0) # Up to 3x penalty for late aborts
            reward += sprt_penalty
            reward_components["sprt_abort_penalty"] = sprt_penalty

        # 5. Novelty Bonus (Epsilon-novelty to prevent deterministic collapse)
        if is_novel:
            # Dynamically scale novelty based on how "stuck" we are (e.g. history size)
            r_novelty = 0.1 + (len(self.history) / 32.0) * 0.1
            reward += r_novelty
            reward_components["novelty_bonus"] = r_novelty

        logger.info(f"Reward Components: {reward_components}")
        return float(reward), reward_components

    def step(self, result: EvaluationResult, action_patch: str, causality_leak: bool = False, abort_step: int = 0) -> Dict[str, Any]:
        """
        Simulates one step in the MDP. Agent takes an action (code mutation),
        orchestrator runs it, and env calculates the state transition and reward.
        """
        # Compare action_patch against history for novelty
        is_novel = action_patch not in [item['patch'] for item in self.history]

        reward, components = self.calculate_reward(result, is_novel=is_novel, causality_leak=causality_leak, abort_step=abort_step)

        # Update memory (H_t)
        self.history.append({
            'job_id': result.job_id,
            'patch': action_patch,
            'status': result.status,
            'final_bpb': result.final_bpb,
            'reward': reward,
            'components': components
        })

        # Keep only K=32 experiments
        if len(self.history) > 32:
            self.history.pop(0)

        return {
            "reward": reward,
            "sota_bpb": self.sota_bpb,
            "memory_size": len(self.history)
        }
