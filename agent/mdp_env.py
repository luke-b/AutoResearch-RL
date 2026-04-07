import math
import logging
from typing import Dict, Any, List, Optional
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator.orchestrator import EvaluationResult

logger = logging.getLogger("MDP_Env")

class AutoResearchEnv:
    def __init__(self, sota_bpb: float = 1.0):
        """
        MDP Environment for AutoResearch-RL agent.
        """
        self.sota_bpb = sota_bpb
        self.history = []  # H_t: Memory of past K experiments

        # Penalties
        self.p_syntax = 5.0
        self.p_waste = 2.0
        self.p_causality = 100.0

    def calculate_reward(self, result: EvaluationResult, is_novel: bool = True, causality_leak: bool = False, abort_step: int = 0, total_expected_steps: int = 200, use_novelty: bool = True, elapsed_time: float = 0.0, uncertainty: float = 0.0) -> float:
        """
        Calculates the reward r_t for a given action (diff patch execution).
        Formula: r_t = Δbpb_t + r_novelty - p_syntax - p_waste - p_causality - compute_cost
        Includes uncertainty handling.
        """
        reward = 0.0
        reward_components = {}

        if causality_leak:
            logger.error("Causality leak detected! Applying maximum penalty.")
            reward_components["causality"] = -self.p_causality
            return -self.p_causality, reward_components

        if result.status == "ABORTED" and result.error_message == "SyntaxError":
            reward_components["syntax"] = -self.p_syntax
            return -self.p_syntax, reward_components

        if result.status == "ABORTED" and result.error_message == "CapacityLimitExceeded":
            reward_components["capacity"] = -self.p_waste
            return -self.p_waste, reward_components

        # Richer compute cost modeling
        # Penalize wall-clock time scaled appropriately (e.g., -0.01 per second over expected fast baseline)
        compute_cost = -0.005 * elapsed_time
        reward += compute_cost
        reward_components["compute_cost"] = compute_cost

        if result.status == "COMPLETED" and result.final_bpb is not None:
            # Guard against NaN before doing math
            if torch.isnan(torch.tensor(result.final_bpb)):
                reward_components["delta_bpb"] = 0.0
            else:
                # Explicit uncertainty handling: discount the delta by confidence bounds
                safe_final_bpb = result.final_bpb + uncertainty
                delta_bpb = (self.sota_bpb - safe_final_bpb) * 10.0
                reward += delta_bpb
                reward_components["delta_bpb"] = delta_bpb

                if safe_final_bpb < self.sota_bpb:
                    logger.info(f"New SOTA achieved! {safe_final_bpb:.4f} < {self.sota_bpb:.4f} (uncertainty: {uncertainty})")
                    self.sota_bpb = safe_final_bpb

        elif result.status == "ABORTED" and result.error_message == "SPRT_EARLY_STOPPING":
            # Adaptive Compute Waste Penalty
            waste_ratio = abort_step / max(1, total_expected_steps)
            sprt_penalty = -0.5 * (1 + waste_ratio * 3.0) # Up to 4x penalty for late aborts
            reward += sprt_penalty
            reward_components["sprt_abort_penalty"] = sprt_penalty

        if is_novel and use_novelty:
            # Adaptive Novelty Bonus: Increases exponentially as history fills up,
            # encouraging the agent to explore further away from the local minima.
            staleness = len(self.history) / 32.0
            r_novelty = 0.1 * math.exp(staleness)
            reward += r_novelty
            reward_components["novelty_bonus"] = r_novelty

        logger.info(f"Reward Components: {reward_components}")
        return float(reward), reward_components

    def step(self, result: EvaluationResult, action_patch: str, causality_leak: bool = False, abort_step: int = 0, use_novelty: bool = True, elapsed_time: float = 0.0, uncertainty: float = 0.0) -> Dict[str, Any]:
        """
        Simulates one step in the MDP. Agent takes an action (code mutation),
        orchestrator runs it, and env calculates the state transition and reward.
        """
        is_novel = action_patch not in [item['patch'] for item in self.history]

        reward, components = self.calculate_reward(result, is_novel=is_novel, causality_leak=causality_leak, abort_step=abort_step, use_novelty=use_novelty, elapsed_time=elapsed_time, uncertainty=uncertainty)

        self.history.append({
            'job_id': result.job_id,
            'patch': action_patch,
            'status': result.status,
            'final_bpb': result.final_bpb,
            'reward': reward,
            'components': components,
            'remediation': getattr(result, "remediation", None)
        })

        if len(self.history) > 32:
            self.history.pop(0)

        return {
            "reward": reward,
            "sota_bpb": self.sota_bpb,
            "memory_size": len(self.history)
        }
