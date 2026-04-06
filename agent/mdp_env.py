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

    def calculate_reward(self, result: EvaluationResult, is_novel: bool = True, causality_leak: bool = False) -> float:
        """
        Calculates the reward r_t for a given action (diff patch execution).

        Formula: r_t = Δbpb_t + r_novelty - p_syntax - p_waste - p_causality
        """
        reward = 0.0

        # 1. Causality check
        if causality_leak:
            logger.error("Causality leak detected! Applying maximum penalty.")
            return -self.p_causality

        # 2. Syntax / Compilation failure
        if result.status == "ABORTED" and result.error_message == "SyntaxError":
            return -self.p_syntax

        # 3. Constraint waste (OOM, 16MB limit, etc.)
        if result.status == "ABORTED" and result.error_message == "CapacityLimitExceeded":
            return -self.p_waste

        # 4. Improvement metric (Δbpb_t)
        # We want to minimize BPB, so a lower final_bpb means a positive delta.
        # Δbpb_t = (SOTA - final_bpb) * scaling_factor
        if result.status == "COMPLETED" and result.final_bpb is not None:
            # Scale to make small improvements meaningful
            delta_bpb = (self.sota_bpb - result.final_bpb) * 10.0
            reward += delta_bpb

            # Update SOTA if we beat it
            if result.final_bpb < self.sota_bpb:
                logger.info(f"New SOTA achieved! {result.final_bpb:.4f} < {self.sota_bpb:.4f}")
                self.sota_bpb = result.final_bpb
        elif result.status == "ABORTED" and result.error_message == "SPRT_EARLY_STOPPING":
            # Small penalty for wasting GPU time on a bad run, but less than a full crash
            reward -= 0.5

        # 5. Novelty Bonus (Epsilon-novelty to prevent deterministic collapse)
        if is_novel:
            r_novelty = 0.1
            reward += r_novelty

        return float(reward)

    def step(self, result: EvaluationResult, action_patch: str, causality_leak: bool = False) -> Dict[str, Any]:
        """
        Simulates one step in the MDP. Agent takes an action (code mutation),
        orchestrator runs it, and env calculates the state transition and reward.
        """
        # In a real run, we'd compare action_patch against history for novelty
        is_novel = action_patch not in [item['patch'] for item in self.history]

        reward = self.calculate_reward(result, is_novel=is_novel, causality_leak=causality_leak)

        # Update memory (H_t)
        self.history.append({
            'job_id': result.job_id,
            'patch': action_patch,
            'status': result.status,
            'final_bpb': result.final_bpb,
            'reward': reward
        })

        # Keep only K=32 experiments
        if len(self.history) > 32:
            self.history.pop(0)

        return {
            "reward": reward,
            "sota_bpb": self.sota_bpb,
            "memory_size": len(self.history)
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    env = AutoResearchEnv(sota_bpb=1.0)

    # Simulate a good result
    good_result = EvaluationResult(job_id="test-1", status="COMPLETED", final_bpb=0.98, artifact_size=15_000_000)
    step_info = env.step(good_result, action_patch="+ def new_layer(): pass")
    print(f"Good Step Reward: {step_info['reward']}")

    # Simulate a syntax error
    bad_result = EvaluationResult(job_id="test-2", status="ABORTED", final_bpb=None, artifact_size=0, error_message="SyntaxError")
    step_info2 = env.step(bad_result, action_patch="- import os")
    print(f"Syntax Error Reward: {step_info2['reward']}")
