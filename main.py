import logging
import time
import os
import shutil

from orchestrator.orchestrator import Orchestrator
from orchestrator.docker_runner import GPUDispatcher
from agent.ppo_agent import PPOMetaAgent
from agent.mdp_env import AutoResearchEnv
from gpu_cluster.sprt import SPRTFilter
from auditor.causality_auditor import check_causality_leak

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoResearch-RL-Main")

def save_artifact(source_code: str, bpb: float, step: int):
    """Saves the current best script to the artifacts directory."""
    os.makedirs("artifacts", exist_ok=True)
    filename = f"artifacts/train_gpt_bpb_{bpb:.4f}_step_{step}.py"
    with open(filename, "w") as f:
        f.write(source_code)

    # Also save as latest
    shutil.copy(filename, "artifacts/train_gpt_best_latest.py")
    logger.info(f"💾 Saved new artifact to {filename}")

def run_perpetual_loop():
    logger.info("🚀 Starting AutoResearch-RL Perpetual Loop")

    # Initialize Core Components
    orchestrator = Orchestrator()
    agent = PPOMetaAgent(system_prompt="Minimize BPB under 16MB. No causality leaks. 10 min hard limit.")
    env = AutoResearchEnv(sota_bpb=1.0)

    # Load Golden Seed
    with open("seed/train_gpt.py", "r") as f:
        current_best_code = f.read()

    logger.info("Loaded Golden Seed.")

    iteration = 1
    max_iterations = 3 # Run for 3 iterations to test the loop

    while iteration <= max_iterations:
        logger.info(f"\n{'='*50}\n🔄 Starting Iteration {iteration}\n{'='*50}")

        # 1. Gather Telemetry (Mocked for now)
        telemetry = {
            "last_vram_peak_mb": 78000,
            "recent_oom": False,
            "iteration": iteration
        }

        # 2. Agent Action Phase (Policy Evaluation -> Action)
        candidate_code = agent.generate_action(current_best_code, env.history, telemetry)

        # 3. Security Audit (Causality)
        causality_leak = check_causality_leak(candidate_code)

        # 4. Orchestrator Pre-checks (AST & Capacity)
        if causality_leak:
            logger.warning("Causality Audit Failed. Skipping GPU dispatch.")
            # Create a mock result to feed the env the penalty
            result = orchestrator.submit_job(candidate_code)
            result.status = "ABORTED"
            result.error_message = "CausalityLeak"
        else:
            # Smoke & Capacity check via Orchestrator
            if not orchestrator.run_smoke_test(candidate_code):
                 # Submit job just to get the formatted error EvaluationResult
                 result = orchestrator.submit_job(candidate_code)
            elif orchestrator.simulate_compression_and_capacity(candidate_code) > 16_000_000:
                 result = orchestrator.submit_job(candidate_code)
            else:
                # 5. Dispatch to GPU Cluster with SPRT filtering
                sprt = SPRTFilter(sota_threshold=env.sota_bpb)

                def sprt_callback(step: int, loss: float) -> bool:
                    return sprt.update_and_check(step, loss)

                dispatcher = GPUDispatcher(sprt_callback=sprt_callback, time_limit_sec=600)

                job_id = f"job_iter_{iteration}"
                result = dispatcher.dispatch(job_id, candidate_code, num_parameters=12_000_000)

        # 6. Environment Step (Calculate Reward & Update Memory)
        step_info = env.step(result, action_patch="MOCK_PATCH_APPLIED", causality_leak=causality_leak)

        logger.info(f"Iteration Result -> Status: {result.status}, BPB: {result.final_bpb}, Reward: {step_info['reward']:.4f}")

        # 7. Update SOTA and Artifacts
        if result.status == "COMPLETED" and result.final_bpb is not None and result.final_bpb < env.sota_bpb:
            logger.info("🏆 Candidate code accepted as new SOTA!")
            current_best_code = candidate_code
            save_artifact(current_best_code, result.final_bpb, iteration)

        iteration += 1
        time.sleep(1)

    logger.info("🛑 Simulation Limit Reached. Terminating AutoResearch-RL Loop.")

if __name__ == "__main__":
    run_perpetual_loop()
