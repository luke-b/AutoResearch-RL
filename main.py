import logging
import time
import os
import shutil
import json

from orchestrator.orchestrator import Orchestrator, MAX_TIME_SECONDS
from orchestrator.docker_runner import GPUDispatcher
from agent.ppo_agent import PPOMetaAgent
from agent.mdp_env import AutoResearchEnv
from gpu_cluster.sprt import SPRTFilter
from auditor.causality_auditor import check_causality_leak

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoResearch-RL-Main")

AUTORESEARCH_MODE = os.environ.get("AUTORESEARCH_MODE", "CLUSTER").upper()

def log_experiment_json(iteration: int, job_id: str, patch: str, status: str, bpb: float, reward: float, components: dict, causality_leak: bool, abort_step: int, remediation: str = None, category: str = "general", log_file: str = "experiment_logs.jsonl"):
    """Appends structured JSON logs for external analysis/dashboarding."""
    log_entry = {
        "timestamp": time.time(),
        "iteration": iteration,
        "job_id": job_id,
        "status": status,
        "final_bpb": bpb,
        "reward": reward,
        "reward_components": components,
        "causality_leak": causality_leak,
        "abort_step": abort_step,
        "patch": patch,
        "remediation": remediation,
        "category": category
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def save_artifact(source_code: str, bpb: float, step: int):
    """Saves the current best script to the artifacts directory."""
    os.makedirs("artifacts", exist_ok=True)
    filename = f"artifacts/train_gpt_bpb_{bpb:.4f}_step_{step}.py"
    with open(filename, "w") as f:
        f.write(source_code)

    # Also save as latest
    shutil.copy(filename, "artifacts/train_gpt_best_latest.py")
    logger.info(f"💾 Saved new artifact to {filename}")

import argparse

def run_perpetual_loop(max_iterations: int = 1, use_novelty: bool = True, use_sprt: bool = True, use_auditor: bool = True, llm_mode: str = "hf", experiment_id: str = "", treatment: str = "", run: int = 0):
    logger.info(f"🚀 Starting AutoResearch-RL Perpetual Loop (Max Iterations: {max_iterations})")
    logger.info(f"Ablation settings -> Novelty: {use_novelty}, SPRT: {use_sprt}, Auditor: {use_auditor}")
    if experiment_id:
        logger.info(f"Experiment: {experiment_id} | Treatment: {treatment} | Run: {run}")
    logger.info(f"LLM Mode: {llm_mode}")

    # Setup experiment logging directory
    if experiment_id and treatment:
        exp_log_dir = f"experiments/{experiment_id}/data/{treatment}/run_{run:03d}"
        os.makedirs(exp_log_dir, exist_ok=True)
        exp_log_file = f"{exp_log_dir}/experiment_logs.jsonl"
        logger.info(f"Logging to: {exp_log_file}")
    else:
        exp_log_file = "experiment_logs.jsonl"

    # Quality Metrics & Cost Accounting
    metrics = {
        "total_patches_proposed": 0,
        "syntactically_valid": 0,
        "semantic_rollback": 0, # Failed execution or early aborts
        "accepted_sota": 0,
        "total_wall_clock_time_sec": 0.0,
    }

    # Initialize Core Components
    orchestrator = Orchestrator()
    agent = PPOMetaAgent(system_prompt="Minimize BPB under 16MB. No causality leaks. 10 min hard limit.")
    env = AutoResearchEnv(sota_bpb=1.0)

    # Load Golden Seed
    with open("seed/train_gpt.py", "r") as f:
        current_best_code = f.read()

    logger.info("Loaded Golden Seed.")

    # Import random patch generator for baseline mode
    if llm_mode == "none":
        from utils.random_patch_generator import RandomPatchGenerator
        logger.info("Baseline mode: Using random patch generator")

    iteration = 1

    while iteration <= max_iterations:
        logger.info(f"\n{'='*50}\n🔄 Starting Iteration {iteration}\n{'='*50}")

        # 1. Gather Telemetry (Mocked for now)
        telemetry = {
            "last_vram_peak_mb": 78000,
            "recent_oom": False,
            "iteration": iteration
        }

        # 2. Agent Action Phase (Policy Evaluation -> Action)
        if llm_mode == "none":
            # Baseline: Use random patch generator
            from utils.random_patch_generator import RandomPatchGenerator
            from agent.ppo_agent import ASTDiffParser
            
            patch = RandomPatchGenerator.generate_random_patch(current_best_code)
            try:
                candidate_code = ASTDiffParser.apply_patch(current_best_code, patch["search"], patch["replace"])
                logger.info(f"Random patch applied: {patch['search'][:30]}... → {patch['replace'][:30]}...")
            except Exception as e:
                logger.warning(f"Failed to apply random patch: {e}")
                candidate_code = current_best_code
        else:
            # HF Router mode: Use LLM
            candidate_code = agent.generate_action(current_best_code, env.history, telemetry)

        # 3. Security Audit (Causality)
        if use_auditor:
            causality_leak = check_causality_leak(candidate_code)
        else:
            causality_leak = False

        sprt = None
        job_id = f"job_iter_{iteration}"
        # Metrics Tracking
        metrics["total_patches_proposed"] += 1
        start_time = time.time()

        # 4. Orchestrator Pre-checks (AST & Capacity)
        if causality_leak:
            logger.warning("Causality Audit Failed. Skipping GPU dispatch.")
            # Create a mock result to feed the env the penalty
            result = orchestrator.submit_job(candidate_code)
            result.status = "ABORTED"
            result.error_message = "CausalityLeak"
            result.remediation = "Remove forward-looking data references (e.g. data[i+1:] or negative shift)."
            metrics["semantic_rollback"] += 1
        else:
            # Smoke & Capacity check via Orchestrator
            if not orchestrator.run_smoke_test(candidate_code):
                 # Submit job just to get the formatted error EvaluationResult
                 result = orchestrator.submit_job(candidate_code)
            else:
                metrics["syntactically_valid"] += 1
                if orchestrator.simulate_compression_and_capacity(candidate_code, num_parameters_int6=10_000_000, num_parameters_bf16=2_000_000) > 16_000_000:
                     result = orchestrator.submit_job(candidate_code)
                else:
                    # 5. Dispatch to GPU Cluster with SPRT filtering
                    sprt = SPRTFilter(sota_threshold=env.sota_bpb) if use_sprt else None

                    def sprt_callback(step: int, loss: float) -> bool:
                        if not use_sprt: return False
                        return sprt.update_and_check(step, loss)

                    use_docker = False if AUTORESEARCH_MODE == "LOCAL" else True
                    dispatcher = GPUDispatcher(use_docker=use_docker, sprt_callback=sprt_callback, time_limit_sec=MAX_TIME_SECONDS)

                    result = dispatcher.dispatch(job_id, candidate_code, num_parameters=12_000_000)

                    if result.status != "COMPLETED":
                        metrics["semantic_rollback"] += 1

        elapsed_time = time.time() - start_time
        metrics["total_wall_clock_time_sec"] += elapsed_time

        # 6. Environment Step (Calculate Reward & Update Memory)
        abort_step = 0
        uncertainty = 0.0
        if result.status == "ABORTED" and result.error_message == "SPRT_EARLY_STOPPING" and sprt:
             abort_step = len(sprt.step_history) * 10

        # Extract uncertainty from SPRT filter if completed successfully to bound the reward.
        # Clamp to a maximum of 5.0 BPB units as a defense-in-depth guard against any
        # residual numerical instability in the curve fit covariance.
        if result.status == "COMPLETED" and sprt and hasattr(sprt, "last_c_std_err"):
             uncertainty = min(getattr(sprt, "last_c_std_err", 0.0), 5.0)

        applied_patch = "MOCK_PATCH_APPLIED"
        patch_category = "hyperparameter_tuning" if "GPTConfig" in candidate_code else "architecture_mutation"

        step_info = env.step(result, action_patch=applied_patch, causality_leak=causality_leak, abort_step=abort_step, use_novelty=use_novelty, elapsed_time=elapsed_time, uncertainty=uncertainty, category=patch_category)

        logger.info(f"Iteration Result -> Status: {result.status}, BPB: {result.final_bpb}, Reward: {step_info['reward']:.4f}")

        # 7. UPDATE POLICY NETWORK (PPO Learning Step)
        agent.update_policy(step_info['reward'])

        log_experiment_json(
            iteration, job_id, applied_patch, result.status,
            result.final_bpb, step_info['reward'], env.history[-1]['components'],
            causality_leak, abort_step, result.remediation, patch_category,
            log_file=exp_log_file
        )

        # 8. Update SOTA and Artifacts
        if result.status == "COMPLETED" and result.final_bpb is not None and result.final_bpb < env.sota_bpb:
            logger.info("🏆 Candidate code accepted as new SOTA!")
            current_best_code = candidate_code
            save_artifact(current_best_code, result.final_bpb, iteration)
            metrics["accepted_sota"] += 1

        iteration += 1
        time.sleep(1)

    logger.info("🛑 Simulation Limit Reached. Terminating AutoResearch-RL Loop.")

    # Cost Accounting & Quality Metrics Report
    gpu_hours = metrics["total_wall_clock_time_sec"] / 3600.0 * 8 # Simulating 8xH100
    valid_rate = (metrics["syntactically_valid"] / max(1, metrics["total_patches_proposed"])) * 100
    rollback_rate = (metrics["semantic_rollback"] / max(1, metrics["total_patches_proposed"])) * 100
    accept_rate = (metrics["accepted_sota"] / max(1, metrics["total_patches_proposed"])) * 100

    logger.info("\n" + "="*50)
    logger.info("📊 AUTORESEARCH-RL METRICS & COST ACCOUNTING")
    logger.info("="*50)
    logger.info(f"Total Patches Proposed: {metrics['total_patches_proposed']}")
    logger.info(f"Syntactic Validity Rate: {valid_rate:.1f}%")
    logger.info(f"Semantic Rollback Rate: {rollback_rate:.1f}%")
    logger.info(f"Acceptance Rate (SOTA gains): {accept_rate:.1f}%")
    logger.info(f"Total Compute Cost (GPU-hours): {gpu_hours:.2f} (8xH100 equiv)")
    if metrics["accepted_sota"] > 0:
        logger.info(f"Cost per SOTA Gain: {gpu_hours / metrics['accepted_sota']:.2f} GPU-hours")
        logger.info(f"Avg Iterations per SOTA Gain: {metrics['total_patches_proposed'] / metrics['accepted_sota']:.1f}")
    logger.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoResearch-RL Perpetual Loop")
    parser.add_argument("--max_iterations", type=int, default=1, help="Max iterations to run")
    parser.add_argument("--no_novelty", action="store_true", help="Disable novelty bonus")
    parser.add_argument("--no_sprt", action="store_true", help="Disable SPRT early stopping")
    parser.add_argument("--no_auditor", action="store_true", help="Disable causality auditor")
    parser.add_argument("--llm_mode", type=str, default="hf", choices=["hf", "none"], 
                        help="LLM mode: 'hf' for HF router, 'none' for random baseline")
    parser.add_argument("--experiment_id", type=str, default="", help="Experiment ID for tracking")
    parser.add_argument("--treatment", type=str, default="", help="Treatment: baseline or hf_router")
    parser.add_argument("--run", type=int, default=0, help="Run number (for multi-run experiments)")
    args = parser.parse_args()

    run_perpetual_loop(
        max_iterations=args.max_iterations,
        use_novelty=not args.no_novelty,
        use_sprt=not args.no_sprt,
        use_auditor=not args.no_auditor,
        llm_mode=args.llm_mode,
        experiment_id=args.experiment_id,
        treatment=args.treatment,
        run=args.run,
    )
