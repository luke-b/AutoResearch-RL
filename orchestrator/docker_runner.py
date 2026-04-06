import logging
import uuid
import json
import threading
import time
from typing import Callable, Optional
import os
import subprocess
import sys

# Ensure we can import EvaluationResult
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator.orchestrator import EvaluationResult

logger = logging.getLogger("GPU_Dispatcher")

class GPUDispatcher:
    """
    Handles the actual dispatch of the generated script to an isolated execution environment
    (e.g., a Docker container simulating an 8xH100 node).
    """
    def __init__(self, sprt_callback: Callable[[int, float], bool], time_limit_sec: int = 600):
        """
        Args:
            sprt_callback: A function `fn(step, loss) -> bool` that returns True if the run should be aborted.
            time_limit_sec: Max allowed wall-clock time.
        """
        self.sprt_callback = sprt_callback
        self.time_limit_sec = time_limit_sec
        self._active_processes = {}

    def _write_script_to_file(self, job_id: str, source_code: str) -> str:
        # In reality, this might be written to a shared NFS or directly piped into Docker
        file_path = f"/tmp/autoresearch_job_{job_id}.py"
        with open(file_path, "w") as f:
            f.write(source_code)
        return file_path

    def dispatch(self, job_id: str, source_code: str, num_parameters: int) -> EvaluationResult:
        """
        Launches the training job, monitors it via the SPRT callback, and enforces constraints.
        This function blocks until completion or abort.
        """
        logger.info(f"Dispatching Job {job_id} to GPU Node...")

        script_path = self._write_script_to_file(job_id, source_code)

        # We execute the actual candidate script via subprocess
        # In a real environment, this would build a container and run it on a GPU node
        process = subprocess.Popen(
            ["python3", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        self._active_processes[job_id] = process

        final_bpb = None
        status = "FAILED"
        error_message = None

        start_time = time.time()

        try:
            for line in iter(process.stdout.readline, ''):
                if time.time() - start_time > self.time_limit_sec:
                    logger.warning(f"Job {job_id} hit Wall-Clock Time Limit ({self.time_limit_sec}s). Killing.")
                    process.terminate()
                    status = "ABORTED"
                    error_message = "TimeLimitExceeded"
                    break

                try:
                    data = json.loads(line.strip())

                    if "step" in data and "loss" in data:
                        should_abort = self.sprt_callback(data["step"], data["loss"])
                        if should_abort:
                            logger.info(f"Job {job_id} aborted by SPRT Filter.")
                            process.terminate()
                            status = "ABORTED"
                            error_message = "SPRT_EARLY_STOPPING"
                            break

                    if "status" in data and data["status"] == "completed":
                        final_bpb = data.get("final_bpb")
                        status = "COMPLETED"

                except json.JSONDecodeError:
                    # Not JSON output, maybe a crash trace
                    pass

        finally:
            process.stdout.close()
            process.wait()

            # Cleanup
            if os.path.exists(script_path):
                os.remove(script_path)

            del self._active_processes[job_id]

        if process.returncode != 0 and status not in ["ABORTED", "COMPLETED"]:
             status = "FAILED"
             error_message = f"Process crashed with code {process.returncode}"

        return EvaluationResult(
            job_id=job_id,
            status=status,
            final_bpb=final_bpb,
            artifact_size=15_000_000, # Handled by orchestrator limit simulation
            error_message=error_message
        )
