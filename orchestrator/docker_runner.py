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
    def __init__(self, sprt_callback: Callable[[int, float], bool], time_limit_sec: int = 600, use_docker: bool = False):
        """
        Args:
            sprt_callback: A function `fn(step, loss) -> bool` that returns True if the run should be aborted.
            time_limit_sec: Max allowed wall-clock time.
            use_docker: If True, dispatches to a real nvidia-docker container instead of a local python subprocess.
        """
        self.sprt_callback = sprt_callback
        self.time_limit_sec = time_limit_sec
        self.use_docker = use_docker
        self._active_processes = {}

    def _write_script_to_file(self, job_id: str, source_code: str) -> str:
        file_path = f"/tmp/autoresearch_job_{job_id}.py"
        with open(file_path, "w") as f:
            f.write(source_code)
        return file_path

    def dispatch(self, job_id: str, source_code: str, num_parameters: int) -> EvaluationResult:
        """
        Launches the training job, monitors it via the SPRT callback, and enforces constraints.
        This function blocks until completion or abort.
        """
        logger.info(f"Dispatching Job {job_id} to GPU Node (Docker: {self.use_docker})...")

        script_path = self._write_script_to_file(job_id, source_code)

        if self.use_docker:
            # Construct a real Docker command for an 8xH100 node with NVLink
            cmd = [
                "docker", "run", "--rm",
                "--gpus", "all",
                "--ipc=host", # Required for NCCL/NVLink
                "-v", f"{script_path}:/workspace/train_job.py",
                "autoresearch-rl-node:latest" # Image built from Dockerfile.cuda
            ]
        else:
            # Fallback to local subprocess execution
            cmd = ["python3", script_path]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        self._active_processes[job_id] = process

        # We need a robust, non-blocking way to read output and enforce timeouts.
        # If the candidate code enters an infinite loop without printing, a simple readline blocks forever.
        # Thus, we put the I/O read in a separate thread and use process.wait(timeout) in the main thread.
        result_box = {
            "final_bpb": None,
            "status": "FAILED",
            "error_message": None
        }

        def stream_reader():
            try:
                for line in iter(process.stdout.readline, ''):
                    # Fast abort check if another thread killed the process
                    if process.poll() is not None:
                        break

                    try:
                        data = json.loads(line.strip())

                        if "step" in data and "loss" in data:
                            should_abort = self.sprt_callback(data["step"], data["loss"])
                            if should_abort:
                                logger.info(f"Job {job_id} aborted by SPRT Filter.")
                                process.terminate()
                                result_box["status"] = "ABORTED"
                                result_box["error_message"] = "SPRT_EARLY_STOPPING"
                                break

                        if "status" in data and data["status"] == "completed":
                            result_box["final_bpb"] = data.get("final_bpb")
                            result_box["status"] = "COMPLETED"

                    except json.JSONDecodeError:
                        pass # Not JSON
            except Exception as e:
                logger.error(f"Error reading stream for job {job_id}: {e}")

        # Start reading the stream in a background thread
        reader_thread = threading.Thread(target=stream_reader)
        reader_thread.daemon = True
        reader_thread.start()

        try:
            # Main thread strictly enforces the wall-clock timeout regardless of stdout blocking
            process.wait(timeout=self.time_limit_sec)
        except subprocess.TimeoutExpired:
            logger.warning(f"Job {job_id} hit Wall-Clock Time Limit ({self.time_limit_sec}s). Killing.")
            process.terminate()
            process.wait() # Wait for it to actually die
            result_box["status"] = "ABORTED"
            result_box["error_message"] = "TimeLimitExceeded"
        finally:
            # Ensure resources are cleaned up
            if process.stdout:
                process.stdout.close()
            reader_thread.join(timeout=1.0)

            # Cleanup
            if os.path.exists(script_path):
                os.remove(script_path)

            del self._active_processes[job_id]

        if process.returncode != 0 and result_box["status"] not in ["ABORTED", "COMPLETED"]:
             result_box["status"] = "FAILED"
             result_box["error_message"] = f"Process crashed with code {process.returncode}"

        return EvaluationResult(
            job_id=job_id,
            status=result_box["status"],
            final_bpb=result_box["final_bpb"],
            artifact_size=15_000_000, # Handled by orchestrator limit simulation
            error_message=result_box["error_message"]
        )
