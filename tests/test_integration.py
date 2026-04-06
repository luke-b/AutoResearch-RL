import pytest
import sys
import os
import json
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from main
from main import run_perpetual_loop

def test_end_to_end_loop(monkeypatch, caplog):
    """
    Integration test to run a 2-iteration perpetual loop.
    Validates that:
    1. The loop executes without crashing.
    2. The 'experiment_logs.jsonl' is created and populated.
    3. PPO Agent updates weights without NaN crashes.
    """
    # Clean up old logs/artifacts if present
    if os.path.exists("experiment_logs.jsonl"):
        os.remove("experiment_logs.jsonl")
    if os.path.exists("artifacts"):
        import shutil
        shutil.rmtree("artifacts")

    # Capture logs at INFO level
    caplog.set_level(logging.INFO)

    # 2. Execute Loop
    # We will use the mock LLM output because we don't have OPENAI_API_KEY in CI.
    # The GPU dispatcher will run the training simulation script.
    run_perpetual_loop(max_iterations=2)

    # 3. Assertions
    # Check if simulation completed naturally
    assert "Simulation Limit Reached. Terminating AutoResearch-RL Loop." in caplog.text

    # Check PPO updates occurred
    assert "PPO Policy Updated" in caplog.text

    # Check JSON structured logging
    assert os.path.exists("experiment_logs.jsonl")

    with open("experiment_logs.jsonl", "r") as f:
        lines = f.readlines()
        assert len(lines) == 2 # 2 iterations = 2 log lines

        # Verify JSON schema of the log
        last_log = json.loads(lines[-1])
        assert "iteration" in last_log
        assert last_log["iteration"] == 2
        assert "reward" in last_log
        assert "reward_components" in last_log
        assert "patch" in last_log

    # Since we are running real execution with randomly initialized weights,
    # the BPB will be high (e.g. 100+), meaning it won't beat the 1.0 sota_threshold baseline.
    # Therefore, we just check that the loop processed the result and stored a non-NaN final_bpb.
    assert last_log["final_bpb"] is not None
    assert last_log["final_bpb"] > 0
