# AutoResearch-RL Agent Instructions

AutoResearch-RL is an autonomous research pipeline designed for the Parameter-Golf challenge. The framework operates a loop where a meta-agent proposes code patches to a seed model (`train_gpt.py`), runs a short training job to assess the improvement in the validation bits-per-byte (BPB) metric, and computes a reward.

## Setup
- Install dependencies: `pip install -r codex_agent/requirements.txt`
- Run tests: `./codex_agent/ci.sh` (or `pytest -q tests/` from repo root)

## Directory Layout
- `main.py` – Entry point that runs the perpetual loop.
- `agent/ppo_agent.py` – Implements a PPO meta-agent. Uses an AST-based DiffParser to apply modifications.
- `agent/mdp_env.py` – Maintains the reinforcement learning environment, recording history and computing rewards.
- `auditor/causality_auditor.py` – Performs static AST analysis to detect forward-looking operations.
- `gpu_cluster/sprt.py` – Implements a sequential probability ratio test (SPRT) that fits a power-law learning curve to abort unpromising runs early.
- `seed/train_gpt.py` – Defines the golden seed model architecture, training loop, and evaluation.
- `tests/` – Contains unit tests for validation.

## Running experiments
- Execute the perpetual loop using the wrapper: `./codex_agent/run_autoresearch.sh`
- Alternatively, run manually from repo root: `python3 main.py --max_iterations=1`
- Check that the script writes telemetry to `experiment_logs.jsonl` and saves improved candidates in `artifacts/`.
- Use the `MAX_ITERATIONS` environment variable in `codex_agent/.env` or `--max_iterations` argument to limit the loop for debugging.

## Code style and constraints
- Use the AST-based diff parser (see `agent/ppo_agent.py`) when proposing patches. Avoid naive string replacements that may break the code.
- Do not increase the compressed artefact size beyond 16 MB. Monitor the `artifact_size` returned by the orchestrator.
- Never introduce forward-looking data access (e.g., negative indexing or slicing beyond current time). Use `auditor/causality_auditor.py` to verify.
- Keep the training loop within the 10-minute time budget and avoid downloading datasets or models.
- Follow existing naming, formatting and architectural patterns; seek guidance in `README.md` and source code.

## Success criteria
- All tests pass.
- The candidate improves validation BPB or provides a novel architecture that could be promising.
- No abort due to causality violation or SPRT early stopping without progress.
- The agent logs a clear summary of actions and outcomes in `experiment_logs.jsonl`.
