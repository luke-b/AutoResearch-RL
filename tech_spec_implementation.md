# Technical Specification: Dual-Mode Architecture for AutoResearch-RL

## 1. Objective
To implement the Steering Scientific Committee's vision of supporting both High-Compute (Cluster) and Low-Cost (Local/Validation) modes without regressing the Golden Seed baseline. This requires non-destructive, dynamic configuration overrides across the Orchestrator, PPO Agent, and the `train_gpt.py` seed.

## 2. Global Execution Profile
We will introduce a global environment variable, `AUTORESEARCH_MODE`, which can be set to either `CLUSTER` (default) or `LOCAL`. All components will read this variable to adjust their behavior.

## 3. Code-Level Implementation Strategy

### 3.1. `seed/train_gpt.py` (Golden Seed)
**Goal:** Allow the seed to scale down its parameters dynamically without permanently modifying the default `GPTConfig`.
**Changes:**
1.  **Parse Environment Variable:** Add `import os` and read `os.environ.get("AUTORESEARCH_MODE", "CLUSTER").upper()`.
2.  **Dynamic Config Override:** After instantiating `config = GPTConfig()`, apply overrides if `MODE == 'LOCAL'`:
    ```python
    if os.environ.get("AUTORESEARCH_MODE", "CLUSTER").upper() == "LOCAL":
        config.n_embd = 128
        config.n_head = 4
        config.depth_loops = 1
        config.mlp_expansion = 1
        config.block_size = 256
        # Adjust training steps for faster local runs
        total_steps = 50
        eval_stride = 128
    ```
    *Note: The PPO Agent modifies `GPTConfig` via AST parsing. By placing the override logic *after* the `GPTConfig` class definition and default instantiation, the Agent can still mutate the base class, and the local run will just downscale those mutational baselines.*

### 3.2. `orchestrator/orchestrator.py` & `orchestrator/docker_runner.py`
**Goal:** Adjust timeouts and execution contexts dynamically.
**Changes in `orchestrator.py`:**
1.  Read `AUTORESEARCH_MODE`.
2.  Modify the `MAX_TIME_SECONDS` constant dynamically:
    ```python
    MAX_TIME_SECONDS = 1800 if MODE == "LOCAL" else 600
    ```
3.  *Optional but recommended:* When estimating artifacts, the Orchestrator assumes a large model. If `LOCAL`, we could scale down `num_parameters_int6` in the `simulate_artifact_size` function, though keeping the 16MB limit strict is preferred to catch true limit violations.

**Changes in `main.py` (Entrypoint):**
1.  Read `AUTORESEARCH_MODE`.
2.  When initializing `GPUDispatcher`, pass `use_docker=False` if `MODE == 'LOCAL'`, and `use_docker=True` if `MODE == 'CLUSTER'`.
    ```python
    use_docker = False if os.environ.get("AUTORESEARCH_MODE", "CLUSTER").upper() == "LOCAL" else True
    dispatcher = GPUDispatcher(use_docker=use_docker, time_limit_sec=MAX_TIME_SECONDS)
    ```

### 3.3. `agent/ppo_agent.py` (LLM Provider Interface)
**Goal:** Modularize the LLM backend to easily support Local, Mock, or OpenAI providers.
**Changes:**
1.  **Extract Provider Logic:** Create a base class `LLMProvider` with a method `generate_patch(prompt)`.
2.  **Implement Providers:**
    *   `OpenAIProvider(api_key)`: Uses the existing `openai.ChatCompletion` logic.
    *   `MockProvider()`: Returns the existing hardcoded JSON diff.
    *   `LocalModelProvider(endpoint)`: Uses `requests.post` to hit a local HTTP endpoint (e.g., LM Studio or Ollama) using OpenAI-compatible payload structures.
3.  **Agent Integration:** Modify `PPOMetaAgent` to accept an `LLMProvider` instance upon initialization rather than hardcoding the API call. `main.py` will configure and inject the correct provider based on `AUTORESEARCH_MODE` and available environment variables.

### 3.4. Continuous Integration (`.github/workflows/ci.yml`)
**Goal:** Ensure the Validation mode is actively tested.
**Changes:**
1.  Ensure the CI workflow explicitly sets `AUTORESEARCH_MODE=LOCAL`.
2.  Add a step to run `main.py` with `--max_iterations 1` to guarantee a full end-to-end loop (including Agent patch generation, Orchestrator dispatch, and SPRT filtering) executes successfully on GitHub Actions runners.

## 4. Rollout Plan
1.  **Phase 1:** Implement the `AUTORESEARCH_MODE` parser and dynamic overrides in `train_gpt.py`. Add unit tests to verify the config scales down properly.
2.  **Phase 2:** Update `orchestrator.py` and `main.py` to handle dynamic timeouts and the `use_docker` flag based on the profile.
3.  **Phase 3:** Refactor `ppo_agent.py` to use the `LLMProvider` interface.
4.  **Phase 4:** Update CI to run an end-to-end local test.

By executing this strategy, the engineering team will deliver a flexible framework capable of cheap local debugging while preserving the rigorous constraints required for the high-compute cluster environment.
