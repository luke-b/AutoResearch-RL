# AutoResearch-RL: Dual-Mode Architecture Implementation Plan

*(Status: ✅ Completed)*

This document outlines the epics and user stories required to implement the Dual-Mode Architecture (Cluster/Local) as defined in the `tech_spec_implementation.md` specification. All stories and epics below have been successfully executed and tested.

## Epic 1: Enable Dynamic Local Mode in the Golden Seed (`train_gpt.py`)
**Objective:** Allow the seed to scale down parameters dynamically for local execution without destructively modifying the default `GPTConfig` parameters used for the high-compute cluster. This ensures the PPO Agent can still mutate the original base class parameters during local testing.

*   **Story 1.1: Environment Variable Integration in Seed**
    *   **Description:** Update `seed/train_gpt.py` to securely read a global `AUTORESEARCH_MODE` environment variable.
    *   **Acceptance Criteria:**
        *   `os` module is imported at the start of the script.
        *   A constant or variable captures `os.environ.get("AUTORESEARCH_MODE", "CLUSTER").upper()`.
*   **Story 1.2: Dynamic `GPTConfig` Downscaling**
    *   **Description:** Apply configuration overrides immediately after the base `GPTConfig` class is instantiated to simulate a smaller network for local execution.
    *   **Acceptance Criteria:**
        *   If `MODE == 'LOCAL'`, override `n_embd` to 128, `n_head` to 4, `depth_loops` to 1, `mlp_expansion` to 1, and `block_size` to 256 on the config instance.
*   **Story 1.3: Training Loop Adjustments for Local Mode**
    *   **Description:** Reduce the number of training steps and evaluation overhead in local mode to ensure fast iteration.
    *   **Acceptance Criteria:**
        *   If `MODE == 'LOCAL'`, override `total_steps` (e.g., to 50) and `config.eval_stride` (e.g., to 128) just before the training loop starts.
        *   Verify that the local mode successfully completes a sliding window evaluation without exceptions.
*   **Story 1.4: Fix Sliding Window Evaluation Context Size Bug**
    *   **Description:** The sliding window evaluation function currently hardcodes `context_size=2048`. In local mode with a reduced block size (256), this causes an `IndexError` during positional embedding lookups.
    *   **Acceptance Criteria:**
        *   Update `sliding_window_eval` signature to use `context_size=config.block_size` instead of hardcoding `2048`.

## Epic 2: Dynamic Execution Timers and Orchestrator Modalities
**Objective:** Propagate the `AUTORESEARCH_MODE` profile to orchestrator components to dynamically adjust timeouts and Docker usage.

*   **Story 2.1: Dynamic Timeout in Orchestrator**
    *   **Description:** Adjust the global `MAX_TIME_SECONDS` constant based on the active mode in `orchestrator/orchestrator.py`.
    *   **Acceptance Criteria:**
        *   Read `AUTORESEARCH_MODE` in `orchestrator.py`.
        *   Set `MAX_TIME_SECONDS = 1800 if MODE == "LOCAL" else 600`.
*   **Story 2.2: Dynamic GPU Dispatcher Instantiation**
    *   **Description:** Update the main entrypoint (`main.py`) to toggle Docker usage and propagate the dynamic timeout based on the mode.
    *   **Acceptance Criteria:**
        *   Read `AUTORESEARCH_MODE` in `main.py`.
        *   Set a `use_docker` flag to `False` if `LOCAL`, otherwise `True`.
        *   Pass `use_docker` and the dynamically imported `MAX_TIME_SECONDS` to the `GPUDispatcher` initialization.

## Epic 3: PPO Agent LLM Provider Modularity
**Objective:** Modularize the `PPOMetaAgent` to support multiple LLM backends (OpenAI, Mock, Local Model) seamlessly via an injectable interface.

*   **Story 3.1: Define `LLMProvider` Interface**
    *   **Description:** Create a base class or interface for LLM patch generation.
    *   **Acceptance Criteria:**
        *   Create an `LLMProvider` class with a `generate_patch(prompt, temperature)` method.
*   **Story 3.2: Implement Specific Providers**
    *   **Description:** Implement the provider interface for OpenAI, a hardcoded Mock response, and a generic local HTTP endpoint.
    *   **Acceptance Criteria:**
        *   Implement `OpenAIProvider(api_key)` utilizing the existing `openai.ChatCompletion` logic.
        *   Implement `MockProvider()` returning a static JSON diff.
        *   Implement `LocalModelProvider(endpoint)` utilizing `requests` to hit a local OpenAI-compatible endpoint.
*   **Story 3.3: Refactor `PPOMetaAgent`**
    *   **Description:** Modify the agent to accept a provider instance instead of tightly coupling to OpenAI.
    *   **Acceptance Criteria:**
        *   `PPOMetaAgent.__init__` accepts a `provider` argument.
        *   `generate_action` utilizes `self.provider.generate_patch(...)`.
*   **Story 3.4: Wire up Provider Selection in Main Loop**
    *   **Description:** Configure the correct provider in `main.py` based on available environment variables.
    *   **Acceptance Criteria:**
        *   Check for `OPENAI_API_KEY` and `LOCAL_LLM_ENDPOINT` environment variables.
        *   Instantiate the appropriate provider (falling back to Mock).
        *   Inject the provider into the `PPOMetaAgent` instantiation.

## Epic 4: Continuous Integration and Verification
**Objective:** Ensure the newly created Validation/Local mode is actively tested in the GitHub Actions CI pipeline.

*   **Story 4.1: CI End-to-End Local Test**
    *   **Description:** Add a step to the CI workflow to execute a full pass of the perpetual loop in local mode.
    *   **Acceptance Criteria:**
        *   Update `.github/workflows/ci.yml`.
        *   Add a step that exports `AUTORESEARCH_MODE=LOCAL` and runs `python main.py --max_iterations 1`.
        *   Ensure the pipeline succeeds.
