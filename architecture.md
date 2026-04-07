# AutoResearch-RL Directory Structure & Architecture Plan

This document outlines the directory structure and main modules for the AutoResearch-RL framework.

## Directory Structure

```
/app
├── main.py                # The Perpetual Research Loop tying all components together
├── Dockerfile.cuda        # Container configuration for 8xH100 NVLink environments
├── README.md              # Project overview, architecture diagrams, and operational guide
├── REPORT.md              # Detailed implementation status report
├── architecture.md        # This file: Directory structure and high-level architecture
├── experiment_logs.jsonl  # Generated file: Structured telemetry output from main.py
├── orchestrator/
│   ├── orchestrator.py    # CPU Orchestrator: manages MDP cycle, smoke tests, limits
│   ├── docker_runner.py   # GPU Dispatcher: runs subprocesses or nvidia-docker clusters
│   └── api_doc.md         # API Specification between Orchestrator and GPU Cluster
├── agent/
│   ├── ppo_agent.py       # PPO Meta-Agent & DiffParser: interacts with OpenAI API and patches code
│   └── mdp_env.py         # The MDP Environment wrapper calculating the multi-objective reward
├── gpu_cluster/
│   └── sprt.py            # Power-Law SPRT Filter: early stopping for GPU runs using covariance bounds
├── auditor/
│   └── causality_auditor.py # Security layer checking for causality leaks in generated code
├── seed/
│   └── train_gpt.py       # The Golden Seed: highly optimized baseline model with GPTConfig
└── tests/                 # Automated pytest suite
    ├── test_auditor.py
    ├── test_diff_parser.py
    ├── test_orchestrator.py
    └── test_sprt.py
```

## Architecture Layers

**Modul A1: CPU Orchestrator (`orchestrator/orchestrator.py` & `orchestrator/docker_runner.py`)**
Runs on a CPU machine to handle AST verification ("smoke tests"), simulate 16MB limit checks (zstd compression and heterogeneous BF16/Int6 parameter estimation), and dispatch valid candidates to isolated subprocess environments or full CUDA docker containers via the `GPUDispatcher`.
It utilizes a **Dual-Mode Architecture** (`AUTORESEARCH_MODE=LOCAL|CLUSTER`) to seamlessly toggle between local development settings (relaxed timeouts, disabled Docker execution) and strict 10-minute high-compute evaluations. The Orchestrator defines explicit `remediation` rules for failures (`SyntaxError`, `CapacityLimitExceeded`), while the `GPUDispatcher` implements a rigid 60-second heartbeat monitor for runtime anomalies.

**Modul A2: PPO Meta-Agent (`agent/ppo_agent.py` & `agent/mdp_env.py`)**
Serves as the decision-maker, observing a 13D state vector (current best code, parsed hyperparameters, SOTA BPB, experiment abort history) to propose a structural action. An extensible `LLMProvider` interface supports OpenAI APIs, local LLM endpoints, and Mock providers seamlessly.
The `ASTDiffParser` robustly applies these mutations by structurally matching `ast` nodes, falling back to whitespace-insensitive matching when necessary. The agent uses batched generalized advantage estimation (GAE) with multi-epoch PPO updates.
The MDP Environment calculates the dynamic multi-objective reward function including scalable novelty, elapsed wall-clock compute cost, and discounts observed SOTA improvements by the SPRT's projection uncertainty.

**Modul A3: Power-Law SPRT Filter (`gpu_cluster/sprt.py`)**
A mechanism to intercept the training curve early. Uses `scipy.optimize.curve_fit` to extract covariance matrices, forming strict confidence intervals to safely abort runs that statistically will not surpass the SOTA threshold in the 10-minute time constraint. Passes explicit uncertainty metrics back to the environment.

**Modul B: The Golden Seed (`seed/train_gpt.py`)**
The ultimate base state ($c_{base}$). Consolidates tunable features inside a `GPTConfig` block. Encompasses Int6 Quantization, Depth Recurrence, Sliding Window Evaluation, and the Muon Optimizer with SWA/EMA warmdown.
Dynamically asserts causality by registering forward hooks to `QKGainAttention` that prove invariant predictions when perturbing future input tokens. Enforces deterministic benchmarking via the `AUTORESEARCH_SEED` environment variable.

**Causality Auditor (`auditor/causality_auditor.py`)**
Statically and recursively checks the generated diff patches to ensure there is no future-looking (forward) test set data spillage (robust against nested slices, tricky indexing, and adversarial `data.roll` attacks).
