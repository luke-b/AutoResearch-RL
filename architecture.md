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

**Modul A2: PPO Meta-Agent (`agent/ppo_agent.py` & `agent/mdp_env.py`)**
Serves as the decision-maker, observing state (current best code, memory of past $K=32$ experiments, constraints, runtime telemetry) to propose a structural action via OpenAI APIs. The `DiffParser` robustly applies these mutations using whitespace-insensitive matching. The MDP Environment calculates the dynamic multi-objective reward function including scalable novelty and late-abort penalties.

**Modul A3: Power-Law SPRT Filter (`gpu_cluster/sprt.py`)**
A mechanism to intercept the training curve early. Uses `scipy.optimize.curve_fit` to extract covariance matrices, forming strict confidence intervals to safely abort runs that statistically will not surpass the SOTA threshold in the 10-minute time constraint. Includes plateau detection.

**Modul B: The Golden Seed (`seed/train_gpt.py`)**
The ultimate base state ($c_{base}$). Consolidates tunable features inside a `GPTConfig` block. Encompasses Int6 Quantization, Depth Recurrence, learned QK-Gain scaling, Sliding Window Evaluation, dynamic causality runtime assertions, and the Muon Optimizer with SWA/EMA warmdown.

**Causality Auditor (`auditor/causality_auditor.py`)**
Statically and recursively checks the generated diff patches to ensure there is no future-looking (forward) test set data spillage. Complemented by runtime assertions in the seed evaluation loops.
