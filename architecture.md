# AutoResearch-RL Directory Structure & Architecture Plan

This document outlines the directory structure and main modules for the AutoResearch-RL framework.

## Directory Structure

```
/app
├── orchestrator/
│   ├── orchestrator.py    # CPU Orchestrator: manages MDP cycle, smoke tests, limits, job dispatch
│   └── api_doc.md         # API Specification between Orchestrator and GPU Cluster
├── agent/
│   ├── ppo_agent.py       # PPO Meta-Agent: generates code mutations
│   └── mdp_env.py         # The MDP Environment wrapper for the agent
├── gpu_cluster/
│   └── sprt.py            # Power-Law SPRT Filter: early stopping for GPU runs
├── auditor/
│   └── causality_auditor.py # Security layer checking for causality leaks in generated code
└── seed/
    └── train_gpt.py       # The Golden Seed: initial starting point for model architecture
```

## Architecture Layers

**Modul A1: CPU Orchestrator (`orchestrator/orchestrator.py`)**
Runs on a CPU machine to handle AST verification ("smoke tests"), simulate 16MB limit checks (zstd compression and int6 bounds), and dispatch valid candidates to GPU nodes.

**Modul A2: PPO Meta-Agent (`agent/ppo_agent.py` & `agent/mdp_env.py`)**
Serves as the decision-maker, observing state (current best code, memory of past $K=32$ experiments, constraints, runtime telemetry) to propose a structural action (a precise diff patch) optimized by a multi-objective reward function including novelty.

**Modul A3: Power-Law SPRT Filter (`gpu_cluster/sprt.py`)**
A mechanism to intercept the training curve early, aborting runs that statistically will not surpass the SOTA threshold in the 10-minute time constraint.

**Modul B: The Golden Seed (`seed/train_gpt.py`)**
The ultimate base state ($c_{base}$) encompassing Int6 Quantization, Depth Recurrence, learned QK-Gain scaling, Sliding Window Evaluation, Swarm Causal Backoff N-gram Mixer, and the Muon Optimizer.

**Causality Auditor (`auditor/causality_auditor.py`)**
Statically and dynamically checks the generated diff patches to ensure there is no future-looking (forward) test set data spillage.
