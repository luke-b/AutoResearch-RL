# API Specification: Orchestrator <-> GPU Cluster

This document outlines the communication interface between the CPU Orchestrator and the GPU Evaluation Nodes in the AutoResearch-RL framework.

## 1. Orchestrator to GPU Node (Job Submission)

The Orchestrator sends training jobs to the GPU nodes. This payload contains the full target source code and environment specifications.

**Payload Format (JSON over TCP/HTTP or via Docker Run Arguments):**
```json
{
  "job_id": "string (UUID)",
  "code_payload": "string (Base64 encoded or raw text of train_gpt.py)",
  "hyperparameters": {
    "learning_rate": "float",
    "batch_size": "int",
    "max_steps": "int"
  },
  "constraints": {
    "max_time_seconds": 600,
    "max_memory_mb": 16000
  }
}
```

## 2. GPU Node to Orchestrator (Telemetry & SPRT Checkpoints)

During training, the GPU node periodically (e.g., every 50 iterations) streams telemetry data back to the Orchestrator for SPRT evaluation and monitoring.

**Payload Format (JSON Stream):**
```json
{
  "job_id": "string",
  "step": "int",
  "loss": "float",
  "bpb": "float",
  "elapsed_time_seconds": "float",
  "gpu_vram_usage_mb": "float"
}
```

## 3. Orchestrator to GPU Node (Control Signals)

The Orchestrator can send control signals, primarily the `ABORT` signal triggered by the SPRT filter if the training curve is unpromising.

**Payload Format (JSON):**
```json
{
  "job_id": "string",
  "action": "ABORT",
  "reason": "SPRT_EARLY_STOPPING"
}
```

## 4. GPU Node to Orchestrator (Final Result/Completion)

Upon successful completion or failure, the GPU node returns a final state package.

**Payload Format (JSON):**
```json
{
  "job_id": "string",
  "status": "COMPLETED | FAILED | ABORTED",
  "final_bpb": "float",
  "total_time_seconds": "float",
  "artifact_size_bytes": "int",
  "error_trace": "string (if FAILED)"
}
```
