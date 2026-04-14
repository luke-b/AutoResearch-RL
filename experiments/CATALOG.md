# AutoResearch-RL Experiments Catalog

Central index of all experiments, their status, and key findings.

---

## Active Experiments

(None in progress)

---

## Completed Experiments

### 1. LLM Capacity & Architecture Improvement Quality
- **ID**: `exp-2026-04-14-001-llm-capacity`
- **Status**: ❌ **INCONCLUSIVE** (Phase 1 Complete)
- **Start Date**: 2026-04-14
- **Completion Date**: 2026-04-14
- **Directory**: `./exp-2026-04-14-001-llm-capacity/`
- **Goal**: Measure HF LLM contribution to architecture improvements vs. random baselines
- **Phase**: Phase 1 (Baseline vs LLM Comparison)
- **Owner**: Luke B

**Key Results**:
- Random Baseline: 0% acceptance (0/133 iterations)
- HF Router LLM: 0% acceptance (0/101 iterations)
- **Statistical Comparison**: p=0.98, Cohen's d=-0.003 **(NOT significant)**
- **Conclusion**: HF ≈ Random in mock environment

**Root Cause**:
Mock GPU dispatcher (AUTORESEARCH_MODE=LOCAL) doesn't differentiate patch quality. Both treatments fail equally because the environment doesn't run real training or compute meaningful BPB feedback.

**Recommendation**:
- ❌ Phase 2 ablations will also show null results in mock environment
- ✅ Plan Phase 1 Redux on GPU cluster with real Qwen + actual training
- ✅ This experiment validated infrastructure (experiment framework works correctly)

**Key Findings**:
1. Experiment infrastructure operational (logging, analysis, statistics)
2. Both treatments treated identically by mock dispatcher
3. No reward signal to differentiate patch quality (avg reward ~0.0017-0.0018 both treatments)
4. Acceptance rate 0.0% both → neither patch type actually evaluated

**Documentation**: [design.md](./exp-2026-04-14-001-llm-capacity/design.md) | [results.md](./exp-2026-04-14-001-llm-capacity/results.md) | [data/](./exp-2026-04-14-001-llm-capacity/data/)

---

## Experiment Lifecycle

1. **Planned**: Experiment designed but not yet started
2. **In Progress**: Data collection / execution underway
3. **Analysis**: Data collected, analysis in progress
4. **Completed**: Final results documented
5. **Archived**: Old experiments moved to `/archived/` after 6 months

---

## Quick Stats

| | Count |
|---|---|
| Total Experiments | 1 |
| In Progress | 0 |
| Completed | 1 |
| Inconclusive | 1 |
| Archived | 0 |

---

## Directory Structure Reference

```
experiments/
├── CATALOG.md                              (this file)
├── TEMPLATE.md                             (experiment template)
├── exp-2026-04-14-001-llm-capacity/
│   ├── README.md                           (experiment overview)
│   ├── design.md                           (detailed methodology)
│   ├── data/
│   │   ├── baseline/
│   │   │   ├── run_001_experiment_logs.jsonl
│   │   │   └── run_001_metrics.json
│   │   └── hf_router/
│   │       ├── run_001_experiment_logs.jsonl
│   │       └── run_001_metrics.json
│   ├── analysis/
│   │   ├── metrics_summary.ipynb
│   │   ├── plots/
│   │   │   ├── bpb_convergence.png
│   │   │   └── acceptance_rate_comparison.png
│   │   └── statistical_tests.py
│   └── results.md                          (key findings & conclusions)
└── archived/
    └── (old experiments > 6 months)
```

---

## How to Create a New Experiment

1. Copy [TEMPLATE.md](./TEMPLATE.md)
2. Create new directory: `exp-YYYY-MM-DD-###-experiment-name/`
3. Add to this CATALOG with status "Planned"
4. Follow the template structure
5. Update catalog when moving through phases

See [TEMPLATE.md](./TEMPLATE.md) for full instructions.

