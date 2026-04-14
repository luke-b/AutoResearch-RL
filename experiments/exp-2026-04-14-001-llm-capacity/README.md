# Experiment: LLM Contribution to Architecture Improvement Capacity

**Experiment ID**: `exp-2026-04-14-001-llm-capacity`  
**Status**: ✅ **COMPLETED** (Phase 1)  
**Date Started**: 2026-04-14  
**Date Completed**: 2026-04-14  
**Owner**: Luke B  

---

## Executive Summary

**Phase 1 of the LLM capacity validation is complete.** We conducted a controlled comparison of LLM-guided patch generation (Hugging Face Qwen) versus random hyperparameter mutation in the AutoResearch-RL framework.

**Key Findings**:
- Random Baseline: 0% acceptance (0/133 iterations), mean reward 0.0018 ± 0.032
- HF Router LLM: 0% acceptance (0/101 iterations), mean reward 0.0017 ± 0.031
- **Statistical Result**: p=0.9802, Cohen's d=-0.003 → **NO significant difference**
- **Outcome**: Outcome C (HF ≈ Random) due to mock environment limitations

**Root Cause**: The mock GPU dispatcher treats all patches identically. Neither random nor LLM patches produce measurable improvements because the evaluation environment doesn't run real training or compute genuine BPB feedback.

---

## Experiment At a Glance

| Dimension | Value |
|-----------|-------|
| **Primary Hypothesis** | HF LLM >5× higher acceptance rate than random ❌ Not supported |
| **Sample Size** | ~13 baseline runs + ~10 HF runs (~133-101 iterations each) |
| **Treatments** | A) Random patches, B) HF Router LLM-guided |
| **Duration** | 1 day execution, 1 day analysis |
| **Outcome** | Inconclusive due to environmental limitations |
| **Full Results** | See [results.md](./results.md) |

---

## Phase 1 Results

### Data Summary

| Treatment | Runs | Observations | Acceptance | Avg Reward | Std Reward |
|-----------|------|---|---|---|---|
| **Random Baseline** | ~13 | 133 | 0/133 (0.0%) | 0.0018 | 0.0318 |
| **HF Router LLM** | ~10 | 101 | 0/101 (0.0%) | 0.0017 | 0.0311 |

### Statistical Analysis

```
Two-sample t-test (HF vs Baseline Rewards):
  t-statistic: -0.0248
  p-value: 0.9802 (NOT significant)
  Cohen's d: -0.003 (negligible effect size)
  95% CI on difference: [-0.0151, 0.0147]
```

**Interpretation**: No statistically significant difference between treatments. Both perform identically across all metrics.

### Why Outcome C?

While the hypothesis predicted "HF > Random × 5", actual results show HF ≈ Random. This is likely due to:

1. **Mock Environment**: AUTORESEARCH_MODE=LOCAL doesn't differentiate patch quality
2. **No Real Training**: Patches aren't evaluated on actual model training
3. **Uniform Evaluation**: Both random and LLM patches fail equally (0% acceptance)
4. **No BPB Signal**: Reward reflects only novelty + cost, not performance delta

**Key Insight**: This is **not** an LLM failure—it's an **environmental limitation**. The evaluation environment provides no signal to discriminate patch quality, so both methods appear identical.

---

## What This Means

### ✅ Validated
- Experiment infrastructure (logging, aggregation, analysis) works correctly
- Random patch generator produces valid hyperparameter mutations
- HF router integration successfully parses and applies patches
- Statistical analysis pipeline computes metrics properly

### ❌ Inconclusive
- Whether LLM actually provides architectural insight
- Whether method differences exist when meaningful evaluation is possible
- Phase 2 ablations (temperature, context) will also be null in mock mode

### 🔮 Next Steps

**Recommended Path Forward**:
1. **Plan Phase 1 Redux on GPU cluster** with real Qwen model + actual training
   - Real BPB feedback signal needed to discriminate method quality
   - Target: 20+ runs × 20+ iterations per treatment
   - Expected outcome: Clear signal separation if LLM works

2. **Skip Phase 2 ablations** for now
   - Temperature/context variations will also show null results in mock mode
   - Resume after GPU cluster validation

3. **Archive as Proof-of-Concept**
   - This experiment demonstrated experiment framework works
   - Saved as reference for infrastructure validation

See [results.md](./results.md) for **full analysis, limitations, and implications**.

---

## Detailed Documentation

| Document | Status | Purpose |
|----------|--------|---------|
| [design.md](./design.md) | ✅ Complete | Full 3-phase methodology, ablation plans, statistical methods |
| [results.md](./results.md) | ✅ Complete | Comprehensive Phase 1 analysis, conclusions, next steps |
| [analysis/analyze.py](./analysis/analyze.py) | ✅ Complete | Statistical analysis pipeline (t-tests, effect sizes, plots) |
| `./data/baseline/` | ✅ Complete | 13 run directories with experiment_logs.jsonl |
| `./data/hf_router/` | ✅ Complete | 10 run directories with experiment_logs.jsonl |

---

## Key Learnings

1. **Mock environments hide differences** when evaluation is uniform across treatments
2. **Real evaluation infrastructure is critical** for detecting method effectiveness
3. **Statistical power requires signal**—cannot detect effects when both treatments fail equally
4. **Experimental design validates assumptions**—this revealed infrastructure requirements rather than method failure

---

## Reproducibility

### Run the Analysis
To reproduce the Phase 1 analysis:
```bash
cd /workspaces/AutoResearch-RL/experiments/exp-2026-04-14-001-llm-capacity
python3 analysis/analyze.py
```

### Data Location
```
exp-2026-04-14-001-llm-capacity/
├── data/
│   ├── baseline/run_001 - run_013/experiment_logs.jsonl
│   └── hf_router/run_001 - run_010/experiment_logs.jsonl
├── analysis/
│   └── analyze.py (statistical analysis)
├── design.md (methodology)
├── results.md (conclusions)
└── README.md (this file)
```

---

## Timeline

- **2026-04-14**: Phase 1 execution complete (20 run directories, ~234 iterations)
- **2026-04-14**: Statistical analysis reveals null results
- **2026-04-14**: Root cause analysis (mock environment limitations)
- **2026-04-14**: Phase 1 concluded; documentation complete
- **2026-04-21**: Planned Phase 1 Redux on GPU cluster

---

## Contact & Questions

- **Experiment Owner**: Luke B
- **Status**: Phase 1 complete; Phase 2+ on hold pending GPU cluster
- **Results Location**: [results.md](./results.md)
- **Design Details**: [design.md](./design.md)
- **Last Updated**: 2026-04-14

