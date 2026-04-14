# Experiment: Measuring LLM Contribution to Architecture Improvement Capacity

## Research Question
**To what degree does the Hugging Face LLM contribute meaningful architectural improvements versus random/heuristic baselines?**

---

## Experimental Design

### Phase 1: Baseline vs. LLM Comparison (N=10 runs each)

#### Treatment A: Random Baseline (MOCK mode)
- **Hypothesis**: Random hyperparameter mutations provide weak improvement signal
- **Expected outcome**: ~0% acceptance rate, minimal BPB improvement
- **Runtime**: 10 iterations × 10 runs = 100 jobs
- **Command**:
  ```bash
  python3 main.py --max_iterations=10 --llm_mode=none  # Purely random patches
  ```

#### Treatment B: LLM-Guided (HF Router, `temperature=0.5`)
- **Hypothesis**: LLM provides principled patch generation based on system constraints
- **Expected outcome**: >5% acceptance rate, positive BPB trend
- **Runtime**: 10 iterations × 10 runs = 100 jobs
- **Command**:
  ```bash
  HF_TOKEN="<token>" python3 main.py --max_iterations=10 --llm_temperature=0.5
  ```

#### Key Metrics to Track
```json
{
  "bpb_convergence": "Final BPB after 10 iterations",
  "improvement_velocity": "(BPB_initial - BPB_final) / iterations",
  "acceptance_rate": "# Patches that didn't regress / total patches",
  "syntactic_validity": "# Parseable patches / total patches",
  "patch_diversity": "# Unique target variables modified",
  "novelty_score": "% Patches modifying variables not modified in history"
}
```

---

### Phase 2: LLM Ablation Studies (5 runs each)

#### Study 2A: Temperature Sweep
**Question**: Does higher temperature improve patch diversity or hurt quality?

| Temperature | Rationale | Prediction |
|---|---|---|
| 0.1 | Deterministic, conservative | Low diversity, high targeting accuracy |
| 0.5 | Recommended default | Balanced diversity & quality |
| 1.0 | High stochasticity | High diversity, lower acceptance |
| 1.5 | Max supported | Chaotic, likely invalid patches |

**Metric**: Plot acceptance_rate vs. temperature

#### Study 2B: History Context Importance
**Question**: Does the LLM learn better patches when given full failure history?

- **Full Context**: Include all 10 prior failures + innovations
- **No Context**: Only current code, no history → `--disable_history_context`
- **Metric**: Better LLM patches in context group → LLM is learning

#### Study 2C: Patch Specificity
**Question**: Does the LLM target the right hyperparameters?

Track which variables the LLM modifies:
```python
# In ASTDiffParser.parse_llm_json():
target_vars = set()
for patch in patches:
    if 'path' in patch:
        target_vars.add(extract_identifier(patch['path']))

# Expected HF targets: mlp_expansion, lora_rank, block_size, depth_loops
# (the ones that actually affect BPB per the system prompt)
```

---

### Phase 3: LLM Quality Analysis (Deep Dive)

#### 3A: Patch Type Distribution
Categorize patches generated:
```json
{
  "scalar_assignment": "mlp_expansion=3 → mlp_expansion=4",
  "loop_body": "Complex function replacements",
  "initialization": "Class config changes",
  "unfixable": "Patches that failed to apply"
}
```

**Prediction**: HF should favor scalar_assignment (easiest to apply), while mock generates random across all types.

#### 3B: Pattern Learning Detection
After 10 failed iterations, does the LLM stop proposing the same failed patches?

```python
# Track patch uniqueness
failed_patches = [e['patch'] for e in history if e['status'] == 'FAILED']
recent_patches = [agent.generate_action(...) for _ in range(3)]
repeat_rate = len(set(recent_patches) & set(failed_patches)) / len(recent_patches)
```

**Prediction**: LLM repeat_rate << random (LLM learns to avoid previous failures).

#### 3C: Semantic Coherence
Do patches form a logical progression?

- **Example sequence (coherent)**:
  1. Reduce `mlp_expansion: 3 → 2`  (save compute)
  2. Reduce `block_size: 2048 → 1024` (save memory)
  3. Reduce `lora_rank: 4 → 2` (simpler fine-tuning)

- **Example sequence (incoherent)**:
  1. Increase `mlp_expansion: 3 → 4`
  2. Decrease `mlp_expansion: 4 → 2`
  3. Increase `mlp_expansion: 2 → 3` (circular)

**Metric**: Similarity of consecutive patches (context consistency).

---

## Execution Plan

### Week 1: Data Collection
```bash
# Run baseline & LLM treatments in parallel
for run in {1..10}; do
  # Baseline
  timeout 3600 python3 main.py --max_iterations=10 --llm_mode=none \
    > logs/baseline_run_${run}.log 2>&1 &
  
  # HF Router
  timeout 3600 bash -c "HF_TOKEN=... python3 main.py --max_iterations=10" \
    > logs/hf_run_${run}.log 2>&1 &
done
wait
```

### Week 2: Analysis
```python
import pandas as pd
import json

# Load all experiment_logs.jsonl results
results = pd.concat([
    pd.read_json('experiment_logs.jsonl', lines=True)
    for log_file in glob('logs/*.log')
])

# Compute metrics
metrics = {
    'baseline': compute_metrics(results[results['mode']=='none']),
    'hf_router': compute_metrics(results[results['mode']=='hf']),
}

# Statistical tests
from scipy import stats
t_stat, p_val = stats.ttest_ind(
    metrics['baseline']['bpb_convergence'],
    metrics['hf_router']['bpb_convergence']
)
print(f"BPB Improvement: HF vs Baseline, p={p_val:.4f}")
```

---

## Success Criteria

| Metric | Random | HF Router | Required for "Success" |
|--------|--------|-----------|----------------------|
| **Acceptance Rate** | <1% | >5% | HF > Random × 5 |
| **BPB Improvement Velocity** | ~0 | >0.01 per iteration | HF statistically different |
| **Patch Diversity** | High (random) | Targeted to key vars | Concentrated on {mlp_expansion, lora_rank, ...} |
| **Learning Signal** | No learning | Patch innovation > repetition | Repeat rate < 10% by iter 8+ |
| **p-value (t-test)** | — | — | p < 0.05 (significant improvement) |

---

## Interpretation Framework

### Outcome A: HF significantly outperforms random
- ✅ **Conclusion**: LLM provides genuine architectural insight beyond randomness
- **Next step**: Optimize LLM prompting & temperature for better convergence

### Outcome B: HF slightly outperforms random (not significant)
- ⚠️ **Conclusion**: LLM helps marginally; may need:
  - Better system prompt engineering
  - More sophisticated patch validation
  - Longer context windows

### Outcome C: HF ≈ Random
- ❌ **Conclusion**: Current LLM not suited for this task; evaluate:
  - Larger models (e.g., Qwen 7B vs 4B)
  - Fine-tuning on prior successful patches
  - Constraint-aware prompt engineering

---

## Tracking & Logging

Enhance `log_experiment_json()` to track LLM-specific metadata:

```json
{
  "iteration": 1,
  "patch": "mlp_expansion=4",
  "llm_temperature": 0.5,
  "llm_model": "Qwen/Qwen3-4B-Instruct-2507:nscale",
  "patch_category": "scalar_assignment",
  "target_variables": ["mlp_expansion"],
  "acceptance": true,
  "bpb_delta": 0.02,
  "patch_novelty": true,  # Never seen before in history
  "applied_successfully": true,
  "error_message": null
}
```

---

## Timeline
- **Setup**: 1 day (implement logging, test infrastructure)
- **Execution**: 7-14 days (parallel runs, ~200 GPU jobs)
- **Analysis**: 2-3 days (statistical tests, visualization)
- **Report**: 1 day (write findings)

**Total**: ~3 weeks for rigorous evaluation

