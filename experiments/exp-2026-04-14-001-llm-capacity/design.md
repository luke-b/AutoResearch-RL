# Detailed Experimental Design: LLM Capacity Evaluation

**Version**: 1.0  
**Date**: 2026-04-14  
**Experiment**: exp-2026-04-14-001-llm-capacity  

---

## Table of Contents

1. [Phase 1: Baseline vs. LLM Comparison](#phase-1)
2. [Phase 2: LLM Ablation Studies](#phase-2)
3. [Phase 3: LLM Quality Analysis](#phase-3)
4. [Statistical Methods](#statistics)
5. [Interpretation Framework](#interpretation)

---

## Phase 1: Baseline vs. LLM Comparison {#phase-1}

### Objective

Establish a rigorous comparison between:
- **Treatment A**: Random patch generation (no LLM)
- **Treatment B**: LLM-guided patch generation (HF Router)

### Hypothesis

**H₀** (Null): There is no difference in acceptance rate between treatments  
**H₁** (Alternative): HF LLM produces acceptance rate > 5× baseline

### Treatment A: Random Baseline

**Description**:
- Generate random hyperparameter mutations
- No context from history or system state
- No semantic validation

**Patch Generation Logic**:
```python
def generate_random_patch():
    """Generate a random hyperparameter mutation."""
    param = random.choice([
        "mlp_expansion",      # [2, 3, 4, 5]
        "lora_rank",          # [1, 2, 3, 4]
        "depth_loops",        # [1, 2, 3, 4]
        "block_size",         # [512, 1024, 2048]
    ])
    old_val = get_current_value(param)
    new_val = random.choice(valid_range_for(param))
    return {
        "search": f"{param} = {old_val}",
        "replace": f"{param} = {new_val}"
    }
```

**Expected Outcome**:
- Acceptance rate: ~0.1-1% (most random mutations decrease performance)
- BPB improvement velocity: near 0
- Patch diversity: high and uniform across all variables

### Treatment B: LLM-Guided (HF Router)

**Description**:
- Hugging Face Router API with Qwen/Qwen3-4B-Instruct-2507:nscale
- Temperature: 0.5 (balanced for exploration and exploitation)
- Full context: prior code, failure history, telemetry

**System Prompt**:
```
You are a code-mutating PPO agent optimized to output JSON diff patches.

CONSTRAINTS:
- Minimize BPB under 16MB
- No causality leaks
- 10-minute hard limit
- Maintain forward compatibility

SUCCESSFUL PATTERNS FROM HISTORY:
[Recent successful patches...]

FAILED PATTERNS (avoid repeating):
[Recent failed patches...]

TARGET HYPERPARAMETERS: mlp_expansion, lora_rank, depth_loops, block_size
```

**Expected Outcome**:
- Acceptance rate: >5%
- BPB improvement velocity: >0.01 per iteration
- Patch diversity: concentrated on key hyperparameters

### Metrics Collected Per-Iteration

```python
{
    "iteration": int,
    "job_id": str,
    "treatment": "baseline" | "hf_router",
    "run": int,  # 1-10
    
    # Patch details
    "patch": str,
    "patch_category": "scalar_assignment" | "loop_body" | "init" | "unfixable",
    "target_variables": [str],  # ["mlp_expansion", "lora_rank", ...]
    "patch_novelty": bool,  # Never seen before in this run's history
    
    # LLM-specific (treatment B only)
    "llm_temperature": float,
    "llm_model": str,
    
    # Execution results
    "acceptance": bool,
    "bpb_delta": float,  # Delta from prior iteration
    "applied_successfully": bool,
    "error_message": str | null,
    
    # Metadata
    "timestamp": float,
    "wall_clock_seconds": float,
}
```

### Aggregate Metrics

| Metric | Computation | Interpretation |
|--------|-----------|---|
| **Acceptance Rate** | (# accepted patches) / (# proposed patches) | Higher = better patch quality |
| **BPB Convergence Velocity** | (initial_BPB - final_BPB) / iterations | Higher = faster improvement |
| **Patch Diversity** | Cardinality of {target_variables} across all patches | Shows breadth of exploration |
| **Target Concentration** | % patches targeting {mlp_expansion, lora_rank, ...} | Shows if LLM focuses on "right" params |

---

## Phase 2: LLM Ablation Studies {#phase-2}

Conducted *after* Phase 1 baseline, using only HF treatment with variations.

### Study 2A: Temperature Sweep (5 runs each)

**Question**: Does temperature affect patch diversity or quality?

| Temp | Runs | Rationale | Prediction |
|------|------|-----------|-----------|
| 0.1 | 5 | Deterministic, conservative | Low diversity, high accuracy |
| 0.5 | 5 | [Phase 1 baseline] | Balanced |
| 1.0 | 5 | High stochasticity | High diversity, lower quality |
| 1.5 | 5 | Max supported | Chaotic, invalid patches |

**Success Criterion**: Temperature 0.5 should outperform 0.1 and 1.0 in acceptance rate.

### Study 2B: History Context Importance (5 runs each)

**Question**: Does LLM learn better patches with full failure history?

**Conditions**:
- **Full Context**: Include all prior failures + innovations (current default)
- **No Context**: Only current code, no history (`--disable_history_context`)

**Metric**: 
```
context_value = (accuracy_full - accuracy_no_context) / accuracy_no_context
```

**Prediction**: LLM with context > without context (p < 0.05)

### Study 2C: Patch Specificity (Continuous monitoring)

**Question**: Does LLM target the right hyperparameters?

Track which variables are modified:

```python
# In analysis:
expected_targets = {"mlp_expansion", "lora_rank", "depth_loops", "block_size"}
proposed_targets = set()
for patch in all_patches:
    target = extract_identifier(patch['path'])
    proposed_targets.add(target)

specificity = len(proposed_targets & expected_targets) / len(proposed_targets)
print(f"Specificity: {specificity:.2f}")  # Expected: >0.8
```

---

## Phase 3: LLM Quality Analysis {#phase-3}

Deep-dive investigation of patch characteristics.

### 3A: Patch Type Distribution

**Categories**:
1. **Scalar Assignment** (easiest): `mlp_expansion=3 → mlp_expansion=4`
2. **Loop Body** (hard): Complex function replacements
3. **Initialization** (medium): Class config changes
4. **Unfixable** (failure): Patches that couldn't be parsed or applied

**Prediction**:
- Random: Distribution roughly uniform across types
- HF: 60%+ scalar assignments (easiest to apply correctly)

**Computation**:
```python
categories = defaultdict(int)
for patch in all_patches:
    cat = classify_patch(patch)
    categories[cat] += 1

for cat, count in categories.items():
    print(f"{cat}: {count / len(all_patches) * 100:.1f}%")
```

### 3B: Pattern Learning Detection

**Question**: Does the LLM avoid previously-failed patches?

```python
failed_patches = [e['patch'] for e in history if e['status'] == 'FAILED']
recent_5_patches = [agent.generate_action(...) for _ in range(5)]

repeat_rate = len(set(recent_5_patches) & set(failed_patches)) / len(recent_5_patches)
```

**Prediction**:
- Random: repeat_rate ≈ 30-50% (expected by chance)
- HF: repeat_rate < 10% (LLM learns to avoid failures)

### 3C: Semantic Coherence

**Question**: Do patches form a logical progression?

**Example of Coherent Sequence**:
1. Reduce `mlp_expansion: 3 → 2` (save compute)
2. Reduce `block_size: 2048 → 1024` (save memory)
3. Reduce `lora_rank: 4 → 2` (simpler fine-tuning)

**Example of Incoherent Sequence**:
1. Increase `mlp_expansion: 3 → 4`
2. Decrease `mlp_expansion: 4 → 2`
3. Increase `mlp_expansion: 2 → 3` (circular)

**Metric**: Cosine similarity between consecutive patch embeddings.

```python
def patch_embedding(patch):
    """Embed patch as vector of hyperparameter changes."""
    emb = {}
    for var in ["mlp_expansion", "lora_rank", "block_size", "depth_loops"]:
        if var in patch['path']:
            emb[var] = patch['value']
    return emb

coherence = []
for i in range(len(patches) - 1):
    sim = cosine_similarity(
        patch_embedding(patches[i]),
        patch_embedding(patches[i+1])
    )
    coherence.append(sim)

avg_coherence = mean(coherence)
print(f"Avg Coherence: {avg_coherence:.3f}")
```

**Prediction**:
- Random: coherence ≈ 0.1-0.3 (no structure)
- HF: coherence ≈ 0.6-0.8 (consistent direction)

---

## Statistical Methods {#statistics}

### Primary Test: Independent Samples T-Test

Comparing BPB convergence velocity between treatments:

```python
from scipy.stats import ttest_ind

baseline_velocity = [
    (init_bpb - final_bpb) / n_iterations
    for run in baseline_runs
]

hf_velocity = [
    (init_bpb - final_bpb) / n_iterations
    for run in hf_runs
]

t_stat, p_value = ttest_ind(hf_velocity, baseline_velocity)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✓ Significant improvement (p < 0.05)")
else:
    print("✗ No significant improvement")
```

### Effect Size: Cohen's d

```python
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group2) - np.mean(group1)) / pooled_std

d = cohens_d(baseline_velocity, hf_velocity)
print(f"Cohen's d: {d:.3f}")
# d < 0.2: small effect
# 0.2 ≤ d < 0.5: small-to-medium
# 0.5 ≤ d < 0.8: medium-to-large
# d ≥ 0.8: large effect
```

### Multiple Comparisons Correction

If running multiple tests, apply Bonferroni correction:
```
α_corrected = α / number_of_tests
# E.g., if 5 tests: α = 0.05 / 5 = 0.01
```

---

## Interpretation Framework {#interpretation}

### Outcome A: HF Significantly Outperforms Random (p < 0.05, Cohen's d > 0.5)

**Conclusion**: ✅ LLM provides genuine architectural insight beyond randomness

**Evidence Pattern**:
- Acceptance rate: HF >> Random
- Convergence velocity: HF > Random
- Patch types: HF concentrated on scalars; Random uniform
- Pattern learning: HF avoids failed patches; Random repeats

**Actions**:
1. ✅ Approve HF router as primary patch generator
2. 📊 Optimize temperature and prompting for even better performance
3. 💰 Calculate cost/benefit of HF inference vs. improvement gains
4. 🔬 Move to Phase 2 ablations (temperature sweep, etc.)

### Outcome B: HF Slightly Outperforms Random (0.05 < p < 0.1, 0.2 < Cohen's d < 0.5)

**Conclusion**: ⚠️ LLM helps marginally; additional optimization needed

**Evidence Pattern**:
- Acceptance rate: HF > Random, but < 5× difference
- Convergence velocity: HF slightly better (not consistent)
- Pattern learning: Weak signal or random-like

**Actions**:
1. 🔍 Investigate system prompt engineering (clearer constraints?)
2. 📈 Evaluate larger models (Qwen 7B vs 4B)
3. 🎯 Increase context window or history depth
4. 🚀 Fine-tune on successful patches from prior runs
5. 🔬 Proceed with Phase 2 ablations to identify bottlenecks

### Outcome C: HF ≈ Random (p > 0.1)

**Conclusion**: ❌ Current LLM not suited for this task

**Evidence Pattern**:
- Acceptance rate: HF ≈ Random
- Convergence velocity: No significant difference
- Patch types: HF generates same quality as random
- Pattern learning: No evidence of learning

**Actions**:
1. ❌ Do NOT deploy HF router at scale
2. 📋 Investigate failure modes:
   - Is instruction-following broken?
   - Are constraints being ignored?
   - Is context window too small?
3. 🔄 Consider alternatives:
   - Rule-based / heuristic patch generation
   - Fine-tuning on successful patches
   - Evaluate completely different models
4. 🗂️ Archive this experiment and create follow-up

---

## Visualization Plan

Plots to generate:

1. **BPB Convergence Curves**
   - X: Iteration, Y: BPB
   - Lines: Mean ± std for each treatment
   - Title: "BPB Convergence: Random vs LLM-Guided"

2. **Acceptance Rate Comparison**
   - Bar chart: Baseline vs HF Router
   - Error bars: 95% CI

3. **Patch Type Distribution**
   - Stacked bar: Scalar, Loop, Init, Unfixable
   - Side-by-side: Random vs HF

4. **Learning Signal**
   - X: Iteration, Y: Repeat rate
   - Show decline of repeat rate over time for HF

5. **Coherence Over Time**
   - X: Iteration, Y: Patch coherence
   - HF should trend upward; Random flat

---

## Reproducibility Checklist

- [ ] Seed all random number generators (for baseline reproducibility)
- [ ] Log HF API version and model ID
- [ ] Freeze all code versions (git commit SHA)
- [ ] Document exact prompts used
- [ ] Save all raw logs (experiment_logs.jsonl) with timestamps
- [ ] Version analysis notebook (git tracking)
- [ ] Note any environmental variables or config changes

---

## References

- [Prior optimization work in AutoResearch-RL]
- [LLM-guided code generation literature]
- [Statistical testing best practices]

