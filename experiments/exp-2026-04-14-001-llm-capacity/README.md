# Experiment: LLM Contribution to Architecture Improvement Capacity

**Experiment ID**: `exp-2026-04-14-001-llm-capacity`  
**Status**: Planned  
**Date Started**: —  
**Date Completed**: TBD  
**Owner**: Luke B  

---

## Executive Summary

This experiment rigorously measures the contribution of the Hugging Face LLM to architecture improvement quality by comparing LLM-guided patch generation against random baselines. The goal is to quantify whether the LLM provides genuine architectural insight or merely generates patches indistinguishable from random mutation.

**Key Question**: *To what degree does the Hugging Face LLM contribute meaningful architectural improvements versus random/heuristic baselines?*

---

## Experiment At a Glance

| Dimension | Details |
|-----------|---------|
| **Primary Hypothesis** | HF LLM provides >5× higher acceptance rate than random patches |
| **Sample Size** | 10 runs × 10 iterations per treatment = 100 jobs/treatment |
| **Treatments** | A) Random patches (baseline), B) HF Router LLM-guided |
| **Duration** | ~2 weeks execution, ~1 week analysis |
| **Success Criteria** | p < 0.05 improvement in BPB convergence velocity |

---

## Research Motivation

1. **AutoResearch-RL needs an intelligent patch generator**  
   → The PPO agent relies on the LLM for proposing code modifications
   
2. **HF Router is expensive**  
   → Need empirical evidence that LLM quality justifies the cost vs. randomness
   
3. **LLM instruction-following is uncertain**  
   → Does the LLM actually understand the architecture constraints (BPB, causality)?

---

## Experimental Design

### Treatments

**Treatment A: Random Baseline**
- Hyperparameter mutations from uniform random ranges
- No context from history
- Expected: ~0% acceptance, minimal BPB change

**Treatment B: LLM-Guided (HF Router)**
- Hugging Face Router API with Qwen 4B model
- Full context: prior code + failure history + telemetry
- Temperature = 0.5 (balanced exploration/exploitation)
- Expected: >5% acceptance, positive BPB trend

### Data & Metrics

**Per-iteration tracking**:
```json
{
  "iteration": 1,
  "patch": "mlp_expansion=3→4",
  "llm_temperature": 0.5,
  "target_variables": ["mlp_expansion"],
  "acceptance": true,
  "bpb_delta": 0.02,
  "patch_novelty": true,
  "applied_successfully": true
}
```

**Aggregate metrics**:
- **Acceptance Rate**: % patches that didn't regress
- **BPB Convergence Velocity**: (initial_BPB - final_BPB) / iterations
- **Patch Diversity**: # unique target variables modified
- **Pattern Learning**: Do failed patches get repeated?

### Success Criteria

| Metric | Baseline | LLM Target | Required for Success |
|--------|----------|------------|----------------------|
| Acceptance Rate | <1% | >5% | HF > Random × 5 |
| BPB Velocity | ~0 | >0.01/iter | p < 0.05 difference |
| Patch Diversity | High | Targeted | Concentrated on key hyperparams |
| Learning Signal | None | Yes | Repeat rate < 10% by iter 8+ |

---

## Conduction Plan

### Phase 1: Setup & Validation (Day 1)
- [ ] Implement enhanced logging (patch category, novelty, etc.)
- [ ] Create baseline random generator
- [ ] Test HF router connectivity
- [ ] Validate analysis pipeline

### Phase 2: Data Collection (Days 2-7)
- [ ] Run 10 baseline treatments (100 GPU jobs total)
- [ ] Run 10 HF treatments (100 GPU jobs total)
- [ ] Monitor for failures, collect logs

### Phase 3: Analysis (Days 8-9)
- [ ] Compute metrics per treatment
- [ ] Run t-tests and effect size calculations
- [ ] Generate visualization plots
- [ ] Document all findings

### Phase 4: Reporting (Day 10)
- [ ] Write results.md with conclusions
- [ ] Create presentation-ready plots
- [ ] Recommend follow-up experiments

---

## Expected Outcomes

### Outcome A: HF Significantly Outperforms Random (p < 0.05)
✅ **Conclusion**: LLM provides genuine architectural value  
**Actions**:
- Optimize LLM temperature & prompting
- Roll out HF router as primary patch generator
- Invest in cost optimization for inference

### Outcome B: HF Slightly Better Than Random (0.05 < p < 0.1)
⚠️ **Conclusion**: LLM helps but not dramatically  
**Actions**:
- Investigate system prompt engineering
- Try larger models (Qwen 7B vs 4B)
- Increase context window size

### Outcome C: HF ≈ Random (p > 0.1)
❌ **Conclusion**: Current LLM inadequate  
**Actions**:
- Consider fine-tuning on successful patches
- Evaluate alternative models
- Return to rule-based patch generation

---

## Execution Commands

### Baseline (Random)
```bash
for run in {1..10}; do
  timeout 7200 python3 main.py \
    --max_iterations=10 \
    --llm_mode=none \
    --experiment_id=exp-2026-04-14-001 \
    --treatment=baseline \
    --run=$run \
    > logs/baseline_run_${run}.log 2>&1 &
done
wait
```

### HF Router Treatment
```bash
for run in {1..10}; do
  timeout 7200 bash -c "
    HF_TOKEN='...' python3 main.py \
      --max_iterations=10 \
      --llm_temperature=0.5 \
      --experiment_id=exp-2026-04-14-001 \
      --treatment=hf_router \
      --run=$run \
      > logs/hf_router_run_${run}.log 2>&1
  " &
done
wait
```

---

## Artifacts & Outputs

| Path | Description | Status |
|------|-------------|--------|
| `./data/baseline/` | Raw logs from baseline runs | Pending |
| `./data/hf_router/` | Raw logs from HF router runs | Pending |
| `./analysis/metrics_summary.ipynb` | Notebook with metric computation | Pending |
| `./analysis/plots/` | Visualization plots | Pending |
| `./results.md` | Final conclusions & findings | Pending |

---

## Detailed Methodology

See [design.md](./design.md) for the full experimental design including:
- Ablation studies (temperature sweep, history context)
- Deep-dive patch quality analysis
- Pattern learning detection methodology
- Statistical test details

---

## Timeline

- **Scheduled Start**: [You set this]
- **Data Collection Window**: 7-14 days
- **Analysis Complete**: +3-7 days
- **Report Ready**: +1-2 days

---

## Related Experiments

This experiment will inform:
- **Exp-2026-04-21-###**: LLM temperature optimization study
- **Exp-2026-05-01-###**: Fine-tuning on successful patches
- **Exp-2026-05-15-###**: Larger model evaluation (Qwen 7B)

---

## Questions & Contact

- **Questions?** Ask in `#autoresearch-experiments` Slack
- **Owner**: Luke B (@luke-b)
- **Last Updated**: 2026-04-14

