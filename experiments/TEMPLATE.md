# Experiment Template

Copy this template and fill in sections for each new experiment. Replace all `[BRACKETS]` with your values.

---

# [Experiment Title]

**Experiment ID**: `exp-YYYY-MM-DD-###-short-name`  
**Status**: Planned / In Progress / Analysis / Completed  
**Date Started**: [YYYY-MM-DD]  
**Date Completed**: [YYYY-MM-DD or TBD]  
**Owner**: [Your Name]  

---

## 1. Research Question & Motivation

**Primary Question**: [What is the core question you're answering?]

**Hypothesis**: [What do you predict will happen?]

**Motivation**: [Why does this matter? What gap does this fill?]

---

## 2. Experiment Design

### 2.1 Treatments / Conditions

| Name | Description | Expected Outcome |
|------|-------------|------------------|
| [Treatment A] | [What varies?] | [What should happen?] |
| [Treatment B] | [What varies?] | [What should happen?] |

### 2.2 Sample Size & Duration

- **Runs per treatment**: [N]
- **Iterations per run**: [M]
- **Total GPU jobs**: [N × M × treatments]
- **Estimated time**: [hours/days]

### 2.3 Metrics

| Metric | Definition | Why It Matters |
|--------|-----------|---|
| [Metric 1] | [How is it computed?] | [What does it tell us?] |
| [Metric 2] | [How is it computed?] | [What does it tell us?] |

### 2.4 Success Criteria

- [ ] Criterion 1: [Quantitative threshold]
- [ ] Criterion 2: [Quantitative threshold]
- [ ] Criterion 3: [Quantitative threshold]

---

## 3. Conduction & Execution

### 3.1 Setup

```bash
# Command to run experiment
[your command here]
```

### 3.2 Data Collection

- **Logging format**: JSON lines (`experiment_logs.jsonl`)
- **Additional metadata**: [Any extra fields to track?]
- **Storage location**: `./exp-YYYY-MM-DD-###-short-name/data/`

### 3.3 Execution Timeline

| Phase | Timeline | Status |
|-------|----------|--------|
| Setup & validation | [date range] | Not started |
| Data collection (treatment A) | [date range] | Not started |
| Data collection (treatment B) | [date range] | Not started |
| Analysis | [date range] | Not started |
| Report writing | [date range] | Not started |

---

## 4. Results & Findings

### 4.1 Summary Metrics

| Treatment | Metric 1 | Metric 2 | Metric 3 |
|-----------|----------|----------|----------|
| [A] | [value] | [value] | [value] |
| [B] | [value] | [value] | [value] |

### 4.2 Statistical Analysis

```
T-test results (Metric 1: A vs B):
  t-statistic: [value]
  p-value: [value]
  Significant? [Yes/No]
```

### 4.3 Key Findings

1. **Finding 1**: [What did you learn?]
   - Supporting evidence: [data/plots]
   
2. **Finding 2**: [What did you learn?]
   - Supporting evidence: [data/plots]

### 4.4 Interpretation

**What does this mean?** [Narrative explanation of results]

**Surprising results?** [Any unexpected outcomes?]

**Limitations**: [What could bias these results?]

---

## 5. Conclusions & Next Steps

### 5.1 Answer to Research Question

[Direct answer to your primary question from section 1]

### 5.2 Implications

- For [component/system]: [insight]
- For [component/system]: [insight]

### 5.3 Recommended Next Experiments

- [ ] [Follow-up experiment 1]
- [ ] [Follow-up experiment 2]

### 5.4 Code/Config Changes

If this experiment suggests changes to the codebase:

```python
# Before
[old code]

# After
[new code]

# PR/Commit: [link or hash]
```

---

## 6. Artifacts & Outputs

- **Logs**: `./data/`
- **Analysis notebook**: `./analysis/metrics_summary.ipynb`
- **Plots**: `./analysis/plots/`
- **Raw data**: `./data/[treatment]/run_*/experiment_logs.jsonl`

---

## References & Related Work

- [Prior experiment or paper that motivated this]
- [External research or baseline]

---

## Contact & Questions

**Questions about this experiment?** Reach out to [owner name].

