# AutoResearch-RL Experiments Directory

Centralized repository for all experimental work on AutoResearch-RL, including design, execution, analysis, and results documentation.

---

## Quick Start

### View All Experiments

```bash
cat experiments/CATALOG.md
```

### Create a New Experiment

1. Copy the template:
   ```bash
   cp experiments/TEMPLATE.md experiments/exp-YYYY-MM-DD-###-description/README.md
   ```

2. Create directory structure:
   ```bash
   mkdir -p experiments/exp-YYYY-MM-DD-###-description/{data,analysis/plots}
   ```

3. Fill in [TEMPLATE.md](./TEMPLATE.md) sections

4. Update [CATALOG.md](./CATALOG.md) with new experiment entry

### Run Existing Experiment

```bash
cd experiments/exp-2026-04-14-001-llm-capacity
cat README.md          # Overview
cat design.md          # Detailed methodology
python analysis/analyze.py   # Analyze results (after experiment runs)
cat results.md         # View findings
```

---

## Directory Structure

```
experiments/
│
├── CATALOG.md                              # Index of all experiments
├── TEMPLATE.md                             # Template for new experiments
├── README.md                               # This file
│
├── exp-2026-04-14-001-llm-capacity/        # Example experiment
│   ├── README.md                            # Experiment overview
│   ├── design.md                            # Detailed methodology
│   ├── data/
│   │   ├── baseline/
│   │   │   ├── run_001/
│   │   │   │   └── experiment_logs.jsonl   # Raw logs from execution
│   │   │   ├── run_002/
│   │   │   └── ...
│   │   └── hf_router/
│   │       ├── run_001/
│   │       └── ...
│   ├── analysis/
│   │   ├── analyze.py                      # Analysis script
│   │   ├── plots/
│   │   │   ├── bpb_convergence.png
│   │   │   └── acceptance_rate_comparison.png
│   │   └── metrics_summary.ipynb          # [Optional] Jupyter analysis
│   └── results.md                          # Key findings & conclusions
│
└── archived/                               # Old experiments (> 6 months)
    └── exp-2026-01-15-###-old-experiment/
```

---

## Experiment Lifecycle

Each experiment progresses through defined phases:

### 1. **Planned**
- Experiment design complete
- Methodology documented
- Ready to execute
- Status: Waiting for approval or resource allocation

### 2. **In Progress**
- Data collection underway
- Observations being logged
- No analysis yet
- Status: Running jobs, collecting logs

### 3. **Analysis**
- Data collection complete
- Metrics computed
- Statistical tests run
- Status: Analyzing results

### 4. **Completed**
- Final results documented
- Conclusions drawn
- Recommendations made
- Status: Ready for review or publication

### 5. **Archived**
- Experiment > 6 months old
- Moved to `/archived/` for reference
- Status: Historical record

---

## What to Document

### README.md (Experiment Overview)

- **Executive summary**: 1-2 paragraph overview
- **Research question**: What are you testing?
- **Hypothesis**: What do you predict?
- **Treatments**: What conditions are you comparing?
- **Sample size**: How many runs/iterations?
- **Expected outcome**: What would success look like?
- **Execution plan**: How to run the experiment
- **Timeline**: When will it be done?

### design.md (Detailed Methodology)

- **Experimental design**: Full technical details
- **Metrics**: How are things measured?
- **Statistical methods**: What tests are used?
- **Success criteria**: Quantitative thresholds
- **Ablation studies**: Variations to test
- **Interpretation framework**: How to interpret results

### data/ (Raw Experimental Artifacts)

- **Organized by treatment**: `baseline/`, `hf_router/`, etc.
- **One directory per run**: `run_001/`, `run_002/`, ...
- **Logs: experiment_logs.jsonl**: Raw observations
- **Metadata**: Any config files or parameters used

### analysis/ (Analysis & Visualization)

- **analyze.py**: Script to compute metrics and generate plots
- **plots/**: PNG/PDF figures (auto-generated or manual)
- **Optional: metrics_summary.ipynb**: Jupyter notebook for exploration

### results.md (Key Findings)

- **Quantitative results**: Table of metrics
- **Statistical tests**: p-values, effect sizes
- **Key findings**: 3-5 main insights
- **Interpretation**: What do results mean?
- **Conclusions**: Answer to research question
- **Next steps**: Recommended follow-up work

---

## Best Practices

### 1. Version Control

Commit everything to git:
```bash
git add experiments/
git commit -m "Add exp-2026-04-14-001: LLM capacity evaluation"
```

### 2. Reproducibility

Include in README:
- [ ] Exact command to run
- [ ] Git commit SHA of code version
- [ ] Python/library versions
- [ ] Random seeds (if applicable)
- [ ] Environment variables

### 3. Documentation

- Use clear section headers
- Include units for all metrics
- Link to related experiments
- Reference figures by filename

### 4. Data Organization

```bash
# Good: Organized by treatment and run
data/baseline/run_001/experiment_logs.jsonl
data/baseline/run_002/experiment_logs.jsonl
data/hf_router/run_001/experiment_logs.jsonl

# Bad: Flat and unclear
data/logs_1.jsonl
data/logs_2.jsonl
```

### 5. Analysis Comments

```python
# In analyze.py: explain every metric
def acceptance_rate(logs):
    """
    Percentage of proposed patches that improved or maintained BPB.
    
    High rate (>5%) indicates good patch quality.
    Low rate (<1%) suggests random or poor heuristic.
    """
    return (logs['acceptance'].sum() / len(logs)) * 100
```

---

## Useful Commands

### Create New Experiment Quickly

```bash
# 1. Generate directory structure
exp_id="exp-$(date +%Y-%m-%d)-001-my-experiment"
mkdir -p "experiments/${exp_id}"/{data,analysis/plots}

# 2. Copy template
cp experiments/TEMPLATE.md "experiments/${exp_id}/README.md"

# 3. Update catalog
echo "- [$exp_id](./experiments/${exp_id}/README.md)" >> experiments/CATALOG.md
```

### Check Experiment Status

```bash
# List all experiments with status
grep -E "^\| " experiments/CATALOG.md

# View specific experiment
cat experiments/exp-2026-04-14-001-llm-capacity/README.md
```

### Run Analysis

```bash
# From experiment directory
cd experiments/exp-2026-04-14-001-llm-capacity
python analysis/analyze.py
```

### Archive Old Experiment

```bash
# Move to archive after 6+ months
mv experiments/exp-2026-01-15-### experiments/archived/
```

---

## Example: LLM Capacity Experiment

This directory includes a complete example experiment: **exp-2026-04-14-001-llm-capacity**

**Purpose**: Measure how much a Hugging Face LLM contributes to architecture improvements vs. random patches.

**Files to review**:
- [README.md](./exp-2026-04-14-001-llm-capacity/README.md) - High-level overview
- [design.md](./exp-2026-04-14-001-llm-capacity/design.md) - Full methodology
- [analysis/analyze.py](./exp-2026-04-14-001-llm-capacity/analysis/analyze.py) - How to compute metrics
- [results.md](./exp-2026-04-14-001-llm-capacity/results.md) - Template for results

Use this as a reference when designing new experiments.

---

## Experiment Tracking

### In Catalog

Every experiment has a status badge:
- **Planned** 🔵
- **In Progress** 🟡
- **Analysis** 🟣
- **Completed** ✅
- **Archived** 📦

Example:
```markdown
| exp-2026-04-14-001 | Planned | LLM Capacity | Luke B |
```

### By Phase

Track how long experiments take:
- Setup: 1-2 days
- Execution: 3-14 days (depends on GPU availability)
- Analysis: 2-3 days
- Writing: 1-2 days
- **Total**: ~1-3 weeks per experiment

---

## Guidelines for Results

### When to Document Results

1. **After data collection is complete** (all runs finished)
2. **Before** analyzing or drawing conclusions
3. **Review with team** before archiving

### What to Include

- [ ] All quantitative metrics in a table
- [ ] Statistical test results (p-values, effect sizes)
- [ ] Plots generated by analysis script
- [ ] Clear interpretation of findings
- [ ] Alignment with success criteria from design.md
- [ ] Recommended next experiments

### What to Avoid

- ❌ Over-interpreting small differences
- ❌ Selective reporting (mention surprising results too!)
- ❌ Making causal claims from correlational data
- ❌ Forgiving issues for "expected reasons"
- ❌ Releasing results before peer review

---

## Questions?

- **How do I structure a new experiment?** → See [TEMPLATE.md](./TEMPLATE.md)
- **What goes in each section?** → See exp-2026-04-14-001 for examples
- **How do I update the catalog?** → Edit [CATALOG.md](./CATALOG.md)
- **Where should raw data go?** → `./[experiment]/data/[treatment]/run_###/`
- **How do I organize analysis?** → See `./analysis/analyze.py` example

---

## Last Updated

April 14, 2026

