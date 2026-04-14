#!/usr/bin/env python3
"""
Analysis script for exp-2026-04-14-001-llm-capacity

Computes metrics from raw logs and generates plots.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_DIR = Path(__file__).parent / "data"
BASELINE_DIR = DATA_DIR / "baseline"
HF_DIR = DATA_DIR / "hf_router"
OUTPUT_DIR = Path(__file__).parent / "analysis"

def load_logs(treatment_dir):
    """Load all experiment_logs.jsonl files from a treatment directory."""
    all_logs = []
    for log_file in treatment_dir.glob("run_*/experiment_logs.jsonl"):
        with open(log_file) as f:
            for line in f:
                if line.strip():
                    try:
                        all_logs.append(json.loads(line))
                    except:
                        pass
    return pd.DataFrame(all_logs) if all_logs else pd.DataFrame()

def compute_metrics(df, treatment_name):
    """Compute summary metrics for a treatment."""
    metrics = {
        "treatment": treatment_name,
        "total_iterations": len(df),
        "acceptance_rate": df["acceptance"].sum() / len(df) if "acceptance" in df else 0,
        "mean_bpb_delta": df["bpb_delta"].mean() if "bpb_delta" in df else 0,
        "std_bpb_delta": df["bpb_delta"].std() if "bpb_delta" in df else 0,
    }
    
    # BPB Convergence Velocity
    if "bpb_delta" in df:
        total_improvement = df["bpb_delta"].sum()
        iterations = len(df)
        metrics["bpb_convergence_velocity"] = total_improvement / iterations if iterations > 0 else 0
    
    return metrics

def plot_bpb_convergence(baseline_df, hf_df):
    """Plot BPB convergence over iterations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Aggregate by treatment and iteration
    baseline_by_iter = baseline_df.groupby("iteration")["bpb_delta"].mean()
    hf_by_iter = hf_df.groupby("iteration")["bpb_delta"].mean()
    
    ax.plot(baseline_by_iter.index, baseline_by_iter.values, marker='o', label='Random Baseline', linewidth=2)
    ax.plot(hf_by_iter.index, hf_by_iter.values, marker='s', label='HF Router LLM', linewidth=2)
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("BPB Delta (improvement)")
    ax.set_title("BPB Convergence: Random vs LLM-Guided")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.savefig(OUTPUT_DIR / "plots" / "bpb_convergence.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: bpb_convergence.png")

def plot_acceptance_rate(baseline_metrics, hf_metrics):
    """Plot acceptance rate comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    treatments = ["Baseline", "HF Router"]
    rates = [
        baseline_metrics["acceptance_rate"] * 100,
        hf_metrics["acceptance_rate"] * 100,
    ]
    
    bars = ax.bar(treatments, rates, color=['#ff6b6b', '#51cf66'], alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel("Acceptance Rate (%)")
    ax.set_title("Acceptance Rate: Random vs LLM-Guided")
    ax.set_ylim(0, max(rates) * 1.2)
    
    fig.savefig(OUTPUT_DIR / "plots" / "acceptance_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: acceptance_rate_comparison.png")

def statistical_test(baseline_df, hf_df):
    """Run t-test comparing convergence velocities."""
    # Compute velocity per run
    baseline_velocities = []
    hf_velocities = []
    
    # Assuming runs are grouped by "run" column
    for run in baseline_df["run"].unique():
        subset = baseline_df[baseline_df["run"] == run]
        velocity = subset["bpb_delta"].sum() / len(subset) if len(subset) > 0 else 0
        baseline_velocities.append(velocity)
    
    for run in hf_df["run"].unique():
        subset = hf_df[hf_df["run"] == run]
        velocity = subset["bpb_delta"].sum() / len(subset) if len(subset) > 0 else 0
        hf_velocities.append(velocity)
    
    t_stat, p_value = stats.ttest_ind(hf_velocities, baseline_velocities)
    
    # Cohen's d
    cohens_d = (np.mean(hf_velocities) - np.mean(baseline_velocities)) / np.sqrt(
        ((len(hf_velocities)-1) * np.var(hf_velocities, ddof=1) + 
         (len(baseline_velocities)-1) * np.var(baseline_velocities, ddof=1)) / 
        (len(hf_velocities) + len(baseline_velocities) - 2)
    )
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "significant": p_value < 0.05,
    }

def main():
    print("=" * 60)
    print("Analysis: exp-2026-04-14-001-llm-capacity")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading experimental data...")
    try:
        baseline_df = load_logs(BASELINE_DIR)
        print(f"  ✓ Baseline: {len(baseline_df)} observations")
    except FileNotFoundError:
        print("  ✗ Baseline data not found. Have you run the baseline treatment?")
        baseline_df = pd.DataFrame()
    
    try:
        hf_df = load_logs(HF_DIR)
        print(f"  ✓ HF Router: {len(hf_df)} observations")
    except FileNotFoundError:
        print("  ✗ HF Router data not found. Have you run the HF treatment?")
        hf_df = pd.DataFrame()
    
    if baseline_df.empty or hf_df.empty:
        print("\n❌ Cannot proceed with analysis. Run treatments first.")
        return
    
    # Compute metrics
    print("\n📊 Computing metrics...")
    baseline_metrics = compute_metrics(baseline_df, "baseline")
    hf_metrics = compute_metrics(hf_df, "hf_router")
    
    print(f"\n{'Treatment':<20} {'Acceptance Rate':<20} {'BPB Velocity':<20}")
    print("-" * 60)
    for metrics in [baseline_metrics, hf_metrics]:
        print(
            f"{metrics['treatment']:<20} "
            f"{metrics['acceptance_rate']*100:>6.2f}% {' '*12} "
            f"{metrics['bpb_convergence_velocity']:>6.4f} {' '*12}"
        )
    
    # Statistical test
    print("\n🔬 Statistical Analysis...")
    test_results = statistical_test(baseline_df, hf_df)
    print(f"  t-statistic: {test_results['t_statistic']:.4f}")
    print(f"  p-value: {test_results['p_value']:.4f}")
    print(f"  Cohen's d: {test_results['cohens_d']:.3f}")
    print(f"  Significant (p < 0.05)? {'✓ Yes' if test_results['significant'] else '✗ No'}")
    
    # Plots
    print("\n🎨 Generating visualizations...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "plots").mkdir(exist_ok=True)
    
    if not baseline_df.empty and not hf_df.empty:
        plot_bpb_convergence(baseline_df, hf_df)
        plot_acceptance_rate(baseline_metrics, hf_metrics)
    
    print("\n✅ Analysis complete!")
    print(f"   Outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
