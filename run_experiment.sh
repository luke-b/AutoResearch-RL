#!/bin/bash
# Launch exp-2026-04-14-001-llm-capacity baseline and HF router treatments

EXPERIMENT_ID="exp-2026-04-14-001-llm-capacity"
MAX_ITERATIONS=10
NUM_RUNS=10

echo "=========================================="
echo "AutoResearch-RL Experiment Launcher"
echo "Experiment: $EXPERIMENT_ID"
echo "Iterations: $MAX_ITERATIONS per run"
echo "Runs: $NUM_RUNS per treatment"
echo "=========================================="

# Store PIDs for monitoring
declare -a pids

# BASELINE TREATMENT (10 runs)
echo ""
echo "Launching BASELINE treatment (random patches)..."
for run in $(seq 1 $NUM_RUNS); do
    timeout 7200 python3 main.py \
        --max_iterations=$MAX_ITERATIONS \
        --llm_mode=none \
        --experiment_id=$EXPERIMENT_ID \
        --treatment=baseline \
        --run=$run \
        > logs/baseline_run_${run}.log 2>&1 &
    pid=$!
    pids+=($pid)
    echo "  Baseline run $run launched (PID: $pid)"
    sleep 2  # Stagger starts
done

# HF ROUTER TREATMENT (10 runs)
echo ""
echo "Launching HF_ROUTER treatment (LLM-guided patches)..."
if [ -z "$HF_TOKEN" ]; then
    echo "  WARNING: HF_TOKEN not set. HF treatment will fall back to MOCK mode."
fi

for run in $(seq 1 $NUM_RUNS); do
    timeout 7200 bash -c "
        HF_TOKEN='$HF_TOKEN' python3 main.py \
            --max_iterations=$MAX_ITERATIONS \
            --llm_mode=hf \
            --experiment_id=$EXPERIMENT_ID \
            --treatment=hf_router \
            --run=$run
    " > logs/hf_router_run_${run}.log 2>&1 &
    pid=$!
    pids+=($pid)
    echo "  HF Router run $run launched (PID: $pid)"
    sleep 2  # Stagger starts
done

echo ""
echo "All jobs launched! Waiting for completion..."
echo "Total jobs: $((NUM_RUNS * 2))"
echo ""

# Wait for all jobs to complete
failed=0
for pid in "${pids[@]}"; do
    if ! wait $pid; then
        failed=$((failed + 1))
    fi
done

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "Failed jobs: $failed / $((NUM_RUNS * 2))"
echo "Data location: experiments/$EXPERIMENT_ID/data/"
echo "=========================================="
echo ""
echo "Next step: Run analysis"
echo "  cd experiments/$EXPERIMENT_ID"
echo "  python analysis/analyze.py"
