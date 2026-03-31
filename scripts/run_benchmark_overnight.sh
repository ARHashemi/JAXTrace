#!/usr/bin/env bash
# =============================================================================
# Overnight L2 Benchmark Suite
#
# Runs all benchmark configurations and logs each to a separate file.
# Designed to run unattended; safe to leave overnight.
#
# Usage:
#   chmod +x scripts/run_benchmark_overnight.sh
#   nohup scripts/run_benchmark_overnight.sh &> logs/overnight_master.log &
#
# Or simply:
#   scripts/run_benchmark_overnight.sh 2>&1 | tee logs/overnight_master.log
# =============================================================================

set -euo pipefail

# --- Configuration (edit these) ----------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
BENCHMARK="$SCRIPT_DIR/benchmark_l2_accuracy.py"
LOGDIR="$SCRIPT_DIR/logs/overnight_$(date +%Y%m%d_%H%M%S)"

# Mesh path — adjust if different on the target machine
MESH_INPUT="${MESH_INPUT:-data/FLA/post}"

# Timing parameters
WARMUP=3
TIMING_RUNS=7

# Common args shared across all runs
COMMON_ARGS=(
    --input "$MESH_INPUT"
    --warmup-runs "$WARMUP"
    --timing-runs "$TIMING_RUNS"
    --seed 42
    --point-in-tet-tol 1e-6
)
# -----------------------------------------------------------------------------

mkdir -p "$LOGDIR"

echo "============================================================"
echo "Overnight L2 Benchmark Suite"
echo "============================================================"
echo "Start time : $(date)"
echo "Host       : $(hostname)"
echo "Python     : $($PYTHON --version 2>&1)"
echo "Log dir    : $LOGDIR"
echo "Mesh input : $MESH_INPUT"
echo "============================================================"
echo ""

# Helper: run one benchmark and log it
run_test() {
    local name="$1"
    shift
    local logfile="$LOGDIR/${name}.log"

    echo "------------------------------------------------------------"
    echo "[$(date +%H:%M:%S)] START: $name"
    echo "  Args: $*"
    echo "  Log:  $logfile"
    echo "------------------------------------------------------------"

    local t_start=$(date +%s)

    # Run benchmark; tee to both console (summary) and log file
    $PYTHON "$BENCHMARK" "${COMMON_ARGS[@]}" "$@" > "$logfile" 2>&1
    local exit_code=$?

    local t_end=$(date +%s)
    local elapsed=$((t_end - t_start))
    local minutes=$((elapsed / 60))
    local seconds=$((elapsed % 60))

    if [ $exit_code -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] DONE:  $name  (${minutes}m ${seconds}s)"
    else
        echo "[$(date +%H:%M:%S)] FAILED: $name  (exit code $exit_code, ${minutes}m ${seconds}s)"
        echo "  See $logfile for details"
    fi
    echo ""
}

# =============================================================================
# Test 1: Full accuracy benchmark (N=10k, all perturbations, all intra-element)
#
# This is the primary paper result: found rate, accuracy, PIT tests,
# level distribution, performance metrics.
# =============================================================================
run_test "01_accuracy_full_10k" \
    --n-particles 10000 \
    --perturbations 0.0 \
    --position-types centroid random near_face near_edge near_vertex

# =============================================================================
# Test 2: Multi-perturbation sweep (N=10k)
#
# Tests robustness across perturbation levels 0.0 to 4.0x.
# Skips intra-element (already covered in test 1) for speed.
# =============================================================================
run_test "02_perturbation_sweep_10k" \
    --n-particles 10000 \
    --perturbations 0.0 0.1 0.2 0.5 0.7 1.0 2.0 4.0 \
    --skip-intra

# =============================================================================
# Test 3: Scalability sweep (3x3x3 only, N_p = 1k to 100k)
#
# Measures throughput scaling: queries/s, us/query, PIT tests/s.
# This is the key data for the performance section.
# =============================================================================
run_test "03_scalability_sweep" \
    --n-particles 10000 \
    --perturbations 0.0 \
    --skip-intra \
    --skip-failure-analysis \
    --scalability \
    --scalability-sizes 1000 2000 5000 10000 20000 50000 100000

# =============================================================================
# Test 4: Large-batch scalability (push to 200k/500k if GPU memory allows)
#
# Tests whether throughput saturates at high N_p.
# May OOM on GPUs with <8GB — reduce batch-size if needed.
# =============================================================================
run_test "04_scalability_large" \
    --n-particles 10000 \
    --perturbations 0.0 \
    --skip-intra \
    --skip-failure-analysis \
    --scalability \
    --scalability-sizes 100000 200000 500000 \
    --batch-size 100000

# =============================================================================
# Test 5: Restricted seeding region (refined zone near weld)
#
# Same region used in the previous benchmark logs.
# =============================================================================
run_test "05_accuracy_weld_region" \
    --n-particles 10000 \
    --perturbations 0.0 \
    --seed-x 0.08 0.38 \
    --seed-y 0.25 0.75 \
    --seed-z 0.50 1.00 \
    --position-types centroid random near_face near_edge near_vertex

# =============================================================================
# Done
# =============================================================================
echo "============================================================"
echo "All tests complete!"
echo "End time: $(date)"
echo "Logs in:  $LOGDIR"
echo ""
echo "Log files:"
ls -lh "$LOGDIR"/*.log 2>/dev/null
echo "============================================================"
