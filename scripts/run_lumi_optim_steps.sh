#!/bin/bash
# SLURM directives — fill in your project ID and log directory before
# submitting, or override at submission time:
#   sbatch --account=project_XXXXXXXXX --output=... run_lumi_optim_steps.sh
#SBATCH --job-name=jaxtrace_optim
#SBATCH --partition=small-g
#SBATCH --account=project_XXXXXXXXX
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT="${SLURM_JOB_ACCOUNT:-project_XXXXXXXXX}"
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif
JAXTRACE="/project/${PROJECT}/${USER}/JAXTrace"
PKGS="/project/${PROJECT}/${USER}/required-packages"
INPUT="/scratch/${PROJECT}/${USER}/data/<CASE>.gid/post"

FLASH_BASE="/flash/${PROJECT}/${USER}/optim_${SLURM_JOB_ID}"
SCRATCH_BASE="/scratch/${PROJECT}/${USER}/outputs/optim_${SLURM_JOB_ID}"
MONITOR_LOG="/scratch/${PROJECT}/${USER}/logs/jaxtrace_optim_${SLURM_JOB_ID}_monitor.log"

mkdir -p $FLASH_BASE $SCRATCH_BASE

# ── Environment ───────────────────────────────────────────────────────────────
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

export JAX_PLATFORMS=rocm
export ROCR_VISIBLE_DEVICES=0
export XLA_FLAGS="--xla_gpu_autotune_level=4 --xla_gpu_enable_latency_hiding_scheduler=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MIOPEN_FIND_MODE=1
export TF_CPP_MIN_LOG_LEVEL=2
export HSA_ENABLE_SDMA=0

# ── Monitor (background) ──────────────────────────────────────────────────────
(
  echo "=== GPU & Memory Monitor === Job $SLURM_JOB_ID === $(date) ==="
  while true; do
    echo "--- $(date '+%Y-%m-%d %H:%M:%S') ---"
    if command -v rocm-smi &>/dev/null; then
      rocm-smi --showuse --showmemuse --showtemp 2>/dev/null | grep -E 'GPU|%|MiB|Temperature'
    fi
    free -h | head -2
    echo ""
    sleep 30
  done
) > $MONITOR_LOG 2>&1 &
MONITOR_PID=$!

# ── Common srun wrapper ───────────────────────────────────────────────────────
run_step() {
    local STEP_NAME=$1
    local EXTRA_ARGS=$2
    local OUT_DIR=$FLASH_BASE/$STEP_NAME

    mkdir -p $OUT_DIR

    echo ""
    echo "========================================================================"
    echo "STEP: $STEP_NAME"
    echo "ARGS: $EXTRA_ARGS"
    echo "OUTPUT: $OUT_DIR"
    echo "Started: $(date)"
    echo "========================================================================"

    srun --gpus-per-task=1 \
      singularity exec --cleanenv \
      --env PYTHONPATH=$JAXTRACE:$PKGS \
      --env JAX_PLATFORMS=$JAX_PLATFORMS \
      --env ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
      --env "XLA_FLAGS=$XLA_FLAGS" \
      --env XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE \
      --env XLA_PYTHON_CLIENT_ALLOCATOR=$XLA_PYTHON_CLIENT_ALLOCATOR \
      --env MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH \
      --env MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_CUSTOM_CACHE_DIR \
      --env MIOPEN_FIND_MODE=$MIOPEN_FIND_MODE \
      --env TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL \
      --env HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA \
      $SIF \
      python $JAXTRACE/benchmark_femuss_comparison.py \
        --input   $INPUT \
        --output  $OUT_DIR \
        --precision float32 \
        --femuss-start 500 \
        --n-steps 100 \
        --log-interval 10 \
        --export-freq 100 \
        --boundary-walls "x_max=outlet,y_min=outlet" \
        --n-groups 0 \
        $EXTRA_ARGS

    local EXIT=$?
    echo "Finished: $(date)  exit=$EXIT"
    return $EXIT
}

# ── Step 1: Baseline ──────────────────────────────────────────────────────────
# Validated fused kernel, default fori_loop L2, no changes.
# Expected: ~1.56 s/step (RTX 5000 reference).  LUMI baseline.
run_step "step1_baseline" ""

# ── Step 2: Module-level _CELL_OFFSETS_3x3x3 fix ─────────────────────────────
# Same fused kernel, but _CELL_OFFSETS_3x3x3 is now a module-level constant
# instead of being re-created inside the fori_loop body at every trace.
# Expected: same throughput or slightly better (constant hoisting only).
run_step "step2_const_offsets" ""
# Note: the fix is in jaxtrace/gpu/search/mesh_aligned_point_location.py —
# no extra CLI flag needed; it is always active on this branch.

# ── Step 3: Vectorized L2 (gather + parallel PIT) inside fused mode ──────────
# Replaces the carry-dependent early-exit fori_loop in L2 with a flat
# gather of up to 512 candidates followed by jax.vmap(test_one).
# Expected: faster if XLA can parallelise the 512 PIT tests better on MI250X.
# May be slower if the dynamic copy loop in gather_l2_candidates is the bottleneck.
run_step "step3_l2_vectorized" "--l2-vectorized"

# ── Step 4: Split kernel (separate vmap per L0/L1/L2/interp per stage) ────────
# Each RK4 stage launches L0, L1, L2, interp as independent vmap'd kernels.
# This avoids any nested vmap/lax.cond interaction but increases kernel launches.
# Uses the default (non-vectorized) fori_loop L2 path to isolate the split effect.
# Expected: unknown — may be faster (independent register budgets) or slower
# (more kernel launch overhead, dynamic loop still present).
run_step "step4_split_kernel" "--rk4-mode split"

# ── Cleanup ───────────────────────────────────────────────────────────────────
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

echo ""
echo "All steps finished at $(date)"
echo "Moving results to scratch..."
mv $FLASH_BASE/* $SCRATCH_BASE/ 2>/dev/null
rmdir $FLASH_BASE 2>/dev/null
echo "Done. Results in $SCRATCH_BASE"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "THROUGHPUT SUMMARY"
echo "========================================================================"
for STEP in step1_baseline step2_const_offsets step3_l2_vectorized step4_split_kernel; do
    LOG=$(ls $SCRATCH_BASE/$STEP/*.out 2>/dev/null | head -1)
    if [ -z "$LOG" ]; then
        # Try the SLURM stdout which goes to the main .out file
        echo "$STEP: see main log"
    else
        TPUT=$(grep "p\*step/s\|p.step/s" $LOG | tail -1)
        echo "$STEP: $TPUT"
    fi
done
