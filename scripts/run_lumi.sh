#!/bin/bash
#SBATCH --job-name=jaxtrace
#SBATCH --partition=small-g
#SBATCH --account=project_465002752
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project_465002752/hashemia/logs/%x_%j.out
#SBATCH --error=/scratch/project_465002752/hashemia/logs/%x_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif
JAXTRACE=/project/project_465002752/hashemia/JAXTrace
PKGS=/project/project_465002752/hashemia/required-packages
INPUT=/scratch/project_465001942/Cases-Edgar/new/cylA.gid/post

# Fast NVMe for active writing
FLASH_OUT=/flash/project_465002752/hashemia/run_$SLURM_JOB_ID
# Final long-term storage
SCRATCH_OUT=/scratch/project_465002752/hashemia/outputs/run_$SLURM_JOB_ID
# Monitoring log
MONITOR_LOG=/scratch/project_465002752/hashemia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_monitor.log

mkdir -p $FLASH_OUT $SCRATCH_OUT

# ── MIOpen cache to RAM (avoids slow disk I/O for kernel tuning DB) ───────────
export MIOPEN_USER_DB_PATH="/tmp/hashemia-miopen-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

# ── ROCm / XLA performance flags for MI250X on LUMI-G ────────────────────────
# Restrict JAX to ROCm only — avoids CPU fallback overhead when cuda is listed
export JAX_PLATFORMS=rocm

# Use only 1 GCD (half of MI250X die) — JAX sees it as 1 GPU.
# ROCR_VISIBLE_DEVICES=0 picks the first GCD. All 64 GB HBM2e available.
export ROCR_VISIBLE_DEVICES=0

# XLA flags: higher autotuning level for ROCm kernels (default is 3 on ROCm).
# Level 4 tries more tile configs; adds ~30s to compilation but can give 20%+
# speedup on fori_loop-heavy kernels that use GEMM-like patterns.
export XLA_FLAGS="--xla_gpu_autotune_level=4 --xla_gpu_enable_latency_hiding_scheduler=true"

# Prevent XLA from preallocating all HBM — use on-demand allocation.
# Already set in Python via os.environ, but set here too for subprocess safety.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Disable MIOpen find-mode on first run (uses tuning DB we just pointed to RAM).
# FIND_MODE=3 = FIND_ENFORCE (always search); use 1 (NORMAL) for production.
export MIOPEN_FIND_MODE=1

# Suppress TF/XLA C++ log spam
export TF_CPP_MIN_LOG_LEVEL=2

# HIP heap size for large temporary buffers during XLA fusion
export HSA_ENABLE_SDMA=0          # Disable SDMA — GPU→CPU copies via blit engine
                                  # faster for large buffers on MI250X

# ── GPU & Memory Monitor (background) ────────────────────────────────────────
# Logs GPU utilization, VRAM, and host memory every 30s
(
  echo "=== GPU & Memory Monitor === Job $SLURM_JOB_ID === $(date) ==="
  echo ""
  while true; do
    echo "--- $(date '+%Y-%m-%d %H:%M:%S') ---"

    # AMD GPU stats via rocm-smi
    if command -v rocm-smi &>/dev/null; then
      rocm-smi --showuse --showmemuse --showtemp 2>/dev/null | grep -E 'GPU|%|MiB|Temperature' || \
      rocm-smi 2>/dev/null | tail -n +3
    fi

    # Host memory
    echo ""
    free -h | head -2
    echo ""

    # Disk usage on flash output
    if [ -d "$FLASH_OUT" ]; then
      N_VTU=$(find $FLASH_OUT -name '*.vtu' 2>/dev/null | wc -l)
      DISK_USAGE=$(du -sh $FLASH_OUT 2>/dev/null | cut -f1)
      echo "Output: $N_VTU VTU files, $DISK_USAGE on flash"
    fi

    echo ""
    sleep 30
  done
) > $MONITOR_LOG 2>&1 &
MONITOR_PID=$!

# ── Run simulation → writes to Flash ──────────────────────────────────────────
echo "Starting simulation at $(date)"
echo "Monitor log: $MONITOR_LOG"
echo ""

# All defaults now match FEMUSS behavior:
#   --failed-substage zero_vel   (k[i]=0 for failed substages)
#   --levelset-mode zero_vel     (zero velocity inside tool)
#   --bbox-clamp OFF             (no substep bbox clamping)
#   --boundary-proj ON           (boundary projection recovery)
#   --boundary-proj-tol 1e-6     (FEMUSS tolerance)
#   --point-in-tet-tol 1e-6      (FEMUSS tolerance)
#   --pin-velocity ON            (reconstruct composite velocity field)
#   --pin-rpm -600               (FEMUSS PROCESS_PARAMETERS RPM)
#   --l0-skip-boundary ON        (skip L0 cache for mixed-LS elements, fresh search like FEMUSS)
#   --l0-skip-band 0             (L0 skip band: 0=mixed-sign only, e.g. 0.5e-3 for ±0.5mm)
#   --enhanced-search-band 0     (0=off; e.g. 1e-3 for ±1mm node-L1+5x5x5 band)
#   --l1-method face             (global L1; 'node' for all-node; or use --enhanced-search-band)
#   --l2-neighborhood 3          (global L2; 5 for all-5x5x5; or use --enhanced-search-band)
srun --gpus-per-task=1 \
  singularity exec --cleanenv \
  --env PYTHONPATH=$JAXTRACE:$PKGS \
  --env JAX_PLATFORMS=$JAX_PLATFORMS \
  --env ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
  --env XLA_FLAGS="$XLA_FLAGS" \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE \
  --env XLA_PYTHON_CLIENT_ALLOCATOR=$XLA_PYTHON_CLIENT_ALLOCATOR \
  --env MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH \
  --env MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_CUSTOM_CACHE_DIR \
  --env MIOPEN_FIND_MODE=$MIOPEN_FIND_MODE \
  --env TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL \
  --env HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA \
  $SIF \
  python $JAXTRACE/benchmark_femuss_comparison.py \
    --input  $INPUT \
    --output $FLASH_OUT \
    --precision float32 \
    --boundary-walls "x_max=outlet,y_min=outlet" \
    --n-groups 5

SIM_EXIT=$?

# ── Stop monitor ──────────────────────────────────────────────────────────────
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

echo ""
echo "Simulation exited with code $SIM_EXIT at $(date)"

# ── Move results to Scratch after job ─────────────────────────────────────────
echo "Moving results to scratch..."
mv $FLASH_OUT/* $SCRATCH_OUT/ 2>/dev/null
rmdir $FLASH_OUT 2>/dev/null
echo "Done. Results in $SCRATCH_OUT"
