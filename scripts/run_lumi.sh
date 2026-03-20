#!/bin/bash
#SBATCH --job-name=jaxtrace
#SBATCH --partition=small-g
#SBATCH --account=project_465002752
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
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

# ── MIOpen cache to RAM ────────────────────────────────────────────────────────
export MIOPEN_USER_DB_PATH="/tmp/hashemia-miopen-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

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
#   --l1-method face             (face-based L1; use 'node' for wider search)
#   --l2-neighborhood 3          (3x3x3 L2; use 5 for 5x5x5 wider search)
srun singularity exec --cleanenv \
  --env PYTHONPATH=$JAXTRACE:$PKGS \
  $SIF \
  python $JAXTRACE/benchmark_femuss_comparison.py \
    --input  $INPUT \
    --output $FLASH_OUT \
    --l1-method node \
    --l2-neighborhood 5

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
