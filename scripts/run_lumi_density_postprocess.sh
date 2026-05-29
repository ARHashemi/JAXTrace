#!/bin/bash
# SLURM batch script: offline density-field post-processing on LUMI.
# Drives run_density_postprocess.py against an already-written
# particles.vtkhdf file. Override at submit time, e.g.:
#   sbatch --account=project_XXXXXXXXX \
#          --export=ALL,PARTICLES=/scratch/.../particles.vtkhdf \
#          scripts/run_lumi_density_postprocess.sh
#
#SBATCH --job-name=jt-density-pp
#SBATCH --partition=small-g
#SBATCH --account=project_XXXXXXXXX
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# ── [1] Paths ────────────────────────────────────────────────────────────────
PROJECT="${SLURM_JOB_ACCOUNT:-project_XXXXXXXXX}"

# Singularity image (same as run_lumi.sh).
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif

JAXTRACE="/project/${PROJECT}/${USER}/JAXTrace"
PKGS="/project/${PROJECT}/${USER}/required-packages"

# REQUIRED: path to the particles.vtkhdf to process. Override via
# `--export=ALL,PARTICLES=...` on the sbatch line, or edit below.
PARTICLES="${PARTICLES:-/scratch/${PROJECT}/${USER}/outputs/<RUN>/particles.vtkhdf}"

# OUTPUT_DIR: destination directory for density_*.vtkhdf / .vti and the
# time-averaged file. Defaults to a sibling 'density' folder next to PARTICLES.
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$PARTICLES")/density}"

# Optional: a velocity mesh PVTU/PVD used ONLY for inside-mesh masking. Leave
# empty to disable masking (the voxel grid then evaluates everywhere within
# the bounding box).
VELOCITY_MESH=""                  # e.g. /scratch/.../<case>.gid/post/0eule/<case>_159.pvtu

# ── [2] Voxel grid ───────────────────────────────────────────────────────────
# BOUNDS_MODE picks how the grid bounding box is determined:
#   prepass  -- 2-pass read of PARTICLES to take the union over all timesteps.
#   mesh     -- use the velocity-mesh bbox (requires VELOCITY_MESH set).
#   explicit -- use BOUNDS below verbatim.
BOUNDS_MODE=prepass               # prepass | mesh | explicit
BOUNDS=""                         # "XMIN XMAX YMIN YMAX ZMIN ZMAX" (for explicit)
RESOLUTION=128                    # cubic resolution; ignored if VOXEL_SIZE set
VOXEL_SIZE=""                     # absolute voxel edge length [m]; overrides RESOLUTION
PAD_FRACTION=0.0
NO_MASK_INSIDE_MESH=0             # 1 = skip masking (faster; lower fidelity near boundary)

# ── [3] Kernel / bandwidth ───────────────────────────────────────────────────
KERNEL=wendland_c2                # wendland_c2|wendland_c4|cubic_spline|gaussian|epanechnikov|quintic_spline
BANDWIDTH_MODE=fixed              # fixed | scott | silverman | knn_adaptive
BANDWIDTH=""                      # explicit h (fixed mode); "" = factor * voxel_size
BANDWIDTH_FACTOR=2.0
BANDWIDTH_REFRESH_EVERY=0
KNN_K=32
KNN_SAFETY=1.2

# ── [4] Engine ───────────────────────────────────────────────────────────────
ENGINE=auto                       # auto | brute | octree
AUTO_THRESHOLD=5e10
BRUTE_QUERY_CHUNK=8192
OCTREE_CELLS_PER_DIM=64
OCTREE_MAX_NEIGHBORS=256
PARTICLE_BUCKET=4096

# ── [5] Output toggles ───────────────────────────────────────────────────────
OUTPUT_FORMAT=vtkhdf              # vtkhdf | vti
FILENAME_STEM=density
NO_PER_STEP=0
NO_TIME_AVERAGE=0
NO_PARTICLE_DENSITY=0
NORMALIZATION=pdf                 # pdf | mass | unnormalized

# Subsample the trajectory (cheap way to reduce wall time on large files).
STEP_STRIDE=1
MAX_STEPS=""                      # "" = all
STEP_RANGE=""                     # "START END" -- process steps in [START, END)
STEP_TAIL=""                      # process the LAST N steps (overrides STEP_RANGE / MAX_STEPS)

# Compression: gzip | lzf | blosc | none.
COMPRESSION=gzip                  # gzip | lzf | blosc | none. ParaView only
                                  # reads gzip; lzf/blosc need a custom HDF5.
COMPRESSION_OPTS=1
BLOSC_THREADS=4

# Background trajectory reader queue depth. 0 disables prefetch.
READ_PREFETCH=4

# ── [6] Performance ──────────────────────────────────────────────────────────
XLA_PREALLOC=0                    # 1 = preallocate ~75% HBM at startup
MONITOR_INTERVAL=30
BENCHMARK_MODE=0

# =============================================================================
# END USER CONFIGURATION
# =============================================================================

_LOCAL_OVERRIDES="$(dirname "$0")/run_lumi_density_postprocess.local.sh"
if [ -f "$_LOCAL_OVERRIDES" ]; then
    echo "[config] Sourcing local overrides: $_LOCAL_OVERRIDES"
    # shellcheck source=/dev/null
    source "$_LOCAL_OVERRIDES"
fi

if [ ! -f "$PARTICLES" ]; then
    echo "ERROR: PARTICLES not found: $PARTICLES" >&2
    echo "Set PARTICLES in this script or via --export=ALL,PARTICLES=..." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
MONITOR_LOG="${OUTPUT_DIR}/density_pp_${SLURM_JOB_ID}_monitor.log"

# ── MIOpen cache to RAM ──────────────────────────────────────────────────────
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

# ── ROCm / XLA flags ─────────────────────────────────────────────────────────
export JAX_PLATFORMS=rocm
export ROCR_VISIBLE_DEVICES=0
export XLA_FLAGS="--xla_gpu_autotune_level=4 --xla_gpu_enable_latency_hiding_scheduler=true"
if [ "$BENCHMARK_MODE" = "1" ] || [ "$XLA_PREALLOC" = "1" ]; then
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    unset  XLA_PYTHON_CLIENT_ALLOCATOR
    echo "[perf] XLA preallocating allocator ENABLED"
else
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
fi
export MIOPEN_FIND_MODE=1
export TF_CPP_MIN_LOG_LEVEL=2
export HSA_ENABLE_SDMA=0
export PYTHONUNBUFFERED=1

# ── GPU & Memory Monitor (background) ────────────────────────────────────────
MONITOR_PID=""
MONITOR_PGID=""
if [ "$BENCHMARK_MODE" != "1" ] && [ "$MONITOR_INTERVAL" -gt 0 ] 2>/dev/null; then
  setsid bash -c '
    echo "=== Density PP Monitor === Job '"$SLURM_JOB_ID"' === $(date) ==="
    while true; do
      echo "--- $(date '\''+%Y-%m-%d %H:%M:%S'\'') ---"
      command -v rocm-smi &>/dev/null && rocm-smi --showuse --showmemuse --showtemp 2>/dev/null \
        | grep -E "GPU|%|MiB|Temperature" || true
      free -h | head -2
      echo ""
      sleep '"${MONITOR_INTERVAL}"'
    done
  ' > "$MONITOR_LOG" 2>&1 &
  MONITOR_PID=$!
  MONITOR_PGID=$MONITOR_PID
fi
_cleanup_monitor() {
  if [ -n "${MONITOR_PGID:-}" ]; then
    kill -- -"$MONITOR_PGID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap _cleanup_monitor EXIT

# ── Build CLI argument list ─────────────────────────────────────────────────
ARGS=(
    --particles      "$PARTICLES"
    --output-dir     "$OUTPUT_DIR"
    --output-format  "$OUTPUT_FORMAT"
    --filename-stem  "$FILENAME_STEM"
    --resolution     "$RESOLUTION"
    --pad-fraction   "$PAD_FRACTION"
    --kernel         "$KERNEL"
    --bandwidth-mode "$BANDWIDTH_MODE"
    --bandwidth-factor "$BANDWIDTH_FACTOR"
    --bandwidth-refresh-every "$BANDWIDTH_REFRESH_EVERY"
    --knn-k          "$KNN_K"
    --knn-safety     "$KNN_SAFETY"
    --engine         "$ENGINE"
    --auto-threshold "$AUTO_THRESHOLD"
    --brute-query-chunk    "$BRUTE_QUERY_CHUNK"
    --octree-cells-per-dim "$OCTREE_CELLS_PER_DIM"
    --octree-max-neighbors "$OCTREE_MAX_NEIGHBORS"
    --particle-bucket "$PARTICLE_BUCKET"
    --normalization  "$NORMALIZATION"
    --step-stride    "$STEP_STRIDE"
    --compression       "$COMPRESSION"
    --compression-opts  "$COMPRESSION_OPTS"
    --blosc-threads     "$BLOSC_THREADS"
    --read-prefetch     "$READ_PREFETCH"
)
[ -n "$BANDWIDTH" ]      && ARGS+=( --bandwidth   "$BANDWIDTH" )
[ -n "$VOXEL_SIZE" ]     && ARGS+=( --voxel-size  "$VOXEL_SIZE" )
[ -n "$VELOCITY_MESH" ]  && ARGS+=( --velocity-mesh "$VELOCITY_MESH" )
[ -n "$BOUNDS" ]         && ARGS+=( --bounds $BOUNDS )
case "$BOUNDS_MODE" in
    explicit)
        if [ -z "$BOUNDS" ]; then
            echo "ERROR: BOUNDS_MODE=explicit requires BOUNDS to be set." >&2
            exit 2
        fi
        ;;
    mesh)
        if [ -z "$VELOCITY_MESH" ]; then
            echo "ERROR: BOUNDS_MODE=mesh requires VELOCITY_MESH to be set." >&2
            exit 2
        fi
        ARGS+=( --bounds-from mesh )
        ;;
    prepass) ;;
    *)
        echo "ERROR: unknown BOUNDS_MODE='$BOUNDS_MODE' (expected prepass|mesh|explicit)" >&2
        exit 2
        ;;
esac
[ "$NO_MASK_INSIDE_MESH"  = "1" ] && ARGS+=( --no-mask-inside-mesh )
[ "$NO_PER_STEP"          = "1" ] && ARGS+=( --no-per-step )
[ "$NO_TIME_AVERAGE"      = "1" ] && ARGS+=( --no-time-average )
[ "$NO_PARTICLE_DENSITY"  = "1" ] && ARGS+=( --no-particle-density )
[ -n "$MAX_STEPS" ]               && ARGS+=( --max-steps "$MAX_STEPS" )
[ -n "$STEP_RANGE" ]              && ARGS+=( --step-range $STEP_RANGE )
[ -n "$STEP_TAIL" ]               && ARGS+=( --step-tail "$STEP_TAIL" )

ALLOC_ENV=()
if [ -n "${XLA_PYTHON_CLIENT_ALLOCATOR:-}" ]; then
    ALLOC_ENV=( --env "XLA_PYTHON_CLIENT_ALLOCATOR=$XLA_PYTHON_CLIENT_ALLOCATOR" )
fi

echo "Starting density post-processing at $(date)"
echo "  Particles: $PARTICLES"
echo "  Output:    $OUTPUT_DIR"
echo "  CLI args:  ${ARGS[*]}"
echo ""

# Forward SIGTERM (from #SBATCH --signal=B:SIGTERM@120) to the srun child.
_forward_sigterm() {
    echo "[trap] Forwarding SIGTERM to srun step (PID $SRUN_PID)..."
    if [ -n "${SRUN_PID:-}" ]; then
        scancel --signal=TERM --batch ${SLURM_JOB_ID:-} 2>/dev/null
        kill -TERM "$SRUN_PID" 2>/dev/null
    fi
}
trap _forward_sigterm SIGTERM

srun --gpus-per-task=1 \
    singularity exec --cleanenv \
    --env PYTHONPATH=$JAXTRACE:$PKGS \
    --env JAX_PLATFORMS=$JAX_PLATFORMS \
    --env ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
    --env XLA_FLAGS="$XLA_FLAGS" \
    --env XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE \
    "${ALLOC_ENV[@]}" \
    --env MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH \
    --env MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_CUSTOM_CACHE_DIR \
    --env MIOPEN_FIND_MODE=$MIOPEN_FIND_MODE \
    --env TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL \
    --env HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA \
    --env PYTHONUNBUFFERED=$PYTHONUNBUFFERED \
    $SIF \
    python $JAXTRACE/run_density_postprocess.py "${ARGS[@]}" &
SRUN_PID=$!
wait $SRUN_PID
PP_EXIT=$?

_cleanup_monitor

echo ""
echo "Density post-processing exited with code $PP_EXIT at $(date)"

# Stash the monitor log next to the outputs for traceability.
[ -f "$MONITOR_LOG" ] && mv "$MONITOR_LOG" "${OUTPUT_DIR}/logs/" 2>/dev/null || true

exit $PP_EXIT
