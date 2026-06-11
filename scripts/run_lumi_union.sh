#!/bin/bash -l
# =============================================================================
# run_lumi_union.sh — JAXTrace density-union (deduplication + optional KDE) on LUMI
#
# Drives run_density_union.py to build the union/deduplicated material
# distribution from a particles.vtkhdf trajectory.
#
# Two invocation modes (the script detects which by checking $SLURM_JOB_ID):
#
#   Standalone:        sbatch --account=project_XXXXXXXXX \
#                        --export=ALL,PARTICLES=/scratch/.../particles.vtkhdf \
#                        run_lumi_union.sh
#                      The #SBATCH directives below allocate a fresh job.
#
#   Hook from run_lumi.sh: bash run_lumi_union.sh
#                      Runs inside the parent's allocation; #SBATCH
#                      directives are ignored by bash. srun re-uses the
#                      enclosing allocation automatically.
#
# Outputs (under OUTPUT_DIR):
#   <stem>_union.vtkhdf            -- deduplicated PolyData point cloud
#   <stem>_union.npy               -- optional NumPy mirror (WRITE_NPY=1)
#   <stem>_union_density.vtkhdf    -- optional voxel-grid density on the
#                                     unified cloud (WRITE_DENSITY=1)
# =============================================================================
#SBATCH --job-name=jt-union
#SBATCH --partition=small-g
#SBATCH --account=project_XXXXXXXXX
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# ── [1] Paths ────────────────────────────────────────────────────────────────
PROJECT="${SLURM_JOB_ACCOUNT:-project_XXXXXXXXX}"

SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif
JAXTRACE="/project/${PROJECT}/${USER}/JAXTrace"
PKGS="/project/${PROJECT}/${USER}/required-packages"

# PARTICLES: input particles.vtkhdf produced by run_tracking. When unset,
# we auto-find the latest run_*/particles.vtkhdf inside the case folder's
# post_pt/ subdir (matches the layout run_lumi.sh writes when
# OUTPUT_TARGET=case). Override with --export=ALL,PARTICLES=...
# OUTPUT_CASE_SUBFOLDER must match what run_lumi.sh used.
CASE_DIR="${CASE_DIR:-$(dirname "$(readlink -f "$0")")}"
OUTPUT_CASE_SUBFOLDER="${OUTPUT_CASE_SUBFOLDER:-post_pt}"
if [ -z "${PARTICLES:-}" ]; then
    PARTICLES="$(ls -dt "$CASE_DIR/$OUTPUT_CASE_SUBFOLDER"/run_*/particles.vtkhdf 2>/dev/null | head -1)"
fi

# OUTPUT_DIR: FINAL destination for the union outputs. Defaults to a
# sibling 'union' folder next to the particles file.
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$PARTICLES")/union}"

# FLASH_DIR: NVMe staging area on LUMI. Active per-step writes hit here
# during the run; results are rsync'd to OUTPUT_DIR at the end. Empty
# disables staging.
FLASH_DIR="${FLASH_DIR:-/flash/${PROJECT}/${USER}}"

FILENAME_STEM=particles

# ── [2] Step selection ──────────────────────────────────────────────────────
STEP_STRIDE=1
MAX_STEPS=""                      # "" = all
STEP_RANGE=""                     # "START END" -- process steps in [START, END)
STEP_TAIL=""                      # process the LAST N steps

# ── [3] Dedup ───────────────────────────────────────────────────────────────
# Modes:
#   batch       -- concatenate all selected steps, dedup once at the end.
#                  Fast; memory ∝ N_total. Default.
#   incremental -- per step, drop particles already covered by survivors of
#                  previous steps. Smaller peak memory.
#   none        -- emit raw concatenation, no dedup (large file).
DEDUP_MODE=batch

# Per-axis dedup tolerance in metres. Empty ⇒ TOLERANCE_FRACTION × Δp_axis
# from the step-0 seeding (anisotropic).
TOLERANCE=""
TOLERANCE_FRACTION=0.01

# ── [4] PointData propagation ───────────────────────────────────────────────
NO_POINT_DATA=0
FIELDS=""
REDUCE_MAX_FIELDS=""

# ── [5] Region of interest ──────────────────────────────────────────────────
ROI_FRACTION="0.5 1.0 0.0 1.0 0.0 1.0"
ROI_BOX=""

# ── [6] Output toggles ──────────────────────────────────────────────────────
NO_WRITE_CLOUD=0
WRITE_NPY=0
WRITE_DENSITY=1

# ── [7] Density pass (only if WRITE_DENSITY=1) ──────────────────────────────
KERNEL=wendland_c2
BANDWIDTH_MODE=initial_spacing
BANDWIDTH=""
BANDWIDTH_XYZ=""
BANDWIDTH_FACTOR=1.5
RESOLUTION=128
RESOLUTION_XYZ=""
VOXEL_SIZE=""
VOXEL_SIZE_XYZ=""
VOXEL_SIZE_FROM_PARTICLES=1
NORMALIZATION=pdf

# ── [8] Compression ─────────────────────────────────────────────────────────
COMPRESSION=gzip
COMPRESSION_OPTS=1

# ── [9] JAX memory & performance ────────────────────────────────────────────
XLA_PREALLOC=0
VRAM_FRACTION=0.95
MONITOR_INTERVAL=100
BENCHMARK_MODE=0

# =============================================================================
# END USER CONFIGURATION
# =============================================================================

# ── Local overrides ──────────────────────────────────────────────────────────
_LOCAL_OVERRIDES="$(dirname "$0")/run_lumi_union.local.sh"
if [ -f "$_LOCAL_OVERRIDES" ]; then
    echo "[config] Sourcing local overrides: $_LOCAL_OVERRIDES"
    # shellcheck source=/dev/null
    source "$_LOCAL_OVERRIDES"
fi

if [ -z "$PARTICLES" ] || [ ! -f "$PARTICLES" ]; then
    echo "ERROR: PARTICLES not found." >&2
    echo "  CASE_DIR=$CASE_DIR" >&2
    echo "  OUTPUT_CASE_SUBFOLDER=$OUTPUT_CASE_SUBFOLDER" >&2
    echo "  PARTICLES='$PARTICLES'" >&2
    echo "Set PARTICLES via --export=ALL,PARTICLES=... or in run_lumi_union.local.sh." >&2
    exit 2
fi

RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/logs"

# ── Flash staging ────────────────────────────────────────────────────────────
# Per-step (and per-cell) writes can be heavy; stage them on /flash NVMe
# and rsync to the case folder at the end. Same pattern as run_lumi.sh.
if [ -n "$FLASH_DIR" ] && mkdir -p "$FLASH_DIR" 2>/dev/null && [ -w "$FLASH_DIR" ]; then
    _FLASH_RUN_DIR="${FLASH_DIR}/union_${RUN_ID}"
    mkdir -p "$_FLASH_RUN_DIR"
    EFFECTIVE_OUTPUT_DIR="$_FLASH_RUN_DIR"
    echo "[stage] writing union to flash: $EFFECTIVE_OUTPUT_DIR"
    echo "[stage] will rsync to final dir at end: $OUTPUT_DIR"
else
    EFFECTIVE_OUTPUT_DIR="$OUTPUT_DIR"
    echo "[stage] no flash staging (FLASH_DIR='$FLASH_DIR'); writing directly to $OUTPUT_DIR"
fi

MONITOR_LOG="${OUTPUT_DIR}/logs/density_union_${RUN_ID}_monitor.log"
RUN_LOG="${OUTPUT_DIR}/logs/density_union_${RUN_ID}.log"

if [ "${__JAXTRACE_UNION_LOG_ATTACHED:-}" != "1" ]; then
    export __JAXTRACE_UNION_LOG_ATTACHED=1
    exec > >(tee -a "$RUN_LOG") 2>&1
    echo "[log] Mirroring full output to $RUN_LOG"
fi

# ── MIOpen cache to RAM ──────────────────────────────────────────────────────
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${RUN_ID}"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

# ── Environment ──────────────────────────────────────────────────────────────
export JAX_PLATFORMS=rocm
export ROCR_VISIBLE_DEVICES=0
export XLA_FLAGS="--xla_gpu_autotune_level=4 --xla_gpu_enable_latency_hiding_scheduler=true"
if [ "$BENCHMARK_MODE" = "1" ]; then XLA_PREALLOC=1; fi
if [ "$XLA_PREALLOC" = "1" ]; then
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION=$VRAM_FRACTION
    unset  XLA_PYTHON_CLIENT_ALLOCATOR
    echo "[perf] XLA_PREALLOC=ON  — reserving ${VRAM_FRACTION} of VRAM at startup"
else
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
    export XLA_PYTHON_CLIENT_MEM_FRACTION=$VRAM_FRACTION
    echo "[perf] XLA_PREALLOC=OFF — on-demand allocator, cap: ${VRAM_FRACTION} VRAM"
fi
export MIOPEN_FIND_MODE=1
export TF_CPP_MIN_LOG_LEVEL=2
export HSA_ENABLE_SDMA=0
export PYTHONUNBUFFERED=1

# ── GPU & memory monitor (background) ───────────────────────────────────────
MONITOR_PID=""
MONITOR_PGID=""
if [ "$WRITE_DENSITY" = "1" ] && [ "$BENCHMARK_MODE" != "1" ] && [ "${MONITOR_INTERVAL}" -gt 0 ] 2>/dev/null; then
    setsid bash -c '
        echo "=== Density Union Monitor === Run '"${RUN_ID}"' === $(date) ==="
        while true; do
            echo "--- $(date '\''+%Y-%m-%d %H:%M:%S'\'') ---"
            if command -v rocm-smi >/dev/null 2>&1; then
                rocm-smi --showuse --showmemuse --showtemp 2>/dev/null \
                | grep -E "GPU|%|MiB|Temperature"
            fi
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
trap _cleanup_monitor EXIT INT TERM

# ── Build CLI argument list ──────────────────────────────────────────────────
ARGS=(
    --particles      "$PARTICLES"
    --output-dir     "$EFFECTIVE_OUTPUT_DIR"
    --filename-stem  "$FILENAME_STEM"
    --step-stride    "$STEP_STRIDE"
    --dedup-mode     "$DEDUP_MODE"
    --tolerance-fraction "$TOLERANCE_FRACTION"
    --compression       "$COMPRESSION"
    --compression-opts  "$COMPRESSION_OPTS"
)
[ -n "$MAX_STEPS" ]               && ARGS+=( --max-steps "$MAX_STEPS" )
[ -n "$STEP_RANGE" ]              && ARGS+=( --step-range $STEP_RANGE )
[ -n "$STEP_TAIL" ]               && ARGS+=( --step-tail "$STEP_TAIL" )
[ -n "$TOLERANCE" ]               && ARGS+=( --tolerance $TOLERANCE )
[ "$NO_POINT_DATA"  = "1" ]       && ARGS+=( --no-point-data )
[ -n "$FIELDS" ]                  && ARGS+=( --fields $FIELDS )
[ -n "$REDUCE_MAX_FIELDS" ]       && ARGS+=( --reduce-max-fields $REDUCE_MAX_FIELDS )
if [ -n "$ROI_BOX" ]; then
    ARGS+=( --roi-box $ROI_BOX )
elif [ -n "$ROI_FRACTION" ]; then
    ARGS+=( --roi-fraction $ROI_FRACTION )
fi
[ "$NO_WRITE_CLOUD" = "1" ]       && ARGS+=( --no-write-cloud )
[ "$WRITE_NPY"      = "1" ]       && ARGS+=( --write-npy )
[ "$WRITE_DENSITY"  = "1" ]       && ARGS+=( --write-density )

if [ "$WRITE_DENSITY" = "1" ]; then
    ARGS+=(
        --kernel           "$KERNEL"
        --bandwidth-mode   "$BANDWIDTH_MODE"
        --bandwidth-factor "$BANDWIDTH_FACTOR"
        --resolution       "$RESOLUTION"
        --normalization    "$NORMALIZATION"
    )
    [ -n "$BANDWIDTH" ]       && ARGS+=( --bandwidth      "$BANDWIDTH" )
    [ -n "$BANDWIDTH_XYZ" ]   && ARGS+=( --bandwidth-xyz  $BANDWIDTH_XYZ )
    [ -n "$VOXEL_SIZE" ]      && ARGS+=( --voxel-size     "$VOXEL_SIZE" )
    [ -n "$VOXEL_SIZE_XYZ" ]  && ARGS+=( --voxel-size-xyz $VOXEL_SIZE_XYZ )
    [ -n "$RESOLUTION_XYZ" ]  && ARGS+=( --resolution-xyz $RESOLUTION_XYZ )
    [ "$VOXEL_SIZE_FROM_PARTICLES" = "1" ] && ARGS+=( --voxel-size-from-particles )
fi

# ── Print run summary ────────────────────────────────────────────────────────
echo "======================================================"
echo " JAXTrace — Density Union (LUMI)"
echo "======================================================"
echo " Run ID:        $RUN_ID"
echo " Particles:     $PARTICLES"
echo " Output:        $OUTPUT_DIR"
echo " Dedup mode:    $DEDUP_MODE  (tol_fraction=$TOLERANCE_FRACTION  tol='$TOLERANCE')"
echo " Step stride:   $STEP_STRIDE  (tail='$STEP_TAIL'  range='$STEP_RANGE'  max='$MAX_STEPS')"
if [ -n "$ROI_BOX" ]; then
    echo " ROI (abs):     $ROI_BOX"
elif [ -n "$ROI_FRACTION" ]; then
    echo " ROI (frac):    $ROI_FRACTION"
else
    echo " ROI:           none (full domain)"
fi
echo " Write cloud:   $([ "$NO_WRITE_CLOUD" = "1" ] && echo no || echo yes)"
echo " Write density: $WRITE_DENSITY"
echo " CLI args:      ${ARGS[*]}"
echo " Started:       $(date)"
echo "======================================================"
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
    --env XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION \
    --env MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH \
    --env MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_CUSTOM_CACHE_DIR \
    --env MIOPEN_FIND_MODE=$MIOPEN_FIND_MODE \
    --env TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL \
    --env HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA \
    --env PYTHONUNBUFFERED=$PYTHONUNBUFFERED \
    $SIF \
    python $JAXTRACE/run_density_union.py "${ARGS[@]}" &
SRUN_PID=$!
wait $SRUN_PID
PP_EXIT=$?

_cleanup_monitor

echo ""
echo "Density union exited with code $PP_EXIT at $(date)"

# ── Rsync flash → final destination ──────────────────────────────────────────
# Same pattern as run_lumi.sh: rsync (or cp -a fallback) for a true
# directory merge; loud failures via WARNING.
if [ "$EFFECTIVE_OUTPUT_DIR" != "$OUTPUT_DIR" ]; then
    echo "[stage] moving outputs $EFFECTIVE_OUTPUT_DIR -> $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    _XFER_RC=0
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --remove-source-files "$EFFECTIVE_OUTPUT_DIR"/ "$OUTPUT_DIR"/
        _XFER_RC=$?
    else
        cp -a "$EFFECTIVE_OUTPUT_DIR"/. "$OUTPUT_DIR"/
        _XFER_RC=$?
        [ "$_XFER_RC" = 0 ] && rm -rf "$EFFECTIVE_OUTPUT_DIR"/*
    fi
    if [ "$_XFER_RC" != "0" ]; then
        echo "WARNING: transfer from $EFFECTIVE_OUTPUT_DIR to $OUTPUT_DIR" \
             "failed with rc=$_XFER_RC. Results remain on /flash;" \
             "you can manually rsync them to the destination." >&2
    else
        find "$EFFECTIVE_OUTPUT_DIR" -depth -type d -empty -delete 2>/dev/null
    fi
fi

# Copy any SLURM stdout/stderr into the results folder when we own the allocation.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SLURM_OUT="${SLURM_SUBMIT_DIR:-.}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
    SLURM_ERR="${SLURM_SUBMIT_DIR:-.}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
    mkdir -p "$OUTPUT_DIR/logs"
    [ -f "$SLURM_OUT" ] && cp "$SLURM_OUT" "$OUTPUT_DIR/logs/" 2>/dev/null
    [ -f "$SLURM_ERR" ] && cp "$SLURM_ERR" "$OUTPUT_DIR/logs/" 2>/dev/null
fi
[ -f "$MONITOR_LOG" ] && mv -f "$MONITOR_LOG" "$OUTPUT_DIR/logs/" 2>/dev/null

echo ""
echo "======================================================"
echo " Done. Exit code: $PP_EXIT"
echo " Results: $OUTPUT_DIR"
echo "======================================================"

exit $PP_EXIT
