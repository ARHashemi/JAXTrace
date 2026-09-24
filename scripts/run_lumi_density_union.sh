#!/bin/bash
# SLURM batch script: offline union/deduplicate of a particles.vtkhdf
# trajectory into the static material distribution map on LUMI.
# Drives run_density_union.py. Override at submit time, e.g.:
#   sbatch --account=project_XXXXXXXXX \
#          --export=ALL,PARTICLES=/scratch/.../particles.vtkhdf \
#          scripts/run_lumi_density_union.sh
#
#SBATCH --job-name=jt-density-union
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

SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif

JAXTRACE="/project/${PROJECT}/${USER}/JAXTrace"
PKGS="/project/${PROJECT}/${USER}/required-packages"

PARTICLES="${PARTICLES:-/scratch/${PROJECT}/${USER}/outputs/<RUN>/particles.vtkhdf}"

# OUTPUT_DIR: FINAL destination on scratch. Defaults to a 'union' folder
# next to the particles file.
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$PARTICLES")/union}"

# FLASH_DIR: optional fast-disk staging directory (LUMI /flash).
FLASH_DIR="${FLASH_DIR:-/flash/${PROJECT:-project_XXXXXXXXX}/${USER}}"

FILENAME_STEM=density

# ── [2] Step selection ──────────────────────────────────────────────────────
STEP_STRIDE=1
MAX_STEPS=""
STEP_RANGE=""
STEP_TAIL=""

# ── [3] Dedup ───────────────────────────────────────────────────────────────
DEDUP_MODE=batch                  # batch | incremental | none
TOLERANCE=""                      # "" ⇒ TOLERANCE_FRACTION × Δp_axis (step 0)
TOLERANCE_FRACTION=0.5

# ── [3b] Co-moving substraction (shifted-frame / sPOD preprocessing) ────────
# Subtracts the bulk advection V_adv from particles (ξ = x - x_0 - V·(t-t_0))
# and from the time-averaged density (Δρ̄ = ρ̄ - ⟨ρ_ref⟩).
DRIFT_VELOCITY=""                 # "VX VY VZ" in m/s. Empty = no co-moving.
NO_COMOVING=0
NO_EMBED_COMOVING=0
WRITE_COMOVING_CLOUD_SEPARATE=0
WRITE_COMOVING_DENSITY_SEPARATE=0
TIME_ORIGIN=""
T_PASS=""

# Reference-density method: 'uniform' (recommended) or 'drift' (legacy).
COMOVING_REFERENCE=uniform
REFERENCE_BOX=""                  # auto when empty (ROI bbox if set, else density bbox).

# Re-run only — operate on already-produced union files.
FROM_UNION=""
FROM_UNION_DENSITY=""

# JUST_COMOVING auto-fills FROM_UNION and FROM_UNION_DENSITY from the
# standard output names. The trajectory union and density pass are
# skipped; only the co-moving fields are added to the existing files.
JUST_COMOVING=0

# ── [4] PointData propagation ───────────────────────────────────────────────
# By default every per-particle field present in the trajectory file
# (ParticleID, Group, Temperature, MaxTemperature, ElementID, Escaped, …)
# is carried into the unified cloud. Each survivor inherits the FIRST-SEEN
# particle's attributes.
NO_POINT_DATA=0                   # 1 = drop all PointData, keep positions only
FIELDS=""                         # whitelist of field names, space-separated.
                                  # Empty = all fields available in the file.
REDUCE_MAX_FIELDS=""              # scalar fields to MAX-reduce across the
                                  # collapsed group instead of first-seen.

# ── [5] Region of interest ──────────────────────────────────────────────────
# When set, particles falling OUTSIDE the ROI are dropped before dedup, and
# the optional voxel-grid density evaluation is anchored to the ROI bbox.
# ROI_BOX wins when both are set non-empty.
#   ROI_FRACTION:  "XLO XHI YLO YHI ZLO ZHI" in [0, 1] of trajectory bbox.
#   ROI_BOX:       absolute "XMIN XMAX YMIN YMAX ZMIN ZMAX" in metres.
# Both empty = no ROI.
ROI_FRACTION=""
ROI_BOX=""

# ── [6] Output toggles ──────────────────────────────────────────────────────
NO_WRITE_CLOUD=0                  # 1 = skip <stem>_union.vtkhdf
WRITE_NPY=0                       # 1 = also write <stem>_union.npy
WRITE_DENSITY=0                   # 1 = also build voxel-grid density

# Streamline-union: one polyline per particle (Lagrangian trajectory bundle).
WRITE_STREAMLINES=0
STREAMLINE_PARTICLE_STRIDE=20
STREAMLINE_STEP_STRIDE=20
STREAMLINE_FIELDS=""
STREAMLINE_ROI_MODE=seed

# ── [5] Density pass (only if WRITE_DENSITY=1) ──────────────────────────────
KERNEL=wendland_c2                # wendland_c2|wendland_c4|cubic_spline|gaussian|epanechnikov|quintic_spline
BANDWIDTH_MODE=initial_spacing    # fixed | scott | silverman | knn_adaptive | initial_spacing
BANDWIDTH=""                      # scalar h [m]
BANDWIDTH_XYZ=""                  # "HX HY HZ"
BANDWIDTH_FACTOR=1.5
RESOLUTION=128
RESOLUTION_XYZ=""
VOXEL_SIZE=""
VOXEL_SIZE_XYZ=""
VOXEL_SIZE_FROM_PARTICLES=0
NORMALIZATION=pdf

# ── [6] Compression ─────────────────────────────────────────────────────────
COMPRESSION=gzip                  # gzip | lzf | blosc | none
COMPRESSION_OPTS=1

# ── [7] Performance (only relevant if WRITE_DENSITY=1) ──────────────────────
XLA_PREALLOC=0
MONITOR_INTERVAL=30
BENCHMARK_MODE=0

# =============================================================================
# END USER CONFIGURATION
# =============================================================================

_LOCAL_OVERRIDES="$(dirname "$0")/run_lumi_density_union.local.sh"
if [ -f "$_LOCAL_OVERRIDES" ]; then
    echo "[config] Sourcing local overrides: $_LOCAL_OVERRIDES"
    # shellcheck source=/dev/null
    source "$_LOCAL_OVERRIDES"
fi

# JUST_COMOVING auto-fills the re-run inputs from the standard output names.
if [ "${JUST_COMOVING:-0}" = "1" ]; then
    [ -z "$FROM_UNION" ] && \
        FROM_UNION="${OUTPUT_DIR}/${FILENAME_STEM}_union.vtkhdf"
    [ -z "$FROM_UNION_DENSITY" ] && \
        FROM_UNION_DENSITY="${OUTPUT_DIR}/${FILENAME_STEM}_union_density.vtkhdf"
    echo "[config] JUST_COMOVING=1: re-run inputs auto-filled"
    echo "  FROM_UNION         = $FROM_UNION"
    echo "  FROM_UNION_DENSITY = $FROM_UNION_DENSITY"
    if [ ! -f "$FROM_UNION" ]; then
        echo "ERROR: JUST_COMOVING expected union cloud at $FROM_UNION" >&2
        exit 2
    fi
    if [ ! -f "$FROM_UNION_DENSITY" ]; then
        echo "ERROR: JUST_COMOVING expected union density at $FROM_UNION_DENSITY" >&2
        exit 2
    fi
fi

if [ ! -f "$PARTICLES" ]; then
    echo "ERROR: PARTICLES not found: $PARTICLES" >&2
    echo "Set PARTICLES in this script or via --export=ALL,PARTICLES=..." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

if [ -n "$FLASH_DIR" ] && mkdir -p "$FLASH_DIR" 2>/dev/null && [ -w "$FLASH_DIR" ]; then
    _FLASH_RUN_DIR="${FLASH_DIR}/union_${SLURM_JOB_ID}"
    mkdir -p "$_FLASH_RUN_DIR"
    EFFECTIVE_OUTPUT_DIR="$_FLASH_RUN_DIR"
    echo "[stage] writing union to flash: $EFFECTIVE_OUTPUT_DIR"
    echo "[stage] will move to final dir at end: $OUTPUT_DIR"
else
    EFFECTIVE_OUTPUT_DIR="$OUTPUT_DIR"
    echo "[stage] no flash staging (FLASH_DIR='$FLASH_DIR'); writing directly to $OUTPUT_DIR"
fi

MONITOR_LOG="${OUTPUT_DIR}/density_union_${SLURM_JOB_ID}_monitor.log"

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

# ── GPU & Memory Monitor (only while density pass runs) ──────────────────────
MONITOR_PID=""
MONITOR_PGID=""
if [ "$WRITE_DENSITY" = "1" ] && [ "$BENCHMARK_MODE" != "1" ] && [ "$MONITOR_INTERVAL" -gt 0 ] 2>/dev/null; then
  setsid bash -c '
    echo "=== Density Union Monitor === Job '"$SLURM_JOB_ID"' === $(date) ==="
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
[ "$WRITE_STREAMLINES" = "1" ]    && ARGS+=( --write-streamlines )
[ -n "$STREAMLINE_PARTICLE_STRIDE" ] && ARGS+=( --streamline-particle-stride "$STREAMLINE_PARTICLE_STRIDE" )
[ -n "$STREAMLINE_STEP_STRIDE" ]     && ARGS+=( --streamline-step-stride "$STREAMLINE_STEP_STRIDE" )
[ -n "$STREAMLINE_FIELDS" ]          && ARGS+=( --streamline-fields $STREAMLINE_FIELDS )
[ -n "$STREAMLINE_ROI_MODE" ]        && ARGS+=( --streamline-roi-mode "$STREAMLINE_ROI_MODE" )

# ── Co-moving substraction ───────────────────────────────────────────────
[ -n "$DRIFT_VELOCITY" ]                && ARGS+=( --drift-velocity $DRIFT_VELOCITY )
[ "$NO_COMOVING"            = "1" ]     && ARGS+=( --no-comoving )
[ "$NO_EMBED_COMOVING"      = "1" ]     && ARGS+=( --no-embed-comoving )
[ "$WRITE_COMOVING_CLOUD_SEPARATE"   = "1" ] && ARGS+=( --write-comoving-cloud-separate )
[ "$WRITE_COMOVING_DENSITY_SEPARATE" = "1" ] && ARGS+=( --write-comoving-density-separate )
[ -n "$TIME_ORIGIN" ]                   && ARGS+=( --time-origin "$TIME_ORIGIN" )
[ -n "$T_PASS" ]                        && ARGS+=( --t-pass "$T_PASS" )
[ -n "$COMOVING_REFERENCE" ]            && ARGS+=( --comoving-reference "$COMOVING_REFERENCE" )
[ -n "$REFERENCE_BOX" ]                 && ARGS+=( --reference-box $REFERENCE_BOX )
[ -n "$FROM_UNION" ]                    && ARGS+=( --from-union "$FROM_UNION" )
[ -n "$FROM_UNION_DENSITY" ]            && ARGS+=( --from-union-density "$FROM_UNION_DENSITY" )

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

ALLOC_ENV=()
if [ -n "${XLA_PYTHON_CLIENT_ALLOCATOR:-}" ]; then
    ALLOC_ENV=( --env "XLA_PYTHON_CLIENT_ALLOCATOR=$XLA_PYTHON_CLIENT_ALLOCATOR" )
fi

echo "Starting density union at $(date)"
echo "  Particles:   $PARTICLES"
echo "  Output:      $OUTPUT_DIR"
echo "  Dedup mode:  $DEDUP_MODE  (tol_fraction=$TOLERANCE_FRACTION  tol='$TOLERANCE')"
echo "  CLI args:    ${ARGS[*]}"
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
    python $JAXTRACE/run_density_union.py "${ARGS[@]}" &
SRUN_PID=$!
wait $SRUN_PID
PP_EXIT=$?

_cleanup_monitor

echo ""
echo "Density union exited with code $PP_EXIT at $(date)"

# Merge results from flash → final destination. Same pattern as
# run_jaxtrace.sh — rsync with --remove-source-files (or cp -a fallback)
# so an existing SCRATCH_OUT/<subdir> is merged into rather than rejected.
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

[ -f "$MONITOR_LOG" ] && mv "$MONITOR_LOG" "${OUTPUT_DIR}/logs/" 2>/dev/null || true

exit $PP_EXIT
