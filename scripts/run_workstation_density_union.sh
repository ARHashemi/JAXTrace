#!/bin/bash
# =============================================================================
# run_workstation_density_union.sh
#
# Build the union/deduplicated material distribution from a particles.vtkhdf
# trajectory on a single-node NVIDIA workstation. Drives run_density_union.py.
#
# Outputs (under OUTPUT_DIR):
#   <stem>_union.vtkhdf            -- deduplicated PolyData point cloud
#   <stem>_union.npy               -- optional NumPy mirror (WRITE_NPY=1)
#   <stem>_union_density.vtkhdf    -- optional voxel-grid density on the
#                                     unified cloud (WRITE_DENSITY=1)
#
# Usage:
#   bash run_workstation_density_union.sh
#   PARTICLES=/path/to/particles.vtkhdf bash run_workstation_density_union.sh
# =============================================================================

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# ── [1] Paths ────────────────────────────────────────────────────────────────
VENV=/flash/shared/jax/.venv
JAXTRACE=/flash/shared/jax/JAXTrace

PARTICLES="${PARTICLES:-/scratch/users/${USER}/outputs/<RUN>/particles.vtkhdf}"

# OUTPUT_DIR: FINAL destination on scratch (or case folder). Defaults to a
# 'union' folder next to the particles file.
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$PARTICLES")/union}"

# FLASH_DIR: optional fast-disk staging directory. When set, Python writes
# the union outputs into a per-run subfolder under here, and the shell moves
# them to OUTPUT_DIR at the end of the run. Empty disables staging.
FLASH_DIR="${FLASH_DIR:-/flash/users/${USER}}"

FILENAME_STEM=density

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
# from the step-0 seeding (anisotropic), so two particles within a half-cell
# of each other collapse to one. Override e.g. with TOLERANCE="1e-3 4e-4 1e-4".
TOLERANCE=""
TOLERANCE_FRACTION=0.5

# ── [3b] Co-moving substraction (shifted-frame / sPOD preprocessing) ────────
# Subtracts the bulk advection V_adv from particles (ξ = x - x_0 - V·(t-t_0))
# and from the time-averaged density (ρ_ref(x,t) = ρ_0(x - V·(t-t_0)),
# Δρ̄ = ρ̄ - ⟨ρ_ref⟩) so the residual reveals stirring near the tool.
#
# Default: ENABLED with V_adv = INLET_VELOCITY · ê_x when DRIFT_VELOCITY is
# set; turn off with NO_COMOVING=1. Set DRIFT_VELOCITY="" to skip entirely.
DRIFT_VELOCITY=""                 # "VX VY VZ" in m/s. Empty = no co-moving.
                                  # Match INLET_VELOCITY from the tracking run
                                  # for the standard sPOD setup.
NO_COMOVING=0                     # 1 = master off-switch (overrides DRIFT_VELOCITY).
NO_EMBED_COMOVING=0               # 1 = do NOT add co-moving fields to the main
                                  # union/density files.
WRITE_COMOVING_CLOUD_SEPARATE=0   # 1 = also emit <stem>_union_comoving.vtkhdf.
WRITE_COMOVING_DENSITY_SEPARATE=0 # 1 = also emit <stem>_union_density_comoving.vtkhdf.
TIME_ORIGIN=""                    # override t_0; default = trajectory step-0 time.
T_PASS=""                         # override T_pass; default = t_end - t_start.

# Reference-density method.
#   uniform (default) -- ρ_unif from a synthetic uniform Cartesian grid at
#                        the seeding spacing Δp filling REFERENCE_BOX (defaults
#                        to ROI bbox if set, otherwise density grid bbox).
#                        Residual ρ̄ - ρ_unif is positive where material drifted
#                        into a previously-empty region, negative where it
#                        depleted, ~zero in pure-advection regions inside
#                        the seeding. Also writes 'normalized_density' = ρ̄/ρ_unif.
#   drift             -- legacy time-integrated shift of step-0 density along V.
COMOVING_REFERENCE=uniform
REFERENCE_BOX=""                  # "XMIN XMAX YMIN YMAX ZMIN ZMAX" in metres.
                                  # Empty = auto (ROI or density bbox).

# Re-run only — operate on already-produced union files. When set, the
# trajectory union/density pass is skipped and only the co-moving fields
# are added to existing files. PARTICLES is still required (for step-0
# positions and ParticleID lookup).
FROM_UNION=""                     # e.g. "$OUTPUT_DIR/${FILENAME_STEM}_union.vtkhdf"
FROM_UNION_DENSITY=""             # e.g. "$OUTPUT_DIR/${FILENAME_STEM}_union_density.vtkhdf"

# JUST_COMOVING: convenience for the common "I already ran the union, now
# just add the substraction" path. When 1, auto-fills FROM_UNION and
# FROM_UNION_DENSITY from the standard output names in $OUTPUT_DIR (only
# the ones that don't already have an explicit value above). The
# trajectory union and density pass are then skipped.
JUST_COMOVING=0

# ── [4] PointData propagation ───────────────────────────────────────────────
# By default every per-particle field present in the trajectory file
# (ParticleID, Group, Temperature, MaxTemperature, ElementID, Escaped, …)
# is carried into the unified cloud. Each survivor inherits the FIRST-SEEN
# particle's attributes.
NO_POINT_DATA=0                   # 1 = drop all PointData, keep positions only
FIELDS=""                         # whitelist of field names, space-separated.
                                  # Empty = all fields available in the file.
                                  # Example: FIELDS="ParticleID Group MaxTemperature"
REDUCE_MAX_FIELDS=""              # scalar fields to MAX-reduce across the
                                  # collapsed group instead of first-seen.
                                  # Typical: "MaxTemperature Temperature".
                                  # Useful when you want "hottest the cell ever was".

# ── [5] Region of interest ──────────────────────────────────────────────────
# When set, particles falling OUTSIDE the ROI are dropped before dedup, and
# the optional voxel-grid density evaluation is anchored to the ROI bbox.
# Default is fractional (matches SEED_FRACTION convention from run_jaxtrace.sh).
# ROI_FRACTION wins when both are set non-empty.
#   ROI_FRACTION:  "XLO XHI YLO YHI ZLO ZHI", each fraction in [0, 1]
#                  of the trajectory union bbox.
#                  Example: "0.0 0.3 0.0 1.0 0.0 1.0" = first 30% of X, full Y/Z.
#   ROI_BOX:       absolute "XMIN XMAX YMIN YMAX ZMIN ZMAX" in metres.
# Both empty = no ROI (whole trajectory bbox).
ROI_FRACTION=""
ROI_BOX=""

# ── [6] Output toggles ──────────────────────────────────────────────────────
NO_WRITE_CLOUD=0                  # 1 = skip <stem>_union.vtkhdf
WRITE_NPY=0                       # 1 = also write <stem>_union.npy
WRITE_DENSITY=0                   # 1 = also build voxel-grid density

# Streamline-union: one polyline per particle (Lagrangian trajectory bundle).
# Independent of the dedup union; reads particles.vtkhdf directly so it works
# even with JUST_COMOVING=1. File size ≈ N_lines × N_vert × (12 + 4·n_fields)
# bytes; the defaults below give ~50 MB on a 360k × 4000-step trajectory.
WRITE_STREAMLINES=0               # 1 = also emit <stem>_union_streamlines.vtkhdf
STREAMLINE_PARTICLE_STRIDE=20     # keep every Nth particle by ParticleID
STREAMLINE_STEP_STRIDE=20         # keep every Nth step (vertices per line)
STREAMLINE_FIELDS=""              # per-vertex PointData fields (space-separated).
                                  # Empty = auto: Temperature + MaxTemperature
                                  # if they exist in the trajectory. VertexTime
                                  # is always added.
STREAMLINE_ROI_MODE=seed          # seed | none. seed = keep particles whose
                                  # step-0 position is in the ROI box.

# ── [5] Density pass (only if WRITE_DENSITY=1) ──────────────────────────────
KERNEL=wendland_c2                # wendland_c2|wendland_c4|cubic_spline|gaussian|epanechnikov|quintic_spline
BANDWIDTH_MODE=initial_spacing    # fixed | scott | silverman | knn_adaptive | initial_spacing
BANDWIDTH=""                      # scalar h [m] (fixed mode)
BANDWIDTH_XYZ=""                  # "HX HY HZ" per-axis (fixed mode)
BANDWIDTH_FACTOR=1.5
RESOLUTION=128
RESOLUTION_XYZ=""                 # "NX NY NZ"
VOXEL_SIZE=""                     # scalar voxel edge [m]
VOXEL_SIZE_XYZ=""                 # "HX HY HZ"
VOXEL_SIZE_FROM_PARTICLES=0       # 1 = derive voxel size from Δp_axis (step 0)
NORMALIZATION=pdf                 # pdf | mass | unnormalized

# ── [6] Compression ─────────────────────────────────────────────────────────
COMPRESSION=gzip                  # gzip | lzf | blosc | none
COMPRESSION_OPTS=1                # gzip level (1-9) or blosc clevel

# ── [7] JAX memory & performance (only relevant if WRITE_DENSITY=1) ─────────
XLA_PREALLOC=1
VRAM_FRACTION=0.9
MONITOR_INTERVAL=30
BENCHMARK_MODE=0

# =============================================================================
# END USER CONFIGURATION
# =============================================================================

_LOCAL_OVERRIDES="$(dirname "$0")/run_workstation_density_union.local.sh"
if [ -f "$_LOCAL_OVERRIDES" ]; then
    echo "[config] Sourcing local overrides: $_LOCAL_OVERRIDES"
    # shellcheck source=/dev/null
    source "$_LOCAL_OVERRIDES"
fi

# JUST_COMOVING auto-fills the re-run inputs from the standard output
# names so the user does not have to type two long paths. Only fills
# the slots that the user did not set explicitly above.
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
        echo "Run the full union first, or set FROM_UNION explicitly." >&2
        exit 2
    fi
    if [ ! -f "$FROM_UNION_DENSITY" ]; then
        echo "ERROR: JUST_COMOVING expected union density at $FROM_UNION_DENSITY" >&2
        exit 2
    fi
fi

if [ ! -f "$PARTICLES" ]; then
    echo "ERROR: PARTICLES not found: $PARTICLES" >&2
    echo "Set PARTICLES in this script or as an env var on the command line." >&2
    exit 2
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/logs"

if [ -n "$FLASH_DIR" ] && mkdir -p "$FLASH_DIR" 2>/dev/null && [ -w "$FLASH_DIR" ]; then
    _FLASH_RUN_DIR="${FLASH_DIR}/union_${RUN_ID}"
    mkdir -p "$_FLASH_RUN_DIR"
    EFFECTIVE_OUTPUT_DIR="$_FLASH_RUN_DIR"
    echo "[stage] writing union to flash: $EFFECTIVE_OUTPUT_DIR"
    echo "[stage] will move to final dir at end: $OUTPUT_DIR"
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

# ── Activate venv ────────────────────────────────────────────────────────────
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
    echo "[env] Activated venv: $VENV ($(python --version))"
else
    echo "[warn] venv not found at $VENV — using current Python: $(which python)"
fi

# ── CUDA / JAX env (only relevant for the optional density pass) ─────────────
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS="--xla_gpu_autotune_level=4 --xla_gpu_enable_latency_hiding_scheduler=true"
CUDA_CACHE_DIR="/tmp/${USER}-xla-cache-${RUN_ID}"
export CUDA_CACHE_PATH="$CUDA_CACHE_DIR"
mkdir -p "$CUDA_CACHE_DIR"
[ "$BENCHMARK_MODE" = "1" ] && XLA_PREALLOC=1
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
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

# ── GPU & Memory Monitor (only while density pass runs) ──────────────────────
MONITOR_PID=""
MONITOR_PGID=""
if [ "$WRITE_DENSITY" = "1" ] && [ "$BENCHMARK_MODE" != "1" ] && [ "${MONITOR_INTERVAL}" -gt 0 ] 2>/dev/null; then
    setsid bash -c '
        echo "=== Density Union Monitor === Run '"${RUN_ID}"' === $(date) ==="
        while true; do
            echo "--- $(date '\''+%Y-%m-%d %H:%M:%S'\'') ---"
            nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
                       --format=csv,noheader,nounits \
            | awk -F", " '\''{printf "  GPU  Temp:%s°C  Util:%s%%  VRAM:%s/%s MiB  Power:%sW\n",$1,$2,$3,$4,$5}'\''
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
# ROI_BOX wins when both are set; CLI itself also enforces this precedence.
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

# ── Co-moving substraction (drives the sPOD-style residuals) ────────────────
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

# ── Print run summary ────────────────────────────────────────────────────────
echo "======================================================"
echo " JAXTrace — Density Union (Workstation)"
echo "======================================================"
echo " Run ID:        $RUN_ID"
echo " Particles:     $PARTICLES"
echo " Output:        $OUTPUT_DIR"
echo " Dedup mode:    $DEDUP_MODE  (tol_fraction=$TOLERANCE_FRACTION  tol='$TOLERANCE')"
echo " Step stride:   $STEP_STRIDE  (tail='$STEP_TAIL'  range='$STEP_RANGE'  max='$MAX_STEPS')"
echo " PointData:     $([ "$NO_POINT_DATA" = "1" ] && echo dropped || echo "kept (fields='${FIELDS:-all}')")"
[ -n "$REDUCE_MAX_FIELDS" ] && echo " Reduce-max:    $REDUCE_MAX_FIELDS"
if [ -n "$ROI_BOX" ]; then
    echo " ROI (abs):     $ROI_BOX"
elif [ -n "$ROI_FRACTION" ]; then
    echo " ROI (frac):    $ROI_FRACTION"
else
    echo " ROI:           none (full domain)"
fi
echo " Write cloud:   $([ "$NO_WRITE_CLOUD" = "1" ] && echo no || echo yes)"
echo " Write npy:     $WRITE_NPY"
echo " Write density: $WRITE_DENSITY"
if [ -n "$DRIFT_VELOCITY" ] && [ "$NO_COMOVING" != "1" ]; then
    echo " Co-moving:     ON  V_adv=($DRIFT_VELOCITY)  embed=$([ "$NO_EMBED_COMOVING" = "1" ] && echo no || echo yes)"
    echo "                separate cloud=$WRITE_COMOVING_CLOUD_SEPARATE  separate density=$WRITE_COMOVING_DENSITY_SEPARATE"
else
    echo " Co-moving:     OFF"
fi
[ -n "$FROM_UNION" ]         && echo " Re-run cloud:  $FROM_UNION"
[ -n "$FROM_UNION_DENSITY" ] && echo " Re-run dens.:  $FROM_UNION_DENSITY"
if [ "$WRITE_DENSITY" = "1" ]; then
    echo " Kernel:        $KERNEL  (bw-mode=$BANDWIDTH_MODE)"
    echo " GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
fi
echo " CLI args:      ${ARGS[*]}"
echo " Started:       $(date)"
echo "======================================================"
echo ""

python "${JAXTRACE}/run_density_union.py" "${ARGS[@]}"
PP_EXIT=$?

_cleanup_monitor

echo ""
echo "Density union exited with code $PP_EXIT at $(date)"

# Merge results from flash → final destination. Use the same pattern as
# run_jaxtrace.sh: `rsync -a --remove-source-files` (or `cp -a` fallback)
# so existing same-name subfolders are merged into rather than refused.
# `mv FLASH/*` would silently leave files stranded when SCRATCH_OUT/<sub>
# already exists.
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

rm -rf "$CUDA_CACHE_DIR"

echo ""
echo "======================================================"
echo " Done. Exit code: $PP_EXIT"
echo " Results: $OUTPUT_DIR"
echo "======================================================"

exit $PP_EXIT
