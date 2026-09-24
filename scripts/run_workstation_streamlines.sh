#!/bin/bash
# =============================================================================
# run_workstation_streamlines.sh
#
# Streamlines-only union runner: emits one polyline per particle for a
# previously-tracked particles.vtkhdf trajectory. Does NOT recompute the
# spatial union or density — those are the job of run_union.sh.
#
# Outputs (under OUTPUT_DIR):
#   <stem>_union_streamlines.vtkhdf   -- PolyData line bundle, one line
#                                         per particle, per-vertex
#                                         VertexTime / Temperature / etc.
#
# Usage:
#   bash run_workstation_streamlines.sh
#   PARTICLES=/path/to/particles.vtkhdf bash run_workstation_streamlines.sh
# =============================================================================

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# ── [1] Paths ────────────────────────────────────────────────────────────────
VENV=/flash/shared/jax/.venv
JAXTRACE=/flash/shared/jax/JAXTrace

# CASE_DIR resolves to the folder this script lives in. Stamped to a
# host-native absolute path by generate_jaxtrace_union_scripts.sh.
CASE_DIR="$(cd "$(dirname "$0")" && pwd -P)"
OUTPUT_CASE_SUBFOLDER=post_pt
if [ -z "${PARTICLES:-}" ]; then
    PARTICLES="$(ls -dt "$CASE_DIR/$OUTPUT_CASE_SUBFOLDER"/run_*/particles.vtkhdf 2>/dev/null | head -1)"
fi

# OUTPUT_DIR: final destination. Defaults to the 'union' folder next to
# the trajectory so streamlines land alongside the spatial union.
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$PARTICLES")/union}"

# FLASH_DIR: optional fast-disk staging directory.
FLASH_DIR="${FLASH_DIR:-/flash/users/${USER}}"

FILENAME_STEM=particles

# ── [2] Streamline parameters ───────────────────────────────────────────────
STREAMLINE_PARTICLE_STRIDE=20     # keep every Nth particle (5% of 360k = 18k lines)
STREAMLINE_STEP_STRIDE=20         # keep every Nth step (200 vertices @ N_STEPS=4000)
STREAMLINE_FIELDS=""              # auto-detect Temperature + MaxTemperature when ""
STREAMLINE_ROI_MODE=none          # seed | none. 'seed' interprets the ROI
                                  # below as fractions/coords of the *trajectory*
                                  # bbox and keeps particles whose step-0 lies
                                  # inside it. For welding setups the seed
                                  # box sits at the inlet (upstream), so the
                                  # trajectory ROI usually does NOT overlap
                                  # the seed → 0 particles. Default 'none'
                                  # keeps every Nth particle regardless.

# ── [3] ROI (applied with STREAMLINE_ROI_MODE=seed) ─────────────────────────
# ROI_FRACTION:  "XLO XHI YLO YHI ZLO ZHI", each in [0, 1] of trajectory bbox.
# ROI_BOX:       absolute "XMIN XMAX YMIN YMAX ZMIN ZMAX" in metres (wins
#                when both set).
ROI_FRACTION="0.5 1.0 0.0 1.0 0.0 1.0"
ROI_BOX=""

# ── [4] Compression / I/O ────────────────────────────────────────────────────
COMPRESSION=gzip                  # gzip | lzf | blosc | none
COMPRESSION_OPTS=1

# ── [5] Performance ─────────────────────────────────────────────────────────
MONITOR_INTERVAL=0                # 0 disables monitor; streamlines are I/O-bound

# =============================================================================
# END USER CONFIGURATION
# =============================================================================

_LOCAL_OVERRIDES="$(dirname "$0")/run_workstation_streamlines.local.sh"
if [ -f "$_LOCAL_OVERRIDES" ]; then
    echo "[config] Sourcing local overrides: $_LOCAL_OVERRIDES"
    # shellcheck source=/dev/null
    source "$_LOCAL_OVERRIDES"
fi

if [ -z "$PARTICLES" ] || [ ! -f "$PARTICLES" ]; then
    echo "ERROR: PARTICLES not found." >&2
    echo "  CASE_DIR=$CASE_DIR" >&2
    echo "  PARTICLES='$PARTICLES'" >&2
    echo "Set PARTICLES via env var:" >&2
    echo "  PARTICLES=/path/to/particles.vtkhdf bash run_streamlines.sh" >&2
    exit 2
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/logs"

if [ -n "$FLASH_DIR" ] && mkdir -p "$FLASH_DIR" 2>/dev/null && [ -w "$FLASH_DIR" ]; then
    _FLASH_RUN_DIR="${FLASH_DIR}/streamlines_${RUN_ID}"
    mkdir -p "$_FLASH_RUN_DIR"
    EFFECTIVE_OUTPUT_DIR="$_FLASH_RUN_DIR"
    echo "[stage] writing streamlines to flash: $EFFECTIVE_OUTPUT_DIR"
    echo "[stage] will move to final dir at end: $OUTPUT_DIR"
else
    EFFECTIVE_OUTPUT_DIR="$OUTPUT_DIR"
    echo "[stage] no flash staging; writing directly to $OUTPUT_DIR"
fi
RUN_LOG="${OUTPUT_DIR}/logs/streamlines_${RUN_ID}.log"

if [ "${__JAXTRACE_STREAMLINES_LOG_ATTACHED:-}" != "1" ]; then
    export __JAXTRACE_STREAMLINES_LOG_ATTACHED=1
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

# JAX is not used by streamlines; skip the CUDA env for a faster, lower-RAM run.
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu  # avoid CUDA init; streamlines do not need a GPU.

# ── Build CLI argument list ──────────────────────────────────────────────────
ARGS=(
    --particles      "$PARTICLES"
    --output-dir     "$EFFECTIVE_OUTPUT_DIR"
    --filename-stem  "$FILENAME_STEM"
    --skip-union
    --write-streamlines
    --streamline-particle-stride "$STREAMLINE_PARTICLE_STRIDE"
    --streamline-step-stride     "$STREAMLINE_STEP_STRIDE"
    --streamline-roi-mode        "$STREAMLINE_ROI_MODE"
    --compression       "$COMPRESSION"
    --compression-opts  "$COMPRESSION_OPTS"
)
[ -n "$STREAMLINE_FIELDS" ] && ARGS+=( --streamline-fields $STREAMLINE_FIELDS )
if [ -n "$ROI_BOX" ]; then
    ARGS+=( --roi-box $ROI_BOX )
elif [ -n "$ROI_FRACTION" ]; then
    ARGS+=( --roi-fraction $ROI_FRACTION )
fi

# ── Print run summary ────────────────────────────────────────────────────────
echo "======================================================"
echo " JAXTrace — Streamlines (Workstation)"
echo "======================================================"
echo " Run ID:        $RUN_ID"
echo " Particles:     $PARTICLES"
echo " Output:        $OUTPUT_DIR"
echo " Particle str:  $STREAMLINE_PARTICLE_STRIDE"
echo " Step stride:   $STREAMLINE_STEP_STRIDE"
echo " Fields:        ${STREAMLINE_FIELDS:-auto}"
echo " ROI mode:      $STREAMLINE_ROI_MODE"
if [ -n "$ROI_BOX" ]; then
    echo " ROI (abs):     $ROI_BOX"
elif [ -n "$ROI_FRACTION" ]; then
    echo " ROI (frac):    $ROI_FRACTION"
else
    echo " ROI:           none (full domain)"
fi
echo " CLI args:      ${ARGS[*]}"
echo " Started:       $(date)"
echo "======================================================"
echo ""

python "${JAXTRACE}/run_density_union.py" "${ARGS[@]}"
SL_EXIT=$?

echo ""
echo "Streamlines exited with code $SL_EXIT at $(date)"

# Merge results from flash → final destination.
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
        echo "WARNING: transfer failed (rc=$_XFER_RC). Results remain on /flash." >&2
    else
        find "$EFFECTIVE_OUTPUT_DIR" -depth -type d -empty -delete 2>/dev/null
    fi
fi

echo ""
echo "======================================================"
echo " Done. Exit code: $SL_EXIT"
echo " Results: $OUTPUT_DIR"
echo "======================================================"

exit $SL_EXIT
