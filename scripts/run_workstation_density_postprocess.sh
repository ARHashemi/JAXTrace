#!/bin/bash
# =============================================================================
# run_workstation_density_postprocess.sh
#
# Offline density-field post-processing for a single-node NVIDIA workstation.
# Drives run_density_postprocess.py against an already-written
# particles.vtkhdf file.
#
# Usage:
#   bash run_workstation_density_postprocess.sh
#   PARTICLES=/path/to/particles.vtkhdf bash run_workstation_density_postprocess.sh
# =============================================================================

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# ── [1] Paths ────────────────────────────────────────────────────────────────
VENV=/flash/shared/jax/.venv
JAXTRACE=/flash/shared/jax/JAXTrace
# PKGS=/flash/shared/jax/required-packages

# REQUIRED: path to the particles.vtkhdf to process. Override on the command
# line, e.g. PARTICLES=/path/to/particles.vtkhdf bash <this-script>
PARTICLES="${PARTICLES:-/scratch/users/${USER}/outputs/<RUN>/particles.vtkhdf}"

# OUTPUT_DIR: FINAL destination for the density outputs (typically on the
# scratch / case folder). Defaults to a 'density' folder next to the
# particles file.
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$PARTICLES")/density}"

# FLASH_DIR: optional fast-disk staging directory. When set, Python writes
# the density files into a per-run subfolder under here, and the shell
# moves them to OUTPUT_DIR at the end of the run. Empty disables staging
# (Python writes directly to OUTPUT_DIR).
#   recommended: /flash/users/$USER
# Leave empty to skip staging.
FLASH_DIR="${FLASH_DIR:-/flash/users/${USER}}"

# Optional: a velocity mesh PVTU/PVD used ONLY for inside-mesh masking.
VELOCITY_MESH=""                  # e.g. /scratch/.../<case>.gid/post/0eule/<case>_159.pvtu

# ── [2] Voxel grid ───────────────────────────────────────────────────────────
# BOUNDS_MODE picks how the grid bounding box is determined:
#   prepass  -- 2-pass read of PARTICLES to take the union over all timesteps.
#   mesh     -- use the velocity-mesh bbox (requires VELOCITY_MESH set).
#   explicit -- use BOUNDS below verbatim.
BOUNDS_MODE=prepass               # prepass | mesh | explicit
BOUNDS=""                         # "XMIN XMAX YMIN YMAX ZMIN ZMAX" (for explicit)
RESOLUTION=128                    # cubic grid resolution (used if no override)
RESOLUTION_XYZ=""                 # "NX NY NZ" per-axis; overrides RESOLUTION
VOXEL_SIZE=""                     # scalar voxel edge length [m]; overrides RESOLUTION
VOXEL_SIZE_XYZ=""                 # "HX HY HZ" per-axis; overrides VOXEL_SIZE / RESOLUTION
VOXEL_SIZE_FROM_PARTICLES=0       # 1 = size voxels from step-0 inter-particle Δp_axis
                                  # (overrides RESOLUTION* and VOXEL_SIZE*)
PAD_FRACTION=0.0
# Inside-mesh masking: when 1, voxels inside any element of the velocity mesh
# pass through; outside voxels are zeroed. Caveat: the mask is built once
# from VELOCITY_MESH (a single PVTU snapshot). If the velocity mesh is
# time-dependent (e.g. a tool-following moving window), set this to 1 to
# disable masking; otherwise the static mask wipes out most of the domain
# the particles actually traverse over the full trajectory.
NO_MASK_INSIDE_MESH=0

# ── [3] Kernel / bandwidth ───────────────────────────────────────────────────
KERNEL=wendland_c2                # wendland_c2|wendland_c4|cubic_spline|gaussian|epanechnikov|quintic_spline
BANDWIDTH_MODE=fixed              # fixed | scott | silverman | knn_adaptive | initial_spacing
BANDWIDTH=""                      # scalar h [m] (fixed mode)
BANDWIDTH_XYZ=""                  # "HX HY HZ" per-axis (fixed mode)
# h = BANDWIDTH_FACTOR * voxel_spacing_per_axis (or initial particle spacing
# when BANDWIDTH_MODE=initial_spacing). Increase on anisotropic grids so the
# kernel spans multiple voxels in every direction.
BANDWIDTH_FACTOR=2.0
BANDWIDTH_REFRESH_EVERY=0
KNN_K=32
KNN_SAFETY=1.2

# ── [4] Engine ───────────────────────────────────────────────────────────────
ENGINE=auto                       # auto | brute | octree
AUTO_THRESHOLD=5e10
BRUTE_QUERY_CHUNK=8192
OCTREE_TARGET_N_PER_CELL=9        # particle-hash backend: target ~particles/cell
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
#   gzip  -- universally readable (DEFAULT). ParaView's bundled vtkhdf5
#            does NOT link LZF or blosc, so files written with those
#            compressors fail to open in ParaView with
#            "H5Dread ... Error reading array".
#   lzf   -- ~4x faster writer than gzip-1, but only readable from h5py
#            or a custom HDF5 install with the LZF filter registered.
#   blosc -- multi-thread compression via hdf5plugin; same reader caveat.
#   none  -- no compression, fastest writer, ~3x bigger files.
COMPRESSION=gzip
COMPRESSION_OPTS=1                # gzip level (1-9) or blosc clevel
BLOSC_THREADS=4

# Background trajectory reader queue depth. 0 disables prefetch (synchronous
# read on the main thread). 4-8 typically closes the GPU-idle gap.
READ_PREFETCH=4

# ── [6] JAX memory & performance ─────────────────────────────────────────────
XLA_PREALLOC=1
VRAM_FRACTION=0.9
BENCHMARK_MODE=0
MONITOR_INTERVAL=30

# =============================================================================
# END USER CONFIGURATION
# =============================================================================

_LOCAL_OVERRIDES="$(dirname "$0")/run_workstation_density_postprocess.local.sh"
if [ -f "$_LOCAL_OVERRIDES" ]; then
    echo "[config] Sourcing local overrides: $_LOCAL_OVERRIDES"
    # shellcheck source=/dev/null
    source "$_LOCAL_OVERRIDES"
fi

if [ ! -f "$PARTICLES" ]; then
    echo "ERROR: PARTICLES not found: $PARTICLES" >&2
    echo "Set PARTICLES in this script or as an env var on the command line." >&2
    exit 2
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/logs"

# Flash-staging logic. When FLASH_DIR is set AND writable AND not the same
# physical location as OUTPUT_DIR, Python writes to a per-run subfolder
# under FLASH_DIR. We mv to OUTPUT_DIR at the end of the run, regardless
# of whether Python exited normally or via SIGTERM. Empty FLASH_DIR or an
# unwritable one skips staging and Python writes directly to OUTPUT_DIR.
if [ -n "$FLASH_DIR" ] && mkdir -p "$FLASH_DIR" 2>/dev/null && [ -w "$FLASH_DIR" ]; then
    _FLASH_RUN_DIR="${FLASH_DIR}/density_${RUN_ID}"
    mkdir -p "$_FLASH_RUN_DIR"
    EFFECTIVE_OUTPUT_DIR="$_FLASH_RUN_DIR"
    echo "[stage] writing density to flash: $EFFECTIVE_OUTPUT_DIR"
    echo "[stage] will move to final dir at end: $OUTPUT_DIR"
else
    EFFECTIVE_OUTPUT_DIR="$OUTPUT_DIR"
    echo "[stage] no flash staging (FLASH_DIR='$FLASH_DIR'); writing directly to $OUTPUT_DIR"
fi
MONITOR_LOG="${OUTPUT_DIR}/logs/density_pp_${RUN_ID}_monitor.log"
RUN_LOG="${OUTPUT_DIR}/logs/density_pp_${RUN_ID}.log"

# Mirror stdout/stderr.
if [ "${__JAXTRACE_DENSITY_LOG_ATTACHED:-}" != "1" ]; then
    export __JAXTRACE_DENSITY_LOG_ATTACHED=1
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

# ── CUDA / JAX env ───────────────────────────────────────────────────────────
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

# ── GPU & Memory Monitor ─────────────────────────────────────────────────────
MONITOR_PID=""
MONITOR_PGID=""
if [ "$BENCHMARK_MODE" != "1" ] && [ "${MONITOR_INTERVAL}" -gt 0 ] 2>/dev/null; then
    setsid bash -c '
        echo "=== Density PP Monitor === Run '"${RUN_ID}"' === $(date) ==="
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
    --octree-target-n-per-cell "$OCTREE_TARGET_N_PER_CELL"
    --particle-bucket "$PARTICLE_BUCKET"
    --normalization  "$NORMALIZATION"
    --step-stride    "$STEP_STRIDE"
    --compression       "$COMPRESSION"
    --compression-opts  "$COMPRESSION_OPTS"
    --blosc-threads     "$BLOSC_THREADS"
    --read-prefetch     "$READ_PREFETCH"
)
[ -n "$BANDWIDTH" ]       && ARGS+=( --bandwidth      "$BANDWIDTH" )
[ -n "$BANDWIDTH_XYZ" ]   && ARGS+=( --bandwidth-xyz  $BANDWIDTH_XYZ )
[ -n "$VOXEL_SIZE" ]      && ARGS+=( --voxel-size     "$VOXEL_SIZE" )
[ -n "$VOXEL_SIZE_XYZ" ]  && ARGS+=( --voxel-size-xyz $VOXEL_SIZE_XYZ )
[ -n "$RESOLUTION_XYZ" ]  && ARGS+=( --resolution-xyz $RESOLUTION_XYZ )
[ "$VOXEL_SIZE_FROM_PARTICLES" = "1" ] && ARGS+=( --voxel-size-from-particles )
[ -n "$VELOCITY_MESH" ]   && ARGS+=( --velocity-mesh "$VELOCITY_MESH" )
[ -n "$BOUNDS" ]          && ARGS+=( --bounds $BOUNDS )
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

# ── Print run summary ────────────────────────────────────────────────────────
VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
VRAM_CAP_MB=$(echo "$VRAM_TOTAL $VRAM_FRACTION" | awk '{printf "%.0f", $1*$2}')

echo "======================================================"
echo " JAXTrace — Density Post-Processing (Workstation)"
echo "======================================================"
echo " Run ID:       $RUN_ID"
echo " Particles:    $PARTICLES"
echo " Output:       $OUTPUT_DIR"
echo " Bounds mode:  $BOUNDS_MODE"
echo " Resolution:   $RESOLUTION  (voxel_size='$VOXEL_SIZE')"
echo " Kernel:       $KERNEL  (bw-mode=$BANDWIDTH_MODE)"
echo " Engine:       $ENGINE"
echo " GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo " VRAM cap:     ${VRAM_FRACTION} (≈ ${VRAM_CAP_MB} / ${VRAM_TOTAL} MiB)"
echo " CLI args:     ${ARGS[*]}"
echo " Started:      $(date)"
echo "======================================================"
echo ""

python "${JAXTRACE}/run_density_postprocess.py" "${ARGS[@]}"
PP_EXIT=$?

_cleanup_monitor

echo ""
echo "Density post-processing exited with code $PP_EXIT at $(date)"

# Move from flash to final OUTPUT_DIR. Done regardless of exit code so a
# partially-completed run still leaves recoverable output in scratch.
if [ "$EFFECTIVE_OUTPUT_DIR" != "$OUTPUT_DIR" ]; then
    echo "[stage] moving outputs $EFFECTIVE_OUTPUT_DIR -> $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    # Use mv -f so existing same-name files in OUTPUT_DIR get overwritten
    # cleanly. Handles the case where a previous run wrote intermediate
    # artefacts (e.g. a partial density.vtkhdf) that this run is replacing.
    mv -f "$EFFECTIVE_OUTPUT_DIR"/* "$OUTPUT_DIR"/ 2>/dev/null
    rmdir "$EFFECTIVE_OUTPUT_DIR" 2>/dev/null || true
fi

rm -rf "$CUDA_CACHE_DIR"

echo ""
echo "======================================================"
echo " Done. Exit code: $PP_EXIT"
echo " Results: $OUTPUT_DIR"
echo "======================================================"

exit $PP_EXIT
