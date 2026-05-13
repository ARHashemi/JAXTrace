#!/bin/bash
# =============================================================================
# run_workstation.sh — JAXTrace launcher for a single-node NVIDIA workstation.
#
# Targets a CUDA GPU and a Python venv (no SLURM, no Singularity).
# Paths default to a /flash (NVMe) + /scratch (HDD) layout; override the
# top-level path variables in scripts/run_workstation.local.sh to match
# your host.
#
# Usage:
#   Foreground:            bash run_workstation.sh
#   Background (nohup):    nohup bash run_workstation.sh > run.log 2>&1 &
#   Task-spooler queue:    TS_SOCKET=/tmp/gpu_queue ts bash run_workstation.sh
#   Inside screen/tmux:    screen -S jaxtrace; bash run_workstation.sh
# =============================================================================

# =============================================================================
# USER CONFIGURATION — edit these groups to control the production run.
# =============================================================================

# ── [1] Paths ─────────────────────────────────────────────────────────────────
VENV=/flash/shared/jax/.venv                # path to shared Python venv
JAXTRACE=/flash/shared/jax/JAXTrace
# PKGS=/flash/shared/jax/required-packages

# INPUT: path to the case folder. Accepts either '<case>.gid' or
# '<case>.gid/post'. Ignored when AUTO_DETECT_CASE=1.
INPUT="/flash/users/${USER}/data/<CASE>.gid"

# AUTO_DETECT_CASE: when 1, INPUT is replaced by the directory containing
# this script at runtime. Use this when a copy of the script is placed
# inside each case folder (e.g. .../A1.gid/run_jaxtrace.sh).
AUTO_DETECT_CASE=0

# Subfolders inside <case>.gid/post/ that contain the mesh PVTU files and
# the FEMUSS particle PVTU files. Set to "" when the files sit directly in
# <case>.gid/post/ without an inner subfolder.
MESH_SUBDIR=""             # mesh PVTU subfolder name; "" if none
FEMUSS_SUBDIR=""           # FEMUSS particles subfolder name; "" if none

# Auto-derivation overrides — leave blank to let the script infer them
# from the case folder name (e.g. cylA.gid -> stem 'cylA').
CASE_STEM=""                 # case-stem string used in file patterns
MESH_PATTERN=""              # e.g. "cylA_{timestep}.pvtu"
FEMUSS_PATTERN=""            # e.g. "cylA_pt_{timestep}.pvtu"

# Absolute path overrides. When set, the corresponding *_SUBDIR is ignored
# and the given path is used directly.
MESH_DIR=""                  # directory containing the mesh PVTU files
FEMUSS_DIR=""                # directory containing the FEMUSS particle files

# Optional tag appended to the auto-generated output folder name.
RUN_TAG=""

# OUTPUT_TARGET selects where JAXTrace results are written.
#   scratch -- /flash/users/$USER/run_<RUN_ID> during the run, then moved to
#              /scratch/users/$USER/outputs/<run_folder> at the end.
#   case    -- <case>.gid/<OUTPUT_CASE_SUBFOLDER>, written in place.
OUTPUT_TARGET=scratch
OUTPUT_CASE_SUBFOLDER=post_pt    # used when OUTPUT_TARGET=case

# ── [2] Precision & velocity field ───────────────────────────────────────────
PRECISION=float32          # float32 | float64
VEL_START=159
VEL_END=159
VELOCITY_FIELD=Displacement
LEVELSET_FIELD=LEVEL

# ── [3] Simulation control ───────────────────────────────────────────────────
N_STEPS=2684               # total number of RK4 steps
DT=0.0025                  # RK4 timestep size [s]
LOG_INTERVAL=10            # print + flush stats every N steps
EXPORT_FREQ=1              # export every N steps (combine with NO_EXPORT=1
                           # to disable export entirely)
NO_EXPORT=0                # 1 = no particle export

# ── [4] Particle seeding ──────────────────────────────────────────────────────
# SEED_SOURCE: femuss | box | grid | box-frac | grid-frac | file
#   femuss     — load initial positions from a FEMUSS particle PVTU
#   box        — uniform random inside an absolute box (SEED_BOX, N_PARTICLES)
#   grid       — uniform grid inside an absolute box (SEED_BOX, SEED_GRID)
#   box-frac   — uniform random inside a fractional sub-box of the mesh bbox
#                (SEED_FRACTION, N_PARTICLES)
#   grid-frac  — uniform grid inside a fractional sub-box of the mesh bbox
#                (SEED_FRACTION, SEED_GRID)
#   file       — load positions from a .npy / .npz (SEED_FILE)
SEED_SOURCE=femuss
FEMUSS_START=0
# Absolute box bounds (used by box / grid). Order: XMIN XMAX YMIN YMAX ZMIN ZMAX
SEED_BOX="-0.01 0.01 -0.01 0.01 0.0 0.002"
# Per-axis fractions of the mesh bbox (used by box-frac / grid-frac).
# Order: XLO XHI YLO YHI ZLO ZHI, each in [0, 1] with lo < hi.
# Example: "0.0 0.2 0.0 1.0 0.0 1.0" = first 20% of X, full Y/Z.
SEED_FRACTION="0.0 0.2 0.0 1.0 0.0 1.0"
# Grid resolution (used by grid / grid-frac). Particle count = NX*NY*NZ.
SEED_GRID="50 70 30"
N_PARTICLES=300000          # used by box / box-frac (ignored by grid modes)
SEED_FILE=""
SEED=42

# ── [5] Optional FEMUSS comparison ───────────────────────────────────────────
FEMUSS_COMPARE=1           # 1 = enable, 0 = disable

# ── [6] Search / RK4 kernel ──────────────────────────────────────────────────
RK4_MODE=fused             # fused | split
L1_METHOD=face             # face | node
L2_NEIGHBORHOOD=3          # 3 | 5
L0_SKIP_BAND=0.0
ENHANCED_SEARCH_BAND=0.0
REGISTRATION=""

# ── [7] Boundary / level-set behaviour ───────────────────────────────────────
# BOUNDARY_WALLS: per-wall behaviour as comma-separated 'wall=mode' pairs,
# where wall is one of {x_min, x_max, y_min, y_max, z_min, z_max} and mode
# is either:
#   clamp  -- particles crossing this wall are projected back inside the
#             bounding box (default for any wall not listed)
#   outlet -- particles crossing this wall leave the domain; their
#             element_id is set to -1 and tracking stops for them
# Set to "" to clamp every wall.
BOUNDARY_WALLS="x_max=outlet"
BOUNDARY_PROJ_TOL=1e-6     # inward offset applied when clamping to a wall [m]
POINT_IN_TET_TOL=1e-6      # numerical tolerance for point-in-tet test
LEVELSET_MODE=zero_vel     # how to handle a particle inside the tool region:
                           #   zero_vel  -- velocity at that step is set to 0
                           #   skip_step -- the RK4 step is skipped entirely
FAILED_SUBSTAGE=zero_vel   # policy when an RK4 substage falls outside the mesh:
                           #   zero_vel       -- treat the substage as v=0
                           #   last_valid_vel -- reuse the last interpolated v
                           #   skip_step      -- abandon the step, freeze particle
INTERPOLATION_METHOD=direct_inverse  # P1 velocity interpolation method:
                                     #   direct_inverse | gram_matrix

# ── [8] Pin velocity ──────────────────────────────────────────────────────────
PIN_VELOCITY=1             # 1 = on, 0 = off
PIN_RPM=-600
PIN_CENTER="0.0 0.0 0.0"
PIN_AXIS="0.0 0.0 1.0"
PIN_TILT=0.0

# ── [9] Particle export options ───────────────────────────────────────────────
# EXPORT_FORMAT: container format for the per-step particle output.
#   vtkhdf -- single .vtkhdf archive containing all timesteps.
#             Requires ParaView >= 6.0 / VTK >= 9.4 to read.
#   vtu    -- one .vtu file per step, loaded in ParaView as a numbered
#             series. Choose this for older ParaView installations.
EXPORT_FORMAT=vtkhdf

N_GROUPS=5                 # number of particle groups by initial X; 0 disables
EXPORT_ELEMENT_IDS=0       # 1 = include each particle's host ElementID
EXPORT_ESCAPED_FLAG=0      # 1 = include a per-particle 'Escaped' UInt8 field
                           # (set to 1 the first time element_id<0; useful
                           # for filtering out lost particles in ParaView)

# Temperature export — both flags share the same per-step P1 evaluation, so
# enabling both costs the same as enabling either one.
TRACK_MAX_TEMPERATURE=0    # 1 = export running maximum of TEMPERATURE_FIELD
                           # along each particle's trajectory as 'MaxTemperature'
EXPORT_TEMPERATURE=0       # 1 = export the instantaneous TEMPERATURE_FIELD at
                           # the current particle position as 'Temperature'
TEMPERATURE_FIELD=Temperature  # PVTU field name to read for the above flags

# ── [10] JAX memory & performance ────────────────────────────────────────────
# XLA_PREALLOC controls JAX's GPU allocator strategy.
#   1 -- preallocate VRAM_FRACTION of total VRAM at startup (fixed pool)
#   0 -- on-demand allocator, grows up to VRAM_FRACTION as needed
# Preallocation is faster but blocks other processes from using that VRAM.
XLA_PREALLOC=1

# VRAM_FRACTION: fraction of total GPU VRAM available to this job.
#   With XLA_PREALLOC=0 it acts as a soft cap.
#   With XLA_PREALLOC=1 it determines the reserved pool size.
# Typical values on a 32 GB GPU:
#   0.90  sole GPU user
#   0.45  two parallel jobs sharing the GPU
#   0.30  three parallel jobs
VRAM_FRACTION=0.9

# BENCHMARK_MODE: 1 forces XLA_PREALLOC=1 and disables the background
# GPU/RAM monitor (eliminates monitor overhead for timing measurements).
BENCHMARK_MODE=0

# MONITOR_INTERVAL: seconds between GPU/RAM log entries. 0 disables the
# background monitor.
MONITOR_INTERVAL=30

# =============================================================================
# END USER CONFIGURATION — below this line is infrastructure.
# =============================================================================

# ── Local overrides (untracked) ──────────────────────────────────────────────
# If a sibling file named run_workstation.local.sh exists, source it here so
# host-specific paths and per-experiment knobs override the defaults above
# without modifying the tracked script. The local file is gitignored, so
# `git pull` will never collide with your customisations.
#
# Example contents for scripts/run_workstation.local.sh:
#   INPUT=/path/to/my/<CASE>.gid
#   N_PARTICLES=192000
#   N_STEPS=8000
#   SEED_SOURCE=grid
#   EXPORT_ESCAPED_FLAG=1
#   TRACK_MAX_TEMPERATURE=1
_LOCAL_OVERRIDES="$(dirname "$0")/run_workstation.local.sh"
if [ -f "$_LOCAL_OVERRIDES" ]; then
    echo "[config] Sourcing local overrides: $_LOCAL_OVERRIDES"
    # shellcheck source=/dev/null
    source "$_LOCAL_OVERRIDES"
fi

# ── Generate unique run ID (replaces $SLURM_JOB_ID) ──────────────────────────
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"

# ── Auto-detect case folder from script location if requested ────────────────
# When AUTO_DETECT_CASE=1, the script's own parent directory is used as
# INPUT, regardless of whatever was set above. This is the pattern when you
# copy run_workstation.sh into each case folder (e.g.
#   /scratch/.../A1.gid/run_jaxtrace.sh
# ) and want every case to "just work" without editing INPUT each time.
if [ "${AUTO_DETECT_CASE:-0}" = "1" ]; then
    _SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
    INPUT="$_SCRIPT_DIR"
    echo "[case] AUTO_DETECT_CASE=1: INPUT='$INPUT'"
fi

# ── Derive case name ──────────────────────────────────────────────────────────
if [ -n "$CASE_STEM" ]; then
    _CASE="$CASE_STEM"
else
    _CASE=$(basename "$INPUT" .gid)
    _CASE=$(basename "$_CASE" /post)
fi

# ── Resolve absolute case directory (used by OUTPUT_TARGET=case) ─────────────
_CASE_DIR="$(cd "$INPUT" 2>/dev/null && pwd -P || echo "$INPUT")"
# If user passed '<case>.gid/post', strip the trailing /post so _CASE_DIR
# points at the .gid folder for OUTPUT_TARGET=case.
if [ "$(basename "$_CASE_DIR")" = "post" ]; then
    _CASE_DIR="$(dirname "$_CASE_DIR")"
fi

# ── Output paths ──────────────────────────────────────────────────────────────
# Two layouts:
#   OUTPUT_TARGET=scratch  (default): use /flash for active IO then move to
#                                     /scratch/users/$USER/outputs/<folder>.
#   OUTPUT_TARGET=case:               write straight into the case folder so
#                                     mesh and JAXTrace outputs sit together.
FLASH_OUT=/flash/users/${USER}/run_${RUN_ID}
LOG_DIR=""
case "${OUTPUT_TARGET:-scratch}" in
    case)
        SUB="${OUTPUT_CASE_SUBFOLDER:-post_pt}"
        SCRATCH_OUT="${_CASE_DIR}/${SUB}"
        SCRATCH_BASE="$(dirname "$SCRATCH_OUT")"
        LOG_DIR="${SCRATCH_OUT}/logs"
        # Run python directly into the case folder (no flash→case copy).
        FLASH_OUT="$SCRATCH_OUT"
        echo "[output] OUTPUT_TARGET=case: writing results to '$SCRATCH_OUT'"
        ;;
    scratch)
        SCRATCH_BASE=/scratch/users/${USER}/outputs
        if [ -z "$RUN_TAG" ]; then
            SCRATCH_FOLDER="${_CASE}_jaxtrace_${RUN_ID}"
        else
            SCRATCH_FOLDER="${RUN_TAG}_${RUN_ID}"
        fi
        SCRATCH_OUT="${SCRATCH_BASE}/${SCRATCH_FOLDER}"
        LOG_DIR="${SCRATCH_BASE}/logs"
        ;;
    *)
        echo "ERROR: OUTPUT_TARGET='${OUTPUT_TARGET}' not in {scratch, case}" >&2
        exit 2
        ;;
esac
MONITOR_LOG="${LOG_DIR}/$(basename "$SCRATCH_OUT")_monitor.log"

mkdir -p "$FLASH_OUT" "$SCRATCH_OUT" "$LOG_DIR"

# ── Mirror this script's output to a log file next to the results ────────────
# The terminal still sees everything; tee duplicates each line to log.txt
# inside the run's scratch folder. Cost is negligible: the script and
# run_tracking.py together emit only a few KB of text over a multi-hour run,
# so this is bytes per second of disk I/O, fully absorbed by the page cache.
# Skip the re-exec when already mirroring (avoids infinite recursion when the
# script restarts itself).
RUN_LOG="${SCRATCH_OUT}/log.txt"
if [ "${__JAXTRACE_LOG_ATTACHED:-}" != "1" ]; then
    export __JAXTRACE_LOG_ATTACHED=1
    # exec replaces stdout/stderr with the tee pipe for the rest of this
    # process; subsequent commands' output is captured automatically. The
    # ${PIPESTATUS[0]} machinery preserves the python exit code through tee.
    exec > >(tee -a "$RUN_LOG") 2>&1
    echo "[log] Mirroring full output to $RUN_LOG"
fi

# ── Activate Python venv ──────────────────────────────────────────────────────
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
    echo "[env] Activated venv: $VENV ($(python --version))"
else
    echo "[warn] venv not found at $VENV — using current Python: $(which python)"
fi

# ── CUDA / NVIDIA environment (replaces ROCm/MIOpen from LUMI) ───────────────
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0

# XLA performance flags (CUDA equivalent of LUMI ROCm flags)
export XLA_FLAGS="--xla_gpu_autotune_level=4 --xla_gpu_enable_latency_hiding_scheduler=true"

# CUDA kernel cache (equivalent to MIOpen cache on LUMI)
CUDA_CACHE_DIR="/tmp/${USER}-xla-cache-${RUN_ID}"
export XLA_FLAGS="$XLA_FLAGS --xla_gpu_cuda_data_dir=/usr/local/cuda"
export CUDA_CACHE_PATH="$CUDA_CACHE_DIR"
mkdir -p "$CUDA_CACHE_DIR"

# ── JAX memory allocator (controlled by XLA_PREALLOC and VRAM_FRACTION) ──────
# BENCHMARK_MODE forces preallocation on
[ "$BENCHMARK_MODE" = "1" ] && XLA_PREALLOC=1

if [ "$XLA_PREALLOC" = "1" ]; then
    # Preallocate VRAM_FRACTION of VRAM at startup
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION=$VRAM_FRACTION
    unset  XLA_PYTHON_CLIENT_ALLOCATOR
    echo "[perf] XLA_PREALLOC=ON  — reserving ${VRAM_FRACTION} of VRAM at startup"
else
    # On-demand allocator: grows up to VRAM_FRACTION but doesn't pre-reserve
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
    export XLA_PYTHON_CLIENT_MEM_FRACTION=$VRAM_FRACTION
    echo "[perf] XLA_PREALLOC=OFF — on-demand allocator, cap: ${VRAM_FRACTION} VRAM"
fi

# Suppress verbose TF/JAX logs
export TF_CPP_MIN_LOG_LEVEL=2

# Python path
# export PYTHONPATH="${JAXTRACE}:${PKGS}:${PYTHONPATH:-}"

# ── GPU & Memory Monitor (nvidia-smi replaces rocm-smi) ──────────────────────
MONITOR_PID=""
if [ "$BENCHMARK_MODE" != "1" ] && [ "${MONITOR_INTERVAL}" -gt 0 ] 2>/dev/null; then
(
    echo "=== GPU & Memory Monitor === Run ${RUN_ID} === $(date) ==="
    echo "Interval: ${MONITOR_INTERVAL}s"
    echo ""
    while true; do
        echo "--- $(date '+%Y-%m-%d %H:%M:%S') ---"
        nvidia-smi \
            --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
            --format=csv,noheader,nounits \
        | awk -F', ' '{printf "  GPU  Temp:%s°C  Util:%s%%  VRAM:%s/%s MiB  Power:%sW\n",$1,$2,$3,$4,$5}'
        echo ""
        free -h | head -2
        echo ""
        sleep "$MONITOR_INTERVAL"
    done
) > "$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!
fi

# ── Build CLI argument list ───────────────────────────────────────────────────
ARGS=(
    --input              "$INPUT"
    --output             "$FLASH_OUT"
    --mesh-subdir        "$MESH_SUBDIR"
    --femuss-subdir      "$FEMUSS_SUBDIR"
    --precision          "$PRECISION"
    --vel-range          "$VEL_START" "$VEL_END"
    --velocity-field     "$VELOCITY_FIELD"
    --levelset-field     "$LEVELSET_FIELD"
    --n-steps            "$N_STEPS"
    --dt                 "$DT"
    --log-interval       "$LOG_INTERVAL"
    --export-freq        "$EXPORT_FREQ"
    --seed-source        "$SEED_SOURCE"
    --seed               "$SEED"
    --rk4-mode           "$RK4_MODE"
    --l1-method          "$L1_METHOD"
    --l2-neighborhood    "$L2_NEIGHBORHOOD"
    --l0-skip-band       "$L0_SKIP_BAND"
    --enhanced-search-band "$ENHANCED_SEARCH_BAND"
    --boundary-proj-tol  "$BOUNDARY_PROJ_TOL"
    --point-in-tet-tol   "$POINT_IN_TET_TOL"
    --levelset-mode      "$LEVELSET_MODE"
    --failed-substage    "$FAILED_SUBSTAGE"
    --interpolation-method "$INTERPOLATION_METHOD"
    --pin-rpm            "$PIN_RPM"
    --pin-center         $PIN_CENTER
    --pin-axis           $PIN_AXIS
    --pin-tilt           "$PIN_TILT"
    --n-groups           "$N_GROUPS"
)

[ -n "$CASE_STEM"        ] && ARGS+=( --case-stem          "$CASE_STEM"        )
[ -n "$MESH_PATTERN"     ] && ARGS+=( --mesh-pattern        "$MESH_PATTERN"     )
[ -n "$FEMUSS_PATTERN"   ] && ARGS+=( --femuss-pattern      "$FEMUSS_PATTERN"   )
[ -n "$MESH_DIR"         ] && ARGS+=( --mesh-dir            "$MESH_DIR"         )
[ -n "$FEMUSS_DIR"       ] && ARGS+=( --femuss-dir          "$FEMUSS_DIR"       )
[ -n "$RUN_TAG"          ] && ARGS+=( --run-tag             "$RUN_TAG"          )
[ -n "$BOUNDARY_WALLS"   ] && ARGS+=( --boundary-walls      "$BOUNDARY_WALLS"   )
[ -n "$REGISTRATION"     ] && ARGS+=( --registration        "$REGISTRATION"     )
[ "$NO_EXPORT"          = "1" ] && ARGS+=( --no-export           )
ARGS+=( --export-format "$EXPORT_FORMAT" )
[ "$EXPORT_ELEMENT_IDS" = "1" ] && ARGS+=( --export-element-ids  )
[ "$EXPORT_ESCAPED_FLAG" = "1" ] && ARGS+=( --export-escaped-flag )
[ "$TRACK_MAX_TEMPERATURE" = "1" ] && ARGS+=( --track-max-temperature )
[ "$EXPORT_TEMPERATURE"     = "1" ] && ARGS+=( --export-temperature )
if [ "$TRACK_MAX_TEMPERATURE" = "1" ] || [ "$EXPORT_TEMPERATURE" = "1" ]; then
    ARGS+=( --temperature-field "$TEMPERATURE_FIELD" )
fi
[ "$PIN_VELOCITY"       = "0" ] && ARGS+=( --no-pin-velocity      )

case "$SEED_SOURCE" in
    femuss)
        ARGS+=( --femuss-start "$FEMUSS_START" )
        [ "$FEMUSS_COMPARE" = "1" ] && ARGS+=( --femuss-compare )
        ;;
    box)
        ARGS+=( --seed-box $SEED_BOX --n-particles "$N_PARTICLES" )
        ;;
    grid)
        ARGS+=( --seed-box $SEED_BOX --seed-grid $SEED_GRID )
        ;;
    box-frac)
        ARGS+=( --seed-fraction $SEED_FRACTION --n-particles "$N_PARTICLES" )
        ;;
    grid-frac)
        ARGS+=( --seed-fraction $SEED_FRACTION --seed-grid $SEED_GRID )
        ;;
    file)
        ARGS+=( --seed-file "$SEED_FILE" )
        ;;
    *)
        echo "ERROR: unknown SEED_SOURCE='$SEED_SOURCE' (expected femuss|box|grid|box-frac|grid-frac|file)" >&2
        exit 2
        ;;
esac

# ── Print run summary ─────────────────────────────────────────────────────────
VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
VRAM_CAP_MB=$(echo "$VRAM_TOTAL $VRAM_FRACTION" | awk '{printf "%.0f", $1*$2}')

echo "======================================================"
echo " JAXTrace — FSW GPU Workstation Run"
echo "======================================================"
echo " Run ID:       $RUN_ID"
echo " User:         $USER"
echo " Case:         $_CASE"
echo " Seed source:  $SEED_SOURCE"
echo " Precision:    $PRECISION"
echo " GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo " XLA prealloc: $( [ "$XLA_PREALLOC" = "1" ] && echo "ON" || echo "OFF (on-demand)" )"
echo " VRAM cap:     ${VRAM_FRACTION} (≈ ${VRAM_CAP_MB} / ${VRAM_TOTAL} MiB)"
echo " Flash out:    $FLASH_OUT"
echo " Scratch out:  $SCRATCH_OUT"
echo " Monitor log:  $MONITOR_LOG"
echo " Started:      $(date)"
echo "======================================================"
echo ""

# ── Run simulation ─────────────────────────────────────────────────────────────
python "${JAXTRACE}/run_tracking.py" "${ARGS[@]}"
SIM_EXIT=$?

# ── Stop monitor ──────────────────────────────────────────────────────────────
if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" 2>/dev/null
    wait "$MONITOR_PID" 2>/dev/null
fi

echo ""
echo "Simulation exited with code $SIM_EXIT at $(date)"

# ── Move results from flash → scratch ─────────────────────────────────────────
# With OUTPUT_TARGET=case, FLASH_OUT == SCRATCH_OUT, so the move is a no-op.
if [ "$FLASH_OUT" != "$SCRATCH_OUT" ]; then
    echo "Moving results from /flash to /scratch..."
    mv "$FLASH_OUT"/* "$SCRATCH_OUT"/ 2>/dev/null
    rmdir "$FLASH_OUT" 2>/dev/null
fi
[ -f "$MONITOR_LOG" ] && mv "$MONITOR_LOG" "$SCRATCH_OUT/logs/" 2>/dev/null

# Cleanup CUDA cache
rm -rf "$CUDA_CACHE_DIR"

echo ""
echo "======================================================"
echo " Done. Exit code: $SIM_EXIT"
echo " Results: $SCRATCH_OUT"
echo " Logs:    $LOG_DIR"
echo "======================================================"

exit $SIM_EXIT
