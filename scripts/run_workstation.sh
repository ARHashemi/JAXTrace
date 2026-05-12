#!/bin/bash
# =============================================================================
# run_workstation.sh — JAXTrace runner for FSW GPU Workstation (fsw_gpu)
#
# Adapted from run_lumi.sh:
#   - No SLURM — direct execution or via task-spooler (ts)
#   - NVIDIA RTX 5090 / CUDA (replaces AMD MI250X / ROCm)
#   - Python venv (replaces Singularity container)
#   - /flash (NVMe) + /scratch (HDD) storage layout
#
# Usage:
#   Foreground:               bash run_workstation.sh
#   Background (survives SSH): nohup bash run_workstation.sh \
#                               > $SCRATCH/logs/run_$(date +%Y%m%d_%H%M%S).out 2>&1 &
#   Via task-spooler queue:   TS_SOCKET=/tmp/gpu_queue ts bash run_workstation.sh
#   Inside screen:            screen -S jaxtrace; bash run_workstation.sh
#   Inside tmux:              tmux new -s jaxtrace; bash run_workstation.sh
# =============================================================================

# =============================================================================
# USER CONFIGURATION — edit these groups to control the production run.
# =============================================================================

# ── [1] Paths ─────────────────────────────────────────────────────────────────
VENV=/flash/shared/jax/.venv                # path to shared Python venv
JAXTRACE=/flash/shared/jax/JAXTrace
# PKGS=/flash/shared/jax/required-packages

# INPUT: point at the FEMUSS case folder. Either '.gid' or '.gid/post'.
INPUT=/flash/users/ali/data/cylA.gid
MESH_SUBDIR=0eule          # subfolder under post/ with mesh PVTU
FEMUSS_SUBDIR=1part        # subfolder under post/ with FEMUSS particles
CASE_STEM=""               # override auto-detected case stem (empty = auto)
MESH_PATTERN=""            # override, e.g. "cylA_{timestep}.pvtu" (empty = auto)
FEMUSS_PATTERN=""          # override, e.g. "cylA_pt_{timestep}.pvtu" (empty = auto)
MESH_DIR=""                # direct path override for mesh files
FEMUSS_DIR=""              # direct path override for FEMUSS particle files
RUN_TAG=""                 # optional custom subfolder; empty = auto

# ── [2] Precision & velocity field ───────────────────────────────────────────
PRECISION=float32          # float32 | float64
VEL_START=159
VEL_END=159
VELOCITY_FIELD=Displacement
LEVELSET_FIELD=LEVEL

# ── [3] Simulation control ───────────────────────────────────────────────────
N_STEPS=2684
DT=0.0025
LOG_INTERVAL=10
EXPORT_FREQ=1              # every N steps; 0 + NO_EXPORT=1 to disable
NO_EXPORT=0                # 1 = disable all VTU output (timing run)

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
# Per-wall semantics for BOUNDARY_WALLS (comma-separated "wall=mode" pairs):
#   <wall>=clamp   particles are pulled back inside (default for every wall)
#   <wall>=outlet  the wall does NOT clamp; particles pass through and are
#                  treated as escaped (their element_id becomes -1)
# Any wall not listed uses the default 'clamp'. To clamp every wall, set "".
#
# Default for this case: only x_max is an outlet; everything else clamps,
# including z_max (so particles leaving through the top get projected back
# inside via the boundary-projection mechanism).
BOUNDARY_WALLS="x_max=outlet"
BOUNDARY_PROJ_TOL=1e-6                  # inward tolerance for boundary projection
POINT_IN_TET_TOL=1e-6
LEVELSET_MODE=zero_vel     # zero_vel | skip_step
FAILED_SUBSTAGE=zero_vel   # zero_vel | last_valid_vel | skip_step
INTERPOLATION_METHOD=direct_inverse  # direct_inverse | gram_matrix

# ── [8] Pin velocity ──────────────────────────────────────────────────────────
PIN_VELOCITY=1             # 1 = on, 0 = off
PIN_RPM=-600
PIN_CENTER="0.0 0.0 0.0"
PIN_AXIS="0.0 0.0 1.0"
PIN_TILT=0.0

# ── [9] Particle export options ───────────────────────────────────────────────
# EXPORT_FORMAT: vtkhdf | vtu
#   vtkhdf (default) -- single transient .vtkhdf archive for the whole run.
#                       Requires ParaView >= 6.0 / VTK >= 9.4.
#                       Much faster to write and transfer than per-step VTUs.
#   vtu              -- legacy: one .vtu per step, opened in ParaView as a
#                       numbered series. Use this on older ParaView installs.
EXPORT_FORMAT=vtkhdf
N_GROUPS=5
EXPORT_ELEMENT_IDS=0       # 1 = include ElementID field
EXPORT_ESCAPED_FLAG=0      # 1 = add 'Escaped' (0/1) per-particle flag set
                           # when element_id<0 at any step; useful as a
                           # Paraview Threshold filter to remove escapees
TRACK_MAX_TEMPERATURE=0    # 1 = track per-particle max of TEMPERATURE_FIELD
                           # over the trajectory (exported as 'MaxTemperature').
                           # Adds ~1-3% per RK4 step and ~1.3 GB GPU memory
                           # for the extra (n_timesteps, n_nodes) scalar stack.
TEMPERATURE_FIELD=Temperature  # PVTU field name when TRACK_MAX_TEMPERATURE=1

# ── [10] JAX memory & performance ────────────────────────────────────────────
#
# XLA_PREALLOC:
#   1 → JAX preallocates VRAM_FRACTION of total VRAM at startup.
#       Faster kernel launches; no other JAX process can share that VRAM.
#       Use when you are the ONLY GPU user or in BENCHMARK_MODE.
#   0 → On-demand allocator. JAX grows its pool as needed up to VRAM_FRACTION.
#       Safe for shared use (default for production on this workstation).
XLA_PREALLOC=1

# VRAM_FRACTION: fraction of total GPU VRAM this job may use.
#   With XLA_PREALLOC=0: JAX will not exceed this fraction (soft cap).
#   With XLA_PREALLOC=1: JAX reserves exactly this fraction at startup.
#
#   Guidelines for this workstation (RTX 5090, 32 GB VRAM):
#     Sole GPU user:       0.90  (~28.8 GB)
#     Two jobs in parallel: 0.45 (~14.4 GB each)
#     Three jobs:          0.30  (~9.6 GB each)
#     Leave headroom:      0.45  (default — safe for shared use)
VRAM_FRACTION=0.9

# BENCHMARK_MODE=1: disables the background monitor and forces XLA_PREALLOC=1.
# Use for pure timing runs only.
BENCHMARK_MODE=0

# MONITOR_INTERVAL: seconds between GPU/RAM log entries. 0 = disable monitor.
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
#   INPUT=/flash/users/ali/data/A2.gid
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

# ── Derive case name ──────────────────────────────────────────────────────────
if [ -n "$CASE_STEM" ]; then
    _CASE="$CASE_STEM"
else
    _CASE=$(basename "$INPUT" .gid)
    _CASE=$(basename "$_CASE" /post)
fi

# ── Output paths ──────────────────────────────────────────────────────────────
FLASH_OUT=/flash/users/${USER}/run_${RUN_ID}
SCRATCH_BASE=/scratch/users/${USER}/outputs
if [ -z "$RUN_TAG" ]; then
    SCRATCH_FOLDER="${_CASE}_jaxtrace_${RUN_ID}"
else
    SCRATCH_FOLDER="${RUN_TAG}_${RUN_ID}"
fi
SCRATCH_OUT="${SCRATCH_BASE}/${SCRATCH_FOLDER}"
LOG_DIR="${SCRATCH_BASE}/logs"
MONITOR_LOG="${LOG_DIR}/${SCRATCH_FOLDER}_monitor.log"

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
[ "$TRACK_MAX_TEMPERATURE" = "1" ] && ARGS+=( --track-max-temperature --temperature-field "$TEMPERATURE_FIELD" )
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
echo "Moving results from /flash to /scratch..."
mv "$FLASH_OUT"/* "$SCRATCH_OUT"/ 2>/dev/null
rmdir "$FLASH_OUT" 2>/dev/null
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
