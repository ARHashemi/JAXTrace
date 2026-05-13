#!/bin/bash
# SLURM directives. SLURM does not expand shell variables here; fill in your
# LUMI project ID and log directory before submitting. Override at submission
# time with:
#   sbatch --account=project_XXXXXXXXX --output=... --error=... run_lumi.sh
#SBATCH --job-name=jaxtrace
#SBATCH --partition=small-g
#SBATCH --account=project_XXXXXXXXX
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --signal=B:SIGTERM@120   # SIGTERM batch shell 120s before time limit
                                  # so run_tracking.py can flush its VTKHDF
                                  # archive before SLURM SIGKILLs everything
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# USER CONFIGURATION — edit these groups to control the production run.
# Runs run_tracking.py (the general-purpose driver), not the FEMUSS benchmark.
# =============================================================================

# ── [1] Paths ────────────────────────────────────────────────────────────────
# LUMI project ID. Defaults to the SLURM account used to submit the job
# (SLURM_JOB_ACCOUNT), which is set automatically when run via `sbatch`.
PROJECT="${SLURM_JOB_ACCOUNT:-project_XXXXXXXXX}"

# Singularity image with JAX + ROCm.
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif

# Per-user paths derived from $PROJECT and $USER. Override in a local config
# (scripts/run_lumi.local.sh, untracked) if your layout differs.
JAXTRACE="/project/${PROJECT}/${USER}/JAXTrace"
PKGS="/project/${PROJECT}/${USER}/required-packages"

# INPUT: path to the case folder. Accepts either '<case>.gid' or
# '<case>.gid/post'. Ignored when AUTO_DETECT_CASE=1.
INPUT="/scratch/${PROJECT}/${USER}/data/<CASE>.gid"

# AUTO_DETECT_CASE: when 1, INPUT is replaced by the directory containing
# this script at runtime. Use this when a copy of the script is placed
# inside each case folder (e.g. .../A1.gid/run_jaxtrace.sh).
AUTO_DETECT_CASE=0

# Subfolders inside <case>.gid/post/ that contain the mesh PVTU files and
# the FEMUSS particle PVTU files. Set to "" when the files sit directly in
# <case>.gid/post/ without an inner subfolder.
MESH_SUBDIR=""                # mesh PVTU subfolder name; "" if none
FEMUSS_SUBDIR=""              # FEMUSS particles subfolder name; "" if none

# Auto-derivation overrides — leave blank to let the script infer them
# from the case folder name (e.g. cylA.gid -> stem 'cylA').
CASE_STEM=""                  # case-stem string used in file patterns
MESH_PATTERN=""               # e.g. "cylA_{timestep}.pvtu"
FEMUSS_PATTERN=""             # e.g. "cylA_pt_{timestep}.pvtu"

# Absolute path overrides. When set, the corresponding *_SUBDIR is ignored
# and the given path is used directly.
MESH_DIR=""                   # directory containing the mesh PVTU files
FEMUSS_DIR=""                 # directory containing the FEMUSS particle files

# Optional tag appended to the auto-generated output folder name.
RUN_TAG=""

# OUTPUT_TARGET selects where JAXTrace results are written.
#   scratch -- /flash/${PROJECT}/${USER}/run_<JOB_ID> during the run, then
#              moved to /scratch/${PROJECT}/${USER}/outputs/<run_folder>.
#   case    -- <case>.gid/<OUTPUT_CASE_SUBFOLDER>, written in place.
OUTPUT_TARGET=scratch
OUTPUT_CASE_SUBFOLDER=post_pt   # used when OUTPUT_TARGET=case

# ── [2] Precision & velocity field ───────────────────────────────────────────
PRECISION=float32             # float32 | float64
VEL_START=159                 # velocity timestep cyclic start
VEL_END=159                   # velocity timestep cyclic end
VELOCITY_FIELD=Displacement
LEVELSET_FIELD=LEVEL

# ── [3] Simulation control ───────────────────────────────────────────────────
N_STEPS=2684                  # total number of RK4 steps
DT=0.0025                     # RK4 timestep size [s]
LOG_INTERVAL=10               # print + flush stats every N steps
EXPORT_FREQ=1                 # export every N steps (combine with NO_EXPORT=1
                              # to disable export entirely)
NO_EXPORT=0                   # 1 = no particle export

# ── [4] Particle seeding ─────────────────────────────────────────────────────
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
FEMUSS_START=0                # used when SEED_SOURCE=femuss
# Absolute box bounds (used by box / grid). Order: XMIN XMAX YMIN YMAX ZMIN ZMAX
SEED_BOX="-0.01 0.01 -0.01 0.01 0.0 0.002"
# Per-axis fractions of the mesh bbox (used by box-frac / grid-frac).
# Order: XLO XHI YLO YHI ZLO ZHI, each in [0, 1] with lo < hi.
# Example: "0.0 0.2 0.0 1.0 0.0 1.0" = first 20% of X, full Y/Z.
SEED_FRACTION="0.0 0.2 0.0 1.0 0.0 1.0"
# Grid resolution (used by grid / grid-frac). Particle count = NX*NY*NZ.
SEED_GRID="50 70 30"
N_PARTICLES=100000            # used by box / box-frac (ignored by grid modes)
SEED_FILE=""                  # used when SEED_SOURCE=file
SEED=42

# ── [5] Optional FEMUSS comparison ───────────────────────────────────────────
# Enable only when SEED_SOURCE=femuss and reference file exists.
FEMUSS_COMPARE=1              # 1 = enable, 0 = disable

# ── [6] Search / RK4 kernel ──────────────────────────────────────────────────
RK4_MODE=fused                # fused | split
L1_METHOD=face                # face | node
L2_NEIGHBORHOOD=3             # 3 | 5
L0_SKIP_BAND=0.0              # 0 = mixed-sign only; e.g. 0.5e-3 for ±0.5mm
ENHANCED_SEARCH_BAND=0.0      # 0 = off; e.g. 1e-3 for ±1mm node-L1+5x5x5
REGISTRATION=""               # "" = default | vertex_multi | parent_cube

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
BOUNDARY_PROJ_TOL=1e-6        # inward offset applied when clamping to a wall [m]
POINT_IN_TET_TOL=1e-6         # numerical tolerance for point-in-tet test
LEVELSET_MODE=zero_vel        # how to handle a particle inside the tool region:
                              #   zero_vel  -- velocity at that step is set to 0
                              #   skip_step -- the RK4 step is skipped entirely
FAILED_SUBSTAGE=zero_vel      # policy when an RK4 substage falls outside the mesh:
                              #   zero_vel       -- treat the substage as v=0
                              #   last_valid_vel -- reuse the last interpolated v
                              #   skip_step      -- abandon the step, freeze particle
INTERPOLATION_METHOD=direct_inverse   # P1 velocity interpolation method:
                                      #   direct_inverse | gram_matrix

# ── [8] Pin velocity (FEMUSS FSW equivalent) ─────────────────────────────────
PIN_VELOCITY=1                # 1 = on, 0 = off
PIN_RPM=-600
PIN_CENTER="0.0 0.0 0.0"
PIN_AXIS="0.0 0.0 1.0"
PIN_TILT=0.0

# ── [9] Particle export options ──────────────────────────────────────────────
# EXPORT_FORMAT: container format for the per-step particle output.
#   vtkhdf -- single .vtkhdf archive containing all timesteps.
#             Requires ParaView >= 6.0 / VTK >= 9.4 to read.
#   vtu    -- one .vtu file per step, loaded in ParaView as a numbered
#             series. Choose this for older ParaView installations.
EXPORT_FORMAT=vtkhdf

N_GROUPS=5                    # number of particle groups by initial X; 0 disables
EXPORT_ELEMENT_IDS=0          # 1 = include each particle's host ElementID
EXPORT_ESCAPED_FLAG=0         # 1 = include a per-particle 'Escaped' UInt8 field
                              # (set to 1 the first time element_id<0; useful
                              # for filtering out lost particles in ParaView)

# Temperature export — both flags share the same per-step P1 evaluation, so
# enabling both costs the same as enabling either one.
TRACK_MAX_TEMPERATURE=0       # 1 = export running maximum of TEMPERATURE_FIELD
                              # along each particle's trajectory as 'MaxTemperature'
EXPORT_TEMPERATURE=0          # 1 = export the instantaneous TEMPERATURE_FIELD at
                              # the current particle position as 'Temperature'
TEMPERATURE_FIELD=Temperature # PVTU field name to read for the above flags

# ── [10] Performance / monitoring ────────────────────────────────────────────
# BENCHMARK_MODE: 1 forces XLA_PREALLOC=1 and disables the background
# GPU/RAM monitor (eliminates monitor overhead for timing measurements).
BENCHMARK_MODE=0

# XLA_PREALLOC controls JAX's GPU allocator strategy.
#   1 -- preallocate ~75% of total HBM at startup (fixed pool)
#   0 -- on-demand allocator, grows as needed
# Preallocation is faster but blocks other processes from using that HBM.
XLA_PREALLOC=0

# MONITOR_INTERVAL: seconds between GPU/RAM log entries. 0 disables the
# background monitor. Ignored when BENCHMARK_MODE=1.
MONITOR_INTERVAL=30

# =============================================================================
# END USER CONFIGURATION — below this line is infrastructure.
# =============================================================================

# ── Local overrides (untracked) ──────────────────────────────────────────────
# If scripts/run_lumi.local.sh exists alongside this file, source it here.
# Use it for per-user paths and per-experiment knobs that you don't want to
# commit. The .gitignore in this repo already excludes *.local.sh so the
# file will never be picked up by `git add` or collide with `git pull`.
_LOCAL_OVERRIDES="$(dirname "$0")/run_lumi.local.sh"
if [ -f "$_LOCAL_OVERRIDES" ]; then
    echo "[config] Sourcing local overrides: $_LOCAL_OVERRIDES"
    # shellcheck source=/dev/null
    source "$_LOCAL_OVERRIDES"
fi

# ── Auto-detect case folder from script location if requested ──────────────
if [ "${AUTO_DETECT_CASE:-0}" = "1" ]; then
  _SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
  INPUT="$_SCRIPT_DIR"
  echo "[case] AUTO_DETECT_CASE=1: INPUT='$INPUT'"
fi

# ── Derive case name (needed for default output folder) ─────────────────────
if [ -n "$CASE_STEM" ]; then
  _CASE="$CASE_STEM"
else
  _CASE=$(basename "$INPUT" .gid)
  _CASE=$(basename "$_CASE" /post)   # strip trailing /post if present
fi

# ── Resolve absolute case directory (used by OUTPUT_TARGET=case) ───────────
_CASE_DIR="$(cd "$INPUT" 2>/dev/null && pwd -P || echo "$INPUT")"
if [ "$(basename "$_CASE_DIR")" = "post" ]; then
  _CASE_DIR="$(dirname "$_CASE_DIR")"
fi

# ── Output paths ────────────────────────────────────────────────────────────
# Two layouts:
#   OUTPUT_TARGET=scratch (default): /flash for active IO, then move to
#                                    /scratch/${PROJECT}/${USER}/outputs/.
#   OUTPUT_TARGET=case             : write straight into the case folder.
FLASH_OUT="/flash/${PROJECT}/${USER}/run_${SLURM_JOB_ID}"
case "${OUTPUT_TARGET:-scratch}" in
  case)
    SUB="${OUTPUT_CASE_SUBFOLDER:-post_pt}"
    SCRATCH_OUT="${_CASE_DIR}/${SUB}"
    SCRATCH_BASE="$(dirname "$SCRATCH_OUT")"
    FLASH_OUT="$SCRATCH_OUT"   # python writes directly, no flash hop
    echo "[output] OUTPUT_TARGET=case: writing to '$SCRATCH_OUT'"
    ;;
  scratch)
    SCRATCH_BASE="/scratch/${PROJECT}/${USER}/outputs"
    SCRATCH_FOLDER=""             # auto: "${_CASE}_jaxtrace_${SLURM_JOB_ID}"
    if [ -z "$SCRATCH_FOLDER" ]; then
      SCRATCH_FOLDER="${_CASE}_jaxtrace_${SLURM_JOB_ID}"
    fi
    SCRATCH_OUT="${SCRATCH_BASE}/${SCRATCH_FOLDER}"
    ;;
  *)
    echo "ERROR: OUTPUT_TARGET='${OUTPUT_TARGET}' not in {scratch, case}" >&2
    exit 2
    ;;
esac
MONITOR_LOG="${SCRATCH_BASE}/$(basename "$SCRATCH_OUT")_monitor.log"

mkdir -p $FLASH_OUT $SCRATCH_OUT

# ── MIOpen cache to RAM ──────────────────────────────────────────────────────
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

# ── ROCm / XLA performance flags for MI250X on LUMI-G ───────────────────────
export JAX_PLATFORMS=rocm
export ROCR_VISIBLE_DEVICES=0
export XLA_FLAGS="--xla_gpu_autotune_level=4 --xla_gpu_enable_latency_hiding_scheduler=true"
# Memory allocator: default = on-demand (platform). BENCHMARK_MODE or
# XLA_PREALLOC=1 switches to pooled preallocation (faster kernel launches,
# but reserves ~75% HBM up-front and can OOM if HBM is shared).
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
# Disable HSA SDMA for host<->device transfers (set to 1 to enable).
export HSA_ENABLE_SDMA=0

# ── GPU & Memory Monitor (background) ───────────────────────────────────────
# Logs rocm-smi + free at $MONITOR_INTERVAL seconds when BENCHMARK_MODE != 1.
# The monitor runs in its own process group via `setsid` so the cleanup
# `kill -- -$MONITOR_PGID` reaches both the bash subshell AND its current
# `sleep` child (a plain `kill` to the bash PID would only fire after the
# sleep returns, leaving the monitor alive up to $MONITOR_INTERVAL seconds
# after the run completes).
MONITOR_PID=""
MONITOR_PGID=""
if [ "$BENCHMARK_MODE" != "1" ] && [ "$MONITOR_INTERVAL" -gt 0 ] 2>/dev/null; then
  setsid bash -c '
    echo "=== GPU & Memory Monitor === Job '"$SLURM_JOB_ID"' === $(date) ==="
    echo "Interval: '"${MONITOR_INTERVAL}"'s"
    echo ""
    while true; do
      echo "--- $(date '\''+%Y-%m-%d %H:%M:%S'\'') ---"
      if command -v rocm-smi &>/dev/null; then
        rocm-smi --showuse --showmemuse --showtemp 2>/dev/null \
          | grep -E "GPU|%|MiB|Temperature" \
          || rocm-smi 2>/dev/null | tail -n +3
      fi
      echo ""
      free -h | head -2
      echo ""
      sleep '"${MONITOR_INTERVAL}"'
    done
  ' > "$MONITOR_LOG" 2>&1 &
  MONITOR_PID=$!
  MONITOR_PGID=$MONITOR_PID   # setsid makes the new process its own pgleader
fi

# Ensure the monitor is reaped on any kind of exit (normal, error, SIGTERM
# from SLURM, Ctrl-C). trap EXIT is bash's unconditional cleanup hook.
_cleanup_monitor() {
  if [ -n "${MONITOR_PGID:-}" ]; then
    kill -- -"$MONITOR_PGID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap _cleanup_monitor EXIT

# ── Build CLI argument list from user config ────────────────────────────────
ARGS=(
  --input  "$INPUT"
  --output "$FLASH_OUT"
  --mesh-subdir   "$MESH_SUBDIR"
  --femuss-subdir "$FEMUSS_SUBDIR"
  --precision      "$PRECISION"
  --vel-range      "$VEL_START" "$VEL_END"
  --velocity-field "$VELOCITY_FIELD"
  --levelset-field "$LEVELSET_FIELD"
  --n-steps        "$N_STEPS"
  --dt             "$DT"
  --log-interval   "$LOG_INTERVAL"
  --export-freq    "$EXPORT_FREQ"
  --seed-source    "$SEED_SOURCE"
  --seed           "$SEED"
  --rk4-mode       "$RK4_MODE"
  --l1-method      "$L1_METHOD"
  --l2-neighborhood "$L2_NEIGHBORHOOD"
  --l0-skip-band   "$L0_SKIP_BAND"
  --enhanced-search-band "$ENHANCED_SEARCH_BAND"
  --boundary-proj-tol    "$BOUNDARY_PROJ_TOL"
  --point-in-tet-tol     "$POINT_IN_TET_TOL"
  --levelset-mode        "$LEVELSET_MODE"
  --failed-substage      "$FAILED_SUBSTAGE"
  --interpolation-method "$INTERPOLATION_METHOD"
  --pin-rpm    "$PIN_RPM"
  --pin-center $PIN_CENTER
  --pin-axis   $PIN_AXIS
  --pin-tilt   "$PIN_TILT"
  --n-groups   "$N_GROUPS"
)

# Optional / conditional flags
[ -n "$CASE_STEM" ]       && ARGS+=( --case-stem "$CASE_STEM" )
[ -n "$MESH_PATTERN" ]    && ARGS+=( --mesh-pattern "$MESH_PATTERN" )
[ -n "$FEMUSS_PATTERN" ]  && ARGS+=( --femuss-pattern "$FEMUSS_PATTERN" )
[ -n "$MESH_DIR" ]        && ARGS+=( --mesh-dir "$MESH_DIR" )
[ -n "$FEMUSS_DIR" ]      && ARGS+=( --femuss-dir "$FEMUSS_DIR" )
[ -n "$RUN_TAG" ]         && ARGS+=( --run-tag "$RUN_TAG" )
[ -n "$BOUNDARY_WALLS" ]  && ARGS+=( --boundary-walls "$BOUNDARY_WALLS" )
[ -n "$REGISTRATION" ]    && ARGS+=( --registration "$REGISTRATION" )
[ "$NO_EXPORT" = "1" ]    && ARGS+=( --no-export )
ARGS+=( --export-format "$EXPORT_FORMAT" )
[ "$EXPORT_ELEMENT_IDS" = "1" ] && ARGS+=( --export-element-ids )
[ "$EXPORT_ESCAPED_FLAG" = "1" ] && ARGS+=( --export-escaped-flag )
[ "$TRACK_MAX_TEMPERATURE" = "1" ] && ARGS+=( --track-max-temperature )
[ "$EXPORT_TEMPERATURE"     = "1" ] && ARGS+=( --export-temperature )
if [ "$TRACK_MAX_TEMPERATURE" = "1" ] || [ "$EXPORT_TEMPERATURE" = "1" ]; then
  ARGS+=( --temperature-field "$TEMPERATURE_FIELD" )
fi
[ "$PIN_VELOCITY" = "0" ] && ARGS+=( --no-pin-velocity )

# Seed-source specific
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

# ── Run simulation → writes to Flash ────────────────────────────────────────
echo "Starting simulation at $(date)"
echo "Monitor log: $MONITOR_LOG"
echo "Script:      run_tracking.py"
echo "Seed source: $SEED_SOURCE"
echo "CLI args:    ${ARGS[*]}"
echo ""

# Only forward XLA_PYTHON_CLIENT_ALLOCATOR when it is set (prealloc mode unsets it).
ALLOC_ENV=()
if [ -n "${XLA_PYTHON_CLIENT_ALLOCATOR:-}" ]; then
  ALLOC_ENV=( --env "XLA_PYTHON_CLIENT_ALLOCATOR=$XLA_PYTHON_CLIENT_ALLOCATOR" )
fi

# Trap SIGTERM so that when SLURM signals 120 s before the time limit
# (--signal=B:SIGTERM@120), we forward it to the srun job step. srun in
# turn forwards SIGTERM to the singularity/python child, which lets
# run_tracking.py's signal handler flush the VTKHDF archive cleanly.
# The trap MUST be a one-liner that calls scancel/kill rather than waits,
# otherwise bash blocks waiting for the foreground job to finish.
_forward_sigterm() {
  echo "[trap] Forwarding SIGTERM to srun step (PID $SRUN_PID)..."
  if [ -n "${SRUN_PID:-}" ]; then
    # scancel --signal forwards to the job step's tasks (python).
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
  $SIF \
  python $JAXTRACE/run_tracking.py "${ARGS[@]}" &
SRUN_PID=$!
wait $SRUN_PID
SIM_EXIT=$?

# ── Stop monitor ─────────────────────────────────────────────────────────────
# Handled unconditionally by the EXIT trap installed alongside the monitor.
# Calling it explicitly here closes the log before the move/copy steps so
# the archived monitor log captures only the active run.
_cleanup_monitor

echo ""
echo "Simulation exited with code $SIM_EXIT at $(date)"

# ── Move results to Scratch after job ───────────────────────────────────────
# When OUTPUT_TARGET=case, FLASH_OUT == SCRATCH_OUT and there's nothing to
# move; python wrote directly into the case folder.
if [ "$FLASH_OUT" != "$SCRATCH_OUT" ]; then
  echo "Moving results from flash to scratch..."
  mv $FLASH_OUT/* $SCRATCH_OUT/ 2>/dev/null
  rmdir $FLASH_OUT 2>/dev/null
fi

# Copy SLURM stdout/stderr and monitor log into the results folder. The
# default --output / --error paths in the #SBATCH header are relative to
# SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from).
SLURM_OUT="${SLURM_SUBMIT_DIR:-.}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
SLURM_ERR="${SLURM_SUBMIT_DIR:-.}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
mkdir -p "$SCRATCH_OUT/logs"
[ -f "$SLURM_OUT" ]   && cp "$SLURM_OUT" "$SCRATCH_OUT/logs/"
[ -f "$SLURM_ERR" ]   && cp "$SLURM_ERR" "$SCRATCH_OUT/logs/"
[ -f "$MONITOR_LOG" ] && mv "$MONITOR_LOG" "$SCRATCH_OUT/logs/"

echo "Done. All results, logs and monitoring in:"
echo "  $SCRATCH_OUT"
