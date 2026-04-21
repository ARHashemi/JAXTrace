#!/bin/bash
#SBATCH --job-name=jaxtrace
#SBATCH --partition=small-g
#SBATCH --account=project_465002752
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project_465002752/hashemia/logs/%x_%j.out
#SBATCH --error=/scratch/project_465002752/hashemia/logs/%x_%j.err

# =============================================================================
# USER CONFIGURATION — edit these groups to control the production run.
# Runs run_tracking.py (the general-purpose driver), not the FEMUSS benchmark.
# =============================================================================

# ── [1] Paths ────────────────────────────────────────────────────────────────
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif
JAXTRACE=/project/project_465002752/hashemia/JAXTrace_stable
PKGS=/project/project_465002752/hashemia/required-packages
# INPUT: point at the FEMUSS case folder. Either '<case>.gid' or '<case>.gid/post'.
# Mesh / particle file patterns auto-derive from the '<case>.gid' stem.
# To override completely, set MESH_DIR and/or FEMUSS_DIR below.
INPUT=/scratch/project_465001942/Cases-Edgar/new/cylA.gid
MESH_SUBDIR=0eule             # subfolder under post/ with mesh PVTU
FEMUSS_SUBDIR=1part           # subfolder under post/ with FEMUSS particles
CASE_STEM=""                  # override auto-detected case stem (empty = auto)
MESH_PATTERN=""               # override, e.g. "cylA_{timestep}.pvtu"  (empty = auto)
FEMUSS_PATTERN=""             # override, e.g. "cylA_pt_{timestep}.pvtu" (empty = auto)
# Direct path overrides — bypass INPUT/<case>.gid/post/<subdir> convention entirely.
# When set, MESH_SUBDIR / FEMUSS_SUBDIR are ignored for the corresponding path.
MESH_DIR=""                   # e.g. "/scratch/.../my_mesh_files"
FEMUSS_DIR=""                 # e.g. "/scratch/.../my_particle_files"
RUN_TAG=""                    # optional custom subfolder; empty = auto

# ── [2] Precision & velocity field ───────────────────────────────────────────
PRECISION=float32             # float32 | float64
VEL_START=159                 # velocity timestep cyclic start
VEL_END=159                   # velocity timestep cyclic end
VELOCITY_FIELD=Displacement
LEVELSET_FIELD=LEVEL

# ── [3] Simulation control ───────────────────────────────────────────────────
N_STEPS=2684
DT=0.0025
LOG_INTERVAL=10
EXPORT_FREQ=1                 # every N steps; set 0 with NO_EXPORT=1 to disable
NO_EXPORT=0                   # 1 = disable all VTU output (timing run)

# ── [4] Particle seeding ─────────────────────────────────────────────────────
# SEED_SOURCE: femuss | box | file
SEED_SOURCE=femuss
FEMUSS_START=0                # used when SEED_SOURCE=femuss
# Box seeding (used when SEED_SOURCE=box). Order: XMIN XMAX YMIN YMAX ZMIN ZMAX
SEED_BOX="-0.01 0.01 -0.01 0.01 0.0 0.002"
N_PARTICLES=100000            # used when SEED_SOURCE=box
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
BOUNDARY_WALLS="x_max=outlet,y_min=outlet"   # "" = all clamp
BOUNDARY_PROJ_TOL=1e-6
POINT_IN_TET_TOL=1e-6
LEVELSET_MODE=zero_vel        # zero_vel | skip_step
FAILED_SUBSTAGE=zero_vel      # zero_vel | last_valid_vel | skip_step
INTERPOLATION_METHOD=direct_inverse   # direct_inverse | gram_matrix

# ── [8] Pin velocity (FEMUSS FSW equivalent) ─────────────────────────────────
PIN_VELOCITY=1                # 1 = on, 0 = off
PIN_RPM=-600
PIN_CENTER="0.0 0.0 0.0"
PIN_AXIS="0.0 0.0 1.0"
PIN_TILT=0.0

# ── [9] VTU export options ───────────────────────────────────────────────────
N_GROUPS=5                    # 0 disables group export
EXPORT_ELEMENT_IDS=0          # 1 = include ElementID field

# ── [10] Performance / benchmark tuning ──────────────────────────────────────
# BENCHMARK_MODE=1: disables the background monitor and flips XLA to
# preallocating allocator (faster but reserves ~75% of HBM up-front).
# Use for pure timing runs; leave 0 for normal production.
BENCHMARK_MODE=0
# XLA_PREALLOC=1 forces XLA_PYTHON_CLIENT_PREALLOCATE=true regardless of mode.
# 0 keeps the on-demand allocator (safer; matches validated runs).
XLA_PREALLOC=0
# Background monitor interval (seconds). Set high (e.g. 300) or 0 to disable
# in production mode. BENCHMARK_MODE=1 forces monitor off regardless.
MONITOR_INTERVAL=30

# =============================================================================
# END USER CONFIGURATION — below this line is infrastructure.
# =============================================================================

# ── Derive case name (needed for default output folder) ─────────────────────
if [ -n "$CASE_STEM" ]; then
  _CASE="$CASE_STEM"
else
  _CASE=$(basename "$INPUT" .gid)
  _CASE=$(basename "$_CASE" /post)   # strip trailing /post if present
fi

# ── Output paths (flash for active IO, scratch for long-term) ───────────────
FLASH_OUT=/flash/project_465002752/hashemia/run_$SLURM_JOB_ID
# SCRATCH_BASE: parent directory on scratch where the results folder is created.
SCRATCH_BASE=/scratch/project_465002752/hashemia/outputs
# SCRATCH_FOLDER: name of the results folder inside SCRATCH_BASE.
# Default: <case>_jaxtrace_<jobid>. Override to use a custom name.
SCRATCH_FOLDER=""             # empty = auto: "${_CASE}_jaxtrace_${SLURM_JOB_ID}"
if [ -z "$SCRATCH_FOLDER" ]; then
  SCRATCH_FOLDER="${_CASE}_jaxtrace_${SLURM_JOB_ID}"
fi
SCRATCH_OUT="${SCRATCH_BASE}/${SCRATCH_FOLDER}"
MONITOR_LOG="${SCRATCH_BASE}/${SCRATCH_FOLDER}_monitor.log"

mkdir -p $FLASH_OUT $SCRATCH_OUT

# ── MIOpen cache to RAM ──────────────────────────────────────────────────────
export MIOPEN_USER_DB_PATH="/tmp/hashemia-miopen-$SLURM_JOB_ID"
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
# HSA SDMA: leaving disabled (validated on MI250X; enable via local edit to test).
export HSA_ENABLE_SDMA=0

# ── GPU & Memory Monitor (background) ───────────────────────────────────────
# BENCHMARK_MODE disables the monitor entirely. Otherwise the monitor logs
# rocm-smi + free at $MONITOR_INTERVAL seconds. We deliberately do NOT run
# find / du on the output dir every tick -- that becomes O(N) per tick with
# many VTU files and hurts timing on large runs.
MONITOR_PID=""
if [ "$BENCHMARK_MODE" != "1" ] && [ "$MONITOR_INTERVAL" -gt 0 ] 2>/dev/null; then
  (
    echo "=== GPU & Memory Monitor === Job $SLURM_JOB_ID === $(date) ==="
    echo "Interval: ${MONITOR_INTERVAL}s"
    echo ""
    while true; do
      echo "--- $(date '+%Y-%m-%d %H:%M:%S') ---"
      if command -v rocm-smi &>/dev/null; then
        rocm-smi --showuse --showmemuse --showtemp 2>/dev/null \
          | grep -E 'GPU|%|MiB|Temperature' \
          || rocm-smi 2>/dev/null | tail -n +3
      fi
      echo ""
      free -h | head -2
      echo ""
      sleep "$MONITOR_INTERVAL"
    done
  ) > $MONITOR_LOG 2>&1 &
  MONITOR_PID=$!
fi

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
[ "$EXPORT_ELEMENT_IDS" = "1" ] && ARGS+=( --export-element-ids )
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
  file)
    ARGS+=( --seed-file "$SEED_FILE" )
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
  python $JAXTRACE/run_tracking.py "${ARGS[@]}"

SIM_EXIT=$?

# ── Stop monitor ─────────────────────────────────────────────────────────────
if [ -n "$MONITOR_PID" ]; then
  kill $MONITOR_PID 2>/dev/null
  wait $MONITOR_PID 2>/dev/null
fi

echo ""
echo "Simulation exited with code $SIM_EXIT at $(date)"

# ── Move results to Scratch after job ───────────────────────────────────────
echo "Moving results from flash to scratch..."
mv $FLASH_OUT/* $SCRATCH_OUT/ 2>/dev/null
rmdir $FLASH_OUT 2>/dev/null

# Copy SLURM stdout/stderr and monitor log into the results folder
SLURM_OUT="/scratch/project_465002752/hashemia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
SLURM_ERR="/scratch/project_465002752/hashemia/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
mkdir -p "$SCRATCH_OUT/logs"
[ -f "$SLURM_OUT" ]   && cp "$SLURM_OUT" "$SCRATCH_OUT/logs/"
[ -f "$SLURM_ERR" ]   && cp "$SLURM_ERR" "$SCRATCH_OUT/logs/"
[ -f "$MONITOR_LOG" ] && mv "$MONITOR_LOG" "$SCRATCH_OUT/logs/"

echo "Done. All results, logs and monitoring in:"
echo "  $SCRATCH_OUT"
