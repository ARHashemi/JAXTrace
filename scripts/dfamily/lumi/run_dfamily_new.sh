#!/bin/bash
# SLURM directives. SLURM does not expand shell variables here; fill in your
# LUMI project ID and log directory before submitting. Override at submission
# time with:
#   sbatch --account=project_XXXXXXXXX --output=... --error=... run_lumi.sh
#SBATCH --job-name=dfam_new
#SBATCH --partition=small-g
#SBATCH --account=project_465002752
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=36:00:00   # D2 took 13h22 at 288k particles; D1 has deeper cells
#SBATCH --array=1-4       # 1=D1 2=D2 3=D3 4=D4; override with --array=1,3,4
#SBATCH --signal=B:SIGTERM@120   # SIGTERM batch shell 120s before time limit
                                  # so run_tracking.py can flush its VTKHDF
                                  # archive before SLURM SIGKILLs everything
#SBATCH --output=/scratch/project_465002752/hashemia/logs/%x_%A_%a.out
#SBATCH --error=/scratch/project_465002752/hashemia/logs/%x_%A_%a.err
# =============================================================================
# USER CONFIGURATION — edit these groups to control the production run.
# Runs run_tracking.py (the general-purpose driver), not the FEMUSS benchmark.
# =============================================================================

# ── [1] Paths ────────────────────────────────────────────────────────────────
# LUMI project ID. Defaults to the SLURM account used to submit the job
# (SLURM_JOB_ACCOUNT), which is set automatically when run via `sbatch`.
PROJECT="${SLURM_JOB_ACCOUNT:-project_465002752}"

# Singularity image with JAX + ROCm.
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif

# Per-user paths derived from $PROJECT and $USER. Override in a local config
# (scripts/run_lumi.local.sh, untracked) if your layout differs.
JAXTRACE="/projappl/${PROJECT}/hashemia/JAXTrace"   # dev repo: stable is releases only
PKGS="/projappl/${PROJECT}/hashemia/required-packages"

# INPUT: path to the case folder. Accepts either '<case>.gid' or
# '<case>.gid/post'. Ignored when AUTO_DETECT_CASE=1.
# Per-case table, selected by $SLURM_ARRAY_TASK_ID. Each case has its own mesh
# file stem (verified against the case folders: D1=A1_, D2=A2_, D3=B4_, D4=C2_),
# so the pattern CANNOT be hardcoded across the family.
CASES_ROOT="/scratch/project_465002752/lorenzgl/Cases/PinShapes/D-ConcavityTilt"
_IDX="${SLURM_ARRAY_TASK_ID:-2}"
case "$_IDX" in
  1) _CASE_NAME=D1; _MESH_STEM=A1; _VEL_END=199 ;;
  2) _CASE_NAME=D2; _MESH_STEM=A2; _VEL_END=199 ;;
  3) _CASE_NAME=D3; _MESH_STEM=B4; _VEL_END=199 ;;
  4) _CASE_NAME=D4; _MESH_STEM=C2; _VEL_END=199 ;;   # 0..201 exist; 199 keeps the family identical
  *) echo "ERROR: SLURM_ARRAY_TASK_ID='$_IDX' not in 1..4" >&2; exit 2 ;;
esac
INPUT="${CASES_ROOT}/${_CASE_NAME}.gid"
echo "[case] array task $_IDX -> $_CASE_NAME (mesh stem ${_MESH_STEM}_)"

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
MESH_PATTERN="${_MESH_STEM}_{timestep}.pvtu"     # per-case stem from the table above
FEMUSS_PATTERN=""             # e.g. "cylA_pt_{timestep}.pvtu"

# Absolute path overrides. When set, the corresponding *_SUBDIR is ignored
# and the given path is used directly.
MESH_DIR=""                   # directory containing the mesh PVTU files
FEMUSS_DIR=""                 # directory containing the FEMUSS particle files

# Optional tag appended to the auto-generated output folder name.
RUN_TAG="${_CASE_NAME}_new"

# OUTPUT_TARGET selects where JAXTrace results are written.
#   scratch -- /flash/${PROJECT}/${USER}/run_<JOB_ID> during the run, then
#              moved to /scratch/${PROJECT}/${USER}/outputs/<run_folder>.
#   case    -- <case>.gid/<OUTPUT_CASE_SUBFOLDER>, written in place.
# Reuse the built octree across runs of the same case. The octree build is
# ~90% of the build stage and the mesh is static for every PinShapes case, so
# a repeat run loads it in seconds. The key hashes the mesh arrays, every
# build option and the build modules' source, so any change rebuilds.
# OCTREE_CACHE: where the built search structure is cached. The build is ~90%
# of the build stage and these meshes are static, so a repeat run loads it in
# under a millisecond instead of rebuilding. The key hashes the mesh arrays,
# every build option and the build modules' source, so any change rebuilds
# automatically.
#
# Set OCTREE_CACHE_IN_CASE=1 to keep it beside the case instead of in scratch,
# as <case>.gid/octree_cache/. One entry is a single .h5 file, so it travels
# with the case and is simple to archive or delete. Resolved after the case
# directory is known, further down.
OCTREE_CACHE_IN_CASE=0
OCTREE_CACHE="/scratch/${PROJECT}/${USER}/octree_cache"

# OCTREE_CACHE_FORMAT: dir = a directory of .npy files (default, needs only
# numpy); hdf5 = one self-contained .h5 per entry, which is what makes keeping
# the cache beside a case tidy. Both are memory-mapped on read and cost the
# same to open (0.41ms vs 0.77ms for a 0.275 GiB D2 octree, against a 963s
# build), and entries of either format are read automatically, so switching
# never strands an existing cache.
#
# hdf5 WORKS on LUMI -- verified h5py 3.16.0 / HDF5 2.0.0 inside the container
# via PYTHONPATH=$PKGS, contiguous offset and byte-exact memmap -- but dir is
# the default because it depends on nothing. Asking for hdf5 where h5py is
# missing falls back to a directory rather than losing the build.
OCTREE_CACHE_FORMAT=dir

#   case-flash -- write to /flash during the run (fast IO, not the shared
#                 Lustre case tree), then MOVE into <case>.gid/<subfolder>
#                 once tracking has finished.
OUTPUT_TARGET=case-flash
OUTPUT_CASE_SUBFOLDER=post_pt_new   # used when OUTPUT_TARGET=case or case-flash

# ── [2] Precision & velocity field ───────────────────────────────────────────
PRECISION=float32             # float32 | float64
VEL_START=34                 # velocity timestep cyclic start
VEL_END="${_VEL_END:-199}"   # velocity timestep cyclic end (per-case table)
VELOCITY_FIELD=Displacement
LEVELSET_FIELD=LEVEL

# ── [3] Simulation control ───────────────────────────────────────────────────
N_STEPS=8000                 # total number of RK4 steps
DT=0.0003629                     # RK4 timestep size [s]
LOG_INTERVAL=100               # print + flush stats every N steps
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
SEED_SOURCE=grid
FEMUSS_START=0                # used when SEED_SOURCE=femuss
# Absolute box bounds (used by box / grid). Order: XMIN XMAX YMIN YMAX ZMIN ZMAX
SEED_BOX="-0.0199 -0.0111 -0.01499 0.01499 -0.00599 -0.00019"
# Per-axis fractions of the mesh bbox (used by box-frac / grid-frac).
# Order: XLO XHI YLO YHI ZLO ZHI, each in [0, 1] with lo < hi.
# Example: "0.0 0.2 0.0 1.0 0.0 1.0" = first 20% of X, full Y/Z.
SEED_FRACTION="0.0 0.2 0.0 1.0 0.0 1.0"
# Grid resolution (used by grid / grid-frac). Particle count = NX*NY*NZ.
SEED_GRID="60 120 40"
N_PARTICLES=100000            # used by box / box-frac (ignored by grid modes)
SEED_FILE=""                  # used when SEED_SOURCE=file
SEED=42

# ── [5] Optional FEMUSS comparison ───────────────────────────────────────────
# Enable only when SEED_SOURCE=femuss and reference file exists.
FEMUSS_COMPARE=0              # 1 = enable, 0 = disable

# ── [6] Search / RK4 kernel ──────────────────────────────────────────────────
RK4_MODE=fused                # fused | split
L1_METHOD=face                # face | node
L2_NEIGHBORHOOD=3             # 3 | 5
L0_SKIP_BAND=0.0              # 0 = mixed-sign only; e.g. 0.5e-3 for ±0.5mm
ENHANCED_SEARCH_BAND=0.0      # 0 = off; e.g. 1e-3 for ±1mm node-L1+5x5x5
REGISTRATION=""               # "" = default | vertex_multi | parent_cube

# ── [6b] Build-stage options (octree cache + registration) ──────────────────
# OCTREE_CACHE is set in section [1]. These control how it behaves and how the
# octree is built. All were added after the D-family investigation.

# NO_OCTREE_CACHE: 1 = ignore AND do not write the cache, forcing a full
# rebuild even when OCTREE_CACHE points at a populated directory. The cache
# key already hashes the mesh arrays, every build option and the build
# modules' source, so a genuine change invalidates itself; use this only to
# force a rebuild deliberately (e.g. to time the build).
NO_OCTREE_CACHE=0

# EXACT_TET_BOX: 1 = at build time, register each tetrahedron only in the
# cells it GENUINELY reaches, instead of every cell its bounding box touches.
# A tet fills at most ~1/6 of its AABB, so this lowers max cell occupancy and
# with it the static search bound. Measured on D2: occupancy 79 -> 56, 1.41x
# fewer tests per query, coverage preserved (registration remains complete).
# Costs an exact tet/box test per candidate cell during the build and nothing
# at run time. Left OFF here so this family matches the D2 reference run that
# produced the 1.06% loss figure; turn it on for new work.
EXACT_TET_BOX=0

# MAX_ELEMS_PER_CELL: static per-cell loop bound for the parent_cube L2
# search. 0 = derive it from the built octree's max occupancy, which is the
# correct behaviour on any mesh and what fixed the D family: the old hardcoded
# 32 silently truncated cells holding 79 (D2) and 115 (D1) elements, which was
# the dominant cause of the 2.93% loss. Set a value only to reproduce a
# historical result.
MAX_ELEMS_PER_CELL=0

# ── [7] Boundary / level-set behaviour ───────────────────────────────────────
# BOUNDARY_WALLS: per-wall behaviour as comma-separated 'wall=mode' pairs,
# where wall is one of {x_min, x_max, y_min, y_max, z_min, z_max} and mode
# is either:
#   clamp  -- particles crossing this wall are projected back inside the
#             bounding box (default for any wall not listed)
#   outlet -- particles crossing this wall leave the domain; their
#             element_id is set to -1 and tracking stops for them
# Set to "" to clamp every wall.
BOUNDARY_WALLS="x_max=clamp"
BOUNDARY_PROJ_TOL=1e-6        # inward offset applied when clamping to a wall [m]
POINT_IN_TET_TOL=1e-6         # numerical tolerance for point-in-tet test
LEVELSET_MODE=zero_vel        # how to handle a particle inside the tool region:
                              #   zero_vel  -- velocity at that step is set to 0
                              #   skip_step -- the RK4 step is skipped entirely
FAILED_SUBSTAGE=last_valid_vel      # policy when an RK4 substage falls outside the mesh:
                              #   zero_vel       -- treat the substage as v=0
                              #   last_valid_vel -- reuse the last interpolated v
                              #   skip_step      -- abandon the step, freeze particle
INTERPOLATION_METHOD=direct_inverse   # P1 velocity interpolation method:
                                      #   direct_inverse | gram_matrix

# ── [8] Pin velocity (FEMUSS FSW equivalent) ─────────────────────────────────
PIN_VELOCITY=0                # 1 = on, 0 = off
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
EXPORT_FORMAT=vtu

N_GROUPS=5                    # number of particle groups by initial X; 0 disables
EXPORT_ELEMENT_IDS=1          # 1 = include each particle's host ElementID
EXPORT_ESCAPED_FLAG=1         # 1 = include a per-particle 'Escaped' UInt8 field
                              # (set to 1 the first time element_id<0; useful
                              # for filtering out lost particles in ParaView)

# Temperature export — both flags share the same per-step P1 evaluation, so
# enabling both costs the same as enabling either one.
TRACK_MAX_TEMPERATURE=1       # 1 = export running maximum of TEMPERATURE_FIELD
                              # along each particle's trajectory as 'MaxTemperature'
EXPORT_TEMPERATURE=1          # 1 = export the instantaneous TEMPERATURE_FIELD at
                              # the current particle position as 'Temperature'
TEMPERATURE_FIELD=Temperature # PVTU field name to read for the above flags

# MESH_LOAD_WORKERS: threads for parsing velocity snapshots. Snapshot loading
# is CPU-bound XML parsing (VTK releases the GIL), measured 2.25x at 8 threads
# on the real multi-snapshot loader. Note that os.cpu_count() inside the
# container reports the NODE's 128 cores, not this job's cgroup share, so the
# library's "auto" would pick 8 whatever we request -- set it explicitly.
#
# Billing: small-g nodes are 128 CPU / 8 GPU, so 16 CPU is the proportional
# share for --gpus-per-node=1, and LUMI-G bills GPU-hours. Using 8 threads
# inside a 16-CPU request therefore costs no extra allocation. Loading is only
# ~14% of wall clock, so there is nothing to gain from going wider.
MESH_LOAD_WORKERS=8

# ── [10] Performance / monitoring ────────────────────────────────────────────
# BENCHMARK_MODE: 1 forces XLA_PREALLOC=1 and disables the background
# GPU/RAM monitor (eliminates monitor overhead for timing measurements).
BENCHMARK_MODE=0

# XLA_PREALLOC controls JAX's GPU allocator strategy.
#   1 -- preallocate ~75% of total HBM at startup (fixed pool)
#   0 -- on-demand allocator, grows as needed
# Preallocation is faster but blocks other processes from using that HBM.
XLA_PREALLOC=1

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
# Under SLURM, $0 points at a temporary copy of the batch script under
# /var/spool/slurmd/<jobdir>, not at the user-visible script path. SLURM
# exposes the original working directory as $SLURM_SUBMIT_DIR, which is
# the directory `sbatch` was invoked from -- that is what we want when
# the user does `cd <case>.gid && sbatch run_jaxtrace.sh`.
# Outside SLURM we fall back to $(dirname "$0").
if [ "${AUTO_DETECT_CASE:-0}" = "1" ]; then
  if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    INPUT="$SLURM_SUBMIT_DIR"
    echo "[case] AUTO_DETECT_CASE=1: using SLURM_SUBMIT_DIR='$INPUT'"
  else
    _SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
    INPUT="$_SCRIPT_DIR"
    echo "[case] AUTO_DETECT_CASE=1: INPUT='$INPUT'"
  fi
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

# ── Octree cache beside the case, if asked ─────────────────────────────────
# Done here because it needs _CASE_DIR, which is resolved just above.
if [ "${OCTREE_CACHE_IN_CASE:-0}" = "1" ]; then
  OCTREE_CACHE="${_CASE_DIR}/octree_cache"
  echo "[octree-cache] keeping the cache beside the case: $OCTREE_CACHE"
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
  case-flash)
    # Active IO on /flash, final results in the case folder. FLASH_OUT stays
    # distinct from SCRATCH_OUT, so the existing move step below relocates it.
    SUB="${OUTPUT_CASE_SUBFOLDER:-post_pt}"
    SCRATCH_OUT="${_CASE_DIR}/${SUB}"
    SCRATCH_BASE="$(dirname "$SCRATCH_OUT")"
    echo "[output] OUTPUT_TARGET=case-flash:"
    echo "           during run : $FLASH_OUT"
    echo "           final      : $SCRATCH_OUT"
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
    echo "ERROR: OUTPUT_TARGET='${OUTPUT_TARGET}' not in {scratch, case, case-flash}" >&2
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
[ -n "$OCTREE_CACHE" ]    && ARGS+=( --octree-cache "$OCTREE_CACHE" )
[ -n "$OCTREE_CACHE_FORMAT" ] && ARGS+=( --octree-cache-format "$OCTREE_CACHE_FORMAT" )
[ -n "$MESH_LOAD_WORKERS" ] && ARGS+=( --mesh-load-workers "$MESH_LOAD_WORKERS" )
[ "$NO_OCTREE_CACHE" = "1" ] && ARGS+=( --no-octree-cache )
[ "$EXACT_TET_BOX" = "1" ]   && ARGS+=( --exact-tet-box )
[ "${MAX_ELEMS_PER_CELL:-0}" != "0" ] && ARGS+=( --max-elems-per-cell "$MAX_ELEMS_PER_CELL" )
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
  --env PYTHONUNBUFFERED=1 \
  $SIF \
  python -u $JAXTRACE/run_tracking.py "${ARGS[@]}" &
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
  _n_src=$(find "$FLASH_OUT" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)
  echo "Moving $_n_src entries from '$FLASH_OUT' to '$SCRATCH_OUT'..."
  mkdir -p "$SCRATCH_OUT"
  # No 2>/dev/null here: a failed move must be visible in the log, because on
  # case-flash the destination is the shared case tree and /flash is purged.
  if mv "$FLASH_OUT"/* "$SCRATCH_OUT"/; then
    rmdir "$FLASH_OUT" 2>/dev/null
    _n_dst=$(find "$SCRATCH_OUT" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)
    echo "Move OK: $_n_dst entries now in '$SCRATCH_OUT'"
  else
    echo "ERROR: move failed; results are STILL ON FLASH at '$FLASH_OUT'" >&2
    echo "ERROR: retrieve them before /flash is purged." >&2
  fi
fi

# Copy SLURM stdout/stderr and monitor log into the results folder. The
# default --output / --error paths in the #SBATCH header are relative to
# SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from).
# The #SBATCH header writes to an absolute logs dir with %x_%A_%a, so build
# the same name here rather than guessing a path relative to the submit dir.
_LOGDIR="/scratch/${PROJECT}/${USER}/logs"
_LOGSTEM="${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
SLURM_OUT="${_LOGDIR}/${_LOGSTEM}.out"
SLURM_ERR="${_LOGDIR}/${_LOGSTEM}.err"
mkdir -p "$SCRATCH_OUT/logs"
[ -f "$SLURM_OUT" ]   && cp "$SLURM_OUT" "$SCRATCH_OUT/logs/"
[ -f "$SLURM_ERR" ]   && cp "$SLURM_ERR" "$SCRATCH_OUT/logs/"
[ -f "$MONITOR_LOG" ] && mv "$MONITOR_LOG" "$SCRATCH_OUT/logs/"

echo "Done. All results, logs and monitoring in:"
echo "  $SCRATCH_OUT"
