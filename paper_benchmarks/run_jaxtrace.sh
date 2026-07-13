#!/bin/bash
# =============================================================================
# run_jaxtrace.sh — Paper Section 7 L0/L1/L2 hit-rate diagnostic on cylA.
#
# This is a copy of scripts/run_workstation.sh tuned for the cylA case
# (mirroring /flash/users/ali/data/cylA.gid/run_jaxtrace.sh's overrides:
# FEMUSS seeding, mesh subdirs, patterns, LOG_INTERVAL=100, etc.), with
# the paper-diagnostic knob HIT_STATS_LOG=1 pre-enabled so a run of this
# script produces a hit_stats.csv alongside the usual outputs.
#
# The tracking kernel is unchanged from the production run.  The
# --hit-stats-log path in run_tracking.py only adds one extra pass of
# the L0/L1/L2 search closures at each --log-interval boundary, then
# writes the classification result to hit_stats.csv.  Adjacent to the
# tracking search itself (which fires every RK4 sub-step for every
# active particle), that added cost is negligible.
#
# Usage:
#   Foreground:            bash paper_benchmarks/run_jaxtrace.sh
#   Background (nohup):    nohup bash paper_benchmarks/run_jaxtrace.sh > run.log 2>&1 &
#   Task-spooler queue:    TS_SOCKET=/tmp/gpu_queue ts bash paper_benchmarks/run_jaxtrace.sh
#   Inside screen/tmux:    screen -S jaxtrace; bash paper_benchmarks/run_jaxtrace.sh
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
INPUT="/flash/users/ali/data/cylA.gid"

# AUTO_DETECT_CASE: when 1, INPUT is replaced by the directory containing
# this script at runtime. Use this when a copy of the script is placed
# inside each case folder (e.g. .../A1.gid/run_jaxtrace.sh).
AUTO_DETECT_CASE=0

# Subfolders inside <case>.gid/post/ that contain the mesh PVTU files and
# the FEMUSS particle PVTU files. Set to "" when the files sit directly in
# <case>.gid/post/ without an inner subfolder.
MESH_SUBDIR="0eule"
FEMUSS_SUBDIR="1part"

# Auto-derivation overrides — leave blank to let the script infer them
# from the case folder name (e.g. cylA.gid -> stem 'cylA').
CASE_STEM=""                 # case-stem string used in file patterns
MESH_PATTERN="cylA_{timestep}.pvtu"
FEMUSS_PATTERN="cylA_pt_{timestep}.pvtu"

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
OUTPUT_TARGET=case
OUTPUT_CASE_SUBFOLDER=post_pt_hit_stats    # under <case>.gid/

# ENABLE_UNION: when 1, run `bash run_union.sh` (sitting next to this
# script) after the tracking python finishes and outputs have been
# rsync'd to SCRATCH_OUT. run_union.sh is expected to read its input
# from SCRATCH_OUT (typically <case>.gid/post_pt/run_*/particles.vtkhdf)
# and write its outputs to <particles_dir>/union/. The hook is a single
# inline bash invocation — no separate process, no SLURM dependency.
# Auto-stamped by generate_jaxtrace_scripts.sh --enable-union.
ENABLE_UNION=0

# ── [2] Precision & velocity field ───────────────────────────────────────────
PRECISION=float32          # float32 | float64

# VELOCITY_SOURCE picks the velocity-field backend:
#   mesh      — load nodal velocity from --input PVTU files; per-RK4-stage
#               interpolate via L0/L1/L2 octree search. Default for every
#               cohort case today.
#   analytic  — call a user-supplied JAX function on every sub-stage. No
#               mesh load, no search, no interpolation. Particles are
#               seeded inside DOMAIN_BBOX (or the velocity module's own
#               default). Inlet drift / FEMUSS / level-set / pin-velocity
#               are mesh-only and are silently ignored on the analytic
#               path; embed any upstream behaviour in velocity_fn itself.
VELOCITY_SOURCE=mesh

# VELOCITY_MODULE: path to the user .py exposing
#     build_provider(domain_bbox, dt, t_start=0.0) -> AnalyticVelocityProvider
# Required when VELOCITY_SOURCE=analytic. Reference fields ship under
# $JAXTRACE/jaxtrace/analytic_fields/:
#   uniform.py                       — v(x) = (V_ref, 0, 0). Sanity baseline.
#   divergence_free_recirculation.py — streamfunction-derived field from
#                                       FSW Internal Summary §A (steady,
#                                       divergence-free, recirculation).
VELOCITY_MODULE=""

# DOMAIN_BBOX: six floats "XMIN XMAX YMIN YMAX ZMIN ZMAX" for the
# analytic path. When empty, the velocity module's own default bbox is
# used. Mesh path ignores this.
DOMAIN_BBOX=""

# ── [2b] Mesh-path velocity field (ignored on analytic path) ─────────────────
VEL_START=159
VEL_END=159
VELOCITY_FIELD=Displacement
LEVELSET_FIELD=LEVEL

# ── [3] Simulation control ───────────────────────────────────────────────────
N_STEPS=2684               # total number of RK4 steps
DT=0.0025                  # RK4 timestep size [s]
LOG_INTERVAL=100            # print + flush stats every N steps
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
SEED_FRACTION="-0.2 0.04 0.01 0.99 0.001 0.999"
# Grid resolution (used by grid / grid-frac). Particle count = NX*NY*NZ.
SEED_GRID="60 120 50"
N_PARTICLES=300000          # used by box / box-frac (ignored by grid modes)
SEED_FILE=""
SEED=42

# ── [5] Optional FEMUSS comparison ───────────────────────────────────────────
FEMUSS_COMPARE=0           # Sec.7 diagnostic run does not need comparison

# ── [6] Search / RK4 kernel ──────────────────────────────────────────────────
RK4_MODE=fused             # fused | split
L1_METHOD=face             # face | node
L2_NEIGHBORHOOD=3          # 3 | 5
L0_SKIP_BAND=0.0
ENHANCED_SEARCH_BAND=0.0
REGISTRATION=""

# ORPHAN_FALLBACK: how to handle non-Kuhn tetrahedra that have no
# Kuhn face/node neighbour. By default (=1) they get a private octree
# cell built from the global median Kuhn cell_size/level so the
# spatial search can still find them. Set to 0 to drop them from the
# octree (legacy behaviour) — particles landing inside these tets
# will be reported as lost.
ORPHAN_FALLBACK=1

# HYBRID_NON_KUHN: registration strategy for non-Kuhn elements.
# 1 (default) — non-Kuhn tets are registered in every cell their AABB
#               overlaps (typically 1-4 cells). This closes coverage
#               holes on meshes with a high non-Kuhn fraction (the
#               classic symptom: particles geometrically inside the
#               domain flagged as Escaped at element faces).
# 0           — non-Kuhn tets registered by centroid only (1 cell;
#               legacy behaviour). Use for benchmarking or when the
#               mesh is mostly Kuhn anyway. Kuhn elements are
#               unaffected by this flag — they always use the cheap
#               single-cell parent-cube registration.
HYBRID_NON_KUHN=1

# ── [7] Boundary / level-set behaviour ───────────────────────────────────────
# BOUNDARY_WALLS: per-wall behaviour as comma-separated 'wall=mode' pairs,
# where wall is one of {x_min, x_max, y_min, y_max, z_min, z_max} and mode
# is one of:
#   clamp     -- particles crossing this wall are projected back inside
#                the bounding box (default for any wall not listed)
#   outlet    -- particles crossing this wall leave the domain; their
#                element_id stays -1 and tracking stops for them
#   ballistic -- particles crossing this wall continue with their last
#                in-domain velocity; element_id stays -1, Escaped=1
#   freeze    -- particles crossing this wall stop at the escape point;
#                element_id stays -1, Escaped=1
# Modes are independent per wall and do not interfere with each other.
# Performance: with no wall set to ballistic or freeze, per-step cost is
# identical to today. Set to "" to clamp every wall.
BOUNDARY_WALLS=""
BOUNDARY_PROJ_TOL=1e-6     # inward offset applied when clamping to a wall [m]
POINT_IN_TET_TOL=1e-6      # numerical tolerance for point-in-tet test

# INLET_WALL allows the seed box to extend outside the mesh on one named
# wall. Particles seeded outside the mesh on that face are kept alive
# with element_id=-1 and drift at INLET_VELOCITY along the inward normal
# until they cross the wall and the kernel's spatial search assigns a
# real host element. Other faces of the seed box are cropped to the mesh
# bounding box; for SEED_SOURCE=grid the grid spacing is preserved and
# the final particle count is reduced accordingly.
# Set INLET_WALL="" to disable (no inlet, no cropping warnings).
INLET_WALL=""              # "" | x_min | x_max | y_min | y_max | z_min | z_max
INLET_VELOCITY=0.0         # signed scalar [m/s]; positive = into the mesh
LEVELSET_ENABLE=1          # 1 = read LEVEL field, mask velocity inside tool;
                           # 0 = ignore the LEVEL field entirely, use raw mesh
                           # velocities everywhere (no tool masking, no
                           # boundary-element L0 skip).
LEVELSET_MODE=zero_vel     # how to handle a particle inside the tool region
                           # when LEVELSET_ENABLE=1:
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

# ── [8b] Paper Sec.7 diagnostic ──────────────────────────────────────────────
# HIT_STATS_LOG: when 1, run_tracking.py appends a row to hit_stats.csv
# every LOG_INTERVAL steps classifying all currently-alive particles by
# which search level (L0/L1/L2) would find them if a cold lookup were
# issued right now.  Adds one extra L0+L1+L2 batch call per log tick
# (negligible against RK4).  This copy of run_workstation.sh has it
# enabled by default because that is the whole point of this launcher.
# Set to 0 to run the same cylA config without the diagnostic overhead.
HIT_STATS_LOG=1

# ── [9] Particle export options ───────────────────────────────────────────────
# EXPORT_FORMAT: container format for the per-step particle output.
#   vtkhdf -- single .vtkhdf archive containing all timesteps.
#             Requires ParaView >= 6.0 / VTK >= 9.4 to read.
#   vtu    -- one .vtu file per step, loaded in ParaView as a numbered
#             series. Choose this for older ParaView installations.
EXPORT_FORMAT=vtkhdf

N_GROUPS=5                 # number of particle groups by initial X; 0 disables
EXPORT_ELEMENT_IDS=0       # 1 = include each particle's host ElementID
EXPORT_ESCAPED_FLAG=1      # 1 = include a per-particle 'Escaped' UInt8 field
                           # (set to 1 the first time element_id<0; useful
                           # for filtering out lost particles in ParaView)

# Temperature export — both flags share the same per-step P1 evaluation, so
# enabling both costs the same as enabling either one.
TRACK_MAX_TEMPERATURE=1    # 1 = export running maximum of TEMPERATURE_FIELD
                           # along each particle's trajectory as 'MaxTemperature'
EXPORT_TEMPERATURE=1       # 1 = export the instantaneous TEMPERATURE_FIELD at
                           # the current particle position as 'Temperature'
TEMPERATURE_FIELD=Temperature  # PVTU field name to read for the above flags

# ── [10] JAX memory & performance ────────────────────────────────────────────
# XLA_PREALLOC controls JAX's GPU allocator strategy.
#   1 -- preallocate VRAM_FRACTION of total VRAM at startup (fixed pool)
#   0 -- on-demand allocator, grows up to VRAM_FRACTION as needed
# Preallocation is faster but blocks other processes from using that VRAM.
XLA_PREALLOC=0

# VRAM_FRACTION: fraction of total GPU VRAM available to this job.
#   With XLA_PREALLOC=0 it acts as a soft cap.
#   With XLA_PREALLOC=1 it determines the reserved pool size.
# Typical values on a 32 GB GPU:
#   0.90  sole GPU user
#   0.45  two parallel jobs sharing the GPU
#   0.30  three parallel jobs
VRAM_FRACTION=0.95

# BENCHMARK_MODE: 1 forces XLA_PREALLOC=1 and disables the background
# GPU/RAM monitor (eliminates monitor overhead for timing measurements).
BENCHMARK_MODE=0

# MONITOR_INTERVAL: seconds between GPU/RAM log entries. 0 disables the
# background monitor.
MONITOR_INTERVAL=100

# ── [11] Online density estimation (opt-in) ──────────────────────────────────
# When DENSITY_ENABLE=0 the runner is never constructed and adds zero overhead
# to the tracking loop. When ON, a density field is computed on a uniform
# voxel grid at the same cadence as EXPORT_FREQ (override with
# DENSITY_EXPORT_FREQ) and written as VTKHDF ImageData or per-step VTI.
# A time-averaged field with mean / coverage / peak fields is written at exit.
DENSITY_ENABLE=0                  # 1 = enable, 0 = disable (default)

# Output (default: <run output>/density)
DENSITY_OUTPUT_DIR=""             # absolute path; "" = <output>/density
DENSITY_OUTPUT_FORMAT=vtkhdf      # vtkhdf | vti
DENSITY_FILENAME_STEM=density

# Kernel + bandwidth
DENSITY_KERNEL=wendland_c2        # wendland_c2|wendland_c4|cubic_spline|gaussian|epanechnikov|quintic_spline
DENSITY_BANDWIDTH_MODE=fixed      # fixed | scott | silverman | knn_adaptive
DENSITY_BANDWIDTH=""              # explicit h (fixed mode); "" = factor * voxel_size
DENSITY_BANDWIDTH_FACTOR=2.0
DENSITY_BANDWIDTH_REFRESH_EVERY=0 # 0 = compute once; N = recompute every N steps
DENSITY_KNN_K=32                  # k-NN neighbors for knn_adaptive
DENSITY_KNN_SAFETY=1.2

# Voxel grid
DENSITY_BOUNDS=""                 # "XMIN XMAX YMIN YMAX ZMIN ZMAX"; "" = use mesh bbox
DENSITY_BOUNDS_FROM=mesh          # mesh | particles  (ignored if DENSITY_BOUNDS set)
DENSITY_RESOLUTION=128            # cubic grid resolution; ignored if voxel-size set
DENSITY_VOXEL_SIZE=""             # physical voxel edge length [m]; overrides resolution
DENSITY_PAD_FRACTION=0.0
DENSITY_NO_MASK_INSIDE_MESH=0     # 1 = skip inside-mesh masking

# Engine
DENSITY_ENGINE=auto               # auto | brute | octree
DENSITY_AUTO_THRESHOLD=5e10
DENSITY_BRUTE_QUERY_CHUNK=8192
DENSITY_OCTREE_TARGET_N_PER_CELL=9
DENSITY_PARTICLE_BUCKET=4096

# Output toggles
DENSITY_NO_PER_STEP=0             # 1 = no per-step grid file (still computes for time-avg)
DENSITY_NO_TIME_AVERAGE=0         # 1 = skip the final time-average file
DENSITY_NO_PARTICLE_DENSITY=0     # 1 = skip 'Density' scalar in particles export
DENSITY_EXPORT_FREQ=""            # "" = same as EXPORT_FREQ
DENSITY_NORMALIZATION=pdf         # pdf | mass | unnormalized

# Compression: gzip | lzf | blosc | none
DENSITY_COMPRESSION=gzip          # gzip | lzf | blosc | none. ParaView only
                                  # reads gzip; lzf/blosc need a custom HDF5.
DENSITY_COMPRESSION_OPTS=1
DENSITY_BLOSC_THREADS=4

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
# When AUTO_DETECT_CASE=1, INPUT is replaced by the directory containing
# this script at runtime. Useful when a copy of the script is placed inside
# each case folder, e.g. /scratch/.../A1.gid/run_jaxtrace.sh. When invoked
# under SLURM, the script lives in a private staging dir and SLURM_SUBMIT_DIR
# is the user-visible launch directory; prefer it when set.
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
# /flash is local NVMe; /scratch and shared case folders sit on slow HDD.
# Hot-path JAXTrace IO (particle export, monitor log, run log) ALWAYS
# stages on /flash; we only differ in where results land at the end:
#   OUTPUT_TARGET=scratch  (default): move to /scratch/users/$USER/outputs/<folder>.
#   OUTPUT_TARGET=case:               move to <case>.gid/<OUTPUT_CASE_SUBFOLDER>
#                                     so mesh + results sit together.
# Earlier versions of this script collapsed FLASH_OUT onto SCRATCH_OUT
# in case-mode, which sent every per-step VTKHDF write straight to the
# shared HDD and tanked throughput. We never want that — the flash→
# target move at the end is cheap (one big sequential copy).
FLASH_OUT=/flash/users/${USER}/run_${RUN_ID}
LOG_DIR=""
case "${OUTPUT_TARGET:-scratch}" in
    case)
        SUB="${OUTPUT_CASE_SUBFOLDER:-post_pt}"
        SCRATCH_OUT="${_CASE_DIR}/${SUB}"
        SCRATCH_BASE="$(dirname "$SCRATCH_OUT")"
        LOG_DIR="${SCRATCH_OUT}/logs"
        echo "[output] OUTPUT_TARGET=case: staging on '$FLASH_OUT', final '$SCRATCH_OUT'"
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

# Create the output directories early. We capture stdout+stderr so the
# dry-run probe below (and the user later) can see whether anything
# failed silently — e.g. permission denied on a shared mount, which is
# the most common cause of OUTPUT_TARGET=case crashing later in the
# pipeline.
_MKDIR_OUT="$(mkdir -p "$FLASH_OUT" "$SCRATCH_OUT" "$LOG_DIR" 2>&1)"
_MKDIR_RC=$?

# ── Dry-run probe (no Python, no GPU; just verify the path plumbing) ─────────
# Enable with: JAXTRACE_DRY_RUN=1 bash run_jaxtrace.sh
# Prints every derived path and tests writability on FLASH_OUT,
# SCRATCH_OUT, and LOG_DIR. Exits with code 0 if all writes succeed,
# code 3 otherwise. Use this before submitting a long run to confirm
# OUTPUT_TARGET=case / OUTPUT_TARGET=scratch are wired correctly on
# this host.
if [ "${JAXTRACE_DRY_RUN:-0}" = "1" ]; then
    echo
    echo "================ DRY RUN ================"
    echo " Case stem:        $_CASE"
    echo " Case dir:         $_CASE_DIR"
    echo " INPUT:            $INPUT"
    echo " OUTPUT_TARGET:    ${OUTPUT_TARGET:-scratch}"
    echo " OUTPUT subfolder: ${OUTPUT_CASE_SUBFOLDER:-post_pt}"
    echo " FLASH_OUT:        $FLASH_OUT"
    echo " SCRATCH_BASE:     $SCRATCH_BASE"
    echo " SCRATCH_OUT:      $SCRATCH_OUT"
    echo " LOG_DIR:          $LOG_DIR"
    echo " MONITOR_LOG:      $MONITOR_LOG"
    echo " RUN_LOG:          $RUN_LOG"
    echo
    echo " mkdir -p rc:      $_MKDIR_RC"
    [ -n "$_MKDIR_OUT" ] && echo " mkdir -p stderr:  $_MKDIR_OUT"
    echo
    # Quick writability check on each directory. Anything we cannot
    # touch here will fail the real run too — usually because the case
    # folder lives on a read-only share or the user lacks group write.
    _probe_writable() {
        local d="$1"
        local label="$2"
        if [ ! -d "$d" ]; then
            echo " [FAIL] $label does not exist: $d"
            return 1
        fi
        local probe="$d/.jaxtrace_dryrun_probe_$$"
        if : > "$probe" 2>/dev/null; then
            rm -f "$probe"
            echo " [ OK ] $label is writable: $d"
            return 0
        else
            echo " [FAIL] $label is NOT writable: $d"
            return 1
        fi
    }
    _probe_writable "$FLASH_OUT"   "FLASH_OUT"
    _rc_a=$?
    _probe_writable "$SCRATCH_OUT" "SCRATCH_OUT"
    _rc_b=$?
    _probe_writable "$LOG_DIR"     "LOG_DIR"
    _rc_c=$?
    echo "=========================================="
    [ $((_rc_a + _rc_b + _rc_c)) = 0 ] && exit 0 || exit 3
fi
# Non-dry-run path: if mkdir actually failed, fail loudly now instead
# of letting Python crash with an opaque file-not-found error later.
if [ "$_MKDIR_RC" != "0" ]; then
    echo "ERROR: mkdir -p failed (rc=$_MKDIR_RC): $_MKDIR_OUT" >&2
    exit 4
fi

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

# ── GPU & Memory Monitor (nvidia-smi) ────────────────────────────────────────
# The monitor runs in its own process group via `setsid` so the cleanup
# `kill -- -$MONITOR_PGID` reaches both the bash subshell AND its current
# `sleep` child (a plain `kill` to the bash PID would wait for the sleep
# to finish, leaving the monitor alive for up to $MONITOR_INTERVAL seconds
# after the run completes).
MONITOR_PID=""
MONITOR_PGID=""
if [ "$BENCHMARK_MODE" != "1" ] && [ "${MONITOR_INTERVAL}" -gt 0 ] 2>/dev/null; then
setsid bash -c '
    echo "=== GPU & Memory Monitor === Run '"${RUN_ID}"' === $(date) ==="
    echo "Interval: '"${MONITOR_INTERVAL}"'s"
    echo ""
    while true; do
        echo "--- $(date '\''+%Y-%m-%d %H:%M:%S'\'') ---"
        nvidia-smi \
            --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
            --format=csv,noheader,nounits \
        | awk -F", " '\''{printf "  GPU  Temp:%s°C  Util:%s%%  VRAM:%s/%s MiB  Power:%sW\n",$1,$2,$3,$4,$5}'\''
        echo ""
        free -h | head -2
        echo ""
        sleep '"${MONITOR_INTERVAL}"'
    done
' > "$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!
MONITOR_PGID=$MONITOR_PID   # setsid makes the new process its own pgleader
fi

# Ensure the monitor is reaped no matter how this script ends — normal exit,
# user Ctrl-C, killed terminal, or unhandled error. trap EXIT is bash's
# unconditional cleanup hook.
_cleanup_monitor() {
    if [ -n "${MONITOR_PGID:-}" ]; then
        kill -- -"$MONITOR_PGID" 2>/dev/null || true
        # Reap the process so it doesn't linger as a zombie.
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
}
trap _cleanup_monitor EXIT INT TERM

# ── Build CLI argument list ───────────────────────────────────────────────────
ARGS=(
    --input              "$INPUT"
    --output             "$FLASH_OUT"
    --mesh-subdir        "$MESH_SUBDIR"
    --femuss-subdir      "$FEMUSS_SUBDIR"
    --precision          "$PRECISION"
    --velocity-source    "$VELOCITY_SOURCE"
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
if [ -n "${INLET_WALL:-}" ]; then
    ARGS+=( --inlet-wall "$INLET_WALL" --inlet-velocity "$INLET_VELOCITY" )
fi
[ -n "$REGISTRATION"     ] && ARGS+=( --registration        "$REGISTRATION"     )
# Analytic-velocity knobs. Only --velocity-source is always passed (its
# default is "mesh"); the others are only added when set, so they don't
# clutter the mesh-path CLI.
[ -n "$VELOCITY_MODULE"  ] && ARGS+=( --velocity-module     "$VELOCITY_MODULE"  )
[ -n "$DOMAIN_BBOX"      ] && ARGS+=( --domain-bbox         $DOMAIN_BBOX        )
[ "$ORPHAN_FALLBACK"    = "0" ] && ARGS+=( --no-orphan-fallback  )
[ "$HYBRID_NON_KUHN"    = "0" ] && ARGS+=( --no-hybrid-non-kuhn  )
[ "$LEVELSET_ENABLE"    = "0" ] && ARGS+=( --no-levelset         )
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
[ "$HIT_STATS_LOG"      = "1" ] && ARGS+=( --hit-stats-log        )

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

# Density estimation flags (only appended when DENSITY_ENABLE=1).
if [ "${DENSITY_ENABLE:-0}" = "1" ]; then
    ARGS+=( --density-enable
            --density-output-format    "$DENSITY_OUTPUT_FORMAT"
            --density-filename-stem    "$DENSITY_FILENAME_STEM"
            --density-kernel           "$DENSITY_KERNEL"
            --density-bandwidth-mode   "$DENSITY_BANDWIDTH_MODE"
            --density-bandwidth-factor "$DENSITY_BANDWIDTH_FACTOR"
            --density-bandwidth-refresh-every "$DENSITY_BANDWIDTH_REFRESH_EVERY"
            --density-knn-k            "$DENSITY_KNN_K"
            --density-knn-safety       "$DENSITY_KNN_SAFETY"
            --density-bounds-from      "$DENSITY_BOUNDS_FROM"
            --density-resolution       "$DENSITY_RESOLUTION"
            --density-pad-fraction     "$DENSITY_PAD_FRACTION"
            --density-engine           "$DENSITY_ENGINE"
            --density-auto-threshold   "$DENSITY_AUTO_THRESHOLD"
            --density-brute-query-chunk     "$DENSITY_BRUTE_QUERY_CHUNK"
            --density-octree-target-n-per-cell "$DENSITY_OCTREE_TARGET_N_PER_CELL"
            --density-particle-bucket  "$DENSITY_PARTICLE_BUCKET"
            --density-normalization    "$DENSITY_NORMALIZATION"
            --density-compression      "$DENSITY_COMPRESSION"
            --density-compression-opts "$DENSITY_COMPRESSION_OPTS"
            --density-blosc-threads    "$DENSITY_BLOSC_THREADS"
    )
    [ -n "$DENSITY_OUTPUT_DIR" ]  && ARGS+=( --density-output-dir  "$DENSITY_OUTPUT_DIR" )
    [ -n "$DENSITY_BANDWIDTH" ]   && ARGS+=( --density-bandwidth   "$DENSITY_BANDWIDTH" )
    [ -n "$DENSITY_BOUNDS" ]      && ARGS+=( --density-bounds      $DENSITY_BOUNDS )
    [ -n "$DENSITY_VOXEL_SIZE" ]  && ARGS+=( --density-voxel-size  "$DENSITY_VOXEL_SIZE" )
    [ -n "$DENSITY_EXPORT_FREQ" ] && ARGS+=( --density-export-freq "$DENSITY_EXPORT_FREQ" )
    [ "$DENSITY_NO_MASK_INSIDE_MESH" = "1" ] && ARGS+=( --density-no-mask-inside-mesh )
    [ "$DENSITY_NO_PER_STEP"         = "1" ] && ARGS+=( --density-no-per-step )
    [ "$DENSITY_NO_TIME_AVERAGE"     = "1" ] && ARGS+=( --density-no-time-average )
    [ "$DENSITY_NO_PARTICLE_DENSITY" = "1" ] && ARGS+=( --density-no-particle-density )
fi

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
# Handled unconditionally by the EXIT trap installed alongside the monitor.
# Calling _cleanup_monitor explicitly here triggers it before the move/log
# steps so the monitor log captures only the active run, not the cleanup.
_cleanup_monitor

echo ""
echo "Simulation exited with code $SIM_EXIT at $(date)"

# ── Move results from flash → final destination ──────────────────────────────
# FLASH_OUT is always on /flash (NVMe staging); SCRATCH_OUT is the final
# destination (/scratch by default, or inside the case folder when
# OUTPUT_TARGET=case). We need a TRUE merge here — earlier runs may have
# left an empty (or stale) subdir at SCRATCH_OUT/run_<...>/, and plain
# `mv FLASH/*` refuses to merge directories. Use rsync when available
# (atomic, handles partial trees), fall back to `cp -a` + `rm`.
#
# We also DO NOT silence stderr any more: the previous version did
# `mv ... 2>/dev/null` and a refused merge was reported as success.
# Real outputs ended up stranded on /flash while empty placeholder
# directories sat in the case folder. Loud failures are better.
if [ "$FLASH_OUT" != "$SCRATCH_OUT" ]; then
    echo "Moving results from $FLASH_OUT to $SCRATCH_OUT..."
    _XFER_RC=0
    if command -v rsync >/dev/null 2>&1; then
        # -a: preserve timestamps/perms; --remove-source-files clears
        # /flash incrementally so a partial xfer is still useful.
        rsync -a --remove-source-files "$FLASH_OUT"/ "$SCRATCH_OUT"/
        _XFER_RC=$?
    else
        # Portable fallback: copy-then-delete. cp -a preserves
        # timestamps/perms; the trailing /. merges the contents.
        cp -a "$FLASH_OUT"/. "$SCRATCH_OUT"/
        _XFER_RC=$?
        [ "$_XFER_RC" = 0 ] && rm -rf "$FLASH_OUT"/*
    fi
    if [ "$_XFER_RC" != "0" ]; then
        echo "WARNING: transfer from $FLASH_OUT to $SCRATCH_OUT" \
             "failed with rc=$_XFER_RC. Results remain on /flash;" \
             "you can manually rsync them to the destination." >&2
    else
        # Prune any now-empty directories left behind on /flash.
        find "$FLASH_OUT" -depth -type d -empty -delete 2>/dev/null
    fi
fi
if [ -f "$MONITOR_LOG" ]; then
    mkdir -p "$SCRATCH_OUT/logs"
    mv -f "$MONITOR_LOG" "$SCRATCH_OUT/logs/" || \
        echo "WARNING: failed to move $MONITOR_LOG to $SCRATCH_OUT/logs/" >&2
fi

# ── Optional union postprocess ───────────────────────────────────────────────
# When ENABLE_UNION=1 and run_union.sh exists next to this script, run
# it after the tracking results have landed in SCRATCH_OUT. We only fire
# the hook if the tracking python itself exited cleanly — there's no
# point unioning a half-written trajectory.
if [ "${ENABLE_UNION:-0}" = "1" ] && [ "$SIM_EXIT" = "0" ]; then
    _UNION_SH="$(dirname "$0")/run_union.sh"
    if [ -x "$_UNION_SH" ] || [ -f "$_UNION_SH" ]; then
        echo ""
        echo "================ UNION POSTPROCESS ================"
        echo " Running $_UNION_SH"
        echo "==================================================="
        bash "$_UNION_SH"
        _UNION_RC=$?
        echo "Union postprocess exited with code $_UNION_RC"
    else
        echo "WARNING: ENABLE_UNION=1 but $_UNION_SH not found; skipping union." >&2
    fi
fi

# Cleanup CUDA cache
rm -rf "$CUDA_CACHE_DIR"

echo ""
echo "======================================================"
echo " Done. Exit code: $SIM_EXIT"
echo " Results: $SCRATCH_OUT"
echo " Logs:    $LOG_DIR"
echo "======================================================"

exit $SIM_EXIT
