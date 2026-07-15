#!/bin/bash
# =============================================================================
# run_jaxtrace_rom.sh  —  Particle tracking on the ROM-reconstructed mesh
# velocity field for one or more cases in the FOM cohort.
#
# Runs the same JAXTrace tracking pipeline as paper_benchmarks/run_jaxtrace.sh
# but with three changes:
#   (a) --input points at the ROM-reconstructed case folder produced by
#       paper_benchmarks/reconstruct_rom_velocities.sh (i.e. a folder
#       whose <case>.gid/post/ contains a single ROM-derived PVTU
#       identical in mesh to the original FOM PVTU but with the
#       Displacement field replaced by the ROM reconstruction).
#   (b) VEL_START = VEL_END = 0 (the reconstruction produced one
#       snapshot at output-timestep 0).  The tracking loop reuses that
#       single snapshot for every RK4 step.
#   (c) OUTPUT_CASE_SUBFOLDER lands under the ORIGINAL FOM case folder
#       as post_pt_rom_<formula>/ so a downstream comparison script
#       can find both post_pt/ (full-order tracking) and post_pt_rom/
#       (ROM-reconstructed tracking) side by side.
#
# All other tracking parameters mirror run_jaxtrace.sh exactly (same
# dt, same N_STEPS, same seeding source, same pin velocity, same
# level-set, same search tuning).  Only the velocity field differs.
#
# Usage:
#   bash paper_benchmarks/run_jaxtrace_rom.sh
#
# Override via env vars:
#   CASES="4 1"           bash paper_benchmarks/run_jaxtrace_rom.sh
#   ROM_FORMULA=c_over_sig bash paper_benchmarks/run_jaxtrace_rom.sh
#
# All env-var overrides:
#   FOM_ROOT       parent of <case>.gid FOM folders   (default: /scratch/shared/ROM/FOM)
#   ROM_ROOT       parent of <case>.gid ROM folders   (default: /flash/users/${USER}/data/ROM_recon/${ROM_FORMULA})
#   CASE_PREFIX    case-folder prefix                  (default: cylindrical)
#   CASES          case indices, space-separated       (default: "4 1")
#   ROM_FORMULA    formula label (for output folder)   (default: centered)
#   N_STEPS        RK4 timesteps                        (default: 2684)
#   DT             timestep size                        (default: 0.0025)
#   LOG_INTERVAL   log every N steps                    (default: 100)
#   PYTHON         python interpreter                   (default: python)
#   Everything else (seeding, PIN_RPM, etc.) is COPIED verbatim from
#   /flash/users/ali/data/cylA.gid/run_jaxtrace.sh values.  Edit this
#   file if any per-case run has a different physics config.
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAXTRACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── [1] Paths ────────────────────────────────────────────────────────────────
VENV=/flash/shared/jax/.venv
JAXTRACE="$JAXTRACE_ROOT"

FOM_ROOT="${FOM_ROOT:-/scratch/shared/ROM/FOM}"
ROM_FORMULA="${ROM_FORMULA:-centered}"
ROM_ROOT="${ROM_ROOT:-/flash/users/${USER:-ali}/data/ROM_recon/${ROM_FORMULA}}"
CASE_PREFIX="${CASE_PREFIX:-cylindrical}"
CASES="${CASES:-4 1}"

# ── [2] Time / precision ─────────────────────────────────────────────────────
PRECISION=float32
VEL_START=0                  # only one reconstructed snapshot exists
VEL_END=0
VELOCITY_FIELD=Displacement
LEVELSET_FIELD=LEVEL

N_STEPS="${N_STEPS:-2684}"
DT="${DT:-0.0025}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
EXPORT_FREQ=1

# ── [3] Seeding — same as cylA production run ────────────────────────────────
# Note: FEMUSS particles live in the ORIGINAL FOM case folder, not the
# ROM one.  We point --input at the ROM folder for the MESH/velocity,
# and override --femuss-dir to point at the FOM particles.
SEED_SOURCE=femuss
FEMUSS_START=0
SEED=42

# ── [4] Search / RK4 ─────────────────────────────────────────────────────────
RK4_MODE=fused
L1_METHOD=face
L2_NEIGHBORHOOD=3
L0_SKIP_BAND=0.0
ENHANCED_SEARCH_BAND=0.0
REGISTRATION=""
ORPHAN_FALLBACK=1
HYBRID_NON_KUHN=1

# ── [5] Boundary / level-set ─────────────────────────────────────────────────
BOUNDARY_WALLS=""
BOUNDARY_PROJ_TOL=1e-6
POINT_IN_TET_TOL=1e-6
INLET_WALL=""
INLET_VELOCITY=0.0
LEVELSET_ENABLE=1
LEVELSET_MODE=zero_vel
FAILED_SUBSTAGE=zero_vel
INTERPOLATION_METHOD=direct_inverse

# ── [6] Pin velocity — MUST match the cylA production settings.  This is
#        NOT case-dependent in the current cylA config; if you know it
#        varies per case, edit here or override PIN_RPM in the env. ──────
PIN_VELOCITY=1
PIN_RPM="${PIN_RPM:--600}"
PIN_CENTER="0.0 0.0 0.0"
PIN_AXIS="0.0 0.0 1.0"
PIN_TILT=0.0

# ── [7] Export ────────────────────────────────────────────────────────────────
EXPORT_FORMAT=vtkhdf
NO_EXPORT=0
N_GROUPS=5
EXPORT_ELEMENT_IDS=0
EXPORT_ESCAPED_FLAG=1
TRACK_MAX_TEMPERATURE=1
EXPORT_TEMPERATURE=1
TEMPERATURE_FIELD=Temperature

# ── [8] Hit-stats ────────────────────────────────────────────────────────────
HIT_STATS_LOG=1

# ── [9] JAX memory ───────────────────────────────────────────────────────────
XLA_PREALLOC=0
VRAM_FRACTION=0.95
BENCHMARK_MODE=0
MONITOR_INTERVAL=0    # off; we're running short passes

PYTHON="${PYTHON:-python}"

# ── Activate venv ────────────────────────────────────────────────────────────
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
    echo "[env] Activated venv: $VENV ($(python --version))"
else
    echo "[warn] venv not found at $VENV — using current Python: $(which python)"
fi

# ── CUDA / XLA env ───────────────────────────────────────────────────────────
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS="--xla_gpu_autotune_level=4 --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_cuda_data_dir=/usr/local/cuda"
CUDA_CACHE_DIR="/tmp/${USER}-xla-cache-rom-$$"
export CUDA_CACHE_PATH="$CUDA_CACHE_DIR"
mkdir -p "$CUDA_CACHE_DIR"

if [ "$XLA_PREALLOC" = "1" ]; then
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION=$VRAM_FRACTION
    unset XLA_PYTHON_CLIENT_ALLOCATOR
else
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
    export XLA_PYTHON_CLIENT_MEM_FRACTION=$VRAM_FRACTION
fi
export TF_CPP_MIN_LOG_LEVEL=2

# ── Loop over cases ──────────────────────────────────────────────────────────
RC=0
for CASE in $CASES; do
    CASE_ID=$(printf "%03d" "$CASE")
    CASE_STEM="${CASE_PREFIX}_${CASE_ID}"

    FOM_CASE_DIR="$FOM_ROOT/${CASE_STEM}.gid"
    ROM_CASE_DIR="$ROM_ROOT/${CASE_STEM}.gid"
    ROM_MESH_PVTU="$ROM_CASE_DIR/post/${CASE_PREFIX}_0.pvtu"

    if [ ! -f "$ROM_MESH_PVTU" ]; then
        echo "ERROR: ROM PVTU missing for case $CASE_ID: $ROM_MESH_PVTU" >&2
        echo "  Run paper_benchmarks/reconstruct_rom_velocities.sh first." >&2
        RC=1
        continue
    fi
    if [ ! -d "$FOM_CASE_DIR/post/1part" ]; then
        # Fall back to no FEMUSS seeding available — user should override
        # SEED_SOURCE.  We continue but warn loudly.
        echo "WARN: FEMUSS particles at $FOM_CASE_DIR/post/1part not found; " \
             "SEED_SOURCE=femuss will fail.  Override SEED_SOURCE or provide " \
             "the 1part folder." >&2
    fi

    OUT_ROOT_CASE="$FOM_CASE_DIR/post_pt_rom_${ROM_FORMULA}"
    mkdir -p "$OUT_ROOT_CASE"

    LOG_FILE="$OUT_ROOT_CASE/tracking.log"

    echo
    echo "============================================================"
    echo " CASE ${CASE_ID}"
    echo "============================================================"
    echo "  ROM velocity : $ROM_MESH_PVTU"
    echo "  FOM particles: $FOM_CASE_DIR/post/1part"
    echo "  Output       : $OUT_ROOT_CASE"
    echo "  Log          : $LOG_FILE"
    echo "  N_STEPS/DT   : $N_STEPS / $DT"
    echo "  Started      : $(date)"
    echo "------------------------------------------------------------"

    ARGS=(
        --input              "$ROM_CASE_DIR"
        --output             "$OUT_ROOT_CASE"
        --mesh-subdir        "post"
        --mesh-pattern       "${CASE_PREFIX}_{timestep}.pvtu"
        --femuss-dir         "$FOM_CASE_DIR/post/1part"
        --femuss-pattern     "${CASE_PREFIX}_pt_{timestep}.pvtu"
        --precision          "$PRECISION"
        --velocity-source    "mesh"
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
    [ "$PIN_VELOCITY" = "0" ] && ARGS+=( --no-pin-velocity )
    [ "$HIT_STATS_LOG" = "1" ] && ARGS+=( --hit-stats-log )
    [ "$SEED_SOURCE" = "femuss" ] && ARGS+=( --femuss-start "$FEMUSS_START" )

    set -x
    "$PYTHON" -u "$JAXTRACE/run_tracking.py" "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
    CRC=${PIPESTATUS[0]}
    set +x
    echo "  [CASE ${CASE_ID}] exit code: $CRC"
    if [ "$CRC" != "0" ]; then RC=$CRC; fi

    # Write a tiny manifest linking this run to the ROM formula + case
    cat > "$OUT_ROOT_CASE/ROM_TRACKING_MANIFEST.txt" <<EOF
case_prefix   : $CASE_PREFIX
case_idx      : $CASE_ID
rom_formula   : $ROM_FORMULA
rom_pvtu      : $ROM_MESH_PVTU
fom_case_dir  : $FOM_CASE_DIR
n_steps       : $N_STEPS
dt            : $DT
seed_source   : $SEED_SOURCE
pin_rpm       : $PIN_RPM
finished      : $(date)
exit_code     : $CRC
EOF
done

rm -rf "$CUDA_CACHE_DIR"

echo
echo "============================================================"
echo " Done.  Aggregate exit code: $RC"
echo " Finished: $(date)"
echo
echo " Next step: compare final particle positions between:"
echo "   FOM tracking:  <case>.gid/post_pt/femuss_0_to_${N_STEPS}/particles.vtkhdf"
echo "   ROM tracking:  <case>.gid/post_pt_rom_${ROM_FORMULA}/femuss_0_to_${N_STEPS}/particles.vtkhdf"
echo
echo " Use paper_benchmarks/compare_rom_vs_fom_tracking.py once both are done."
echo "============================================================"
exit "$RC"
