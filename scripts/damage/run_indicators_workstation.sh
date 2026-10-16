#!/usr/bin/env bash
# =============================================================================
# Damage-indicator run on the fsw-gpu workstation (NOT LUMI, NOT SLURM).
#
#   bash run_indicators_workstation.sh            # short smoke test
#   MODE=full bash run_indicators_workstation.sh  # end-to-end D2 run
#   MODE=smoke N_STEPS=50 bash run_indicators_workstation.sh
#
# WHY THIS EXISTS, AND HOW IT RELATES TO THE LUMI SCRIPTS
# -------------------------------------------------------
# The authoritative per-case scripts are the run_jaxtrace.sh files that live
# beside each case under
#   /scratch/project_465002752/lorenzgl/Cases/PinShapes/<family>/<case>.gid/
# They are SLURM + Singularity, and their physics configuration is what the
# production runs used. ⚠️ Those folders are READ-ONLY for us.
#
# This script mirrors that configuration EXACTLY for the physics and search
# parameters, and replaces only the launch mechanism: no sbatch, no srun, no
# container -- just the workstation venv against a local copy of the case. The
# point is to validate the new indicator code on fast local hardware before
# spending GPU-hours on LUMI.
#
# ⚠️ KEEP THE PHYSICS BLOCK IN SYNC with the reference script. If the reference
# changes, re-diff it; a silent divergence here would make the workstation test
# validate something the overnight run does not do.
# =============================================================================
set -uo pipefail

# ── [1] Environment (workstation) ────────────────────────────────────────────
JAXTRACE="${JAXTRACE:-/flash/shared/jax/JAXTrace}"
VENV_PY="${VENV_PY:-/flash/shared/jax/.venv/bin/python}"

# Local copy of the case. ⚠️ The folder is D2.gid but the file stem inside is
# A2 -- that is a real naming quirk of this case, not an error. CASE_STEM is
# left empty so run_tracking.py auto-detects it.
INPUT="${INPUT:-/scratch/shared/PinShapes/D-ConcavityTilt/D2.gid}"
CASE_TAG="${CASE_TAG:-D2}"

MODE="${MODE:-smoke}"
OUT_ROOT="${OUT_ROOT:-/scratch/shared/damage_indicators}"

# ── [2] Run size ─────────────────────────────────────────────────────────────
# smoke: a few steps on a coarse seed grid, just to prove the pipeline runs and
#        the four accumulators come out with sane values.
# full : the production step count for D2 from the reference script.
case "$MODE" in
  smoke)
    N_STEPS="${N_STEPS:-50}"
    SEED_GRID="${SEED_GRID:-20 40 14}"
    EXPORT_FREQ="${EXPORT_FREQ:-25}"
    VEL_END="${VEL_END:-40}"
    ;;
  full)
    N_STEPS="${N_STEPS:-8000}"
    SEED_GRID="${SEED_GRID:-60 120 40}"
    EXPORT_FREQ="${EXPORT_FREQ:-50}"
    VEL_END="${VEL_END:-199}"
    ;;
  *) echo "ERROR: MODE must be 'smoke' or 'full' (got '$MODE')" >&2; exit 2 ;;
esac

# ── [3] Physics / search — MIRRORS the reference run_jaxtrace.sh for D2 ──────
PRECISION=float32
VEL_START=34
VELOCITY_FIELD=Displacement
LEVELSET_FIELD=LevelSet
DT=0.0003629
LOG_INTERVAL=10
SEED_SOURCE=grid
SEED_FRACTION="0.0 0.2 0.0 1.0 0.0 1.0"
SEED=42
RK4_MODE=fused
L1_METHOD=face
L2_NEIGHBORHOOD=3
L0_SKIP_BAND=0.0
ENHANCED_SEARCH_BAND=0.0
BOUNDARY_PROJ_TOL=1e-6
POINT_IN_TET_TOL=1e-6
LEVELSET_MODE=zero_vel
FAILED_SUBSTAGE=last_valid_vel
INTERPOLATION_METHOD=direct_inverse
PIN_RPM=-600
PIN_CENTER="0.0 0.0 0.0"
PIN_AXIS="0.0 0.0 1.0"
PIN_TILT=0.0
N_GROUPS=5
BOUNDARY_WALLS="x_max=clamp"
MESH_LOAD_WORKERS=8

# ── [4] The indicators under test ────────────────────────────────────────────
# models driver is REQUIRED: the Piwnik-Plata integrand is p/sigma_bar = -eta,
# and eta only exists in the 3-slot [edot, eta, s1_ratio] stack.
DAMAGE_DRIVER=models
DAMAGE_ORDER="${DAMAGE_ORDER:-1}"
RESIDENCE_THR="${RESIDENCE_THR:-1.0}"

# ── [5] GPU env (CUDA on the workstation, not ROCm) ──────────────────────────
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="$JAXTRACE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT_ROOT}/${CASE_TAG}_${MODE}_${STAMP}"
mkdir -p "$OUT" || { echo "ERROR: cannot create $OUT" >&2; exit 1; }
LOG="${OUT}/run.log"

ARGS=(
  --input  "$INPUT"
  --output "$OUT"
  --precision      "$PRECISION"
  --vel-range      "$VEL_START" "$VEL_END"
  --velocity-field "$VELOCITY_FIELD"
  --levelset-field "$LEVELSET_FIELD"
  --n-steps        "$N_STEPS"
  --dt             "$DT"
  --log-interval   "$LOG_INTERVAL"
  --export-freq    "$EXPORT_FREQ"
  --seed-source    "$SEED_SOURCE"
  --seed-grid      $SEED_GRID
  --seed-fraction  $SEED_FRACTION
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
  --boundary-walls "$BOUNDARY_WALLS"
  --mesh-load-workers "$MESH_LOAD_WORKERS"
  # the indicators
  --damage
  --damage-driver "$DAMAGE_DRIVER"
  --damage-order  "$DAMAGE_ORDER"
  --damage-indicators
  --residence-edot-threshold "$RESIDENCE_THR"
  --damage-out "${OUT}/damage_models.npz"
)

echo "============================================================"
echo " workstation indicator run"
echo "   mode      : $MODE"
echo "   case      : $INPUT  (tag $CASE_TAG)"
echo "   steps     : $N_STEPS  dt $DT   vel [$VEL_START,$VEL_END]"
echo "   seed grid : $SEED_GRID  frac $SEED_FRACTION"
echo "   driver    : $DAMAGE_DRIVER  order $DAMAGE_ORDER"
echo "   t_res thr : $RESIDENCE_THR 1/s"
echo "   out       : $OUT"
echo "   python    : $VENV_PY"
echo "============================================================"

if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: no python at $VENV_PY" >&2; exit 1
fi
if [ ! -d "$INPUT" ]; then
  echo "ERROR: case folder not found: $INPUT" >&2; exit 1
fi

t0=$(date +%s)
"$VENV_PY" -u "$JAXTRACE/run_tracking.py" "${ARGS[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
t1=$(date +%s)

echo
echo "============================================================"
echo " exit $rc   elapsed $(( t1 - t0 ))s"
if [ "$rc" -eq 0 ]; then
  echo " damage archive:"
  ls -la "${OUT}/damage_models.npz" 2>/dev/null || echo "   (missing!)"
  echo
  echo " indicator summary from the log:"
  grep -E "\[damage\]" "$LOG" | tail -20
else
  echo " FAILED — last 25 log lines:"
  tail -25 "$LOG"
fi
echo "============================================================"
exit $rc
