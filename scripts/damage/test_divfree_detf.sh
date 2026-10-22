#!/usr/bin/env bash
# =============================================================================
# THE det F GATE on the divergence-free projection. Two arms on D2.
#
#   bash test_divfree_detf.sh              # both arms
#   ARM=raw bash test_divfree_detf.sh      # one arm only
#   ARM=df  bash test_divfree_detf.sh
#   N_STEPS=400 bash test_divfree_detf.sh  # quick shakeout
#
# WHAT THIS ANSWERS
# -----------------
# §4z already showed, with an INDEPENDENT numpy integrator, that ln J along a
# pathline falls from 7.6e-04 to 1.7e-18 under the projection, with |det F - 1|
# exactly 0 and trajectories shifting <= 2.9e-04 relative. This runs the SAME
# comparison through the PRODUCTION kernel, which is the thing that will ship:
# the vmapped RK4, the octree search, the level-set masking, the detachment
# columns and the real accumulators.
#
# ⚠️ TIME-DEPENDENT BY DEFAULT. §4ab moved the projection IN-KERNEL (it wraps
# the active interpolant and reads tr A from arrays the kernel already holds),
# so it no longer rides the steady-only gradient-recovery path and needs no
# per-frame array. VEL_END therefore defaults to D2's production 199 -- 166
# frames -- and this is a test of what will actually ship.
# Set VEL_END=34 to compare against the steady §4z numbers instead.
#
# ⚠️ The case folder is READ-ONLY. Output goes to /scratch only.
# =============================================================================
set -uo pipefail

PROJECT="${SLURM_JOB_ACCOUNT:-project_465002752}"
SIF="${SIF:-/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif}"
JAXTRACE="${JAXTRACE:-/projappl/${PROJECT}/hashemia/JAXTrace}"
PKGS="${PKGS:-/projappl/${PROJECT}/hashemia/required-packages}"
INPUT="${INPUT:-/scratch/${PROJECT}/lorenzgl/Cases/PinShapes/D-ConcavityTilt/D2.gid}"
OUT_ROOT="${OUT_ROOT:-/scratch/${PROJECT}/hashemia/divfree_detf}"
OCTREE_CACHE="${OCTREE_CACHE:-/scratch/${PROJECT}/hashemia/octree_cache}"

# ⚠️ PATHS. D2's PVTUs live DIRECTLY in post/ with an "A2_" stem, not in
# post/0eule/ with a "D2_" stem -- the D family reuses A/B/C stems (D1->A1,
# D2->A2, D3->B4, D4->C2). run_tracking.py's defaults assume 0eule/ and the
# case name, so BOTH must be passed explicitly; omitting them is what made the
# first attempt at this gate fail with "D2_34.pvtu not found". Detected from
# disk rather than hard-coded so this script works on any case.
MESH_SUBDIR="${MESH_SUBDIR-}"
POST_DIR="$INPUT/post"
[ -n "$MESH_SUBDIR" ] && POST_DIR="$POST_DIR/$MESH_SUBDIR"
if [ -z "${MESH_PATTERN:-}" ]; then
    _stem=$(ls "$POST_DIR"/*_34.pvtu 2>/dev/null | head -1             | xargs -r basename | sed 's/_34\.pvtu$//')
    if [ -z "$_stem" ]; then
        echo "ERROR: no *_34.pvtu under $POST_DIR -- cannot infer the stem" >&2
        exit 2
    fi
    MESH_PATTERN="${_stem}_{timestep}.pvtu"
fi
echo "[paths] post dir  : $POST_DIR"
echo "[paths] mesh patt : $MESH_PATTERN"

# ── Physics: D2's run_jaxtrace_D2.sh, with the two forced divergences above ──
PRECISION=float32
VEL_START="${VEL_START:-34}"
VEL_END="${VEL_END:-199}"         # D2 production range (166 frames)
VELOCITY_FIELD=Displacement
LEVELSET_FIELD=LEVEL
TEMPERATURE_FIELD=Temperature
N_STEPS="${N_STEPS:-2000}"
DT=0.0003629
LOG_INTERVAL=200
SEED_SOURCE=grid
SEED_FRACTION="0.0 0.2 0.0 1.0 0.0 1.0"
SEED_GRID="${SEED_GRID:-40 40 20}"
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

# ── Damage: det F is the quantity under test ──────────────────────────────────
DAMAGE_DRIVER=models
DAMAGE_ORDER=1
RESIDENCE_THR=1.0
TAU_REF_EDOT=1.0

export JAX_PLATFORMS="${JAX_PLATFORMS:-rocm}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONPATH="$JAXTRACE:$PKGS"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3

mkdir -p "$OUT_ROOT" "$OCTREE_CACHE"

run_arm () {
    local tag="$1" extra="$2"
    local stamp; stamp="$(date +%Y%m%d_%H%M%S)"
    local out="${OUT_ROOT}/D2_${tag}_${stamp}"
    mkdir -p "$out" || return 1
    local log="${out}/run.log"
    {
        echo "arm         : $tag"
        echo "extra args  : $extra"
        echo "case        : $INPUT"
        echo "vel         : [$VEL_START,$VEL_END]  (steady so recovery builds)"
        echo "n_steps     : $N_STEPS   dt $DT"
        echo "started     : $(date -Is)"
        echo "repo_commit : $(cd "$JAXTRACE" && git rev-parse --short HEAD 2>/dev/null)"
    } > "${out}/RUN_INFO.txt"

    local args=(
      --input "$INPUT" --output "$out"
      --mesh-subdir "$MESH_SUBDIR"
      --mesh-pattern "$MESH_PATTERN"
      --precision "$PRECISION"
      --vel-range "$VEL_START" "$VEL_END"
      --velocity-field "$VELOCITY_FIELD"
      --levelset-field "$LEVELSET_FIELD"
      --n-steps "$N_STEPS" --dt "$DT" --log-interval "$LOG_INTERVAL"
      --seed-source "$SEED_SOURCE" --seed-fraction $SEED_FRACTION
      --seed-grid $SEED_GRID --seed "$SEED"
      --rk4-mode "$RK4_MODE" --l1-method "$L1_METHOD"
      --l2-neighborhood "$L2_NEIGHBORHOOD" --l0-skip-band "$L0_SKIP_BAND"
      --enhanced-search-band "$ENHANCED_SEARCH_BAND"
      --boundary-proj-tol "$BOUNDARY_PROJ_TOL"
      --point-in-tet-tol "$POINT_IN_TET_TOL"
      --levelset-mode "$LEVELSET_MODE"
      --failed-substage "$FAILED_SUBSTAGE"
      --interpolation-method "$INTERPOLATION_METHOD"
      --pin-rpm "$PIN_RPM" --pin-center $PIN_CENTER --pin-axis $PIN_AXIS
      --pin-tilt "$PIN_TILT" --n-groups "$N_GROUPS"
      --temperature-field "$TEMPERATURE_FIELD"
      --octree-cache "$OCTREE_CACHE"
      --damage --damage-driver "$DAMAGE_DRIVER" --damage-order "$DAMAGE_ORDER"
      --damage-indicators --residence-edot-threshold "$RESIDENCE_THR"
      --damage-detachment --damage-tau-ref-edot "$TAU_REF_EDOT"
      --damage-divergence analytic
      --damage-out "${out}/damage_models.npz"
      --no-export
    )
    # shellcheck disable=SC2086
    [ -n "$extra" ] && args+=( $extra )

    echo "============================================================"
    echo " ARM: $tag     out: $out"
    echo " extra: ${extra:-<none>}"
    echo "============================================================"
    local t0; t0=$(date +%s)
    singularity exec -B /scratch/${PROJECT} -B /projappl/${PROJECT} \
        "$SIF" python3 -u "$JAXTRACE/run_tracking.py" "${args[@]}" 2>&1 \
        | tee "$log"
    local rc=${PIPESTATUS[0]}
    echo " arm $tag exit $rc  elapsed $(( $(date +%s) - t0 ))s"
    # ⚠️ Surface whether recovery actually BUILT and whether the projection
    # actually RAN. A silent fallback is the failure mode this guards against.
    echo " ---- recovery / projection lines ----"
    grep -E "gradient-recovery|divergence-free" "$log" | head -12
    echo " ---- det F gate ----"
    grep -iE "det F|detF|ln_J|ln J|GATE|occupanc" "$log" | tail -16
    echo "$tag,$rc,$out" >> "${OUT_ROOT}/arms.csv"
}

ARM="${ARM:-both}"
case "$ARM" in
  raw) run_arm raw "--gradient-recovery 0" ;;
  df)  run_arm df  "--gradient-recovery 0 --divergence-free" ;;
  both)
    # ⚠️ Raw P1 in BOTH arms, so the ONLY difference is the projection.
    # Gradient recovery is off: it is steady-only and irrelevant now that the
    # projection is in-kernel, and leaving it on would confound the arms.
    run_arm raw "--gradient-recovery 0"
    run_arm df  "--gradient-recovery 0 --divergence-free" ;;
  *) echo "ARM must be raw | df | both" >&2; exit 2 ;;
esac

echo
echo "============================================================"
echo " Compare det F between the arms:"
echo "   python3 $JAXTRACE/scripts/damage/detachment_report.py <out>/damage_models.npz"
echo "============================================================"
