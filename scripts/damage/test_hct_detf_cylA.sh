#!/usr/bin/env bash
# =============================================================================
# Does the HCT reconstruction reduce the det F error? Two arms on cylA.
#
#   bash test_hct_detf_cylA.sh            # both arms, raw P1 then hct_cubic
#   ARM=p1  bash test_hct_detf_cylA.sh    # one arm only
#   ARM=hct bash test_hct_detf_cylA.sh
#   N_STEPS=200 bash test_hct_detf_cylA.sh   # quick shakeout
#
# WHY cylA AND NOT A D-FAMILY CASE
# ---------------------------------
# ⚠️ HCT is STEADY-ONLY today: run_tracking.py builds the reconstruction under
# `if n_frames == 1` and otherwise prints "SKIPPED ... falling back to raw P1".
# cylA has VEL_START=VEL_END=159 -- ONE snapshot -- so HCT actually builds.
#
# ⚠️ The A family on LUMI is NOT steady despite the cylindrical tool: its
# per-case scripts are VEL_START=34 VEL_END=199, i.e. 166 snapshots. Checked
# A1-A4 and the *old variants; all 166. So they cannot test HCT either.
#
# WHAT THIS ANSWERS, AND WHAT IT DOES NOT
# ----------------------------------------
# ✅ Answers: does the divergence of the RECONSTRUCTION drive the det F error?
#    The SPR-recovered nodal gradient has ~15x lower relative divergence than
#    raw P1 (3.31e-03 vs 4.87e-02 median, measured on D2 snapshot 34), and HCT
#    interpolates through that gradient. If det F improves by a comparable
#    factor, the detachment method is viable on an HCT field.
# ⚠️ Does NOT answer: whether the production time-dependent runs can use it.
#    A steady field is a different problem, and the Bernstein array is ~2.6 GB
#    PER FRAME (n_elements, 4 subtets, 20 coeffs, 3), so 166 frames is ~430 GB.
#    This run BOUNDS the achievable gain; it does not deliver it.
#
# ⚠️ Mirrors cylA's own run_jaxtrace.sh for physics, with two deliberate
# divergences, both forced:
#   OUTPUT_TARGET  : the reference writes into the case folder (post_pt).
#                    We write to /scratch, leaving the case untouched.
#   SEED_SOURCE    : the reference uses femuss (reads 1part/ particle files).
#                    We use the same source so the trajectories are the
#                    reference's -- do NOT switch to grid, which would change
#                    which particles are measured and make the arms
#                    incomparable with everything else on this case.
# =============================================================================
set -uo pipefail

JAXTRACE="${JAXTRACE:-/flash/shared/jax/JAXTrace}"
VENV_PY="${VENV_PY:-/flash/shared/jax/.venv/bin/python}"
INPUT="${INPUT:-/flash/users/ali/data/cylA.gid}"
OUT_ROOT="${OUT_ROOT:-/scratch/shared/hct_detf}"

# ── Physics: cylA's run_jaxtrace.sh, verbatim ────────────────────────────────
MESH_SUBDIR="0eule"
FEMUSS_SUBDIR="1part"
PRECISION=float32
VEL_START=159
VEL_END=159                  # ⚠️ == VEL_START: ONE snapshot, so HCT builds
VELOCITY_FIELD=Displacement
LEVELSET_FIELD=LEVEL
N_STEPS="${N_STEPS:-2684}"
DT=0.0025
LOG_INTERVAL=100
SEED_SOURCE=femuss
FEMUSS_START=0
SEED=42
LEVELSET_MODE=zero_vel
FAILED_SUBSTAGE=zero_vel
INTERPOLATION_METHOD=direct_inverse
POINT_IN_TET_TOL=1e-6
BOUNDARY_PROJ_TOL=1e-6
ENHANCED_SEARCH_BAND=0.0
PIN_RPM=-600
PIN_CENTER="0.0 0.0 0.0"
PIN_AXIS="0.0 0.0 1.0"
PIN_TILT=0.0
N_GROUPS=5
TEMPERATURE_FIELD=Temperature

# ── Damage + detachment: det F is the quantity under test ────────────────────
DAMAGE_DRIVER=models
DAMAGE_ORDER=1
RESIDENCE_THR=1.0
TAU_REF_EDOT=1.0

# ⚠️ On-demand allocator. HCT's Bernstein array is large and the preallocating
# allocator claims ~75 % of HBM regardless of need -- measured fatal on this
# 32.6 GB card for the plain damage run already.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONPATH="$JAXTRACE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3

run_arm () {
    local tag="$1" extra_args="$2"
    local stamp; stamp="$(date +%Y%m%d_%H%M%S)"
    local out="${OUT_ROOT}/cylA_${tag}_${stamp}"
    mkdir -p "$out" || { echo "ERROR: cannot create $out" >&2; return 1; }
    local log="${out}/run.log"
    cp "$0" "${out}/run_script.sh" 2>/dev/null
    {
        echo "arm           : $tag"
        echo "extra args    : $extra_args"
        echo "case          : $INPUT"
        echo "vel           : [$VEL_START,$VEL_END]  (steady: HCT can build)"
        echo "n_steps       : $N_STEPS   dt $DT"
        echo "started       : $(date -Is)"
        echo "repo_commit   : $(cd "$JAXTRACE" && git rev-parse --short HEAD 2>/dev/null)"
    } > "${out}/RUN_INFO.txt"

    local args=(
      --input "$INPUT" --output "$out"
      --mesh-subdir "$MESH_SUBDIR" --femuss-subdir "$FEMUSS_SUBDIR"
      --precision "$PRECISION"
      --vel-range "$VEL_START" "$VEL_END"
      --velocity-field "$VELOCITY_FIELD"
      --levelset-field "$LEVELSET_FIELD"
      --n-steps "$N_STEPS" --dt "$DT" --log-interval "$LOG_INTERVAL"
      --seed-source "$SEED_SOURCE" --femuss-start "$FEMUSS_START" --seed "$SEED"
      --levelset-mode "$LEVELSET_MODE"
      --failed-substage "$FAILED_SUBSTAGE"
      --interpolation-method "$INTERPOLATION_METHOD"
      --point-in-tet-tol "$POINT_IN_TET_TOL"
      --boundary-proj-tol "$BOUNDARY_PROJ_TOL"
      --enhanced-search-band "$ENHANCED_SEARCH_BAND"
      --pin-rpm "$PIN_RPM" --pin-center $PIN_CENTER --pin-axis $PIN_AXIS
      --pin-tilt "$PIN_TILT" --n-groups "$N_GROUPS"
      --temperature-field "$TEMPERATURE_FIELD"
      --damage --damage-driver "$DAMAGE_DRIVER" --damage-order "$DAMAGE_ORDER"
      --damage-indicators --residence-edot-threshold "$RESIDENCE_THR"
      --damage-detachment --damage-tau-ref-edot "$TAU_REF_EDOT"
      --damage-out "${out}/damage_models.npz"
    )
    # shellcheck disable=SC2086
    [ -n "$extra_args" ] && args+=( $extra_args )

    echo "============================================================"
    echo " ARM: $tag        out: $out"
    echo " extra: ${extra_args:-<none>}"
    echo "============================================================"
    local t0; t0=$(date +%s)
    "$VENV_PY" -u "$JAXTRACE/run_tracking.py" "${args[@]}" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    echo " arm $tag exit $rc  elapsed $(( $(date +%s) - t0 ))s"
    # ⚠️ Did HCT actually build, or silently fall back? The run prints
    # "SKIPPED" on a multi-frame field. Surface it either way.
    grep -E "gradient-recovery" "$log" | head -4
    echo "$out"
}

ARM="${ARM:-both}"
case "$ARM" in
  p1)   run_arm p1  "--gradient-recovery 0" ;;
  hct)  run_arm hct "--gradient-recovery 1 --recovery-method hct_cubic" ;;
  both) run_arm p1  "--gradient-recovery 0"
        run_arm hct "--gradient-recovery 1 --recovery-method hct_cubic" ;;
  *) echo "ARM must be p1 | hct | both" >&2; exit 2 ;;
esac

echo
echo "============================================================"
echo " Compare det F between the arms:"
echo "   python3 scripts/damage/detachment_report.py <out>/damage_models.npz"
echo "============================================================"
