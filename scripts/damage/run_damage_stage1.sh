#!/usr/bin/env bash
# =============================================================================
# Stage 1 — damage fields on the original FOM mesh, all cases, all rheologies.
#
# RUN THIS ON THE WORKSTATION.  It is CPU/RAM bound (NumPy, not GPU) but the
# cohort is 20 x 180k-node meshes and cylA is 800k nodes / 3M tets, so it wants
# the workstation's memory and cores rather than a laptop.
#
#   bash run_damage_stage1.sh              # everything
#   CASES="000 003" bash run_damage_stage1.sh
#   MODELS="norton" bash run_damage_stage1.sh
#   DRY_RUN=1 bash run_damage_stage1.sh    # print the plan, run nothing
#
# Outputs land in $OUTDIR:
#   dmg_<case>_<model>.npz     nodal fields
#   dmg_<case>_<model>.json    diagnostics + acceptance table
#   stage1_summary.csv         one row per (case, model) — the table to discuss
#   logs/<case>_<model>.log    full stdout
# =============================================================================
set -uo pipefail

# ----------------------------------------------------------------------------
# Paths — adjust only if the mounts move.
# ----------------------------------------------------------------------------
REPO="${REPO:-/flash/shared/jax/JAXTrace}"
COHORT_ROOT="${COHORT_ROOT:-/scratch/shared/ROM/FOM}"
CYLA_DIR="${CYLA_DIR:-/flash/users/ali/data/cylA.gid}"
PINSHAPES_ROOT="${PINSHAPES_ROOT:-/scratch/shared/PinShapes/Supermesh}"
OUTDIR="${OUTDIR:-/scratch/shared/ROM/damage/stage1}"

# Norton tables come from the case's OWN .mat when it has one (cylA, A2 —
# both Al6063).  The 20-case cohort ships no .mat, so it falls back to cylA's.
# Set MAT_FILE=... to force one file for every case.
MAT_FALLBACK="${MAT_FALLBACK:-${CYLA_DIR}/cylA.mat}"

# ----------------------------------------------------------------------------
# What to run.
# ----------------------------------------------------------------------------
CASES="${CASES:-cylA A2 000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019}"
MODELS="${MODELS:-constant norton sellars_tegart}"
DRY_RUN="${DRY_RUN:-0}"

# The workstation's virtualenv. /usr/bin/python3 has no numpy/vtk, so the venv
# interpreter is the default rather than a fallback. Override with PY=... .
VENV="${VENV:-/flash/shared/jax/.venv}"
if [[ -z "${PY:-}" ]]; then
    if [[ -x "$VENV/bin/python" ]]; then
        PY="$VENV/bin/python"
    else
        PY="$(command -v python3 || command -v python || true)"
    fi
fi

echo "============================================================"
echo " Stage 1 — damage fields on FOM mesh"
echo "============================================================"
echo "  repo      : $REPO"
echo "  cohort    : $COHORT_ROOT"
echo "  cylA      : $CYLA_DIR"
echo "  pinshapes : $PINSHAPES_ROOT"
echo "  material  : ${MAT_FILE:-<per-case, fallback $MAT_FALLBACK>}"
echo "  outdir    : $OUTDIR"
echo "  cases     : $CASES"
echo "  models    : $MODELS"
echo "  venv      : $VENV"
echo "  python    : $PY"
echo

_precheck_fail=0
if [[ -z "$PY" ]]; then
    echo "ERROR: no interpreter found. Expected the venv at $VENV," >&2
    echo "       or set PY=/path/to/python explicitly." >&2
    _precheck_fail=1
elif ! "$PY" -c "import numpy, vtk" >/dev/null 2>&1; then
    echo "ERROR: $PY cannot import numpy and vtk." >&2
    echo "       The system python3 lacks them; use the venv:" >&2
    echo "         PY=$VENV/bin/python bash \$0" >&2
    _precheck_fail=1
fi
if [[ ! -d "$REPO" ]]; then
    echo "ERROR: repo not found at $REPO" >&2
    echo "       These defaults are the WORKSTATION's own paths. If you are seeing" >&2
    echo "       this on another machine that mounts the workstation, override e.g.:" >&2
    echo "         REPO=\$MOUNT/flash/shared/jax/JAXTrace \\" >&2
    echo "         COHORT_ROOT=\$MOUNT/scratch/shared/ROM/FOM \\" >&2
    echo "         CYLA_DIR=\$MOUNT/flash/users/ali/data/cylA.gid \\" >&2
    echo "         OUTDIR=\$MOUNT/scratch/shared/ROM/damage/stage1 bash \$0" >&2
    _precheck_fail=1
fi
if [[ ! -f "${MAT_FILE:-$MAT_FALLBACK}" ]]; then
    echo "ERROR: material file not found at ${MAT_FILE:-$MAT_FALLBACK}" >&2
    echo "       the 'norton' model needs it; use MODELS=\"constant sellars_tegart\"" >&2
    _precheck_fail=1
fi
[[ "$_precheck_fail" != "0" ]] && exit 1

# Only create output directories once the inputs check out, so a wrong-machine
# invocation fails with a clear message instead of a mkdir permission error.
if [[ "$DRY_RUN" == "0" ]]; then
    if ! mkdir -p "$OUTDIR/logs"; then
        echo "ERROR: cannot create $OUTDIR/logs" >&2
        exit 1
    fi
fi

# ----------------------------------------------------------------------------
# Sanity: the analytic regression tests must pass before any result is trusted.
# test_skewed_linear_field guards the shape-gradient transpose bug (B1).
# ----------------------------------------------------------------------------
echo "--- analytic regression tests ---"
if ! ( cd "$REPO" && "$PY" -m jaxtrace.damage.test_fields ); then
    echo "ERROR: analytic tests FAILED — not running the sweep." >&2
    exit 1
fi
echo

resolve_mat() {
    # A case's own .mat if present, else the fallback.  MAT_FILE overrides all.
    local case="$1"
    if [[ -n "${MAT_FILE:-}" ]]; then echo "$MAT_FILE"; return; fi
    case "$case" in
        cylA) [[ -f "${CYLA_DIR}/cylA.mat" ]] && { echo "${CYLA_DIR}/cylA.mat"; return; } ;;
        *)
            local own="${PINSHAPES_ROOT}/${case}.gid/${case}.mat"
            [[ -f "$own" ]] && { echo "$own"; return; } ;;
    esac
    echo "$MAT_FALLBACK"
}

resolve_pvtu() {
    # Echo the final-step PVTU for a case name.  Each family has its own
    # layout, so add a branch here when a new family appears.
    local case="$1"
    case "$case" in
        cylA)
            # post/0eule/, fields only at the LAST step (159)
            echo "${CYLA_DIR}/post/0eule/cylA_159.pvtu" ;;
        A2)
            # PinShapes: flat post/, fields at every step, last is 151
            echo "${PINSHAPES_ROOT}/A2.gid/post/A2_151.pvtu" ;;
        [0-9][0-9][0-9])
            echo "${COHORT_ROOT}/cylindrical_${case}.gid/post/cylindrical_119.pvtu" ;;
        *)
            # Unknown name: try the PinShapes layout, picking the last step.
            local d="${PINSHAPES_ROOT}/${case}.gid/post"
            if [[ -d "$d" ]]; then
                ls "$d"/${case}_*.pvtu 2>/dev/null | sort -V | tail -1
            fi ;;
    esac
}

SUMMARY="$OUTDIR/stage1_summary.csv"
if [[ "$DRY_RUN" == "0" && ! -f "$SUMMARY" ]]; then
    echo "case,model,n_nodes,n_elems,incompressibility,edot_med,edot_max,sigeq_med_MPa,eta_med,adv_wake_eta_med,adv_wake_frac_pos,ret_wake_eta_med,ret_wake_frac_pos,growth_ratio_adv_ret,tensile_flank,matches_literature,rotation,advancing_side,runtime_s" > "$SUMMARY"
fi

n_ok=0; n_skip=0; n_fail=0

for case in $CASES; do
    PVTU="$(resolve_pvtu "$case")"
    if [[ ! -f "$PVTU" ]]; then
        echo "  SKIP $case — no PVTU at $PVTU"
        n_skip=$((n_skip+1))
        continue
    fi

    for model in $MODELS; do
        TAG="${case}_${model}"
        OUT_NPZ="$OUTDIR/dmg_${TAG}.npz"
        LOG="$OUTDIR/logs/${TAG}.log"

        if [[ -f "$OUT_NPZ" ]]; then
            echo "  skip $TAG (exists)"
            n_skip=$((n_skip+1))
            continue
        fi

        MAT_FOR_CASE="$(resolve_mat "$case")"
        echo "  run  $TAG"
        if [[ "$DRY_RUN" != "0" ]]; then
            echo "         $PVTU"
            continue
        fi

        if ( cd "$REPO" && "$PY" -m jaxtrace.damage.run_stage1 \
                --pvtu "$PVTU" \
                --case "$case" \
                --model "$model" \
                --mat "$MAT_FOR_CASE" \
                --outdir "$OUTDIR" \
                --summary "$SUMMARY" ) > "$LOG" 2>&1; then
            tail -n 3 "$LOG" | sed 's/^/         /'
            n_ok=$((n_ok+1))
        else
            echo "         FAILED — see $LOG"
            tail -n 15 "$LOG" | sed 's/^/         /'
            n_fail=$((n_fail+1))
        fi
    done
done

echo
echo "============================================================"
echo "  ok=$n_ok  skipped=$n_skip  failed=$n_fail"
echo "  summary : $SUMMARY"
echo "  logs    : $OUTDIR/logs/"
echo "============================================================"

if [[ "$DRY_RUN" == "0" && -f "$SUMMARY" ]]; then
    echo
    echo "--- survey summary (no pass/fail; criterion chosen after the full cohort) ---"
    column -s, -t < "$SUMMARY" | cut -c1-200
fi
