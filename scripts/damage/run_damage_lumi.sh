#!/usr/bin/env bash
# =============================================================================
# Stage 1 damage survey on LUMI — PinShapes families + the ROM FOM cohort.
#
#   bash run_damage_lumi.sh                      # everything, norton only
#   MODELS="norton sellars_tegart" bash run_damage_lumi.sh
#   CASES="lumi:D-ConcavityTilt/D1 rom:000" bash run_damage_lumi.sh
#   DRY_RUN=1 bash run_damage_lumi.sh            # print the plan
#
# CODE   -> /projappl/<proj>/<user>/JAXTrace
# OUTPUT -> /scratch/<proj>/<user>/damage/stage1
#
# Runs inside the LUMI JAX/ROCm container with required-packages on
# PYTHONPATH, matching the per-case run_jaxtrace.sh scripts.  No --bind:
# LUMI auto-binds /scratch, /projappl and /flash.
# =============================================================================
set -uo pipefail

# ⚠️ NOT $PROJECT: LUMI exports that as a full path (/project/project_...),
# which would produce doubled paths.  Use a private name.
PROJ_ID="${PROJ_ID:-project_465002752}"
PROJ_ID="$(basename "$PROJ_ID")"
USERDIR="${USERDIR:-hashemia}"

REPO="${REPO:-/projappl/${PROJ_ID}/${USERDIR}/JAXTrace}"
PKGS="${PKGS:-/projappl/${PROJ_ID}/${USERDIR}/required-packages}"
OUTDIR="${OUTDIR:-/scratch/${PROJ_ID}/${USERDIR}/damage/stage1}"

CASES_ROOT="${CASES_ROOT:-/scratch/${PROJ_ID}/lorenzgl/Cases}"
PINSHAPES="${PINSHAPES:-${CASES_ROOT}/PinShapes}"
ROM_ROOT="${ROM_ROOT:-${CASES_ROOT}/ROM/FOM_cases}"

SIF="${SIF:-/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif}"

MODELS="${MODELS:-norton}"
DRY_RUN="${DRY_RUN:-0}"

# PHASE: how to treat tools that are periodic at omega (threaded, fluted,
# flats, tilted).  A single snapshot is ONE arbitrary rotational phase.
#   snapshot  last step only (fast; correct only for plain cylindrical tools)
#   probe     sample PHASE_N phases across the revolution — cheap diagnostic
#   full      average every step of the revolution (~100 GB read per case)
# The revolution window comes from each case's own run_jaxtrace.sh
# (VEL_START..VEL_END); cases without one fall back to snapshot.
PHASE="${PHASE:-snapshot}"
PHASE_N="${PHASE_N:-8}"

# ---------------------------------------------------------------------------
# Case list.  Built by discovery so new cases are picked up automatically.
# ---------------------------------------------------------------------------
if [[ -z "${CASES:-}" ]]; then
    CASES=""
    if [[ "${FAMILIES:-both}" == "rom" ]]; then :; else
    for fam in "$PINSHAPES"/*/; do
        [[ -d "$fam" ]] || continue
        famname="$(basename "$fam")"
        for g in "$fam"*.gid; do
            [[ -d "$g" ]] || continue
            CASES="$CASES lumi:${famname}/$(basename "$g" .gid)"
        done
    done
    fi
    # FAMILIES selects which trees to discover: pinshapes | rom | both.
    # Without this a PinShapes-only stage also picked up all 22 ROM cases and
    # re-ran them (harmlessly, but at full cost).
    if [[ "${FAMILIES:-both}" != "pinshapes" ]]; then
        for g in "$ROM_ROOT"/cylindrical_*.gid; do
            [[ -d "$g" ]] || continue
            n="$(basename "$g" .gid)"; n="${n#cylindrical_}"
            CASES="$CASES rom:${n}"
        done
    fi
    CASES="${CASES# }"
fi

echo "============================================================"
echo " Stage 1 damage survey — LUMI"
echo "============================================================"
echo "  repo      : $REPO"
echo "  pinshapes : $PINSHAPES"
echo "  rom       : $ROM_ROOT"
echo "  outdir    : $OUTDIR"
echo "  models    : $MODELS"
echo "  families  : ${FAMILIES:-both}"
echo "  phase     : $PHASE$( [[ "$PHASE" == "probe" ]] && echo " (N=$PHASE_N)" )"
echo "  cases     : $(echo "$CASES" | wc -w) found"
echo

_fail=0
[[ -d "$REPO" ]] || { echo "ERROR: repo not found: $REPO" >&2; _fail=1; }
[[ -d "$PINSHAPES" || -d "$ROM_ROOT" ]] || { echo "ERROR: no case tree found" >&2; _fail=1; }
[[ "$_fail" == "0" ]] || exit 1

[[ "$DRY_RUN" == "0" ]] && { mkdir -p "$OUTDIR/logs" || exit 1; }

SUMMARY="$OUTDIR/stage1_summary.csv"
if [[ "$DRY_RUN" == "0" && ! -f "$SUMMARY" ]]; then
    echo "case,model,n_nodes,n_elems,incompressibility,edot_med,edot_max,sigeq_med_MPa,eta_med,adv_wake_eta_med,adv_wake_frac_pos,ret_wake_eta_med,ret_wake_frac_pos,growth_ratio_adv_ret,tensile_flank,matches_literature,rotation,advancing_side,runtime_s" > "$SUMMARY"
fi

# Highest-numbered <something>_<n>.pvtu in a post dir.
#
# ⚠️ The file prefix does NOT have to match the case folder.  These cases were
# produced by copying a reference case, renaming the folder, changing the
# parameters and re-running FEMUSS -- so e.g. D4.gid/post contains C2_*.pvtu.
# Those files ARE D4's own output and carry D4's parameters; the stale prefix
# is only a naming artefact.  Never filter on the folder name: take the
# highest-numbered PVTU whatever it is called, and report the prefix so the
# mismatch stays visible.
_last_pvtu() {
    local dir="$1"
    [[ -d "$dir" ]] || return 0
    ls "$dir"/*_[0-9]*.pvtu 2>/dev/null \
        | sed 's/.*_\([0-9]*\)\.pvtu/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-
}

resolve_pvtu() {
    local c="$1" r="" f="" n=""
    case "$c" in
        lumi:*)
            r="${c#lumi:}"; f="${r%%/*}"; n="${r##*/}"
            _last_pvtu "${PINSHAPES}/${f}/${n}.gid/post" ;;
        rom:*)
            n="${c#rom:}"
            _last_pvtu "${ROM_ROOT}/cylindrical_${n}.gid/post" ;;
    esac
}

# VEL_START/VEL_END from a case's run_jaxtrace.sh = one full tool revolution.
_vel_range() {
    local gid="$1" f="$gid/run_jaxtrace.sh" vs ve
    [[ -f "$f" ]] || return 0
    vs=$(grep -m1 "^VEL_START=" "$f" 2>/dev/null | sed 's/.*=//;s/[^0-9].*//')
    ve=$(grep -m1 "^VEL_END=" "$f" 2>/dev/null | sed 's/.*=//;s/[^0-9].*//')
    [[ -n "$vs" && -n "$ve" ]] && echo "$vs $ve"
}

resolve_gid() {
    local c="$1" r="" f="" n=""
    case "$c" in
        lumi:*) r="${c#lumi:}"; f="${r%%/*}"; n="${r##*/}"
                echo "${PINSHAPES}/${f}/${n}.gid" ;;
        rom:*)  echo "${ROM_ROOT}/cylindrical_${c#rom:}.gid" ;;
    esac
}

resolve_mat() {
    local c="$1" d="" m="" r="" f="" n=""
    case "$c" in
        lumi:*)
            r="${c#lumi:}"; f="${r%%/*}"; n="${r##*/}"
            d="${PINSHAPES}/${f}/${n}.gid" ;;
        rom:*)
            d="${ROM_ROOT}/cylindrical_${c#rom:}.gid" ;;
    esac
    m=$(ls "$d"/*.mat 2>/dev/null | head -1)
    echo "$m"
}

run_one() {
    local c="$1" model="$2"
    local pvtu mat tag log
    pvtu="$(resolve_pvtu "$c")"
    if [[ -z "$pvtu" || ! -f "$pvtu" ]]; then
        echo "  SKIP $c — no usable PVTU"
        return 2
    fi
    mat="$(resolve_mat "$c")"
    # --case must NOT contain the model: run_stage1 appends it when naming
    # outputs, so passing "<case>_<model>" produced dmg_<case>_norton_norton.
    local casename
    casename="$(echo "$c" | tr '/:' '__')"
    tag="${casename}_${model}"
    log="$OUTDIR/logs/${tag}.log"

    local suffix=""
    [[ "$PHASE" != "snapshot" ]] && suffix="_rev"
    if [[ -f "$OUTDIR/dmg_${tag}${suffix}.npz" ]]; then
        echo "  skip $tag (exists)"; return 3
    fi

    local base pfx
    base="$(basename "$pvtu")"; pfx="${base%_*}"
    echo "  run  $tag"
    # Flag only a genuine mismatch: PinShapes cases are named after the
    # folder, so a different prefix means the case was copied from another.
    # ROM cohort files are always "cylindrical_*" by design -- not a mismatch.
    case "$c" in
        lumi:*)
            [[ "${c##*/}" == "$pfx" ]] || \
                echo "         note: files named ${pfx}_* — copied from $pfx, re-run with this case's parameters" ;;
    esac
    local phase_args=()
    if [[ "$PHASE" != "snapshot" ]]; then
        local gid rng vs ve stride
        gid="$(resolve_gid "$c")"
        rng="$(_vel_range "$gid")"
        if [[ -n "$rng" ]]; then
            vs="${rng%% *}"; ve="${rng##* }"
            if [[ "$PHASE" == "probe" ]]; then
                stride=$(( (ve - vs + 1) / PHASE_N )); (( stride < 1 )) && stride=1
            else
                stride=1
            fi
            phase_args=(--phase-range "$vs" "$ve" --phase-stride "$stride")
            echo "         revolution $vs..$ve stride $stride ($PHASE)"
        else
            echo "         no VEL_START/VEL_END — snapshot for this case"
        fi
    fi

    if [[ "$DRY_RUN" != "0" ]]; then
        echo "         $pvtu"
        [[ -n "$mat" ]] && echo "         mat: $mat"
        return 0
    fi

    local args=(--pvtu "$pvtu" --case "$casename" --model "$model"
                --outdir "$OUTDIR" --summary "$SUMMARY")
    [[ -n "$mat" ]] && args+=(--mat "$mat")
    (( ${#phase_args[@]} )) && args+=("${phase_args[@]}")


    if singularity exec --cleanenv --env PYTHONPATH="$REPO:$PKGS" "$SIF" \
         python3 -m jaxtrace.damage.run_stage1 "${args[@]}" > "$log" 2>&1; then
        tail -n 2 "$log" | sed 's/^/         /'
        return 0
    else
        echo "         FAILED — see $log"
        tail -n 8 "$log" | sed 's/^/         /'
        return 1
    fi
}

n_ok=0; n_skip=0; n_fail=0
for c in $CASES; do
    for m in $MODELS; do
        run_one "$c" "$m"
        case $? in
            0) n_ok=$((n_ok+1)) ;;
            1) n_fail=$((n_fail+1)) ;;
            *) n_skip=$((n_skip+1)) ;;
        esac
    done
done

echo
echo "============================================================"
echo "  ok=$n_ok  skipped=$n_skip  failed=$n_fail"
echo "  summary : $SUMMARY"
echo "============================================================"
