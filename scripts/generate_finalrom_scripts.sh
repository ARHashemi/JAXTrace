#!/bin/bash
# =============================================================================
# generate_finalrom_scripts.sh
#
# Generate, for every cylindrical_0NN.gid case, a self-contained pair of
# scripts that re-run particle tracking and compute the FINAL-STEP density
# + co-moving density (NOT the time-averaged union) for the 2nd-stage ROM:
#
#   run_jaxtrace_finalrom.sh   tracking with a COMMON dt for all cases
#                              (dt = 1.875e-3; N_STEPS = round(t_max/dt) per
#                              case so v_adv*t_max = 0.075 is preserved),
#                              then fires the union hook below.
#   run_union_finalrom.sh      run_density_union.py on the LAST step only
#                              (--step-tail 1, --dedup-mode none), with
#                              --drift-velocity <v_adv> 0 0 so the embedded
#                              co-moving (uniform-reference) density is built.
#
# Outputs use a distinct run folder (post_pt/run_finalrom/) and filename
# stem (finalrom_*) so NOTHING from the existing union/average runs is
# overwritten.
#
# Why a common dt is valid: tracking integrates x += dt*v(x) over a FROZEN
# velocity field (VEL_START=VEL_END), so the continuous trajectory depends
# only on t_max = dt*N_STEPS, not on dt. dt is purely the numerical step;
# 1.875e-3 is the finest dt already used in the set, hence accurate enough
# for every case. t_max (and pin revolutions, advection distance) stay
# per-case via N_STEPS.
#
# Usage:   bash scripts/generate_finalrom_scripts.sh [FOM_ROOT]
#          (default FOM_ROOT = /home/arhashemi/fsw-gpu/scratch/shared/ROM/FOM)
# =============================================================================
set -euo pipefail

FOM_ROOT="${1:-/home/arhashemi/fsw-gpu/scratch/shared/ROM/FOM}"

# Common integration step for ALL cases (finest currently in use).
DT_COMMON="1.8750000000e-03"
# Design invariant: v_adv * t_max = 0.075  ->  N_STEPS = 0.075/(v_adv*dt).
VT_INVARIANT="0.075"

# Templates: a known-good tracking script (RPM=-400, structurally clean)
# and the newer union script (carries the co-moving block).
TRACK_TEMPLATE="${FOM_ROOT}/cylindrical_014.gid/run_jaxtrace.sh"
UNION_TEMPLATE="${FOM_ROOT}/cylindrical_001.gid/run_union.sh"

[ -f "$TRACK_TEMPLATE" ] || { echo "missing track template $TRACK_TEMPLATE" >&2; exit 1; }
[ -f "$UNION_TEMPLATE" ] || { echo "missing union template $UNION_TEMPLATE" >&2; exit 1; }

# Remote-native case-dir prefix the scripts expect (they run on the
# workstation where the mount root is /scratch, not the local mount path).
REMOTE_PREFIX="/scratch/shared/ROM/FOM"

read_var () { # read_var <file> <VARNAME> -> value (first match, strip comment)
    grep -E "^$2=" "$1" | head -1 | sed -E "s/^$2=//; s/[[:space:]]*#.*//" | tr -d '"'
}

n_done=0
for case_sh in "$FOM_ROOT"/cylindrical_0*.gid/run_jaxtrace.sh; do
    case_dir="$(dirname "$case_sh")"
    cnum="$(basename "$case_dir" | sed -E 's/cylindrical_([0-9]+)\.gid/\1/')"

    v_adv="$(read_var "$case_sh" INLET_VELOCITY)"
    # N_STEPS = round( 0.075 / (v_adv * dt) )
    n_steps="$(python3 -c "print(round($VT_INVARIANT/(float('$v_adv')*float('$DT_COMMON'))))")"

    remote_case="${REMOTE_PREFIX}/cylindrical_${cnum}.gid"

    # ---- run_jaxtrace_finalrom.sh : common dt, per-case N_STEPS, new tag ----
    out_track="${case_dir}/run_jaxtrace_finalrom.sh"
    sed -e "s|^INPUT=.*|INPUT=\"${remote_case}\"|" \
        -e "s|^INLET_VELOCITY=.*|INLET_VELOCITY=${v_adv}|" \
        -e "s|^DT=.*|DT=${DT_COMMON}|" \
        -e "s|^N_STEPS=.*|N_STEPS=${n_steps}|" \
        -e "s|^RUN_TAG=.*|RUN_TAG=\"run_finalrom\"|" \
        -e 's|_UNION_SH="$(dirname "$0")/run_union.sh"|_UNION_SH="$(dirname "$0")/run_union_finalrom.sh"|' \
        "$TRACK_TEMPLATE" > "$out_track"
    # The PIN_RPM of the template is -400; restore the case's own value.
    rpm="$(read_var "$case_sh" PIN_RPM)"
    sed -i -e "s|^PIN_RPM=.*|PIN_RPM=${rpm}|" "$out_track"
    chmod +x "$out_track" 2>/dev/null || true

    # ---- run_union_finalrom.sh : last step only, no dedup, co-moving on ----
    out_union="${case_dir}/run_union_finalrom.sh"
    # Use the NEW run folder explicitly so the union targets the finalrom
    # particles, and write to a finalrom stem so nothing is overwritten.
    sed -e "s|^CASE_DIR=.*|CASE_DIR=\"${remote_case}\"|" \
        -e "s|^FILENAME_STEM=.*|FILENAME_STEM=finalrom|" \
        -e "s|^STEP_TAIL=.*|STEP_TAIL=1                        # FINAL step only (no time-average)|" \
        -e "s|^DEDUP_MODE=.*|DEDUP_MODE=none                   # single step: dedup is a no-op|" \
        -e "s|^DRIFT_VELOCITY=.*|DRIFT_VELOCITY=\"${v_adv} 0 0\"   # co-moving subtraction ON|" \
        -e "s|^WRITE_DENSITY=.*|WRITE_DENSITY=1|" \
        "$UNION_TEMPLATE" > "$out_union"
    # Point PARTICLES at the new run_finalrom folder (override the
    # latest-run auto-detect so it can't pick an old run_grid-frac_*).
    # Insert an explicit default just after the CASE_DIR line.
    python3 - "$out_union" "$remote_case" <<'PYEOF'
import sys, re
path, remote_case = sys.argv[1], sys.argv[2]
s = open(path).read()
needle = 'if [ -z "${PARTICLES:-}" ]; then'
inject = (
    'if [ -z "${PARTICLES:-}" ]; then\n'
    '    # finalrom: target the new run_finalrom folder explicitly.\n'
    '    PARTICLES="$CASE_DIR/$OUTPUT_CASE_SUBFOLDER/run_finalrom/particles.vtkhdf"\n'
    '    [ -f "$PARTICLES" ] || PARTICLES=""\n'
)
if 'run_finalrom/particles.vtkhdf' not in s:
    s = s.replace(needle, inject, 1)
    open(path, 'w').write(s)
PYEOF
    chmod +x "$out_union" 2>/dev/null || true

    echo "case ${cnum}: v_adv=${v_adv} rpm=${rpm} -> N_STEPS=${n_steps} (dt=${DT_COMMON})"
    n_done=$((n_done+1))
done

echo ""
echo "Generated ${n_done} case script pairs (run_jaxtrace_finalrom.sh + run_union_finalrom.sh)."
echo "Common dt = ${DT_COMMON}; outputs -> post_pt/run_finalrom/, stem 'finalrom' (existing runs untouched)."
