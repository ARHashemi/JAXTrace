#!/bin/bash
# =============================================================================
# launch_finalstep_density_all.sh
#
# Compute FINAL-STEP density (+ co-moving) for every case from the EXISTING
# particle trajectories (no re-tracking). Sequential. Each case runs its
# run_finalstep_density.sh -> run_grid-frac_*/union/finalstep_union_density.vtkhdf
# on the common max-final-step grid at seed resolution. The time-averaged
# union density (particles_union_density.vtkhdf) is left untouched.
#
# Usage (on the workstation):
#   bash scripts/launch_finalstep_density_all.sh
#   CASES="002 010" bash scripts/launch_finalstep_density_all.sh
#   nohup bash scripts/launch_finalstep_density_all.sh > finalstep_density.log 2>&1 &
# =============================================================================
set -uo pipefail
FOM_ROOT="${FOM_ROOT:-/scratch/shared/ROM/FOM}"

if [ -n "${CASES:-}" ]; then CASE_LIST=( $CASES ); else
    CASE_LIST=(); for d in "$FOM_ROOT"/cylindrical_0*.gid; do
        CASE_LIST+=( "$(basename "$d" | sed -E 's/cylindrical_([0-9]+)\.gid/\1/')" ); done
fi

echo "=== finalstep density: ${#CASE_LIST[@]} cases, started $(date) ==="
n_ok=0; n_fail=0; failed=()
for c in "${CASE_LIST[@]}"; do
    sh="$FOM_ROOT/cylindrical_${c}.gid/run_finalstep_density.sh"
    [ -f "$sh" ] || { echo "[case $c] MISSING $sh"; n_fail=$((n_fail+1)); failed+=("$c"); continue; }
    echo ""; echo "############ case $c ($(date)) ############"
    bash "$sh"; rc=$?
    if [ "$rc" = 0 ]; then echo "[case $c] OK"; n_ok=$((n_ok+1));
    else echo "[case $c] FAILED rc=$rc"; n_fail=$((n_fail+1)); failed+=("$c"); fi
done
echo ""; echo "=== done: $n_ok ok, $n_fail failed $([ $n_fail -gt 0 ] && echo "(${failed[*]})"); $(date) ==="
exit $([ "$n_fail" -eq 0 ] && echo 0 || echo 1)
