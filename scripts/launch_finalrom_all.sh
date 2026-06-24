#!/bin/bash
# =============================================================================
# launch_finalrom_all.sh
#
# Run the final-step density + co-moving ROM pipeline for every case,
# sequentially, on the workstation. Each case's run_jaxtrace_finalrom.sh
# does tracking (common dt = 1.875e-3, per-case N_STEPS) and then fires
# run_union_finalrom.sh (last step only, no dedup, co-moving on).
#
# Existing union/average outputs are NOT touched: new results land in
# post_pt/run_finalrom/ with the 'finalrom' filename stem.
#
# Usage (on the workstation, where the FOM root is /scratch/...):
#   bash scripts/launch_finalrom_all.sh
#   CASES="004 007 010" bash scripts/launch_finalrom_all.sh   # subset
#   FOM_ROOT=/scratch/shared/ROM/FOM bash scripts/launch_finalrom_all.sh
#
# Background / queue:
#   nohup bash scripts/launch_finalrom_all.sh > finalrom_all.log 2>&1 &
# =============================================================================
set -uo pipefail

FOM_ROOT="${FOM_ROOT:-/scratch/shared/ROM/FOM}"

# Default: all 20 cases, in order. Override with CASES="004 007 ...".
if [ -n "${CASES:-}" ]; then
    CASE_LIST=( $CASES )
else
    CASE_LIST=()
    for d in "$FOM_ROOT"/cylindrical_0*.gid; do
        CASE_LIST+=( "$(basename "$d" | sed -E 's/cylindrical_([0-9]+)\.gid/\1/')" )
    done
fi

echo "======================================================"
echo " finalrom batch: ${#CASE_LIST[@]} cases"
echo " FOM_ROOT = $FOM_ROOT"
echo " cases    = ${CASE_LIST[*]}"
echo " started  = $(date)"
echo "======================================================"

n_ok=0; n_fail=0; failed=()
for cnum in "${CASE_LIST[@]}"; do
    sh="$FOM_ROOT/cylindrical_${cnum}.gid/run_jaxtrace_finalrom.sh"
    if [ ! -f "$sh" ]; then
        echo "[case $cnum] MISSING $sh — skipping" >&2
        n_fail=$((n_fail+1)); failed+=("$cnum"); continue
    fi
    echo ""
    echo "###################### case $cnum ######################"
    echo "[$(date)] bash $sh"
    bash "$sh"
    rc=$?
    if [ "$rc" = "0" ]; then
        echo "[case $cnum] OK"
        n_ok=$((n_ok+1))
    else
        echo "[case $cnum] FAILED rc=$rc" >&2
        n_fail=$((n_fail+1)); failed+=("$cnum")
    fi
done

echo ""
echo "======================================================"
echo " finalrom batch done: $n_ok ok, $n_fail failed"
[ "$n_fail" -gt 0 ] && echo " failed cases: ${failed[*]}"
echo " finished = $(date)"
echo "======================================================"
exit $([ "$n_fail" -eq 0 ] && echo 0 || echo 1)
