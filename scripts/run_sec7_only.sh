#!/bin/bash
# =============================================================================
# run_sec7_only.sh — Catch-up sec7 run after a partial run_paper_benchmarks.sh
#
# Use this when sec6 finished cleanly but sec7 never started (so we have
# sec6_l2_accuracy.log + manifest.json in RESULTS_DIR but no sec7 log and
# no aggregated markdown). It runs benchmark_femuss_comparison.py with
# the same arguments the driver would have used, writes the sec7 log
# next to the existing sec6 log, then re-invokes the aggregator so the
# same markdown report fills in the missing sec7 section.
#
# Usage:
#   bash scripts/run_sec7_only.sh                                # default results dir
#   RESULTS_DIR=/path/to/dir bash scripts/run_sec7_only.sh
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAXTRACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default to the latest paper_benchmarks_* directory if not specified.
RESULTS_DIR="${RESULTS_DIR:-$(ls -dt $JAXTRACE_ROOT/results/paper_benchmarks_* 2>/dev/null | head -1)}"

if [ -z "$RESULTS_DIR" ] || [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: no results directory found under $JAXTRACE_ROOT/results/" >&2
    echo "  Set RESULTS_DIR=/absolute/path explicitly." >&2
    exit 2
fi

MESH_BASE="${MESH_BASE:-/flash/users/${USER:-ali}/data/cylA.gid/post}"
# IMPORTANT: keep these defaults SINGLE-quoted. Bash brace-expansion is
# disabled for braces with no comma/.. inside (per POSIX), but it bites
# anyway when expanded through the "${VAR:-default}" form: bash
# silently moves the closing brace to the end, turning
#   cylA_{timestep}.pvtu
# into
#   cylA_{timestep.pvtu}
# Python then sees a malformed format placeholder and crashes with
# "AttributeError: 'int' object has no attribute 'pvtu'".
# Using single-quoted literals and an explicit default-test sidesteps
# the expansion. The Python script receives the pattern verbatim.
: "${MESH_PATTERN:=$(printf '%s' 'cylA_{timestep}.pvtu')}"
: "${FEMUSS_PATTERN:=$(printf '%s' 'cylA_pt_{timestep}.pvtu')}"
VEL_START="${VEL_START:-159}"
VEL_END="${VEL_END:-159}"
SEC7_FEMUSS_START="${SEC7_FEMUSS_START:-0}"
SEC7_N_STEPS="${SEC7_N_STEPS:-2684}"
SEC7_DT="${SEC7_DT:-0.0025}"
PYTHON="${PYTHON:-python}"

SEC7_LOG="$RESULTS_DIR/sec7_femuss_comparison.log"
SEC7_OUT="$RESULTS_DIR/sec7_femuss_comparison"
mkdir -p "$SEC7_OUT"

echo "============================================================"
echo " Sec 7 catch-up run"
echo "============================================================"
echo " Results dir:    $RESULTS_DIR"
echo " Mesh base:      $MESH_BASE"
echo " Mesh pattern:   $MESH_PATTERN"
echo " FEMUSS pattern: $FEMUSS_PATTERN"
echo " Vel range:      $VEL_START..$VEL_END"
echo " N_STEPS / dt:   $SEC7_N_STEPS / $SEC7_DT"
echo " Output dir:     $SEC7_OUT"
echo " Log:            $SEC7_LOG"
echo " Started:        $(date)"
echo "============================================================"

if [ ! -d "$MESH_BASE/0eule" ] || [ ! -d "$MESH_BASE/1part" ]; then
    echo "ERROR: expected $MESH_BASE to contain 0eule/ and 1part/ subdirs" >&2
    exit 3
fi

# Run sec7. Use -u for unbuffered Python so the log is streaming.
set -x
"$PYTHON" -u "$JAXTRACE_ROOT/benchmark_femuss_comparison.py" \
    --input "$MESH_BASE" \
    --output "$SEC7_OUT" \
    --mesh-pattern "$MESH_PATTERN" \
    --femuss-pattern "$FEMUSS_PATTERN" \
    --vel-range "$VEL_START" "$VEL_END" \
    --femuss-start "$SEC7_FEMUSS_START" \
    --n-steps "$SEC7_N_STEPS" \
    --dt "$SEC7_DT" \
    --registration parent_cube \
    --rk4-mode fused \
    --precision float64 \
    2>&1 | tee "$SEC7_LOG"
SEC7_EXIT=${PIPESTATUS[0]}
set +x

echo
echo " Sec 7 exit code: $SEC7_EXIT"
echo " Finished:        $(date)"
echo

# Re-aggregate.
AGG_PY="$JAXTRACE_ROOT/scripts/aggregate_paper_results.py"
if [ -f "$AGG_PY" ]; then
    REPORT="$RESULTS_DIR/RTX5090_BENCHMARK_REPORT.md"
    echo "============================================================"
    echo " Re-aggregating into $REPORT"
    echo "============================================================"
    "$PYTHON" "$AGG_PY" \
        --results-dir "$RESULTS_DIR" \
        --output "$REPORT" \
        2>&1 | tee -a "$RESULTS_DIR/aggregate.log"
fi

echo
echo "Done. Sec 7 exit: $SEC7_EXIT"
exit $SEC7_EXIT
