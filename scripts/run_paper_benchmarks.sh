#!/bin/bash
# =============================================================================
# run_paper_benchmarks.sh — Reproduce the JAXTrace paper's Section 6 + 7
# benchmark numbers on the workstation, with the AABB-overlap variant
# (currently shown as "---" in the paper) included at all perturbation
# levels.
#
# Outputs go to:
#   $RESULTS_DIR/sec6_l2_accuracy.log
#   $RESULTS_DIR/sec6_l2_accuracy.json         (if the script emits one)
#   $RESULTS_DIR/sec7_femuss_comparison.log
#   $RESULTS_DIR/sec7_femuss_comparison/       (per-step CSVs, deviation maps)
#   $RESULTS_DIR/manifest.json                 (hardware + script versions)
#
# The companion aggregator scripts/aggregate_paper_results.py reads the
# log files and emits a markdown doc shaped like the paper's sec6/sec7
# tables.
#
# Usage:
#   bash scripts/run_paper_benchmarks.sh                 # full run, default mesh
#   MESH_BASE=/path/to/cylA.gid bash scripts/run_paper_benchmarks.sh
#   SKIP_SEC7=1 bash scripts/run_paper_benchmarks.sh     # just sec6
#   SKIP_SEC6=1 bash scripts/run_paper_benchmarks.sh     # just sec7
#
# Expected runtime on RTX 5090: ~60–90 minutes.
# =============================================================================

set -uo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
# JAXTrace repo root: the directory two levels up from this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAXTRACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Mesh + FEMUSS data. The default points at the workstation's local
# /flash copy. Override via env vars.
MESH_BASE="${MESH_BASE:-/flash/users/${USER:-ali}/data/cylA.gid/post}"
MESH_SUBDIR="${MESH_SUBDIR:-0eule}"
FEMUSS_SUBDIR="${FEMUSS_SUBDIR:-1part}"
# IMPORTANT: keep these defaults SINGLE-quoted. Bash expansion through
# "${VAR:-default}" with the literal braces silently corrupts
# "cylA_{timestep}.pvtu" into "cylA_{timestep.pvtu}" — Python then
# crashes with "AttributeError: 'int' object has no attribute 'pvtu'".
# Use ${VAR:=...} with a printf-built literal to keep the placeholder
# intact. (Bug seen in the 2026-06-15 workstation run.)
: "${MESH_PATTERN:=$(printf '%s' 'cylA_{timestep}.pvtu')}"
: "${FEMUSS_PATTERN:=$(printf '%s' 'cylA_pt_{timestep}.pvtu')}"

# Velocity timestep range used in the paper.
VEL_START="${VEL_START:-159}"
VEL_END="${VEL_END:-159}"

# Sec6 §6.2 batch size for accuracy + per-method timing.
NP_ACCURACY="${NP_ACCURACY:-10000}"

# Sec6 §6.4 scalability sweep sizes.
SCALABILITY_SIZES="${SCALABILITY_SIZES:-1000 2000 5000 10000 20000 50000 100000 200000 500000}"

# Sec7 application: match the paper's published configuration.
SEC7_FEMUSS_START="${SEC7_FEMUSS_START:-0}"
SEC7_N_STEPS="${SEC7_N_STEPS:-2684}"
SEC7_DT="${SEC7_DT:-0.0025}"

# Output directory.
RESULTS_DIR="${RESULTS_DIR:-${JAXTRACE_ROOT}/results/paper_benchmarks_$(date +%Y%m%d_%H%M%S)}"

# Toggles.
SKIP_SEC6="${SKIP_SEC6:-0}"
SKIP_SEC7="${SKIP_SEC7:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"

# Python interpreter — the workstation's shared venv when available.
PYTHON="${PYTHON:-python}"

# ── Pre-flight checks ───────────────────────────────────────────────────────
echo "============================================================"
echo " JAXTrace — Paper benchmark suite"
echo "============================================================"
echo " JAXTrace root:  $JAXTRACE_ROOT"
echo " Mesh base:      $MESH_BASE"
echo " Mesh sub:       $MESH_SUBDIR  (pattern '$MESH_PATTERN')"
echo " FEMUSS sub:     $FEMUSS_SUBDIR  (pattern '$FEMUSS_PATTERN')"
echo " Vel range:      $VEL_START..$VEL_END"
echo " Sec6 N_p:       $NP_ACCURACY"
echo " Sec6 scaling:   $SCALABILITY_SIZES"
echo " Sec7 steps/dt:  $SEC7_N_STEPS / $SEC7_DT"
echo " Results dir:    $RESULTS_DIR"
echo " Python:         $($PYTHON --version 2>&1)"
echo "============================================================"

if [ ! -d "$MESH_BASE/$MESH_SUBDIR" ]; then
    echo "ERROR: mesh dir not found: $MESH_BASE/$MESH_SUBDIR" >&2
    echo "  Edit MESH_BASE / MESH_SUBDIR at the top of this script" >&2
    echo "  or pass them as env vars." >&2
    exit 2
fi
if [ "$SKIP_SEC7" != "1" ] && [ ! -d "$MESH_BASE/$FEMUSS_SUBDIR" ]; then
    echo "WARNING: FEMUSS particle dir not found: $MESH_BASE/$FEMUSS_SUBDIR" >&2
    echo "  Sec7 will be skipped automatically." >&2
    SKIP_SEC7=1
fi

mkdir -p "$RESULTS_DIR"

# ── Manifest: hardware + git commit + start time ────────────────────────────
MANIFEST="$RESULTS_DIR/manifest.json"
{
    echo "{"
    echo "  \"started\": \"$(date -Iseconds)\","
    echo "  \"hostname\": \"$(hostname)\","
    echo "  \"jaxtrace_root\": \"$JAXTRACE_ROOT\","
    echo "  \"jaxtrace_commit\": \"$(cd "$JAXTRACE_ROOT" && git rev-parse HEAD 2>/dev/null || echo 'unknown')\","
    echo "  \"jaxtrace_branch\": \"$(cd "$JAXTRACE_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')\","
    echo "  \"mesh_base\": \"$MESH_BASE\","
    echo "  \"mesh_pattern\": \"$MESH_PATTERN\","
    echo "  \"vel_range\": [$VEL_START, $VEL_END],"
    echo "  \"sec6_n_particles\": $NP_ACCURACY,"
    echo "  \"sec7_n_steps\": $SEC7_N_STEPS,"
    echo "  \"sec7_dt\": $SEC7_DT,"
    if command -v nvidia-smi >/dev/null 2>&1; then
        _gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)
        echo "  \"gpu\": \"$_gpu\","
    fi
    _jax_ver=$($PYTHON -c "import jax; print(jax.__version__)" 2>/dev/null || echo "unknown")
    echo "  \"jax_version\": \"$_jax_ver\","
    echo "  \"python\": \"$($PYTHON --version 2>&1)\""
    echo "}"
} > "$MANIFEST"
echo
echo "[manifest] wrote $MANIFEST"
echo

# ── Sec 6: accuracy + timing + scaling for all 7 methods ────────────────────
SEC6_EXIT=0
if [ "$SKIP_SEC6" != "1" ]; then
    SEC6_LOG="$RESULTS_DIR/sec6_l2_accuracy.log"
    echo "============================================================"
    echo " [1/2] Sec 6  — benchmark_l2_accuracy.py"
    echo "============================================================"
    echo " Output:  $SEC6_LOG"
    echo " Started: $(date)"
    echo
    # The --registration=all switch adds the AABB-overlap variant
    # alongside vertex-multi and parent_cube, replacing the "---" entries
    # in the paper's Table 3/4. --float64 matches the paper's precision.
    set -x
    "$PYTHON" "$JAXTRACE_ROOT/benchmark_l2_accuracy.py" \
        --input "$MESH_BASE" \
        --mesh-subdir "$MESH_SUBDIR" \
        --vel-range "$VEL_START" "$VEL_END" \
        --n-particles "$NP_ACCURACY" \
        --perturbations 0.0 0.1 0.2 0.5 0.7 1.0 \
        --position-types centroid random near_face near_edge near_vertex \
        --warmup-runs 3 \
        --timing-runs 7 \
        --scalability \
        --scalability-sizes $SCALABILITY_SIZES \
        --registration all \
        --cost-analysis \
        --float64 \
        2>&1 | tee "$SEC6_LOG"
    SEC6_EXIT=${PIPESTATUS[0]}
    set +x
    echo
    echo " Sec 6 exit code: $SEC6_EXIT"
    echo
fi

# ── Sec 7: FEMUSS application comparison ────────────────────────────────────
SEC7_EXIT=0
if [ "$SKIP_SEC7" != "1" ]; then
    SEC7_LOG="$RESULTS_DIR/sec7_femuss_comparison.log"
    SEC7_OUT="$RESULTS_DIR/sec7_femuss_comparison"
    mkdir -p "$SEC7_OUT"
    echo "============================================================"
    echo " [2/2] Sec 7  — benchmark_femuss_comparison.py"
    echo "============================================================"
    echo " Output:  $SEC7_LOG"
    echo " OutDir:  $SEC7_OUT"
    echo " Started: $(date)"
    echo
    set -x
    "$PYTHON" "$JAXTRACE_ROOT/benchmark_femuss_comparison.py" \
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
    echo
fi

# ── Aggregate into markdown doc ─────────────────────────────────────────────
if [ "$SKIP_AGGREGATE" != "1" ]; then
    AGG_PY="$JAXTRACE_ROOT/scripts/aggregate_paper_results.py"
    if [ -f "$AGG_PY" ]; then
        echo "============================================================"
        echo " Aggregating results into markdown..."
        echo "============================================================"
        "$PYTHON" "$AGG_PY" \
            --results-dir "$RESULTS_DIR" \
            --output "$RESULTS_DIR/RTX5090_BENCHMARK_REPORT.md" \
            2>&1 | tee -a "$RESULTS_DIR/aggregate.log"
        AGG_EXIT=${PIPESTATUS[0]}
        echo " Aggregate exit code: $AGG_EXIT"
    else
        echo "WARNING: $AGG_PY not found; skipping aggregation." >&2
    fi
fi

echo
echo "============================================================"
echo " Done."
echo " Results dir: $RESULTS_DIR"
echo " Sec6 exit:   $SEC6_EXIT"
echo " Sec7 exit:   $SEC7_EXIT"
echo "============================================================"

# Finalise manifest with end time and exit codes
{
    head -n -1 "$MANIFEST"
    echo "  ,"
    echo "  \"finished\": \"$(date -Iseconds)\","
    echo "  \"sec6_exit\": $SEC6_EXIT,"
    echo "  \"sec7_exit\": $SEC7_EXIT"
    echo "}"
} > "${MANIFEST}.new" && mv "${MANIFEST}.new" "$MANIFEST"

# Overall exit: 0 only if every run we actually attempted succeeded.
OVERALL=0
[ "$SKIP_SEC6" != "1" ] && [ "$SEC6_EXIT" != "0" ] && OVERALL=$SEC6_EXIT
[ "$SKIP_SEC7" != "1" ] && [ "$SEC7_EXIT" != "0" ] && OVERALL=$SEC7_EXIT
exit $OVERALL
