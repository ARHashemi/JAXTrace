#!/usr/bin/env bash
#
# Run a single MALMO (mesh, variant) with a configurable timeout, capturing
# the run log. Useful for chasing individual pathological configurations
# (e.g. centroid/aabb on StanfordBunny where XLA compile is slow) without
# rerunning the whole sweep.
#
# Usage:
#   scripts/scone_bench/run_malmo_single.sh \
#       <vtu_path> <variant> [n_points=100000] [out_dir=./malmo_runs/<mesh>__<variant>]
#
# Environment:
#   MALMO_TIMEOUT   seconds; default 3600 (1 h) — much longer than the sweep default
#                    so we can wait out slow XLA compiles honestly.
#   JAXTRACE_VENV   default /flash/shared/jax/.venv

set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <vtu_path> <variant> [n_points] [out_dir]" >&2
  echo "  variant ∈ {vertex_multi, centroid, aabb}" >&2
  exit 2
fi

VTU="$(readlink -f "$1")"
VARIANT="$2"
N_POINTS="${3:-100000}"

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"

MESH_NAME="$(basename "$VTU")"
MESH_NAME="${MESH_NAME%.*}"

OUT="${4:-${REPO_ROOT}/malmo_runs/${MESH_NAME}__${VARIANT}}"

JAXTRACE_VENV="${JAXTRACE_VENV:-/flash/shared/jax/.venv}"
if [[ -f "${JAXTRACE_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${JAXTRACE_VENV}/bin/activate"
  echo "activated venv: ${JAXTRACE_VENV}"
fi
PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1

MALMO_TIMEOUT="${MALMO_TIMEOUT:-3600}"
mkdir -p "$OUT"
RUN_LOG="$OUT/run.log"

echo "=== run_malmo_single.sh"
echo "    vtu     : $VTU"
echo "    variant : $VARIANT"
echo "    n_pts   : $N_POINTS"
echo "    out_dir : $OUT"
echo "    timeout : ${MALMO_TIMEOUT} s"
echo "    log     : $RUN_LOG"
echo

# Unbuffered log to disk + terminal, no pipe swallow of exit code
timeout --signal=TERM --kill-after=60s "$MALMO_TIMEOUT" \
    stdbuf -oL -eL "$PYTHON" \
    "$REPO_ROOT/scripts/scone_bench/bench_malmo_pointloc.py" \
    --vtu "$VTU" --variant "$VARIANT" --n-points "$N_POINTS" \
    --out-dir "$OUT" 2>&1 | tee "$RUN_LOG"
ec=${PIPESTATUS[0]}

echo
if [[ $ec -eq 0 ]]; then
  echo "=== OK. result.json: $OUT/result.json"
elif [[ $ec -eq 124 || $ec -eq 137 ]]; then
  echo "=== TIMED OUT after ${MALMO_TIMEOUT} s (ec=$ec)"
  cat > "$OUT/result.json" <<STUBJSON
{
  "mesh": "${MESH_NAME}",
  "method": "MALMO_${VARIANT}",
  "n_points_queried": ${N_POINTS},
  "found_rate": null,
  "wall_seconds": {"query_min_of_3": null},
  "throughput_queries_per_second": null,
  "process": {"max_rss_kb": null},
  "timed_out": true,
  "timeout_seconds": ${MALMO_TIMEOUT}
}
STUBJSON
else
  echo "=== FAILED with ec=$ec"
fi

exit $ec
