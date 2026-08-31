#!/usr/bin/env bash
#
# Sweep one polyMesh across the four SCONE point-location methods
# {none, octree, patchSingle, patchMulti} and both threading regimes
# {omp=1, omp=$(nproc)}. Each run produces its own scone_runs/... directory
# with input.txt, stdout.log, time.log, and result.json.
#
# Usage:
#   scripts/scone_bench/sweep_pointloc.sh <polymesh_dir> [pop] [cycles]
#
# Default pop=50000, cycles=10. Bump for larger meshes if you want more work
# per method to average out noise.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <polymesh_dir> [pop=50000] [cycles=10]" >&2
  exit 2
fi

POLYMESH="$1"
POP="${2:-50000}"
CYCLES="${3:-10}"

SCRIPT_DIR="$(readlink -f "$(dirname "$0")")"
DRIVER="${SCRIPT_DIR}/run_pointloc.sh"

# Unbuffered tee-log of the sweep
if [[ "${SWEEP_LOG_ACTIVE:-0}" != "1" ]]; then
  REPO_ROOT="$(readlink -f "${SCRIPT_DIR}/../..")"
  LOG_DIR="${REPO_ROOT}/sweep_logs"
  mkdir -p "$LOG_DIR"
  MESH_TAG="$(basename "$POLYMESH")"
  LOG_FILE="${LOG_DIR}/sweep_${MESH_TAG}_$(date +%Y%m%d_%H%M%S).log"
  echo "logging to $LOG_FILE"
  export SWEEP_LOG_ACTIVE=1
  exec stdbuf -oL -eL bash -c "'$0' $*" 2>&1 | tee "$LOG_FILE"
fi

NCPU="$(nproc)"

for OMP in 1 "$NCPU"; do
  for METHOD in none octree patchSingle patchMulti; do
    echo
    echo "############################################################"
    echo "#  method=${METHOD}  omp=${OMP}  pop=${POP}  cycles=${CYCLES}"
    echo "############################################################"
    if ! "$DRIVER" "$POLYMESH" "$METHOD" "$POP" "$CYCLES" "$OMP"; then
      echo "  (^ that run failed; continuing sweep)" >&2
    fi
  done
done

echo
echo "=== sweep complete ==="
echo "aggregate the results with:"
echo "    scripts/scone_bench/collect_results.py --root $(pwd)/scone_runs --csv $(pwd)/scone_runs/summary.csv"
