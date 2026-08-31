#!/usr/bin/env bash
#
# Test whether SCONE's octree "Ray has lost correct material" fatal is a
# float-precision ray-leakage issue at the mesh boundary. We rerun octree
# on one fuel-pin mesh with progressively larger BOX_PAD values (halfwidth =
# max(|xmin|,|xmax|) * BOX_PAD). If a large-enough pad avoids the failure,
# it's a float-boundary issue in SCONE that we can quote in the response
# letter; if octree still fails at BOX_PAD=10, it is a genuine SCONE octree
# bug independent of our input.
#
# Usage:
#   scripts/scone_bench/sweep_octree_pad.sh [mesh_name] [pop] [cycles]
#
# Defaults: FinalFuelPinTet298, pop=2000, cycles=4

set -euo pipefail

SCRIPT_DIR="$(readlink -f "$(dirname "$0")")"
DRIVER="${SCRIPT_DIR}/run_pointloc.sh"

MESH_NAME="${1:-FinalFuelPinTet298}"
POP="${2:-2000}"
CYCLES="${3:-4}"

POLYMESH="/flash/shared/jax/ii1o33-SCONE-00d7668/IntegrationTestFiles/Geometry/Meshes/OpenFOAM/${MESH_NAME}"

# Unbuffered tee-log of the sweep
if [[ "${SWEEP_LOG_ACTIVE:-0}" != "1" ]]; then
  REPO_ROOT="$(readlink -f "${SCRIPT_DIR}/../..")"
  LOG_DIR="${REPO_ROOT}/sweep_logs"
  mkdir -p "$LOG_DIR"
  LOG_FILE="${LOG_DIR}/octree_pad_${MESH_NAME}_$(date +%Y%m%d_%H%M%S).log"
  echo "logging to $LOG_FILE"
  export SWEEP_LOG_ACTIVE=1
  exec stdbuf -oL -eL bash -c "'$0' $*" 2>&1 | tee "$LOG_FILE"
fi

# Wipe any previous octree pad-sweep runs on this mesh
rm -rf "$(pwd)/scone_runs/${MESH_NAME}__octree_pad"*

for PAD in 1.05 1.5 2.0 5.0 10.0; do
  # Create a labelled OUT_DIR by symlinking under a pad-tagged path.
  # Easier: rename OUT_DIR after each run by tweaking BOX_PAD env + also
  # making sure the driver's dir naming is unique per pad.
  echo
  echo "############################################################"
  echo "# octree omp=1  BOX_PAD=${PAD}  mesh=${MESH_NAME}"
  echo "############################################################"
  BOX_PAD="$PAD" "$DRIVER" "$POLYMESH" octree "$POP" "$CYCLES" 1 || true

  # Move the just-produced OUT_DIR to a pad-tagged name so successive
  # pad values don't clobber each other.
  SRC="$(pwd)/scone_runs/${MESH_NAME}__octree__omp1"
  DST="$(pwd)/scone_runs/${MESH_NAME}__octree_pad${PAD}__omp1"
  if [[ -d "$SRC" ]]; then
    rm -rf "$DST"
    mv "$SRC" "$DST"
  fi
done

echo
echo "=== octree pad sweep complete ==="
echo "Per-pad status:"
for d in $(pwd)/scone_runs/${MESH_NAME}__octree_pad*__omp1; do
  if [[ ! -d "$d" ]]; then continue; fi
  pad="$(basename "$d" | sed -E 's/.*octree_pad([0-9.]+)__omp1/\1/')"
  # Was there a SIGSEGV, a SCONE fatal, or clean exit?
  if grep -q "SIGSEGV" "$d/stdout.log" 2>/dev/null; then
    status="SEGV"
  elif grep -q "Fatal has occurred" "$d/stdout.log" 2>/dev/null; then
    status="FAIL"
    reason=$(grep -A2 "Because:" "$d/stdout.log" | tail -1 | sed 's/^ *//')
    status="FAIL ($reason)"
  else
    status="OK"
  fi
  echo "  pad=${pad}  ${status}"
done
