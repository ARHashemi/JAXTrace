#!/usr/bin/env bash
#
# Fill in the two intermediate microfluidics mesh sizes (solution_2, solution_3)
# for the mesh-size scaling curve.  MALMO-aabb only (the winning variant on this
# geometry; vertex_multi is known to fail on flat/extruded meshes and centroid
# tracks aabb closely).
#
# Usage:
#   scripts/scone_bench/sweep_malmo_microfluidics_scaling.sh [n_points]

set -euo pipefail

N_POINTS="${1:-100000}"

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
RTX_DATA="${RTXADVECT_DATA:-/flash/shared/jax/JAXTrace/3rdParty/RTXAdvect/dataset}"

JAXTRACE_VENV="${JAXTRACE_VENV:-/flash/shared/jax/.venv}"
if [[ -f "${JAXTRACE_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${JAXTRACE_VENV}/bin/activate"
fi
PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1

MALMO_TIMEOUT="${MALMO_TIMEOUT:-900}"

MESH_NAMES=(microfluidics_199k microfluidics_698k)
MESH_PATHS=(
  "$RTX_DATA/microfludics/solution_2.vtu"
  "$RTX_DATA/microfludics/solution_3.vtu"
)

mkdir -p "$REPO_ROOT/malmo_runs_rtxadvect"

for i in "${!MESH_NAMES[@]}"; do
  mesh="${MESH_NAMES[$i]}"
  VTU="${MESH_PATHS[$i]}"

  [[ -f "$VTU" ]] || { echo "missing $VTU"; continue; }

  for variant in aabb centroid; do
    OUT="$REPO_ROOT/malmo_runs_rtxadvect/${mesh}__${variant}"
    echo
    echo "############################################################"
    echo "# MALMO $variant on $mesh  (n_points=$N_POINTS)"
    echo "############################################################"
    mkdir -p "$OUT"
    RUN_LOG="$OUT/run.log"
    timeout --signal=TERM --kill-after=30s "$MALMO_TIMEOUT" \
         "$PYTHON" "$REPO_ROOT/scripts/scone_bench/bench_malmo_pointloc.py" \
         --vtu "$VTU" --variant "$variant" --n-points "$N_POINTS" \
         --out-dir "$OUT" > "$RUN_LOG" 2>&1
    tail -20 "$RUN_LOG"
  done
done
echo
echo "=== microfluidics scaling infill complete ==="
