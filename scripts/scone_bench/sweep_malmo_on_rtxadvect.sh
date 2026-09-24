#!/usr/bin/env bash
#
# Run MALMO point-location on the benchmark meshes distributed with the
# RTXAdvect reference implementation (Wang et al. 2022, CPC 271:108221):
#   - microfludics/solution_1.vtu (~41 k tets)   -- microfluidic channel, tightly packed sliver tets
#   - microfludics/solution_4.vtu (~1.66 M tets) -- large refinement of the same geometry
#   - porousMedia/solution_porousmedia.vtu (~820 k tets) -- pore-scale flow, high dynamic range
#
# The RTXAdvect distribution ships single VTU files with cell-centred
# Pressure / Velocity fields.  bench_malmo_pointloc.py already accepts a
# plain VTU via --vtu, so no format conversion is needed.
#
# Usage:
#   scripts/scone_bench/sweep_malmo_on_rtxadvect.sh [n_points]
#
# Default n_points = 100000.

set -euo pipefail

N_POINTS="${1:-100000}"

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"

# RTXAdvect clone (already cloned as a submodule / vendored under 3rdParty).
RTX_DATA="${RTXADVECT_DATA:-/flash/shared/jax/JAXTrace/3rdParty/RTXAdvect/dataset}"

# Optional tee-log for the full sweep.
if [[ "${SWEEP_LOG_ACTIVE:-0}" != "1" ]]; then
  LOG_DIR="${REPO_ROOT}/sweep_logs"
  mkdir -p "$LOG_DIR"
  LOG_FILE="${LOG_DIR}/malmo_on_rtxadvect_$(date +%Y%m%d_%H%M%S).log"
  echo "logging to $LOG_FILE"
  export SWEEP_LOG_ACTIVE=1
  exec stdbuf -oL -eL bash -c "'$0' $*" 2>&1 | tee "$LOG_FILE"
fi

JAXTRACE_VENV="${JAXTRACE_VENV:-/flash/shared/jax/.venv}"
if [[ -f "${JAXTRACE_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${JAXTRACE_VENV}/bin/activate"
  echo "activated venv: ${JAXTRACE_VENV}"
  python3 -c "import numpy, vtk; print(f'  numpy={numpy.__version__}, vtk={vtk.VTK_VERSION}')" || {
    echo "ERROR: venv activated but numpy/vtk still not importable" >&2
    exit 3
  }
else
  echo "WARNING: JAXTRACE_VENV=${JAXTRACE_VENV} not found; using system python3" >&2
fi
PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1

MALMO_TIMEOUT="${MALMO_TIMEOUT:-900}"

if [[ ! -d "$RTX_DATA" ]]; then
  echo "ERROR: RTXAdvect dataset directory not found: $RTX_DATA" >&2
  exit 2
fi

# name -> vtu-path table (Bash 3-compatible).
MESH_NAMES=(microfluidics_41k microfluidics_1p66M porousMedia_820k)
MESH_PATHS=(
  "$RTX_DATA/microfludics/solution_1.vtu"
  "$RTX_DATA/microfludics/solution_4.vtu"
  "$RTX_DATA/porousMedia/solution_porousmedia.vtu"
)

mkdir -p "$REPO_ROOT/malmo_runs_rtxadvect"

for i in "${!MESH_NAMES[@]}"; do
  mesh="${MESH_NAMES[$i]}"
  VTU="${MESH_PATHS[$i]}"

  if [[ ! -f "$VTU" ]]; then
    echo "  (missing $VTU; skipping $mesh)"
    continue
  fi

  for variant in vertex_multi centroid aabb; do
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
    ec=$?
    tail -60 "$RUN_LOG"
    if [[ $ec -ne 0 ]]; then
      if [[ $ec -eq 124 || $ec -eq 137 ]]; then
        echo "  (MALMO $variant TIMED OUT after ${MALMO_TIMEOUT}s on $mesh; continuing)"
        cat > "$OUT/result.json" <<STUBJSON
{
  "mesh": "$mesh",
  "method": "MALMO_${variant}",
  "n_points_queried": $N_POINTS,
  "found_rate": null,
  "wall_seconds": {"query_min_of_3": null},
  "throughput_queries_per_second": null,
  "process": {"max_rss_kb": null},
  "timed_out": true,
  "timeout_seconds": ${MALMO_TIMEOUT}
}
STUBJSON
      else
        echo "  (MALMO $variant FAILED (ec=$ec) on $mesh; continuing)"
        cat > "$OUT/result.json" <<STUBJSON
{
  "mesh": "$mesh",
  "method": "MALMO_${variant}",
  "n_points_queried": $N_POINTS,
  "found_rate": null,
  "wall_seconds": {"query_min_of_3": null},
  "throughput_queries_per_second": null,
  "process": {"max_rss_kb": null},
  "failed": true,
  "exit_code": $ec
}
STUBJSON
      fi
    fi
  done
done

echo
echo "=== MALMO-on-RTXAdvect sweep complete ==="
echo "results under: $REPO_ROOT/malmo_runs_rtxadvect/"
