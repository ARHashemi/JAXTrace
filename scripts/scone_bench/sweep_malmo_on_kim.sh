#!/usr/bin/env bash
#
# For each bundled Kim et al. polyMesh, convert to VTU and run MALMO
# point-location. Companion to the SCONE sweep so the response letter has
# both sides of the comparison on the same meshes.
#
# Usage:
#   scripts/scone_bench/sweep_malmo_on_kim.sh [n_points]
#
# Default n_points=100000.

set -euo pipefail

N_POINTS="${1:-100000}"

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
KIM_MESHES="/flash/shared/jax/ii1o33-SCONE-00d7668/IntegrationTestFiles/Geometry/Meshes/OpenFOAM"

# Unbuffered tee-log of everything the sweep prints (relaunch trick).
# When SWEEP_LOG_ACTIVE=1 is set we skip re-entry — the child process is the
# one actually writing to disk in parallel with the terminal.
if [[ "${SWEEP_LOG_ACTIVE:-0}" != "1" ]]; then
  LOG_DIR="${REPO_ROOT}/sweep_logs"
  mkdir -p "$LOG_DIR"
  LOG_FILE="${LOG_DIR}/malmo_on_kim_$(date +%Y%m%d_%H%M%S).log"
  echo "logging to $LOG_FILE"
  export SWEEP_LOG_ACTIVE=1
  # stdbuf -oL -eL forces line-buffered stdout+stderr for the whole subtree
  exec stdbuf -oL -eL bash -c "'$0' $*" 2>&1 | tee "$LOG_FILE"
fi

# Workstation venv with numpy + vtk + jax. Override via env if needed.
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

# Per-(mesh,variant) time budget (seconds). Default 10 min; can be raised for
# meshes/variants with pathologically-long XLA compile times (e.g.
# MALMO centroid/aabb on StanfordBunny, where the mesh's Kuhn-level span is
# large so the search's fori_loop unrolls into a big graph).
MALMO_TIMEOUT="${MALMO_TIMEOUT:-600}"

if [[ ! -d "$KIM_MESHES" ]]; then
  echo "ERROR: Kim mesh directory not found: $KIM_MESHES" >&2
  exit 2
fi

# The meshes we ran SCONE on (see scone_runs/)
MESHES=(
  FinalFuelPinTet63
  FinalFuelPinTet137
  FinalFuelPinTet298
  FinalFuelPinTet1820
  FinalFuelPinTet2856
  StanfordBunny_LowPoly
  FinalFuelPinPoly264
  FinalFuelPinPoly940
  FinalFuelPinPoly1560
)

mkdir -p "$REPO_ROOT/kim_meshes_vtu"
mkdir -p "$REPO_ROOT/malmo_runs"

for mesh in "${MESHES[@]}"; do
  SRC="$KIM_MESHES/$mesh"
  VTU="$REPO_ROOT/kim_meshes_vtu/$mesh.vtu"

  if [[ ! -f "$VTU" ]]; then
    echo
    echo "=== converting $mesh to VTU"
    "$REPO_ROOT/scripts/openfoam_polymesh_convert.py" \
        --from-polymesh "$SRC" \
        --to-vtu "$VTU" || {
      echo "  (conversion failed for $mesh; skipping)"
      continue
    }
  fi

  for variant in vertex_multi centroid aabb; do
    OUT="$REPO_ROOT/malmo_runs/${mesh}__${variant}"
    echo
    echo "############################################################"
    echo "# MALMO $variant on $mesh   (n_points=$N_POINTS)"
    echo "############################################################"
    mkdir -p "$OUT"
    # Hard per-(mesh,variant) timeout. Pathological configurations (e.g.
    # centroid/aabb on non-octree-aligned meshes) may take longer than
    # any interesting workload; we prefer to record "timed out" and continue.
    #
    # NOTE: the exit code of a pipeline is the LAST command's exit code, so
    # we can't pipe through 'tail' and still read timeout's ec.  Log to a
    # file, then tail the file after the fact.
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
  "n_found": null,
  "mean_pit_tests": null,
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
echo "=== MALMO-on-Kim sweep complete ==="
echo "aggregate with:"
echo "    ls -1 $REPO_ROOT/malmo_runs/*/result.json | while read f; do"
echo "      python3 -c \"import json; d=json.load(open('\$f')); \\"
echo "        print(f\\\"{d['mesh']:<28} {d['method']:<20} \\"
echo "              rate={d['found_rate']*100:5.1f}%  q={d['n_points_queried']}  \\"
echo "              t={d['wall_seconds']['query_min_of_3']*1000:8.2f} ms  \\"
echo "              tput={d['throughput_queries_per_second']/1e6:6.1f} Mq/s\\\")\""
echo "    done"
