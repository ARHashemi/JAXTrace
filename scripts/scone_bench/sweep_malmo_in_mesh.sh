#!/usr/bin/env bash
#
# Follow-up MALMO sweep with in-mesh sampling: each query point is placed
# inside a known random tet via barycentric coordinates, and the true host
# element ID is recorded.  This lets us report BOTH:
#
#   found_rate         — MALMO returned some host (>=0)
#   correct_rate_strict — MALMO returned the SAME tet the point came from
#
# This is the same benchmark design used in the paper's Sec 6
# (sec6_validation.tex) and is what the reviewer's R2-3 question should
# be answered against.  Sampling uniformly in the AABB (the old default)
# under-reports Bunny's found_rate because 73% of the AABB is empty space.
#
# Usage:
#   scripts/scone_bench/sweep_malmo_in_mesh.sh [n_points]
#
# Env:
#   MALMO_TIMEOUT       default 1800 s per (mesh, variant)
#   FSW_PAPER_MESH      default cylA_119.pvtu; skip if not present
#   OUT_ROOT            default ./malmo_runs_in_mesh
#   JAXTRACE_VENV       default /flash/shared/jax/.venv

set -uo pipefail

N_POINTS="${1:-100000}"

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/malmo_runs_in_mesh}"
KIM_MESHES="/flash/shared/jax/ii1o33-SCONE-00d7668/IntegrationTestFiles/Geometry/Meshes/OpenFOAM"
FSW_PAPER_MESH="${FSW_PAPER_MESH:-/flash/users/ali/data/cylA.gid/post/0eule/cylA_119.pvtu}"
MALMO_TIMEOUT="${MALMO_TIMEOUT:-1800}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/sweep_logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/malmo_in_mesh_${TS}.log"

if [[ "${SWEEP_LOG_ACTIVE:-0}" != "1" ]]; then
  export SWEEP_LOG_ACTIVE=1
  echo "logging to $MAIN_LOG"
  exec stdbuf -oL -eL bash -c "'$0' $*" 2>&1 | tee "$MAIN_LOG"
fi

JAXTRACE_VENV="${JAXTRACE_VENV:-/flash/shared/jax/.venv}"
if [[ -f "${JAXTRACE_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${JAXTRACE_VENV}/bin/activate"
  python3 -c "import numpy, vtk, jax; print(f'  numpy={numpy.__version__}  vtk={vtk.VTK_VERSION}  jax={jax.__version__}')" || true
fi
PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1

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

echo "############################################################"
echo "# MALMO IN-MESH SAMPLING SWEEP"
echo "#  start   : $(date)"
echo "#  n_points: $N_POINTS"
echo "#  timeout : ${MALMO_TIMEOUT}s per (mesh, variant)"
echo "#  out     : $OUT_ROOT"
echo "############################################################"

mkdir -p "$REPO_ROOT/kim_meshes_vtu"
mkdir -p "$OUT_ROOT"

# ------- Kim meshes -------
for mesh in "${MESHES[@]}"; do
  SRC="$KIM_MESHES/$mesh"
  VTU="$REPO_ROOT/kim_meshes_vtu/$mesh.vtu"

  if [[ ! -f "$VTU" ]]; then
    echo; echo "=== converting $mesh to VTU"
    "$REPO_ROOT/scripts/openfoam_polymesh_convert.py" \
        --from-polymesh "$SRC" --to-vtu "$VTU" \
      || { echo "  (conversion failed for $mesh; skipping)"; continue; }
  fi

  for variant in vertex_multi centroid aabb; do
    OUT="$OUT_ROOT/${mesh}__${variant}"
    mkdir -p "$OUT"
    echo
    echo "############################################################"
    echo "# MALMO $variant on $mesh   (in-mesh, n=$N_POINTS)"
    echo "############################################################"
    RUN_LOG="$OUT/run.log"
    timeout --signal=TERM --kill-after=30s "$MALMO_TIMEOUT" \
        "$PYTHON" "$REPO_ROOT/scripts/scone_bench/bench_malmo_pointloc.py" \
        --vtu "$VTU" --variant "$variant" --n-points "$N_POINTS" \
        --sampling in_mesh \
        --out-dir "$OUT" > "$RUN_LOG" 2>&1
    ec=$?
    tail -60 "$RUN_LOG"
    if [[ $ec -ne 0 ]]; then
      if [[ $ec -eq 124 || $ec -eq 137 ]]; then
        echo "  (TIMED OUT after ${MALMO_TIMEOUT}s; continuing)"
        cat > "$OUT/result.json" <<STUBJSON
{"mesh":"$mesh","method":"MALMO_${variant}","sampling":"in_mesh","timed_out":true,"timeout_seconds":${MALMO_TIMEOUT},"n_points_queried":${N_POINTS}}
STUBJSON
      else
        echo "  (FAILED ec=$ec; continuing)"
      fi
    fi
  done
done

# ------- FSW paper mesh -------
if [[ -f "$FSW_PAPER_MESH" ]]; then
  echo
  echo "############################################################"
  echo "# FSW paper mesh: $FSW_PAPER_MESH"
  echo "############################################################"
  for variant in vertex_multi centroid aabb; do
    OUT="$OUT_ROOT/FSW_paper__${variant}"
    mkdir -p "$OUT"
    RUN_LOG="$OUT/run.log"
    echo
    echo "== MALMO $variant on FSW paper (in-mesh, n=$N_POINTS)"
    timeout --signal=TERM --kill-after=60s "$MALMO_TIMEOUT" \
        "$PYTHON" "$REPO_ROOT/scripts/scone_bench/bench_malmo_pointloc.py" \
        --vtu "$FSW_PAPER_MESH" --variant "$variant" --n-points "$N_POINTS" \
        --sampling in_mesh \
        --out-dir "$OUT" > "$RUN_LOG" 2>&1
    ec=$?
    tail -60 "$RUN_LOG"
    if [[ $ec -ne 0 ]]; then
      if [[ $ec -eq 124 || $ec -eq 137 ]]; then
        echo "  (TIMED OUT after ${MALMO_TIMEOUT}s; continuing)"
      else
        echo "  (FAILED ec=$ec; continuing)"
      fi
    fi
  done
else
  echo
  echo "== FSW_PAPER_MESH not found, skipping FSW paper mesh runs"
fi

# ------- Final table -------
echo
echo "############################################################"
echo "# IN-MESH SAMPLING RESULTS"
echo "############################################################"
ls "$OUT_ROOT"/*/result.json 2>/dev/null | while read f; do
  python3 -c "
import json
d = json.load(open('$f'))
mesh = d.get('mesh','?')[:24]
method = d.get('method','?').replace('MALMO_','')
if d.get('timed_out'): print(f'{mesh:<24} {method:<15} TIMED OUT'); exit()
if d.get('found_rate') is None: print(f'{mesh:<24} {method:<15} FAILED'); exit()
fr = d['found_rate']*100
cs = (d.get('correct_rate_strict') or 0)*100
fill = (d.get('fill_ratio') or 0)*100
qms = d['wall_seconds']['query_min_of_3']*1000
tput = d['throughput_queries_per_second']/1e6
pit = d.get('mean_pit_tests', 0) or 0
print(f'{mesh:<24} {method:<15} found={fr:6.2f}%  correct_strict={cs:6.2f}%  fill_bbox={fill:5.2f}%  t={qms:8.2f}ms  tput={tput:6.2f}Mq/s  PIT={pit:7.1f}')"
done

echo
echo "############################################################"
echo "# DONE   $(date)"
echo "############################################################"
