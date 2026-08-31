#!/usr/bin/env bash
#
# Re-run RTXAdvect on the StanfordBunny mesh with IN-MESH particle
# sampling (barycentric-in-tet) instead of the axis-aligned bbox
# sampling that gave the misleading 27.24% found rate.  Now that
# Phase 14 wired --input-particles, the exe can read a particle file
# directly.
#
# Requires: Phase 14 patch applied (port_rtxadvect_step14.sh).
#
# Usage:
#   scripts/scone_bench/rerun_rtxadvect_bunny_inmesh.sh [n_particles] [n_steps]
# Defaults: 1000000 particles, 100 steps (matches the overnight scale sweep).

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
cd "$REPO_ROOT"

N_PART="${1:-1000000}"
N_STEPS="${2:-100}"
MESH_NAME="StanfordBunny_LowPoly"
VTU="$REPO_ROOT/kim_meshes_vtu/${MESH_NAME}.vtu"

EXE="$REPO_ROOT/3rdParty/RTXAdvect/build/cudaParticleAdvection"
OWL_LIB_DIR="$REPO_ROOT/3rdParty/RTXAdvect/build/owl/owl"
export LD_LIBRARY_PATH="$OWL_LIB_DIR:${LD_LIBRARY_PATH:-}"

JAXTRACE_VENV="${JAXTRACE_VENV:-/flash/shared/jax/.venv}"
if [[ -f "$JAXTRACE_VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$JAXTRACE_VENV/bin/activate"
fi

STAGE_DIR="$REPO_ROOT/rtxadvect_meshes"
mkdir -p "$STAGE_DIR"

META="${STAGE_DIR}/${MESH_NAME}.meta.json"
if [[ ! -f "$META" ]]; then
  echo "== converting VTU → RTXAdvect mesh"
  python3 scripts/scone_bench/vtu_to_rtxadvect.py \
      --vtu "$VTU" --out-prefix "${STAGE_DIR}/${MESH_NAME}"
fi

verts=$(python3 -c "import json; print(json.load(open('$META'))['verts_dat'])")
cells=$(python3 -c "import json; print(json.load(open('$META'))['cells_dat'])")
zvel=$(python3  -c "import json; print(json.load(open('$META'))['zero_velocity_dat'])")

# Generate in-mesh particles (barycentric-in-tet, ground-truth tet id)
PARTICLES="${STAGE_DIR}/${MESH_NAME}.inmesh_${N_PART}.particles.dat"
echo
echo "== generating ${N_PART} in-mesh particles"
python3 scripts/scone_bench/vtu_to_rtxadvect_particles.py \
    --vtu "$VTU" -n "$N_PART" --out "$PARTICLES" --seed 42

# Run
OUT="$REPO_ROOT/rtxadvect_runs_inmesh/${MESH_NAME}"
mkdir -p "$OUT"
RUN_LOG="$OUT/run.log"

echo
echo "=========================================================="
echo "== RTXAdvect  IN-MESH  test on $MESH_NAME"
echo "=========================================================="
echo "  exe        : $EXE"
echo "  mesh       : $MESH_NAME  ($VTU)"
echo "  particles  : $N_PART  (in-mesh barycentric)"
echo "  steps      : $N_STEPS"
echo "=========================================================="
echo

T0=$(date +%s.%N)
( cd "$OUT" &&
  "$EXE" \
      --num-particles "$N_PART" --num-steps "$N_STEPS" \
      --input_mesh "$verts" "$cells" \
      --input_tet_velocity_field "$zvel" \
      -dt 1e-3 \
      --input-particles "$PARTICLES" \
      --save-streamline-to-vtk "streamline.vtk" ) > "$RUN_LOG" 2>&1
ec=$?
T1=$(date +%s.%N)
WALL=$(python3 -c "print(f'{${T1}-${T0}:.6f}')")

# Parse into a result.json
python3 scripts/scone_bench/parse_rtxadvect_log.py \
    --log "$RUN_LOG" --meta "$META" \
    --out "$OUT/result.json" \
    --n-particles "$N_PART" --n-steps "$N_STEPS" \
    --wall-seconds "$WALL"

# Tag it as in-mesh so build_joint_table.py can distinguish
python3 -c "
import json
d = json.load(open('$OUT/result.json'))
d['sampling'] = 'in_mesh_barycentric'
d['shell_exit_code'] = $ec
json.dump(d, open('$OUT/result.json', 'w'), indent=2)
"

echo
echo "=========================================================="
echo "== RTXAdvect in-mesh done"
echo "  exit code : $ec"
echo "  wall time : ${WALL} s"
echo "  log       : $RUN_LOG"
echo "  result    : $OUT/result.json"
echo "=========================================================="
echo
echo "--- key numbers ---"
grep -E "NumParticles|Out-of-domain|Simulation RunTime|Simulation Performance|Init RunTime|BVH Construction|Total Time|Performance =" "$RUN_LOG"
