#!/usr/bin/env bash
#
# Smoke test the newly-built RTXAdvect against Kim et al.'s smallest tet
# mesh (FinalFuelPinTet63).  Runs 100 particles for 1 step with a
# zero-velocity field so we exercise the host-cell locator + BVH init
# + particle seeding without needing a real advection loop.
#
# Prints wall time + tail of stdout so we can see exactly what the tool
# emits (needed to write the results parser for the sweep).

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
cd "$REPO_ROOT"

EXE="$REPO_ROOT/3rdParty/RTXAdvect/build/cudaParticleAdvection"
OWL_LIB_DIR="$REPO_ROOT/3rdParty/RTXAdvect/build/owl/owl"
export LD_LIBRARY_PATH="$OWL_LIB_DIR:${LD_LIBRARY_PATH:-}"

# Venv for the VTU→verts.dat converter
JAXTRACE_VENV="${JAXTRACE_VENV:-/flash/shared/jax/.venv}"
if [[ -f "$JAXTRACE_VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$JAXTRACE_VENV/bin/activate"
fi

# Convert the smallest Kim mesh
KIM_ROOT="/flash/shared/jax/ii1o33-SCONE-00d7668/IntegrationTestFiles/Geometry/Meshes/OpenFOAM"
MESH_NAME="FinalFuelPinTet63"
VTU="$REPO_ROOT/kim_meshes_vtu/${MESH_NAME}.vtu"
STAGE="$REPO_ROOT/rtxadvect_meshes/${MESH_NAME}"

if [[ ! -f "$VTU" ]]; then
  echo "== converting $MESH_NAME polyMesh → VTU"
  python3 scripts/openfoam_polymesh_convert.py \
      --from-polymesh "$KIM_ROOT/$MESH_NAME" \
      --to-vtu "$VTU"
fi

echo "== converting VTU → RTXAdvect (verts.dat + cells.dat + zero_velocity.dat)"
python3 scripts/scone_bench/vtu_to_rtxadvect.py \
    --vtu "$VTU" --out-prefix "$STAGE"

# Read the seeding-box from the meta.json
BBOX=$(python3 -c "
import json; d = json.load(open('${STAGE}.meta.json'))
print(d['seeding_box_flag'])
")

OUT_DIR="$REPO_ROOT/rtxadvect_smoke_${MESH_NAME}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo
echo "=========================================================="
echo "== RTXAdvect smoke test"
echo "=========================================================="
echo "  exe        : $EXE"
echo "  mesh       : $MESH_NAME (63 tets)"
echo "  seeding box: $BBOX"
echo "  particles  : 100"
echo "  steps      : 1"
echo "=========================================================="
echo

T0=$(date +%s.%N)
"$EXE" \
    --num-particles 100 --num-steps 1 \
    --input_mesh "${STAGE}.verts.dat" "${STAGE}.cells.dat" \
    --input_tet_velocity_field "${STAGE}.zero_velocity.dat" \
    -dt 1e-6 \
    --seeding-box $BBOX \
    --save-streamline-to-vtk "smoke_streamline.vtk" 2>&1 | tee smoke.log
ec=${PIPESTATUS[0]}
T1=$(date +%s.%N)
WALL=$(python3 -c "print(f'{${T1}-${T0}:.3f}')")

echo
echo "=========================================================="
echo "== smoke test done"
echo "  exit code : $ec"
echo "  wall time : ${WALL} s"
echo "  log       : $OUT_DIR/smoke.log"
if [[ -f "$OUT_DIR/smoke_streamline.vtk" ]]; then
  echo "  streamline VTK: $OUT_DIR/smoke_streamline.vtk ($(du -h "$OUT_DIR/smoke_streamline.vtk" | cut -f1))"
fi
echo "=========================================================="
