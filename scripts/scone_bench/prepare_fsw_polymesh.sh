#!/usr/bin/env bash
#
# One-shot preparation: convert one or more JAXTrace VTU/PVTU meshes into
# OpenFOAM polyMesh directories that SCONE can consume, and stage them
# under scone_meshes/ inside the JAXTrace root for the SCONE benchmark
# driver.
#
# Usage:
#   scripts/scone_bench/prepare_fsw_polymesh.sh <mesh.vtu-or-pvtu> [<name>]
#     - <name> defaults to the input file stem
#
# Produces:
#   ./scone_meshes/<name>/
#     ├── points          (OpenFOAM ASCII polyMesh)
#     ├── faces
#     ├── owner
#     ├── neighbour
#     └── cellZones
#
# Prints the polyMesh path so it can be piped straight into run_pointloc.sh.
#
# Notes:
#   - SCONE has a hard-coded pathLen=100 (SharedModules/numPrecision.f90),
#     so we symlink into scone_meshes/ with a short name.
#   - JAXTrace's bridge (jaxtrace.io.openfoam_polymesh) handles VTU tets
#     directly and VTU polyhedra via the VTK_POLYHEDRON path.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <mesh.vtu-or-pvtu> [<short-name>]" >&2
  exit 2
fi

# Workstation venv with numpy + vtk. Override via env if needed.
JAXTRACE_VENV="${JAXTRACE_VENV:-/flash/shared/jax/.venv}"
if [[ -f "${JAXTRACE_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${JAXTRACE_VENV}/bin/activate"
fi

SRC="$(readlink -f "$1")"
if [[ ! -f "$SRC" ]]; then
  echo "ERROR: $SRC does not exist" >&2
  exit 2
fi

NAME="${2:-$(basename "$SRC")}"
NAME="${NAME%.*}"                     # strip extension
NAME="${NAME//[^A-Za-z0-9_-]/_}"      # sanitise

# Repo root is two levels up from this script
REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
OUT_DIR="${REPO_ROOT}/scone_meshes/${NAME}"
mkdir -p "${REPO_ROOT}/scone_meshes"

echo "=== prepare_fsw_polymesh.sh"
echo "    src : $SRC"
echo "    out : $OUT_DIR"

# Convert via the bridge CLI
"${REPO_ROOT}/scripts/openfoam_polymesh_convert.py" \
    --from-vtu "$SRC" \
    --to-polymesh "$OUT_DIR"

echo
echo "polyMesh directory ready. To run SCONE on it:"
echo "    scripts/scone_bench/run_pointloc.sh $OUT_DIR patchSingle 2000 4 1"
echo "    scripts/scone_bench/sweep_pointloc.sh $OUT_DIR 2000 4"
echo
echo "$OUT_DIR"
