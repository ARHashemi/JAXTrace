#!/usr/bin/env bash
#
# Submit the two confirmation tests for the "wrong grid pitch" hypothesis.
#
# Why these two
# -------------
# Established so far (all measured):
#   * D2 has cells of two DIFFERENT sizes sharing one octree level, a factor of
#     2 apart, in 99.86% of cells. A1 has 0.00% — only float round-off.
#   * A1 loses zero particles; D2 loses thousands.
#   * Registration coverage is adequate for the lost particles (audit: 150/150
#     had their true host in a visited cell).
#   * `aabb` registration cuts the loss 8,027 -> 3,540 (56%) but cannot
#     eliminate it, which is what you expect if the INDEX is wrong rather than
#     the registration.
#
# Not yet established: that the search actually computes an out-of-neighbourhood
# cell index FOR THE PARTICLES IT LOSES. That is what test 2 settles, and it is
# the decisive one. Test 1 explains WHY the sizes mix (anisotropy), which
# determines which fix is appropriate.
#
# Test 1 — why_factor2: per level, the distinct (dx,dy,dz) triples present and
#          the per-cell anisotropy. Confirms (or refutes) that `level` collapses
#          anisotropic cells of different size onto one level, because it is
#          derived from mean([dx,dy,dz]).
#
# Test 2 — test_index_mismatch: THE DECISIVE TEST. Per particle, compares the
#          index the search uses, floor(pos / level_cell_sizes[level]), against
#          the stored grid index of the cell its true host is registered in.
#          Lost particles are compared against an equal-sized control sample of
#          SURVIVORS, so a high mismatch rate only counts if the survivors do
#          not share it.
#
# Both are CPU-only (no GPU), pure NumPy, JAX pinned to the CPU backend.
# They need ~240 GB for a 10.8M-element D mesh, so they must be batch jobs:
# the LUMI login-node cgroup caps per-user memory at 96 GB and the octree build
# dies with MemoryError there.
#
# Usage
#   scripts/dfamily/sbatch_confirm_tests.sh              # submit both, D2 + A1
#   scripts/dfamily/sbatch_confirm_tests.sh --dry-run    # print, submit nothing
#   scripts/dfamily/sbatch_confirm_tests.sh --only index # just the decisive one
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

PROJECT="${PROJECT:-project_465002752}"
PROJECT="$(basename "$PROJECT")"
USER_NAME="${USER:-hashemia}"

JAXTRACE="${JAXTRACE:-/projappl/${PROJECT}/${USER_NAME}/JAXTrace}"
PKGS="/projappl/${PROJECT}/${USER_NAME}/required-packages"
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif
OUTDIR="/scratch/${PROJECT}/${USER_NAME}/dfamily"
LOGS="/scratch/${PROJECT}/${USER_NAME}/logs"

CASES_ROOT="/scratch/${PROJECT}/lorenzgl/Cases/PinShapes"
D2="${CASES_ROOT}/D-ConcavityTilt/D2.gid"
A1="${CASES_ROOT}/A-FlatsVariations/A1.gid"

# The two tracking runs to test. parent_cube is the production default and the
# one with the big loss; aabb is the mitigated run, useful to check whether its
# RESIDUAL losses are also index mismatches.
RUN_PC="${OUTDIR}/sweep_D2/parent_cube_results"
RUN_AABB="${OUTDIR}/sweep_D2/aabb_results"

DRY=0
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --only)    ONLY="$2"; shift 2 ;;
    *) echo "ERROR: unknown option '$1'" >&2; exit 2 ;;
  esac
done

want() { [ -z "$ONLY" ] || printf '%s' ",$ONLY," | grep -q ",$1,"; }

submit() {
  local name="$1" mem="$2" tlimit="$3" script="$4"; shift 4
  local wrap="srun singularity exec --cleanenv \
--env PYTHONPATH=${JAXTRACE}:${PKGS} \
--env JAX_PLATFORMS=cpu --env TF_CPP_MIN_LOG_LEVEL=2 \
${SIF} python3 -u ${script} $*"
  if [ "$DRY" = "1" ]; then
    echo "--- would submit: $name (mem=$mem time=$tlimit)"
    echo "    $script $*"
    return
  fi
  local id
  id=$(sbatch --parsable \
    --job-name="$name" --account="$PROJECT" --partition=small \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem="$mem" --time="$tlimit" \
    --output="${LOGS}/${name}_%j.out" --error="${LOGS}/${name}_%j.err" \
    --wrap="$wrap")
  echo "  submitted $name as job $id"
}

mkdir -p "$LOGS" "$OUTDIR"

echo "Confirmation tests for the wrong-grid-pitch hypothesis"
echo "  repo    : $JAXTRACE"
echo "  outputs : $OUTDIR"
echo

# ---- Test 1: why do sizes mix within a level? -------------------------------
if want why; then
  echo "[test 1] anisotropy / distinct size-triples per level"
  submit "aniso_D2" 240G 01:30:00 \
    "${JAXTRACE}/scripts/dfamily/why_factor2.py" "$D2"
  submit "aniso_A1" 240G 01:30:00 \
    "${JAXTRACE}/scripts/dfamily/why_factor2.py" "$A1"
  echo
fi

# ---- Test 2: THE DECISIVE ONE ------------------------------------------------
if want index; then
  echo "[test 2] index mismatch, lost vs surviving particles (DECISIVE)"
  if [ -d "$RUN_PC" ]; then
    submit "idxmis_pc" 240G 02:00:00 \
      "${JAXTRACE}/scripts/dfamily/test_index_mismatch.py" \
      --run "$RUN_PC" --case "$D2" --registration parent_cube \
      --max-particles 300 \
      --out "${OUTDIR}/05_D2_index_mismatch_parent_cube.md"
  else
    echo "  SKIP idxmis_pc: no results at $RUN_PC" >&2
  fi
  if [ -d "$RUN_AABB" ]; then
    submit "idxmis_aabb" 240G 02:00:00 \
      "${JAXTRACE}/scripts/dfamily/test_index_mismatch.py" \
      --run "$RUN_AABB" --case "$D2" --registration aabb \
      --max-particles 300 \
      --out "${OUTDIR}/05_D2_index_mismatch_aabb.md"
  else
    echo "  SKIP idxmis_aabb: no results at $RUN_AABB" >&2
  fi
  echo
fi

if [ "$DRY" = "1" ]; then
  echo "Dry run — nothing submitted."
else
  echo "Watch:   squeue -u $USER_NAME"
  echo "Reports: $OUTDIR/05_D2_index_mismatch_*.md"
  echo "Logs:    $LOGS/{aniso,idxmis}_*.out"
fi
