#!/usr/bin/env bash
#
# Complete the in-mesh-seeded RTXAdvect campaign with the FSW mesh.
#
# sweep_inmesh_rtxadvect.sh covers the four static VTU meshes but skips
# FSW, whose source is a time-dependent PVTU.  The FSW mesh is however
# already staged in RTXAdvect .dat form, so the run only needs an
# in-mesh particle file at the campaign's particle count and the same
# runner invocation the other four meshes used.
#
# Running this puts all five meshes in one campaign
# (rtxadvect_runs_inmesh_seeded/), so the paper's RTXAdvect timing row
# comes from a single protocol instead of three.
#
# Usage:
#   scripts/scone_bench/rerun_rtxadvect_fsw_inmesh.sh [n_particles] [n_steps]

set -uo pipefail

N_PART="${1:-100000}"
N_STEPS="${2:-1}"
SEED="${SEED:-42}"

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
cd "$REPO_ROOT"

EXE_CANDIDATES=(
  "$REPO_ROOT/3rdParty/RTXAdvect/build/cudaParticleAdvection"
  "$REPO_ROOT/3rdParty/RTXAdvect/build/bin/cudaParticleAdvection"
)
EXE=""
for c in "${EXE_CANDIDATES[@]}"; do
  [[ -x "$c" ]] && { EXE="$c"; break; }
done
[[ -n "$EXE" ]] || { echo "ERROR: RTXAdvect exe not found" >&2; exit 2; }
echo "using RTXAdvect executable: $EXE"

OWL_LIB_DIR="$REPO_ROOT/3rdParty/RTXAdvect/build/owl/owl"
[[ -f "$OWL_LIB_DIR/libowl.so" ]] && \
  export LD_LIBRARY_PATH="$OWL_LIB_DIR:${LD_LIBRARY_PATH:-}"

JAXTRACE_VENV="${JAXTRACE_VENV:-/flash/shared/jax/.venv}"
[[ -f "$JAXTRACE_VENV/bin/activate" ]] && . "$JAXTRACE_VENV/bin/activate"
PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1

OUT_ROOT="$REPO_ROOT/rtxadvect_runs_inmesh_seeded"
STAGE_DIR="$REPO_ROOT/rtxadvect_meshes"
PART_DIR="$REPO_ROOT/rtxadvect_particles"
mkdir -p "$OUT_ROOT" "$PART_DIR"

FSW_PVTU="${FSW_PVTU:-/flash/users/ali/data/cylA.gid/post/0eule/cylA_119.pvtu}"
meta="$STAGE_DIR/FSW_paper.meta.json"
parts="$PART_DIR/FSW.particles.dat"

[[ -f "$meta" ]] || {
  echo "ERROR: FSW mesh not staged: $meta" >&2
  echo "       stage it with vtu_to_rtxadvect.py first." >&2
  exit 2
}

echo
echo "############################################################"
echo "# RTXAdvect IN-MESH: FSW  (n_particles=$N_PART)"
echo "############################################################"

# In-mesh particle seeds: same sampler and seed as MALMO and as the
# other four meshes of this campaign.
if [[ ! -f "$parts" ]]; then
  [[ -f "$FSW_PVTU" ]] || { echo "ERROR: missing $FSW_PVTU" >&2; exit 2; }
  echo "== generating in-mesh particles for FSW"
  "$PYTHON" "$REPO_ROOT/scripts/scone_bench/make_inmesh_particles.py" \
      --vtu "$FSW_PVTU" --out "$parts" \
      --n-particles "$N_PART" --seed "$SEED" \
    || { echo "particle generation failed" >&2; exit 3; }
else
  echo "== reusing existing particle file: $parts"
fi

out="$OUT_ROOT/FSW"
mkdir -p "$out"
"$PYTHON" "$REPO_ROOT/scripts/scone_bench/bench_rtxadvect_single.py" \
    --exe "$EXE" \
    --meta "$meta" \
    --out-dir "$out" \
    --n-particles "$N_PART" \
    --n-steps "$N_STEPS" \
    --dt 1e-3 \
    --particles "$parts" \
    --parser "$REPO_ROOT/scripts/scone_bench/parse_rtxadvect_log.py"
ec=$?
tail -25 "$out/run.log" 2>/dev/null
echo "-> $out/result.json (ec=$ec)"

echo
echo "=== FSW IN-MESH RTXAdvect RUN COMPLETE ==="
echo "All five meshes now in: $OUT_ROOT/*/result.json"
