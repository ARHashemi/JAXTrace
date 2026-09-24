#!/usr/bin/env bash
#
# Re-run RTXAdvect with IN-MESH seeding so its correctness column is
# measured on the same query distribution as MALMO's.
#
# RTXAdvect's default --seeding-box samples uniformly in the mesh AABB.
# On the three non-convex meshes (Bunny, Microfluidics, PorousMedia)
# roughly a third of those samples fall outside the mesh volume, so the
# reported found-rate tracks the bbox fill ratio (~66%) rather than any
# property of the search. This sweep feeds RTXAdvect an explicit particle
# file drawn with the same volume-weighted barycentric sampler MALMO uses,
# via the tool's own --input-particles flag.
#
# Usage:
#   scripts/scone_bench/sweep_inmesh_rtxadvect.sh [n_particles] [n_steps]

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

RTX_DATA="${RTXADVECT_DATA:-$REPO_ROOT/3rdParty/RTXAdvect/dataset}"
KIM_VTU="${KIM_VTU_DIR:-$REPO_ROOT/kim_meshes_vtu}"

OUT_ROOT="$REPO_ROOT/rtxadvect_runs_inmesh_seeded"
STAGE_DIR="$REPO_ROOT/rtxadvect_meshes_rtxadvect"
PART_DIR="$REPO_ROOT/rtxadvect_particles"
mkdir -p "$OUT_ROOT" "$STAGE_DIR" "$PART_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/sweep_logs"; mkdir -p "$LOG_DIR"

# label : vtu
LABELS=(Microfluidics PorousMedia Bunny FP-8.7k)
PATHS=(
  "$RTX_DATA/microfludics/solution_4.vtu"
  "$RTX_DATA/porousMedia/solution_porousmedia.vtu"
  "$KIM_VTU/StanfordBunny_LowPoly.vtu"
  "$KIM_VTU/FinalFuelPinPoly1560.vtu"
)

for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  vtu="${PATHS[$i]}"

  if [[ ! -f "$vtu" ]]; then
    echo "== SKIP $label (missing $vtu)"
    continue
  fi

  stage="$STAGE_DIR/$label"
  meta="${stage}.meta.json"
  parts="$PART_DIR/${label}.particles.dat"

  echo
  echo "############################################################"
  echo "# RTXAdvect IN-MESH: $label"
  echo "############################################################"

  # 1. Mesh -> RTXAdvect .dat format (skip if already staged)
  if [[ ! -f "$meta" ]]; then
    echo "== staging mesh $label"
    "$PYTHON" "$REPO_ROOT/scripts/scone_bench/vtu_to_rtxadvect.py" \
        --vtu "$vtu" --out-prefix "$stage" \
      || { echo "  mesh staging failed; skipping"; continue; }
  fi

  # 2. In-mesh particle seeds (same sampler + seed as MALMO)
  if [[ ! -f "$parts" ]]; then
    echo "== generating in-mesh particles for $label"
    "$PYTHON" "$REPO_ROOT/scripts/scone_bench/make_inmesh_particles.py" \
        --vtu "$vtu" --out "$parts" \
        --n-particles "$N_PART" --seed "$SEED" \
      || { echo "  particle generation failed; skipping"; continue; }
  fi

  # 3. Run RTXAdvect with --input-particles
  out="$OUT_ROOT/$label"
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
done

echo
echo "=== RTXAdvect IN-MESH SWEEP COMPLETE ==="
echo "results: $OUT_ROOT/*/result.json"
echo "particles: $PART_DIR/"
echo
echo "Compare against the bbox-seeded runs in rtxadvect_runs_rtxadvect/"
echo "to confirm the ~66% found-rate there is a sampling artefact."
