#!/usr/bin/env bash
#
# Tier-B extras for Section 6. All cheap; none blocks the main tables.
#
#   1. MALMO-C column for the mesh-size scaling curve
#      (centroid variant on the two microfluidics endpoints; the two
#      intermediate refinements are already done).
#   2. Level-distribution diagnostic on PorousMedia — the FSW mesh is the
#      only one currently reported, and PorousMedia has genuine multi-level
#      structure worth showing.
#   3. RSS capture for RTXAdvect on FP-8.7k and FSW, which are the two
#      gaps ('---') in the peak-memory table.
#
# Usage:
#   scripts/scone_bench/sweep_optional_extras.sh [n_points]

set -uo pipefail

N_POINTS="${1:-100000}"

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
cd "$REPO_ROOT"

JAXTRACE_VENV="${JAXTRACE_VENV:-/flash/shared/jax/.venv}"
[[ -f "$JAXTRACE_VENV/bin/activate" ]] && . "$JAXTRACE_VENV/bin/activate"
PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

RTX_DATA="${RTXADVECT_DATA:-/flash/shared/jax/JAXTrace/3rdParty/RTXAdvect/dataset}"
KIM_VTU="${KIM_VTU_DIR:-$REPO_ROOT/kim_meshes_vtu}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/sweep_logs"; mkdir -p "$LOG_DIR"

MALMO_TIMEOUT="${MALMO_TIMEOUT:-1800}"

# ---------------------------------------------------------------------------
echo
echo "############################################################"
echo "# [1/3] MALMO-C scaling column (microfluidics endpoints)"
echo "############################################################"

SCALE_LABELS=(microfluidics_41k microfluidics_1p66M)
SCALE_PATHS=(
  "$RTX_DATA/microfludics/solution_1.vtu"
  "$RTX_DATA/microfludics/solution_4.vtu"
)

mkdir -p "$REPO_ROOT/malmo_runs_rtxadvect"
for i in "${!SCALE_LABELS[@]}"; do
  mesh="${SCALE_LABELS[$i]}"
  vtu="${SCALE_PATHS[$i]}"
  [[ -f "$vtu" ]] || { echo "  skip $mesh (missing $vtu)"; continue; }

  out="$REPO_ROOT/malmo_runs_rtxadvect/${mesh}__centroid"
  if [[ -f "$out/result.json" ]]; then
    echo "  $mesh centroid already present; skipping"
    continue
  fi
  mkdir -p "$out"
  echo "== MALMO centroid on $mesh"
  timeout --signal=TERM --kill-after=30s "$MALMO_TIMEOUT" \
    "$PYTHON" "$REPO_ROOT/scripts/scone_bench/bench_malmo_pointloc.py" \
      --vtu "$vtu" --variant centroid --n-points "$N_POINTS" \
      --out-dir "$out" > "$out/run.log" 2>&1
  tail -15 "$out/run.log"
done

# ---------------------------------------------------------------------------
echo
echo "############################################################"
echo "# [2/3] Level distribution on PorousMedia"
echo "############################################################"

POROUS_VTU="$RTX_DATA/porousMedia/solution_porousmedia.vtu"
if [[ -f "$POROUS_VTU" ]]; then
  log="$LOG_DIR/level_dist_PorousMedia_${TS}.log"
  echo "== benchmark_l2_accuracy on PorousMedia (level distribution)"
  "$PYTHON" -u "$REPO_ROOT/benchmark_l2_accuracy.py" \
      --vtu "$POROUS_VTU" \
      --mesh-label PorousMedia \
      --n-particles "$N_POINTS" \
      --registration parent_cube \
      --perturbations 0.0 \
      --skip-intra \
      --skip-failure-analysis \
      --warmup-runs 2 --timing-runs 3 \
      --seed 42 --float64 \
      2>&1 | tee "$log"
  echo "-> $log"
else
  echo "  skip (missing $POROUS_VTU)"
fi

# ---------------------------------------------------------------------------
echo
echo "############################################################"
echo "# [3/3] RTXAdvect RSS capture (FP-8.7k, FSW)"
echo "############################################################"

EXE=""
for c in "$REPO_ROOT/3rdParty/RTXAdvect/build/cudaParticleAdvection" \
         "$REPO_ROOT/3rdParty/RTXAdvect/build/bin/cudaParticleAdvection"; do
  [[ -x "$c" ]] && { EXE="$c"; break; }
done

if [[ -n "$EXE" ]]; then
  OWL_LIB_DIR="$REPO_ROOT/3rdParty/RTXAdvect/build/owl/owl"
  [[ -f "$OWL_LIB_DIR/libowl.so" ]] && \
    export LD_LIBRARY_PATH="$OWL_LIB_DIR:${LD_LIBRARY_PATH:-}"

  RSS_LABELS=(FP-8.7k FSW_paper)
  RSS_VTUS=(
    "$KIM_VTU/FinalFuelPinPoly1560.vtu"
    "${FSW_PAPER_MESH:-/flash/users/ali/data/cylA.gid/post/0eule/cylA_119.pvtu}"
  )
  OUT_ROOT="$REPO_ROOT/rtxadvect_runs_rss"
  STAGE_DIR="$REPO_ROOT/rtxadvect_meshes"
  mkdir -p "$OUT_ROOT" "$STAGE_DIR"

  for i in "${!RSS_LABELS[@]}"; do
    label="${RSS_LABELS[$i]}"
    vtu="${RSS_VTUS[$i]}"
    [[ -f "$vtu" ]] || { echo "  skip $label (missing $vtu)"; continue; }

    stage="$STAGE_DIR/$label"
    meta="${stage}.meta.json"
    if [[ ! -f "$meta" ]]; then
      "$PYTHON" "$REPO_ROOT/scripts/scone_bench/vtu_to_rtxadvect.py" \
          --vtu "$vtu" --out-prefix "$stage" \
        || { echo "  staging failed for $label"; continue; }
    fi

    out="$OUT_ROOT/$label"; mkdir -p "$out"
    echo "== RTXAdvect RSS run: $label"
    "$PYTHON" "$REPO_ROOT/scripts/scone_bench/bench_rtxadvect_single.py" \
        --exe "$EXE" --meta "$meta" --out-dir "$out" \
        --n-particles "$N_POINTS" --n-steps 1 --dt 1e-3 \
        --parser "$REPO_ROOT/scripts/scone_bench/parse_rtxadvect_log.py"
    echo "-> $out/result.json"
  done
else
  echo "  skip (RTXAdvect exe not found)"
fi

echo
echo "=== OPTIONAL EXTRAS COMPLETE ==="
