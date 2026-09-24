#!/usr/bin/env bash
#
# Full method x mesh matrix for Section 6.
#
# Runs benchmark_l2_accuracy.py — which implements ALL seven methods
# (MALMO 1x1x1 / 3x3x3 / 5x5x5 vertex, 3x3x3 parent-cube, 3x3x3 AABB,
# and the Morton-linear w-band baselines via search_radius) and ALL
# THREE query protocols (in-mesh, perturbation sigma-sweep,
# intra-element position classes) — across the five benchmark meshes.
#
# Previously this harness could only open the time-dependent FSW PVTU
# sequence. The --vtu flag added to benchmark_l2_accuracy.py lets the
# same code path run on the static meshes.
#
# Usage:
#   scripts/scone_bench/sweep_full_matrix.sh [n_particles]
#
# Default n_particles = 10000 (the paper protocol for the accuracy
# tables). Logs land in sweep_logs/full_matrix_<mesh>_<ts>.log and are
# parsed afterwards by paper_benchmarks/sec6_postprocess.py.

set -uo pipefail

N_PART="10000"
ONLY=""
REGISTRATION="all"
TAG=""

# ---- Args: [n_particles] [--only LABEL[,LABEL...]] -------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      ONLY="$2"; shift 2 ;;
    --only=*)
      ONLY="${1#*=}"; shift ;;
    --registration)
      REGISTRATION="$2"; shift 2 ;;
    --registration=*)
      REGISTRATION="${1#*=}"; shift ;;
    --tag)
      TAG="$2"; shift 2 ;;
    --tag=*)
      TAG="${1#*=}"; shift ;;
    -h|--help)
      cat <<'USAGE'
Usage: sweep_full_matrix.sh [n_particles] [--only LABEL[,LABEL...]]

  n_particles   Query batch size (default 10000, the paper protocol).
  --only        Comma-separated subset of mesh labels to run. Valid labels:
                  FP-8.7k  Bunny  Microfluidics  PorousMedia  FSW
  --registration Which registration strategies to run:
                  all | vertex | parent_cube | aabb | both   (default: all)
  --tag         Suffix appended to log filenames, to keep a targeted
                rerun from overwriting or being confused with a full run.

Examples:
  sweep_full_matrix.sh 10000
  sweep_full_matrix.sh 10000 --only Bunny,Microfluidics,PorousMedia
  sweep_full_matrix.sh 10000 --only Bunny,PorousMedia --registration vertex --tag vfix
USAGE
      exit 0 ;;
    *)
      N_PART="$1"; shift ;;
  esac
done

# Return 0 when a label should run under the current --only filter.
should_run() {
  [[ -z "$ONLY" ]] && return 0
  local want="$1" tok
  IFS=',' read -ra _sel <<< "$ONLY"
  for tok in "${_sel[@]}"; do
    [[ "$tok" == "$want" ]] && return 0
  done
  return 1
}

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
cd "$REPO_ROOT"

JAXTRACE_VENV="${JAXTRACE_VENV:-/flash/shared/jax/.venv}"
if [[ -f "${JAXTRACE_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${JAXTRACE_VENV}/bin/activate"
  echo "activated venv: ${JAXTRACE_VENV}"
fi
PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_CPP_MIN_LOG_LEVEL=2
export JAX_ENABLE_X64=1

HARNESS="$REPO_ROOT/benchmark_l2_accuracy.py"
[[ -f "$HARNESS" ]] || { echo "ERROR: harness not found: $HARNESS" >&2; exit 2; }

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/sweep_logs"
mkdir -p "$LOG_DIR"

# ---- Paper protocol --------------------------------------------------------
PERTURBATIONS=(0.0 0.1 0.2 0.5 0.7 1.0)
POSITION_TYPES=(centroid random near_face near_edge near_vertex)
WARMUP=3
TIMED=7
SEED=42
TOL=1e-6
BATCH=50000

# Set IN_DOMAIN_ORACLE=1 to determine the in-domain reference set with an
# exhaustive MALMO^AABB 5x5x5 search instead of the mesh bounding box, so
# that N_fail excludes queries perturbed outside a non-convex mesh volume.
ORACLE_OPT=()
[[ "${IN_DOMAIN_ORACLE:-0}" == "1" ]] && ORACLE_OPT=(--in-domain-oracle)

# ---- Static meshes (label : vtu path) --------------------------------------
RTX_DATA="${RTXADVECT_DATA:-/flash/shared/jax/JAXTrace/3rdParty/RTXAdvect/dataset}"
KIM_VTU="${KIM_VTU_DIR:-$REPO_ROOT/kim_meshes_vtu}"

STATIC_LABELS=(FP-8.7k Bunny Microfluidics PorousMedia)
STATIC_PATHS=(
  "$KIM_VTU/FinalFuelPinPoly1560.vtu"
  "$KIM_VTU/StanfordBunny_LowPoly.vtu"
  "$RTX_DATA/microfludics/solution_4.vtu"
  "$RTX_DATA/porousMedia/solution_porousmedia.vtu"
)

# ---- FSW (time-dependent PVTU path) ----------------------------------------
FSW_INPUT="${JAXTRACE_MESH_DIR:-/flash/users/ali/data/cylA.gid/post}"
FSW_SUBDIR="0eule"
FSW_PATTERN="cylA_{timestep}.pvtu"
FSW_TS=159

run_static() {
  local label="$1" vtu="$2"
  local log="$LOG_DIR/full_matrix_${label}${TAG:+_$TAG}_${TS}.log"

  if [[ ! -f "$vtu" ]]; then
    echo "== SKIP $label (missing $vtu)"
    return 0
  fi

  echo
  echo "############################################################"
  echo "# FULL MATRIX: $label"
  echo "#   vtu: $vtu"
  echo "#   log: $log"
  echo "############################################################"

  "$PYTHON" -u "$HARNESS" \
      --vtu "$vtu" \
      --mesh-label "$label" \
      --n-particles "$N_PART" \
      --batch-size "$BATCH" \
      --warmup-runs "$WARMUP" \
      --timing-runs "$TIMED" \
      --perturbations "${PERTURBATIONS[@]}" \
      --position-types "${POSITION_TYPES[@]}" \
      --registration "$REGISTRATION" \
      --seed "$SEED" \
      --point-in-tet-tol "$TOL" \
      --float64 \
      "${ORACLE_OPT[@]}" \
      2>&1 | tee "$log"

  echo "-> $log"
}

run_fsw() {
  local log="$LOG_DIR/full_matrix_FSW${TAG:+_$TAG}_${TS}.log"
  if [[ ! -f "${FSW_INPUT}/${FSW_SUBDIR}/cylA_${FSW_TS}.pvtu" ]]; then
    echo "== SKIP FSW (missing ${FSW_INPUT}/${FSW_SUBDIR}/cylA_${FSW_TS}.pvtu)"
    return 0
  fi

  echo
  echo "############################################################"
  echo "# FULL MATRIX: FSW (time-dependent PVTU)"
  echo "#   log: $log"
  echo "############################################################"

  "$PYTHON" -u "$HARNESS" \
      --input "$FSW_INPUT" \
      --mesh-subdir "$FSW_SUBDIR" \
      --mesh-pattern "$FSW_PATTERN" \
      --vel-range "$FSW_TS" "$FSW_TS" \
      --mesh-label FSW \
      --n-particles "$N_PART" \
      --batch-size "$BATCH" \
      --warmup-runs "$WARMUP" \
      --timing-runs "$TIMED" \
      --perturbations "${PERTURBATIONS[@]}" \
      --position-types "${POSITION_TYPES[@]}" \
      --registration "$REGISTRATION" \
      --seed "$SEED" \
      --point-in-tet-tol "$TOL" \
      --float64 \
      "${ORACLE_OPT[@]}" \
      2>&1 | tee "$log"

  echo "-> $log"
}

echo "=== FULL MATRIX SWEEP  (n_particles=$N_PART, ts=$TS) ==="

[[ -n "$ONLY" ]] && echo "    (meshes restricted to: $ONLY)"
[[ "$REGISTRATION" != "all" ]] && echo "    (registration restricted to: $REGISTRATION)"
[[ -n "$TAG" ]] && echo "    (log tag: $TAG)"

for i in "${!STATIC_LABELS[@]}"; do
  label="${STATIC_LABELS[$i]}"
  if ! should_run "$label"; then
    echo "== SKIP $label (not in --only)"
    continue
  fi
  run_static "$label" "${STATIC_PATHS[$i]}" \
    || echo "  ($label exited non-zero; continuing)"
done

if should_run FSW; then
  run_fsw || echo "  (FSW exited non-zero; continuing)"
else
  echo "== SKIP FSW (not in --only)"
fi

echo
echo "=== FULL MATRIX SWEEP COMPLETE ==="
echo "logs: $LOG_DIR/full_matrix_*_${TS}.log"
echo
echo "next: parse each log with"
echo "    python3 paper_benchmarks/sec6_postprocess.py \\"
echo "        --log <log> --json <out.json> --report <out.md>"
