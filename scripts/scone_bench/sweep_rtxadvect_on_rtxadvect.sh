#!/usr/bin/env bash
#
# Run RTXAdvect on its own reference meshes (microfluidics, porousMedia)
# so we have head-to-head numbers vs. MALMO on the same inputs.
#
# Companion to sweep_malmo_on_rtxadvect.sh.  Requires setup_rtxadvect.sh
# to have been run.
#
# Usage:
#   scripts/scone_bench/sweep_rtxadvect_on_rtxadvect.sh [n_particles] [n_steps]
#
# Defaults: 100000 particles, 1 step (zero-velocity field, so RTXAdvect
# is exercised as a pure point-location + host-reporting tool, matching
# the MALMO methodology).

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
cd "$REPO_ROOT"

N_PART="${1:-100000}"
N_STEPS="${2:-1}"

# Locate the RTXAdvect executable.
EXE_CANDIDATES=(
  "$REPO_ROOT/3rdParty/RTXAdvect/build/cudaParticleAdvection"
  "$REPO_ROOT/3rdParty/RTXAdvect/build/bin/cudaParticleAdvection"
  "$REPO_ROOT/3rdParty/RTXAdvect/build/cudaParticleAdvection.exe"
)
EXE=""
for c in "${EXE_CANDIDATES[@]}"; do
  if [[ -x "$c" ]]; then EXE="$c"; break; fi
done
if [[ -z "$EXE" ]]; then
  echo "ERROR: RTXAdvect executable not found.  Ran setup_rtxadvect.sh?" >&2
  echo "Searched:" >&2
  printf '  %s\n' "${EXE_CANDIDATES[@]}" >&2
  exit 2
fi
echo "using RTXAdvect executable: $EXE"

OWL_LIB_DIR="$REPO_ROOT/3rdParty/RTXAdvect/build/owl/owl"
if [[ -f "$OWL_LIB_DIR/libowl.so" ]]; then
  export LD_LIBRARY_PATH="$OWL_LIB_DIR:${LD_LIBRARY_PATH:-}"
  echo "LD_LIBRARY_PATH prepended with: $OWL_LIB_DIR"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/sweep_logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/rtxadvect_on_rtxadvect_${TS}.log"

if [[ "${SWEEP_LOG_ACTIVE:-0}" != "1" ]]; then
  export SWEEP_LOG_ACTIVE=1
  echo "logging to $MAIN_LOG"
  exec stdbuf -oL -eL bash -c "'$0' $*" 2>&1 | tee "$MAIN_LOG"
fi

JAXTRACE_VENV="${JAXTRACE_VENV:-/flash/shared/jax/.venv}"
if [[ -f "$JAXTRACE_VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$JAXTRACE_VENV/bin/activate"
fi
PYTHON="${PYTHON:-python3}"

RTX_DATA="${RTXADVECT_DATA:-$REPO_ROOT/3rdParty/RTXAdvect/dataset}"
OUT_ROOT="$REPO_ROOT/rtxadvect_runs_rtxadvect"
STAGE_DIR="$REPO_ROOT/rtxadvect_meshes_rtxadvect"
mkdir -p "$OUT_ROOT" "$STAGE_DIR"

MESH_NAMES=(microfluidics_41k microfluidics_1p66M porousMedia_820k)
MESH_PATHS=(
  "$RTX_DATA/microfludics/solution_1.vtu"
  "$RTX_DATA/microfludics/solution_4.vtu"
  "$RTX_DATA/porousMedia/solution_porousmedia.vtu"
)

run_one() {
  local vtu="$1" tag="$2"
  local out="$OUT_ROOT/$tag"
  mkdir -p "$out"
  local stage="$STAGE_DIR/$tag"
  local meta="${stage}.meta.json"
  if [[ ! -f "$meta" ]]; then
    echo "== converting $tag"
    "$PYTHON" "$REPO_ROOT/scripts/scone_bench/vtu_to_rtxadvect.py" \
        --vtu "$vtu" --out-prefix "$stage" \
      || { echo "  conversion failed"; return 1; }
  fi

  echo
  echo "############################################################"
  echo "# RTXAdvect on $tag  (n_part=$N_PART, n_steps=$N_STEPS)"
  echo "############################################################"

  "$PYTHON" "$REPO_ROOT/scripts/scone_bench/bench_rtxadvect_single.py" \
      --exe "$EXE" \
      --meta "$meta" \
      --out-dir "$out" \
      --n-particles "$N_PART" \
      --n-steps "$N_STEPS" \
      --dt 1e-3 \
      --parser "$REPO_ROOT/scripts/scone_bench/parse_rtxadvect_log.py"
  local ec=$?
  tail -30 "$out/run.log"
  echo "-> $out/result.json (ec=$ec)"
}

for i in "${!MESH_NAMES[@]}"; do
  vtu="${MESH_PATHS[$i]}"
  tag="${MESH_NAMES[$i]}"
  if [[ ! -f "$vtu" ]]; then
    echo "== skip $tag (missing $vtu)"
    continue
  fi
  run_one "$vtu" "$tag" || echo "  (RTXAdvect on $tag exited non-zero; continuing)"
done

echo
echo "############################################################"
echo "# RTXAdvect-on-RTXAdvect SWEEP RESULTS"
echo "############################################################"
ls "$OUT_ROOT"/*/result.json 2>/dev/null | while read f; do
  "$PYTHON" -c "
import json
d = json.load(open('$f'))
mesh = d.get('mesh','?')
ec = d.get('shell_exit_code')
wall = (d.get('wall_seconds') or {}).get('outer', 0.0)
found = d.get('found_rate')
found_s = f'{found*100:5.1f}%' if found is not None else '   ?%'
tput = d.get('throughput_Mqps')
tput_s = f'{tput:8.2f} Mqps' if tput is not None else '     ? Mqps'
status = 'OK' if ec==0 else f'FAIL(ec={ec})'
print(f'{mesh:<24} RTXAdvect  {status:<12} wall={wall:8.3f}s  found={found_s}  tput={tput_s}  n_part={d.get(\"n_points_queried\")}')"
done

echo
echo "DONE  $(date)"
