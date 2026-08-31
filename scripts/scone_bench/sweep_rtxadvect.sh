#!/usr/bin/env bash
#
# Sweep RTXAdvect (Wang et al. 2022, CPC) across the same 9 Kim meshes
# + the FSW paper mesh.  Uses zero-velocity field so the tool acts as a
# pure point-location + host-reporting benchmark comparable to MALMO.
#
# Requires setup_rtxadvect.sh to have completed successfully.
#
# Usage:
#   scripts/scone_bench/sweep_rtxadvect.sh [n_particles] [n_steps]
#
# Defaults: 100000 particles, 1 step (particles stay put in zero field).

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
cd "$REPO_ROOT"

N_PART="${1:-100000}"
N_STEPS="${2:-1}"

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

# The build produces libowl.so under build/owl/owl/ — must be on LD path
OWL_LIB_DIR="$REPO_ROOT/3rdParty/RTXAdvect/build/owl/owl"
if [[ -f "$OWL_LIB_DIR/libowl.so" ]]; then
  export LD_LIBRARY_PATH="$OWL_LIB_DIR:${LD_LIBRARY_PATH:-}"
  echo "LD_LIBRARY_PATH prepended with: $OWL_LIB_DIR"
else
  echo "WARN: libowl.so not found at $OWL_LIB_DIR — runtime linking may fail"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/sweep_logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/rtxadvect_${TS}.log"

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

KIM_MESHES=(
  FinalFuelPinTet63 FinalFuelPinTet137 FinalFuelPinTet298
  FinalFuelPinTet1820 FinalFuelPinTet2856
  FinalFuelPinPoly264 FinalFuelPinPoly940 FinalFuelPinPoly1560
  StanfordBunny_LowPoly
)
FSW_PAPER_MESH="${FSW_PAPER_MESH:-/flash/users/ali/data/cylA.gid/post/0eule/cylA_119.pvtu}"

OUT_ROOT="$REPO_ROOT/rtxadvect_runs"
mkdir -p "$OUT_ROOT"
STAGE_DIR="$REPO_ROOT/rtxadvect_meshes"
mkdir -p "$STAGE_DIR"

run_one() {
  local vtu="$1" tag="$2"
  local out="$OUT_ROOT/$tag"
  mkdir -p "$out"
  local stage="$STAGE_DIR/$tag"
  local meta="${stage}.meta.json"
  # Convert VTU -> RTXAdvect format if not already done
  if [[ ! -f "$meta" ]]; then
    echo "== converting $tag"
    "$PYTHON" "$REPO_ROOT/scripts/scone_bench/vtu_to_rtxadvect.py" \
        --vtu "$vtu" --out-prefix "$stage" \
      || { echo "  conversion failed"; return 1; }
  fi
  # Read metadata + seeding-box
  local sbox
  sbox=$("$PYTHON" -c "import json; d=json.load(open('$meta')); print(d['seeding_box_flag'])")
  local verts cells zvel
  verts=$("$PYTHON" -c "import json; print(json.load(open('$meta'))['verts_dat'])")
  cells=$("$PYTHON" -c "import json; print(json.load(open('$meta'))['cells_dat'])")
  zvel=$("$PYTHON"  -c "import json; print(json.load(open('$meta'))['zero_velocity_dat'])")

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

# Sweep Kim meshes
for m in "${KIM_MESHES[@]}"; do
  vtu="$REPO_ROOT/kim_meshes_vtu/$m.vtu"
  if [[ ! -f "$vtu" ]]; then
    # Fallback: convert from Kim polyMesh via the existing bridge
    src="/flash/shared/jax/ii1o33-SCONE-00d7668/IntegrationTestFiles/Geometry/Meshes/OpenFOAM/$m"
    if [[ -d "$src" ]]; then
      "$REPO_ROOT/scripts/openfoam_polymesh_convert.py" \
          --from-polymesh "$src" --to-vtu "$vtu" \
        || { echo "== skip $m (bridge failed)"; continue; }
    else
      echo "== skip $m (no source)"; continue
    fi
  fi
  run_one "$vtu" "$m" || echo "  (RTXAdvect on $m exited non-zero; continuing)"
done

# FSW paper mesh
if [[ -f "$FSW_PAPER_MESH" ]]; then
  run_one "$FSW_PAPER_MESH" "FSW_paper" \
    || echo "  (RTXAdvect on FSW paper exited non-zero; continuing)"
fi

# Aggregate
echo
echo "############################################################"
echo "# RTXAdvect SWEEP RESULTS"
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
print(f'{mesh:<26} RTXAdvect  {status:<12} wall={wall:8.3f}s  found={found_s}  tput={tput_s}  n_part={d.get(\"n_points_queried\")}')"
done

echo
echo "DONE  $(date)"
