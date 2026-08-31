#!/usr/bin/env bash
#
# Overnight orchestrator for the CMAME revision.  Reruns MALMO on Kim's
# meshes with the extractor fixes, runs MALMO on the full 3M-tet FSW
# paper mesh, converts the FSW mesh to polyMesh, runs SCONE on it,
# sweeps the newly-ported RTXAdvect (Wang 2022) at meaningful scale
# across all meshes, then aggregates everything.
#
# Everything is logged unbuffered to sweep_logs/overnight_YYYYMMDD_HHMMSS.log.
# All scripts within honour timeouts; nothing can hang forever.
#
# Estimated runtime: 2 to 5 h depending on how MALMO scales on
# the 3M-tet mesh + how many RTXAdvect meshes are queued.  Safe to
# leave overnight.
#
# Steps:
#   1/5  MALMO on Kim's 9 meshes (extractor fixes)
#   2/5  MALMO on 3M-tet FSW paper mesh (3 variants)
#   3/5  SCONE on FSW paper mesh (all pointloc methods, ~1 h)
#   4/5  RTXAdvect at scale (1M particles x 100 steps) on all meshes
#   5/5  Aggregate results
#
# Usage:
#   scripts/scone_bench/overnight_run.sh
#
# Env overrides:
#   FSW_PAPER_MESH        default /flash/users/ali/data/cylA.gid/post/0eule/cylA_119.pvtu
#   FSW_MALMO_TIMEOUT     default 3600 s (1 h) per MALMO variant on the 3M-tet mesh
#   KIM_MALMO_TIMEOUT     default 1800 s (30 min) per Kim (mesh, variant) row
#   SCONE_MALMO_POP       default 1000 (small: 3M mesh is huge, patchSingle needs GBs)
#   SCONE_MALMO_CYCLES    default 1
#   RTX_N_PARTICLES       default 1000000
#   RTX_N_STEPS           default 100
#   SKIP_MALMO_KIM=1      skip STEP 1 (already have results)
#   SKIP_MALMO_FSW=1      skip STEP 2
#   SKIP_SCONE_FSW=1      skip STEP 3
#   SKIP_RTXADVECT=1      skip STEP 4

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
cd "$REPO_ROOT"

FSW_PAPER_MESH="${FSW_PAPER_MESH:-/flash/users/ali/data/cylA.gid/post/0eule/cylA_119.pvtu}"
FSW_MALMO_TIMEOUT="${FSW_MALMO_TIMEOUT:-3600}"
KIM_MALMO_TIMEOUT="${KIM_MALMO_TIMEOUT:-1800}"
SCONE_POP="${SCONE_POP:-1000}"
SCONE_CYCLES="${SCONE_CYCLES:-1}"
RTX_N_PARTICLES="${RTX_N_PARTICLES:-1000000}"
RTX_N_STEPS="${RTX_N_STEPS:-100}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/sweep_logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="${LOG_DIR}/overnight_${TS}.log"

# Self-relaunch under stdbuf+tee so the main log is unbuffered.
if [[ "${OVERNIGHT_ACTIVE:-0}" != "1" ]]; then
  export OVERNIGHT_ACTIVE=1
  echo "main log: $MAIN_LOG"
  exec stdbuf -oL -eL bash -c "'$0' $*" 2>&1 | tee "$MAIN_LOG"
fi

echo "############################################################"
echo "# OVERNIGHT ORCHESTRATOR"
echo "#   start: $(date)"
echo "#   host : $(hostname)"
echo "#   log  : $MAIN_LOG"
echo "############################################################"

# Activate the venv (subprocesses inherit)
JAXTRACE_VENV="${JAXTRACE_VENV:-/flash/shared/jax/.venv}"
if [[ -f "${JAXTRACE_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${JAXTRACE_VENV}/bin/activate"
  echo "activated venv: ${JAXTRACE_VENV}"
  python3 -c "import numpy, vtk, jax; print(f'  numpy={numpy.__version__}  vtk={vtk.VTK_VERSION}  jax={jax.__version__}')" || true
fi
export PYTHONUNBUFFERED=1

step() {
  echo
  echo "============================================================"
  echo "== STEP: $*"
  echo "== $(date)"
  echo "============================================================"
}

# --------------------------------------------------------------
# STEP 1 · Rerun MALMO-on-Kim sweep with the extractor fixes
# --------------------------------------------------------------
if [[ "${SKIP_MALMO_KIM:-0}" == "1" ]]; then
  step "1/5 · MALMO on Kim's 9 meshes  (SKIPPED via SKIP_MALMO_KIM=1)"
else
step "1/5 · MALMO on Kim's 9 meshes with extractor fixes"

# Wipe old runs so results are clean
rm -rf "$REPO_ROOT/malmo_runs"

MALMO_TIMEOUT="$KIM_MALMO_TIMEOUT" \
    scripts/scone_bench/sweep_malmo_on_kim.sh 100000 \
  || echo "  (Kim sweep exited non-zero, continuing)"

# Aggregate results
echo
echo "--- Kim MALMO results table ---"
ls "$REPO_ROOT/malmo_runs"/*/result.json 2>/dev/null | while read f; do
  python3 -c "
import json
d = json.load(open('$f'))
mesh = d.get('mesh','?')[:26]
method = d.get('method','?').replace('MALMO_','')
if d.get('timed_out'): print(f'{mesh:<26} {method:<15} TIMED OUT'); exit()
if d.get('failed') or d.get('found_rate') is None: print(f'{mesh:<26} {method:<15} FAILED'); exit()
r = d['found_rate']*100
qms = d['wall_seconds']['query_min_of_3']*1000
tput = d['throughput_queries_per_second']/1e6
pit = d.get('mean_pit_tests', 0) or 0
print(f'{mesh:<26} {method:<15} rate={r:6.2f}%  t={qms:8.2f}ms  tput={tput:7.2f}Mq/s  PIT={pit:8.1f}')"
done
fi  # end SKIP_MALMO_KIM

# --------------------------------------------------------------
# STEP 2 · MALMO on the full 3M-tet FSW paper mesh (all variants)
# --------------------------------------------------------------
if [[ "${SKIP_MALMO_FSW:-0}" == "1" ]]; then
  step "2/5 · MALMO on the full FSW paper mesh  (SKIPPED via SKIP_MALMO_FSW=1)"
else
step "2/5 · MALMO on the full FSW paper mesh: $FSW_PAPER_MESH"

if [[ ! -f "$FSW_PAPER_MESH" ]]; then
  echo "  ERROR: FSW_PAPER_MESH not found; skipping FSW MALMO runs"
else
  for variant in vertex_multi centroid aabb; do
    out_dir="$REPO_ROOT/malmo_runs_fsw_paper/${variant}"
    step "  MALMO $variant on FSW paper mesh"
    MALMO_TIMEOUT="$FSW_MALMO_TIMEOUT" \
        scripts/scone_bench/run_malmo_single.sh \
            "$FSW_PAPER_MESH" "$variant" 100000 "$out_dir" \
      || echo "    (variant $variant exited non-zero; continuing)"
  done
fi
fi  # end SKIP_MALMO_FSW

# --------------------------------------------------------------
# STEP 3 · Convert FSW paper mesh to polyMesh + run SCONE on it
# --------------------------------------------------------------
if [[ "${SKIP_SCONE_FSW:-0}" == "1" ]]; then
  step "3/5 · SCONE on FSW paper mesh  (SKIPPED via SKIP_SCONE_FSW=1)"
else
step "3/5 · Convert FSW paper mesh to polyMesh + run SCONE"

if [[ ! -f "$FSW_PAPER_MESH" ]]; then
  echo "  skipping (no FSW mesh)"
else
  rm -rf "$REPO_ROOT/scone_meshes/FSW_paper"
  echo "  converting FSW paper PVTU -> polyMesh (may take minutes on 3M tets)..."
  if timeout 3600 scripts/scone_bench/prepare_fsw_polymesh.sh \
         "$FSW_PAPER_MESH" FSW_paper; then
    echo "  polyMesh written to $REPO_ROOT/scone_meshes/FSW_paper"
    # Sweep SCONE across the four methods; each has its own driver
    # timeouts.  Note SCONE_POP is small — patchSingle uses ~2 GB per
    # 300 tets, so on 3M tets patchSingle would need ~20 TB.  We run
    # it anyway to record the OOM as data.
    BOX_PAD=5.0 scripts/scone_bench/sweep_pointloc.sh \
        "$REPO_ROOT/scone_meshes/FSW_paper" "$SCONE_POP" "$SCONE_CYCLES" \
      || echo "  (SCONE FSW sweep exited non-zero; continuing)"
  else
    echo "  ERROR: polyMesh conversion failed or timed out"
  fi
fi
fi  # end SKIP_SCONE_FSW

# --------------------------------------------------------------
# STEP 4 · RTXAdvect at meaningful scale (1M particles × 100 steps)
#         across all Kim meshes + FSW paper mesh
# --------------------------------------------------------------
if [[ "${SKIP_RTXADVECT:-0}" == "1" ]]; then
  step "4/5 · RTXAdvect at scale  (SKIPPED via SKIP_RTXADVECT=1)"
else
step "4/5 · RTXAdvect at scale: ${RTX_N_PARTICLES} particles × ${RTX_N_STEPS} steps"

# Move any previous scale-1 runs aside so we don't overwrite them
if [[ -d "$REPO_ROOT/rtxadvect_runs" ]]; then
  BAK="$REPO_ROOT/rtxadvect_runs_100kx1_${TS}"
  mv "$REPO_ROOT/rtxadvect_runs" "$BAK"
  echo "  previous rtxadvect_runs/ moved to ${BAK##*/}"
fi

# Verify executable is available
RTX_EXE="$REPO_ROOT/3rdParty/RTXAdvect/build/cudaParticleAdvection"
if [[ ! -x "$RTX_EXE" ]]; then
  echo "  ERROR: RTXAdvect executable not found at $RTX_EXE — skipping STEP 4"
  echo "         Run scripts/scone_bench/setup_rtxadvect.sh + port_rtxadvect_step*.sh first."
else
  scripts/scone_bench/sweep_rtxadvect.sh "$RTX_N_PARTICLES" "$RTX_N_STEPS" \
    || echo "  (RTXAdvect scale sweep exited non-zero; continuing)"
fi
fi  # end SKIP_RTXADVECT

# --------------------------------------------------------------
# STEP 5 · Aggregate everything
# --------------------------------------------------------------
step "5/5 · Aggregate results"

# All SCONE
if compgen -G "$REPO_ROOT/scone_runs/*/result.json" > /dev/null; then
  scripts/scone_bench/collect_results.py \
      --root scone_runs --csv scone_runs/summary_final.csv || true
fi

# All MALMO
echo
echo "--- Final MALMO summary (Kim + FSW paper) ---"
for d in "$REPO_ROOT/malmo_runs" "$REPO_ROOT/malmo_runs_fsw_paper"; do
  [[ -d "$d" ]] || continue
  ls "$d"/*/result.json 2>/dev/null | while read f; do
    python3 -c "
import json
d = json.load(open('$f'))
mesh = d.get('mesh','?')[:26]
method = d.get('method','?')
if d.get('timed_out'): print(f'{mesh:<26} {method:<24} TIMED OUT'); exit()
if d.get('failed') or d.get('found_rate') is None: print(f'{mesh:<26} {method:<24} FAILED'); exit()
r = d['found_rate']*100
qms = d['wall_seconds']['query_min_of_3']*1000
tput = d['throughput_queries_per_second']/1e6
pit = d.get('mean_pit_tests', 0) or 0
rss = d.get('process',{}).get('max_rss_kb') or 0
rss_mb = rss/1024
print(f'{mesh:<26} {method:<24} rate={r:6.2f}%  t={qms:9.2f}ms  tput={tput:7.2f}Mq/s  PIT={pit:8.1f}  RSS={rss_mb:8.1f}MB')"
  done
done

# RTXAdvect summary
echo
echo "--- RTXAdvect summary (all meshes, ${RTX_N_PARTICLES} × ${RTX_N_STEPS}) ---"
if compgen -G "$REPO_ROOT/rtxadvect_runs/*/result.json" > /dev/null; then
  ls "$REPO_ROOT/rtxadvect_runs"/*/result.json 2>/dev/null | while read f; do
    python3 -c "
import json
d = json.load(open('$f'))
mesh = d.get('mesh','?')[:26]
ec = d.get('shell_exit_code')
ws = d.get('wall_seconds') or {}
outer = ws.get('outer', 0.0)
bvh = ws.get('bvh_ms')
sim = ws.get('simulation_ms')
found = d.get('found_rate')
found_s = f'{found*100:6.2f}%' if found is not None else '     ?%'
nout = d.get('n_out_of_domain', '?')
tput = d.get('throughput_Mqps')
tput_s = f'{tput:8.2f} Mq/s' if tput is not None else '        ? Mq/s'
status = 'OK' if ec == 0 else f'FAIL(ec={ec})'
bvh_s = f'{bvh:8.1f}ms' if bvh is not None else '        ?'
sim_s = f'{sim:8.1f}ms' if sim is not None else '        ?'
print(f'{mesh:<26} RTXAdvect  {status:<12} found={found_s} n_out={nout:>7} BVH={bvh_s} sim={sim_s} tput={tput_s} outer={outer:8.2f}s')"
  done
else
  echo "  (no RTXAdvect results in rtxadvect_runs/)"
fi

# Try to build the joint 3-way table (MALMO + SCONE + RTXAdvect)
echo
echo "--- Building joint 3-way summary table ---"
python3 scripts/scone_bench/build_joint_table.py 2>&1 | tail -20 || \
  echo "  (joint table build failed — likely need to extend build_joint_table.py for RTXAdvect)"

echo
echo "############################################################"
echo "# OVERNIGHT ORCHESTRATOR complete"
echo "#   end : $(date)"
echo "#   log : $MAIN_LOG"
echo "############################################################"
