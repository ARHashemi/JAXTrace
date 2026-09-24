#!/usr/bin/env bash
#
# Re-sweep the R2-6 T1-T5 tables on a warmed workstation so the
# headline paper numbers reflect steady-state throughput rather than
# the cold-cache/cold-GPU state some rows were measured in on 2026-08-27.
#
# Same 10-mesh cohort, same 3 MALMO variants, same N_p (100_000),
# same in-mesh barycentric sampling as the original R2-6 sweep, but
# with two changes:
#
#   1. A warm-up phase (a few MALMO runs on a discardable small mesh)
#      before the real sweep, so the GPU + JAX kernel cache are in
#      steady-state before the timed runs begin.
#
#   2. Output goes to malmo_runs_in_mesh_warmed/ (NOT malmo_runs_in_mesh/)
#      so the original R2-6 numbers remain intact for provenance.
#
# When done, run build_joint_table.py with
#   --malmo-root malmo_runs_in_mesh_warmed
# to regenerate the joint_summary and the paper_table_*.{md,tex}
# with the warmed numbers.
#
# Expected wall time: ~2 h total on a warm GPU (10 meshes x 3 variants
# x few seconds to few minutes each; FSW dominates at ~5-30 s per run
# depending on warm state).
#
# Usage:
#   scripts/scone_bench/resweep_T1_T5_warmed.sh
#
# Env overrides (all optional):
#   FSW_MESH        default /flash/users/ali/data/cylA.gid/post/0eule/cylA_119.pvtu
#   N_POINTS        default 100000  (must match the R2-6 default to be comparable)

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
cd "$REPO_ROOT"

FSW_MESH="${FSW_MESH:-/flash/users/ali/data/cylA.gid/post/0eule/cylA_119.pvtu}"
N_POINTS="${N_POINTS:-100000}"
OUT_ROOT="$REPO_ROOT/malmo_runs_in_mesh_warmed"

: "${JAXTRACE_VENV:=/flash/shared/jax/.venv}"
[[ -f "$JAXTRACE_VENV/bin/activate" ]] && source "$JAXTRACE_VENV/bin/activate"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/sweep_logs"
mkdir -p "$LOG_DIR" "$OUT_ROOT"
MAIN_LOG="$LOG_DIR/resweep_T1_T5_warmed_${TS}.log"

if [[ "${SWEEP_LOG_ACTIVE:-0}" != "1" ]]; then
  export SWEEP_LOG_ACTIVE=1
  exec stdbuf -oL -eL bash -c "'$0' $*" 2>&1 | tee "$MAIN_LOG"
fi

echo "############################################################"
echo "# T1-T5 WARMED RE-SWEEP"
echo "#   start : $(date)"
echo "#   host  : $(hostname)"
echo "#   n_p   : $N_POINTS"
echo "#   out   : $OUT_ROOT"
echo "############################################################"

# ---- 0. Warm-up phase: 2 rounds of the 3 variants on the smallest
#         Kim mesh, discarded.  Puts the GPU into a stable clock state
#         and populates the JAX kernel cache.
echo
echo "== WARM-UP =="
WARM_TMP="$(mktemp -d -t malmo_warmup.XXXXXX)"
trap 'rm -rf "$WARM_TMP"' EXIT
WARMUP_MESH="$REPO_ROOT/kim_meshes_vtu/FinalFuelPinTet63.vtu"
if [[ ! -f "$WARMUP_MESH" ]]; then
  echo "  ERROR: warm-up mesh not found at $WARMUP_MESH"; exit 3
fi
for round in 1 2; do
  for v in aabb centroid vertex_multi; do
    echo "  warm-up round $round: $v on Tet63"
    python3 scripts/scone_bench/bench_malmo_pointloc.py \
        --vtu "$WARMUP_MESH" \
        --out-dir "$WARM_TMP/round${round}__$v" \
        --variant "$v" \
        --n-points "$N_POINTS" \
        --sampling in_mesh \
      > /dev/null 2>&1 || echo "    (warm-up failed for $v; continuing)"
  done
done
echo "== WARM-UP done — GPU and JAX kernel cache should be in steady state =="
echo

# ---- 1. Real sweep — the 10-mesh cohort, 3 variants each.
KIM_MESHES=(
  FinalFuelPinTet63 FinalFuelPinTet137 FinalFuelPinTet298
  FinalFuelPinTet1820 FinalFuelPinTet2856
  FinalFuelPinPoly264 FinalFuelPinPoly940 FinalFuelPinPoly1560
  StanfordBunny_LowPoly
)

run_mesh() {
  local vtu="$1" tag="$2"
  echo
  echo "########  $tag  ########"
  for v in vertex_multi centroid aabb; do
    local out="$OUT_ROOT/${tag}__$v"
    if [[ -f "$out/result.json" ]]; then
      echo "  skip $v (result.json exists)"
      continue
    fi
    mkdir -p "$out"
    echo "== $v =="
    python3 scripts/scone_bench/bench_malmo_pointloc.py \
        --vtu "$vtu" \
        --out-dir "$out" \
        --variant "$v" \
        --n-points "$N_POINTS" \
        --sampling in_mesh \
      || echo "  ($v exited non-zero; continuing)"
  done
}

for m in "${KIM_MESHES[@]}"; do
  VTU="$REPO_ROOT/kim_meshes_vtu/${m}.vtu"
  if [[ ! -f "$VTU" ]]; then
    echo "  skip $m (VTU not staged)"
    continue
  fi
  run_mesh "$VTU" "$m"
done

# FSW paper mesh
if [[ -f "$FSW_MESH" ]]; then
  run_mesh "$FSW_MESH" "FSW_paper"
else
  echo "  skip FSW (mesh not found at $FSW_MESH)"
fi

# ---- 2. Regenerate the joint table and paper tables against
#         the warmed cohort.
echo
echo "########  Regenerating joint_summary and paper tables  ########"

python3 scripts/scone_bench/build_joint_table.py \
   --malmo-root      "$OUT_ROOT" \
   --malmo-fsw-root  "$REPO_ROOT/malmo_runs_fsw_paper" \
   --scone-root      "$REPO_ROOT/scone_runs_padfix" \
   --scone-fsw-root  "$REPO_ROOT/scone_runs" \
   --rtx-root        "$REPO_ROOT/rtxadvect_runs" \
   --csv             "$REPO_ROOT/joint_summary_warmed.csv" \
   --md              "$REPO_ROOT/joint_summary_warmed.md" \
  || echo "  build_joint_table.py exited non-zero (check output)"

# Also emit paper_table_*_warmed.{md,tex}, so the originals stay
# untouched until we decide which set the paper should use.
python3 scripts/scone_bench/build_paper_tables.py \
   --csv "$REPO_ROOT/joint_summary_warmed.csv" \
   --out-dir "$REPO_ROOT/paper_tables_warmed" \
  || echo "  build_paper_tables.py exited non-zero (check output)"

echo
echo "############################################################"
echo "# T1-T5 WARMED RE-SWEEP complete"
echo "#   end : $(date)"
echo "#   log : $MAIN_LOG"
echo "############################################################"
echo
echo "Compare warmed vs cold numbers with:"
echo "  diff <(head -30 joint_summary.md) <(head -30 joint_summary_warmed.md)"
