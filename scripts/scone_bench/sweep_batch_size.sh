#!/usr/bin/env bash
#
# Sweep MALMO variants at multiple batch sizes for R2-7 (kernel-only
# timing, larger batches).
#
# For each of a fixed set of batch sizes N_p, run bench_malmo_pointloc.py
# on each MALMO variant (aabb, centroid, vertex_multi) on a target mesh.
# The bench script already uses jax.block_until_ready() + best-of-3
# trials, so the reported query_min_of_3 is a kernel-only-equivalent
# time; wall_seconds_outer captures the full-process time for
# reference.
#
# Usage:
#   scripts/scone_bench/sweep_batch_size.sh <MESH_TAG> <VTU_PATH>
#     [batch_size_list]  (default: "10000 50000 100000 200000 500000")
#
# Example (FSW mesh):
#   scripts/scone_bench/sweep_batch_size.sh FSW_paper \\
#       /flash/users/ali/data/cylA.gid/post/0eule/cylA_119.pvtu
#
# The output goes to malmo_runs_batch_sweep/<mesh>__<variant>__np<N>/result.json,
# so build_paper_tables.py can (with a small extension) emit a T6 batch-
# size scaling table.

set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <MESH_TAG> <VTU_PATH> [batch_size_list]" >&2
  exit 2
fi

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
cd "$REPO_ROOT"

MESH_NAME="$1"
VTU="$2"
BATCH_SIZES="${3:-10000 50000 100000 200000 500000}"

: "${JAXTRACE_VENV:=/flash/shared/jax/.venv}"
[[ -f "$JAXTRACE_VENV/bin/activate" ]] && source "$JAXTRACE_VENV/bin/activate"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/sweep_logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/batch_sweep_${MESH_NAME}_${TS}.log"

if [[ "${SWEEP_LOG_ACTIVE:-0}" != "1" ]]; then
  export SWEEP_LOG_ACTIVE=1
  exec stdbuf -oL -eL bash -c "'$0' $*" 2>&1 | tee "$MAIN_LOG"
fi

echo "############################################################"
echo "# BATCH-SIZE SWEEP for $MESH_NAME"
echo "#   start: $(date)"
echo "#   host : $(hostname)"
echo "#   sizes: $BATCH_SIZES"
echo "############################################################"

for variant in vertex_multi centroid aabb; do
  for n in $BATCH_SIZES; do
    out_dir="$REPO_ROOT/malmo_runs_batch_sweep/${MESH_NAME}__${variant}__np${n}"
    if [[ -f "$out_dir/result.json" ]]; then
      echo "  skip $variant np=$n (result.json exists)"
      continue
    fi
    mkdir -p "$out_dir"
    echo
    echo "== $variant  N_p=$n  =="
    python3 scripts/scone_bench/bench_malmo_pointloc.py \
        --vtu "$VTU" \
        --out-dir "$out_dir" \
        --variant "$variant" \
        --n-points "$n" \
        --sampling in_mesh \
      || echo "  ($variant np=$n exited non-zero; continuing)"
  done
done

echo
echo "############################################################"
echo "# BATCH-SIZE SWEEP complete"
echo "#   end  : $(date)"
echo "#   log  : $MAIN_LOG"
echo "############################################################"
