#!/usr/bin/env bash
#
# R1-g float-precision ablation: sweep MALMO across
# {float32, float64} x {PIT tolerance 1e-4, 1e-6, 1e-8} on one mesh.
#
# Precision is controlled at the process level via JAX_ENABLE_X64:
#   JAX_ENABLE_X64=1 -> float64 default
#   JAX_ENABLE_X64=0 -> float32 default
# PIT tolerance is passed via bench_malmo_pointloc.py's --tol.
#
# We sweep the three MALMO variants (aabb, centroid, vertex_multi)
# on one target mesh at a fixed batch size (100k, the R2-6 default),
# for all 6 (precision, tolerance) combinations = 18 runs total per
# variant, 54 runs per mesh.  Estimated ~1 minute per run on the
# Kim meshes, ~5 minutes per run on FSW = ~5 min or ~4.5 h
# respectively.
#
# Recommended usage:
#   scripts/scone_bench/sweep_float_ablation.sh FSW_paper \
#       /flash/users/ali/data/cylA.gid/post/0eule/cylA_119.pvtu
#
# Output layout:
#   malmo_runs_float_ablation/<MESH>__<variant>__<prec>__tol<E>/result.json

set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <MESH_TAG> <VTU_PATH>" >&2
  exit 2
fi

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
cd "$REPO_ROOT"

MESH_NAME="$1"
VTU="$2"

: "${JAXTRACE_VENV:=/flash/shared/jax/.venv}"
[[ -f "$JAXTRACE_VENV/bin/activate" ]] && source "$JAXTRACE_VENV/bin/activate"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/sweep_logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/float_ablation_${MESH_NAME}_${TS}.log"

if [[ "${SWEEP_LOG_ACTIVE:-0}" != "1" ]]; then
  export SWEEP_LOG_ACTIVE=1
  exec stdbuf -oL -eL bash -c "'$0' $*" 2>&1 | tee "$MAIN_LOG"
fi

echo "############################################################"
echo "# R1-g FLOAT-ABLATION SWEEP for $MESH_NAME"
echo "#   start : $(date)"
echo "#   host  : $(hostname)"
echo "############################################################"

for prec in fp32 fp64; do
  if [[ "$prec" == "fp32" ]]; then
    export JAX_ENABLE_X64=0
  else
    export JAX_ENABLE_X64=1
  fi
  echo
  echo "########  precision = $prec  (JAX_ENABLE_X64=$JAX_ENABLE_X64)  ########"

  for tol in 1e-4 1e-6 1e-8; do
    for variant in vertex_multi centroid aabb; do
      # Tolerance suffix without dots (1e-6 -> tol6, 1e-4 -> tol4)
      tol_tag="tol$(python3 -c "import math,sys; print(int(round(-math.log10(float(sys.argv[1])))))" "$tol")"
      out_dir="$REPO_ROOT/malmo_runs_float_ablation/${MESH_NAME}__${variant}__${prec}__${tol_tag}"
      if [[ -f "$out_dir/result.json" ]]; then
        echo "  skip $variant $prec $tol_tag (exists)"
        continue
      fi
      mkdir -p "$out_dir"
      echo
      echo "== $variant  $prec  tol=$tol =="
      python3 scripts/scone_bench/bench_malmo_pointloc.py \
          --vtu "$VTU" \
          --out-dir "$out_dir" \
          --variant "$variant" \
          --n-points 100000 \
          --sampling in_mesh \
          --tol "$tol" \
        || echo "  ($variant $prec $tol_tag exited non-zero; continuing)"
    done
  done
done

echo
echo "############################################################"
echo "# FLOAT-ABLATION SWEEP complete"
echo "#   end  : $(date)"
echo "#   log  : $MAIN_LOG"
echo "############################################################"
