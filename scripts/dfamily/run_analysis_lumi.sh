#!/usr/bin/env bash
#
# Run analyze_frozen_particles.py on LUMI inside the same singularity image
# and PYTHONPATH that run_jaxtrace.sh uses, so VTK/numpy resolve identically.
#
# Pattern copied from the per-case run_jaxtrace.sh:
#   SIF      lumi-jax-rocm singularity image
#   PKGS     /projappl/<proj>/hashemia/required-packages
#   JAXTRACE repo root, prepended to PYTHONPATH
#
# Usage
#   scripts/dfamily/run_analysis_lumi.sh <run_dir> [extra args to the python script]
#   scripts/dfamily/run_analysis_lumi.sh --script <other.py> [args...]
#
# With --script the named python file is run instead, and NO run_dir is
# prepended: pass whatever arguments that script expects. Used by
# registration_sweep.sh --compare, which takes several --run NAME=DIR pairs.
#
# Example
#   scripts/dfamily/run_analysis_lumi.sh \
#       /scratch/project_465002752/hashemia/outputs/D2_jaxtrace_1234567 \
#       --level-set /scratch/.../D2.gid/post/A2_34.pvtu \
#       --out docs/dfamily/02_D2_frozen_report.md
#
# Runs on the login node: it is pure file reading, no GPU, a few minutes.
set -euo pipefail

# --script <path> runs a different analysis script with no implicit run_dir.
SCRIPT_OVERRIDE=""
if [ "${1:-}" = "--script" ]; then
  SCRIPT_OVERRIDE="${2:?--script needs a path}"
  shift 2
  RUN_DIR=""
else
  RUN_DIR="${1:?usage: run_analysis_lumi.sh <run_dir> [args...]   or   --script <py> [args...]}"
  shift || true
fi

# PROJECT may be exported in the environment as a full path
# (e.g. /project/project_465002752). Reduce it to the bare project id.
PROJECT="${PROJECT:-project_465002752}"
PROJECT="$(basename "$PROJECT")"
SIF="${SIF:-/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif}"
PKGS="${PKGS:-/projappl/${PROJECT}/hashemia/required-packages}"

# Repo root = two levels up from this script
JAXTRACE="${JAXTRACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
SCRIPT="${SCRIPT_OVERRIDE:-$JAXTRACE/scripts/dfamily/analyze_frozen_particles.py}"

[ -f "$SCRIPT" ] || { echo "ERROR: not found: $SCRIPT" >&2; exit 2; }
if [ -n "$RUN_DIR" ] && [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: run dir not found: $RUN_DIR" >&2; exit 2
fi
[ -f "$SIF" ]     || { echo "ERROR: singularity image not found: $SIF" >&2; exit 2; }

echo "[env] SIF      $SIF"
echo "[env] JAXTRACE $JAXTRACE"
echo "[env] PKGS     $PKGS"
echo "[env] script   $SCRIPT"
[ -n "$RUN_DIR" ] && echo "[env] run dir  $RUN_DIR"
echo

singularity exec --cleanenv \
  --env PYTHONPATH="$JAXTRACE:$PKGS" \
  --env TF_CPP_MIN_LOG_LEVEL=2 \
  --env JAX_PLATFORMS=cpu \
  "$SIF" \
  python "$SCRIPT" ${RUN_DIR:+"$RUN_DIR"} "$@"
