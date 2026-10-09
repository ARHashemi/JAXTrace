#!/usr/bin/env bash
#
# ONE script to run the whole D-family diagnostic on LUMI.
#
#   step 1  prepare a diagnostic copy of the case run script (ElementID ON)
#   step 2  submit it
#   step 3  when it finishes, analyse the output
#
# The colleague's case folder is NEVER written to. Everything lands in
# /scratch/<project>/<user>/dfamily/.
#
# Usage
#   # prepare + submit (default)
#   scripts/dfamily/dfamily_diag.sh D2
#
#   # prepare only, do not submit
#   scripts/dfamily/dfamily_diag.sh D2 --no-submit
#
#   # analyse a finished run
#   scripts/dfamily/dfamily_diag.sh D2 --analyze <run_dir>
#
#   # different length
#   scripts/dfamily/dfamily_diag.sh D2 --steps 3000 --freq 20 --vel 40
#
set -euo pipefail

CASE="${1:?usage: dfamily_diag.sh <D1|D2|D3|D4|...> [options]}"; shift || true

# PROJECT may be exported in the environment as a full path
# (e.g. /project/project_465002752). Reduce it to the bare project id.
PROJECT="${PROJECT:-project_465002752}"
PROJECT="$(basename "$PROJECT")"
USER_NAME="${USER:-hashemia}"
CASES_ROOT="${CASES_ROOT:-/scratch/${PROJECT}/lorenzgl/Cases/PinShapes}"
FAMILY="${FAMILY:-D-ConcavityTilt}"
N_STEPS=1500
EXPORT_FREQ=10
# velocity snapshots to load. Mesh loading dominates wall clock (~1.4 min per
# snapshot on D2), and 1500 steps span ~1.0 cycle of 30 snapshots. 0 = all.
N_VEL=30
SUBMIT=1
ANALYZE_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --steps)     N_STEPS="$2"; shift 2 ;;
    --freq)      EXPORT_FREQ="$2"; shift 2 ;;
    --vel)       N_VEL="$2"; shift 2 ;;
    --no-submit) SUBMIT=0; shift ;;
    --analyze)
      if [ $# -ge 2 ] && [ "${2#-}" = "$2" ]; then ANALYZE_DIR="$2"; shift 2
      else ANALYZE_DIR="AUTO"; shift; fi ;;
    --family)    FAMILY="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
JAXTRACE="$(cd "$HERE/../.." && pwd -P)"
CASE_DIR="${CASES_ROOT}/${FAMILY}/${CASE}.gid"
DEST="/scratch/${PROJECT}/${USER_NAME}/dfamily/${CASE}_diag"

# ----------------------------------------------------------------- analyse --
# Default analyse target: the folder our own diagnostic run writes to.
RESULTS_PARENT="/scratch/${PROJECT}/${USER_NAME}/dfamily"
RESULTS_DIR="${RESULTS_PARENT}/${CASE}_results"

if [ "$ANALYZE_DIR" = "AUTO" ]; then
  if [ -d "$RESULTS_DIR" ]; then
    ANALYZE_DIR="$RESULTS_DIR"
  else
    echo "ERROR: no diagnostic results at $RESULTS_DIR" >&2
    echo "       Run '$0 $CASE' first, or pass a folder:" >&2
    echo "       $0 $CASE --analyze <dir>" >&2
    exit 2
  fi
fi

if [ -n "$ANALYZE_DIR" ]; then
  [ -d "$ANALYZE_DIR" ] || { echo "ERROR: not a directory: $ANALYZE_DIR" >&2; exit 2; }
  # mesh prefix is NOT always the case name: D2.gid/post holds A2_*.vtu
  MESH_PVTU=$(ls "${CASE_DIR}/post/"*_34.pvtu 2>/dev/null | head -1 || true)
  # Documents do NOT belong on LUMI. Write the report beside the results on
  # scratch; pull it into the local repo's docs/dfamily/ with scp.
  REPORT="${RESULTS_PARENT}/02_${CASE}_frozen_report.md"
  ARGS=( "$ANALYZE_DIR" --out "$REPORT" )
  if [ -n "$MESH_PVTU" ]; then
    echo "[mesh] level set from: $MESH_PVTU"
    ARGS+=( --level-set "$MESH_PVTU" )
  else
    echo "[mesh] WARNING: no *_34.pvtu under ${CASE_DIR}/post — running without --level-set"
  fi
  "$HERE/run_analysis_lumi.sh" "${ARGS[@]}"
  echo
  echo "Report written to (on LUMI scratch):"
  echo "  $REPORT"
  echo "Copy it into the local repo with:"
  echo "  scp lumi:$REPORT <local-repo>/docs/dfamily/"
  exit 0
fi

# ----------------------------------------------------------------- prepare --
[ -d "$CASE_DIR" ] || { echo "ERROR: case not found: $CASE_DIR" >&2; exit 2; }
"$HERE/make_diag_run.sh" "$CASE_DIR" "$N_STEPS" "$EXPORT_FREQ" "$N_VEL"

echo
echo "Diff against the original (review this):"
diff "${CASE_DIR}/run_jaxtrace.sh" "${DEST}/run_jaxtrace_diag.sh" || true
echo

if [ "$SUBMIT" = "1" ]; then
  cd "$DEST"
  JOB=$(sbatch --parsable run_jaxtrace_diag.sh)
  echo "Submitted job $JOB"
  echo
  echo "Watch:    squeue -u $USER_NAME"
  echo "Logs:     /scratch/${PROJECT}/${USER_NAME}/logs/jaxtrace_diag_${JOB}.out"
  echo
  echo "Results will be written to:"
  echo "  $RESULTS_DIR"
  echo
  echo "When it finishes:"
  echo "  $0 $CASE --analyze"
else
  echo "Not submitted (--no-submit). To submit:"
  echo "  cd $DEST && sbatch run_jaxtrace_diag.sh"
fi
