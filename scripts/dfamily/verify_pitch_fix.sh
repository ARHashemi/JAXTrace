#!/usr/bin/env bash
#
# Verify the per-axis-median pitch fix with a FULL production-length tracking
# run, built from the case's own run_jaxtrace.sh.
#
# What the fix is
# ---------------
# One line in jaxtrace/gpu/search/mesh_aligned_octree_gpu.py:
#     level_cell_sizes_cpu[level] = np.median(level_sizes, axis=0)
#                            (was: level_sizes[0])
#
# It is a PRE-BUILD change only. The time-marching path never touches cell
# sizes -- grep over jaxtrace/gpu/tracking/ for cell_size returns nothing --
# and all nine search sites already read level_cell_sizes[level] and divide
# each axis by its own pitch, so cuboid cells need no kernel change. There is
# no new branch anywhere in the per-step code, hence no throughput cost.
#
# Why the case script cannot be used unmodified
#   JAXTRACE=.../JAXTrace_stable  -> predates the fix AND drops non-Kuhn
#                                    orphans from the octree; the patch lives
#                                    in the dev repo that production used.
#   OUTPUT_TARGET=case            -> would write into the colleague's case
#                                    folder. We never touch those.
#   EXPORT_ELEMENT_IDS=0          -> without it the lost-particle count cannot
#                                    be measured at all.
# Everything else is left exactly as the case defines it: N_STEPS=8000,
# VEL_START/END=34..199 (the full time-dependent sequence), the case's own
# seeding, dt, pin kinematics and level-set mode.
#
# Usage
#   scripts/dfamily/verify_pitch_fix.sh D2                 # generate + submit
#   scripts/dfamily/verify_pitch_fix.sh D2 --no-submit      # generate only
#   scripts/dfamily/verify_pitch_fix.sh D2 --analyze        # after it finishes
#   scripts/dfamily/verify_pitch_fix.sh A1 --family A       # the control
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

CASE="${1:?usage: verify_pitch_fix.sh <CASE> [--family A|B|C|D] [--no-submit|--analyze]}"
shift || true

FAMILY="D"
SUBMIT=1
ANALYZE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --family)    FAMILY="$2"; shift 2 ;;
    --no-submit) SUBMIT=0; shift ;;
    --analyze)   ANALYZE=1; shift ;;
    *) echo "ERROR: unknown option '$1'" >&2; exit 2 ;;
  esac
done

case "$FAMILY" in
  A) FAM_DIR="A-FlatsVariations" ;;
  B) FAM_DIR="B-FluteVariations" ;;
  C) FAM_DIR="C-ThreadsVariations" ;;
  D) FAM_DIR="D-ConcavityTilt" ;;
  *) echo "ERROR: --family must be A, B, C or D" >&2; exit 2 ;;
esac

PROJECT="${PROJECT:-project_465002752}"
PROJECT="$(basename "$PROJECT")"
USER_NAME="${USER:-hashemia}"

# The patch lives in the DEV repo, which is also what the production p4rk4
# array used. JAXTrace_stable must not be used here.
JAXTRACE_REPO="${JAXTRACE_REPO:-/projappl/${PROJECT}/${USER_NAME}/JAXTrace}"

CASE_DIR="/scratch/${PROJECT}/lorenzgl/Cases/PinShapes/${FAM_DIR}/${CASE}.gid"
ROOT="/scratch/${PROJECT}/${USER_NAME}/dfamily/pitchfix_${CASE}"
RESULTS="${ROOT}/${CASE}_results"
DEST="${ROOT}/${CASE}_run"

# ------------------------------------------------------------------ analyse --
if [ "$ANALYZE" = "1" ]; then
  [ -d "$RESULTS" ] || { echo "ERROR: no results at $RESULTS" >&2; exit 2; }
  MESH_PVTU=$(ls "${CASE_DIR}/post/"*_34.pvtu 2>/dev/null | head -1 || true)
  ARGS=( "$RESULTS" --out "${ROOT}/06_${CASE}_pitchfix_report.md" )
  [ -n "$MESH_PVTU" ] && ARGS+=( --level-set "$MESH_PVTU" )
  "$HERE/run_analysis_lumi.sh" "${ARGS[@]}"
  echo
  echo "Report: ${ROOT}/06_${CASE}_pitchfix_report.md"
  echo "Pull it local with:"
  echo "  scp lumi:${ROOT}/06_${CASE}_pitchfix_report.md <repo>/docs/dfamily/"
  exit 0
fi

# ----------------------------------------------------------------- generate --
[ -d "$CASE_DIR" ] || { echo "ERROR: case not found: $CASE_DIR" >&2; exit 2; }
SRC="$CASE_DIR/run_jaxtrace.sh"
[ -f "$SRC" ] || { echo "ERROR: no run_jaxtrace.sh in $CASE_DIR" >&2; exit 2; }

mkdir -p "$DEST"
cp "$SRC" "$DEST/run_jaxtrace_pitchfix.sh"

python3 - "$DEST/run_jaxtrace_pitchfix.sh" "$CASE_DIR" "$ROOT" "$CASE" \
         "$JAXTRACE_REPO" <<'PYEOF'
import os, re, sys
path, case_dir, root, case, repo = sys.argv[1:6]
s = open(path, encoding="utf-8").read()

def setvar(src, var, value):
    """Replace the first assignment of VAR=, keeping any trailing comment.

    Matches a full double-quoted value before a bare token: a plain \S* stops
    at the first space and would leave the tail of a quoted multi-word value
    (SEED_BOX, SEED_GRID) dangling after the new one.
    """
    pat = re.compile(rf'^({var}=)("(?:[^"\\]|\\.)*"|\S*)(.*)$', re.M)
    out, n = pat.subn(lambda m: f"{m.group(1)}{value}{m.group(3)}", src, count=1)
    if not n:
        print(f"WARNING: {var}= not found in the case script", file=sys.stderr)
    return out

# The three changes that are strictly necessary, and nothing else.
s = setvar(s, "EXPORT_ELEMENT_IDS", "1")      # without this nothing is measurable
s = setvar(s, "OUTPUT_TARGET",      "scratch")  # never write into the case folder
s = setvar(s, "AUTO_DETECT_CASE",   "0")
s = setvar(s, "INPUT",              f'"{case_dir}"')
s = setvar(s, "RUN_TAG",            f'"pitchfix_{case}"')
s = setvar(s, "EXPORT_FORMAT",      "vtu")
s = setvar(s, "LOG_INTERVAL",       "100")

# Unbuffered python, so the log is readable DURING the run instead of only
# after it exits. The case scripts invoke a plain `python`, whose stdout is
# block-buffered when redirected to a file; on an 8000-step run that means the
# log sits at ~1.3 kB for hours and the only way to tell which phase the job is
# in is to probe /proc for open file descriptors and rchar growth. The
# production array (sbatch_phase4_rk4_array.sh:281,284) already used both
# `srun --unbuffered` and `python3 -u`; the per-case scripts do not.
#
# PYTHONUNBUFFERED is set via the environment rather than by editing the
# command line, so it survives regardless of how the script spells the python
# invocation.
s, nu = re.subn(r'^(\s*)python (\$JAXTRACE/run_tracking\.py)',
                lambda m: f'{m.group(1)}python -u {m.group(2)}', s, count=1,
                flags=re.M)
if not nu:
    print("WARNING: could not add -u to the python invocation", file=sys.stderr)

# Also ask srun not to buffer, and export PYTHONUNBUFFERED into the container.
s, ns = re.subn(r'^(\s*)srun (?!--unbuffered)(\S)',
                lambda m: f'{m.group(1)}srun --unbuffered {m.group(2)}', s,
                count=1, flags=re.M)
s, ne = re.subn(r'^(\s*)--env TF_CPP_MIN_LOG_LEVEL=',
                lambda m: f'{m.group(1)}--env PYTHONUNBUFFERED=1 \\\n'
                          f'{m.group(1)}--env TF_CPP_MIN_LOG_LEVEL=',
                s, count=1, flags=re.M)
print(f"  unbuffered: python -u={bool(nu)} srun={bool(ns)} PYTHONUNBUFFERED={bool(ne)}")

# EXPORT_FREQ: the case default is 1, i.e. a 10-14 MB VTU every step = 8,000
# files and ~80-110 GB for an 8000-step run. That made job 22638923 write-bound
# at 0.4-0.5 step/s. Nothing in this verification needs per-step geometry: the
# loss counts come from the per-step "active/lost" log line and
# search_stats.csv, and the final loss GEOMETRY only needs the last few files.
# Export every 25 steps (321 files) instead, unless EXPORT_FREQ is overridden.
s = setvar(s, "EXPORT_FREQ", os.environ.get("EXPORT_FREQ", "25"))

# Pin the repo: the case script points at JAXTrace_stable, which predates the
# fix and also drops non-Kuhn orphans, so it would measure the wrong thing.
s, n = re.subn(r'^(\s*)JAXTRACE="[^"]*"',
               lambda m: f'{m.group(1)}JAXTRACE="{repo}"', s, count=1, flags=re.M)
if not n:
    raise SystemExit("ERROR: no JAXTRACE= line found; refusing to run on the "
                     "wrong repo")

# Output location, pinned so --analyze needs no job-id hunt.
s, nb = re.subn(r'^(\s*)SCRATCH_BASE="/scratch/\$\{PROJECT\}/\$\{USER\}/outputs"',
                lambda m: f'{m.group(1)}SCRATCH_BASE="{root}"', s, count=1, flags=re.M)
s, nf = re.subn(r'^(\s*)SCRATCH_FOLDER=""',
                lambda m: f'{m.group(1)}SCRATCH_FOLDER="{case}_results"', s,
                count=1, flags=re.M)
if not (nb and nf):
    raise SystemExit(f"ERROR: output pin failed (base={nb}, folder={nf})")

# --hit-stats-log is deliberately NOT added.
#
# It killed job 22638923 at step 899 of 8000, after the tracking itself had run
# cleanly with zero losses:
#     Failed to load HSACO: HIP_ERROR_NoBinaryForGpu
#     run_tracking.py:3043 -> _hit_probe_closures.hit_probe_step(...)
# The probe compiles a SEPARATE GPU kernel from the tracking step, and on this
# ROCm/JAX build that kernel intermittently fails to load. The failure is in
# the diagnostic, not in the physics or the search, but it aborts the run.
#
# The per-step "active=N lost=M (+K)" line that run_tracking.py always prints,
# plus search_stats.csv, already give the loss counts we need. Set
# HIT_STATS=1 in the environment to re-enable the probe if you accept the risk.
if os.environ.get("HIT_STATS") == "1":
    anchor = 'ARGS+=( --export-format "$EXPORT_FORMAT" )'
    if anchor not in s:
        raise SystemExit("ERROR: ARGS+=( --export-format ... ) anchor not found")
    s = s.replace(anchor, anchor + '\nARGS+=( --hit-stats-log )', 1)
    print("  NOTE: --hit-stats-log ENABLED via HIT_STATS=1 (crashed job 22638923)",
          file=sys.stderr)

# Wall time. Full-length run: 166 snapshots of Lustre reads (~1.4 min each,
# doubled for slack) + ~120 min octree build on a 10.8M-element mesh + 8000
# tracking steps. The first D2 diagnostic died at a 1h30 limit with nothing
# written, so budget generously -- this is a one-off verification.
s = re.sub(r'^#SBATCH --time=\S+', '#SBATCH --time=12:00:00', s, count=1, flags=re.M)
s = re.sub(r'^#SBATCH --job-name=\S+', f'#SBATCH --job-name=pitchfix_{case}',
           s, count=1, flags=re.M)

open(path, "w", encoding="utf-8").write(s)
PYEOF

echo "Case     : $CASE_DIR"
echo "Repo     : $JAXTRACE_REPO  (must contain the median-pitch fix)"
echo "Run dir  : $DEST"
echo "Results  : $RESULTS"
echo
echo "Diff against the case's own script (review this):"
diff "$SRC" "$DEST/run_jaxtrace_pitchfix.sh" || true
echo

if [ "$SUBMIT" = "1" ]; then
  JOB=$(cd "$DEST" && sbatch --parsable run_jaxtrace_pitchfix.sh)
  echo "Submitted job $JOB"
  echo
  echo "Watch:   squeue -u $USER_NAME"
  echo "Logs:    /scratch/${PROJECT}/${USER_NAME}/logs/pitchfix_${CASE}_${JOB}.out"
  echo "Analyse: $0 $CASE --family $FAMILY --analyze"
else
  echo "Not submitted. To submit:"
  echo "  cd $DEST && sbatch run_jaxtrace_pitchfix.sh"
fi
