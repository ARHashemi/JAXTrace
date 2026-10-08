#!/usr/bin/env bash
#
# Build a DIAGNOSTIC copy of a case's run_jaxtrace.sh under the user's own
# scratch, with ElementID export turned on. The original case folder is
# never touched.
#
# What it changes vs the case's own script:
#   EXPORT_ELEMENT_IDS=1    <- the whole point: lets us tell a search failure
#                              (ElementID<0) from a zero velocity (>=0)
#   OUTPUT_TARGET=scratch   <- write to /scratch/<proj>/<user>/dfamily/...
#                              instead of into the colleague's case folder
#   N_STEPS / EXPORT_FREQ   <- shortened by default; the freeze signature is
#                              already obvious within a few hundred steps
#
# Usage
#   scripts/dfamily/make_diag_run.sh <case.gid> [n_steps] [export_freq]
#
# Example
#   scripts/dfamily/make_diag_run.sh \
#     /scratch/project_465002752/lorenzgl/Cases/PinShapes/D-ConcavityTilt/D2.gid 1500 10
#
# Then:  cd <printed dir> && sbatch run_jaxtrace_diag.sh
set -euo pipefail

CASE_DIR="${1:?usage: make_diag_run.sh <case.gid> [n_steps] [export_freq]}"
N_STEPS="${2:-1500}"
EXPORT_FREQ="${3:-10}"

CASE_DIR="$(cd "$CASE_DIR" && pwd -P)"
SRC="$CASE_DIR/run_jaxtrace.sh"
[ -f "$SRC" ] || { echo "ERROR: no run_jaxtrace.sh in $CASE_DIR" >&2; exit 2; }

CASE_NAME="$(basename "$CASE_DIR" .gid)"
PROJECT="${PROJECT:-project_465002752}"
USER_NAME="${USER:-hashemia}"
DEST="/scratch/${PROJECT}/${USER_NAME}/dfamily/${CASE_NAME}_diag"

mkdir -p "$DEST"
cp "$SRC" "$DEST/run_jaxtrace_diag.sh"

python3 - "$DEST/run_jaxtrace_diag.sh" "$CASE_DIR" "$DEST" "$N_STEPS" "$EXPORT_FREQ" <<'PYEOF'
import re, sys
path, case_dir, dest, n_steps, export_freq = sys.argv[1:6]
s = open(path, encoding="utf-8").read()

def setvar(src, name, value):
    """Replace the first assignment of NAME=, preserving any trailing comment."""
    pat = re.compile(rf'^({name}=)(\S*)(.*)$', re.M)
    if not pat.search(src):
        raise SystemExit(f"ERROR: {name}= not found in run script")
    return pat.sub(lambda m: f"{m.group(1)}{value}{m.group(3)}", src, count=1)

s = setvar(s, "EXPORT_ELEMENT_IDS", "1")
s = setvar(s, "N_STEPS",            n_steps)
s = setvar(s, "EXPORT_FREQ",        export_freq)
s = setvar(s, "OUTPUT_TARGET",      "scratch")
s = setvar(s, "AUTO_DETECT_CASE",   "0")
s = setvar(s, "INPUT",              f'"{case_dir}"')
s = setvar(s, "RUN_TAG",            '"dfamily_diag"')
s = setvar(s, "EXPORT_FORMAT",      "vtu")
s = setvar(s, "JOB_NAME",           "jaxtrace_diag") if "JOB_NAME=" in s else s

# keep the job short: the signature appears early
s = re.sub(r'^#SBATCH --time=\S+', '#SBATCH --time=01:30:00', s, count=1, flags=re.M)
s = re.sub(r'^#SBATCH --job-name=\S+', '#SBATCH --job-name=jaxtrace_diag', s, count=1, flags=re.M)

open(path, "w", encoding="utf-8").write(s)
print("  patched:", path)
PYEOF

chmod +x "$DEST/run_jaxtrace_diag.sh"

cat <<EOF

Diagnostic run prepared.

  case   : $CASE_DIR   (untouched)
  script : $DEST/run_jaxtrace_diag.sh
  steps  : $N_STEPS   export every $EXPORT_FREQ

Changed from the original: EXPORT_ELEMENT_IDS=1, OUTPUT_TARGET=scratch,
N_STEPS=$N_STEPS, EXPORT_FREQ=$EXPORT_FREQ, time limit 1h30.

Review the diff first:
  diff "$SRC" "$DEST/run_jaxtrace_diag.sh"

Submit:
  cd "$DEST" && sbatch run_jaxtrace_diag.sh

When it finishes, analyse with:
  python scripts/dfamily/analyze_frozen_particles.py \\
      /scratch/${PROJECT}/${USER_NAME}/outputs/<run_folder> \\
      --level-set ${CASE_DIR}/post/<MESH>_34.pvtu \\
      --out docs/dfamily/02_${CASE_NAME}_frozen_report.md
EOF
