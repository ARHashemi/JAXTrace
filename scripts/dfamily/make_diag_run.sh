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
#   JAXTRACE                <- pinned to the SAME repo the production array
#                              (sbatch_phase4_rk4_array.sh) used. The per-case
#                              run_jaxtrace.sh points at JAXTrace_stable, which
#                              predates the octree orphan_fallback fix and so
#                              DROPS non-Kuhn elements with no Kuhn neighbour
#                              from the octree entirely (>1.3M on D2, which is
#                              >50% non-Kuhn). A run on that build is NOT
#                              comparable to production: its ElementID<0 counts
#                              are inflated by elements never registered.
#   VEL_END                 <- trimmed so we load only as many velocity
#                              snapshots as the shortened run actually needs.
#                              The velocity sequence is CYCLIC, so a subset is
#                              physically valid: it just repeats sooner. This
#                              matters enormously -- each D2 snapshot costs
#                              ~1.4 min of Lustre reads (189 GB over 166
#                              snapshots), so loading all of them takes ~4 h
#                              while 1500 RK4 steps only span 0.19 of one full
#                              cycle. 30 snapshots = ~1.0 cycle for 1500 steps.
#
# Usage
#   scripts/dfamily/make_diag_run.sh <case.gid> [n_steps] [export_freq] [n_vel]
#
# Example
#   scripts/dfamily/make_diag_run.sh \
#     /scratch/project_465002752/lorenzgl/Cases/PinShapes/D-ConcavityTilt/D2.gid 1500 10
#
# Then:  cd <printed dir> && sbatch run_jaxtrace_diag.sh
set -euo pipefail

CASE_DIR="${1:?usage: make_diag_run.sh <case.gid> [n_steps] [export_freq] [n_vel]}"
N_STEPS="${2:-1500}"
EXPORT_FREQ="${3:-10}"
N_VEL="${4:-30}"          # how many velocity snapshots to load (0 = keep all)
# Match the production array's repo, NOT the case script's JAXTrace_stable.
JAXTRACE_REPO="${JAXTRACE_REPO:-/projappl/${PROJECT:-project_465002752}/hashemia/JAXTrace}"

CASE_DIR="$(cd "$CASE_DIR" && pwd -P)"
SRC="$CASE_DIR/run_jaxtrace.sh"
[ -f "$SRC" ] || { echo "ERROR: no run_jaxtrace.sh in $CASE_DIR" >&2; exit 2; }

CASE_NAME="$(basename "$CASE_DIR" .gid)"
# PROJECT may be exported in the environment as a full path
# (e.g. /project/project_465002752). Reduce it to the bare project id.
PROJECT="${PROJECT:-project_465002752}"
PROJECT="$(basename "$PROJECT")"
USER_NAME="${USER:-hashemia}"
DEST="/scratch/${PROJECT}/${USER_NAME}/dfamily/${CASE_NAME}_diag"

mkdir -p "$DEST"
cp "$SRC" "$DEST/run_jaxtrace_diag.sh"

python3 - "$DEST/run_jaxtrace_diag.sh" "$CASE_DIR" "$DEST" "$N_STEPS" "$EXPORT_FREQ" "$N_VEL" "$JAXTRACE_REPO" <<'PYEOF'
import re, sys
path, case_dir, dest, n_steps, export_freq, n_vel, jaxtrace_repo = sys.argv[1:8]
s = open(path, encoding="utf-8").read()

def setvar(src, name, value):
    """Replace the first assignment of NAME=, preserving any trailing comment.

    Match a full double-quoted value before falling back to a bare token:
    a plain \\S* stops at the first space, so a quoted multi-word value
    (SEED_BOX, SEED_GRID) would keep the tail of the OLD value after the new
    one and silently produce a wrong setting.
    """
    pat = re.compile(rf'^({name}=)("(?:[^"\\]|\\.)*"|\S*)(.*)$', re.M)
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
# ---- pin the repo to production's -------------------------------------------
# Production used REPO=/projappl/<proj>/<user>/JAXTrace (the dev repo, where
# orphan_fallback exists and every non-Kuhn element is registered). The case
# scripts point at JAXTrace_stable. Match production or the octree differs.
s, n_jt = re.subn(r'^(\s*)JAXTRACE="[^"]*"',
                  lambda m: f'{m.group(1)}JAXTRACE="{jaxtrace_repo}"',
                  s, count=1, flags=re.M)
if not n_jt:
    print("WARNING: no JAXTRACE= line found; repo not pinned", file=sys.stderr)

# ---- trim the velocity sequence ------------------------------------------
# Mesh loading, not tracking, dominates the wall clock: topology is read once
# but the velocity field is re-read from every snapshot in [VEL_START, VEL_END].
# The sequence is cycled (cycle_period = n_snapshots * velocity_dt), so taking
# a contiguous subset starting at VEL_START stays physically meaningful.
n_vel = int(n_vel)
vel_start = vel_end = None
m = re.search(r'^VEL_START=(\d+)', s, re.M)
if m:
    vel_start = int(m.group(1))
m = re.search(r'^VEL_END=(\d+)', s, re.M)
if m:
    vel_end = int(m.group(1))

n_vel_used = None
if n_vel > 0 and vel_start is not None and vel_end is not None:
    avail = vel_end - vel_start + 1
    n_vel_used = min(n_vel, avail)
    s = setvar(s, "VEL_END", str(vel_start + n_vel_used - 1))
elif n_vel > 0:
    print("WARNING: VEL_START/VEL_END not found; velocity sequence not trimmed",
          file=sys.stderr)


# Pin the output folder to a deterministic, named path so the user never
# has to hunt for a job-id folder. The case script leaves SCRATCH_FOLDER
# empty and falls back to "<case>_jaxtrace_<jobid>"; we set it explicitly.
base = dest.rsplit("/", 1)[0]
folder = dest.rsplit("/", 1)[1].replace("_diag", "") + "_results"
# Only the OUTPUT_TARGET=scratch branch defines SCRATCH_BASE=".../outputs";
# match that exact line so the OUTPUT_TARGET=case branch is left alone.
s, nb = re.subn(r'^(\s*)SCRATCH_BASE="/scratch/\$\{PROJECT\}/\$\{USER\}/outputs"',
                lambda m: f'{m.group(1)}SCRATCH_BASE="{base}"', s, count=1, flags=re.M)
s, nf = re.subn(r'^(\s*)SCRATCH_FOLDER=""',
                lambda m: f'{m.group(1)}SCRATCH_FOLDER="{folder}"', s, count=1, flags=re.M)
if not (nb and nf):
    raise SystemExit(f"ERROR: output-folder pin failed (base={nb}, folder={nf})")
s = setvar(s, "EXPORT_FORMAT",      "vtu")
s = setvar(s, "JOB_NAME",           "jaxtrace_diag") if "JOB_NAME=" in s else s

# keep the job short: the signature appears early
# Wall time = snapshot loading + octree build + tracking.
#   loading: ~1.4 min per snapshot off Lustre, doubled for slack (two jobs
#            contending for the same files roughly halves each one's rate).
#   octree : MEASURED at ~90+ min on a 10.8M-element D mesh. The cost is
#            build_node_to_elements (mesh_aligned_octree_vertex_multi.py:70),
#            a pure-Python loop over n_elements x 4 nodes building ~1.9M
#            python sets -- 43M iterations for D2. Both parent_cube and aabb
#            import it, so it is paid regardless of registration.
#   The first sweep attempt used a 45 min allowance and was at risk of
#   hitting its 3:30 wall with no VTUs written; budget 120 min instead.
_snaps = n_vel_used if n_vel_used else (
    vel_end - vel_start + 1 if (vel_end and vel_start) else 166)
_hours = max(2, int((_snaps * 1.4 * 2.0 + 120) // 60) + 1)
s = re.sub(r'^#SBATCH --time=\S+', f'#SBATCH --time={_hours:02d}:30:00',
           s, count=1, flags=re.M)
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
