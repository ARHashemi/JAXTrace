#!/usr/bin/env bash
#
# Registration sweep: does the octree registration strategy explain the
# D-family host-loss, or is the SEARCH itself at fault?
#
# Background
# ----------
# The revised paper's Table T2 (paper_table_found.md) measures location
# correctness per registration method over the 10-mesh R2-6 cohort, which
# INCLUDES fully non-Kuhn meshes (Bunny 100%, Porous media 100%,
# Microfluidics 100%, FP-8.7k 93.8% non-Kuhn):
#
#     MALMO aabb          100.00% on all 10 meshes
#     MALMO centroid       99.64% worst (Poly940)
#     MALMO vertex_multi   96.12% worst (Bunny)
#
# So `aabb` is the only coverage-complete option, and `centroid`/parent_cube
# already loses queries in the paper's own data. The D-family meshes are
# 50.6-59.3% non-Kuhn (vs 0.06% for the FSW/cylA mesh), i.e. squarely in the
# regime where the registration choice matters.
#
# What this sweep does
# --------------------
# Runs the SAME case with only the registration differing, each with
# EXPORT_ELEMENT_IDS=1, then compares the frozen counts and the loss annulus.
#
#   aabb          every element in every cell its AABB overlaps  (coverage-complete)
#   parent_cube   one parent cube; non-Kuhn borrow a neighbour   (the production default)
#   pc-nohybrid   parent_cube with --no-hybrid-non-kuhn          (isolates the AABB-overlap fix)
#   pc-noorphan   parent_cube with --no-orphan-fallback          (isolates the orphan fix)
#   vertex_multi  every cell its 4 vertices touch                (the legacy approach)
#
# Reading the result
#   aabb ~0 frozen while parent_cube loses thousands
#       -> registration coverage is the root cause; the search kernel is fine,
#          and the fix is --registration aabb (or porting hybrid+orphan).
#   aabb still loses particles in the same r 7-9mm annulus
#       -> the defect is in the SEARCH, and the band settings become relevant.
#
# Cost control (why this is cheap)
# --------------------------------
# The losses are already localised: they occur at r 6.4-9.5 mm, z -4..0 mm,
# and the particles that suffer them START at r 11.1-14.2 mm and need ~670
# steps to advect in. Seeding DIRECTLY in the annulus removes that travel, so
# a few hundred steps suffice instead of 1500. The D family needs the real
# time-dependent velocity sequence (NOT a single snapshot), so VEL stays at a
# sequence; --steps and the tight seed box are what keep it short.
#
# Usage
#   scripts/dfamily/registration_sweep.sh D2                 # all variants
#   scripts/dfamily/registration_sweep.sh D2 --only aabb,parent_cube
#   scripts/dfamily/registration_sweep.sh D2 --steps 400 --vel 30
#   scripts/dfamily/registration_sweep.sh D2 --no-submit     # generate only
#   scripts/dfamily/registration_sweep.sh D2 --compare       # analyse results
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
JAXTRACE="$(cd "$HERE/../.." && pwd -P)"

CASE="${1:?usage: registration_sweep.sh <CASE> [options]   e.g. D2}"
shift || true

# PROJECT may be exported as a full path (/project/project_465002752).
PROJECT="${PROJECT:-project_465002752}"
PROJECT="$(basename "$PROJECT")"
USER_NAME="${USER:-hashemia}"

# Production used the DEV repo, not JAXTrace_stable (which predates the
# orphan_fallback / hybrid_non_kuhn fixes and silently drops elements).
JAXTRACE_REPO="${JAXTRACE_REPO:-/projappl/${PROJECT}/${USER_NAME}/JAXTrace}"

N_STEPS=400
EXPORT_FREQ=10
N_VEL=30
SUBMIT=1
COMPARE=0
ONLY=""

# Concentrated seed shell around the measured loss annulus, in METRES.
# Measured from job 22614710: losses at r 6.38-9.50 mm, z -3.0..0 mm.
# A slightly wider box so the annulus is fully bracketed; seeds landing
# inside the tool simply fail host assignment and are excluded by analysis.
SEED_BOX="-0.0100 0.0100 -0.0100 0.0100 -0.0045 -0.0001"
SEED_GRID="80 80 24"          # 153,600 seeds, ~50% inside the r[6,10] shell

while [ $# -gt 0 ]; do
  case "$1" in
    --steps)     N_STEPS="$2"; shift 2 ;;
    --freq)      EXPORT_FREQ="$2"; shift 2 ;;
    --vel)       N_VEL="$2"; shift 2 ;;
    --only)      ONLY="$2"; shift 2 ;;
    --seed-box)  SEED_BOX="$2"; shift 2 ;;
    --seed-grid) SEED_GRID="$2"; shift 2 ;;
    --no-submit) SUBMIT=0; shift ;;
    --compare)   COMPARE=1; shift ;;
    *) echo "ERROR: unknown option '$1'" >&2; exit 2 ;;
  esac
done

CASE_DIR="/scratch/${PROJECT}/lorenzgl/Cases/PinShapes/D-ConcavityTilt/${CASE}.gid"
SWEEP_ROOT="/scratch/${PROJECT}/${USER_NAME}/dfamily/sweep_${CASE}"

# variant : registration : extra run_tracking flags
# --hit-stats-log is added to EVERY variant: at each --log-interval step it
# classifies the surviving particles by which search level (L0 cached element /
# L1 neighbour hop / L2 global octree / miss) would find them right now, into
# hit_stats.csv. That is the intermediate logging that says WHICH stage fails,
# rather than only that the final host was lost. run_tracking.py also always
# writes search_stats.csv with per-step n_lost/new_lost.
HIT="--hit-stats-log"
VARIANTS=(
  "aabb:aabb:$HIT"
  "parent_cube:parent_cube:$HIT"
  "pc-nohybrid:parent_cube:$HIT --no-hybrid-non-kuhn"
  "pc-noorphan:parent_cube:$HIT --no-orphan-fallback"
  "vertex_multi:vertex_multi:$HIT"
)

# ------------------------------------------------------------------ compare --
if [ "$COMPARE" = "1" ]; then
  echo "Comparing registration variants under $SWEEP_ROOT"
  echo
  ARGS=()
  for v in "${VARIANTS[@]}"; do
    name="${v%%:*}"
    d="${SWEEP_ROOT}/${name}_results"
    [ -d "$d" ] && ARGS+=( --run "${name}=${d}" )
  done
  if [ ${#ARGS[@]} -eq 0 ]; then
    echo "ERROR: no results under $SWEEP_ROOT — run the sweep first." >&2
    exit 2
  fi
  MESH_PVTU=$(ls "${CASE_DIR}/post/"*_34.pvtu 2>/dev/null | head -1 || true)
  [ -n "$MESH_PVTU" ] && ARGS+=( --level-set "$MESH_PVTU" )
  ARGS+=( --out "${SWEEP_ROOT}/03_${CASE}_registration_sweep.md" )
  exec "$HERE/run_analysis_lumi.sh" --script "$HERE/compare_registration.py" "${ARGS[@]}"
fi

# ----------------------------------------------------------------- generate --
[ -d "$CASE_DIR" ] || { echo "ERROR: case not found: $CASE_DIR" >&2; exit 2; }
SRC="$CASE_DIR/run_jaxtrace.sh"
[ -f "$SRC" ] || { echo "ERROR: no run_jaxtrace.sh in $CASE_DIR" >&2; exit 2; }

mkdir -p "$SWEEP_ROOT"
echo "Sweep root : $SWEEP_ROOT"
echo "Case       : $CASE_DIR"
echo "Repo       : $JAXTRACE_REPO"
echo "Steps      : $N_STEPS (export every $EXPORT_FREQ), velocity snapshots: $N_VEL"
echo "Seed box   : $SEED_BOX   grid $SEED_GRID"
echo

JOBS=()
for v in "${VARIANTS[@]}"; do
  name="${v%%:*}"; rest="${v#*:}"
  reg="${rest%%:*}"; extra="${rest#*:}"

  if [ -n "$ONLY" ] && ! printf '%s' ",$ONLY," | grep -q ",$name,"; then
    continue
  fi

  DEST="${SWEEP_ROOT}/${name}_diag"
  RESULTS="${SWEEP_ROOT}/${name}_results"
  mkdir -p "$DEST"
  cp "$SRC" "$DEST/run_jaxtrace_sweep.sh"

  python3 - "$DEST/run_jaxtrace_sweep.sh" "$CASE_DIR" "$SWEEP_ROOT" "$name" \
           "$N_STEPS" "$EXPORT_FREQ" "$N_VEL" "$JAXTRACE_REPO" \
           "$reg" "$extra" "$SEED_BOX" "$SEED_GRID" <<'PYEOF'
import re, sys
(path, case_dir, sweep_root, name, n_steps, export_freq, n_vel,
 repo, reg, extra, seed_box, seed_grid) = sys.argv[1:13]
s = open(path, encoding="utf-8").read()

def setvar(src, var, value):
    """Replace the first assignment of VAR=, preserving any trailing comment.

    The value may be a QUOTED string containing spaces (SEED_BOX, SEED_GRID),
    so match a full double-quoted value first and only then fall back to a
    bare token. A plain \\S* would stop at the first space and leave the rest
    of the old value dangling after the new one.
    """
    pat = re.compile(rf'^({var}=)("(?:[^"\\]|\\.)*"|\S*)(.*)$', re.M)
    out, n = pat.subn(lambda m: f"{m.group(1)}{value}{m.group(3)}", src, count=1)
    if not n:
        print(f"WARNING: {var}= not found in the case script", file=sys.stderr)
    return out

s = setvar(s, "EXPORT_ELEMENT_IDS", "1")
s = setvar(s, "N_STEPS",            n_steps)
s = setvar(s, "EXPORT_FREQ",        export_freq)
s = setvar(s, "OUTPUT_TARGET",      "scratch")
s = setvar(s, "AUTO_DETECT_CASE",   "0")
s = setvar(s, "INPUT",              f'"{case_dir}"')
s = setvar(s, "RUN_TAG",            f'"sweep_{name}"')
s = setvar(s, "EXPORT_FORMAT",      "vtu")
s = setvar(s, "REGISTRATION",       f'"{reg}"')
# hit_stats.csv rows are written every LOG_INTERVAL steps; 25 gives ~16 rows
# across a 400-step run instead of 4.
s = setvar(s, "LOG_INTERVAL",       "25")

# Concentrated seeding: start the particles IN the measured loss annulus so
# the failure appears within a few hundred steps instead of ~670.
s = setvar(s, "SEED_SOURCE", "grid")
s = setvar(s, "SEED_BOX",    f'"{seed_box}"')
s = setvar(s, "SEED_GRID",   f'"{seed_grid}"')

# Pin the repo: the case script points at JAXTrace_stable, which lacks
# hybrid_non_kuhn / orphan_fallback entirely and DROPS non-Kuhn orphans.
s, n_jt = re.subn(r'^(\s*)JAXTRACE="[^"]*"',
                  lambda m: f'{m.group(1)}JAXTRACE="{repo}"', s, count=1, flags=re.M)
if not n_jt:
    print("WARNING: no JAXTRACE= line found; repo not pinned", file=sys.stderr)

# Trim the velocity sequence. The D family needs a time-DEPENDENT field, so we
# keep a sequence (never a single snapshot); n_vel only bounds how much of the
# cyclic sequence is loaded, since each snapshot costs ~1.4 min of Lustre reads.
n_vel = int(n_vel)
vs = re.search(r'^VEL_START=(\d+)', s, re.M)
ve = re.search(r'^VEL_END=(\d+)',   s, re.M)
if n_vel > 0 and vs and ve:
    start, end = int(vs.group(1)), int(ve.group(1))
    used = min(n_vel, end - start + 1)
    s = setvar(s, "VEL_END", str(start + used - 1))
else:
    used = (int(ve.group(1)) - int(vs.group(1)) + 1) if (vs and ve) else 166

# Pin the output so the comparison tool can find it without a job-id hunt.
s, nb = re.subn(r'^(\s*)SCRATCH_BASE="/scratch/\$\{PROJECT\}/\$\{USER\}/outputs"',
                lambda m: f'{m.group(1)}SCRATCH_BASE="{sweep_root}"', s, count=1, flags=re.M)
s, nf = re.subn(r'^(\s*)SCRATCH_FOLDER=""',
                lambda m: f'{m.group(1)}SCRATCH_FOLDER="{name}_results"', s, count=1, flags=re.M)
if not (nb and nf):
    raise SystemExit(f"ERROR: output pin failed (base={nb}, folder={nf})")

# Extra per-variant run_tracking flags. The case script builds its argument
# list as `ARGS+=( ... )` lines, so append a matching line rather than trying
# to splice a bare flag into an existing one.
if extra:
    anchor = 'ARGS+=( --export-format "$EXPORT_FORMAT" )'
    if anchor not in s:
        raise SystemExit("ERROR: ARGS+=( --export-format ...) anchor not found; "
                         "cannot inject extra flags safely")
    s = s.replace(anchor, anchor + f'\nARGS+=( {extra} )', 1)

# Wall time = loading + octree build + tracking.
#   loading: ~1.4 min/snapshot off Lustre, doubled (concurrent sweep jobs
#            contend for the same files).
#   octree : MEASURED >90 min on a 10.8M-element D mesh. The dominant cost is
#            build_node_to_elements (vertex_multi.py:70), a pure-Python loop
#            over n_elements x 4 nodes -- 43M iterations for D2 -- which BOTH
#            parent_cube and aabb import, so registration choice barely
#            changes it. A 60 min allowance was too small and risked the
#            3:30 wall with nothing written.
hours = max(3, int((used * 1.4 * 2.0 + 150) // 60) + 1)
s = re.sub(r'^#SBATCH --time=\S+', f'#SBATCH --time={hours:02d}:30:00', s, count=1, flags=re.M)
s = re.sub(r'^#SBATCH --job-name=\S+', f'#SBATCH --job-name=sweep_{name}', s, count=1, flags=re.M)

open(path, "w", encoding="utf-8").write(s)
print(f"  [{name}] registration={reg} {extra}  time={hours:02d}:30  snapshots={used}")
PYEOF

  if [ "$SUBMIT" = "1" ]; then
    JOB=$(cd "$DEST" && sbatch --parsable run_jaxtrace_sweep.sh)
    JOBS+=("$name=$JOB")
    echo "  [$name] submitted job $JOB -> $RESULTS"
  else
    echo "  [$name] generated $DEST/run_jaxtrace_sweep.sh (not submitted)"
  fi
done

echo
if [ "$SUBMIT" = "1" ] && [ ${#JOBS[@]} -gt 0 ]; then
  printf 'Submitted: %s\n' "${JOBS[*]}"
  echo
  echo "Watch:   squeue -u $USER_NAME"
  echo "Compare: $0 $CASE --compare"
else
  echo "Nothing submitted. To submit: $0 $CASE"
fi
