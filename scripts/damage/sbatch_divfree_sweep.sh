#!/usr/bin/env bash
#SBATCH --job-name=divfree
#SBATCH --partition=small
#SBATCH --account=project_465002752
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --array=0-16
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=/scratch/project_465002752/hashemia/logs/%x_%A_%a.out
#SBATCH --error=/scratch/project_465002752/hashemia/logs/%x_%A_%a.err
# =============================================================================
# ⚠️⚠️ SUPERSEDED AND DISABLED. DO NOT RUN THIS TO PRODUCE A FIELD.
#
# The BDM1 patch solve this drives produces a field that is divergence-free
# but whose GRADIENT is ~1e6 too large and ORTHOGONAL to the FOM's
# (cos = 0.0016, measured on the shipped C1_34.npz over 7.59M elements). It
# matches the centroid velocity only by cancellation between two huge wrong
# terms. Tracking integrates the GRADIENT -- det F and every damage
# accumulator depend on A -- so the output is UNUSABLE however clean its
# divergence looks. See docs/damage/DIVERGENCE_FREE_PLAN.md 4x.
#
# ✅ THE REPLACEMENT is the closed form A' = A - (tr A/3) I, applied in-kernel
# by run_tracking.py --divergence-free (4y, 4ab). It is better on every axis
# at once: divergence 3.0e-18 vs 9.2e-17, gradient cos 0.99998 vs 0.0016, and
# no solve at all -- 13 s per snapshot against 1511 s on 16 cores, or zero
# extra cost when done in-kernel.
#
# This script is kept because the diagnostics built on it (4t-4x) are the
# evidence for WHY the patch approach fails, and that reasoning is worth
# keeping. It refuses to run without ALLOW_SUPERSEDED=1.
# =============================================================================
if [ "${ALLOW_SUPERSEDED:-0}" != "1" ]; then
    cat >&2 <<'MSG'
REFUSING TO RUN: this sweep produces a field whose gradient is ~1e6 too large
and orthogonal to the FOM's, which is unusable for tracking or damage.
Use  run_tracking.py --divergence-free  instead (DIVERGENCE_FREE_PLAN.md 4y).
To run anyway for diagnostic purposes:  ALLOW_SUPERSEDED=1 sbatch ...
MSG
    exit 2
fi
# =============================================================================
# BDM1 divergence-free projection across the pin-shape families.
#
#   sbatch sbatch_divfree_sweep.sh                # all 16 cases
#   sbatch --array=0-3 sbatch_divfree_sweep.sh    # D family only
#
# ONE CASE PER ARRAY TASK, 3 snapshots each. The point is to find out whether
# the PRESERVATION quality measured on cylA (p50 0.186) and D2 (p50 4.7e-04)
# is case-dependent, and therefore whether it can affect tracking and damage.
#
# ⚠️ CPU-ONLY. The projection is scipy sparse LU with NO GPU use, and splu is
# single-threaded -- which is why --cpus-per-task=16 and --jobs 16 matter: the
# parallelism is ACROSS PATCHES, not inside the solve.
#
# ⚠️ MEMORY. Measured on the workstation: D2 (10.8M elements) peaked well
# inside 58 GB with 12 workers. 120G with 16 workers is deliberate headroom,
# since a patch that lands in a refined region can be larger than average.
# The per-task log records peak RSS so the next sweep can be sized from data.
#
# ⚠️ TIME. MEASURED ON LUMI (pilot job 23058480_8, case C1, 7.59M elements):
# 1511 s solve + load = 1813 s wall for ONE snapshot at 1322 % CPU. So 3
# snapshots is ~90 min and 6 h is ~4x headroom. Peak RSS 10.4 GB per the
# time -v report; the node showed 67 GB actually resident across 18 forked
# workers, so 120G is ample.
#
# ⚠️ The array is 0-16 to cover all 17 *.gid dirs. Two exit cleanly without
# work: A1 has NO pvtu output at all, and D2old is superseded.
# =============================================================================
set -uo pipefail

REPO="${REPO:-/projappl/project_465002752/hashemia/JAXTrace}"
CASES_ROOT="${CASES_ROOT:-/scratch/project_465002752/lorenzgl/Cases/PinShapes}"
OUT_ROOT="${OUT_ROOT:-/scratch/project_465002752/hashemia/divfree}"
SNAPSHOTS="${SNAPSHOTS:-34 100 199}"
SEED_SIZE="${SEED_SIZE:-10000}"
LAYERS="${LAYERS:-3}"
JOBS="${JOBS:-16}"

mkdir -p "$OUT_ROOT" /scratch/project_465002752/hashemia/logs

# ⚠️ The case list must match --array. Built from the directory listing so it
# cannot silently drift from what is on disk.
mapfile -t CASE_DIRS < <(ls -d "$CASES_ROOT"/*/*.gid 2>/dev/null | sort)
N=${#CASE_DIRS[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "$N" ]; then
    echo "task $IDX >= $N cases found under $CASES_ROOT -- nothing to do"
    exit 0
fi
CASE_DIR="${CASE_DIRS[$IDX]}"
CASE=$(basename "$CASE_DIR" .gid)
echo "=========================================================="
echo " task $IDX/$N   case $CASE"
echo " dir   $CASE_DIR"
echo " node  $(hostname)   cpus ${SLURM_CPUS_PER_TASK:-?}   mem ${SLURM_MEM_PER_NODE:-?}M"
echo " start $(date -Is)"
echo "=========================================================="

# ⚠️ The PVTU stem is NOT always the case name: D2.gid holds A2_*.pvtu. Detect
# it from the files rather than assuming, which has bitten this project before.
POST="$CASE_DIR/post"
[ -d "$POST/0eule" ] && POST="$POST/0eule"
STEM=$(ls "$POST"/*_34.pvtu 2>/dev/null | head -1 | xargs -r basename | sed 's/_34\.pvtu$//')
if [ -z "$STEM" ]; then
    echo "⚠️ no *_34.pvtu under $POST -- skipping $CASE"
    exit 0
fi
echo " stem  $STEM   post $POST"

# ⚠️ The container lacks psutil (jaxtrace.utils imports it at package import
# time), so required-packages MUST be on PYTHONPATH -- same as the production
# tracking scripts. Without it the import fails before any work starts.
SIF="${SIF:-/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif}"
PKGS="${PKGS:-/projappl/project_465002752/hashemia/required-packages}"
export JAX_PLATFORMS=cpu          # ⚠️ CPU-only: this is a scipy sparse solve.
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:$PKGS"
# ⚠️ splu is single-threaded, and so is the BLAS used by the dense 12x12
# inversions. Parallelism comes from --jobs ACROSS patches, so pin the
# per-process thread count to 1 or 16 workers will oversubscribe the node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

RUN=(singularity exec -B /scratch/project_465002752 -B /projappl/project_465002752 "$SIF" python3 -u)

for TS in $SNAPSHOTS; do
    OUT="$OUT_ROOT/${CASE}_${TS}.npz"
    echo "---------- $CASE ts=$TS  $(date -Is)"
    T0=$(date +%s)
    # ⚠️ /usr/bin/time -v gives peak RSS, which is the number needed to size
    # the next sweep. SLURM's own accounting is coarser and arrives late.
    #
    # ⚠️ Its report goes to STDERR and the solver's progress to STDOUT. Do NOT
    # pipe them together through grep: the pipe makes the child's stdout a
    # non-tty, Python block-buffers it, and the log stays EMPTY for the whole
    # run (observed: 12 min of silence while 18 workers were at 100% CPU).
    # Keep stdout unpiped and send only time's stderr to a file.
    TVLOG="$OUT_ROOT/${CASE}_${TS}.time"
    /usr/bin/time -v -o "$TVLOG" \
        "${RUN[@]}" "$REPO/scripts/damage/build_divfree_field.py" \
        --case "$POST" --pattern "${STEM}_{timestep}.pvtu" --timestep "$TS" \
        --seed-size "$SEED_SIZE" --layers "$LAYERS" --jobs "$JOBS" \
        --out "$OUT"
    RC=$?
    # ⚠️ RSS here is the MAX OVER THE PROCESS TREE, and with fork+COW the
    # workers share the mesh physically while each one counts it in its own
    # RSS. Measured on C1: 94 GB summed over 18 procs, 67 GB actually
    # resident. So treat this as an upper bound when sizing --mem.
    if [ -f "$TVLOG" ]; then
        grep -E "Maximum resident|Elapsed .wall|Percent of CPU" "$TVLOG" \
            | sed 's/^[[:space:]]*/  time: /'
    fi
    echo "  ts=$TS exit $RC  elapsed $(( $(date +%s) - T0 ))s"
done
echo " done  $(date -Is)"
