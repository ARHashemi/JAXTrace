#!/usr/bin/env bash
#SBATCH --job-name=damage_full
#SBATCH --partition=small
#SBATCH --account=project_465002752
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-19%10
#SBATCH --signal=B:SIGTERM@300
#SBATCH --output=/scratch/project_465002752/hashemia/logs/%x_%A_%a.out
#SBATCH --error=/scratch/project_465002752/hashemia/logs/%x_%A_%a.err
# =============================================================================
# Full-revolution damage survey — ONE CASE PER ARRAY TASK.
#
#   sbatch sbatch_damage_full_array.sh
#
# Why an array rather than a single long job: a full revolution is ~166
# timesteps at ~50 s each, so ~2.2 h per case.  Twenty cases in sequence is
# ~44 h, which exceeds any single allocation.  As an array they run in
# parallel (%10 = at most 10 concurrent) and finish in a few hours.
#
# --array must match the number of PinShapes cases.  Check with:
#   ls -d /scratch/project_465002752/lorenzgl/Cases/PinShapes/*/*.gid | wc -l
# and adjust 0-<N-1> if cases are added.
#
# Each task is skip-guarded, so a resubmit only redoes what is missing.
# =============================================================================
set -uo pipefail

PROJ_ID="${PROJ_ID:-project_465002752}"
PROJ_ID="$(basename "$PROJ_ID")"
USERDIR="${USERDIR:-hashemia}"

REPO="/projappl/${PROJ_ID}/${USERDIR}/JAXTrace"
SWEEP="${REPO}/scripts/damage/run_damage_lumi.sh"
PINSHAPES="/scratch/${PROJ_ID}/lorenzgl/Cases/PinShapes"
OUTDIR="/scratch/${PROJ_ID}/${USERDIR}/damage/stage1_pinshapes_full"
MODELS="${MODELS:-norton}"

mkdir -p "/scratch/${PROJ_ID}/${USERDIR}/logs"

# Build the same ordered case list in every task so the index is stable.
CASES=()
for fam in "$PINSHAPES"/*/; do
    [[ -d "$fam" ]] || continue
    famname="$(basename "$fam")"
    for g in "$fam"*.gid; do
        [[ -d "$g" ]] || continue
        CASES+=("lumi:${famname}/$(basename "$g" .gid)")
    done
done

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= ${#CASES[@]} )); then
    echo "array index $IDX >= ${#CASES[@]} cases — nothing to do"
    exit 0
fi
CASE="${CASES[$IDX]}"

echo "############################################################"
echo "# full-revolution damage — array task $IDX"
echo "#   case   : $CASE"
echo "#   job    : ${SLURM_ARRAY_JOB_ID:-}_${IDX}"
echo "#   node   : $(hostname)"
echo "#   started: $(date -Is)"
echo "############################################################"

_child=""
trap '[[ -n "$_child" ]] && kill -TERM "$_child" 2>/dev/null' SIGTERM

env PHASE=full MODELS="$MODELS" FAMILIES=pinshapes CASES="$CASE" OUTDIR="$OUTDIR" \
    bash "$SWEEP" &
_child=$!
wait $_child
rc=$?

echo "# finished $(date -Is) rc=$rc"
exit $rc
