#!/usr/bin/env bash
#SBATCH --job-name=damage
#SBATCH --partition=small
#SBATCH --account=project_465002752
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --signal=B:SIGTERM@300
#SBATCH --output=/scratch/project_465002752/hashemia/logs/%x_%j.out
#SBATCH --error=/scratch/project_465002752/hashemia/logs/%x_%j.err
# =============================================================================
# Overnight Stage-1 damage survey — all LUMI cases.
#
#   sbatch sbatch_damage_overnight.sh                  # the full plan below
#   sbatch --time=04:00:00 sbatch_damage_overnight.sh  # shorter wall time
#   PHASE=probe sbatch sbatch_damage_overnight.sh      # override a stage
#
# Partition is `small` (CPU), NOT small-g: this workload is NumPy/VTK on the
# host.  No GPU is requested, so it does not compete with tracking jobs.
#
# Two stages, cheapest first, so partial results are still useful if the job
# is cut short:
#
#   1. ROM cohort, snapshot     — 22 plain cylindrical cases, genuinely steady (~2 min)
#   2. PinShapes, 8-phase probe — 20 periodic cases, phase diagnostic (~2 h)
#
# The FULL revolution (~166 steps x 20 cases = ~44 h serial) does NOT belong
# here: submit `sbatch_damage_full_array.sh` instead, which runs one case per
# array task in parallel.
#
# Both stages are skip-guarded, so re-submitting resumes rather than repeats.
# =============================================================================
set -uo pipefail

PROJ_ID="${PROJ_ID:-project_465002752}"
PROJ_ID="$(basename "$PROJ_ID")"
USERDIR="${USERDIR:-hashemia}"

REPO="/projappl/${PROJ_ID}/${USERDIR}/JAXTrace"
SWEEP="${REPO}/scripts/damage/run_damage_lumi.sh"
OUTBASE="/scratch/${PROJ_ID}/${USERDIR}/damage"

MODELS="${MODELS:-norton}"
PHASE_N="${PHASE_N:-8}"

mkdir -p "/scratch/${PROJ_ID}/${USERDIR}/logs"

echo "############################################################"
echo "# Stage-1 damage survey — overnight"
echo "#   job      : ${SLURM_JOB_ID:-<interactive>}"
echo "#   node     : $(hostname)"
echo "#   started  : $(date -Is)"
echo "#   models   : $MODELS"
echo "############################################################"
echo

# Forward SIGTERM so an in-flight case exits cleanly and the skip guard
# leaves the completed ones intact on resubmission.
_child=""
_term() {
    echo "[$(date -Is)] SIGTERM received — stopping after the current case."
    [[ -n "$_child" ]] && kill -TERM "$_child" 2>/dev/null
}
trap _term SIGTERM

run_stage() {
    local label="$1"; shift
    echo
    echo "=========================================================="
    echo "  STAGE: $label"
    echo "  started $(date -Is)"
    echo "=========================================================="
    env "$@" bash "$SWEEP" &
    _child=$!
    wait $_child
    local rc=$?
    echo "  STAGE $label finished rc=$rc at $(date -Is)"
    return $rc
}

# ── 1. ROM cohort — steady, snapshot is the correct treatment ────────────────
ROM_CASES=""
for g in /scratch/${PROJ_ID}/lorenzgl/Cases/ROM/FOM_cases/cylindrical_*.gid; do
    [[ -d "$g" ]] || continue
    n="$(basename "$g" .gid)"; ROM_CASES="$ROM_CASES rom:${n#cylindrical_}"
done
run_stage "ROM cohort (snapshot)" \
    PHASE=snapshot MODELS="$MODELS" FAMILIES=rom CASES="${ROM_CASES# }" \
    OUTDIR="$OUTBASE/stage1_rom"

# ── 2. PinShapes — 8-phase probe ─────────────────────────────────────────────
# Cheap diagnostic: does the rotational phase change the answer?  If the
# spread is small the snapshot numbers stand; if large, stage 3 is mandatory.
run_stage "PinShapes (${PHASE_N}-phase probe)" \
    PHASE=probe PHASE_N="$PHASE_N" MODELS="$MODELS" FAMILIES=pinshapes \
    OUTDIR="$OUTBASE/stage1_pinshapes_probe"

# The full revolution runs as a separate job array — see
# scripts/damage/sbatch_damage_full_array.sh.

echo
echo "############################################################"
echo "# finished $(date -Is)"
for d in "$OUTBASE"/stage1_*; do
    [[ -d "$d" ]] || continue
    n=$(ls "$d"/dmg_*.npz 2>/dev/null | wc -l)
    echo "#   $(basename "$d"): $n result(s)"
    [[ -f "$d/stage1_summary.csv" ]] && echo "#     $d/stage1_summary.csv"
done
echo "############################################################"
