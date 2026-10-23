#!/usr/bin/env bash
#SBATCH --job-name=dtp_fam
#SBATCH --partition=small-g
#SBATCH --account=project_465002752
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=36:00:00
#SBATCH --signal=B:SIGTERM@180
#SBATCH --array=0-2
#SBATCH --output=/scratch/project_465002752/hashemia/logs/%x_%A_%a.out
#SBATCH --error=/scratch/project_465002752/hashemia/logs/%x_%A_%a.err
# =============================================================================
# DTP on a real FSW family, with the full damage indicator set.
#
#   sbatch sbatch_dtp_family.sh                    # A family (default)
#   FAMILY=C sbatch --array=0-3 sbatch_dtp_family.sh
#   VELOCITY_MODEL=raw sbatch sbatch_dtp_family.sh  # the control arm
#
# ⚠️ CASE SELECTION IS BY DATA, NOT BY NAME. Surveyed 2026-10-23:
#     12 cases ship 167 SERIAL .vtu, timesteps 33..199, ZERO gaps
#      4 cases (A3, B2, D2, D3) ship partitioned .pvtu, heavily gapped
#        (22, 22, 26 and 3 files) -- vel-range 34..199 is INVALID for them
#        and is what killed the earlier gate job on a missing A2_72.pvtu.
# This script therefore skips a case whose requested range is not contiguous
# rather than failing 40 minutes into the mesh load.
#
# ⚠️ DETACHMENT IS RECORDED BUT NOT ACTING. --damage-detachment accumulates
# ln_J and tau_min as columns; --detach is the flag that would make a particle
# stop following the field, and it is deliberately ABSENT so trajectories are
# comparable across velocity models. Every threshold stays a post-process.
#
# ⚠️ tau_y IS AN ASSUMPTION, not a measurement. 10.8 MPa is tau_y = sigma_y/sqrt3
# at 500 C from [Din21] (DETACHMENT_METHOD.md section 2), paired with
# --damage-tau-ref-temp 500 so the two are consistent; the measured cylA peak
# is 509 C. A 111-paper search found NO validated temperature-resolved Bingham
# yield stress for AA6061/AA6063. Bu scales LINEARLY in tau_y, so the particle
# RANKING -- the result that matters -- is unaffected by the choice.
# =============================================================================
set -uo pipefail

PROJECT="${SLURM_JOB_ACCOUNT:-project_465002752}"
SIF="${SIF:-/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif}"
JAXTRACE="${JAXTRACE:-/projappl/${PROJECT}/hashemia/JAXTrace}"
PKGS="${PKGS:-/projappl/${PROJECT}/hashemia/required-packages}"
CASES="${CASES:-/scratch/${PROJECT}/lorenzgl/Cases/PinShapes}"
OUT_ROOT="${OUT_ROOT:-/scratch/${PROJECT}/hashemia/dtp_family}"
OCTREE="${OCTREE:-/scratch/${PROJECT}/hashemia/octree_cache}"

FAMILY="${FAMILY:-A}"
VELOCITY_MODEL="${VELOCITY_MODEL:-dtp}"
VEL_START="${VEL_START:-34}"; VEL_END="${VEL_END:-199}"
N_STEPS="${N_STEPS:-8000}"; DT="${DT:-0.0003629}"

mkdir -p "$OUT_ROOT" "$OCTREE" "/scratch/${PROJECT}/hashemia/logs"

mapfile -t CASE_DIRS < <(ls -d "$CASES"/${FAMILY}-*/${FAMILY}[0-9].gid 2>/dev/null | sort)
IDX=${SLURM_ARRAY_TASK_ID:-0}
[ "$IDX" -ge "${#CASE_DIRS[@]}" ] && { echo "task $IDX >= ${#CASE_DIRS[@]} cases"; exit 0; }
CASE_DIR="${CASE_DIRS[$IDX]}"; CASE=$(basename "$CASE_DIR" .gid)

POST="$CASE_DIR/post"; [ -d "$POST/0eule" ] && POST="$POST/0eule"
# ⚠️ Detect the stem AND the extension from disk: the D family reuses A/B/C
# stems (D1->A1, D2->A2, D3->B4, D4->C2) and the two formats coexist.
FIRST=$(ls "$POST"/*.pvtu 2>/dev/null | head -1)
EXT=pvtu
[ -z "$FIRST" ] && { FIRST=$(ls "$POST"/*.vtu 2>/dev/null | head -1); EXT=vtu; }
[ -z "$FIRST" ] && { echo "no .pvtu or .vtu under $POST -- skipping $CASE"; exit 0; }
STEM=$(basename "$FIRST" | sed "s/_[0-9]*\.${EXT}\$//")

# ⚠️ Refuse a non-contiguous range rather than dying on the first gap.
MISSING=0
for t in $(seq "$VEL_START" "$VEL_END"); do
    [ -f "$POST/${STEM}_${t}.${EXT}" ] || MISSING=$((MISSING+1))
done
if [ "$MISSING" -gt 0 ]; then
    echo "SKIP $CASE: $MISSING of $((VEL_END-VEL_START+1)) snapshots missing in"
    echo "     ${STEM}_{$VEL_START..$VEL_END}.${EXT} under $POST"
    echo "     (this case ships a gapped set; pick a contiguous sub-range)"
    exit 0
fi

OUT="$OUT_ROOT/${CASE}_${VELOCITY_MODEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
echo "=============================================================="
echo " case $CASE   model $VELOCITY_MODEL   $STEM_*.$EXT  $VEL_START..$VEL_END"
echo " out  $OUT"
echo " node $(hostname)  $(date -Is)"
echo "=============================================================="

export JAX_PLATFORMS=rocm
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONPATH="$JAXTRACE:$PKGS"
export PYTHONUNBUFFERED=1

ARGS=(
  --input "$CASE_DIR" --output "$OUT"
  --mesh-subdir "$(basename "$POST")"
  --mesh-pattern "${STEM}_{timestep}.${EXT}"
  --precision float32
  --vel-range "$VEL_START" "$VEL_END"
  --velocity-field Displacement --levelset-field LEVEL
  --temperature-field Temperature
  --n-steps "$N_STEPS" --dt "$DT" --log-interval 200 --export-freq 20
  --seed-source grid-frac --seed-fraction 0.0 0.2 0.0 1.0 0.0 1.0
  --seed-grid 60 120 40 --seed 42
  --rk4-mode fused --l1-method face --l2-neighborhood 3
  --levelset-mode zero_vel --failed-substage last_valid_vel
  --interpolation-method direct_inverse
  --pin-rpm -600 --pin-center 0.0 0.0 0.0 --pin-axis 0.0 0.0 1.0
  --pin-tilt 0.0 --n-groups 5
  --octree-cache "$OCTREE"
  --velocity-model "$VELOCITY_MODEL"
  # ── damage: every indicator on, detachment RECORDED not ACTING ──────────
  --damage --damage-driver models --damage-order 1
  --damage-indicators --residence-edot-threshold 1.0
  --damage-detachment
  --damage-divergence analytic
  --damage-tau-ref-edot 1.0
  --damage-tau-ref-temp 500
  --damage-tau-y 10.8e6
  --damage-out "$OUT/damage_models.npz"
)

{ echo "case=$CASE model=$VELOCITY_MODEL stem=$STEM ext=$EXT"
  echo "range=$VEL_START..$VEL_END steps=$N_STEPS dt=$DT"
  echo "tau_y=10.8e6 Pa (Din21, 500 C) tau_ref_temp=500 C"
  echo "commit=$(cd "$JAXTRACE" && git rev-parse --short HEAD 2>/dev/null)"
  echo "started=$(date -Is)"; } > "$OUT/RUN_INFO.txt"

/usr/bin/time -v -o "$OUT/run.time" \
  singularity exec -B /scratch/${PROJECT} -B /projappl/${PROJECT} "$SIF" \
  python3 -u "$JAXTRACE/run_tracking.py" "${ARGS[@]}"
RC=$?
echo " exit $RC  $(date -Is)"
grep -E "Maximum resident|Elapsed .wall|Percent of CPU" "$OUT/run.time" 2>/dev/null
