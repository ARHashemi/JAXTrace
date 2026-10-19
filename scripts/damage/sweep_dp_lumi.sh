#!/usr/bin/env bash
#SBATCH --job-name=dpsweep
#SBATCH --partition=small
#SBATCH --account=project_465002752
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=/scratch/project_465002752/hashemia/damage_indicators/dp/slurm_%A_%a.out
#SBATCH --error=/scratch/project_465002752/hashemia/damage_indicators/dp/slurm_%A_%a.err
# =============================================================================
# A.4 contact-pressure differential B across all 16 PinShapes cases.
#
#   sbatch sweep_dp_lumi.sh
#
# ⚠️ CPU PARTITION, not small-g. B is a PRESSURE-FIELD diagnostic: it reads the
# nodal Pressure from the PVTUs and compares the front and rear of the tool at
# each instant. No particle is integrated, no element is searched, no GPU is
# touched. Asking for a GPU here would waste an allocation.
#
# ⚠️ Why B matters more than its own number. Every other indicator we have
# (lnPhi, C, A_edot, t_res, w) is a PATHLINE integral over the same 288,000
# trajectories, the same RK4 stepping and the same L0/L1/L2 search. A systematic
# error in that machinery moves all of them together. B needs none of it, so it
# is the first genuine EXTERNAL check on the particle apparatus.
#
# Criterion (VERIFIED against the full text): Shi et al., Int. J. Mech. Sci. 241
# (2023) 107969, Eq. (19), Fig. 21 -- voids form when B > 15 MPa, where
# B = P_max(front, theta=0) - P_min(rear, theta=180).
# ⚠️ That threshold is from THEIR alloy and tool, so the RANKING across our 16
# cases is the primary result; absolute pass/fail is indicative.
# =============================================================================
set -uo pipefail

REPO=/projappl/project_465002752/hashemia/JAXTrace
PKGS=/projappl/project_465002752/hashemia/required-packages
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif
CASES_ROOT=/scratch/project_465002752/lorenzgl/Cases/PinShapes
OUT=/scratch/project_465002752/hashemia/damage_indicators/dp
mkdir -p "$OUT"

# ⚠️ Per-case family folder AND vel-range. B2 ends at 201 where every other case
# ends at 199 -- the same outlier the tracking wrapper preserves.
CASES=(A1 A2 A3 A4 B1 B2 B3 B4 C1 C2 C3 C4 D1 D2 D3 D4)
FAMS=(A-FlatsVariations A-FlatsVariations A-FlatsVariations A-FlatsVariations
      B-FluteVariations B-FluteVariations B-FluteVariations B-FluteVariations
      C-ThreadsVariations C-ThreadsVariations C-ThreadsVariations C-ThreadsVariations
      D-ConcavityTilt D-ConcavityTilt D-ConcavityTilt D-ConcavityTilt)

i=${SLURM_ARRAY_TASK_ID:-0}
C=${CASES[$i]}; F=${FAMS[$i]}
VEL_END=199; [ "$C" = "B2" ] && VEL_END=201

echo "=== $C ($F), vel 34..$VEL_END ==="
singularity exec --cleanenv --env PYTHONPATH="$PKGS:$REPO" "$SIF" \
  python "$REPO/scripts/damage/indicator_dp.py" \
    --case "$CASES_ROOT/$F/$C.gid" \
    --vel-range 34 "$VEL_END" \
    --pin-rpm -600 \
    --workers 16 \
    --out "$OUT/dp_$C.csv"
rc=$?
echo "=== $C exit $rc ==="
exit $rc
