#!/bin/bash
# Static-mesh certification over EVERY loaded timestep for ALL 20 PinShapes
# cases, with the piece index rotating per step.
#
# Why this rather than --pieces=0: at the measured 1.77 piece-reads/s on LUMI
# /scratch, hashing every piece at every step costs ~3.3 h per case, so a 12 h
# job certifies only ~4 cases -- and since the survey walks families
# alphabetically those four are all A-family, leaving the D family (the one
# under investigation, 55% non-Kuhn) unmeasured. Rotating one piece per step
# covers all 166 steps of all 20 cases in ~3,320 reads, i.e. minutes.
#
# What it still catches: motion confined to one piece at one step. Verified on
# a synthetic case where the 5-sample screen returns a false negative
# ("STATIC") and this mode correctly reports "MOVES at ts17/piece3".
#
# What it does not claim: that every piece was checked at every step. A mesh
# that moved only in a piece this run never samples at that step would be
# missed -- but it would have to evade a rotating index across 166 steps.
#
# Both srun --unbuffered and python3 -u are set so progress lines appear in the
# log as they happen; a buffered run looks identical to a stalled one.
#
# Reads only. Nothing is written to any case folder.
#SBATCH --job-name=meshsurvey_rot
#SBATCH --account=project_465002752
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project_465002752/hashemia/logs/meshsurvey_rot_%j.out
#SBATCH --error=/scratch/project_465002752/hashemia/logs/meshsurvey_rot_%j.err
set -euo pipefail
P=/projappl/project_465002752/hashemia/required-packages
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif
srun --unbuffered singularity exec --cleanenv --env PYTHONPATH="$P" \
  --env JAX_PLATFORMS=cpu --env TF_CPP_MIN_LOG_LEVEL=2 \
  --env PYTHONUNBUFFERED=1 "$SIF" \
  python3 -u /scratch/project_465002752/hashemia/dfamily/mesh_survey.py \
    --all-steps --pieces=1 \
    /scratch/project_465002752/lorenzgl/Cases/PinShapes
