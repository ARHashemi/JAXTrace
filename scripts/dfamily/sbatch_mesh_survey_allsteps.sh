#!/bin/bash
# Exhaustive static-mesh certification: EVERY loaded timestep, EVERY piece,
# for all 20 PinShapes cases.
#
# The default 5-sample screen can miss motion confined to one piece at one
# step -- verified on a synthetic case that the screen called STATIC and this
# mode correctly calls MOVES. This run leaves nothing inferred from a sample.
#
# Reads only. Nothing is written to any case folder.
#SBATCH --job-name=meshsurvey_all
#SBATCH --account=project_465002752
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_465002752/hashemia/logs/meshsurvey_all_%j.out
#SBATCH --error=/scratch/project_465002752/hashemia/logs/meshsurvey_all_%j.err
set -euo pipefail
P=/projappl/project_465002752/hashemia/required-packages
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif
srun --unbuffered singularity exec --cleanenv --env PYTHONPATH="$P" \
  --env JAX_PLATFORMS=cpu --env TF_CPP_MIN_LOG_LEVEL=2 "$SIF" \
  python3 -u /scratch/project_465002752/hashemia/dfamily/mesh_survey.py \
    --all-steps --pieces=0 \
    /scratch/project_465002752/lorenzgl/Cases/PinShapes
