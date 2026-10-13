#!/bin/bash
#SBATCH --job-name=meshsurvey
#SBATCH --account=project_465002752
#SBATCH --partition=small
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_465002752/hashemia/logs/meshsurvey_%j.out
#SBATCH --error=/scratch/project_465002752/hashemia/logs/meshsurvey_%j.err
set -euo pipefail
P=/projappl/project_465002752/hashemia/required-packages
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif
srun --unbuffered singularity exec --cleanenv --env PYTHONPATH="$P" \
  --env JAX_PLATFORMS=cpu --env TF_CPP_MIN_LOG_LEVEL=2 "$SIF" \
  python3 -u /scratch/project_465002752/hashemia/dfamily/mesh_survey.py \
    /scratch/project_465002752/lorenzgl/Cases/PinShapes
