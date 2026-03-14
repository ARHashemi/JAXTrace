#!/bin/bash
#SBATCH --job-name=jaxtrace
#SBATCH --partition=small-g
#SBATCH --account=project_465002752
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project_465002752/hashemia/logs/%x_%j.out
#SBATCH --error=/scratch/project_465002752/hashemia/logs/%x_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif
JAXTRACE=/project/project_465002752/hashemia/JAXTrace
PKGS=/project/project_465002752/hashemia/required-packages
INPUT=/scratch/project_465001942/Cases-Edgar/new

# Fast NVMe for active writing
FLASH_OUT=/flash/project_465002752/hashemia/run_$SLURM_JOB_ID
# Final long-term storage
SCRATCH_OUT=/scratch/project_465002752/hashemia/outputs/run_$SLURM_JOB_ID

mkdir -p $FLASH_OUT $SCRATCH_OUT

# ── MIOpen cache to RAM ────────────────────────────────────────────────────────
export MIOPEN_USER_DB_PATH="/tmp/hashemia-miopen-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

# ── Run simulation → writes to Flash ──────────────────────────────────────────
srun singularity exec --cleanenv \
  --env PYTHONPATH=$JAXTRACE:$PKGS \
  $SIF \
  python $JAXTRACE/benchmark_femuss_comparison.py \
    --input  $INPUT \
    --output $FLASH_OUT

# ── Move results to Scratch after job ─────────────────────────────────────────
echo "Simulation done. Moving results to scratch..."
mv $FLASH_OUT/* $SCRATCH_OUT/
rmdir $FLASH_OUT
echo "Done. Results in $SCRATCH_OUT"
