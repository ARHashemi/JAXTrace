#!/bin/bash
#SBATCH --job-name=jt-diagnose
#SBATCH --partition=small-g
#SBATCH --account=project_465002752
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_465002752/hashemia/logs/%x_%j.out
#SBATCH --error=/scratch/project_465002752/hashemia/logs/%x_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif
JAXTRACE=/project/project_465002752/hashemia/JAXTrace
PKGS=/project/project_465002752/hashemia/required-packages
INPUT=/scratch/project_465001942/Cases-Edgar/new/cylA.gid/post

# Output
SCRATCH_OUT=/scratch/project_465002752/hashemia/outputs/diagnose_$SLURM_JOB_ID
mkdir -p $SCRATCH_OUT

# ── MIOpen cache to RAM ────────────────────────────────────────────────────────
export MIOPEN_USER_DB_PATH="/tmp/hashemia-miopen-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

# ── Run diagnostic ───────────────────────────────────────────────────────────
echo "Starting deviation diagnostic at $(date)"
echo "Output: $SCRATCH_OUT"
echo ""

srun singularity exec --cleanenv \
  --env PYTHONPATH=$JAXTRACE:$PKGS \
  $SIF \
  python $JAXTRACE/diagnose_femuss_deviation.py \
    --input  $INPUT \
    --output-dir $SCRATCH_OUT \
    --n-steps 100 \
    --compare-freq 5 \
    --femuss-start 0 \
    --y-range -0.01 0.01 \
    --z-range -0.005 0.0 \
    --error-threshold 1e-5 \
    --boundary-proj \
    --pin-velocity \
    --point-in-tet-tol 1e-6 \
    --l0-skip-band 1e-3 \
    --enhanced-search-band 2e-3

echo ""
echo "Finished at $(date) with exit code $?"
echo "Results in $SCRATCH_OUT"
