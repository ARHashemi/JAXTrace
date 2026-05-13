#!/bin/bash
# SLURM directives — fill in your project ID and log directory before
# submitting, or override at submission time:
#   sbatch --account=project_XXXXXXXXX --output=... run_lumi_diagnose.sh
#SBATCH --job-name=jt-diagnose
#SBATCH --partition=small-g
#SBATCH --account=project_XXXXXXXXX
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT="${SLURM_JOB_ACCOUNT:-project_XXXXXXXXX}"
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif
JAXTRACE="/project/${PROJECT}/${USER}/JAXTrace"
PKGS="/project/${PROJECT}/${USER}/required-packages"
INPUT="/scratch/${PROJECT}/${USER}/data/<CASE>.gid/post"

# Output
SCRATCH_OUT="/scratch/${PROJECT}/${USER}/outputs/diagnose_${SLURM_JOB_ID}"
mkdir -p $SCRATCH_OUT

# ── MIOpen cache to RAM (avoids slow disk I/O for kernel tuning DB) ───────────
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

# ── ROCm / XLA performance flags for MI250X on LUMI-G ────────────────────────
export JAX_PLATFORMS=rocm
export ROCR_VISIBLE_DEVICES=0
export XLA_FLAGS="--xla_gpu_autotune_level=4 --xla_gpu_enable_latency_hiding_scheduler=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MIOPEN_FIND_MODE=1
export TF_CPP_MIN_LOG_LEVEL=2
export HSA_ENABLE_SDMA=0

# ── Run diagnostic ───────────────────────────────────────────────────────────
echo "Starting deviation diagnostic at $(date)"
echo "Output: $SCRATCH_OUT"
echo ""

srun --gpus-per-task=1 \
  singularity exec --cleanenv \
  --env PYTHONPATH=$JAXTRACE:$PKGS \
  --env JAX_PLATFORMS=$JAX_PLATFORMS \
  --env ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
  --env XLA_FLAGS="$XLA_FLAGS" \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE \
  --env XLA_PYTHON_CLIENT_ALLOCATOR=$XLA_PYTHON_CLIENT_ALLOCATOR \
  --env MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH \
  --env MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_CUSTOM_CACHE_DIR \
  --env MIOPEN_FIND_MODE=$MIOPEN_FIND_MODE \
  --env TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL \
  --env HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA \
  $SIF \
  python $JAXTRACE/diagnose_femuss_deviation.py \
    --input  $INPUT \
    --output-dir $SCRATCH_OUT \
    --precision float32 \
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
