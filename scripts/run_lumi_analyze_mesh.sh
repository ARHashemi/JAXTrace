#!/bin/bash
#SBATCH --job-name=jt-mesh-analyze
#SBATCH --partition=small-g
#SBATCH --account=project_465002752
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_465002752/hashemia/logs/%x_%j.out
#SBATCH --error=/scratch/project_465002752/hashemia/logs/%x_%j.err

# Runs two analyses sequentially:
#   1) analyze_mesh_variation.py — full sweep over all mesh files in the
#      cyclic sequence, writing the CSV row-by-row so a partial run still
#      yields usable output.
#   2) diff_two_meshes.py — focused two-PVTU diff (step A vs step B) for
#      spatial localisation of changed elements, refinement-level breakdown,
#      level-set proximity, and the fixed-cells-varying-registration check.

# ── Paths ─────────────────────────────────────────────────────────────────────
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif
JAXTRACE=/project/project_465002752/hashemia/JAXTrace
PKGS=/project/project_465002752/hashemia/required-packages
INPUT=/scratch/project_465002752/lorenzgl/Cases/PinShapes/C-ThreadsVariations/C3.gid/post

PATTERN="C3_{timestep}.pvtu"
START=0
END=151
STRIDE=1
DIFF_STEP_A=0
DIFF_STEP_B=50           # ~1/3 of the way through the cycle
LEVELSET_FIELD=""        # set to e.g. "LevelSet" if available; leave empty to skip

# Output
SCRATCH_OUT=/scratch/project_465002752/hashemia/outputs/analyze_mesh_$SLURM_JOB_ID
mkdir -p $SCRATCH_OUT

CSV_OUT=$SCRATCH_OUT/mesh_variation_report.csv
DIFF_TXT=$SCRATCH_OUT/diff_a${DIFF_STEP_A}_b${DIFF_STEP_B}.txt
DIFF_VTU=$SCRATCH_OUT/diff_a${DIFF_STEP_A}_b${DIFF_STEP_B}.vtu
ENVELOPE_JSON=$SCRATCH_OUT/fine_zone_envelope.json

# Fine-zone envelope sweep parameters
ENV_START=0
ENV_END=149
ENV_STRIDE=10
FINEST_FRAC=0.20

# ── MIOpen cache to RAM (avoids slow disk I/O for kernel tuning DB) ───────────
export MIOPEN_USER_DB_PATH="/tmp/hashemia-miopen-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

# ── ROCm / XLA flags (analysis is CPU/VTK-bound but we keep them for parity) ──
export JAX_PLATFORMS=rocm
export ROCR_VISIBLE_DEVICES=0
export XLA_FLAGS="--xla_gpu_autotune_level=4 --xla_gpu_enable_latency_hiding_scheduler=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MIOPEN_FIND_MODE=1
export TF_CPP_MIN_LOG_LEVEL=2
export HSA_ENABLE_SDMA=0
export PYTHONUNBUFFERED=1

echo "Starting mesh analysis at $(date)"
echo "Output: $SCRATCH_OUT"
echo ""

# Build singularity exec invocation as an array so quoting survives
# variable expansion (a string-valued $SING would word-split XLA_FLAGS).
SING_ARGS=(
  singularity exec --cleanenv
  --env "PYTHONPATH=$JAXTRACE:$PKGS"
  --env "JAX_PLATFORMS=$JAX_PLATFORMS"
  --env "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"
  --env "XLA_FLAGS=$XLA_FLAGS"
  --env "XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE"
  --env "XLA_PYTHON_CLIENT_ALLOCATOR=$XLA_PYTHON_CLIENT_ALLOCATOR"
  --env "MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH"
  --env "MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_CUSTOM_CACHE_DIR"
  --env "MIOPEN_FIND_MODE=$MIOPEN_FIND_MODE"
  --env "TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL"
  --env "HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA"
  --env "PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
  "$SIF"
)

# ── 1) Full sweep ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "[1/3] analyze_mesh_variation.py — full sweep $START..$END (stride $STRIDE)"
echo "============================================================"
srun --gpus-per-task=1 "${SING_ARGS[@]}" \
  python $JAXTRACE/analyze_mesh_variation.py \
    --input "$INPUT" \
    --pattern "$PATTERN" \
    --start $START \
    --end $END \
    --stride $STRIDE \
    --output "$CSV_OUT" \
    --max-sample 3000000

echo ""
echo "Sweep finished at $(date)"
echo ""

# ── 2) Two-PVTU diff ──────────────────────────────────────────────────────────
echo "============================================================"
echo "[2/3] diff_two_meshes.py — A=$DIFF_STEP_A vs B=$DIFF_STEP_B"
echo "============================================================"

DIFF_ARGS=(
  --input "$INPUT"
  --pattern "$PATTERN"
  --step-a $DIFF_STEP_A
  --step-b $DIFF_STEP_B
  --output "$DIFF_TXT"
  --export-changed-vtu "$DIFF_VTU"
)
if [ -n "$LEVELSET_FIELD" ]; then
  DIFF_ARGS+=(--levelset-field "$LEVELSET_FIELD")
fi

srun --gpus-per-task=1 "${SING_ARGS[@]}" \
  python $JAXTRACE/diff_two_meshes.py "${DIFF_ARGS[@]}"

echo ""

# ── 3) Fine-zone envelope scan ───────────────────────────────────────────────
echo "============================================================"
echo "[3/3] scan_fine_zone_envelope.py — $ENV_START..$ENV_END stride $ENV_STRIDE"
echo "============================================================"
srun --gpus-per-task=1 "${SING_ARGS[@]}" \
  python $JAXTRACE/scan_fine_zone_envelope.py \
    --input "$INPUT" \
    --pattern "$PATTERN" \
    --start $ENV_START \
    --end $ENV_END \
    --stride $ENV_STRIDE \
    --finest-frac $FINEST_FRAC \
    --output "$ENVELOPE_JSON"

echo ""
echo "Finished at $(date)"
echo "Results in $SCRATCH_OUT"
ls -la $SCRATCH_OUT
