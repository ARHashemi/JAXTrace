#!/bin/bash
# Launch diagnose_mesh_coverage.py on LUMI for a specific case. Fill in
# the case-specific lines (CASE_DIR, MESH_PATTERN, VEL_START) below
# before submitting:
#   sbatch --account=project_XXXXXXXXX scripts/run_lumi_mesh_diagnostic.sh
#
# This is a CPU-friendly diagnostic — it loads the mesh, classifies
# Kuhn / non-Kuhn elements, builds the octree, and verifies that the
# 3x3x3 cell neighbourhood around each element's registered cell
# actually covers every vertex of the element. Coverage gaps explain
# the "particle inside the domain marked Escaped" symptom.
#
# Output is written to stdout and captured by the SLURM .out file.
#
#SBATCH --job-name=jaxtrace_diag
#SBATCH --partition=small-g
#SBATCH --account=project_XXXXXXXXX
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT="${SLURM_JOB_ACCOUNT:-project_XXXXXXXXX}"
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif
JAXTRACE="/project/${PROJECT}/${USER}/JAXTrace"
PKGS="/project/${PROJECT}/${USER}/required-packages"

# ── Edit these per case ─────────────────────────────────────────────────
CASE_DIR="/scratch/${PROJECT}/${USER}/data/<CASE>.gid/post"
MESH_PATTERN="<stem>_{timestep}.pvtu"
VEL_START=0
REGISTRATION="parent_cube"   # parent_cube | vertex_multi
MAX_ELEMENTS=0               # 0 = all
HALF_WINDOW=1                # 1 = 3x3x3, 2 = 5x5x5
# Optional: provide a .npy of (N,3) stuck-particle positions to replay
LOST_POSITIONS=""            # "" to skip
# ────────────────────────────────────────────────────────────────────────

export JAX_PLATFORMS=rocm
export ROCR_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ARGS=(
  --input "$CASE_DIR"
  --mesh-pattern "$MESH_PATTERN"
  --vel-start "$VEL_START"
  --registration "$REGISTRATION"
  --max-elements "$MAX_ELEMENTS"
  --half-window "$HALF_WINDOW"
)
[ -n "$LOST_POSITIONS" ] && ARGS+=( --lost-positions "$LOST_POSITIONS" )

srun --gpus-per-task=1 \
  singularity exec --cleanenv \
    --env PYTHONPATH="$JAXTRACE:$PKGS" \
    --env JAX_PLATFORMS="$JAX_PLATFORMS" \
    --env ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    --env TF_CPP_MIN_LOG_LEVEL="$TF_CPP_MIN_LOG_LEVEL" \
    --env XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE" \
    "$SIF" \
    python "$JAXTRACE/diagnose_mesh_coverage.py" "${ARGS[@]}"
