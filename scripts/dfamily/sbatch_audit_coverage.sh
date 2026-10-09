#!/usr/bin/env bash
#SBATCH --job-name=audit_cov
#SBATCH --account=project_465002752
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_465002752/hashemia/logs/audit_cov_%j.out
#SBATCH --error=/scratch/project_465002752/hashemia/logs/audit_cov_%j.err
#
# Run audit_octree_coverage.py as a batch job.
#
# Why not on the login node: the per-user cgroup there caps memory at 96 GB,
# and building the node-to-elements map plus the octree for a 10.8M-element
# D-family mesh in NumPy exceeds that (MemoryError in build_node_to_elements).
# This asks for 240 GB on a CPU-only `small` node — no GPU is needed, the
# audit is pure NumPy and JAX is pinned to the CPU backend.
#
# Usage
#   sbatch scripts/dfamily/sbatch_audit_coverage.sh <results_dir> <case.gid> \
#          [registration] [max_particles]
set -euo pipefail

RUN_DIR="${1:?usage: sbatch_audit_coverage.sh <results_dir> <case.gid> [registration] [max_particles]}"
CASE_DIR="${2:?need <case>.gid}"
REG="${3:-parent_cube}"
MAXP="${4:-150}"

PROJECT="${PROJECT:-project_465002752}"
PROJECT="$(basename "$PROJECT")"
USER_NAME="${USER:-hashemia}"

JAXTRACE="${JAXTRACE:-/projappl/${PROJECT}/${USER_NAME}/JAXTrace}"
PKGS="/projappl/${PROJECT}/${USER_NAME}/required-packages"
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif

CASE_NAME="$(basename "$CASE_DIR" .gid)"
OUT="/scratch/${PROJECT}/${USER_NAME}/dfamily/04_${CASE_NAME}_coverage_audit_${REG}.md"

echo "run dir      : $RUN_DIR"
echo "case         : $CASE_DIR"
echo "registration : $REG"
echo "max particles: $MAXP"
echo "report       : $OUT"
echo

srun singularity exec --cleanenv \
  --env PYTHONPATH="$JAXTRACE:$PKGS" \
  --env JAX_PLATFORMS=cpu \
  --env TF_CPP_MIN_LOG_LEVEL=2 \
  "$SIF" \
  python3 -u "$JAXTRACE/scripts/dfamily/audit_octree_coverage.py" \
    --run "$RUN_DIR" \
    --case "$CASE_DIR" \
    --registration "$REG" \
    --max-particles "$MAXP" \
    --out "$OUT"

echo
echo "Report: $OUT"
