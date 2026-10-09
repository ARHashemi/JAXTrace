#!/usr/bin/env bash
#
# Evaluate how to choose level_cell_sizes[level] so the search indexes cells
# correctly on an anisotropic (cuboid) mesh.
#
# Background
# ----------
# The search already supports CUBOID cells: level_cell_sizes is (max_level+1, 3)
# and mesh_aligned_point_location.py:209-211 divides each axis by its own pitch.
# Nothing in the kernel assumes cubic cells.
#
# The bug is only in how that 3-vector is picked
# (mesh_aligned_octree_gpu.py:452-457):
#
#     level_cell_sizes_cpu[level] = level_sizes[0]      # the FIRST cell seen
#
# On D2, 99.42% of level-14 cells are cuboid (dx=dy=4.6875e-05,
# dz=5.6117e-05) and only 0.58% are cubic — and the first cell happened to be
# one of the cubic outliers. So the search divides z by 4.6875e-05 while the
# real cells are 5.6117e-05 tall, inflating every z index by 1.197x. The error
# grows with depth and reaches |dk| = 14 at z = -4 mm, far outside the +-1 the
# 3x3x3 neighbourhood can absorb.
#
# D2's dominant sizes form a perfectly regular cuboid lattice: dx and dz each
# halve exactly per level and dz/dx is a constant 1.1972 at every level. So one
# per-axis pitch per level is well defined and needs NO extra levels and NO
# kernel change.
#
# What this job measures
# ----------------------
# For six candidate rules (first / mean / median / dominant-mode / min / max)
# it computes, for EVERY cell, the index error the search would incur, and
# counts the cells whose error exceeds +-1 (i.e. unfindable). It then reports
# where the residual cells sit in (r, z), to answer what to do about the
# minority that disobeys the dominant per-level size.
#
# CPU only, no GPU. ~240 GB for a 10.8M-element D mesh, so it must be a batch
# job: the LUMI login-node cgroup caps per-user memory at 96 GB.
#
# Usage
#   sbatch scripts/dfamily/sbatch_pitch_eval.sh <case.gid>
set -euo pipefail

CASE="${1:?usage: sbatch_pitch_eval.sh <case.gid>}"

PROJECT="${PROJECT:-project_465002752}"
PROJECT="$(basename "$PROJECT")"
USER_NAME="${USER:-hashemia}"
JAXTRACE="${JAXTRACE:-/projappl/${PROJECT}/${USER_NAME}/JAXTrace}"
PKGS="/projappl/${PROJECT}/${USER_NAME}/required-packages"
SIF=/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-community-0.5.0.sif

srun singularity exec --cleanenv \
  --env PYTHONPATH="${JAXTRACE}:${PKGS}" \
  --env JAX_PLATFORMS=cpu \
  --env TF_CPP_MIN_LOG_LEVEL=2 \
  "$SIF" \
  python3 -u "${JAXTRACE}/scripts/dfamily/eval_pitch_choice.py" "$CASE"
