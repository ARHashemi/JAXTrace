#!/usr/bin/env bash
# =============================================================================
# sec6_rerun.sh — full Section 6 (Validation) benchmark re-run
# =============================================================================
#
# Runs the existing benchmark_l2_accuracy.py harness with the exact protocol
# quoted in sec6_validation.tex:
#
#   * Batch size N_p = 10,000 for the main tables
#   * Perturbation sweep sigma in {0.0, 0.1, 0.2, 0.5, 0.7, 1.0}
#   * Warm-up runs = 3, timed runs = 7
#   * Registration = all (vertex, parent_cube, aabb) so all three MALMO
#     variants are covered
#   * Scalability sweep with N_p in {1k, 2k, 5k, 10k, 20k, 50k, 100k,
#     200k, 500k} — the union of what tab:timing and tab:scalability use
#   * Intra-element accuracy (5 position classes)
#   * 1x1x1 failure decomposition
#   * Float64 (paper protocol)
#
# The raw stdout of benchmark_l2_accuracy.py is written to
#   paper_benchmarks/sec6_raw.log
# and the derived clean numbers are written by
#   paper_benchmarks/sec6_postprocess.py
# to
#   paper_benchmarks/sec6_numbers.json    (machine-readable)
#   paper_benchmarks/sec6_report.md       (human-readable diff vs paper)
#
# Usage on the workstation:
#   cd /flash/shared/jax/JAXTrace
#   bash paper_benchmarks/sec6_rerun.sh
#
# Optional environment overrides:
#   JAXTRACE_MESH_DIR   root of mesh PVTU files (default: /scratch/...)
#   JAXTRACE_OUTDIR     where to write logs (default: paper_benchmarks/)
#
# The full run takes ~20-40 minutes on an RTX 5090 depending on JIT-cache
# warmth. If a previous JIT-cache exists the run is closer to 20 min.
# =============================================================================

set -euo pipefail

# Locate the repository root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
OUTDIR="${JAXTRACE_OUTDIR:-${REPO_ROOT}/paper_benchmarks}"
LOG="${OUTDIR}/sec6_raw.log"

# ---- Mesh location ---------------------------------------------------------
# The paper's benchmark mesh is /flash/users/ali/data/cylA.gid/, with velocity
# PVTUs under post/0eule/cylA_{timestep}.pvtu, timestep = 159 (final step).
# The harness expects --input to point at the parent that contains the
# --mesh-subdir folder. Override via JAXTRACE_MESH_DIR if the mount path
# differs on your workstation.
MESH_INPUT="${JAXTRACE_MESH_DIR:-/flash/users/ali/data/cylA.gid/post}"
MESH_SUBDIR="0eule"
MESH_PATTERN="cylA_{timestep}.pvtu"
VEL_START=159
VEL_END=159

# Sanity check.
if [[ ! -f "${MESH_INPUT}/${MESH_SUBDIR}/cylA_${VEL_END}.pvtu" ]]; then
    echo "[sec6_rerun] ERROR: expected mesh file not found:"
    echo "             ${MESH_INPUT}/${MESH_SUBDIR}/cylA_${VEL_END}.pvtu"
    echo "             Override JAXTRACE_MESH_DIR to point at the correct post directory."
    exit 1
fi

echo "[sec6_rerun] Repo root: ${REPO_ROOT}"
echo "[sec6_rerun] Outdir:    ${OUTDIR}"
echo "[sec6_rerun] Mesh:      ${MESH_INPUT} (subdir='${MESH_SUBDIR}', pattern='${MESH_PATTERN}', vel-range ${VEL_START} ${VEL_END})"
mkdir -p "${OUTDIR}"

# ---- Paper protocol (fixed) -----------------------------------------------
N_PARTICLES=10000
BATCH_SIZE=50000
WARMUP=3
TIMED=7
PERTURBATIONS=(0.0 0.1 0.2 0.5 0.7 1.0)
SCALABILITY_SIZES=(1000 2000 5000 10000 20000 50000 100000 200000 500000)
POSITION_TYPES=(centroid random near_face near_edge near_vertex)
REGISTRATION=all
SEED=42
POINT_IN_TET_TOL=1e-6

# ---- Environment ----------------------------------------------------------
# Prevent JAX from pre-allocating the entire VRAM pool so the mesh
# preprocessing NumPy stages don't fight for memory.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export TF_CPP_MIN_LOG_LEVEL=2
# For nsys captures if the operator wants them; benign otherwise.
export JAX_ENABLE_X64=1

# ---- Argument construction ------------------------------------------------
HARNESS="${REPO_ROOT}/benchmark_l2_accuracy.py"
if [[ ! -f "${HARNESS}" ]]; then
    echo "[sec6_rerun] ERROR: harness not found at ${HARNESS}" >&2
    exit 1
fi

ARGS=(
    --input "${MESH_INPUT}"
    --mesh-subdir "${MESH_SUBDIR}"
    --mesh-pattern "${MESH_PATTERN}"
    --vel-range "${VEL_START}" "${VEL_END}"
    --n-particles "${N_PARTICLES}"
    --batch-size "${BATCH_SIZE}"
    --warmup-runs "${WARMUP}"
    --timing-runs "${TIMED}"
    --perturbations "${PERTURBATIONS[@]}"
    --position-types "${POSITION_TYPES[@]}"
    --registration "${REGISTRATION}"
    --seed "${SEED}"
    --point-in-tet-tol "${POINT_IN_TET_TOL}"
    --scalability
    --scalability-sizes "${SCALABILITY_SIZES[@]}"
    --float64
)

# ---- Run + tee to log -----------------------------------------------------
echo "[sec6_rerun] Command: python ${HARNESS} ${ARGS[@]}"
echo "[sec6_rerun] Log:     ${LOG}"
echo "[sec6_rerun] Starting at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Note: --scalability re-uses the harness's own N_p sweep. Combined with
# perturbation sweep at N_p=10000 this covers every table's underlying data.
python -u "${HARNESS}" "${ARGS[@]}" 2>&1 | tee "${LOG}"

echo "[sec6_rerun] Finished at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---- Post-process ---------------------------------------------------------
POSTPROC="${SCRIPT_DIR}/sec6_postprocess.py"
if [[ ! -f "${POSTPROC}" ]]; then
    echo "[sec6_rerun] WARNING: post-processor not found at ${POSTPROC}"
    echo "[sec6_rerun]           You can still inspect the raw log at ${LOG}"
    exit 0
fi

echo "[sec6_rerun] Running post-processor..."
python "${POSTPROC}" \
    --log "${LOG}" \
    --json "${OUTDIR}/sec6_numbers.json" \
    --report "${OUTDIR}/sec6_report.md"

echo "[sec6_rerun] DONE."
echo "[sec6_rerun]   Raw log:            ${LOG}"
echo "[sec6_rerun]   Machine numbers:    ${OUTDIR}/sec6_numbers.json"
echo "[sec6_rerun]   Human report:       ${OUTDIR}/sec6_report.md"
