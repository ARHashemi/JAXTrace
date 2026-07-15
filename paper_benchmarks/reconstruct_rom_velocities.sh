#!/bin/bash
# =============================================================================
# reconstruct_rom_velocities.sh  —  Precompute ROM velocity fields on the
# original FOM mesh for one or more cases in the FOM cohort.
#
# Loads the shared FSW-ROM basis + coefficients, evaluates the chosen
# reconstruction formula on the case's own mesh, and writes a PVTU that
# looks byte-identical to a FOM PVTU (same mesh, same 'Displacement'
# field name).  The result can be fed straight to run_tracking.py
# --velocity-source mesh with no other changes: from JAXTrace's point
# of view it IS a mesh-loaded velocity field.
#
# The point is to answer "if we ran particle tracking on the
# ROM-reconstructed field instead of the FOM field, how far do the
# particles drift from the FOM answer?".  This is the cheapest way to
# use the reconstruction (no per-substep function call, no uniform-grid
# projection, no analytic-mode retooling) and gives us a lower bound
# on any downstream ROM-tracking accuracy.
#
# Defaults: two cases — the BEST-reconstructing case (04) and the
# WORST (01), as identified in
# docs/rom_reconstruction_findings.md at ts=119.  Formula defaults to
# 'centered' which matches the FEMUSS SLEPcExternalFilter +
# SnapshotsMean convention (colleague spec).
#
# Usage:
#   bash paper_benchmarks/reconstruct_rom_velocities.sh
#
# Override via env vars:
#   CASES="4 1 7 15"      bash paper_benchmarks/reconstruct_rom_velocities.sh
#   ROM_FORMULA=c_over_sig bash paper_benchmarks/reconstruct_rom_velocities.sh
#
# All env-var overrides:
#   FOM_ROOT       parent of <case>.gid folders   (default: /scratch/shared/ROM/FOM)
#   CASE_PREFIX    case-folder prefix              (default: cylindrical)
#   BASIS          .fswrom.basis path              (default: <FOM_ROOT>/cylindrical.som.fswrom.basis)
#   ROMDATA        .fswrom.romdata path            (default: <FOM_ROOT>/cylindrical.som.fswrom.romdata)
#   CASES          case indices, space-separated   (default: "4 1")
#   ROM_FORMULA    centered|sigma_c|c_over_sig|
#                   no_mean|no_mean_sig             (default: centered)
#   SOURCE_TS      timestep to use as mesh template (default: 119)
#   OUT_ROOT       root of output tree             (default: /flash/users/${USER}/data/ROM_recon)
#   FIELD_GROUP    field name inside basis/romdata (default: Displacement)
#   PYTHON         Python interpreter              (default: python)
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAXTRACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FOM_ROOT="${FOM_ROOT:-/scratch/shared/ROM/FOM}"
CASE_PREFIX="${CASE_PREFIX:-cylindrical}"
BASIS="${BASIS:-$FOM_ROOT/${CASE_PREFIX}.som.fswrom.basis}"
ROMDATA="${ROMDATA:-$FOM_ROOT/${CASE_PREFIX}.som.fswrom.romdata}"
CASES="${CASES:-4 1}"
ROM_FORMULA="${ROM_FORMULA:-centered}"
SOURCE_TS="${SOURCE_TS:-119}"
OUT_ROOT="${OUT_ROOT:-/flash/users/${USER:-ali}/data/ROM_recon/${ROM_FORMULA}}"
FIELD_GROUP="${FIELD_GROUP:-Displacement}"
PYTHON="${PYTHON:-python}"

echo "============================================================"
echo " Reconstruct ROM velocities on the FOM mesh"
echo "============================================================"
echo " FOM_ROOT     : $FOM_ROOT"
echo " CASE_PREFIX  : $CASE_PREFIX"
echo " BASIS        : $BASIS"
echo " ROMDATA      : $ROMDATA"
echo " CASES        : $CASES"
echo " ROM_FORMULA  : $ROM_FORMULA"
echo " SOURCE_TS    : $SOURCE_TS"
echo " FIELD_GROUP  : $FIELD_GROUP"
echo " OUT_ROOT     : $OUT_ROOT"
echo " Started      : $(date)"
echo "============================================================"

if [ ! -f "$BASIS" ]; then
    echo "ERROR: basis file not found: $BASIS" >&2
    exit 3
fi
if [ ! -f "$ROMDATA" ]; then
    echo "ERROR: romdata file not found: $ROMDATA" >&2
    exit 3
fi

mkdir -p "$OUT_ROOT"

RC=0
for CASE in $CASES; do
    CASE_ID=$(printf "%03d" "$CASE")
    echo
    echo "------------------------------------------------------------"
    echo " CASE ${CASE_ID}   (formula: ${ROM_FORMULA})"
    echo "------------------------------------------------------------"
    "$PYTHON" -u "$SCRIPT_DIR/reconstruct_rom_case.py" \
        --fom-root       "$FOM_ROOT" \
        --case           "$CASE" \
        --case-prefix    "$CASE_PREFIX" \
        --source-timestep "$SOURCE_TS" \
        --basis          "$BASIS" \
        --romdata        "$ROMDATA" \
        --formula        "$ROM_FORMULA" \
        --field-group    "$FIELD_GROUP" \
        --out-root       "$OUT_ROOT"
    CRC=$?
    if [ "$CRC" != "0" ]; then
        echo "  [CASE ${CASE_ID}] FAILED (rc=$CRC)" >&2
        RC=$CRC
    fi
done

echo
echo "============================================================"
echo " Done.  Exit code: $RC"
echo " Finished: $(date)"
echo " Output tree at: $OUT_ROOT"
echo "   Each case: $OUT_ROOT/${CASE_PREFIX}_<idx>.gid/post/${CASE_PREFIX}_0.pvtu"
echo "============================================================"
exit "$RC"
