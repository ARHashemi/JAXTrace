#!/bin/bash
# =============================================================================
# generate_finalstep_density.sh
#
# Generate, per case, a script that computes the FINAL-STEP density (+ uniform
# co-moving residual) from the EXISTING particle trajectory — no re-tracking.
# This gives a density snapshot directly comparable to the particle final-step
# study (same physical step), as opposed to the time-averaged union density.
#
#   run_finalstep_density.sh   run_density_union.py --step-tail 1
#                              --dedup-mode none --write-density
#                              --drift-velocity <v_adv> 0 0
#                              on the existing run_grid-frac_*/particles.vtkhdf
#
# COMMON GRID (per user spec): every case is binned onto the SAME absolute
# grid so the snapshots are directly comparable with no per-case resampling:
#   * ROI_BOX = max bounding box of the final-step particle clouds across all
#     20 cases (the runs are length-matched, so final clouds occupy ~the same
#     region);
#   * voxel size = the initial seed-grid spacing Δp (identical for all cases,
#     since seeds are identical), via VOXEL_SIZE_FROM_PARTICLES=1.
#   -> grid ≈ 486 × 120 × 53 ≈ 3.1 M voxels.
#
# Output: run_grid-frac_*/union/finalstep_union_density.vtkhdf (stem
# 'finalstep') — does NOT touch the existing particles_union_density.vtkhdf
# (the time-averaged union).
#
# Usage:  bash scripts/generate_finalstep_density.sh [FOM_ROOT]
# =============================================================================
set -euo pipefail

FOM_ROOT="${1:-/home/arhashemi/fsw-gpu/scratch/shared/ROM/FOM}"
REMOTE_PREFIX="/scratch/shared/ROM/FOM"

# Common absolute ROI box (max final-step bbox across all cases) + 1·Δp margin.
# "XMIN XMAX YMIN YMAX ZMIN ZMAX" in metres.
ROI_BOX_ABS="-0.0088 0.0805 -0.0150 0.0150 -0.0046 0.0004"

# Template: a known-good union script (carries the co-moving block).
UNION_TEMPLATE="${FOM_ROOT}/cylindrical_003.gid/run_union.sh"
[ -f "$UNION_TEMPLATE" ] || { echo "missing union template $UNION_TEMPLATE" >&2; exit 1; }

read_var () { grep -E "^$2=" "$1" | head -1 | sed -E "s/^$2=//; s/[[:space:]]*#.*//" | tr -d '"'; }

n=0
for case_sh in "$FOM_ROOT"/cylindrical_0*.gid/run_jaxtrace.sh; do
    case_dir="$(dirname "$case_sh")"
    cnum="$(basename "$case_dir" | sed -E 's/cylindrical_([0-9]+)\.gid/\1/')"
    v_adv="$(read_var "$case_sh" INLET_VELOCITY)"
    remote_case="${REMOTE_PREFIX}/cylindrical_${cnum}.gid"
    out="${case_dir}/run_finalstep_density.sh"

    sed -e "s|^CASE_DIR=.*|CASE_DIR=\"${remote_case}\"|" \
        -e "s|^FILENAME_STEM=.*|FILENAME_STEM=finalstep|" \
        -e "s|^STEP_TAIL=.*|STEP_TAIL=1                       # FINAL step only|" \
        -e "s|^DEDUP_MODE=.*|DEDUP_MODE=none                  # single step: no dedup|" \
        -e "s|^WRITE_DENSITY=.*|WRITE_DENSITY=1|" \
        -e "s|^ROI_FRACTION=.*|ROI_FRACTION=\"\"                # use absolute common ROI_BOX|" \
        -e "s|^ROI_BOX=.*|ROI_BOX=\"${ROI_BOX_ABS}\"   # common max final-step bbox (all cases)|" \
        -e "s|^VOXEL_SIZE_FROM_PARTICLES=.*|VOXEL_SIZE_FROM_PARTICLES=1     # = seed Δp, identical for all cases|" \
        "$UNION_TEMPLATE" > "$out"

    # Inject DRIFT_VELOCITY (template references but does not declare it),
    # and point PARTICLES at the EXISTING trajectory (newest run_grid-frac_*).
    python3 - "$out" "$v_adv" <<'PYEOF'
import sys
path, v_adv = sys.argv[1], sys.argv[2]
s = open(path).read()
# 1) co-moving declaration after WRITE_DENSITY
if 'DRIFT_VELOCITY=' not in s:
    block = (f'\n# finalstep: co-moving (uniform-reference) ON\n'
             f'DRIFT_VELOCITY="{v_adv} 0 0"\nNO_COMOVING=0\n'
             f'COMOVING_REFERENCE=uniform\n')
    lines = s.splitlines(keepends=True); out=[]; done=False
    for ln in lines:
        out.append(ln)
        if not done and ln.startswith('WRITE_DENSITY='):
            out.append(block); done=True
    s=''.join(out)
# 2) PARTICLES = newest existing run_grid-frac_*/particles.vtkhdf
needle='if [ -z "${PARTICLES:-}" ]; then'
inject=(needle+'\n'
        '    # finalstep: use the EXISTING trajectory (newest run_grid-frac_*).\n'
        '    PARTICLES="$(ls -dt "$CASE_DIR/$OUTPUT_CASE_SUBFOLDER"/run_grid-frac_*/particles.vtkhdf 2>/dev/null | head -1)"\n')
if 'run_grid-frac_*/particles.vtkhdf' not in s:
    s=s.replace(needle, inject, 1)
open(path,'w').write(s)
PYEOF
    chmod +x "$out" 2>/dev/null || true
    echo "case ${cnum}: v_adv=${v_adv} -> run_finalstep_density.sh"
    n=$((n+1))
done
echo ""
echo "Generated ${n} run_finalstep_density.sh scripts."
echo "Common grid: ROI_BOX='${ROI_BOX_ABS}', voxel=seed Δp. Stem 'finalstep' (existing union density untouched)."
