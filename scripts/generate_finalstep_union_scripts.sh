#!/bin/bash
# =============================================================================
# generate_finalstep_union_scripts.sh
#
# Stage B of the three-stage cohort post-processing:
#   A) Variable-dt tracking      (existing run_jaxtrace.sh per case)
#   B) Final-step-only density   (this script: run_union_finalstep.sh per case)
#   C) Fixed-dt tracking + both  (run_jaxtrace_finalrom.sh / run_union_finalrom.sh)
#
# For every cylindrical_NNN.gid in the cohort root, copy the case's
# existing run_union.sh into run_union_finalstep.sh and patch it for
# FINAL-STEP-ONLY density:
#
#   FILENAME_STEM=finalstep                   # separate output file
#   STEP_TAIL=1                               # only the last step
#   DEDUP_MODE=none                           # single step -> no dedup work
#   WRITE_DENSITY=1                           # keep the density pass on
#   ROI_FRACTION=""                           # disabled
#   ROI_BOX="-0.030 0.085 -0.018 0.018 -0.0055 0.0015"
#                                             # absolute box -> runaway-proof
#                                             # voxel-grid sizing
#
# Input particles: post_pt/run_grid-frac_n*/particles.vtkhdf  — the
# auto-detect inside run_union.sh already picks the latest such folder,
# so this script does NOT stamp a PARTICLES path. The result lives at
# post_pt/run_grid-frac_n*/union/<stem>_union_density.vtkhdf.
#
# Existing run_union.sh, run_union_finalrom.sh, and run_jaxtrace_finalrom.sh
# are NOT touched.
#
# Usage:
#   bash generate_finalstep_union_scripts.sh           # all 20 cases
#   bash generate_finalstep_union_scripts.sh 005 008   # subset by case number
# =============================================================================
set -euo pipefail

FOM_ROOT="${FOM_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
OUT_NAME="run_union_finalstep.sh"

# Absolute ROI box, in metres: "XMIN XMAX YMIN YMAX ZMIN ZMAX". Same as
# generate_finalrom_scripts.sh — derived from the 17 clean cases' density
# grids extended upstream to cover the inlet seed region, with generous
# y/z margins. Sizes the voxel grid to ~7.4 M voxels regardless of how
# far any runaway particle wandered.
ROI_BOX_ABS="-0.030 0.085 -0.018 0.018 -0.0055 0.0015"

# Build the case list. If positional args given, treat each as a 3-digit
# case number to process; otherwise discover all cylindrical_0NN.gid.
if [ $# -gt 0 ]; then
    CASES=("$@")
else
    CASES=()
    for d in "$FOM_ROOT"/cylindrical_0*.gid; do
        CASES+=( "$(basename "$d" | sed -E 's/cylindrical_([0-9]+)\.gid/\1/')" )
    done
fi

n_done=0
n_skip=0
for cnum in "${CASES[@]}"; do
    # Normalise to 3-digit zero-padded.
    printf -v cnum3 "%03d" "$((10#$cnum))"
    case_dir="$FOM_ROOT/cylindrical_${cnum3}.gid"
    src="$case_dir/run_union.sh"
    dst="$case_dir/$OUT_NAME"

    if [ ! -f "$src" ]; then
        echo "  $cnum3: SKIP — no run_union.sh template at $src" >&2
        n_skip=$((n_skip + 1))
        continue
    fi

    # Patch four anchored lines:
    #   FILENAME_STEM, STEP_TAIL, DEDUP_MODE, ROI_FRACTION, ROI_BOX.
    # Leave everything else (CASE_DIR, OUTPUT_CASE_SUBFOLDER, WRITE_DENSITY,
    # kernel, bandwidth, voxel-size-from-particles, …) at whatever the case
    # already uses.
    sed \
        -e 's|^FILENAME_STEM=.*|FILENAME_STEM=finalstep              # final-step density (Stage B)|' \
        -e 's|^STEP_TAIL=.*|STEP_TAIL=1                            # final step only|' \
        -e 's|^DEDUP_MODE=.*|DEDUP_MODE=none                       # one step -> no dedup work|' \
        -e "s|^ROI_FRACTION=.*|ROI_FRACTION=\"\"                          # disabled; using absolute ROI_BOX|" \
        -e "s|^ROI_BOX=.*|ROI_BOX=\"$ROI_BOX_ABS\"   # absolute physical box; runaway-proof|" \
        "$src" > "$dst.tmp"
    mv "$dst.tmp" "$dst"
    chmod +x "$dst" 2>/dev/null || true

    printf "  %s: written  %s\n" "$cnum3" "$dst"
    n_done=$((n_done + 1))
done

echo
echo "Summary:"
echo "  written:          $n_done"
echo "  skipped (no src): $n_skip"
[ "$n_skip" = 0 ]
