#!/bin/bash
# =============================================================================
# generate_jaxtrace_union_scripts.sh
#
# Companion to generate_jaxtrace_scripts.sh. For every
# cylindrical_NNN.gid folder in the current directory, drop in a
# per-case run_union.sh that runs the density-union postprocess on the
# particles.vtkhdf produced by that case's tracking run.
#
# Per-case patches:
#   - CASE_DIR:                absolute path to the case folder
#                              (derived as "$COHORT_PREFIX/<case_name>",
#                              same logic as the tracking generator).
#   - OUTPUT_CASE_SUBFOLDER:   matches what run_jaxtrace.sh used at
#                              tracking time (default 'post_pt').
#
# We deliberately do NOT stamp a PARTICLES path. The runner finds the
# latest <case>.gid/<OUTPUT_CASE_SUBFOLDER>/run_*/particles.vtkhdf at
# run time, so re-running tracking with new N_STEPS and then re-running
# union just works without regenerating these scripts.
#
# Templates:
#   --platform=workstation (default)
#       template: ./cylindrical_001.gid/run_union.sh
#   --platform=lumi
#       template: $JAXTRACE_REPO/scripts/run_lumi_union.sh
#
# Existing per-case run_union.sh files are SKIPPED unless --force.
#
# Usage:
#   ./generate_jaxtrace_union_scripts.sh                              # workstation
#   ./generate_jaxtrace_union_scripts.sh --platform=lumi
#   ./generate_jaxtrace_union_scripts.sh --force
#   ./generate_jaxtrace_union_scripts.sh --skip=004,005,006
#   ./generate_jaxtrace_union_scripts.sh --jaxtrace-repo=/path
#   ./generate_jaxtrace_union_scripts.sh --cohort-prefix=/scratch/shared/ROM/FOM
#   ./generate_jaxtrace_union_scripts.sh --output-subfolder=post_pt
# =============================================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
PLATFORM=workstation
FORCE=0
SKIP_LIST=""
JAXTRACE_REPO="/flash/shared/jax/JAXTrace"
CASE_GLOB="cylindrical_*.gid"
COHORT_PREFIX=""
# Subfolder under each <case>.gid where run_jaxtrace.sh writes results
# in OUTPUT_TARGET=case mode. Must match what was used at tracking time
# (the runner uses this to find particles.vtkhdf).
OUTPUT_SUBFOLDER="post_pt"

# Reference template for workstation.
WORKSTATION_TEMPLATE="cylindrical_001.gid/run_union.sh"

# Per-case output filename. The launcher looks for this exact name.
OUT_NAME="run_union.sh"

# ── Parse args ──────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --platform=*)          PLATFORM="${arg#*=}" ;;
        --force)               FORCE=1 ;;
        --skip=*)              SKIP_LIST="${arg#*=}" ;;
        --jaxtrace-repo=*)     JAXTRACE_REPO="${arg#*=}" ;;
        --cohort-prefix=*)     COHORT_PREFIX="${arg#*=}" ;;
        --output-subfolder=*)  OUTPUT_SUBFOLDER="${arg#*=}" ;;
        --help|-h)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Try --help" >&2
            exit 1
            ;;
    esac
done

case "$PLATFORM" in
    workstation|lumi) ;;
    *) echo "ERROR: --platform must be 'workstation' or 'lumi', got '$PLATFORM'" >&2; exit 1 ;;
esac

# ── Pick template ───────────────────────────────────────────────────────────
if [ "$PLATFORM" = "workstation" ]; then
    TEMPLATE_PATH="$WORKSTATION_TEMPLATE"
    if [ ! -f "$TEMPLATE_PATH" ]; then
        echo "ERROR: workstation union template not found at $TEMPLATE_PATH" >&2
        echo "Set up cylindrical_001.gid/run_union.sh first (the canonical reference)." >&2
        exit 1
    fi
else
    TEMPLATE_PATH="$JAXTRACE_REPO/scripts/run_lumi_union.sh"
    if [ ! -f "$TEMPLATE_PATH" ]; then
        echo "ERROR: LUMI union template not found at $TEMPLATE_PATH" >&2
        echo "Pass --jaxtrace-repo=/path/to/JAXTrace to override." >&2
        exit 1
    fi
fi

# ── Resolve cohort prefix ──────────────────────────────────────────────────
if [ -z "$COHORT_PREFIX" ]; then
    COHORT_PREFIX="$PWD"
fi
COHORT_PREFIX="${COHORT_PREFIX%/}"

echo "Template:    $TEMPLATE_PATH"
echo "Platform:    $PLATFORM"
echo "Cohort:      $COHORT_PREFIX"
echo "Force:       $FORCE"
echo "Skip:        ${SKIP_LIST:-<none>}"
echo "Sub-folder:  $OUTPUT_SUBFOLDER"
echo

# ── Patch helper ───────────────────────────────────────────────────────────
# We don't touch PARTICLES — the runner auto-finds it. We do stamp
# CASE_DIR explicitly (some templates default it to $(dirname $0), which
# resolves through symlinks via readlink -f; stamping the user-supplied
# COHORT_PREFIX path keeps mount-prefix surprises from sneaking in) and
# OUTPUT_CASE_SUBFOLDER so the runner looks in the right subdir.
patch_template() {
    local template="$1"
    local output="$2"
    local case_dir="$3"
    local out_sub="$4"

    sed \
        -e "0,/^CASE_DIR=/{s|^CASE_DIR=.*|CASE_DIR=\"$case_dir\"|}" \
        -e "0,/^OUTPUT_CASE_SUBFOLDER=/{s|^OUTPUT_CASE_SUBFOLDER=.*|OUTPUT_CASE_SUBFOLDER=$out_sub|}" \
        "$template" > "$output.tmp"

    mv "$output.tmp" "$output"
    chmod +x "$output"
}

# ── Build skip set ─────────────────────────────────────────────────────────
declare -A SKIP_SET
if [ -n "$SKIP_LIST" ]; then
    IFS=',' read -ra arr <<< "$SKIP_LIST"
    for s in "${arr[@]}"; do
        n=$((10#$s))
        printf -v key "%03d" "$n"
        SKIP_SET[$key]=1
    done
fi

# ── Walk cases ─────────────────────────────────────────────────────────────
n_done=0
n_skipped_existing=0
n_skipped_userlist=0
n_failed=0

shopt -s nullglob
for case_dir in $CASE_GLOB; do
    name=$(basename "$case_dir")
    num="${name#cylindrical_}"
    num="${num%.gid}"
    if [[ ! "$num" =~ ^[0-9]+$ ]]; then
        echo "  $name: skipping (no numeric suffix)"
        continue
    fi

    if [ "${SKIP_SET[$num]:-0}" = 1 ]; then
        echo "  $name: SKIPPED (user --skip)"
        n_skipped_userlist=$((n_skipped_userlist + 1))
        continue
    fi

    out_path="$case_dir/$OUT_NAME"
    if [ -e "$out_path" ] && [ "$FORCE" = 0 ]; then
        echo "  $name: skipping (exists, use --force)"
        n_skipped_existing=$((n_skipped_existing + 1))
        continue
    fi

    case_dir_abs="$COHORT_PREFIX/$name"
    patch_template "$TEMPLATE_PATH" "$out_path" "$case_dir_abs" "$OUTPUT_SUBFOLDER"

    printf "  %s: written  (CASE_DIR=%s, OUTPUT_CASE_SUBFOLDER=%s)\n" \
        "$name" "$case_dir_abs" "$OUTPUT_SUBFOLDER"
    n_done=$((n_done + 1))
done

echo
echo "Summary:"
echo "  written:          $n_done"
echo "  skipped existing: $n_skipped_existing"
echo "  skipped (--skip): $n_skipped_userlist"
echo "  failed:           $n_failed"
[ "$n_failed" = 0 ]
