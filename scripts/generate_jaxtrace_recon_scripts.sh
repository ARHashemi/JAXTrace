#!/bin/bash
# =============================================================================
# generate_jaxtrace_recon_scripts.sh
#
# Walk a cohort of cylindrical_NNN.gid folders and, for each one, generate
# a run_jaxtrace_recon.sh derived from that case's own run_jaxtrace.sh
# via the per-case helper scripts/generate_run_jaxtrace_recon.sh.
#
# This is the bulk sibling of scripts/generate_jaxtrace_scripts.sh — same
# cohort-walking / --skip / --force plumbing, but instead of stamping a
# fresh FOM tracker from the shared template, it turns each case's
# existing FOM runner into its ROM-velocity twin.
#
# What the per-case helper actually does (unchanged from before):
#   * inherits every physics knob from run_jaxtrace.sh (PIN_RPM, DT,
#     N_STEPS, SEED_*, boundary walls, level-set, HCT-3D recovery, ...)
#   * only swaps the velocity-source paths:
#       INPUT                 -> <recon-root>/<case>.gid
#       VEL_START/END         -> 0 0    (recon writes one snapshot)
#       OUTPUT_CASE_SUBFOLDER -> post_pt/rom_<formula>  (nests inside post_pt/)
#       AUTO_DETECT_CASE      -> 0
#       RUN_TAG               -> ""
#   * the ROM PVTU itself must already exist for the case at
#       <recon-root>/<case>.gid/post/cylindrical_0.pvtu
#     (produced by /scratch/shared/ROM/FOM/reconstruct_rom_velocities.sh)
#
# Usage:
#   ./generate_jaxtrace_recon_scripts.sh
#   ./generate_jaxtrace_recon_scripts.sh --force
#   ./generate_jaxtrace_recon_scripts.sh --formula=c_over_sig
#   ./generate_jaxtrace_recon_scripts.sh --skip=004,005
#   ./generate_jaxtrace_recon_scripts.sh --only=001,004
#   ./generate_jaxtrace_recon_scripts.sh --skip-preflight     # don't check for recon PVTU
#   ./generate_jaxtrace_recon_scripts.sh \
#       --recon-root=/scratch/shared/ROM/ROM_recon_centered   # explicit recon-root
#   ./generate_jaxtrace_recon_scripts.sh --jaxtrace-repo=/path
#
# Preflight: unless --skip-preflight is set, each case is checked for a
# ROM PVTU at <recon-root>/<case>.gid/post/cylindrical_0.pvtu.  Cases
# without one are reported and their runner is NOT generated (running it
# would crash immediately in run_tracking.py's mesh loader).  Set
# --skip-preflight when you plan to regenerate the PVTUs later.
#
# Environment override (only for the per-case sed-fix; won't override
# the auto-derived RECON_ROOT below unless --recon-root=... is used):
#   ROM_FORMULA (also settable via --formula=)
# =============================================================================

set -uo pipefail

# ── Defaults ───────────────────────────────────────────────────────────────
FORCE=0
SKIP_LIST=""
ONLY_LIST=""
FORMULA="${ROM_FORMULA:-centered}"
RECON_ROOT=""
SKIP_PREFLIGHT=0
JAXTRACE_REPO="/flash/shared/jax/JAXTrace"
CASE_GLOB="cylindrical_*.gid"
OUT_NAME="run_jaxtrace_recon.sh"
SRC_NAME="run_jaxtrace.sh"

# ── Parse args ────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --force)             FORCE=1 ;;
        --skip=*)            SKIP_LIST="${arg#*=}" ;;
        --only=*)            ONLY_LIST="${arg#*=}" ;;
        --formula=*)         FORMULA="${arg#*=}" ;;
        --recon-root=*)      RECON_ROOT="${arg#*=}" ;;
        --skip-preflight)    SKIP_PREFLIGHT=1 ;;
        --jaxtrace-repo=*)   JAXTRACE_REPO="${arg#*=}" ;;
        --help|-h)
            sed -n '2,48p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Try --help" >&2
            exit 1
            ;;
    esac
done

# ── Locate the per-case helper ────────────────────────────────────────────
HELPER="$JAXTRACE_REPO/scripts/generate_run_jaxtrace_recon.sh"
if [ ! -f "$HELPER" ]; then
    echo "ERROR: per-case helper not found: $HELPER" >&2
    echo "Pass --jaxtrace-repo=/path/to/JAXTrace to override." >&2
    exit 2
fi

# ── Default recon-root: sibling of the current cohort folder ──────────────
# Called from /scratch/shared/ROM/FOM (or wherever the cohort lives), so
# the default is /scratch/shared/ROM/ROM_recon_<formula> — matching the
# convention of /scratch/shared/ROM/FOM/reconstruct_rom_velocities.sh.
if [ -z "$RECON_ROOT" ]; then
    _COHORT_PARENT="$(cd .. && pwd)"
    RECON_ROOT="${_COHORT_PARENT}/ROM_recon_${FORMULA}"
fi

# ── Build skip / only sets ────────────────────────────────────────────────
declare -A SKIP_SET
declare -A ONLY_SET
ONLY_MODE=0
if [ -n "$SKIP_LIST" ]; then
    IFS=',' read -ra arr <<< "$SKIP_LIST"
    for s in "${arr[@]}"; do
        n=$((10#$s))
        printf -v key "%03d" "$n"
        SKIP_SET[$key]=1
    done
fi
if [ -n "$ONLY_LIST" ]; then
    ONLY_MODE=1
    IFS=',' read -ra arr <<< "$ONLY_LIST"
    for s in "${arr[@]}"; do
        n=$((10#$s))
        printf -v key "%03d" "$n"
        ONLY_SET[$key]=1
    done
fi

echo "Helper:      $HELPER"
echo "Formula:     $FORMULA"
echo "Recon-root:  $RECON_ROOT"
echo "Preflight:   $( [ "$SKIP_PREFLIGHT" = 1 ] && echo off || echo on )"
echo "Cohort:      $PWD"
echo "Force:       $FORCE"
echo "Skip:        ${SKIP_LIST:-<none>}"
echo "Only:        ${ONLY_LIST:-<all>}"
echo

# ── Walk cases ────────────────────────────────────────────────────────────
n_done=0
n_skipped_existing=0
n_skipped_userlist=0
n_skipped_no_src=0
n_skipped_no_pvtu=0
n_failed=0

shopt -s nullglob
for case_dir in $CASE_GLOB; do
    name=$(basename "$case_dir")
    num="${name#cylindrical_}"
    num="${num%.gid}"
    if [[ ! "$num" =~ ^[0-9]+$ ]]; then
        continue
    fi

    if [ "${SKIP_SET[$num]:-0}" = 1 ]; then
        echo "  $name: skipped (--skip)"
        n_skipped_userlist=$((n_skipped_userlist + 1))
        continue
    fi
    if [ "$ONLY_MODE" = 1 ] && [ "${ONLY_SET[$num]:-0}" != 1 ]; then
        continue
    fi

    src="$case_dir/$SRC_NAME"
    if [ ! -f "$src" ]; then
        echo "  $name: skipped (no $SRC_NAME — run generate_jaxtrace_scripts.sh first)"
        n_skipped_no_src=$((n_skipped_no_src + 1))
        continue
    fi

    # Preflight: the ROM PVTU must exist for the case (unless disabled).
    if [ "$SKIP_PREFLIGHT" != 1 ]; then
        recon_pvtu="$RECON_ROOT/$name/post/cylindrical_0.pvtu"
        if [ ! -f "$recon_pvtu" ]; then
            echo "  $name: skipped (no ROM PVTU at $recon_pvtu)"
            n_skipped_no_pvtu=$((n_skipped_no_pvtu + 1))
            continue
        fi
    fi

    out="$case_dir/$OUT_NAME"
    if [ -e "$out" ] && [ "$FORCE" = 0 ]; then
        echo "  $name: skipping (exists, use --force)"
        n_skipped_existing=$((n_skipped_existing + 1))
        continue
    fi

    # Delegate to the per-case helper.  It reads the case's own
    # run_jaxtrace.sh, applies the fixed substitutions, and writes
    # run_jaxtrace_recon.sh next to it.
    if ROM_FORMULA="$FORMULA" ROM_RECON_ROOT="$RECON_ROOT" \
        bash "$HELPER" "$case_dir" >/dev/null 2>&1; then
        if [ -f "$out" ]; then
            n_done=$((n_done + 1))
            echo "  $name: written  (formula=$FORMULA, input=$RECON_ROOT/$name)"
        else
            echo "  $name: helper reported success but $out not present" >&2
            n_failed=$((n_failed + 1))
        fi
    else
        rc=$?
        echo "  $name: helper FAILED (exit $rc)" >&2
        n_failed=$((n_failed + 1))
    fi
done

echo
echo "Summary:"
echo "  written:                       $n_done"
echo "  skipped existing (no --force): $n_skipped_existing"
echo "  skipped (--skip):              $n_skipped_userlist"
echo "  skipped (no source runner):    $n_skipped_no_src"
echo "  skipped (no recon PVTU):       $n_skipped_no_pvtu"
echo "  failed:                        $n_failed"
[ "$n_failed" = 0 ]
