#!/bin/bash
# =============================================================================
# launch_jaxtrace_recon.sh
#
# Bundle-launch every cylindrical_NNN.gid/run_jaxtrace_recon.sh in this
# folder.  Sibling of launch_jaxtrace.sh, but targets the ROM-reconstructed-
# velocity tracking runner instead of the FOM tracking runner.
#
# Two modes — pick at launch time:
#
#   --platform=workstation (default)
#       Run cases ONE AT A TIME, foreground. The current case's stdout/err is
#       streamed live; the next case starts only after the previous finishes.
#       Designed for a single-GPU workstation where parallel runs would
#       contend for the device.
#
#   --platform=lumi
#       Submit each case to SLURM with `sbatch`. Cases run concurrently as
#       SLURM allocates resources. Each case writes its own SLURM
#       stdout/err per its #SBATCH --output / --error directives.
#
# Preflight: a case is skipped (reported as MISSING RECON PVTU) if the
# corresponding ROM-reconstructed velocity PVTU does not exist on disk
# for the formula the runner points at.  The default location is
#   /scratch/shared/ROM_recon_<formula>/<case>.gid/post/cylindrical_0.pvtu
# override with --recon-root=/some/path.
#
# Usage:
#   ./launch_jaxtrace_recon.sh                                # workstation, all cases with a recon PVTU
#   ./launch_jaxtrace_recon.sh --platform=lumi                # LUMI parallel submit
#   ./launch_jaxtrace_recon.sh --skip=004,005,006             # skip those cases
#   ./launch_jaxtrace_recon.sh --only=000,001                 # only those cases
#   ./launch_jaxtrace_recon.sh --dry-run                      # print, don't launch
#   ./launch_jaxtrace_recon.sh --formula=c_over_sig           # change expected recon-root suffix
#   ./launch_jaxtrace_recon.sh --recon-root=/other/path       # override recon-root entirely
#   ./launch_jaxtrace_recon.sh --skip-preflight               # don't check for recon PVTU
#
# Expectation: each case folder has its own run_jaxtrace_recon.sh, produced
# by generate_jaxtrace_recon_scripts.sh (or the per-case
# generate_run_jaxtrace_recon.sh helper).  Cases without one are reported
# and skipped.  The ROM PVTUs themselves come from
#   /scratch/shared/ROM/FOM/reconstruct_rom_velocities.sh
# =============================================================================

set -euo pipefail

PLATFORM=workstation
SKIP_LIST=""
ONLY_LIST=""
DRY_RUN=0
FORMULA="centered"
RECON_ROOT=""
SKIP_PREFLIGHT=0
RUN_NAME="run_jaxtrace_recon.sh"
CASE_GLOB="cylindrical_*.gid"

for arg in "$@"; do
    case "$arg" in
        --platform=*)     PLATFORM="${arg#*=}" ;;
        --skip=*)         SKIP_LIST="${arg#*=}" ;;
        --only=*)         ONLY_LIST="${arg#*=}" ;;
        --dry-run)        DRY_RUN=1 ;;
        --formula=*)      FORMULA="${arg#*=}" ;;
        --recon-root=*)   RECON_ROOT="${arg#*=}" ;;
        --skip-preflight) SKIP_PREFLIGHT=1 ;;
        --help|-h)
            sed -n '2,42p' "$0"
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
    *) echo "ERROR: --platform must be 'workstation' or 'lumi'" >&2; exit 1 ;;
esac

# Default recon-root is a sibling of the current cohort folder (FOM/) named
# ROM_recon_<formula>.  Same convention as reconstruct_rom_velocities.sh.
if [ -z "$RECON_ROOT" ]; then
    _COHORT_PARENT="$(cd .. && pwd)"
    RECON_ROOT="${_COHORT_PARENT}/ROM_recon_${FORMULA}"
fi

# Build skip / only sets (3-digit padded keys).
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

echo "Platform:       $PLATFORM"
echo "Formula:        $FORMULA"
echo "Recon-root:     $RECON_ROOT"
echo "Preflight:      $( [ "$SKIP_PREFLIGHT" = 1 ] && echo off || echo on )"
echo "Skip:           ${SKIP_LIST:-<none>}"
echo "Only:           ${ONLY_LIST:-<all>}"
echo "Dry run:        $DRY_RUN"
echo

n_launched=0
n_missing_runner=0
n_missing_pvtu=0
n_skipped=0
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
        n_skipped=$((n_skipped + 1))
        continue
    fi
    if [ "$ONLY_MODE" = 1 ] && [ "${ONLY_SET[$num]:-0}" != 1 ]; then
        continue
    fi

    runner="$case_dir/$RUN_NAME"
    if [ ! -f "$runner" ]; then
        echo "  $name: MISSING $RUN_NAME — run generate_jaxtrace_recon_scripts.sh first"
        n_missing_runner=$((n_missing_runner + 1))
        continue
    fi

    # Preflight: does the ROM-reconstructed PVTU exist for this case?
    # The per-case runner will crash immediately in run_tracking.py's
    # mesh loader if it doesn't, so catch it early with a clearer message.
    if [ "$SKIP_PREFLIGHT" != 1 ]; then
        recon_pvtu="$RECON_ROOT/$name/post/cylindrical_0.pvtu"
        if [ ! -f "$recon_pvtu" ]; then
            echo "  $name: MISSING RECON PVTU at $recon_pvtu"
            echo "                run reconstruct_rom_velocities.sh (CASES=\"${num#0}\") first"
            n_missing_pvtu=$((n_missing_pvtu + 1))
            continue
        fi
    fi

    if [ "$PLATFORM" = "lumi" ]; then
        # Parallel: hand off to SLURM and move on immediately. Each case's
        # run_jaxtrace_recon.sh carries its own #SBATCH directives.
        if [ "$DRY_RUN" = 1 ]; then
            echo "  $name: DRY-RUN — would do: (cd $case_dir && sbatch $RUN_NAME)"
        else
            echo "  $name: submitting..."
            if ( cd "$case_dir" && sbatch "$RUN_NAME" ); then
                n_launched=$((n_launched + 1))
            else
                echo "  $name: sbatch FAILED"
                n_failed=$((n_failed + 1))
            fi
        fi
    else
        # Sequential: run in the foreground, wait for completion before
        # starting the next case so they don't contend for the GPU.
        if [ "$DRY_RUN" = 1 ]; then
            echo "  $name: DRY-RUN — would do: (cd $case_dir && bash $RUN_NAME)"
        else
            echo
            echo "================================================================="
            echo "  Running $name (ROM recon, formula=$FORMULA)"
            echo "  Started: $(date)"
            echo "================================================================="
            if ( cd "$case_dir" && bash "$RUN_NAME" ); then
                n_launched=$((n_launched + 1))
                echo "  $name: finished OK at $(date)"
            else
                rc=$?
                echo "  $name: FAILED (exit $rc) — continuing with the next case"
                n_failed=$((n_failed + 1))
            fi
        fi
    fi
done

echo
echo "Summary:"
echo "  launched/submitted:      $n_launched"
echo "  missing runner:          $n_missing_runner"
echo "  missing recon PVTU:      $n_missing_pvtu"
echo "  skipped (--skip):        $n_skipped"
echo "  failed:                  $n_failed"
[ "$n_failed" = 0 ] && [ "$n_missing_runner" = 0 ]
