#!/bin/bash
# =============================================================================
# launch_jaxtrace.sh
#
# Bundle-launch every cylindrical_NNN.gid/run_jaxtrace.sh in this folder.
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
# Usage:
#   ./launch_jaxtrace.sh                                 # workstation, all cases
#   ./launch_jaxtrace.sh --platform=lumi                 # LUMI parallel submit
#   ./launch_jaxtrace.sh --skip=004,005,006              # skip those cases
#   ./launch_jaxtrace.sh --only=000,001                  # only those cases
#   ./launch_jaxtrace.sh --dry-run                       # print, don't launch
#
# Expectation: each case folder has its own run_jaxtrace.sh, produced by
# generate_jaxtrace_scripts.sh. Cases without one are reported and skipped.
# =============================================================================

set -euo pipefail

PLATFORM=workstation
SKIP_LIST=""
ONLY_LIST=""
DRY_RUN=0
RUN_NAME="run_jaxtrace.sh"
CASE_GLOB="cylindrical_*.gid"

for arg in "$@"; do
    case "$arg" in
        --platform=*)  PLATFORM="${arg#*=}" ;;
        --skip=*)      SKIP_LIST="${arg#*=}" ;;
        --only=*)      ONLY_LIST="${arg#*=}" ;;
        --dry-run)     DRY_RUN=1 ;;
        --help|-h)
            sed -n '2,30p' "$0"
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

echo "Platform:  $PLATFORM"
echo "Skip:      ${SKIP_LIST:-<none>}"
echo "Only:      ${ONLY_LIST:-<all>}"
echo "Dry run:   $DRY_RUN"
echo

n_launched=0
n_missing=0
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
        echo "  $name: MISSING $RUN_NAME — run generate_jaxtrace_scripts.sh first"
        n_missing=$((n_missing + 1))
        continue
    fi

    if [ "$PLATFORM" = "lumi" ]; then
        # Parallel: hand off to SLURM and move on immediately. Each case's
        # run_jaxtrace.sh carries its own #SBATCH directives.
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
            echo "  Running $name"
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
echo "  launched/submitted:  $n_launched"
echo "  missing runner:      $n_missing"
echo "  skipped (--skip):    $n_skipped"
echo "  failed:              $n_failed"
[ "$n_failed" = 0 ] && [ "$n_missing" = 0 ]
