#!/bin/bash
# =============================================================================
# generate_jaxtrace_scripts.sh
#
# For every cylindrical_NNN.gid folder in the current directory, drop in a
# per-case run_jaxtrace.sh patched with:
#   - INPUT:          set to "$COHORT_PREFIX/<case_name>". The cohort
#                     prefix comes from --cohort-prefix=<path> if given,
#                     otherwise from $PWD (the directory you ran the
#                     generator from, as you typed it — no symlink
#                     resolution). On a machine where the cohort folder
#                     is mounted under a different prefix than the host
#                     that will run the simulation, pass
#                     --cohort-prefix=<runtime-native-path> so the
#                     stamped INPUT matches what the runtime host
#                     expects to see.
#   - INLET_VELOCITY: 3rd column of the first line after
#                     '$ Faces constraints => 1=111' in
#                     <case>/data/<stem>.som.fix
#                     (e.g. '1 111 5.0000000000e-03 0.0 0.0  0   0'
#                      → INLET_VELOCITY=5.0000000000e-03)
#   - PIN_RPM:        value after 'RPM:' in <case>/data/<stem>.som.dat
#                     (PIN_VELOCITY stays whatever the template has — your
#                      current template has PIN_VELOCITY=0, which means RPM
#                      is parsed and stored but the pin-velocity reconstruction
#                      is OFF. Edit the template if you want it on by default.)
#
# Other knobs (MESH_PATTERN, INLET_WALL, BOUNDARY_WALLS, …) are taken
# verbatim from the chosen template so this script never has to know about
# them. Edit the template to change anything common to all cases, then
# re-run with --force.
#
# Templates:
#   --platform=workstation (default)
#       template: ./cylindrical_001.gid/run_jaxtrace.sh
#       (canonical workstation flavor you already have set up)
#   --platform=lumi
#       template: $JAXTRACE_REPO/scripts/run_lumi.sh
#       (uses SBATCH directives, Singularity, paths under /project /scratch)
#
# Existing per-case run_jaxtrace.sh files are SKIPPED unless --force is
# given. The canonical template for workstation is itself in
# cylindrical_001.gid, so cylindrical_001 is also skipped by default;
# pass --force to rewrite it from itself (effectively a no-op patch).
#
# Usage:
#   ./generate_jaxtrace_scripts.sh                              # workstation, skip existing
#   ./generate_jaxtrace_scripts.sh --platform=lumi              # LUMI
#   ./generate_jaxtrace_scripts.sh --force                      # overwrite everything
#   ./generate_jaxtrace_scripts.sh --skip=004,005,006           # don't touch these
#   ./generate_jaxtrace_scripts.sh --jaxtrace-repo=/path        # explicit JAXTrace dir
#   ./generate_jaxtrace_scripts.sh \
#       --cohort-prefix=/scratch/shared/ROM/FOM                 # runtime-native cohort path
#   ./generate_jaxtrace_scripts.sh --csv                        # write case_parameters.csv
#   ./generate_jaxtrace_scripts.sh --csv=summary.csv            # write to a specific file
#   ./generate_jaxtrace_scripts.sh --csv --csv-numbers-only     # number-only CSV
#
# CSV: a row is emitted for every case where INLET_VELOCITY and PIN_RPM
# could be parsed, including cases where the per-case runner already
# exists and is being skipped. Columns are:
#   case_name,case_number,inlet_velocity,pin_rpm
# or, with --csv-numbers-only:
#   case_number,inlet_velocity,pin_rpm
# =============================================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
PLATFORM=workstation
FORCE=0
SKIP_LIST=""
# Default JAXTrace repo for LUMI template. Override with --jaxtrace-repo.
JAXTRACE_REPO="/flash/shared/jax/JAXTrace"
CASE_GLOB="cylindrical_*.gid"
# CSV summary of extracted values. Empty = disabled.
# When --csv (no value) is passed, the path defaults to ./case_parameters.csv.
CSV_OUTPUT=""
CSV_NAMES_ONLY=0      # 1 = number column only; 0 (default) = case name + number
# Parent directory under which the generated INPUT paths are written:
#   INPUT="$COHORT_PREFIX/<case_name>"
# Empty (the default) means "use $PWD as the user typed it" — useful when
# you run the generator from the cohort folder via its runtime-native
# path. Override with --cohort-prefix=<path> when the host running the
# generator sees the cohort folder under a different mount than the host
# that will run the generated scripts (e.g. the local machine sees
# /home/<user>/fsw-gpu/scratch/... while the workstation runtime expects
# /scratch/...). We deliberately do NOT call `readlink -f`, because that
# would resolve symlinks and substitute the local mount prefix.
COHORT_PREFIX=""

# Reference template for workstation: must exist as a hand-tuned baseline.
WORKSTATION_TEMPLATE="cylindrical_001.gid/run_jaxtrace.sh"

# Per-case output filename. The launcher below looks for this exact name.
OUT_NAME="run_jaxtrace.sh"

# ── Parse args ──────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --platform=*)        PLATFORM="${arg#*=}" ;;
        --force)             FORCE=1 ;;
        --skip)              ;;  # eaten by next iteration; see --skip=X below
        --skip=*)            SKIP_LIST="${arg#*=}" ;;
        --jaxtrace-repo=*)   JAXTRACE_REPO="${arg#*=}" ;;
        --cohort-prefix=*)   COHORT_PREFIX="${arg#*=}" ;;
        --csv)               CSV_OUTPUT="case_parameters.csv" ;;
        --csv=*)             CSV_OUTPUT="${arg#*=}" ;;
        --csv-numbers-only)  CSV_NAMES_ONLY=1 ;;
        --help|-h)
            sed -n '2,64p' "$0"
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
        echo "ERROR: workstation template not found at $TEMPLATE_PATH" >&2
        echo "Set up cylindrical_001.gid/run_jaxtrace.sh first (the canonical reference)." >&2
        exit 1
    fi
else
    TEMPLATE_PATH="$JAXTRACE_REPO/scripts/run_lumi.sh"
    if [ ! -f "$TEMPLATE_PATH" ]; then
        echo "ERROR: LUMI template not found at $TEMPLATE_PATH" >&2
        echo "Pass --jaxtrace-repo=/path/to/JAXTrace to override." >&2
        exit 1
    fi
fi
# ── Resolve cohort prefix ──────────────────────────────────────────────────
# Order: explicit --cohort-prefix wins; otherwise use $PWD as-is (no
# readlink, no realpath — we trust the user's typed path).
if [ -z "$COHORT_PREFIX" ]; then
    COHORT_PREFIX="$PWD"
fi
# Trim any trailing slash for tidy joins below.
COHORT_PREFIX="${COHORT_PREFIX%/}"

echo "Template: $TEMPLATE_PATH"
echo "Platform: $PLATFORM"
echo "Cohort:   $COHORT_PREFIX"
echo "Force:    $FORCE"
echo "Skip:     ${SKIP_LIST:-<none>}"
echo

# ── Per-case patch helpers ─────────────────────────────────────────────────
# Read INLET_VELOCITY from <case>/data/<stem>.som.fix. The relevant line is
# the FIRST data line after the marker '$ Faces constraints => 1=111'.
# Its 3rd whitespace-separated column is the X advancing velocity.
extract_inlet_velocity() {
    local som_fix="$1"
    [ -f "$som_fix" ] || { echo "" ; return ; }
    awk '
        /^[[:space:]]*\$ Faces constraints => 1=111[[:space:]]*$/ { found=1; next }
        found {
            # Stop on next "$ Faces constraints" block or section marker
            if ($0 ~ /^[[:space:]]*\$/) { exit }
            # First data line: print col 3 and exit
            print $3
            exit
        }
    ' "$som_fix"
}

# Read PIN_RPM from <case>/data/<stem>.som.dat. The relevant line looks like
# "            RPM: -400" — match RPM:, take the next field, strip whitespace.
extract_pin_rpm() {
    local som_dat="$1"
    [ -f "$som_dat" ] || { echo "" ; return ; }
    awk '
        /RPM[[:space:]]*:/ {
            # Drop everything up to and including the colon, take the
            # first whitespace-separated token afterward.
            sub(/.*RPM[[:space:]]*:[[:space:]]*/, "")
            print $1
            exit
        }
    ' "$som_dat"
}

# Patch INPUT, INLET_VELOCITY, PIN_RPM, AUTO_DETECT_CASE in the template.
#
# INPUT is set to "$COHORT_PREFIX/<case_name>", giving each generated
# script an explicit, host-native input path. We also force
# AUTO_DETECT_CASE=0 so the runner uses that stamped INPUT verbatim
# instead of re-deriving the case dir from `pwd -P` at run time —
# this makes the launch behaviour deterministic regardless of CWD or
# symlinks.
#
# We use a temp file rather than sed -i so the template is not modified
# and any failure leaves the output as it was.
patch_template() {
    local template="$1"
    local output="$2"
    local input_path="$3"
    local inlet_v="$4"
    local pin_rpm="$5"

    # First line at column 0 for each variable. The leading-anchor
    # 0,/^VAR=/ form restricts the substitution to the first match so
    # comments or echoes later in the script are not affected.
    sed \
        -e "0,/^INPUT=/{s|^INPUT=.*|INPUT=\"$input_path\"|}" \
        -e "0,/^AUTO_DETECT_CASE=/{s|^AUTO_DETECT_CASE=.*|AUTO_DETECT_CASE=0|}" \
        -e "0,/^INLET_VELOCITY=/{s|^INLET_VELOCITY=.*|INLET_VELOCITY=$inlet_v|}" \
        -e "0,/^PIN_RPM=/{s|^PIN_RPM=.*|PIN_RPM=$pin_rpm|}" \
        "$template" > "$output.tmp"

    mv "$output.tmp" "$output"
    chmod +x "$output"
}


# ── Build skip set ─────────────────────────────────────────────────────────
declare -A SKIP_SET
if [ -n "$SKIP_LIST" ]; then
    IFS=',' read -ra arr <<< "$SKIP_LIST"
    for s in "${arr[@]}"; do
        # Accept either "4" or "004"; normalise to 3-digit
        n=$((10#$s))
        printf -v key "%03d" "$n"
        SKIP_SET[$key]=1
    done
fi

# ── Initialise CSV if requested ────────────────────────────────────────────
# The CSV captures every case where we successfully parsed values, even
# cases where we skipped writing the per-case runner (because it already
# exists). That way the CSV is a complete catalogue of the cohort,
# independent of whether the runners are regenerated.
if [ -n "$CSV_OUTPUT" ]; then
    echo "CSV:      $CSV_OUTPUT"
    if [ "$CSV_NAMES_ONLY" = 1 ]; then
        echo "case_number,inlet_velocity,pin_rpm" > "$CSV_OUTPUT"
    else
        echo "case_name,case_number,inlet_velocity,pin_rpm" > "$CSV_OUTPUT"
    fi
fi
echo

# ── Walk cases ─────────────────────────────────────────────────────────────
n_done=0
n_skipped_existing=0
n_skipped_userlist=0
n_failed=0
n_csv_rows=0

shopt -s nullglob
for case_dir in $CASE_GLOB; do
    # Extract NNN from cylindrical_NNN.gid
    name=$(basename "$case_dir")
    num="${name#cylindrical_}"
    num="${num%.gid}"
    if [[ ! "$num" =~ ^[0-9]+$ ]]; then
        echo "  $name: skipping (no numeric suffix)"
        continue
    fi

    # Skip-list check (user explicitly excluded this case)
    if [ "${SKIP_SET[$num]:-0}" = 1 ]; then
        echo "  $name: SKIPPED (user --skip)"
        n_skipped_userlist=$((n_skipped_userlist + 1))
        continue
    fi

    # Find the SOM files; the stem can vary (some cases share a single
    # cylindrical.* stem inside data/).
    data_dir="$case_dir/data"
    if [ ! -d "$data_dir" ]; then
        echo "  $name: FAILED — no data/ directory"
        n_failed=$((n_failed + 1))
        continue
    fi
    som_fix=$(ls "$data_dir"/*.som.fix 2>/dev/null | head -1)
    som_dat=$(ls "$data_dir"/*.som.dat 2>/dev/null | head -1)
    if [ -z "$som_fix" ] || [ -z "$som_dat" ]; then
        echo "  $name: FAILED — missing .som.fix or .som.dat in $data_dir"
        n_failed=$((n_failed + 1))
        continue
    fi

    inlet_v=$(extract_inlet_velocity "$som_fix")
    pin_rpm=$(extract_pin_rpm "$som_dat")
    if [ -z "$inlet_v" ]; then
        echo "  $name: FAILED — could not parse INLET_VELOCITY from $som_fix"
        n_failed=$((n_failed + 1))
        continue
    fi
    if [ -z "$pin_rpm" ]; then
        echo "  $name: FAILED — could not parse PIN_RPM from $som_dat"
        n_failed=$((n_failed + 1))
        continue
    fi

    # Append a CSV row for this case before we decide whether to write
    # the runner — so a re-run with --csv produces a complete catalogue
    # even when most runners already exist.
    if [ -n "$CSV_OUTPUT" ]; then
        if [ "$CSV_NAMES_ONLY" = 1 ]; then
            echo "$num,$inlet_v,$pin_rpm" >> "$CSV_OUTPUT"
        else
            echo "$name,$num,$inlet_v,$pin_rpm" >> "$CSV_OUTPUT"
        fi
        n_csv_rows=$((n_csv_rows + 1))
    fi

    # Decide whether to write the per-case runner.
    out_path="$case_dir/$OUT_NAME"
    if [ -e "$out_path" ] && [ "$FORCE" = 0 ]; then
        echo "  $name: skipping (exists, use --force)"
        n_skipped_existing=$((n_skipped_existing + 1))
        continue
    fi

    # Compose INPUT from the user-supplied cohort prefix (or $PWD) and
    # the case directory name. The prefix is taken verbatim, so the
    # caller is responsible for ensuring it matches the path the
    # runtime host expects to see (e.g. /scratch/shared/ROM/FOM on the
    # workstation, not /home/<user>/fsw-gpu/scratch/... on the local
    # box).
    input_path="$COHORT_PREFIX/$name"

    patch_template "$TEMPLATE_PATH" "$out_path" "$input_path" "$inlet_v" "$pin_rpm"

    printf "  %s: written  (INPUT=%s, INLET_VELOCITY=%s, PIN_RPM=%s)\n" \
        "$name" "$input_path" "$inlet_v" "$pin_rpm"
    n_done=$((n_done + 1))
done

echo
echo "Summary:"
echo "  written:          $n_done"
echo "  skipped existing: $n_skipped_existing"
echo "  skipped (--skip): $n_skipped_userlist"
echo "  failed:           $n_failed"
if [ -n "$CSV_OUTPUT" ]; then
    echo "  csv rows:         $n_csv_rows  ($CSV_OUTPUT)"
fi
[ "$n_failed" = 0 ]
