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
#   - DT:             value after 'TIME_STEP_SIZE:' in <case>/data/<stem>.dat.
#                     Always stamped (even when --max-steps is not given), so
#                     the runner uses the same dt as the upstream solver.
#   - N_STEPS:        only patched when --max-steps=N is given. We then
#                     compute v_max across all parseable cases (taking each
#                     case's own dt and v), define
#                       D_max = v_max * dt_at_v_max * N
#                     and assign each case
#                       N_STEPS_case = ceil(D_max / (v_case * dt_case))
#                     so every case simulates the same physical travel
#                     distance. The fastest case gets exactly N steps;
#                     every other case gets more.
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
#   ./generate_jaxtrace_scripts.sh --max-steps=2000             # equalise travel distance
#   ./generate_jaxtrace_scripts.sh --csv                        # write case_parameters.csv
#   ./generate_jaxtrace_scripts.sh --csv=summary.csv            # write to a specific file
#   ./generate_jaxtrace_scripts.sh --csv --csv-numbers-only     # number-only CSV
#
# CSV: a row is emitted for every case where all per-case values could
# be parsed, including cases where the per-case runner already exists
# and is being skipped. Columns are:
#   case_name,case_number,inlet_velocity,pin_rpm,dt[,n_steps]
# or, with --csv-numbers-only:
#   case_number,inlet_velocity,pin_rpm,dt[,n_steps]
# The n_steps column is included only when --max-steps was passed.
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
# When non-empty, normalise every case so each one travels the same
# physical distance as the case with the largest INLET_VELOCITY would
# travel in MAX_STEPS steps. For the slower cases we increase N_STEPS
# accordingly. DT is taken from each case's data/<stem>.dat
# (TIME_STEP_SIZE field). The fastest case keeps exactly MAX_STEPS;
# every other case gets ceil(D_max / (v_case * dt_case)) where
# D_max = v_max * dt_max * MAX_STEPS is the reference travel distance.
# Empty = disabled (use the template's N_STEPS / DT verbatim).
MAX_STEPS=""

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
        --max-steps=*)       MAX_STEPS="${arg#*=}" ;;
        --csv)               CSV_OUTPUT="case_parameters.csv" ;;
        --csv=*)             CSV_OUTPUT="${arg#*=}" ;;
        --csv-numbers-only)  CSV_NAMES_ONLY=1 ;;
        --help|-h)
            sed -n '2,78p' "$0"
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

# Validate --max-steps if given.
if [ -n "$MAX_STEPS" ]; then
    if ! [[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: --max-steps must be a positive integer (got '$MAX_STEPS')" >&2
        exit 1
    fi
fi

echo "Template:  $TEMPLATE_PATH"
echo "Platform:  $PLATFORM"
echo "Cohort:    $COHORT_PREFIX"
echo "Force:     $FORCE"
echo "Skip:      ${SKIP_LIST:-<none>}"
if [ -n "$MAX_STEPS" ]; then
    echo "Max steps: $MAX_STEPS  (applied to the case with the largest INLET_VELOCITY)"
else
    echo "Max steps: <unchanged from template>"
fi
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

# Read TIME_STEP_SIZE from <case>/data/<stem>.dat. The relevant line looks
# like "     TIME_STEP_SIZE:           3.7500000000e-03".
extract_time_step_size() {
    local dat="$1"
    [ -f "$dat" ] || { echo "" ; return ; }
    awk '
        /TIME_STEP_SIZE[[:space:]]*:/ {
            sub(/.*TIME_STEP_SIZE[[:space:]]*:[[:space:]]*/, "")
            print $1
            exit
        }
    ' "$dat"
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
    local dt="$6"          # may be "" → keep template's DT
    local n_steps="$7"     # may be "" → keep template's N_STEPS

    # First line at column 0 for each variable. The leading-anchor
    # 0,/^VAR=/ form restricts the substitution to the first match so
    # comments or echoes later in the script are not affected.
    local -a sed_args=(
        -e "0,/^INPUT=/{s|^INPUT=.*|INPUT=\"$input_path\"|}"
        -e "0,/^AUTO_DETECT_CASE=/{s|^AUTO_DETECT_CASE=.*|AUTO_DETECT_CASE=0|}"
        -e "0,/^INLET_VELOCITY=/{s|^INLET_VELOCITY=.*|INLET_VELOCITY=$inlet_v|}"
        -e "0,/^PIN_RPM=/{s|^PIN_RPM=.*|PIN_RPM=$pin_rpm|}"
    )
    if [ -n "$dt" ]; then
        sed_args+=( -e "0,/^DT=/{s|^DT=.*|DT=$dt|}" )
    fi
    if [ -n "$n_steps" ]; then
        sed_args+=( -e "0,/^N_STEPS=/{s|^N_STEPS=.*|N_STEPS=$n_steps|}" )
    fi
    sed "${sed_args[@]}" "$template" > "$output.tmp"

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

# ── Pass 0: scan for reference case (only when --max-steps is given) ──────
# We need to know the case with the largest INLET_VELOCITY so we can
# compute D_max = v_max * dt_at_v_max * MAX_STEPS and scale every other
# case's N_STEPS to match. The pass is silent — failed-to-parse cases
# are just skipped here; pass 1 below will issue user-facing errors.
D_MAX=""
V_MAX=""
DT_AT_VMAX=""
NAME_VMAX=""
if [ -n "$MAX_STEPS" ]; then
    shopt -s nullglob
    for _case_dir in $CASE_GLOB; do
        _name=$(basename "$_case_dir")
        _num="${_name#cylindrical_}"
        _num="${_num%.gid}"
        [[ "$_num" =~ ^[0-9]+$ ]] || continue
        [ "${SKIP_SET[$_num]:-0}" = 1 ] && continue
        _ddir="$_case_dir/data"
        [ -d "$_ddir" ] || continue
        _som_fix=$(ls "$_ddir"/*.som.fix 2>/dev/null | head -1)
        # The case-level data file is <stem>.dat (one dot in basename);
        # cylindrical.som.dat / cylindrical.dom.dat / etc are subdomain
        # files we don't want here.
        _dat=$(find "$_ddir" -maxdepth 1 -type f -regex '.*/[^./]*\.dat$' 2>/dev/null | head -1)
        [ -n "$_som_fix" ] && [ -n "$_dat" ] || continue
        _v=$(extract_inlet_velocity "$_som_fix")
        _dt=$(extract_time_step_size "$_dat")
        [ -n "$_v" ] && [ -n "$_dt" ] || continue
        # Compare floats via awk; if current v beats previous max,
        # remember it together with this case's dt and name.
        if [ -z "$V_MAX" ] || awk -v a="$_v" -v b="$V_MAX" 'BEGIN{exit !(a+0 > b+0)}'; then
            V_MAX="$_v"
            DT_AT_VMAX="$_dt"
            NAME_VMAX="$_name"
        fi
    done
    if [ -z "$V_MAX" ]; then
        echo "ERROR: --max-steps was given but no case yielded a parseable" >&2
        echo "       INLET_VELOCITY + TIME_STEP_SIZE pair. Cannot compute" >&2
        echo "       reference travel distance." >&2
        exit 1
    fi
    D_MAX=$(awk -v v="$V_MAX" -v dt="$DT_AT_VMAX" -v n="$MAX_STEPS" \
        'BEGIN{printf "%.17g", v*dt*n}')
    echo "Ref case:  $NAME_VMAX  (v=$V_MAX m/s, dt=$DT_AT_VMAX s)"
    echo "D_max:     $D_MAX m  (= v_max * dt * MAX_STEPS)"
    echo
fi

# ── Initialise CSV if requested ────────────────────────────────────────────
# The CSV captures every case where we successfully parsed values, even
# cases where we skipped writing the per-case runner (because it already
# exists). That way the CSV is a complete catalogue of the cohort,
# independent of whether the runners are regenerated.
if [ -n "$CSV_OUTPUT" ]; then
    echo "CSV:       $CSV_OUTPUT"
    # Header columns: always include dt (always parsed). n_steps is
    # included only when --max-steps was given (otherwise N_STEPS
    # comes verbatim from the template and isn't meaningful per case).
    if [ "$CSV_NAMES_ONLY" = 1 ]; then
        _hdr="case_number,inlet_velocity,pin_rpm,dt"
    else
        _hdr="case_name,case_number,inlet_velocity,pin_rpm,dt"
    fi
    [ -n "$MAX_STEPS" ] && _hdr="$_hdr,n_steps"
    echo "$_hdr" > "$CSV_OUTPUT"
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
    dat_file=$(find "$data_dir" -maxdepth 1 -type f -regex '.*/[^./]*\.dat$' 2>/dev/null | head -1)
    if [ -z "$som_fix" ] || [ -z "$som_dat" ]; then
        echo "  $name: FAILED — missing .som.fix or .som.dat in $data_dir"
        n_failed=$((n_failed + 1))
        continue
    fi
    if [ -z "$dat_file" ]; then
        echo "  $name: FAILED — missing <stem>.dat in $data_dir"
        n_failed=$((n_failed + 1))
        continue
    fi

    inlet_v=$(extract_inlet_velocity "$som_fix")
    pin_rpm=$(extract_pin_rpm "$som_dat")
    dt=$(extract_time_step_size "$dat_file")
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
    if [ -z "$dt" ]; then
        echo "  $name: FAILED — could not parse TIME_STEP_SIZE from $dat_file"
        n_failed=$((n_failed + 1))
        continue
    fi

    # Compute case-specific N_STEPS so this case travels the same
    # physical distance as the reference (max-velocity) case. The
    # fastest case gets exactly MAX_STEPS; slower cases get more.
    # We compute ceil(D_max / (v_case * dt_case)) in awk to keep float
    # precision; bash arithmetic is integer-only.
    n_steps=""
    if [ -n "$MAX_STEPS" ]; then
        n_steps=$(awk -v d="$D_MAX" -v v="$inlet_v" -v dt="$dt" \
            'BEGIN{
                step = v*dt
                if (step <= 0) { print ""; exit }
                n = d / step
                # ceil
                ni = int(n)
                if (n > ni) ni = ni + 1
                if (ni < 1) ni = 1
                print ni
            }')
        if [ -z "$n_steps" ]; then
            echo "  $name: FAILED — v*dt is non-positive, cannot scale N_STEPS"
            n_failed=$((n_failed + 1))
            continue
        fi
    fi

    # Append a CSV row for this case before we decide whether to write
    # the runner — so a re-run with --csv produces a complete catalogue
    # even when most runners already exist.
    if [ -n "$CSV_OUTPUT" ]; then
        if [ "$CSV_NAMES_ONLY" = 1 ]; then
            _row="$num,$inlet_v,$pin_rpm,$dt"
        else
            _row="$name,$num,$inlet_v,$pin_rpm,$dt"
        fi
        [ -n "$MAX_STEPS" ] && _row="$_row,$n_steps"
        echo "$_row" >> "$CSV_OUTPUT"
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

    # Always patch DT (read from the case's own data file). Patch
    # N_STEPS too only when --max-steps was given; otherwise the
    # template's N_STEPS is preserved (pass "" to patch_template to
    # skip the substitution).
    patch_template "$TEMPLATE_PATH" "$out_path" "$input_path" \
        "$inlet_v" "$pin_rpm" "$dt" "$n_steps"

    if [ -n "$n_steps" ]; then
        printf "  %s: written  (INPUT=%s, INLET_VELOCITY=%s, PIN_RPM=%s, DT=%s, N_STEPS=%s)\n" \
            "$name" "$input_path" "$inlet_v" "$pin_rpm" "$dt" "$n_steps"
    else
        printf "  %s: written  (INPUT=%s, INLET_VELOCITY=%s, PIN_RPM=%s, DT=%s)\n" \
            "$name" "$input_path" "$inlet_v" "$pin_rpm" "$dt"
    fi
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
