#!/bin/bash
# =============================================================================
# generate_run_jaxtrace_recon.sh  —  Generate a run_jaxtrace_recon.sh
# tracker inside a FOM case folder, derived from its existing
# run_jaxtrace.sh but with the velocity source swapped to the
# ROM-reconstructed field.
#
# Rationale:
#   * Each ROM/FOM case folder already contains a run_jaxtrace.sh
#     hand-tuned for that case's physics (pin RPM, dt, N_steps,
#     seeding, boundary walls, ...).  We MUST preserve those values
#     when we run the ROM-reconstructed field, otherwise the FOM-vs-ROM
#     comparison is comparing apples to oranges.
#   * The only tracking changes for ROM tracking are:
#       INPUT           -> the ROM_recon case folder (has one PVTU at ts=0)
#       MESH_SUBDIR     -> "post"     (recon layout uses post/ directly)
#       MESH_PATTERN    -> ${CASE_PREFIX}_{timestep}.pvtu
#       VEL_START/END   -> 0 0        (only one recon snapshot exists)
#       OUTPUT_CASE_SUBFOLDER -> post_pt_rom_<formula>  (sit next to post_pt)
#       RUN_TAG         -> ""         (folder name stability)
#       AUTO_DETECT_CASE-> 0          (INPUT is the recon folder, not this script's dir)
#   * Everything else — PIN_RPM, DT, N_STEPS, SEED_SOURCE, SEED_FRACTION,
#     SEED_GRID, LEVELSET_MODE, EXPORT_*, etc. — is COPIED unchanged from
#     the case's run_jaxtrace.sh.
#
# Usage:
#   bash scripts/generate_run_jaxtrace_recon.sh <case>.gid [<case>.gid ...]
#
# or a directory containing several cases:
#   bash scripts/generate_run_jaxtrace_recon.sh /scratch/shared/ROM/FOM/cylindrical_004.gid
#
# Environment overrides:
#   ROM_FORMULA    formula label     (default: centered)
#   ROM_RECON_ROOT parent of ROM_recon_<formula>/<case>.gid  (default: sibling of FOM/ named ROM_recon_<formula>)
#   OUTPUT_FILENAME output filename inside each case folder  (default: run_jaxtrace_recon.sh)
# =============================================================================

set -uo pipefail

ROM_FORMULA="${ROM_FORMULA:-centered}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-run_jaxtrace_recon.sh}"

if [ $# -lt 1 ]; then
    echo "usage: $0 <case>.gid [<case>.gid ...]" >&2
    exit 2
fi

_generate_for_case() {
    local CASE_DIR="$1"
    if [ ! -d "$CASE_DIR" ]; then
        echo "  [skip] not a directory: $CASE_DIR" >&2
        return 1
    fi
    local SRC="$CASE_DIR/run_jaxtrace.sh"
    if [ ! -f "$SRC" ]; then
        echo "  [skip] $SRC missing" >&2
        return 1
    fi

    # Extract INPUT to figure out FOM_ROOT (parent of the case folder)
    # and CASE_STEM.  Fall back to bash-derivation if INPUT is not set.
    local CASE_STEM
    CASE_STEM="$(basename "$CASE_DIR" .gid)"
    local FOM_ROOT
    FOM_ROOT="$(dirname "$CASE_DIR")"

    # Compute default ROM_RECON_ROOT if the user didn't set it.
    local ROM_RECON_ROOT_LOCAL
    ROM_RECON_ROOT_LOCAL="${ROM_RECON_ROOT:-$(dirname "$FOM_ROOT")/ROM_recon_${ROM_FORMULA}}"

    local RECON_INPUT="$ROM_RECON_ROOT_LOCAL/${CASE_STEM}.gid"
    local DST="$CASE_DIR/$OUTPUT_FILENAME"

    # We'll do the substitutions in a single-pass sed.  The transformations:
    #   INPUT="…"                          -> INPUT="<recon input>"
    #   MESH_SUBDIR="…"                    -> MESH_SUBDIR="post"
    #   MESH_PATTERN="…"                   -> keep as-is (already correct for this case)
    #   VEL_START=<digits>                  -> VEL_START=0
    #   VEL_END=<digits>                    -> VEL_END=0
    #   OUTPUT_CASE_SUBFOLDER=…            -> OUTPUT_CASE_SUBFOLDER=post_pt_rom_<formula>
    #   AUTO_DETECT_CASE=1                 -> AUTO_DETECT_CASE=0    (safer for a script that references the recon INPUT explicitly)
    #   RUN_TAG="…"                        -> RUN_TAG=""
    # Also drop any HIT_STATS_LOG that was already 0 and force =1 so we
    # get L0/L1/L2 stats on the ROM-tracking run too.
    #
    # We do NOT alter DT, N_STEPS, PIN_RPM, SEED_*, BOUNDARY_WALLS, LEVELSET_*,
    # PROJ tolerances, or anything else.

    python3 - "$SRC" "$DST" "$RECON_INPUT" "$ROM_FORMULA" "$CASE_STEM" <<'PY'
import re, sys
from pathlib import Path
src, dst, recon_input, rom_formula, case_stem = sys.argv[1:6]
txt = Path(src).read_text()

def sub_var(text, name, new_value_expr, is_string=False):
    """Replace `<name>=<something>[  # trailing comment]` with `<name>=<new_value_expr>`.
    Preserves the trailing comment if any (with at least one space before #),
    and keeps `<name>=` alignment intact."""
    if is_string:
        new = f'{name}="{new_value_expr}"'
    else:
        new = f'{name}={new_value_expr}'
    # Capture optional trailing whitespace + optional '# comment' separately
    pat = re.compile(
        r'^(\s*)' + re.escape(name) + r'=[^\n#]*?(\s*)(#.*)?$',
        re.MULTILINE,
    )
    def _repl(m):
        indent = m.group(1)
        gap = m.group(2) or ''
        comment = m.group(3) or ''
        if comment and not gap.strip():
            # ensure at least one space between value and comment
            gap = '  ' if not gap else gap
        return f'{indent}{new}{gap}{comment}'
    return pat.sub(_repl, text, count=1)

# Header comment
header_extra = (
    f"\n# =============================================================================\n"
    f"# AUTO-GENERATED from run_jaxtrace.sh by scripts/generate_run_jaxtrace_recon.sh\n"
    f"# Reconstruction formula: {rom_formula}\n"
    f"# Only the velocity-field paths have been swapped.  All other tracking\n"
    f"# knobs (dt, N_STEPS, PIN_RPM, seeding, boundary walls, level-set,\n"
    f"# temperature export) are inherited unchanged from run_jaxtrace.sh so a\n"
    f"# FOM-vs-ROM comparison of final particle positions varies ONE thing:\n"
    f"# the velocity field the RK4 integrator sees.\n"
    f"# =============================================================================\n"
)
# Insert the header block right after the shebang line
lines = txt.splitlines(True)
i = 1 if lines and lines[0].startswith("#!") else 0
lines.insert(i, header_extra)
txt = "".join(lines)

txt = sub_var(txt, 'INPUT',                    recon_input,                    is_string=True)
txt = sub_var(txt, 'AUTO_DETECT_CASE',         '0')
txt = sub_var(txt, 'MESH_SUBDIR',              'post',                         is_string=True)
# Do NOT touch MESH_PATTERN; the recon PVTU keeps the same case-prefix pattern.
txt = sub_var(txt, 'VEL_START',                '0')
txt = sub_var(txt, 'VEL_END',                  '0')
txt = sub_var(txt, 'OUTPUT_CASE_SUBFOLDER',    f'post_pt_rom_{rom_formula}',   is_string=True)
txt = sub_var(txt, 'RUN_TAG',                  '',                              is_string=True)
txt = sub_var(txt, 'HIT_STATS_LOG',            '1')
# The union hook should NOT auto-run against ROM-recon outputs (different
# folder layout expected).  Force it off.
txt = sub_var(txt, 'ENABLE_UNION',             '0')

Path(dst).write_text(txt)
PY

    if [ ! -f "$DST" ]; then
        echo "  [fail] generator produced no output for $CASE_DIR" >&2
        return 2
    fi

    chmod +x "$DST"
    echo "  [OK] $DST"
    return 0
}

for CASE_ARG in "$@"; do
    _generate_for_case "$CASE_ARG"
done
