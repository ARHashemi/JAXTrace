#!/usr/bin/env bash
#
# Submit the pitch-fix verification for the remaining D-family cases, once D2
# has confirmed the fix works.
#
# Each case keeps its OWN MESH_PATTERN (D1=A1_, D2=A2_, D2old=A2_, D3=B4_,
# D4=C2_ — the prefix is NOT the case name), its own seeding, dt and pin
# kinematics. verify_pitch_fix.sh never touches those; it only pins the repo to
# the patched dev tree, redirects output to the user's own scratch, and turns
# on ElementID export plus the hit-level probe.
#
# Jobs are staggered so they do not all hammer Lustre for the same mesh files
# at once: concurrent snapshot loading roughly halved the per-job rate in the
# earlier sweep (85 min instead of 42 for the same 30 snapshots).
#
# Usage
#   scripts/dfamily/submit_dfamily_pitchfix.sh            # D1 D3 D4 D2old
#   scripts/dfamily/submit_dfamily_pitchfix.sh D1 D3      # a subset
#   scripts/dfamily/submit_dfamily_pitchfix.sh --dry-run
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

DRY=0
CASES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    -*) echo "ERROR: unknown option '$1'" >&2; exit 2 ;;
    *) CASES+=("$1"); shift ;;
  esac
done
# D2 is the one already verified, so it is not re-run by default.
[ ${#CASES[@]} -eq 0 ] && CASES=(D1 D3 D4 D2old)

echo "Submitting pitch-fix verification for: ${CASES[*]}"
echo

STAGGER=0
for c in "${CASES[@]}"; do
  if [ "$DRY" = "1" ]; then
    echo "--- would submit $c (after ${STAGGER}s stagger)"
    "$HERE/verify_pitch_fix.sh" "$c" --family D --no-submit >/dev/null 2>&1 \
      && echo "    generated OK" || echo "    GENERATION FAILED" >&2
  else
    [ "$STAGGER" -gt 0 ] && sleep "$STAGGER"
    echo "--- $c"
    "$HERE/verify_pitch_fix.sh" "$c" --family D 2>&1 | grep -E "Submitted|ERROR|WARNING" || true
  fi
  STAGGER=600      # 10 min between launches
done

echo
if [ "$DRY" = "1" ]; then
  echo "Dry run — nothing submitted."
else
  echo "Watch:   squeue -u ${USER:-hashemia}"
  echo "Analyse: scripts/dfamily/verify_pitch_fix.sh <CASE> --analyze"
fi
