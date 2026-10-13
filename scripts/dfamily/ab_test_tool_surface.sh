#!/bin/bash
# A/B test the candidate remedies for the tool-surface losses.
#
# Run AFTER diagnose_tool_surface_loss.py has identified the mechanism. Each
# arm changes exactly ONE knob from the baseline, so the loss delta is
# attributable. All knobs already exist on run_tracking.py -- no code change
# is needed to test any of them.
#
#   baseline   reproduce the production run's settings
#   tol1e4     --point-in-tet-tol 1e-4    T4 candidate: predicate too strict
#   tol1e3     --point-in-tet-tol 1e-3    "  (how far must it go)
#   band       --enhanced-search-band ... widen the search only near the tool
#   l2_5       --l2-neighborhood 5        5x5x5 instead of 3x3x3
#   skipstep   --levelset-mode skip_step  freeze on tool contact instead of
#                                         integrating a zeroed substage
#
# Short runs with seeding concentrated in the measured loss band
# (r 5-9 mm, z -1..+1.4 mm) so each arm costs minutes, not hours. The mesh is
# static (verified), so one velocity snapshot is enough to expose a
# point-location failure -- but losses accumulate over steps, so we use a few
# hundred.
#
# Usage: ab_test_tool_surface.sh [arm ...]      (default: all arms)
# Read-only w.r.t. the case folder; all output under /scratch/users/$USER.
set -uo pipefail

JAXTRACE=/flash/shared/jax/JAXTrace
CASE=/scratch/shared/PinShapes/D-ConcavityTilt/D2.gid
PY=/flash/shared/jax/.venv/bin/python
OUT=/scratch/users/${USER}/d2_ab
mkdir -p "$OUT"

# Short but long enough for losses to accumulate: the production run saw its
# first loss at step 900, but that was with seeds spread over the whole box.
# Seeding directly in the loss band should fail far sooner.
N_STEPS=${N_STEPS:-600}
DT=0.0003629
VEL_START=34
VEL_END=199
NPART=${NPART:-60000}

# Seed box covering the measured loss band, in metres:
#   radius 5-9 mm  -> |x|,|y| <= 9mm ; z -1 .. +1.4 mm
# A box is the available primitive; the annulus is cut out in post.
SEED_BOX="-0.009 0.009 -0.009 0.009 -0.001 0.0014"

run_arm () {
  local name="$1"; shift
  local log="$OUT/${name}.log"
  echo "=== arm $name -> $log"
  {
    echo "=== $name started $(date -Is) ==="
    echo "=== extra args: $* ==="
    "$PY" -u "$JAXTRACE/run_tracking.py" \
      --input "$CASE/post" \
      --mesh-subdir "" \
      --output "$OUT/$name" \
      --n-steps "$N_STEPS" --dt "$DT" \
      --vel-range "$VEL_START" "$VEL_END" \
      --velocity-field Displacement \
      --seed-source box --seed-box $SEED_BOX --n-particles "$NPART" \
      --failed-substage last_valid_vel \
      --export-element-ids \
      --no-export \
      "$@"
    echo "=== EXIT $? at $(date -Is) ==="
  } >> "$log" 2>&1
  grep -E "Final lost|initial assignment|Particles:" "$log" | tail -3
}

ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(baseline tol1e4 tol1e3 band l2_5 skipstep)

for a in "${ARMS[@]}"; do
  case "$a" in
    baseline) run_arm baseline ;;
    tol1e4)   run_arm tol1e4   --point-in-tet-tol 1e-4 ;;
    tol1e3)   run_arm tol1e3   --point-in-tet-tol 1e-3 ;;
    band)     run_arm band     --enhanced-search-band 0.0005 ;;
    l2_5)     run_arm l2_5     --l2-neighborhood 5 ;;
    skipstep) run_arm skipstep --levelset-mode skip_step ;;
    *) echo "unknown arm: $a" >&2 ;;
  esac
done

echo
echo "=== summary: final loss per arm ==="
for f in "$OUT"/*.log; do
  printf "  %-10s %s\n" "$(basename "$f" .log)" \
    "$(grep -oE 'Final lost:[[:space:]]+[0-9,]+ \([0-9.]+%\)' "$f" | tail -1)"
done
