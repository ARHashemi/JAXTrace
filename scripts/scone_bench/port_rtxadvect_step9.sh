#!/usr/bin/env bash
#
# Port RTXAdvect — Phase 9: three tiny mechanical fixes.
#
# Fix 6: thrust::min / thrust::max were removed in Thrust 2.  Since these
#        are called inside __host__ __device__ operator()s, std::min /
#        std::max work (both have __host__ __device__ overloads in
#        CUDA 12+).  Just rename thrust:: → std:: for min/max in this file.
#
# Fix 7: Undo my Phase 3 over-correction.  `%.2f` was correct.  My earlier
#        sed changed `\%` (a bad escape) to `%%` (a literal %), but the
#        right fix was `\%` → `%` — the intent was a format specifier.
#        In `printf("%%.2f", x)` the %% is a literal % and .2f is text,
#        so `x` is unused and the second column of every stats row prints
#        as ".2f" literal.  Change `\t%%.2f\n` → `\t%.2f\n`.
#        (These are five printf lines; guarded by the exact substring.)
#
# Fix 8: C++17 forbids "explicit qualification" of a function definition
#        inside its own namespace.  Three sites:
#          ConvexQuery.cu:438  void advect::convexWallReflect(...)
#          ConvexQuery.cu:498  void advect::testNStracing(...)
#          RTQuery.cu:418      void advect::testRT(...)
#        Fix: strip the `advect::` prefix (they're inside `namespace advect
#        { ... }` already).

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
RTX_ROOT="$REPO_ROOT/3rdParty/RTXAdvect"

PARTICLES="$RTX_ROOT/cuda/particles.cu"
ADV="$RTX_ROOT/cuda/cudaParticleAdvection.cu"
CONVEX="$RTX_ROOT/query/ConvexQuery.cu"
RTQ="$RTX_ROOT/query/RTQuery.cu"

# ------------------------------------------------------------------
# Fix 6: thrust::min|max -> std::min|max in particles.cu
# ------------------------------------------------------------------
echo "== Fix 6: thrust::min|max -> std::min|max in $PARTICLES"
sed -i \
    -e 's/\bthrust::min\b/std::min/g' \
    -e 's/\bthrust::max\b/std::max/g' \
    "$PARTICLES"

# Ensure <algorithm> is included for std::min/max on the host side
if ! grep -q "#include <algorithm>" "$PARTICLES"; then
  # Insert at top-of-file after last existing #include
  python3 - "$PARTICLES" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p).read()
inc_positions = [m.end() for m in re.finditer(r'^#\s*include\s+[<"][^>"]+[>"]\s*$', src, flags=re.M)]
if inc_positions:
    pos = inc_positions[-1]
    src = src[:pos] + "\n#include <algorithm>" + src[pos:]
    open(p, 'w').write(src)
    print("added #include <algorithm>")
PY
fi

# ------------------------------------------------------------------
# Fix 7: fix the printf format strings — %%.2f -> %.2f (5 lines)
# ------------------------------------------------------------------
echo
echo "== Fix 7: %%.2f -> %.2f in $ADV"
sed -i 's/\\t%%.2f\\n/\\t%.2f\\n/g' "$ADV"

# ------------------------------------------------------------------
# Fix 8: strip explicit `advect::` from function definitions inside
#        the advect namespace.  Only touches the three sites the log
#        called out.
# ------------------------------------------------------------------
echo
echo "== Fix 8: strip explicit advect:: from function defs in ConvexQuery.cu + RTQuery.cu"
# Use very targeted sed patterns keyed on the function name so we don't
# touch call sites (which legitimately use advect::)
sed -i 's/^\([[:space:]]*\)void advect::convexWallReflect/\1void convexWallReflect/' "$CONVEX"
sed -i 's/^\([[:space:]]*\)void advect::testNStracing/\1void testNStracing/'         "$CONVEX"
sed -i 's/^\([[:space:]]*\)void advect::testRT/\1void testRT/'                       "$RTQ"

# Show what changed
echo "-- confirm ConvexQuery.cu:"
grep -n "^[[:space:]]*void.*convexWallReflect\|^[[:space:]]*void.*testNStracing" "$CONVEX" | head
echo "-- confirm RTQuery.cu:"
grep -n "^[[:space:]]*void.*testRT" "$RTQ" | head

# ------------------------------------------------------------------
# Rebuild
# ------------------------------------------------------------------
: "${CUDA_HOME:=/usr/local/cuda-13.3}"
: "${OPTIX_ROOT:=/flash/shared/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64}"
: "${CUDA_ARCHS:=120}"
export PATH="$CUDA_HOME/bin:$PATH"

BUILD_DIR="$RTX_ROOT/build"
cd "$BUILD_DIR"

echo
echo "== make -j$(nproc)"
make -j"$(nproc)" 2>&1 | tee /tmp/rtxadvect_make.log | tail -60
ec=${PIPESTATUS[0]}

echo
if [[ $ec -eq 0 ]]; then
  echo "=== PORT STEP 9 SUCCESS"
  find "$BUILD_DIR" -maxdepth 3 -type f -executable 2>/dev/null | head -10
else
  echo "=== PORT STEP 9 FAILED (ec=$ec)"
  echo "First 30 error lines:"
  grep -m 30 -E "error:|undefined|fatal" /tmp/rtxadvect_make.log | head -30
fi
