#!/usr/bin/env bash
#
# Port RTXAdvect — Phase 5: two small portability fixes to RTXAdvect's own
# helpers that were Windows-specific and only tolerated by older toolchains.
#
#   1) cudaHelpers.cuh includes <windows.h> unconditionally.  Guard it
#      behind _WIN32 so Linux nvcc + gcc 15 don't choke.
#   2) HostTetMesh.h:88 uses the 2-arg form assert(cond, "msg") — that
#      form only works with Microsoft's _ASSERTE macro.  Convert to the
#      standard C assert idiom:  assert(cond && "msg").

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
RTX_ROOT="$REPO_ROOT/3rdParty/RTXAdvect"

HELPERS="$RTX_ROOT/cuda/cudaHelpers.cuh"
HOSTMESH="$RTX_ROOT/cuda/HostTetMesh.h"

echo "== Fix 1: guard <windows.h> in $HELPERS"
if [[ -f "$HELPERS" && ! -f "$HELPERS.orig-2020" ]]; then
  cp -v "$HELPERS" "$HELPERS.orig-2020"
fi
# Replace the naked #include <windows.h> with a guarded one.
python3 - "$HELPERS" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p).read()
new = re.sub(
    r'^\s*#\s*include\s*<windows\.h>\s*$',
    '#if defined(_WIN32) || defined(_WIN64)\n#include <windows.h>\n#endif',
    src, count=1, flags=re.M,
)
if new == src:
    print(f"NOTE: no naked #include <windows.h> found in {p} (already guarded?)")
else:
    open(p, 'w').write(new)
    print(f"OK: patched {p}")
PY

echo
echo "== Fix 2: convert 2-arg assert() to 'cond && \"msg\"' in $HOSTMESH"
if [[ -f "$HOSTMESH" && ! -f "$HOSTMESH.orig-2020" ]]; then
  cp -v "$HOSTMESH" "$HOSTMESH.orig-2020"
fi
python3 - "$HOSTMESH" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p).read()
# Match assert(<cond>, "<message>");   -> assert((<cond>) && "<message>");
# Handle multi-line messages carefully but conservatively (single-line only).
pat = re.compile(
    r'assert\s*\(\s*'                      # assert(
    r'([^,";]+(?:\([^)]*\)[^,";]*)*)'      # 1: cond (may contain balanced parens)
    r'\s*,\s*'                              # ,
    r'("[^"]*")\s*'                        # 2: "message"
    r'\)\s*;',                              # );
    re.M
)
def _sub(m):
    cond = m.group(1).strip()
    msg  = m.group(2)
    return f'assert(({cond}) && {msg});'
new, n = pat.subn(_sub, src)
if n == 0:
    print(f"NOTE: no 2-arg assert() found in {p} (already patched?)")
else:
    open(p, 'w').write(new)
    print(f"OK: patched {n} assert() call(s) in {p}")
PY

# Rebuild
: "${CUDA_HOME:=/usr/local/cuda-13.3}"
: "${OPTIX_ROOT:=/flash/shared/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64}"
: "${CUDA_ARCHS:=120}"
export PATH="$CUDA_HOME/bin:$PATH"

BUILD_DIR="$RTX_ROOT/build"
cd "$BUILD_DIR"

echo
echo "== make -j$(nproc)"
make -j"$(nproc)" 2>&1 | tee /tmp/rtxadvect_make.log | tail -100
ec=${PIPESTATUS[0]}

echo
if [[ $ec -eq 0 ]]; then
  echo "=== PORT STEP 5 SUCCESS"
  find "$BUILD_DIR" -maxdepth 3 -type f -executable 2>/dev/null | head -10
else
  echo "=== PORT STEP 5 FAILED (ec=$ec)"
  echo "First 30 error lines:"
  grep -m 30 -E "error:|undefined|fatal" /tmp/rtxadvect_make.log | head -30
  echo
  echo "Full make log at /tmp/rtxadvect_make.log"
fi
