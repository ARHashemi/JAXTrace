#!/usr/bin/env bash
#
# Port RTXAdvect — Phase 8: reapply the <windows.h> guard that Phase 7
# accidentally undid by restoring from .orig-2020.

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
RTX_ROOT="$REPO_ROOT/3rdParty/RTXAdvect"
HELPERS="$RTX_ROOT/cuda/cudaHelpers.cuh"

echo "== reapply <windows.h> guard in $HELPERS"
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

# Rebuild
: "${CUDA_HOME:=/usr/local/cuda-13.3}"
: "${OPTIX_ROOT:=/flash/shared/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64}"
: "${CUDA_ARCHS:=120}"
export PATH="$CUDA_HOME/bin:$PATH"

BUILD_DIR="$RTX_ROOT/build"
cd "$BUILD_DIR"

echo
echo "== make -j$(nproc)"
make -j"$(nproc)" 2>&1 | tee /tmp/rtxadvect_make.log | tail -80
ec=${PIPESTATUS[0]}

echo
if [[ $ec -eq 0 ]]; then
  echo "=== PORT STEP 8 SUCCESS"
  find "$BUILD_DIR" -maxdepth 3 -type f -executable 2>/dev/null | head -10
else
  echo "=== PORT STEP 8 FAILED (ec=$ec)"
  echo "First 30 error lines:"
  grep -m 30 -E "error:|undefined|fatal" /tmp/rtxadvect_make.log | head -30
fi
