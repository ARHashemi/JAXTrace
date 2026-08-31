#!/usr/bin/env bash
#
# Port RTXAdvect — Phase 12: drop the leftover closing '}' at end-of-file.
#
# Phase 11's parser correctly closed the 'namespace advect' block early
# (before main()) but its `after = src[ns_close+1:]` slice left the
# ORIGINAL end-of-namespace '}' intact, which now dangles at global scope.
# Result: cudaParticleAdvection.cu(501): error: expected a declaration.
#
# Delete the trailing dangling '}' + ensure a final newline.

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
RTX_ROOT="$REPO_ROOT/3rdParty/RTXAdvect"
ADV="$RTX_ROOT/cuda/cudaParticleAdvection.cu"

echo "== Fix 11: strip dangling '}' at end of $ADV"
python3 - "$ADV" <<'PY'
import sys, re
p = sys.argv[1]
src = open(p).read()
# Strip trailing whitespace
stripped = src.rstrip()
# If the file ends with a lone '}' (which is the original namespace close),
# remove it.  We also require that the file already contains the marker
# comment we inserted so we don't confuse this with a legitimate close.
if "} // namespace advect (moved main() outside)" not in stripped:
    print("NOTE: marker comment not found — is main() still outside namespace?")
if stripped.endswith("}"):
    stripped = stripped[:-1].rstrip()
    print("OK: dropped trailing '}' (namespace close leftover)")
else:
    print("NOTE: no trailing '}' to strip")
# Ensure a single trailing newline
open(p, 'w').write(stripped + "\n")
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
make -j"$(nproc)" 2>&1 | tee /tmp/rtxadvect_make.log | tail -25
ec=${PIPESTATUS[0]}

echo
if [[ $ec -eq 0 ]]; then
  echo "=== PORT STEP 12 SUCCESS"
  EXE=$(find "$BUILD_DIR" -maxdepth 3 -type f -executable -name "cudaParticleAdvection" 2>/dev/null | head -1)
  echo "  executable: $EXE"
  echo
  echo "=== TRYING --help TO CONFIRM"
  "$EXE" --help 2>&1 | head -30 || "$EXE" 2>&1 | head -20 || true
else
  echo "=== PORT STEP 12 FAILED (ec=$ec)"
  echo "First 30 error lines:"
  grep -m 30 -E "error:|undefined|fatal" /tmp/rtxadvect_make.log | head -30
fi
