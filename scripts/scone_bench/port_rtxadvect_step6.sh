#!/usr/bin/env bash
#
# Port RTXAdvect — Phase 6: three portability / modern-Thrust fixes.
#
#   1) cudaHelpers.cuh has a Windows-only CPUTimer class using
#      LARGE_INTEGER + QueryPerformanceCounter.  Give it a std::chrono
#      Linux/portable implementation.  Semantics preserved.
#
#   2) particles.cu inherits from thrust::unary_function / binary_function,
#      empty tag classes deprecated in Thrust 1.9 and REMOVED in Thrust 2.0
#      (which ships with CUDA 12+).  They're purely for STL dispatch and
#      have no runtime effect.  Delete the inheritance.
#
#   3) cudaParticleAdvection.cu uses \% in printf strings — not a valid
#      escape, warning promoted to error by nvcc 13.  Change \% → %%.

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
RTX_ROOT="$REPO_ROOT/3rdParty/RTXAdvect"

HELPERS="$RTX_ROOT/cuda/cudaHelpers.cuh"
PARTICLES="$RTX_ROOT/cuda/particles.cu"
ADV="$RTX_ROOT/cuda/cudaParticleAdvection.cu"

# Ensure backups exist for all three (some already backed up in earlier steps)
for f in "$HELPERS" "$PARTICLES" "$ADV"; do
  if [[ -f "$f" && ! -f "$f.orig-2020" ]]; then
    cp -v "$f" "$f.orig-2020"
  fi
done

# ------------------------------------------------------------------
# Fix 1 — replace the Win32 CPUTimer with a std::chrono equivalent
# ------------------------------------------------------------------
echo "== Fix 1: replace Win32 CPUTimer with std::chrono in $HELPERS"
python3 - "$HELPERS" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p).read()

# Find the class definition:  class CPUTimer { ... };
# Match from "class CPUTimer" to the matching closing brace + semicolon.
m = re.search(r'class\s+CPUTimer\s*\{.*?\};\s*\n', src, flags=re.S)
if not m:
    print("NOTE: CPUTimer class not found (already patched?)")
    sys.exit(0)

replacement = """class CPUTimer
{
private:
    std::chrono::high_resolution_clock::time_point tStart;
    std::chrono::high_resolution_clock::time_point tEnd;

public:
    CPUTimer(void) {}

    void start(void)
    {
        tStart = std::chrono::high_resolution_clock::now();
    }

    double stop(void)
    {
        tEnd = std::chrono::high_resolution_clock::now();
        return this->TimeInSeconds() * 1000.0;
    }

    long TimeInTicks(void)
    {
        return static_cast<long>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(tEnd - tStart).count());
    }

    double TimeInSeconds(void)
    {
        return std::chrono::duration<double>(tEnd - tStart).count();
    }
};
"""

# Also add #include <chrono> if not already present.
if "#include <chrono>" not in src:
    # Insert it near the top, after the last existing #include line
    inc_positions = [m.end() for m in re.finditer(r'^#\s*include\s+[<"][^>"]+[>"]\s*$', src, flags=re.M)]
    if inc_positions:
        pos = inc_positions[-1]
        src = src[:pos] + "\n#include <chrono>" + src[pos:]

src = src[:m.start()] + replacement + src[m.end():]
open(p, 'w').write(src)
print(f"OK: patched CPUTimer in {p}")
PY

# ------------------------------------------------------------------
# Fix 2 — drop thrust::unary_function / binary_function inheritance
# ------------------------------------------------------------------
echo
echo "== Fix 2: drop thrust::(unary|binary)_function inheritance in $PARTICLES"
python3 - "$PARTICLES" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p).read()

# Delete the ": public thrust::unary_function< ... >" line
new = re.sub(
    r'\n\s*:\s*public\s+thrust::unary_function\s*<[^>]+>\s*',
    '\n',
    src,
)
new = re.sub(
    r'\n\s*:\s*public\s+thrust::binary_function\s*<[^>]+>\s*',
    '\n',
    new,
)
if new == src:
    print("NOTE: no thrust::unary/binary_function inheritance found (already patched?)")
else:
    open(p, 'w').write(new)
    print(f"OK: patched thrust functor inheritance in {p}")
PY

# ------------------------------------------------------------------
# Fix 3 — replace \% with %% in printf strings
# ------------------------------------------------------------------
echo
echo "== Fix 3: replace \\% with %% in $ADV"
python3 - "$ADV" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p).read()
# Only touch the specific pattern "\%" that appears inside a printf format
# string.  Use a simple string replace — \% is not a valid C escape anywhere
# else so this is safe.
new = src.replace(r'\%', '%%')
if new == src:
    print("NOTE: no bad-escape found (already patched?)")
else:
    open(p, 'w').write(new)
    print(f"OK: patched printf escape in {p}")
PY

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
make -j"$(nproc)" 2>&1 | tee /tmp/rtxadvect_make.log | tail -100
ec=${PIPESTATUS[0]}

echo
if [[ $ec -eq 0 ]]; then
  echo "=== PORT STEP 6 SUCCESS"
  find "$BUILD_DIR" -maxdepth 3 -type f -executable 2>/dev/null | head -10
else
  echo "=== PORT STEP 6 FAILED (ec=$ec)"
  echo "First 30 error lines:"
  grep -m 30 -E "error:|undefined|fatal" /tmp/rtxadvect_make.log | head -30
  echo
  echo "Full make log at /tmp/rtxadvect_make.log"
fi
