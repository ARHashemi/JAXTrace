#!/usr/bin/env bash
#
# Port RTXAdvect — Phase 10: generalise the explicit-qualification fix.
#
# 9 more sites in particles.cu (and possibly others we haven't seen yet).
# Instead of listing each function by name, we globally strip 'advect::'
# from any function-definition-start line inside the RTXAdvect .cu/.cpp
# files.  Two rules protect us from over-rewriting:
#   1) We only touch lines whose FIRST non-whitespace token is a C++
#      return type followed by `advect::<identifier>(` — i.e. function
#      definitions.  Call sites like `advect::foo(x, y)` (not at column
#      0-ish) are untouched.
#   2) We only touch files inside 3rdParty/RTXAdvect/ (never OWL).
#
# Also add --expt-relaxed-constexpr to nvcc flags for the RTXAdvect
# targets — std::min/std::max are constexpr __host__ functions in
# libstdc++, and calling them from a __host__ __device__ operator
# triggers a warning that could bite us if -Werror is enabled.

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
RTX_ROOT="$REPO_ROOT/3rdParty/RTXAdvect"

# Files that live in the advect namespace (all .cu/.cpp under RTXAdvect
# except OWL, and except the top-level main-file cudaParticleAdvection.cu
# which doesn't put main() in the namespace).
SOURCES=(
  "$RTX_ROOT/cuda/particles.cu"
  "$RTX_ROOT/cuda/utils.cpp"
  "$RTX_ROOT/query/ConvexQuery.cu"
  "$RTX_ROOT/query/RTQuery.cu"
  "$RTX_ROOT/optix/OptixTetQuery.cpp"
  "$RTX_ROOT/optix/OptixTriQuery.cpp"
)

echo "== Fix 8 (generalised): strip 'advect::' from every function definition"
python3 - "${SOURCES[@]}" <<'PY'
import re, sys
# Match a line whose leading whitespace + one C++ return-type keyword
# (int/void/double/float/bool/auto/T) OR a template-ended '>' pattern
# is followed by 'advect::identifier('
# We just look for the very common pattern:  ^<indent><type> advect::<name>(
pat = re.compile(
    r'^([ \t]*'                                # 1: leading whitespace
    r'[A-Za-z_][A-Za-z_0-9:<>*&\s,]*?'         # C++ return type (may have templates, refs, ptrs)
    r'[ \t]+)'                                  # end of return type
    r'advect::'                                 # to remove
    r'([A-Za-z_][A-Za-z_0-9]*\s*\()',          # 2: function name and open-paren
    re.M,
)
total = 0
for p in sys.argv[1:]:
    src = open(p).read()
    new, n = pat.subn(r'\1\2', src)
    if n:
        open(p, 'w').write(new)
        total += n
        print(f"  {p}: stripped {n} 'advect::' prefix(es)")
print(f"total: {total} stripped")
PY

# ------------------------------------------------------------------
# Also add --expt-relaxed-constexpr and silence -Wformat= (used by
# the printf(%d, long int) noise) for the RTXAdvect targets
# ------------------------------------------------------------------
CMAKE_FILE="$RTX_ROOT/CMakeLists.txt"
if ! grep -q "expt-relaxed-constexpr" "$CMAKE_FILE"; then
  echo
  echo "== add --expt-relaxed-constexpr to nvcc flags"
  cat >> "$CMAKE_FILE" <<'CMAKE_EOF'

# nvcc 13 promotes some warnings; relax constexpr host->device calls
# (std::min/std::max on the device side) and silence printf format
# checks (there are a few benign %d vs long int mismatches in the
# original code).
target_compile_options(optixTetQueries       PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-format>)
target_compile_options(cudaParticleAdvection PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-format>)
CMAKE_EOF
fi

# ------------------------------------------------------------------
# Rebuild
# ------------------------------------------------------------------
: "${CUDA_HOME:=/usr/local/cuda-13.3}"
: "${OPTIX_ROOT:=/flash/shared/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64}"
: "${CUDA_ARCHS:=120}"
export PATH="$CUDA_HOME/bin:$PATH"

BUILD_DIR="$RTX_ROOT/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo
echo "== cmake configure"
cmake .. \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DOptiX_INSTALL_DIR="$OPTIX_ROOT" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DCMAKE_BUILD_TYPE=Release \
    -Wno-dev \
  2>&1 | tail -30

echo
echo "== make -j$(nproc)"
make -j"$(nproc)" 2>&1 | tee /tmp/rtxadvect_make.log | tail -40
ec=${PIPESTATUS[0]}

echo
if [[ $ec -eq 0 ]]; then
  echo "=== PORT STEP 10 SUCCESS"
  find "$BUILD_DIR" -maxdepth 3 -type f -executable 2>/dev/null | head -10
else
  echo "=== PORT STEP 10 FAILED (ec=$ec)"
  echo "First 30 error lines:"
  grep -m 30 -E "error:|undefined|fatal" /tmp/rtxadvect_make.log | head -30
fi
