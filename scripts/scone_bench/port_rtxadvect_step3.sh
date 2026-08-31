#!/usr/bin/env bash
#
# Port RTXAdvect — Phase 3: patch RTXAdvect's own C++ source to use the new
# NVIDIA/OWL API symbol names.  OWL dropped the "LaunchParams" prefix from
# most user-facing setters and renamed the 2D launcher:
#
#   owlLaunchParamsCreate   -> owlParamsCreate
#   owlLaunchParamsSet1i    -> owlParamsSet1i
#   owlLaunchParamsSet1ul   -> owlParamsSet1ul
#   owlLaunchParamsSet3f    -> owlParamsSet3f       (in case it's used)
#   owlLaunchParamsSetBuffer -> owlParamsSetBuffer  (in case it's used)
#   owlParamsLaunch2D       -> owlRayGenLaunch2D
#
# Also silence the CUDA 13 `double4` deprecation warnings by disabling that
# specific warning — the type still works; it's a maintenance renaming for
# alignment-explicit variants (double4_16a).  Fixing that properly would
# require redeclaring every double4-typed function in the API, which is
# scope creep — we just silence the warning.
#
# Only touches files under 3rdParty/RTXAdvect/optix/ and cuda/ (RTXAdvect's
# own source, NOT the OWL submodule).

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
RTX_ROOT="$REPO_ROOT/3rdParty/RTXAdvect"

if [[ ! -d "$RTX_ROOT/optix" ]]; then
  echo "ERROR: $RTX_ROOT/optix not found — did Phase 1 clone succeed?"
  exit 2
fi

FILES=(
  "$RTX_ROOT/optix/OptixTetQuery.cpp"
  "$RTX_ROOT/optix/OptixTriQuery.cpp"
  "$RTX_ROOT/optix/OptixQuery.h"
  "$RTX_ROOT/optix/optixQueryKernel.cu"
  "$RTX_ROOT/cuda/cudaParticleAdvection.cu"
  "$RTX_ROOT/query/ConvexQuery.cu"
  "$RTX_ROOT/query/RTQuery.cu"
)

echo "== patching OWL API symbol renames"

for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || continue
  # Back up once (skip if backup exists)
  if [[ ! -f "$f.orig-2020" ]]; then
    cp -v "$f" "$f.orig-2020"
  fi

  # OWL API renames.  Order matters: replace the most specific first
  # so `owlLaunchParamsSet1i` doesn't get half-replaced by an earlier
  # `owlLaunchParams` rewrite.
  sed -i \
      -e 's/\bowlLaunchParamsCreate\b/owlParamsCreate/g' \
      -e 's/\bowlLaunchParamsSet1i\b/owlParamsSet1i/g' \
      -e 's/\bowlLaunchParamsSet1ul\b/owlParamsSet1ul/g' \
      -e 's/\bowlLaunchParamsSet1f\b/owlParamsSet1f/g' \
      -e 's/\bowlLaunchParamsSet3f\b/owlParamsSet3f/g' \
      -e 's/\bowlLaunchParamsSet4f\b/owlParamsSet4f/g' \
      -e 's/\bowlLaunchParamsSetBuffer\b/owlParamsSetBuffer/g' \
      -e 's/\bowlLaunchParamsSetGroup\b/owlParamsSetGroup/g' \
      -e 's/\bowlLaunchParamsSetTexture\b/owlParamsSetTexture/g' \
      -e 's/\bowlLaunchParamsSetPointer\b/owlParamsSetPointer/g' \
      -e 's/\bowlLaunchParamsSetRaw\b/owlParamsSetRaw/g' \
      -e 's/\bowlParamsLaunch2D\b/owlLaunch2D/g' \
      "$f"
done

# Add the CUDA-13 deprecation-warning silence to the target so we don't
# have to touch every source file.  Append a target_compile_options at
# the end of the top-level CMakeLists.
if ! grep -q "double4-deprecated shim" "$RTX_ROOT/CMakeLists.txt"; then
  echo "== adding double4 deprecation-warning silencer to CMakeLists.txt"
  cat >> "$RTX_ROOT/CMakeLists.txt" <<'CMAKE_EOF'

# double4-deprecated shim — CUDA 13 renamed double4 to double4_16a /
# double4_32a but kept the original as a deprecated alias.  Silence the
# warning here rather than patching every function signature that
# transitively takes double4.
target_compile_options(optixTetQueries      PRIVATE -Wno-deprecated-declarations)
target_compile_options(cudaParticleAdvection PRIVATE -Wno-deprecated-declarations)
CMAKE_EOF
fi

echo
echo "== rebuild"
: "${CUDA_HOME:=/usr/local/cuda-13.3}"
: "${OPTIX_ROOT:=/flash/shared/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64}"
: "${CUDA_ARCHS:=120}"
export PATH="$CUDA_HOME/bin:$PATH"

BUILD_DIR="$RTX_ROOT/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DOptiX_INSTALL_DIR="$OPTIX_ROOT" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DCMAKE_BUILD_TYPE=Release \
    -Wno-dev \
  2>&1 | tail -40

echo
echo "== make -j$(nproc)"
make -j"$(nproc)" 2>&1 | tee /tmp/rtxadvect_make.log | tail -120
ec=${PIPESTATUS[0]}

echo
if [[ $ec -eq 0 ]]; then
  echo "=== PORT STEP 3 SUCCESS"
  find "$BUILD_DIR" -maxdepth 3 -type f -executable 2>/dev/null | head -10
else
  echo "=== PORT STEP 3 FAILED (ec=$ec)"
  echo "First 30 error lines:"
  grep -m 30 -E "error:|undefined|fatal" /tmp/rtxadvect_make.log | head -30
  echo
  echo "Full make log at /tmp/rtxadvect_make.log"
fi
