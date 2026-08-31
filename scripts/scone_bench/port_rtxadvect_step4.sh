#!/usr/bin/env bash
#
# Port RTXAdvect — Phase 4: fix the earlier over-rewrite of owlParamsLaunch2D.
#
# In step 3 we followed GCC's "did you mean owlRayGenLaunch2D" hint, but that
# hint was based on string similarity, not semantics.  The correct rename in
# modern OWL is:
#
#   OLD: owlParamsLaunch2D(rayGen, w, h, launchParams)   -- 4 args
#   NEW: owlLaunch2D      (rayGen, w, h, launchParams)   -- SAME 4 args, just
#                                                          drops "Params" prefix
#
# `owlRayGenLaunch2D(rayGen, w, h)` is a DIFFERENT function that takes no
# launchParams (used when there are no bound params).  RTXAdvect uses
# launchParams so we need owlLaunch2D.

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
RTX_ROOT="$REPO_ROOT/3rdParty/RTXAdvect"

FILES=(
  "$RTX_ROOT/optix/OptixTetQuery.cpp"
  "$RTX_ROOT/optix/OptixTriQuery.cpp"
  "$RTX_ROOT/optix/OptixQuery.h"
  "$RTX_ROOT/optix/optixQueryKernel.cu"
  "$RTX_ROOT/cuda/cudaParticleAdvection.cu"
  "$RTX_ROOT/query/ConvexQuery.cu"
  "$RTX_ROOT/query/RTQuery.cu"
)

echo "== undoing the owlRayGenLaunch2D over-rewrite from step 3"
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || continue
  # Only touch call sites that pass 4 args (rayGen, w, h, launchParams)
  # — a 4-arg call to owlRayGenLaunch2D is guaranteed wrong; a 3-arg call
  # would already be a legitimate use.  We rely on the fact that the
  # RTXAdvect code always uses launchParams, so every RayGenLaunch2D on
  # 4 args is the one we mis-rewrote.  Simplest: rewrite unconditionally
  # to owlLaunch2D — RTXAdvect uses owlLaunch2D everywhere.
  sed -i 's/\bowlRayGenLaunch2D\b/owlLaunch2D/g' "$f"
done

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
  echo "=== PORT STEP 4 SUCCESS"
  find "$BUILD_DIR" -maxdepth 3 -type f -executable 2>/dev/null | head -10
else
  echo "=== PORT STEP 4 FAILED (ec=$ec)"
  echo "First 30 error lines:"
  grep -m 30 -E "error:|undefined|fatal" /tmp/rtxadvect_make.log | head -30
  echo
  echo "Full make log at /tmp/rtxadvect_make.log"
fi
