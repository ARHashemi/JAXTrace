#!/usr/bin/env bash
#
# Port RTXAdvect — Phase 2: patch RTXAdvect's top-level CMakeLists.txt to
# use the modern NVIDIA/OWL API:
#   - include(configure_owl)                   -> removed (OWL now exports targets)
#   - include_directories(${OWL_INCLUDES})     -> removed (target-based)
#   - target_link_libraries(... ${OWL_LIBS})   -> owl::owl
#   - cuda_compile_and_embed(ptxCode …)        -> embed_ptx(OUTPUT_TARGET … )
#   - cuda_add_executable(…)                   -> add_executable(… .cu)
#
# Backs up the original to CMakeLists.txt.orig-2020 the first time.
#
# See scripts/scone_bench/port_rtxadvect_plan.md for the overall plan.

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
RTX_ROOT="$REPO_ROOT/3rdParty/RTXAdvect"
CMAKE_FILE="$RTX_ROOT/CMakeLists.txt"

if [[ ! -f "$CMAKE_FILE" ]]; then
  echo "ERROR: $CMAKE_FILE not found — is RTXAdvect cloned?"
  exit 2
fi

if [[ ! -f "$CMAKE_FILE.orig-2020" ]]; then
  cp -v "$CMAKE_FILE" "$CMAKE_FILE.orig-2020"
fi

cat > "$CMAKE_FILE" <<'CMAKE_EOF'
# ======================================================================== #
# Copyright 2019-2020 The Collaborators                                    #
# Ported to CUDA 13.x + OptiX 9.x + NVIDIA/OWL by JAXTrace revision team,  #
# 2026-08-28.                                                              #
# Licensed under the Apache License, Version 2.0 (see LICENSE).            #
# ======================================================================== #

cmake_minimum_required(VERSION 3.24)
project(RTXAdvect VERSION 0.0.2 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------
# NVIDIA/OWL as a subproject.  New OWL exports targets `owl::owl`
# and provides the `embed_ptx` CMake function via its cmake/ dir.
# ------------------------------------------------------------------
set(owl_dir ${PROJECT_SOURCE_DIR}/owl)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

# embed_ptx() lives in owl/owl/cmake/embed_ptx.cmake; add_subdirectory
# above should already have brought it into scope, but include the
# module path explicitly for clarity.
list(APPEND CMAKE_MODULE_PATH "${owl_dir}/owl/cmake")
include(embed_ptx)

# ------------------------------------------------------------------
# OptixTetQueries library
# ------------------------------------------------------------------
embed_ptx(
  OUTPUT_TARGET
    optixQueryKernel_ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    optix/optixQueryKernel.cu
)

add_library(optixTetQueries STATIC
  optix/internalTypes.h
  optix/OptixQuery.h
  optix/OptixTetQuery.cpp
  optix/OptixTriQuery.cpp
)
target_include_directories(optixTetQueries PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(optixTetQueries
  PUBLIC
    owl::owl
    optixQueryKernel_ptx
)

# ------------------------------------------------------------------
# The main executable
# ------------------------------------------------------------------
add_executable(cudaParticleAdvection
  cuda/common.h
  cuda/utils.cpp
  cuda/particles.cu
  cuda/HostTetMesh.h
  cuda/DeviceTetMesh.cuh
  query/ConvexQuery.h
  query/ConvexQuery.cu
  query/RTQuery.h
  query/RTQuery.cu
  cuda/cudaParticleAdvection.cu
)
target_include_directories(cudaParticleAdvection PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(cudaParticleAdvection
  PRIVATE
    optixTetQueries
    owl::owl
)

# RTX 5090 = Blackwell / sm_120 is picked up by CMAKE_CUDA_ARCHITECTURES
# passed on the command line; OWL sets sensible defaults if unset.

message(STATUS "RTXAdvect ported to NVIDIA/OWL + OptiX 9.x + CUDA 13.x")
CMAKE_EOF

echo "== patched $CMAKE_FILE"
echo "-- backup at $CMAKE_FILE.orig-2020"

# Now rerun the build in a wiped build/ directory
: "${CUDA_HOME:=/usr/local/cuda-13.3}"
: "${OPTIX_ROOT:=/flash/shared/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64}"
: "${CUDA_ARCHS:=120}"

BUILD_DIR="$RTX_ROOT/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

export PATH="$CUDA_HOME/bin:$PATH"
echo
echo "== cmake configure (post-patch)"
echo "     CUDA_HOME    = $CUDA_HOME"
echo "     OPTIX_ROOT   = $OPTIX_ROOT"
echo "     CUDA_ARCHS   = $CUDA_ARCHS"
cmake .. \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DOptiX_INSTALL_DIR="$OPTIX_ROOT" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DCMAKE_BUILD_TYPE=Release \
    -Wno-dev \
  2>&1 | tail -60

echo
echo "== make -j$(nproc)"
make -j"$(nproc)" 2>&1 | tee /tmp/rtxadvect_make.log | tail -100
ec=${PIPESTATUS[0]}

echo
if [[ $ec -eq 0 ]]; then
  echo "=== PORT STEP 2 SUCCESS"
  find "$BUILD_DIR" -maxdepth 3 -type f -executable 2>/dev/null | head -10
else
  echo "=== PORT STEP 2 FAILED (ec=$ec)"
  echo "First 30 error lines:"
  grep -m 30 -E "error:|undefined|fatal" /tmp/rtxadvect_make.log | head -30
  echo
  echo "Full make log at /tmp/rtxadvect_make.log"
  echo
  echo "Next step: read /tmp/rtxadvect_make.log more carefully — most"
  echo "likely we need to patch RTXAdvect's own C++/CUDA source to match"
  echo "OptiX 9 / new OWL naming.  This is Phase 3 of the port plan."
fi
