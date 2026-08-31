#!/usr/bin/env bash
#
# Port RTXAdvect — Phase 1: swap the bundled 2020-vintage OWL submodule for
# the current NVIDIA/OWL and attempt to rebuild against CUDA 13.3 + OptiX 9.1.
#
# See scripts/scone_bench/port_rtxadvect_plan.md for full context.
#
# Usage:
#   scripts/scone_bench/port_rtxadvect_step1.sh
#
# Env (all optional, defaults match the workstation as of 2026-08-28):
#   CUDA_HOME      /usr/local/cuda-13.3
#   OPTIX_ROOT     /flash/shared/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64
#   CUDA_ARCHS     120  (RTX 5090 / Blackwell)
#   OWL_URL        https://github.com/NVIDIA/OWL.git   (new upstream OWL)
#   OWL_BRANCH     master  (or a tag)

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
RTX_ROOT="$REPO_ROOT/3rdParty/RTXAdvect"

: "${CUDA_HOME:=/usr/local/cuda-13.3}"
: "${OPTIX_ROOT:=/flash/shared/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64}"
: "${CUDA_ARCHS:=120}"
: "${OWL_URL:=https://github.com/NVIDIA/OWL.git}"
# NVIDIA/OWL uses 'main' as default; older forks used 'master'. Try main
# first, fall back to master if not present.
: "${OWL_BRANCH:=}"

if [[ ! -d "$RTX_ROOT" ]]; then
  echo "ERROR: RTXAdvect not cloned yet. Run setup_rtxadvect.sh first."
  exit 2
fi

cd "$RTX_ROOT"

# Step 1a — swap OWL
if [[ -d owl && ! -d owl.orig-2020 ]]; then
  echo "== backing up bundled OWL to owl.orig-2020"
  mv owl owl.orig-2020
fi
if [[ ! -f owl/CMakeLists.txt ]]; then
  rm -rf owl
  if [[ -n "$OWL_BRANCH" ]]; then
    echo "== cloning NVIDIA/OWL (branch: $OWL_BRANCH)"
    git clone --depth 1 --branch "$OWL_BRANCH" "$OWL_URL" owl \
      || { echo "ERROR: git clone --branch $OWL_BRANCH failed"; exit 3; }
  else
    # Let git pick the default branch (main for NVIDIA/OWL, master for older forks)
    echo "== cloning NVIDIA/OWL (default branch)"
    git clone --depth 1 "$OWL_URL" owl \
      || { echo "ERROR: git clone $OWL_URL failed"; exit 3; }
  fi
  echo "== OWL default branch: $(cd owl && git rev-parse --abbrev-ref HEAD)"
  echo "== OWL last commit: $(cd owl && git log -1 --format='%h %s (%ar)')"
fi

# Step 1b — wipe old CMake cache
if [[ -d build ]]; then
  echo "== wiping old build/"
  rm -rf build
fi
mkdir -p build && cd build

# Step 1c — configure
export PATH="$CUDA_HOME/bin:$PATH"
echo
echo "== cmake configure"
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
# Save the full make log — first 200 lines of errors are the useful part
make -j"$(nproc)" 2>&1 | tee /tmp/rtxadvect_make.log | tail -100
ec=${PIPESTATUS[0]}

echo
if [[ $ec -eq 0 ]]; then
  echo "=== PORT STEP 1 SUCCESS"
  find "$RTX_ROOT/build" -maxdepth 3 -type f -executable 2>/dev/null | head -10
else
  echo "=== PORT STEP 1 FAILED (ec=$ec)"
  echo "First 20 error lines:"
  grep -m 20 -E "error:|undefined|fatal" /tmp/rtxadvect_make.log | head -20
  echo
  echo "Full make log at /tmp/rtxadvect_make.log"
  echo "Next: read port_rtxadvect_plan.md Phase 2 (fix RTXAdvect surface calls)"
fi
