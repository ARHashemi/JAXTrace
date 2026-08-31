#!/usr/bin/env bash
#
# One-shot setup: clone RTXAdvect (Wang et al. 2022, CPC 264:107954),
# check CUDA + OptiX availability, and build. Everything lands under
# 3rdParty/RTXAdvect/ inside the JAXTrace repo.
#
# References:
#   Wang B, Wald I, Morrical N, Usher W, Mu L, Thompson K, Hughes R
#   "A GPU-accelerated particle tracking method for Eulerian-Lagrangian
#    simulations using hardware ray tracing cores"
#   Computer Physics Communications 264:107954 (2022).
#   Repo: https://github.com/binwang0213/RTXAdvect  (Apache 2.0)
#
# Usage:
#   scripts/scone_bench/setup_rtxadvect.sh [--reclone]
#
# Env:
#   OPTIX_ROOT   path to OptiX 7 SDK; auto-detected under /opt/optix* if unset
#   CUDA_HOME    path to CUDA toolkit; auto-detected via /usr/local/cuda if unset

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
cd "$REPO_ROOT"
mkdir -p 3rdParty
cd 3rdParty

RECLONE=0
if [[ "${1:-}" == "--reclone" ]]; then RECLONE=1; fi

# Compatibility-with-CMake-5+ shim: RTXAdvect's CMakeLists calls
# cmake_minimum_required(VERSION 2.8) which CMake 4+ rejects.  The suggested
# workaround (also printed by cmake itself) is to pass this flag.
CMAKE_COMPAT="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

# Workstation-known-good defaults (2026-08-28).
# Override at the command line with e.g. `CUDA_HOME=... OPTIX_ROOT=... $0`.
: "${CUDA_HOME:=/usr/local/cuda-13.3}"
: "${OPTIX_ROOT:=/flash/shared/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64}"
# RTX 5090 is Blackwell = compute capability 12.0
: "${CUDA_ARCHS:=120}"

if [[ -d RTXAdvect && "$RECLONE" == "0" ]]; then
  echo "== RTXAdvect already cloned at 3rdParty/RTXAdvect"
else
  rm -rf RTXAdvect
  echo "== cloning binwang0213/RTXAdvect"
  git clone --depth 1 https://github.com/binwang0213/RTXAdvect.git \
      || { echo "ERROR: git clone failed"; exit 2; }
fi

cd RTXAdvect

echo
echo "== checking build dependencies"

# CUDA — try harder to find nvcc.  On systems where only the driver is
# installed, nvcc will be absent even though nvidia-smi works.
CUDA_HOME="${CUDA_HOME:-}"
if [[ -z "$CUDA_HOME" ]]; then
  # Common install locations, newest-first
  for p in /usr/local/cuda /usr/local/cuda-* /opt/cuda /opt/cuda-* \
           /usr/lib/nvidia-cuda-toolkit /usr; do
    [[ -x "$p/bin/nvcc" ]] && { CUDA_HOME="$p"; break; }
  done
  # Fall back to whatever's on PATH
  if [[ -z "$CUDA_HOME" ]] && command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
  fi
fi
if [[ -n "$CUDA_HOME" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  echo "  CUDA_HOME: $CUDA_HOME"
  "$CUDA_HOME/bin/nvcc" --version 2>&1 | head -5
else
  echo "  WARN: nvcc not found."
  echo "        Install CUDA Toolkit 12.4+ (Blackwell / RTX 5090 needs 12.4+):"
  echo "          apt-get install cuda-toolkit-12-6   # Ubuntu 22.04+"
  echo "        or download from https://developer.nvidia.com/cuda-downloads"
  echo "        Then rerun with CUDA_HOME=/usr/local/cuda-12.6 $0"
fi

# OptiX 7 SDK — cannot be scripted (NVIDIA developer login required).
if [[ -z "${OPTIX_ROOT:-}" ]]; then
  for candidate in /opt/optix* /opt/NVIDIA-OptiX-SDK* /opt/NVIDIA-OptiX* \
                    /flash/shared/optix* /flash/shared/NVIDIA-OptiX* \
                    "$HOME"/OptiX* "$HOME"/NVIDIA-OptiX*; do
    if [[ -d "$candidate" && -f "$candidate/include/optix.h" ]]; then
      OPTIX_ROOT="$candidate"; break
    fi
  done
fi
if [[ -n "${OPTIX_ROOT:-}" && -f "${OPTIX_ROOT}/include/optix.h" ]]; then
  echo "  OPTIX_ROOT: $OPTIX_ROOT"
  # Emit the OptiX version (grep the VERSION macro in optix.h)
  grep -E "OPTIX_VERSION|#define OPTIX_VERSION" "$OPTIX_ROOT/include/optix.h" 2>/dev/null | head -3
else
  echo "  WARN: OPTIX_ROOT not found."
  echo "        RTX 5090 (Blackwell) requires OptiX 7.7 or later."
  echo "        Download the SDK (needs a free NVIDIA developer account):"
  echo "          https://developer.nvidia.com/designworks/optix/download"
  echo "        Extract under /opt/, then rerun with"
  echo "          OPTIX_ROOT=/opt/NVIDIA-OptiX-SDK-8.x-linux64 $0"
fi

# nvidia-smi sanity
echo
echo "== nvidia-smi"
nvidia-smi 2>&1 | head -20 || echo "  (nvidia-smi not available)"

# Configure + build
BUILD_DIR="$REPO_ROOT/3rdParty/RTXAdvect/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Refuse to proceed if either dependency is missing — no point running
# cmake with broken deps and then puzzling over the errors.
if ! command -v nvcc >/dev/null 2>&1 && [[ -z "$CUDA_HOME" ]]; then
  echo
  echo "STOP: CUDA toolkit is missing.  See instructions above."
  exit 4
fi
if [[ -z "${OPTIX_ROOT:-}" ]]; then
  echo
  echo "STOP: OptiX 7 SDK is missing.  See instructions above."
  exit 4
fi

echo
echo "== cmake configure"
echo "     CUDA_HOME    = $CUDA_HOME"
echo "     OPTIX_ROOT   = $OPTIX_ROOT"
echo "     CUDA_ARCHS   = $CUDA_ARCHS  (RTX 5090 = Blackwell / SM 12.0)"
echo "     compat shim  : $CMAKE_COMPAT"
cmake .. \
    $CMAKE_COMPAT \
    -DOptiX_INSTALL_DIR="$OPTIX_ROOT" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DCMAKE_BUILD_TYPE=Release \
  2>&1 | tail -50

echo
echo "== make -j$(nproc)"
make -j"$(nproc)" 2>&1 | tail -80 || { echo "ERROR: build failed"; exit 3; }

echo
echo "== build products"
find "$BUILD_DIR" -maxdepth 3 -type f -executable 2>/dev/null | head -10

echo
echo "=== SETUP DONE"
echo "Executable(s) above.  Next: run scripts/scone_bench/sweep_rtxadvect.sh"
