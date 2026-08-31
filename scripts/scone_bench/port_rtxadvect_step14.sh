#!/usr/bin/env bash
#
# Port RTXAdvect — Phase 14: expose the (already-implemented) particle-file
# loader on the CLI.
#
# Wang's code has a hidden `seeding_pts_filename` string with a working
# loader (cudaInitParticles(Particle*, int, std::string)) but no CLI flag
# to set it — the string is hard-set to "" at line 82 and never revisited.
# We add `--input-particles <path>` that sets it.  When present, the code
# path already exists to skip bbox seeding and load from file instead.
#
# File format (per particles.cu:141):
#     NumParticles <N>
#     x y z tetID
#     <x> <y> <z> <tetID>   (tetID column read as a word, ignored)
#     ...
#
# After Phase 14 completes, the exe accepts:
#     --input-particles bunny_inmesh.particles.dat

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
RTX_ROOT="$REPO_ROOT/3rdParty/RTXAdvect"
ADV="$RTX_ROOT/cuda/cudaParticleAdvection.cu"

echo "== Fix 14: add --input-particles CLI flag to $ADV"
python3 - "$ADV" <<'PY'
import sys, re
p = sys.argv[1]
src = open(p).read()

# 1. Insert the --input-particles branch right before the existing
#    --save-streamline-to-obj branch.  This is symmetric with how
#    --seeding-box is parsed above it.
insertion = (
    '\t  else if (arg == "--input-particles") {\n'
    '\t\t  seeding_pts_filename = av[++i];\n'
    '\t  }\n'
)
marker = '\t  else if (arg == "--save-streamline-to-obj") {'
if '--input-particles' in src:
    print("NOTE: --input-particles already wired (already patched?)")
elif marker not in src:
    print(f"ERROR: could not find CLI insertion marker; aborting")
    sys.exit(1)
else:
    src = src.replace(marker, insertion + marker)
    open(p, 'w').write(src)
    print("OK: added --input-particles branch to CLI parser")
PY
if [[ $? -ne 0 ]]; then
  echo "FAILED to patch $ADV"; exit 1
fi

: "${CUDA_HOME:=/usr/local/cuda-13.3}"
: "${OPTIX_ROOT:=/flash/shared/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64}"
: "${CUDA_ARCHS:=120}"
export PATH="$CUDA_HOME/bin:$PATH"

BUILD_DIR="$RTX_ROOT/build"
cd "$BUILD_DIR"

echo
echo "== make -j$(nproc)"
make -j"$(nproc)" 2>&1 | tail -20
ec=$?

if [[ $ec -eq 0 ]]; then
  echo "=== PORT STEP 14 SUCCESS"
else
  echo "=== PORT STEP 14 FAILED (ec=$ec)"
  exit $ec
fi
