#!/usr/bin/env bash
#
# Port RTXAdvect — Phase 13: honor the --seeding-box CLI argument.
#
# The original code hard-codes a "Microfludics" seeding box AFTER the CLI
# parsing loop (cudaParticleAdvection.cu around line 249), which overwrites
# whatever --seeding-box passed in.  For our benchmark we need the CLI arg
# to actually take effect.  Comment out the two offending lines.

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
RTX_ROOT="$REPO_ROOT/3rdParty/RTXAdvect"
ADV="$RTX_ROOT/cuda/cudaParticleAdvection.cu"

echo "== Fix 12: stop hardcoding the seeding box after CLI parse in $ADV"
python3 - "$ADV" <<'PY'
import sys, re
p = sys.argv[1]
src = open(p).read()

# Comment out the two lines:
#     double seedBox[6] = { 73.9 + tol, ...};
#     std::copy(seedBox, seedBox+6, SeedingBox);
# Only touch the Microfludics one; the other seedBox lines are already commented.
new = re.sub(
    r'^([ \t]*)(double\s+seedBox\[6\]\s*=\s*\{\s*73\.9[^;]*;\s*)$',
    r'\1// PORTED-OUT: hardcoded Microfluidics seeding box removed to honor --seeding-box CLI\n\1// \2',
    src,
    flags=re.M,
)
new = re.sub(
    r'^([ \t]*)(std::copy\(seedBox,\s*seedBox\s*\+\s*6\s*,\s*SeedingBox\)\s*;\s*)$',
    r'\1// \2',
    new,
    flags=re.M,
)

if new == src:
    print("NOTE: didn't find the hardcoded seedBox / std::copy lines (already patched?)")
    sys.exit(0)
open(p, 'w').write(new)
print("OK: commented out hardcoded Microfluidics seeding-box override")
PY

# Rebuild only the changed file
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
  echo "=== PORT STEP 13 SUCCESS"
else
  echo "=== PORT STEP 13 FAILED (ec=$ec)"
fi
