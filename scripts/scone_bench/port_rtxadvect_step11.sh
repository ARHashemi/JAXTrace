#!/usr/bin/env bash
#
# Port RTXAdvect — Phase 11: two link-time fixes.
#
# Fix 9: main() lives inside `namespace advect { ... }` in
#        cudaParticleAdvection.cu.  Phase 7 stripped 'extern "C"' but that
#        left main() as a member of namespace advect; the linker can't
#        find ::main and produces "undefined reference to `main'".
#        Move main() out of the namespace by (a) closing the namespace
#        just before line 67 and (b) opening a `using namespace advect;`
#        so the body still compiles.  Then remove the outer `}` that
#        was at line 497 since we already closed the namespace earlier.
#
# Fix 10: the OptixTetQuery / OptixTriQuery code references
#         `extern "C" const char ptxCode[]` — the old cuda_compile_and_embed
#         macro emitted a variable literally named `ptxCode`.
#         The modern embed_ptx() defaults to `<basename>_ptx` (i.e.
#         `optixQueryKernel_ptx`).  Rename the embedded symbol back to
#         `ptxCode` by passing EMBEDDED_SYMBOL_NAMES.

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
RTX_ROOT="$REPO_ROOT/3rdParty/RTXAdvect"
ADV="$RTX_ROOT/cuda/cudaParticleAdvection.cu"
CMAKE_FILE="$RTX_ROOT/CMakeLists.txt"

# ------------------------------------------------------------------
# Fix 9: move main() outside namespace advect
# ------------------------------------------------------------------
echo "== Fix 9: move main() outside namespace advect in $ADV"
python3 - "$ADV" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p).read()

# Find the opening 'namespace advect {'
m = re.search(r'^\s*namespace\s+advect\s*\{\s*$', src, flags=re.M)
if not m:
    sys.exit("PARSER ERROR: could not find 'namespace advect {' line")
ns_open = m.start()
ns_open_end = m.end()

# Find the closing '}' at namespace scope (brace-count from ns_open_end).
depth = 1
i = ns_open_end
while i < len(src):
    c = src[i]
    if c == "{": depth += 1
    elif c == "}":
        depth -= 1
        if depth == 0:
            break
    i += 1
if depth != 0:
    sys.exit("PARSER ERROR: unmatched { on namespace advect")
ns_close = i
# Include one trailing newline
if ns_close + 1 < len(src) and src[ns_close + 1] == "\n":
    ns_close += 1

# Find `int main(int ac,` inside the namespace to know where to split
mm = re.search(r'^\s*int\s+main\s*\(\s*int\s+ac\s*,', src, flags=re.M)
if not mm or not (ns_open_end < mm.start() < ns_close):
    print("NOTE: could not find 'int main(int ac,' inside namespace advect (already moved?)")
    sys.exit(0)
main_start = src.rfind("\n", 0, mm.start()) + 1  # start of line

# Compose new source:
#   [everything up to just before main_start]  (still inside namespace)
#   } // namespace advect
#   using namespace advect;
#   int main(int ac, char **av) { ... }
#   [everything after the original ns_close]
# The original '}' at ns_close is DROPPED because we've moved it up.
before = src[:main_start]
main_body = src[main_start:ns_close]
after = src[ns_close+1:] if ns_close+1 < len(src) else ""

# Insert namespace close + using directive between `before` and `main_body`
insert = "} // namespace advect (moved main() outside)\n\nusing namespace advect;\n\n"

new = before + insert + main_body + after
open(p, 'w').write(new)
print(f"OK: moved main() outside 'namespace advect' in {p}")
PY

# ------------------------------------------------------------------
# Fix 10: rename the embedded PTX symbol from optixQueryKernel_ptx
#         back to `ptxCode` (what OptixTetQuery.cpp expects)
# ------------------------------------------------------------------
echo
echo "== Fix 10: rename embedded PTX symbol to ptxCode in $CMAKE_FILE"
# Look for the existing embed_ptx block and add EMBEDDED_SYMBOL_NAMES if absent
if grep -q "EMBEDDED_SYMBOL_NAMES" "$CMAKE_FILE"; then
  echo "NOTE: EMBEDDED_SYMBOL_NAMES already present (already patched?)"
else
  python3 - "$CMAKE_FILE" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p).read()
# Insert EMBEDDED_SYMBOL_NAMES immediately before SOURCES
new = re.sub(
    r'(embed_ptx\(\s*OUTPUT_TARGET\s+optixQueryKernel_ptx\s+PTX_LINK_LIBRARIES\s+owl::owl\s+)(SOURCES)',
    r'\1EMBEDDED_SYMBOL_NAMES\n    ptxCode\n  \2',
    src,
    count=1,
)
if new == src:
    sys.exit("PARSER ERROR: could not locate embed_ptx block")
open(p, 'w').write(new)
print("OK: added EMBEDDED_SYMBOL_NAMES ptxCode to embed_ptx() call")
PY
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
  2>&1 | tail -20

echo
echo "== make -j$(nproc)"
make -j"$(nproc)" 2>&1 | tee /tmp/rtxadvect_make.log | tail -30
ec=${PIPESTATUS[0]}

echo
if [[ $ec -eq 0 ]]; then
  echo "=== PORT STEP 11 SUCCESS"
  find "$BUILD_DIR" -maxdepth 3 -type f -executable 2>/dev/null | head -10
  echo
  echo "=== TRYING --help TO CONFIRM"
  find "$BUILD_DIR" -maxdepth 3 -type f -executable -name "cudaParticleAdvection" -exec {} --help \; 2>&1 | head -30 || true
else
  echo "=== PORT STEP 11 FAILED (ec=$ec)"
  echo "First 30 error lines:"
  grep -m 30 -E "error:|undefined|fatal" /tmp/rtxadvect_make.log | head -30
fi
