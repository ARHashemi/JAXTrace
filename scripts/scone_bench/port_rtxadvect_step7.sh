#!/usr/bin/env bash
#
# Port RTXAdvect — Phase 7: redo Phase 6 correctly.
#
# Phase 6 had two regex bugs:
#   - The CPUTimer replacement used a non-greedy .*? that matched the first
#     };\n inside GPUTimer::TimeInMilliseconds(), so it sliced across two
#     classes instead of just CPUTimer.
#   - The thrust::unary_function line-delete used [^>]+ which can't span
#     nested template angle brackets, leaving orphaned "> {".
#
# We restore both files from their .orig-2020 backups and reapply with
# balanced-brace / bracket-counting parsers.
#
# Fix 3 (printf \% → %%) was correct; we don't touch cudaParticleAdvection.cu.
#
# We ALSO add here the two portability fixes exposed downstream:
#   Fix 4 — cudaParticleAdvection.cu has `extern "C" int main(...)`.
#            nvcc 13 rejects that: `main` cannot be inside a linkage
#            specification.  Change extern "C" int main → int main.
#   Fix 5 — cudaParticleAdvection.cu calls system("pause");  — a Windows-
#            only prompt.  Convert to a no-op / guarded call.

set -uo pipefail

REPO_ROOT="$(readlink -f "$(dirname "$0")/../..")"
RTX_ROOT="$REPO_ROOT/3rdParty/RTXAdvect"

HELPERS="$RTX_ROOT/cuda/cudaHelpers.cuh"
PARTICLES="$RTX_ROOT/cuda/particles.cu"
ADV="$RTX_ROOT/cuda/cudaParticleAdvection.cu"

# ------------------------------------------------------------------
# Restore from backup (Phase 6 mangled two files)
# ------------------------------------------------------------------
echo "== restoring $HELPERS and $PARTICLES from .orig-2020"
for f in "$HELPERS" "$PARTICLES"; do
  if [[ -f "$f.orig-2020" ]]; then
    cp -v "$f.orig-2020" "$f"
  else
    echo "WARN: no backup for $f — skipping restore"
  fi
done

# ------------------------------------------------------------------
# Fix 1 (redo) — replace only the CPUTimer class in cudaHelpers.cuh.
# Use a proper brace-counting parser instead of a non-greedy regex.
# ------------------------------------------------------------------
echo
echo "== Fix 1 (redo): CPUTimer -> std::chrono in $HELPERS"
python3 - "$HELPERS" <<'PY'
import sys, re
p = sys.argv[1]
src = open(p).read()

# Find "class CPUTimer" then walk forward matching braces to find the
# terminating };
idx = src.find("class CPUTimer")
if idx < 0:
    print("NOTE: no 'class CPUTimer' in file (already patched?)")
    sys.exit(0)

# Find the opening { after CPUTimer
open_b = src.find("{", idx)
if open_b < 0:
    sys.exit("PARSER ERROR: no { after class CPUTimer")

# Walk to matching }
depth = 0
i = open_b
while i < len(src):
    c = src[i]
    if c == "{":
        depth += 1
    elif c == "}":
        depth -= 1
        if depth == 0:
            # Include the trailing ;
            end = i + 1
            if end < len(src) and src[end] == ";":
                end += 1
            break
    i += 1
else:
    sys.exit("PARSER ERROR: unmatched { in class CPUTimer")

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
};"""

if "#include <chrono>" not in src:
    inc_positions = [m.end() for m in re.finditer(r'^#\s*include\s+[<"][^>"]+[>"]\s*$', src, flags=re.M)]
    if inc_positions:
        pos = inc_positions[-1]
        src = src[:pos] + "\n#include <chrono>" + src[pos:]
        # After that insertion, the indices we computed shift.  Recompute idx/end.
        shift = len("\n#include <chrono>")
        if idx > pos:  idx += shift
        if open_b > pos: open_b += shift
        if end > pos:  end += shift

src = src[:idx] + replacement + src[end:]
open(p, 'w').write(src)
print(f"OK: replaced CPUTimer class in {p} (bytes idx={idx}..{end})")
PY

# ------------------------------------------------------------------
# Fix 2 (redo) — drop thrust::(unary|binary)_function inheritance.
# Match the ": public thrust::X_function<...>" with balanced-bracket parser.
# ------------------------------------------------------------------
echo
echo "== Fix 2 (redo): drop thrust functor inheritance in $PARTICLES"
python3 - "$PARTICLES" <<'PY'
import sys, re
p = sys.argv[1]
src = open(p).read()

def strip_inheritance(text, name):
    """Find ': public thrust::<name>< ... >' with balanced angle brackets
    and remove it (including preceding whitespace/newline)."""
    marker = f"public thrust::{name}"
    out = []
    i = 0
    n_removed = 0
    while i < len(text):
        j = text.find(marker, i)
        if j < 0:
            out.append(text[i:])
            break
        # Walk backward to find the ':' before "public"
        k = j - 1
        while k >= 0 and text[k] in " \t":
            k -= 1
        if k < 0 or text[k] != ":":
            # Not the pattern we expected; leave it and advance past
            out.append(text[i:j+1])
            i = j + 1
            continue
        # k points at ':'.  Walk further back to also swallow the newline+indent that precedes it.
        line_start = text.rfind("\n", 0, k) + 1
        # Verify the segment between line_start..k is just whitespace
        if text[line_start:k].strip() != "":
            # Not a clean 'own-line' inheritance — do it in place; still remove.
            line_start = k
        # Walk forward from j through "public thrust::<name>" then through balanced <...>
        m = j + len(marker)
        # Skip whitespace to '<'
        while m < len(text) and text[m] in " \t":
            m += 1
        if m >= len(text) or text[m] != "<":
            # No <>; give up on this match
            out.append(text[i:j+1])
            i = j + 1
            continue
        depth = 0
        end = m
        while end < len(text):
            c = text[end]
            if c == "<":
                depth += 1
            elif c == ">":
                depth -= 1
                if depth == 0:
                    end += 1
                    break
            end += 1
        # Now [line_start..end) is the thing we want to remove.
        # Also swallow any trailing spaces (but leave the newline so the
        # next line's brace stays where it is).
        while end < len(text) and text[end] in " \t":
            end += 1
        out.append(text[i:line_start])
        n_removed += 1
        i = end
    return "".join(out), n_removed

src, n1 = strip_inheritance(src, "unary_function")
src, n2 = strip_inheritance(src, "binary_function")
if n1 + n2 == 0:
    print("NOTE: no thrust::(unary|binary)_function inheritance found")
else:
    open(p, 'w').write(src)
    print(f"OK: removed {n1} unary + {n2} binary thrust functor inheritance(s) in {p}")
PY

# ------------------------------------------------------------------
# Fix 4 — extern "C" int main -> int main
# ------------------------------------------------------------------
echo
echo "== Fix 4: 'extern \"C\" int main' -> 'int main' in $ADV"
if [[ ! -f "$ADV.orig-2020" ]]; then
  cp -v "$ADV" "$ADV.orig-2020"
fi
sed -i 's/extern[[:space:]]*"C"[[:space:]]*int[[:space:]]*main/int main/' "$ADV"
grep -n '^int main\|extern "C" int main' "$ADV" | head -3

# ------------------------------------------------------------------
# Fix 5 — system("pause") is Windows-only.  Comment it out.
# ------------------------------------------------------------------
echo
echo "== Fix 5: comment out system(\"pause\") in $ADV"
sed -i 's|system("pause");|/* removed system("pause") — Windows-only */|' "$ADV"

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
make -j"$(nproc)" 2>&1 | tee /tmp/rtxadvect_make.log | tail -80
ec=${PIPESTATUS[0]}

echo
if [[ $ec -eq 0 ]]; then
  echo "=== PORT STEP 7 SUCCESS"
  find "$BUILD_DIR" -maxdepth 3 -type f -executable 2>/dev/null | head -10
else
  echo "=== PORT STEP 7 FAILED (ec=$ec)"
  echo "First 30 error lines:"
  grep -m 30 -E "error:|undefined|fatal" /tmp/rtxadvect_make.log | head -30
fi
