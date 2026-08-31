#!/usr/bin/env bash
#
# Run Kim/Ravoisin/Parks (2026) acceleration structure inside SCONE on a
# given OpenFOAM polyMesh, without any modification to their code.
#
# Requires: SCONE built at $SCONE_BUILD_DIR (default:
#           /home/arhashemi/fsw-gpu/flash/shared/jax/ii1o33-SCONE-00d7668/build)
#           Python 3 with numpy (only for reading the polyMesh bbox).
#
# Usage:
#   scripts/scone_bench/run_pointloc.sh <polymesh_dir> <accelMethod> [pop] [cycles] [omp]
#
# accelMethod ∈ {none, octree, patchSingle, patchMulti}
#   - none         : brute force (no acceleration structure)
#   - octree       : Kim et al.'s optimised linear octree (paper baseline)
#   - patchSingle  : Chen & Yang single-layer patch search
#   - patchMulti   : Kim/Ravoisin/Parks Adaptive Multi-Layered Grid (AMLG)
#
# Output goes to  <cwd>/scone_runs/<mesh_basename>__<accelMethod>/
#   - input.txt   : the exact SCONE input we ran
#   - stdout.log  : full SCONE stdout (contains the Main Timer wall time)
#   - result.json : mesh name, method, pop, cycles, omp, wall time (seconds)

set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<EOF
usage: $0 <polymesh_dir> <accelMethod> [pop=50000] [cycles=10] [omp=1]
  accelMethod ∈ {none, octree, patchSingle, patchMulti}
EOF
  exit 2
fi

POLYMESH="$(readlink -f "$1")"
ACCEL="$2"
POP="${3:-50000}"
CYCLES="${4:-10}"
OMP="${5:-1}"

SCONE_BUILD_DIR="${SCONE_BUILD_DIR:-/flash/shared/jax/ii1o33-SCONE-00d7668/build}"
SCONE_EXE="${SCONE_BUILD_DIR}/scone.out"

SCRIPT_DIR="$(readlink -f "$(dirname "$0")")"
TEMPLATE="${SCRIPT_DIR}/pointloc_template.txt"

case "$ACCEL" in
  none|octree|patchSingle|patchMulti) ;;
  *)
    echo "ERROR: accelMethod must be one of: none, octree, patchSingle, patchMulti (got '${ACCEL}')" >&2
    exit 2
    ;;
esac

for f in "$SCONE_EXE" "$TEMPLATE" "${POLYMESH}/points" "${POLYMESH}/faces" "${POLYMESH}/owner"; do
  if [[ ! -e "$f" ]]; then
    echo "ERROR: required file not found: $f" >&2
    exit 2
  fi
done

# Number of cellZones in the polyMesh. SCONE's meshUniverse requires exactly
# one 'fills' entry per zone. If cellZones is missing, SCONE's reader creates
# a single "Default" zone (see OpenFOAMMesh_class.f90:361), so N_ZONES=1 fits.
if [[ -f "${POLYMESH}/cellZones" ]]; then
  N_ZONES="$(python3 - <<PY
from pathlib import Path
text = Path("${POLYMESH}/cellZones").read_text()
# Strip block-comment header
i = text.find('*/')
if i != -1:
    text = text[i+2:]
# Strip the FoamFile { ... } dictionary
i = text.find('{')
if i != -1:
    depth = 0
    j = i
    while j < len(text):
        if text[j] == '{': depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                text = text[j+1:]
                break
        j += 1
# First non-blank line is the zone count
for ln in text.splitlines():
    s = ln.strip()
    if not s or s.startswith('//'):
        continue
    try:
        print(int(s))
    except ValueError:
        print(1)
    break
else:
    print(1)
PY
)"
else
  N_ZONES=1
fi

# Build the 'fills (...)' payload: N copies of the material name 'mat'
FILLS="$(python3 -c "print(' '.join(['mat'] * ${N_ZONES}))")"

# Compute enclosing box halfwidths from polyMesh 'points' file.
# SCONE's box.cropsBoundingBox (Geometry/Surfaces/CompoundSurfaces/box_class.f90)
# compares abs(mesh_bound) against the box halfwidths directly — it assumes
# the box surface is centred at the world origin (0,0,0). So we ALWAYS place
# the boxContainer at (0,0,0) and set halfwidth per axis to
#   max(|xmin|, |xmax|) * BOX_PAD
# This is geometrically correct for both centred meshes (fuel-pin tets already
# lie in [-h,+h]) and non-centred meshes (bunny lies in [-24, +84] → hx≈84).
#
# BOX_PAD can be overridden via env var to test whether SCONE's octree ray-
# tracing failure "Ray has lost correct material" is caused by float-precision
# rays leaking to the very edge of the mesh. Default 1.05; try 5.0 or 10.0
# to move the containment surface far from the mesh boundary.
BOX_PAD="${BOX_PAD:-1.05}"
HALFWIDTH="$(POLYMESH_DIR="$POLYMESH" BOX_PAD="$BOX_PAD" python3 - <<'PY'
from pathlib import Path
import os, re, sys
POLYMESH = os.environ["POLYMESH_DIR"]
lines = Path(f"{POLYMESH}/points").read_text().splitlines()
xs, ys, zs = [], [], []
in_list = False
for ln in lines:
    s = ln.strip()
    if not in_list:
        if s == "(": in_list = True
        continue
    if s == ")": break
    m = re.match(r"^\(?([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)?", s)
    if m:
        xs.append(float(m.group(1))); ys.append(float(m.group(2))); zs.append(float(m.group(3)))
if not xs:
    sys.exit(f"ERROR: could not parse any points from {POLYMESH}/points")
xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys); zmin, zmax = min(zs), max(zs)
pad = float(os.environ["BOX_PAD"])
hx = max(abs(xmin), abs(xmax)) * pad
hy = max(abs(ymin), abs(ymax)) * pad
hz = max(abs(zmin), abs(zmax)) * pad
hx = max(hx, 1e-6); hy = max(hy, 1e-6); hz = max(hz, 1e-6)
print(f"{hx:.15g} {hy:.15g} {hz:.15g}")
PY
)"

# Box always at world origin (SCONE convention); convert commas for SCONE
ORIGIN="0.0 0.0 0.0"

MESH_NAME="$(basename "$POLYMESH")"
# Include omp thread count in OUT_DIR so single-thread and multi-thread runs
# of the same (mesh, method) don't clobber each other.
OUT_DIR="$(pwd)/scone_runs/${MESH_NAME}__${ACCEL}__omp${OMP}"
mkdir -p "$OUT_DIR"

# SCONE has a hard-coded pathLen=100 (SharedModules/numPrecision.f90). Our
# absolute mesh paths are often longer, so Fortran silently truncates them and
# the reader reports "folder does not exist". Work around by creating a short
# relative symlink 'm/' inside OUT_DIR pointing at the real polyMesh and
# writing 'path ./m/;' into the input while running scone from OUT_DIR.
ln -sfn "$POLYMESH" "${OUT_DIR}/m"
SCONE_PATH_STR="./m/"

INPUT="${OUT_DIR}/input.txt"
sed \
  -e "s|__POLYMESH_PATH__|${SCONE_PATH_STR}|g" \
  -e "s|__ACCEL_METHOD__|${ACCEL}|g" \
  -e "s|__BOX_ORIGIN__|${ORIGIN}|g" \
  -e "s|__BOX_HALFWIDTH__|${HALFWIDTH}|g" \
  -e "s|__POP__|${POP}|g" \
  -e "s|__CYCLES__|${CYCLES}|g" \
  -e "s|__FILLS__|${FILLS}|g" \
  "$TEMPLATE" > "$INPUT"

echo "=== run_pointloc.sh"
echo "    mesh        : $POLYMESH"
echo "    accelMethod : $ACCEL"
echo "    zones/fills : ${N_ZONES}  ->  fills(${FILLS})"
echo "    pop/cycles  : ${POP} / ${CYCLES}"
echo "    omp threads : ${OMP}"
echo "    box origin  : (${ORIGIN})"
echo "    box halfw   : (${HALFWIDTH})  (BOX_PAD=${BOX_PAD})"
echo "    input       : $INPUT"
echo "    scone.out   : $SCONE_EXE"

LOG="${OUT_DIR}/stdout.log"
TIME_LOG="${OUT_DIR}/time.log"
T0=$(date +%s.%N)
# Run scone from OUT_DIR so its short relative './m/' path resolves and stays
# well under SCONE's pathLen=100 limit. Pass an absolute input path.
# Wrap with GNU 'time -v' to capture peak RSS + user/sys CPU (writes to TIME_LOG).
#
# AMLG (patchMulti) uses large automatic arrays inside refineCell2 that
# overflow gfortran's default 8 MB stack -> silent SIGSEGV before any
# print* line flushes. Two mitigations, both applied here:
#   (a) 'ulimit -s unlimited'   raises the main-thread stack.
#   (b) OMP_STACKSIZE=1G        raises each OpenMP worker's stack.
# These affect only the SCONE process launched here; no source changes.
# Temporarily disable 'set -e' so a SCONE segfault does not abort the driver
# before we can write result.json (the whole point of the enriched JSON is to
# CAPTURE failures as structured rows for the CSV).
set +e
(
  cd "$OUT_DIR"
  ulimit -s unlimited 2>/dev/null || true
  export OMP_STACKSIZE=1G
  # Force Fortran unit 6 (stdout) to flush after every write so any diagnostic
  # print survives an early crash. Also disable output buffering via stdbuf.
  export GFORTRAN_UNBUFFERED_ALL=1
  export GFORTRAN_UNBUFFERED_PRECONNECTED=1
  stdbuf -o0 -e0 /usr/bin/time -v -o "$TIME_LOG" "$SCONE_EXE" "$INPUT" --omp "$OMP"
) > "$LOG" 2>&1
ec=$?
set -e
if [[ $ec -ne 0 ]]; then
  echo "SCONE exited with code $ec — see $LOG" >&2
  tail -40 "$LOG" >&2
  # Do NOT abort: we still want to run the result-parser so the crash is
  # captured as a structured row in the CSV. The sweep can continue.
fi
T1=$(date +%s.%N)
WALL=$(python3 -c "print(f'{${T1}-${T0}:.6f}')")

# Parse SCONE stdout + /usr/bin/time output into structured metrics
python3 - <<PY > "${OUT_DIR}/result.json"
import json, re
from pathlib import Path
log = Path("${LOG}").read_text(errors="replace")
tlog = Path("${TIME_LOG}").read_text(errors="replace") if Path("${TIME_LOG}").exists() else ""

# --- SCONE stdout parsing ---
# Cartesian grid banner (patchSingle + patchMulti print this)
def find(pat, txt, cast=str, default=None, flags=0):
    m = re.search(pat, txt, flags)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except Exception:
        return default

grid = {
    "min_angle":        find(r"Minimum angle\s*:\s*([-0-9.Ee+]+)", log, float),
    "min_edge_length":  find(r"Minimum edge length\s*:\s*([-0-9.Ee+]+)", log, float),
    "n_vertices":       find(r"No\. of vertices\s*:\s*([0-9]+)", log, int),
    "n_edges":          find(r"No\. of edges\s*:\s*([0-9]+)", log, int),
    "n_faces":          find(r"No\. of faces\s*:\s*([0-9]+)", log, int),
    "n_elements":       find(r"No\. of elements\s*:\s*([0-9]+)", log, int),
    "n_layers":         find(r"No\. of layers\s*:\s*([0-9]+)", log, int),
    "grid_spacing_coarsest": find(r"Grid spacing for coarsest\s*:\s*([-0-9.Ee+]+)", log, float),
    "grid_spacing_finest":   find(r"Grid spacing for finest\s*:\s*([-0-9.Ee+]+)", log, float),
    # Single-layer patch prints these differently
    "grid_spacing_single":   find(r"^ Grid spacing\s*:\s*([-0-9.Ee+]+)", log, float, flags=re.M),
    "grid_size_x":           find(r"Grid size in x\s*:\s*([0-9]+)", log, int),
    "grid_size_y":           find(r"Grid size in y\s*:\s*([0-9]+)", log, int),
    "grid_size_z":           find(r"Grid size in z\s*:\s*([0-9]+)", log, int),
}
# AMLG (patchMulti) prints its coarsest+finest grid dims on lines like
#   "Grid size in xyz for coarsest       :  23  23  19"
#   "Grid size in xyz for finest         : 529 529 437"
def _xyz(pat, txt):
    m = re.search(pat, txt)
    if not m: return None
    parts = [int(x) for x in m.group(1).split()]
    if len(parts) != 3: return None
    return parts
xyz_coarsest = _xyz(r"Grid size in xyz for coarsest\s*:\s*([-0-9.Ee+ \t]+)$", log if False else log)
xyz_finest   = _xyz(r"Grid size in xyz for finest\s*:\s*([-0-9.Ee+ \t]+)$", log if False else log)
xyz_coarsest = None
xyz_finest = None
for line in log.splitlines():
    m = re.match(r"\s*Grid size in xyz for coarsest\s*:\s*([-0-9. \t]+)", line)
    if m:
        try: xyz_coarsest = [int(x) for x in m.group(1).split()]
        except Exception: pass
    m = re.match(r"\s*Grid size in xyz for finest\s*:\s*([-0-9. \t]+)", line)
    if m:
        try: xyz_finest = [int(x) for x in m.group(1).split()]
        except Exception: pass

def _prod(xyz):
    if not xyz or len(xyz) != 3: return None
    return int(xyz[0]) * int(xyz[1]) * int(xyz[2])

grid["grid_xyz_coarsest"] = xyz_coarsest
grid["grid_xyz_finest"]   = xyz_finest
grid["n_cells_coarsest"]  = _prod(xyz_coarsest)
grid["n_cells_finest"]    = _prod(xyz_finest)

# SCONE's Main Timer line, and any per-cycle times if present
main_timer_s = find(r"Main Timer[^0-9]*([0-9.Ee+]+)\s*s", log, float)
cycle_wall = [float(x) for x in re.findall(r"Cycle\s+\d+.*?wall\s*=\s*([0-9.Ee+]+)", log)]

# --- /usr/bin/time -v parsing ---
def tfind(key, cast=float, default=None):
    m = re.search(rf"^\s*{re.escape(key)}:\s*(.+)$", tlog, re.M)
    if not m:
        return default
    v = m.group(1).strip()
    try:
        return cast(v)
    except Exception:
        return v

# 'Elapsed (wall clock) time' can be h:mm:ss or m:ss.ss — convert
elapsed_raw = tfind("Elapsed (wall clock) time (h:mm:ss or m:ss)", str, None)
def _parse_elapsed(s):
    if s is None: return None
    parts = s.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:  return parts[0]*3600 + parts[1]*60 + parts[2]
    if len(parts) == 2:  return parts[0]*60 + parts[1]
    if len(parts) == 1:  return parts[0]
    return None

time_metrics = {
    "user_cpu_seconds": tfind("User time (seconds)", float),
    "sys_cpu_seconds":  tfind("System time (seconds)", float),
    "elapsed_seconds":  _parse_elapsed(elapsed_raw),
    "percent_cpu":      tfind("Percent of CPU this job got", str),
    "max_rss_kb":       tfind("Maximum resident set size (kbytes)", int),
    "voluntary_ctx_switches":   tfind("Voluntary context switches", int),
    "involuntary_ctx_switches": tfind("Involuntary context switches", int),
    "exit_status":              tfind("Exit status", int),
}

out = {
    "mesh": "${MESH_NAME}",
    "polymesh_dir": "${POLYMESH}",
    "accelerationMethod": "${ACCEL}",
    "n_zones": ${N_ZONES},
    "pop": ${POP},
    "cycles": ${CYCLES},
    "omp_threads": ${OMP},
    "box_pad": ${BOX_PAD},
    "box_halfwidth": [${HALFWIDTH// /,}],
    "shell_exit_code": ${ec},
    "crashed_sigsegv": ${ec} == 139,
    "wall_seconds_outer": ${WALL},
    "main_timer_seconds": main_timer_s,
    "cycle_wall_seconds": cycle_wall,
    "grid": grid,
    "process": time_metrics,
}
print(json.dumps(out, indent=2, default=str))
PY

echo "--- last lines of stdout ---"
tail -15 "$LOG"
echo
echo "=== done: ${OUT_DIR}/result.json"
cat "${OUT_DIR}/result.json"
