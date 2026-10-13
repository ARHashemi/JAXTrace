"""Is the mesh static for every PinShapes case, across the steps it loads?

Decides whether the MALMO search structure can be built once or must be rebuilt
per loaded timestep. Reads only -- no compute job, no writes.

For each case: hash the Points array of one representative piece at several
timesteps spanning that case's own VEL_START..VEL_END. Identical hashes mean
the node coordinates never move, so the grid, Morton codes and registration
are all valid for the whole run. Differing hashes mean the geometry moves and
the structure must be rebuilt (or at least re-registered) per step.
"""
import sys, hashlib, re
import numpy as np, vtk
import xml.etree.ElementTree as ET
from vtk.util.numpy_support import vtk_to_numpy
from pathlib import Path

ROOT = Path(sys.argv[1])
cases = sys.argv[2:] if len(sys.argv) > 2 else None

def points_hash(vtu):
    r = vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(str(vtu)); r.Update()
    g = r.GetOutput()
    pts = g.GetPoints()
    if pts is None:
        return None, 0, 0
    a = vtk_to_numpy(pts.GetData())
    return (hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:12],
            len(a), g.GetNumberOfCells())

rows = []
for fam in sorted(ROOT.iterdir()):
    if not fam.is_dir():
        continue
    for gid in sorted(fam.glob("*.gid")):
        name = gid.name[:-4]
        if cases and name not in cases:
            continue
        rj = gid / "run_jaxtrace.sh"
        post = gid / "post"
        if not rj.exists() or not post.is_dir():
            continue
        txt = rj.read_text(errors="ignore")
        m1 = re.search(r"^VEL_START=(\d+)", txt, re.M)
        m2 = re.search(r"^VEL_END=(\d+)", txt, re.M)
        if not (m1 and m2):
            continue
        vs, ve = int(m1.group(1)), int(m2.group(1))
        pv = sorted(post.glob(f"*_{vs}.pvtu"))
        if not pv:
            continue
        # Derive the prefix by stripping exactly the trailing _<ts>.pvtu, and
        # resolve piece files from the PVTU's own <Piece Source=...> entries
        # rather than constructing names: pieces are numbered from _0, and a
        # hand-built name silently yields zero points if it is wrong.
        prefix = re.sub(rf"_{vs}\.pvtu$", "", pv[0].name)
        # sample across the loaded range, including a full revolution span
        samples = sorted({vs, vs + 1, (vs + ve) // 2, ve - 1, ve})
        ref = None; diffs = []; npts = ncell = 0; nread = 0
        for ts in samples:
            pvt = post / f"{prefix}_{ts}.pvtu"
            if not pvt.exists():
                continue
            try:
                srcs = [p.get("Source")
                        for p in ET.parse(pvt).getroot().iter("Piece")]
            except Exception:
                continue
            if not srcs:
                continue
            f = post / srcs[0]          # same piece index at every timestep
            if not f.exists():
                continue
            h, n, c = points_hash(f)
            if h is None or n == 0:
                continue
            npts, ncell = n, c; nread += 1
            if ref is None:
                ref = (ts, h)
            elif h != ref[1]:
                diffs.append(ts)
        if npts == 0:
            verdict = "NO DATA READ (cannot conclude)"
        elif nread < 2:
            verdict = f"ONLY {nread} TIMESTEP READ (cannot conclude)"
        elif diffs:
            verdict = f"MOVES at {diffs}"
        else:
            verdict = f"STATIC ({nread} steps compared)"
        rows.append((fam.name, name, vs, ve, ve - vs + 1, npts, ncell, verdict))
        print(f"{fam.name:22} {name:8} vel {vs:4}..{ve:<4} "
              f"({ve-vs+1:3} steps)  piece {npts:,}p/{ncell:,}c  {verdict}",
              flush=True)

print()
mv = [r for r in rows if r[-1].startswith("MOVES")]
nc = [r for r in rows if "cannot conclude" in r[-1]]
print(f"cases checked: {len(rows)}   MOVING: {len(mv)}   inconclusive: {len(nc)}")
for r in mv + nc:
    print(f"  {r[0]}/{r[1]}: {r[-1]}")
if not mv and not nc and rows:
    print("\nALL cases static -> one search structure for the whole run is valid:")
    print("  grid, Morton codes AND registration can all be built once.")
