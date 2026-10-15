"""Is snapshot loading I/O-bound or CPU-bound? That decides whether
parallelising it helps, and whether the GPU is relevant at all."""
import os, time, subprocess
from pathlib import Path
import numpy as np

post = Path("/scratch/shared/PinShapes/D-ConcavityTilt/D2.gid/post")
pv = post / "A2_40.pvtu"
import xml.etree.ElementTree as ET
srcs = [p.get("Source") for p in ET.parse(pv).getroot().iter("Piece")]
files = [post / s for s in srcs]
total_bytes = sum(f.stat().st_size for f in files)
print(f"one snapshot = {len(files)} piece files, "
      f"{total_bytes/2**20:.0f} MiB total")

# 1. RAW I/O: just read the bytes, no parsing
def raw_read():
    n = 0
    for f in files:
        with open(f, "rb") as fh:
            n += len(fh.read())
    return n
raw_read()                     # warm the page cache
t = time.perf_counter(); raw_read(); t_warm = time.perf_counter() - t
print(f"\n  raw byte read, page-cached : {t_warm:6.2f}s  "
      f"({total_bytes/2**20/t_warm:.0f} MiB/s)")

# 2. FULL VTK parse of the same files
import vtk
from vtk.util.numpy_support import vtk_to_numpy
def vtk_parse():
    npts = 0
    for f in files:
        r = vtk.vtkXMLUnstructuredGridReader()
        r.SetFileName(str(f)); r.Update()
        g = r.GetOutput()
        npts += g.GetNumberOfPoints()
        vtk_to_numpy(g.GetPoints().GetData())
        vtk_to_numpy(g.GetPointData().GetArray("Displacement"))
    return npts
vtk_parse()
t = time.perf_counter(); n = vtk_parse(); t_parse = time.perf_counter() - t
print(f"  full VTK parse, page-cached: {t_parse:6.2f}s  "
      f"({n:,} points)")
print()
print(f"  => parse / raw-read ratio: {t_parse/t_warm:.1f}x")
if t_parse > 4 * t_warm:
    print("  => DOMINATED BY CPU PARSING, not disk. Multi-core helps;")
    print("     more I/O threads alone would not.")
else:
    print("  => I/O is a large share. Overlapping reads would help.")
