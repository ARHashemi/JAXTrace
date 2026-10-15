"""How much does parallelising the VTK parse actually buy?

Parsing is 245x the raw read cost, so it is CPU-bound and splits across cores.
Two mechanisms are possible:
  threads  -- only helps if the VTK reader releases the GIL during parsing
  processes -- always parallel, but pays pickling to return the arrays
Both measured, against the serial baseline.
"""
import os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import xml.etree.ElementTree as ET
import numpy as np

post = Path("/scratch/shared/PinShapes/D-ConcavityTilt/D2.gid/post")
pv = post / "A2_41.pvtu"
srcs = [p.get("Source") for p in ET.parse(pv).getroot().iter("Piece")]
FILES = [str(post / s) for s in srcs]
print(f"{len(FILES)} piece files, {os.cpu_count()} cores visible")

def parse_one(path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(path); r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    dsp = vtk_to_numpy(g.GetPointData().GetArray("Displacement"))
    return pts.shape[0], float(pts[0, 0]), float(dsp[0, 0])

def serial():
    return [parse_one(f) for f in FILES]

def threaded(n):
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(parse_one, FILES))

def procs(n):
    with ProcessPoolExecutor(max_workers=n) as ex:
        return list(ex.map(parse_one, FILES, chunksize=4))

ref = serial()                     # warm cache + reference values
t = time.perf_counter(); r0 = serial(); t0 = time.perf_counter() - t
print(f"\n  serial            : {t0:6.2f}s")

for n in (4, 8, 16):
    t = time.perf_counter(); r = threaded(n); dt = time.perf_counter() - t
    ok = r == ref
    print(f"  threads={n:<3}       : {dt:6.2f}s  {t0/dt:5.2f}x  identical={ok}")

for n in (4, 8, 16):
    t = time.perf_counter(); r = procs(n); dt = time.perf_counter() - t
    ok = r == ref
    print(f"  processes={n:<3}     : {dt:6.2f}s  {t0/dt:5.2f}x  identical={ok}")
