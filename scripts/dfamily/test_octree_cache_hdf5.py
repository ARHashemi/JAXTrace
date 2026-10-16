"""Verify the single-file HDF5 octree cache against the .npy-directory one.

Checks fidelity (every field byte-identical), that mapping is genuinely used
(not a silent eager read), cross-format reading, and that a corrupt entry
degrades to a rebuild rather than returning a wrong structure.
"""
import json, os, shutil, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from jaxtrace.gpu.search import octree_cache as oc

PASS = FAIL = 0
def check(name, cond, extra=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  PASS  {name}" + (f"  {extra}" if extra else ""))
    else:    FAIL += 1; print(f"  FAIL  {name}" + (f"  {extra}" if extra else ""))

# A stand-in with the same shape as the real structure: a NamedTuple mixing
# arrays, ints, floats, strs and bools, including the awkward cases.
from typing import NamedTuple, Any
class FakeOctree(NamedTuple):
    cell_morton_codes: Any
    cell_levels: Any
    cell_to_elements_offsets: Any
    cell_to_elements_data: Any
    bounds_min: Any
    empty_array: Any            # zero-length: no HDF5 offset, must fall back
    n_cells: int
    max_elements_per_cell: int
    elements_per_cell_mean: float
    method: str
    hybrid: bool

def make(n_cells=5000, n_csr=90000, seed=0):
    r = np.random.default_rng(seed)
    return FakeOctree(
        cell_morton_codes=r.integers(0, 2**62, n_cells, dtype=np.int64),
        cell_levels=r.integers(0, 21, n_cells).astype(np.int8),
        cell_to_elements_offsets=np.sort(r.integers(0, n_csr, n_cells+1)).astype(np.int64),
        cell_to_elements_data=r.integers(0, 10_000_000, n_csr, dtype=np.int32),
        bounds_min=r.random((3,)).astype(np.float32),
        empty_array=np.zeros(0, dtype=np.int32),
        n_cells=n_cells, max_elements_per_cell=65,
        elements_per_cell_mean=9.217, method="parent_cube", hybrid=True)

def same(a, b):
    for f in a._fields:
        x, y = getattr(a, f), getattr(b, f)
        if oc._is_array(x) or oc._is_array(y):
            x, y = np.asarray(x), np.asarray(y)
            if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
                return False, f"{f}: {x.dtype}{x.shape} vs {y.dtype}{y.shape}"
        else:
            if type(x) is not type(y) or x != y:
                return False, f"{f}: {x!r} ({type(x).__name__}) vs {y!r} ({type(y).__name__})"
    return True, ""

T = tempfile.mkdtemp()
try:
    oct_in = make()
    KEY = "testkey" + "0" * 25

    print("1. HDF5 round-trip fidelity")
    check("store(hdf5) returns True", oc.store(T, KEY, oct_in, verbose=False, format="hdf5"))
    p = oc.cache_path(T, KEY, "hdf5")
    check("a single file was written", p.is_file(), f"{p.name}, {p.stat().st_size/2**20:.1f} MiB")
    got = oc.load(T, KEY, FakeOctree, verbose=False)
    check("load returns a structure", got is not None)
    ok, why = same(oct_in, got); check("every field byte-identical", ok, why)

    print("\n2. mapping is genuinely used (not an eager read)")
    check("a big array is a memmap", isinstance(got.cell_to_elements_data, np.memmap),
          type(got.cell_to_elements_data).__name__)
    check("empty array falls back to ndarray", not isinstance(got.empty_array, np.memmap)
          and got.empty_array.size == 0)
    eager = oc.load(T, KEY, FakeOctree, verbose=False, mmap=False)
    check("mmap=False gives a real ndarray", not isinstance(eager.cell_to_elements_data, np.memmap))
    ok, why = same(oct_in, eager); check("eager read also byte-identical", ok, why)

    print("\n3. scalar types survive")
    for f, t in [("n_cells", int), ("elements_per_cell_mean", float),
                 ("method", str), ("hybrid", bool)]:
        v = getattr(got, f)
        check(f"{f} is {t.__name__}", type(v) is t, f"got {type(v).__name__} = {v!r}")

    print("\n4. directory format still works, and both are readable")
    K2 = "dirkey" + "0" * 26
    check("store(dir) works", oc.store(T, K2, oct_in, verbose=False, format="dir"))
    check("a directory was written", oc.cache_path(T, K2, "dir").is_dir())
    g2 = oc.load(T, K2, FakeOctree, verbose=False)
    ok, why = same(oct_in, g2); check("dir entry byte-identical", ok, why)
    check("dir arrays are mapped too", isinstance(g2.cell_to_elements_data, np.memmap))
    # the reader must find each format without being told which
    check("reader sniffs hdf5", oc._existing_entry(T, KEY)[1] == "hdf5")
    check("reader sniffs dir",  oc._existing_entry(T, K2)[1] == "dir")

    print("\n5. a wrong key is a miss, not a wrong answer")
    check("absent key -> None", oc.load(T, "nope" + "0"*28, FakeOctree, verbose=False) is None)

    print("\n6. corruption degrades to a rebuild")
    K3 = "corrupt" + "0"*25
    oc.store(T, K3, oct_in, verbose=False, format="hdf5")
    pc = oc.cache_path(T, K3, "hdf5")
    with open(pc, "r+b") as f:
        f.truncate(pc.stat().st_size // 3)
    check("truncated .h5 -> None", oc.load(T, K3, FakeOctree, verbose=False) is None)
    K4 = "garbage" + "0"*25
    oc.cache_path(T, K4, "hdf5").write_bytes(b"not an hdf5 file at all")
    check("non-HDF5 file -> None", oc.load(T, K4, FakeOctree, verbose=False) is None)

    print("\n7. key mismatch inside the file is caught")
    K5 = "mismatch" + "0"*24
    oc.store(T, K5, oct_in, verbose=False, format="hdf5")
    import h5py
    with h5py.File(oc.cache_path(T, K5, "hdf5"), "r+") as h:
        h.attrs["key"] = "a-different-key"
    check("stored-key mismatch -> None", oc.load(T, K5, FakeOctree, verbose=False) is None)

    print("\n8. overwriting an entry in the other format")
    # a dir entry already exists at K2; writing hdf5 for the same key must not
    # leave the reader seeing a stale directory
    oc.store(T, K2, make(seed=9), verbose=False, format="hdf5")
    check("hdf5 now preferred for that key", oc._existing_entry(T, K2)[1] == "hdf5")
    g3 = oc.load(T, K2, FakeOctree, verbose=False)
    ok, _ = same(make(seed=9), g3); check("reads the NEW content", ok)

    print("\n9. datasets are contiguous (the property mapping depends on)")
    with h5py.File(oc.cache_path(T, KEY, "hdf5"), "r") as h:
        noff = {k: h[k].id.get_offset() for k in h.keys()}
    nonctg = [k for k, v in noff.items() if v is None and k != "empty_array"]
    check("every non-empty dataset has an offset", not nonctg, str(nonctg))

finally:
    shutil.rmtree(T, ignore_errors=True)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
