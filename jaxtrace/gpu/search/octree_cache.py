"""Disk cache for the built mesh-aligned octree.

The octree build is the single largest cost of a short run: measured at
963.5 s of a 1064.6 s build stage on D2 (10.8 M elements), against 263 s of
tracking for 600 steps. And it is *reproducible* — the mesh is static across
every loaded timestep for all 20 PinShapes cases (certified, see
docs/dfamily/08_mesh_is_static_all_cases.md), so for a given case and a given
set of build options the octree is byte-identical on every run.

This caches it. The first run of a case pays the build; every later run loads
arrays from disk in seconds.

CORRECTNESS IS THE WHOLE PROBLEM HERE. A stale cache would silently feed a
wrong search structure into tracking, which is exactly the class of defect
this investigation spent weeks finding. So the key covers everything that can
change the result:

  * the mesh itself — hashes of the node_positions and connectivity ARRAYS,
    not the file path or mtime. A regenerated mesh with identical geometry
    legitimately reuses the cache; a changed mesh cannot, however it was
    produced.
  * every build option — tolerance, orphan_fallback, hybrid_non_kuhn,
    exact_tet_box, and the registration method.
  * THE BUILD SOURCE CODE — a hash of the modules that produce the octree.
    Edit any of them and every cache entry for every mesh is invalidated
    automatically. This is what makes the cache safe to leave enabled while
    the build logic is under development.

If any of those differ, the key differs and the build runs normally. There is
no partial or fuzzy matching: either the key matches exactly or it rebuilds.

ON-DISK FORMAT: one directory per entry, one ``.npy`` per array, opened with
``mmap_mode="r"``. Measured on D2-scale arrays (0.52 GiB, 85 M CSR entries):

    npz, uncompressed   write 0.31 s   read 0.45 s     0.52 GiB   no mmap
    npz, compressed     write 21.08 s  read 2.77 s     0.44 GiB   no mmap
    .npy dir, mmap      write 0.22 s   read 0.0103 s   0.52 GiB   mmap
    .npy dir, full read write 0.22 s   read 0.29 s     0.52 GiB

``.npz`` is a zip container, so its members cannot be memory-mapped and every
array is decompressed or copied on open. Separate ``.npy`` files can be
mapped, which makes opening the cache 43x faster than the uncompressed npz and
effectively free. Compression is counterproductive here: 21 s to write and 6x
slower to read, for 15% of space on a scratch filesystem.

Mapping is the right choice even though the arrays end up on the GPU.
``upload_mesh_aligned_octree_to_gpu`` passes each one to ``jnp.array()``,
which copies host to device; a mapped array satisfies that read directly and
the OS pages it in during the transfer, so nothing is read twice and no
intermediate host copy of the whole structure is materialised.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

# Modules whose contents can change the octree. Hashed into every cache key,
# so editing any of them invalidates the cache without anyone remembering to.
_BUILD_SOURCES = (
    "mesh_aligned_octree_parent_cube.py",
    "mesh_aligned_octree_single_cell.py",
    "mesh_aligned_octree_vertex_multi.py",
    "mesh_aligned_octree_aabb.py",
)

# Scalar (non-array) fields of the octree NamedTuple, stored alongside the
# arrays and restored as Python scalars.
_SCALARS = (
    "n_cells", "n_elements", "cells_per_element_mean",
    "elements_per_cell_mean", "max_elements_per_cell",
    "n_non_kuhn", "n_non_kuhn_registered",
)


def _hash_build_sources() -> str:
    """Hash the source of every module that can affect the built octree."""
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in sorted(_BUILD_SOURCES):
        p = here / name
        # A missing module is itself part of the identity: record the name
        # either way so adding or removing one changes the key.
        h.update(name.encode())
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def compute_cache_key(node_positions: np.ndarray,
                      connectivity: np.ndarray,
                      **options) -> str:
    """A key that changes if the mesh, the options, or the build code change.

    The arrays are hashed by content rather than by file identity, so a mesh
    rewritten with identical geometry reuses the cache and a mesh with any
    difference does not.
    """
    h = hashlib.sha256()
    for arr in (node_positions, connectivity):
        a = np.ascontiguousarray(arr)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    for k in sorted(options):
        h.update(f"{k}={options[k]!r}".encode())
    h.update(_hash_build_sources().encode())
    return h.hexdigest()[:32]


def cache_path(cache_dir: str | os.PathLike, key: str) -> Path:
    """Directory holding one entry: a meta.json plus one .npy per array."""
    return Path(cache_dir) / f"octree_{key}"


def load(cache_dir: str | os.PathLike, key: str, cls,
         verbose: bool = True, mmap: bool = True):
    """Return a cached octree, or None if there is no usable entry.

    Arrays are memory-mapped by default, which makes opening the cache
    effectively free; they are read for real when uploaded to the GPU.

    Any failure to read is treated as a miss rather than an error: a corrupt
    or truncated entry (an interrupted write, a full disk) must cause a
    rebuild, never a crash and never a partial structure.
    """
    d = cache_path(cache_dir, key)
    meta_p = d / "meta.json"
    if not meta_p.exists():
        return None
    try:
        t0 = time.time()
        meta = json.loads(meta_p.read_text())
        if meta.get("key") != key:                 # paranoia: name vs content
            raise ValueError("meta key does not match directory")
        fields = {}
        total = 0
        for name in cls._fields:
            if name in _SCALARS:
                fields[name] = meta["scalars"][name]
            else:
                f = d / f"{name}.npy"
                fields[name] = np.load(f, mmap_mode="r" if mmap else None,
                                       allow_pickle=False)
                total += f.stat().st_size
        obj = cls(**fields)
        if verbose:
            print(f"  [octree-cache] HIT {d.name} "
                  f"({total / 2 ** 20:.0f} MiB "
                  f"{'mapped' if mmap else 'read'} in "
                  f"{time.time() - t0:.2f}s): {obj.n_cells:,} cells, "
                  f"{obj.elements_per_cell_mean:.2f} elem/cell "
                  f"(max {obj.max_elements_per_cell})")
        return obj
    except Exception as exc:                                   # noqa: BLE001
        if verbose:
            print(f"  [octree-cache] unusable ({type(exc).__name__}: {exc}); "
                  f"rebuilding")
        return None


def store(cache_dir: str | os.PathLike, key: str, octree,
          verbose: bool = True) -> bool:
    """Write an octree to the cache. Returns True on success.

    Written to a temporary directory and renamed into place, so an interrupted
    write cannot leave a half-complete entry that a later run would try to
    use. meta.json is written last and is what `load` looks for, so an entry
    without it is never read. A failure to cache is reported and ignored: it
    must never fail the run.
    """
    try:
        final = cache_path(cache_dir, key)
        tmp = final.with_name(final.name + f".tmp{os.getpid()}")
        tmp.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        total = 0
        scalars = {}
        for name in octree._fields:
            v = getattr(octree, name)
            if name in _SCALARS:
                scalars[name] = (float(v) if "mean" in name else int(v))
            else:
                f = tmp / f"{name}.npy"
                np.save(f, np.ascontiguousarray(v), allow_pickle=False)
                total += f.stat().st_size
        # Last, so a partially written entry has no meta.json and is ignored.
        (tmp / "meta.json").write_text(json.dumps(
            {"key": key, "scalars": scalars,
             "build_sources": _hash_build_sources()}, indent=1))
        if final.exists():
            import shutil
            shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        if verbose:
            print(f"  [octree-cache] stored {final.name} "
                  f"({total / 2 ** 20:.0f} MiB in {time.time() - t0:.2f}s)")
        return True
    except Exception as exc:                                   # noqa: BLE001
        if verbose:
            print(f"  [octree-cache] could not store "
                  f"({type(exc).__name__}: {exc}); continuing")
        try:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:                                      # noqa: BLE001
            pass
        return False
