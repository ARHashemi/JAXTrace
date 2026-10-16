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

SINGLE-FILE CONTAINER (``format="hdf5"``): one ``.h5`` per entry instead of a
directory, which is tidier to copy, archive and keep beside a case. It costs
nothing, because mapping is preserved: an HDF5 dataset written *contiguous and
uncompressed* is a flat byte range in the file, so its offset
(``dset.id.get_offset()``) can be handed to ``np.memmap`` directly. h5py has
no mmap of its own; this is the same technique vaex uses. Measured through this
module's own API at D2 scale (0.275 GiB, 5.48 M cells, 50.4 M CSR entries),
median of five opens:

    .npy dir + mmap          write 0.12 s   open 0.41 ms   1 dir, 5 files
    hdf5 + np.memmap offset  write 0.08 s   open 0.77 ms   1 file

HDF5 is ~0.36 ms slower to open, h5py's library-open overhead, against a
963 s build. Both are free; the container is chosen for tidiness, not speed.

What the mapping buys is the thing to protect. Reading the same HDF5 eagerly
costs 0.41 s to open -- 375x more, exactly like npz, because every byte is
copied. Compression is therefore deliberately NOT used: it would make the
datasets non-contiguous, remove the offset and silently force that path.

Zarr was considered and rejected: its advantages are cloud object stores and
parallel writes from many workers, neither of which applies to a single
process writing one local file on Lustre.

Both formats remain supported and are chosen per call. The reader sniffs what
is actually on disk, so entries written by either format are readable
regardless of the current default, and a mixed cache directory is fine.
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

# Fields are classified by VALUE, not by name: anything that is not an array
# is stored in meta.json with its Python type recorded, everything else as a
# .npy. Nothing here knows what a field MEANS, so adding, renaming or removing
# a field in the octree structure needs no change in this module -- and any
# such edit changes the build-source hash anyway, so old entries are
# invalidated rather than misread.
_SCALAR_TYPES = {"int": int, "float": float, "bool": bool, "str": str}


def _is_array(v) -> bool:
    return isinstance(v, np.ndarray) or hasattr(v, "__array_interface__")


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


def cache_path(cache_dir: str | os.PathLike, key: str,
               format: str = "dir") -> Path:
    """Where one entry lives.

    ``format="dir"``  -> a directory: meta.json plus one .npy per array.
    ``format="hdf5"`` -> a single .h5 file holding every array and scalar.
    """
    base = Path(cache_dir) / f"octree_{key}"
    return base.with_suffix(".h5") if format == "hdf5" else base


def _existing_entry(cache_dir: str | os.PathLike, key: str):
    """Find an entry on disk whatever format wrote it.

    Returns (path, format) or (None, None). Checking the filesystem rather
    than trusting the caller's current default means a cache written before a
    format change is still a hit, and a directory holding both kinds is fine.
    The single file is preferred when both exist, as it is the tidier one to
    have been written deliberately.
    """
    h5 = cache_path(cache_dir, key, "hdf5")
    if h5.is_file():
        return h5, "hdf5"
    d = cache_path(cache_dir, key, "dir")
    if (d / "meta.json").exists():
        return d, "dir"
    return None, None


def _h5_read_array(h5path: Path, dset, mmap: bool):
    """One dataset as a numpy array, mapped when that is possible.

    A contiguous, uncompressed HDF5 dataset is a flat byte range, so
    ``np.memmap`` can address it via the stored offset -- the same trick vaex
    uses, since h5py exposes no mmap. ``get_offset()`` returns None for a
    chunked, compressed or empty dataset, and for those we read eagerly. The
    fallback is correctness, not a slow path we expect to hit: `store` writes
    everything contiguous on purpose.
    """
    if mmap and dset.size > 0:
        off = dset.id.get_offset()
        if off is not None:
            return np.memmap(h5path, dtype=dset.dtype, mode="r",
                             offset=off, shape=dset.shape)
    return dset[...]


def _load_hdf5(path: Path, key: str, cls, mmap: bool):
    """Read an entry from a single .h5. Shape mirrors the directory reader."""
    import h5py
    fields, total = {}, 0
    with h5py.File(path, "r") as h:
        if h.attrs.get("key") != key:
            raise ValueError("stored key does not match the requested one")
        scalars = json.loads(h.attrs["scalars"])
        for name in cls._fields:
            if name in scalars:
                raw, tname = scalars[name]
                fields[name] = _SCALAR_TYPES.get(tname, lambda x: x)(raw)
            else:
                dset = h[name]
                total += dset.nbytes
                fields[name] = _h5_read_array(path, dset, mmap)
    return fields, total


def _load_dir(d: Path, key: str, cls, mmap: bool):
    """Read an entry from a directory of .npy files."""
    meta = json.loads((d / "meta.json").read_text())
    if meta.get("key") != key:                     # paranoia: name vs content
        raise ValueError("meta key does not match directory")
    fields, total = {}, 0
    scalars = meta.get("scalars", {})
    for name in cls._fields:
        if name in scalars:
            raw, tname = scalars[name]
            fields[name] = _SCALAR_TYPES.get(tname, lambda x: x)(raw)
        else:
            f = d / f"{name}.npy"
            fields[name] = np.load(f, mmap_mode="r" if mmap else None,
                                   allow_pickle=False)
            total += f.stat().st_size
    return fields, total


def load(cache_dir: str | os.PathLike, key: str, cls,
         verbose: bool = True, mmap: bool = True):
    """Return a cached octree, or None if there is no usable entry.

    Arrays are memory-mapped by default, which makes opening the cache
    effectively free; they are read for real when uploaded to the GPU.

    Any failure to read is treated as a miss rather than an error: a corrupt
    or truncated entry (an interrupted write, a full disk) must cause a
    rebuild, never a crash and never a partial structure.
    """
    path, fmt = _existing_entry(cache_dir, key)
    if path is None:
        return None
    try:
        t0 = time.time()
        if fmt == "hdf5":
            fields, total = _load_hdf5(path, key, cls, mmap)
        else:
            fields, total = _load_dir(path, key, cls, mmap)
        obj = cls(**fields)
        if verbose:
            print(f"  [octree-cache] HIT {path.name} "
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
          verbose: bool = True, format: str = "hdf5") -> bool:
    """Write an octree to the cache. Returns True on success.

    ``format="hdf5"`` (the default) writes one self-contained .h5; "dir"
    writes the original directory of .npy files. Both are readable by `load`
    whatever the current default, so changing this does not strand an entry.

    Written to a temporary path and renamed into place, so an interrupted
    write cannot leave a half-complete entry that a later run would try to
    use. For the directory format meta.json is written last and is what
    `load` looks for; for HDF5 the rename itself is the commit, since a
    half-written .h5 never appears at the final name. A failure to cache is
    reported and ignored: it must never fail the run.

    Every dataset is written CONTIGUOUS and UNCOMPRESSED on purpose. That is
    what gives each one a file offset and so keeps it memory-mappable on
    read; chunking or compressing would silently force the eager read path
    and make opening the cache ~375x slower.
    """
    tmp = None
    try:
        final = cache_path(cache_dir, key, format)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_name(final.name + f".tmp{os.getpid()}")
        t0 = time.time()
        total = 0
        scalars = {}

        if format == "hdf5":
            import h5py
            with h5py.File(tmp, "w") as h:
                for name in octree._fields:
                    v = getattr(octree, name)
                    if _is_array(v):
                        a = np.ascontiguousarray(v)
                        # chunks=None + compression=None => contiguous, which
                        # is what makes the dataset mappable on read.
                        h.create_dataset(name, data=a,
                                         chunks=None, compression=None)
                        total += a.nbytes
                    else:
                        scalars[name] = [v, type(v).__name__]
                h.attrs["key"] = key
                h.attrs["scalars"] = json.dumps(scalars)
                h.attrs["build_sources"] = _hash_build_sources()
        else:
            tmp.mkdir(parents=True, exist_ok=True)
            for name in octree._fields:
                v = getattr(octree, name)
                if _is_array(v):
                    f = tmp / f"{name}.npy"
                    np.save(f, np.ascontiguousarray(v), allow_pickle=False)
                    total += f.stat().st_size
                else:
                    # Record the value AND its type, so it is restored as the
                    # same Python type without this module knowing the
                    # field's meaning.
                    scalars[name] = [v, type(v).__name__]
            # Last, so a partially written entry has no meta.json.
            (tmp / "meta.json").write_text(json.dumps(
                {"key": key, "scalars": scalars,
                 "build_sources": _hash_build_sources()}, indent=1))

        # Replace whatever is already at the final name, in either layout.
        if final.is_dir():
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
            if tmp is not None:
                if tmp.is_dir():
                    import shutil
                    shutil.rmtree(tmp, ignore_errors=True)
                elif tmp.exists():
                    tmp.unlink()
        except Exception:                                      # noqa: BLE001
            pass
        return False
