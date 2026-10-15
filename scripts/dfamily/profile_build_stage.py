"""Where do the 1,094 s of 3_build_upload actually go?

Times each major sub-step of the build on the real D2 mesh, so the
optimisation document ranks measured costs rather than guesses.
"""
import os, sys, time
import numpy as np
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "/flash/shared/jax/JAXTrace")
from pathlib import Path
import jax.numpy as jnp
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import (
    extract_octree_cells_parent_cube)
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import (
    upload_mesh_aligned_octree_to_gpu)
from jaxtrace.gpu.forest import build_element_neighbors_array

post = Path("/scratch/shared/PinShapes/D-ConcavityTilt/D2.gid/post")

t = time.perf_counter()
npos, conn, vel = load_velocity_sequence_from_pvtu(
    base_path=post, file_pattern="A2_{timestep}.pvtu", timestep_range=(34, 34),
    field_name="Displacement", verbose=False)
t_load = time.perf_counter() - t
print(f"  load 1 snapshot                : {t_load:8.1f}s", flush=True)

t = time.perf_counter()
npos, conn, _n, vel = deduplicate_nodes(npos, conn, velocity_sequence=vel,
                                        verbose=False)
conn = conn.astype(np.int32)
t_dedup = time.perf_counter() - t
print(f"  deduplicate_nodes              : {t_dedup:8.1f}s", flush=True)

t = time.perf_counter()
cells = extract_octree_cells_parent_cube(npos, conn, tolerance=1e-6,
                                         verbose=False)
t_oct = time.perf_counter() - t
print(f"  extract_octree_cells_parent_cube: {t_oct:8.1f}s   <- pure Python",
      flush=True)

t = time.perf_counter()
g = upload_mesh_aligned_octree_to_gpu(conn, npos, cells, verbose=False)
t_up = time.perf_counter() - t
print(f"  upload_mesh_aligned_octree      : {t_up:8.1f}s", flush=True)

t = time.perf_counter()
nbrs = build_element_neighbors_array(conn, method='face', verbose=False)
t_nbr = time.perf_counter() - t
print(f"  build_element_neighbors_array   : {t_nbr:8.1f}s", flush=True)

tot = t_oct + t_up + t_nbr
print()
print(f"  build-stage subtotal            : {tot:8.1f}s")
for lbl, v in (("octree extract", t_oct), ("gpu upload", t_up),
               ("neighbours", t_nbr)):
    print(f"    {lbl:22}: {100*v/tot:5.1f}%")
print()
print(f"  mesh: {len(npos):,} nodes, {len(conn):,} elements, "
      f"{cells.n_cells:,} cells")
