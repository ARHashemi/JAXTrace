# jaxtrace/density/inside_mesh.py
"""
Voxel-inside-mesh masking using the mesh-aligned octree from
``jaxtrace.gpu.search.mesh_aligned_point_location``.

A voxel center is considered "inside" the velocity mesh iff a containing
tetrahedral element is found by the 3x3x3 local search. This mask is
precomputed once per run and reused every step.

We deliberately use the ``_where`` variant of the search to match the
correctness guarantees used by the RK4 fully-fused integrator (avoiding
deeply-nested ``lax.cond`` under ``vmap``).
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..gpu.search.mesh_aligned_point_location import (
    search_mesh_aligned_octree_multi_local_where,
)


def compute_inside_mesh_mask(
    voxel_centers: jnp.ndarray,            # (M, 3) device float32
    mesh_octree_gpu,                       # MeshAlignedOctreeGPU
    *,
    chunk: int = 65_536,
    max_tests: int = 600,
) -> jnp.ndarray:
    """
    Return a (M,) boolean device array, True where the voxel center is
    inside a tet of the velocity mesh.

    Chunked over query points to keep the per-call compile / launch grid
    reasonable. The jitted function is shape-stable so it compiles once.
    """
    M = int(voxel_centers.shape[0])

    @jax.jit
    def _search_chunk(Q: jnp.ndarray) -> jnp.ndarray:
        # Q: (chunk, 3); returns boolean (chunk,) where elem_id >= 0.
        def per_point(pos):
            elem_id, _n_tests = search_mesh_aligned_octree_multi_local_where(
                pos, mesh_octree_gpu, jnp.int32(max_tests),
            )
            return elem_id >= 0

        return jax.vmap(per_point)(Q)

    out_chunks = []
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        block = voxel_centers[s:e]
        # Pad the last chunk to a fixed size so we only compile once.
        if block.shape[0] < chunk:
            pad = chunk - block.shape[0]
            block_pad = jnp.concatenate(
                [block, jnp.zeros((pad, 3), dtype=block.dtype)], axis=0,
            )
            mask = _search_chunk(block_pad)[:block.shape[0]]
        else:
            mask = _search_chunk(block)
        out_chunks.append(mask)

    return jnp.concatenate(out_chunks, axis=0)


def inside_mask_to_3d(
    mask_flat: jnp.ndarray, resolution: tuple[int, int, int],
) -> jnp.ndarray:
    """Reshape a flat (M,) mask back to (Nx, Ny, Nz)."""
    nx, ny, nz = resolution
    return mask_flat.reshape((nx, ny, nz))
