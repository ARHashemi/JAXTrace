from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import vmap

from .base import (
    barycentric_coords_triangle,
    barycentric_coords_tetra,
    tri6_shape,
    tet10_shape,
)


class ElementType(str, Enum):
    triangle = "triangle"
    tetra = "tetra"


@dataclass
class UnstructuredMesh:
    """
    Unstructured mesh data.

    nodes: (Pn,3)
    elements: indices array of shape (M,k)
      - k=3 for tri3, k=6 for tri6
      - k=4 for tet4, k=10 for tet10
    etype: triangle or tetra
    order: 1 (linear) or 2 (quadratic)
    """
    nodes: jnp.ndarray
    elements: Optional[jnp.ndarray]
    etype: ElementType
    order: int = 1


def _shape_linear_triangle(lmbda: jnp.ndarray):
    return lmbda  # (3,)


def _shape_linear_tetra(lmbda: jnp.ndarray):
    return lmbda  # (4,)


@jax.jit
def _interpolate_on_tri3(p: jnp.ndarray, tri_nodes: jnp.ndarray, nodal_vals: jnp.ndarray):
    l = barycentric_coords_triangle(p, tri_nodes[0], tri_nodes[1], tri_nodes[2])
    N = _shape_linear_triangle(l)  # (3,)
    return (N[:, None] * nodal_vals).sum(axis=0)


@jax.jit
def _interpolate_on_tri6(p: jnp.ndarray, tri6_nodes: jnp.ndarray, nodal_vals: jnp.ndarray):
    # vertices first: 0..2, edges: 3..5
    l = barycentric_coords_triangle(p, tri6_nodes[0], tri6_nodes[1], tri6_nodes[2])
    Nq = tri6_shape(l)  # (6,)
    return (Nq[:, None] * nodal_vals).sum(axis=0)


@jax.jit
def _interpolate_on_tet4(p: jnp.ndarray, tet_nodes: jnp.ndarray, nodal_vals: jnp.ndarray):
    l = barycentric_coords_tetra(p, tet_nodes[0], tet_nodes[1], tet_nodes[2], tet_nodes[3])
    N = _shape_linear_tetra(l)  # (4,)
    return (N[:, None] * nodal_vals).sum(axis=0)


@jax.jit
def _interpolate_on_tet10(p: jnp.ndarray, tet10_nodes: jnp.ndarray, nodal_vals: jnp.ndarray):
    # vertices 0..3, edges 4..9
    l = barycentric_coords_tetra(p, tet10_nodes[0], tet10_nodes[1], tet10_nodes[2], tet10_nodes[3])
    Nq = tet10_shape(l)  # (10,)
    return (Nq[:, None] * nodal_vals).sum(axis=0)


def _euclidean_sq(a: jnp.ndarray, b: jnp.ndarray):
    d = a - b
    return jnp.dot(d, d)


@jax.jit
def _knn_indices(x: jnp.ndarray, nodes: jnp.ndarray, k: int):
    # O(Pn) scan; for very large meshes, prefer a grid-hash neighbor search placed in utils
    d2 = jnp.sum((nodes - x[None, :]) ** 2, axis=1)
    idx = jnp.argsort(d2)[:k]
    return idx


@dataclass
class UnstructuredField:
    """
    Spatial sampler on an unstructured mesh with optional connectivity.
    If connectivity is absent, uses kNN as pseudo-element nodes, consistent with
    prior nearest-neighbor FE-style interpolation logic[^14,^9,^12].
    """
    mesh: UnstructuredMesh
    # If connectivity is None, use kNN fallback:
    knn_k: Optional[int] = None      # defaults: 3 for tri, 4 for tet, 6/10 for quad
    strict_inside: bool = False      # if True, reject negative barycentric and fallback

    def _default_k(self):
        if self.mesh.order == 1:
            return 3 if self.mesh.etype == ElementType.triangle else 4
        else:
            return 6 if self.mesh.etype == ElementType.triangle else 10

    def _interpolate_single(self, x: jnp.ndarray, nodal_values: jnp.ndarray) -> jnp.ndarray:
        nodes = self.mesh.nodes
        elements = self.mesh.elements
        etype = self.mesh.etype
        order = self.mesh.order
        k = self.knn_k or self._default_k()

        if elements is not None:
            # Simple strategy: choose the element whose centroid is closest to x.
            # More advanced point-in-element queries can be added later.
            elem_nodes = nodes[elements]  # (M,k,3)
            centroids = elem_nodes.mean(axis=1)  # (M,3)
            d2 = jnp.sum((centroids - x[None, :]) ** 2, axis=1)
            e_idx = jnp.argmin(d2)
            enodes = elem_nodes[e_idx]         # (k,3)
            evals = nodal_values[elements[e_idx]]  # (k,C)
        else:
            # kNN fallback: use nearest k nodes as element nodes
            idx = _knn_indices(x, nodes, k)
            enodes = nodes[idx]
            evals = nodal_values[idx]

        if etype == ElementType.triangle:
            if order == 1:
                return _interpolate_on_tri3(x, enodes[:3], evals[:3])
            else:
                # tri6 expects 6 nodes: v1,v2,v3,e12,e23,e31. If we don't have mid-edge nodes,
                # approximate by repeating nearest vertices for mid-edge slots.
                if enodes.shape[0] < 6:
                    # approximate mid-edge nodes by midpoints of vertices
                    v = enodes[:3]
                    mids = jnp.stack([(v[0] + v[1]) * 0.5, (v[1] + v[2]) * 0.5, (v[2] + v[0]) * 0.5], axis=0)
                    enodes_q = jnp.concatenate([v, mids], axis=0)
                    evals_q = jnp.concatenate([evals[:3],
                                               0.5 * (evals[0:1] + evals[1:2]),
                                               0.5 * (evals[1:2] + evals[2:3]),
                                               0.5 * (evals[2:3] + evals[0:1])], axis=0)
                else:
                    enodes_q = enodes[:6]
                    evals_q = evals[:6]
                return _interpolate_on_tri6(x, enodes_q, evals_q)

        else:  # tetra
            if order == 1:
                return _interpolate_on_tet4(x, enodes[:4], evals[:4])
            else:
                # tet10 expects 10 nodes: v1..v4 + 6 edge midpoints
                if enodes.shape[0] < 10:
                    v = enodes[:4]
                    # construct 6 mids: (12,23,31,14,24,34)
                    mids = jnp.stack([
                        0.5 * (v[0] + v[1]),
                        0.5 * (v[1] + v[2]),
                        0.5 * (v[2] + v[0]),
                        0.5 * (v[0] + v[3]),
                        0.5 * (v[1] + v[3]),
                        0.5 * (v[2] + v[3]),
                    ], axis=0)
                    enodes_q = jnp.concatenate([v, mids], axis=0)
                    ev = evals[:4]
                    mids_vals = jnp.stack([
                        0.5 * (ev[0] + ev[1]),
                        0.5 * (ev[1] + ev[2]),
                        0.5 * (ev[2] + ev[0]),
                        0.5 * (ev[0] + ev[3]),
                        0.5 * (ev[1] + ev[3]),
                        0.5 * (ev[2] + ev[3]),
                    ], axis=0)
                    evals_q = jnp.concatenate([ev, mids_vals], axis=0)
                else:
                    enodes_q = enodes[:10]
                    evals_q = evals[:10]
                return _interpolate_on_tet10(x, enodes_q, evals_q)

    def sample_given_values(self, x: jnp.ndarray, nodal_values: jnp.ndarray) -> jnp.ndarray:
        """
        x: (N,3)
        nodal_values: (Pn,C) values on nodes for a single time slice
        returns: (N,C)
        """
        fn = lambda xi: self._interpolate_single(xi, nodal_values)
        return vmap(fn)(x)
