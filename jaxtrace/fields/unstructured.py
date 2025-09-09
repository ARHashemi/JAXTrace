# jaxtrace/fields/unstructured.py
"""
Unstructured mesh field interpolation with barycentric coordinates.

- Supports linear and quadratic triangular/tetrahedral elements.
- Fallback to k-nearest neighbor (kNN) interpolation when connectivity is missing.
- JAX-friendly interpolation kernels; element search remains on CPU (optional).
- Includes utilities for VTK-like data and basic precomputation.

Notes:
- When JAX is available, interpolation kernels are JIT-compiled and vectorized with vmap.
- For GPU-friendly pipelines, prefer use_element_search=False to avoid Python loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict
import numpy as np

# Import JAX utilities with fallback
from ..utils.jax_utils import JAX_AVAILABLE

if JAX_AVAILABLE:
    try:
        import jax
        import jax.numpy as jnp
        from jax import jit, vmap
    except Exception:
        JAX_AVAILABLE = False

if not JAX_AVAILABLE:
    import numpy as jnp  # type: ignore

    # Mock JAX decorators for NumPy fallback
    def jit(fn):
        return fn

    def vmap(fn):
        def wrapper(x):
            return np.asarray([fn(xi) for xi in x], dtype=np.float32)
        return wrapper

from .base import (
    barycentric_coords_triangle,
    barycentric_coords_tetrahedron,
    tri6_shape_functions,
    tet10_shape_functions,
    _ensure_float32,
    _ensure_positions_shape,
)


# ---------------------------------------------------------------------
# Element types and mesh container
# ---------------------------------------------------------------------

class ElementType(str, Enum):
    """Supported element types for unstructured meshes."""
    triangle = "triangle"
    tetra = "tetra"


@dataclass
class UnstructuredMesh:
    """
    Unstructured mesh data container.

    Supports linear and quadratic elements with optional connectivity.

    Attributes
    ----------
    nodes : np.ndarray
        Node coordinates, shape (Pn, 3), float32
    elements : np.ndarray, optional
        Element connectivity, shape (M, k) where:
        - k = 3 for tri3, k = 6 for tri6
        - k = 4 for tet4, k = 10 for tet10
    etype : ElementType
        Element type: triangle or tetra
    order : int
        Element order: 1 (linear) or 2 (quadratic)
    """
    nodes: np.ndarray
    elements: Optional[np.ndarray]
    etype: ElementType
    order: int = 1

    def __post_init__(self):
        # Ensure 3D node coordinates and float32 dtype
        self.nodes = _ensure_positions_shape(self.nodes).astype(np.float32, copy=False)
        if self.elements is not None:
            self.elements = np.asarray(self.elements, dtype=np.int32)
        self._validate_connectivity()

    def _validate_connectivity(self):
        if self.elements is None:
            return
        n_nodes = int(self.nodes.shape[0])
        expected_k = self.expected_nodes_per_element()
        if self.elements.shape[1] != expected_k:
            raise ValueError(
                f"Expected {expected_k} nodes per {self.etype.value} element (order {self.order}), "
                f"got {self.elements.shape[1]}"
            )
        if self.elements.size > 0:
            max_node_idx = int(np.max(self.elements))
            if max_node_idx >= n_nodes:
                raise ValueError(
                    f"Element references node {max_node_idx}, but only {n_nodes} nodes available"
                )

    def expected_nodes_per_element(self) -> int:
        if self.etype == ElementType.triangle:
            return 3 if self.order == 1 else 6
        else:
            return 4 if self.order == 1 else 10

    def get_spatial_bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        bounds_min = jnp.min(jnp.asarray(self.nodes, dtype=jnp.float32), axis=0)
        bounds_max = jnp.max(jnp.asarray(self.nodes, dtype=jnp.float32), axis=0)
        return bounds_min, bounds_max

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def num_elements(self) -> int:
        return int(self.elements.shape[0]) if self.elements is not None else 0


# ---------------------------------------------------------------------
# Shape functions and element interpolation (JAX-jittable)
# ---------------------------------------------------------------------

@jit
def _shape_linear_triangle(lmbda: jnp.ndarray) -> jnp.ndarray:
    """Linear triangle shape functions: identity on barycentric coords."""
    return lmbda  # (3,)


@jit
def _shape_linear_tetra(lmbda: jnp.ndarray) -> jnp.ndarray:
    """Linear tetrahedron shape functions: identity on barycentric coords."""
    return lmbda  # (4,)


@jit
def _interpolate_on_tri3(
    p: jnp.ndarray,
    tri_nodes: jnp.ndarray,
    nodal_vals: jnp.ndarray
) -> jnp.ndarray:
    l = barycentric_coords_triangle(p, tri_nodes[0], tri_nodes[1], tri_nodes[2])
    N = _shape_linear_triangle(l)  # (3,)
    return (N[:, None] * nodal_vals[:3]).sum(axis=0)


@jit
def _interpolate_on_tri6(
    p: jnp.ndarray,
    tri6_nodes: jnp.ndarray,
    nodal_vals: jnp.ndarray
) -> jnp.ndarray:
    l = barycentric_coords_triangle(p, tri6_nodes[0], tri6_nodes[1], tri6_nodes[2])
    Nq = tri6_shape_functions(l)  # (6,)
    return (Nq[:, None] * nodal_vals[:6]).sum(axis=0)


@jit
def _interpolate_on_tet4(
    p: jnp.ndarray,
    tet_nodes: jnp.ndarray,
    nodal_vals: jnp.ndarray
) -> jnp.ndarray:
    l = barycentric_coords_tetrahedron(p, tet_nodes[0], tet_nodes[1], tet_nodes[2], tet_nodes[3])
    N = _shape_linear_tetra(l)  # (4,)
    return (N[:, None] * nodal_vals[:4]).sum(axis=0)


@jit
def _interpolate_on_tet10(
    p: jnp.ndarray,
    tet10_nodes: jnp.ndarray,
    nodal_vals: jnp.ndarray
) -> jnp.ndarray:
    l = barycentric_coords_tetrahedron(p, tet10_nodes[0], tet10_nodes[1], tet10_nodes[2], tet10_nodes[3])
    Nq = tet10_shape_functions(l)  # (10,)
    return (Nq[:, None] * nodal_vals[:10]).sum(axis=0)


# ---------------------------------------------------------------------
# Neighbor search and element search (CPU element search)
# ---------------------------------------------------------------------

def _knn_indices(x: jnp.ndarray, nodes: jnp.ndarray, k: int) -> jnp.ndarray:
    """Find k nearest neighbor indices via brute-force O(P) distance."""
    d2 = jnp.sum((nodes - x[None, :]) ** 2, axis=1)
    idx = jnp.argsort(d2)[:k]
    return idx


def _find_containing_element(x: jnp.ndarray, mesh: UnstructuredMesh) -> Optional[int]:
    """
    Find element containing point x using barycentric coordinate tests.
    CPU-only (Python loop); not jittable. Used only if use_element_search=True.
    """
    if mesh.elements is None:
        return None

    nodes = mesh.nodes
    elements = mesh.elements

    if mesh.etype == ElementType.triangle:
        for elem_idx in range(mesh.num_elements):
            elem = elements[elem_idx]
            if elem.shape[0] < 3:
                continue
            enodes = nodes[elem[:3]]
            bary = barycentric_coords_triangle(
                jnp.asarray(x, dtype=jnp.float32),
                jnp.asarray(enodes[0], dtype=jnp.float32),
                jnp.asarray(enodes[1], dtype=jnp.float32),
                jnp.asarray(enodes[2], dtype=jnp.float32),
            )
            if jnp.all(bary >= -1e-6) and jnp.sum(bary) <= 1.0 + 1e-6:
                return elem_idx
    else:
        for elem_idx in range(mesh.num_elements):
            elem = elements[elem_idx]
            if elem.shape[0] < 4:
                continue
            enodes = nodes[elem[:4]]
            bary = barycentric_coords_tetrahedron(
                jnp.asarray(x, dtype=jnp.float32),
                jnp.asarray(enodes[0], dtype=jnp.float32),
                jnp.asarray(enodes[1], dtype=jnp.float32),
                jnp.asarray(enodes[2], dtype=jnp.float32),
                jnp.asarray(enodes[3], dtype=jnp.float32),
            )
            if jnp.all(bary >= -1e-6) and jnp.sum(bary) <= 1.0 + 1e-6:
                return elem_idx

    return None


# ---------------------------------------------------------------------
# Field class
# ---------------------------------------------------------------------

@dataclass
class UnstructuredField:
    """
    Spatial sampler on unstructured mesh with optional connectivity.

    If connectivity is absent, uses kNN as pseudo-element nodes.
    Supports both linear and quadratic elements.

    Parameters
    ----------
    mesh : UnstructuredMesh
        Mesh geometry and connectivity.
    knn_k : int, optional
        Number of neighbors for kNN fallback (auto-determined if None).
    strict_inside : bool
        If True and use_element_search=True: return zeros for points outside all elements.
    use_element_search : bool
        Whether to search for containing elements (slower, CPU-only),
        or use nearest-centroid element selection (JAX-friendly).
    """
    mesh: UnstructuredMesh
    knn_k: Optional[int] = None
    strict_inside: bool = False
    use_element_search: bool = False

    def __post_init__(self):
        if self.knn_k is None:
            self.knn_k = self._default_k()
        # Clamp k to reasonable upper bound
        max_k = min(self.mesh.num_nodes, 20)
        if self.knn_k > max_k:
            self.knn_k = max_k

    def _default_k(self) -> int:
        if self.mesh.order == 1:
            return 3 if self.mesh.etype == ElementType.triangle else 4
        else:
            return 6 if self.mesh.etype == ElementType.triangle else 10

    def _element_interpolate(
        self,
        x: jnp.ndarray,
        enodes: jnp.ndarray,
        evals: jnp.ndarray,
        etype: ElementType,
        order: int
    ) -> jnp.ndarray:
        if etype == ElementType.triangle:
            if order == 1:
                return _interpolate_on_tri3(x, enodes[:3], evals[:3])
            else:
                if enodes.shape[0] < 6:
                    # Upgrade tri3 to tri6 by inserting mid-edge nodes
                    v = enodes[:3]
                    mids = jnp.stack([
                        0.5 * (v[0] + v[1]),
                        0.5 * (v[1] + v[2]),
                        0.5 * (v[2] + v[0]),
                    ], axis=0)
                    enodes_q = jnp.concatenate([v, mids], axis=0)
                    evals_q = jnp.concatenate([
                        evals[:3],
                        0.5 * (evals[0:1] + evals[1:2]),
                        0.5 * (evals[1:2] + evals[2:3]),
                        0.5 * (evals[2:3] + evals[0:1]),
                    ], axis=0)
                else:
                    enodes_q = enodes[:6]
                    evals_q = evals[:6]
                return _interpolate_on_tri6(x, enodes_q, evals_q)
        else:
            if order == 1:
                return _interpolate_on_tet4(x, enodes[:4], evals[:4])
            else:
                if enodes.shape[0] < 10:
                    # Upgrade tet4 to tet10 by inserting mid-edge nodes
                    v = enodes[:4]
                    mids = jnp.stack([
                        0.5 * (v[0] + v[1]), 0.5 * (v[1] + v[2]), 0.5 * (v[2] + v[0]),
                        0.5 * (v[0] + v[3]), 0.5 * (v[1] + v[3]), 0.5 * (v[2] + v[3]),
                    ], axis=0)
                    enodes_q = jnp.concatenate([v, mids], axis=0)
                    evals_q = jnp.concatenate([
                        evals[:4],
                        0.5 * (evals[0:1] + evals[1:2]),
                        0.5 * (evals[1:2] + evals[2:3]),
                        0.5 * (evals[2:3] + evals[0:1]),
                        0.5 * (evals[0:1] + evals[3:4]),
                        0.5 * (evals[1:2] + evals[3:4]),
                        0.5 * (evals[2:3] + evals[3:4]),
                    ], axis=0)
                else:
                    enodes_q = enodes[:10]
                    evals_q = evals[:10]
                return _interpolate_on_tet10(x, enodes_q, evals_q)

    def _interpolate_single(
        self,
        x: jnp.ndarray,
        nodal_values: jnp.ndarray
    ) -> jnp.ndarray:
        nodes = jnp.asarray(self.mesh.nodes, dtype=jnp.float32)
        etype = self.mesh.etype
        order = self.mesh.order
        k = self.knn_k or self._default_k()

        if self.mesh.elements is not None:
            elements = self.mesh.elements
            if self.use_element_search:
                elem_idx = _find_containing_element(x, self.mesh)
                if elem_idx is not None:
                    idx = elements[elem_idx]
                    enodes = nodes[idx]
                    evals = nodal_values[idx]
                    return self._element_interpolate(x, enodes, evals, etype, order)
                else:
                    if self.strict_inside:
                        # Return zeros if strictly outside
                        C = int(nodal_values.shape[1])
                        return jnp.zeros((C,), dtype=jnp.float32)
                    # Fallback to nearest element centroid if not strict
                    elem_nodes = nodes[elements]  # (M, k, 3)
                    centroids = elem_nodes.mean(axis=1)  # (M, 3)
                    d2 = jnp.sum((centroids - x[None, :]) ** 2, axis=1)
                    e_idx = jnp.argmin(d2)
                    idx = elements[e_idx]
                    enodes = nodes[idx]
                    evals = nodal_values[idx]
                    return self._element_interpolate(x, enodes, evals, etype, order)
            else:
                # Nearest centroid element (JAX-friendly)
                elem_nodes = nodes[elements]  # (M, k, 3)
                centroids = elem_nodes.mean(axis=1)
                d2 = jnp.sum((centroids - x[None, :]) ** 2, axis=1)
                e_idx = jnp.argmin(d2)
                idx = elements[e_idx]
                enodes = nodes[idx]
                evals = nodal_values[idx]
                return self._element_interpolate(x, enodes, evals, etype, order)
        else:
            # kNN fallback
            idx = _knn_indices(x, nodes, k)
            enodes = nodes[idx]
            evals = nodal_values[idx]
            return self._element_interpolate(x, enodes, evals, etype, order)

    def sample(self, positions: np.ndarray, nodal_values: np.ndarray) -> jnp.ndarray:
        """
        Sample field at multiple positions.

        Parameters
        ----------
        positions : array-like
            Query positions, shape (N, 2|3)
        nodal_values : array-like
            Nodal field values, shape (Pn, C) where C is number of components

        Returns
        -------
        array-like
            Interpolated values, shape (N, C); jax.numpy array if JAX available, otherwise numpy array.
        """
        positions_np = _ensure_positions_shape(positions)
        nodal_values_np = _ensure_float32(nodal_values)

        if nodal_values_np.shape[0] != self.mesh.num_nodes:
            raise ValueError(
                f"nodal_values has {nodal_values_np.shape[0]} rows, "
                f"but mesh has {self.mesh.num_nodes} nodes"
            )

        if JAX_AVAILABLE:
            pos = jnp.asarray(positions_np, dtype=jnp.float32)
            vals = jnp.asarray(nodal_values_np, dtype=jnp.float32)
            fn = lambda x: self._interpolate_single(x, vals)
            return vmap(fn)(pos)
        else:
            # NumPy fallback
            results = []
            for i in range(positions_np.shape[0]):
                x = jnp.asarray(positions_np[i], dtype=jnp.float32)
                vals = jnp.asarray(nodal_values_np, dtype=jnp.float32)
                results.append(np.asarray(self._interpolate_single(x, vals)))
            return np.asarray(results, dtype=np.float32)

    def get_spatial_bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return spatial bounds of the mesh domain."""
        return self.mesh.get_spatial_bounds()


# ---------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------

def create_unstructured_field_2d(
    nodes: np.ndarray,
    triangles: Optional[np.ndarray] = None,
    order: int = 1,
    **kwargs
) -> UnstructuredField:
    """
    Create 2D unstructured field on triangular mesh.

    Parameters
    ----------
    nodes : array-like
        Node coordinates, shape (Pn, 2) or (Pn, 3)
    triangles : array-like, optional
        Triangle connectivity, shape (M, 3) for linear or (M, 6) for quadratic
    order : int
        Element order: 1 (linear) or 2 (quadratic)
    **kwargs
        Additional field options

    Returns
    -------
    UnstructuredField
        Configured field sampler
    """
    nodes3 = _ensure_positions_shape(nodes)
    if triangles is not None:
        triangles = np.asarray(triangles, dtype=np.int32)
        if triangles.shape[1] == 6:
            order = 2
        elif triangles.shape[1] == 3:
            order = 1
        else:
            raise ValueError(f"Expected 3 or 6 nodes per triangle, got {triangles.shape[1]}")
    mesh = UnstructuredMesh(nodes=nodes3, elements=triangles, etype=ElementType.triangle, order=order)
    return UnstructuredField(mesh=mesh, **kwargs)


def create_unstructured_field_3d(
    nodes: np.ndarray,
    tetrahedra: Optional[np.ndarray] = None,
    order: int = 1,
    **kwargs
) -> UnstructuredField:
    """
    Create 3D unstructured field on tetrahedral mesh.

    Parameters
    ----------
    nodes : array-like
        Node coordinates, shape (Pn, 3)
    tetrahedra : array-like, optional
        Tetrahedra connectivity, shape (M, 4) for linear or (M, 10) for quadratic
    order : int
        Element order: 1 (linear) or 2 (quadratic)
    **kwargs
        Additional field options

    Returns
    -------
    UnstructuredField
        Configured field sampler
    """
    nodes3 = _ensure_positions_shape(nodes)
    if tetrahedra is not None:
        tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
        if tetrahedra.shape[1] == 10:
            order = 2
        elif tetrahedra.shape[1] == 4:
            order = 1
        else:
            raise ValueError(f"Expected 4 or 10 nodes per tetrahedron, got {tetrahedra.shape[1]}")
    mesh = UnstructuredMesh(nodes=nodes3, elements=tetrahedra, etype=ElementType.tetra, order=order)
    return UnstructuredField(mesh=mesh, **kwargs)


def create_unstructured_from_vtk_data(
    vtk_data: Dict,
    **kwargs
) -> UnstructuredField:
    """
    Create unstructured field from VTK-like data dictionary.

    Parameters
    ----------
    vtk_data : dict
        VTK-like data containing 'points' and optionally:
        - 'connectivity' (np.ndarray)
        - 'cell_types' (np.ndarray of VTK cell type ints)
    **kwargs
        Additional field options

    Returns
    -------
    UnstructuredField
        Configured field sampler
    """
    if 'points' not in vtk_data:
        raise ValueError("VTK data must contain 'points'")

    nodes3 = _ensure_positions_shape(vtk_data['points'])

    elements = vtk_data.get('connectivity')
    cell_types = vtk_data.get('cell_types', None)

    if cell_types is not None:
        cell_types = np.asarray(cell_types)
        unique_types = np.unique(cell_types)
        if unique_types.size > 1:
            raise ValueError(f"Mixed cell types not supported: {unique_types}")

        # VTK cell type constants commonly used
        VTK_TRIANGLE = 5
        VTK_TETRA = 10
        VTK_QUADRATIC_TRIANGLE = 22
        VTK_QUADRATIC_TETRA = 24

        cell_type = int(unique_types[0])
        if cell_type in (VTK_TRIANGLE, VTK_QUADRATIC_TRIANGLE):
            etype = ElementType.triangle
            order = 1 if cell_type == VTK_TRIANGLE else 2
        elif cell_type in (VTK_TETRA, VTK_QUADRATIC_TETRA):
            etype = ElementType.tetra
            order = 1 if cell_type == VTK_TETRA else 2
        else:
            raise ValueError(f"Unsupported VTK cell type: {cell_type}")
    else:
        # Default if no cell type info
        etype = ElementType.tetra
        order = 1

    mesh = UnstructuredMesh(nodes=nodes3, elements=elements, etype=etype, order=order)
    return UnstructuredField(mesh=mesh, **kwargs)


# ---------------------------------------------------------------------
# Precomputation utilities (optional)
# ---------------------------------------------------------------------

def precompute_element_data(field: UnstructuredField) -> Dict:
    """
    Precompute element data for faster element search.

    Returns
    -------
    dict
        {
          'centroids': (M, 3),
          'element_bounds': (M, 2, 3),
          'element_nodes': (M, k, 3),
        }
    """
    if field.mesh.elements is None:
        return {}

    mesh = field.mesh
    nodes = mesh.nodes
    elements = mesh.elements

    elem_nodes = nodes[elements]                   # (M, k, 3)
    centroids = elem_nodes.mean(axis=1)            # (M, 3)
    elem_mins = elem_nodes.min(axis=1)             # (M, 3)
    elem_maxs = elem_nodes.max(axis=1)             # (M, 3)

    return {
        'centroids': centroids.astype(np.float32, copy=False),
        'element_bounds': np.stack([elem_mins, elem_maxs], axis=1).astype(np.float32, copy=False),
        'element_nodes': elem_nodes.astype(np.float32, copy=False),
    }


def optimized_element_search(
    x: jnp.ndarray,
    precomputed_data: Dict,
    mesh: UnstructuredMesh
) -> Optional[int]:
    """
    Optimized element search using precomputed bounding boxes, then barycentric test.

    Parameters
    ----------
    x : jnp.ndarray
        Query point, shape (3,)
    precomputed_data : dict
        Precomputed element data from precompute_element_data()
    mesh : UnstructuredMesh
        Mesh geometry

    Returns
    -------
    Optional[int]
        Element index or None if not found
    """
    if not precomputed_data:
        return _find_containing_element(x, mesh)

    bounds = precomputed_data['element_bounds']  # (M, 2, 3)
    inside_bounds = np.all(
        (np.asarray(x)[None, :] >= bounds[:, 0, :]) & (np.asarray(x)[None, :] <= bounds[:, 1, :]),
        axis=1
    )
    candidates = np.where(inside_bounds)[0]

    nodes = mesh.nodes
    elements = mesh.elements

    for ei in candidates:
        idx = elements[ei]
        enodes = nodes[idx]
        if mesh.etype == ElementType.triangle and enodes.shape[0] >= 3:
            bary = barycentric_coords_triangle(
                jnp.asarray(x, dtype=jnp.float32),
                jnp.asarray(enodes[0], dtype=jnp.float32),
                jnp.asarray(enodes[1], dtype=jnp.float32),
                jnp.asarray(enodes[2], dtype=jnp.float32),
            )
            if jnp.all(bary >= -1e-6) and jnp.sum(bary) <= 1.0 + 1e-6:
                return int(ei)
        elif mesh.etype == ElementType.tetra and enodes.shape[0] >= 4:
            bary = barycentric_coords_tetrahedron(
                jnp.asarray(x, dtype=jnp.float32),
                jnp.asarray(enodes[0], dtype=jnp.float32),
                jnp.asarray(enodes[1], dtype=jnp.float32),
                jnp.asarray(enodes[2], dtype=jnp.float32),
                jnp.asarray(enodes[3], dtype=jnp.float32),
            )
            if jnp.all(bary >= -1e-6) and jnp.sum(bary) <= 1.0 + 1e-6:
                return int(ei)

    return None


# Backwards compatibility alias
UnstructuredSampler = UnstructuredField