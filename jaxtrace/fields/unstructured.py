# jaxtrace/fields/unstructured.py  
"""  
Unstructured mesh field interpolation with barycentric coordinates.  

Supports linear and quadratic triangular/tetrahedral elements with  
fallback to k-nearest neighbor interpolation when connectivity is missing.  
Enhanced with VTK integration and comprehensive error handling.  
"""  

from __future__ import annotations  
from dataclasses import dataclass  
from enum import Enum  
from typing import Optional, Tuple  
import numpy as np  

# Import JAX utilities with fallback  
from ..utils.jax_utils import JAX_AVAILABLE  

if JAX_AVAILABLE:  
    try:  
        import jax  
        import jax.numpy as jnp  
        from jax import vmap  
    except Exception:  
        JAX_AVAILABLE = False  

if not JAX_AVAILABLE:  
    import numpy as jnp  # type: ignore  
    # Mock JAX functions for NumPy fallback  
    class MockJit:  
        def __call__(self, func):  
            return func  
    class MockVmap:  
        def __call__(self, func):  
            def vectorized(x):  
                return np.array([func(xi) for xi in x])  
            return vectorized  
    jax = type('MockJax', (), {'jit': MockJit()})()  
    vmap = MockVmap()  

from .base import (  
    BaseField,  
    barycentric_coords_triangle,  
    barycentric_coords_tetrahedron,  
    tri6_shape_functions,  
    tet10_shape_functions,  
    _ensure_float32,  
    _ensure_positions_shape  
)  


class ElementType(str, Enum):  
    """Supported element types for unstructured meshes."""  
    triangle = "triangle"  
    tetra = "tetra"  


@dataclass  
class UnstructuredMesh:  
    """  
    Unstructured mesh data container.  

    Supports both linear and quadratic elements with optional connectivity.  
    Enhanced with spatial bounds and validation.  
    
    Attributes  
    ----------  
    nodes : jnp.ndarray  
        Node coordinates, shape (Pn, 3)  
    elements : jnp.ndarray, optional  
        Element connectivity, shape (M, k) where:  
        - k=3 for tri3, k=6 for tri6  
        - k=4 for tet4, k=10 for tet10  
    etype : ElementType  
        Element type: triangle or tetra  
    order : int  
        Element order: 1 (linear) or 2 (quadratic)  
    """  
    nodes: jnp.ndarray  
    elements: Optional[jnp.ndarray]  
    etype: ElementType  
    order: int = 1  
    
    def __post_init__(self):  
        """Validate mesh data after initialization."""  
        self.nodes = _ensure_float32(self.nodes)  
        if self.elements is not None:  
            self.elements = np.asarray(self.elements, dtype=np.int32)  
            
        # Validate connectivity  
        self._validate_connectivity()  
    
    def _validate_connectivity(self):  
        """Validate element connectivity against mesh topology."""  
        if self.elements is None:  
            return  # No connectivity to validate  
        
        n_nodes = self.nodes.shape[0]  
        expected_k = self.expected_nodes_per_element()  
        
        if self.elements.shape[1] != expected_k:  
            raise ValueError(  
                f"Expected {expected_k} nodes per {self.etype.value} element (order {self.order}), "  
                f"got {self.elements.shape[1]}"  
            )  
        
        # Check node indices are valid  
        max_node_idx = np.max(self.elements)  
        if max_node_idx >= n_nodes:  
            raise ValueError(f"Element references node {max_node_idx}, but only {n_nodes} nodes available")  
    
    def expected_nodes_per_element(self) -> int:  
        """Get expected number of nodes per element based on type and order."""  
        if self.etype == ElementType.triangle:  
            return 3 if self.order == 1 else 6  
        else:  # tetra  
            return 4 if self.order == 1 else 10  
    
    def get_spatial_bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:  
        """Get spatial bounds of the mesh."""  
        bounds_min = jnp.min(self.nodes, axis=0)  
        bounds_max = jnp.max(self.nodes, axis=0)  
        return bounds_min, bounds_max  
    
    @property  
    def num_nodes(self) -> int:  
        """Number of mesh nodes."""  
        return self.nodes.shape[0]  
    
    @property  
    def num_elements(self) -> int:  
        """Number of mesh elements."""  
        return self.elements.shape[0] if self.elements is not None else 0  


def _shape_linear_triangle(lmbda: jnp.ndarray) -> jnp.ndarray:  
    """Linear triangle shape functions (identity for barycentric coords)."""  
    return lmbda  # (3,)  


def _shape_linear_tetra(lmbda: jnp.ndarray) -> jnp.ndarray:  
    """Linear tetrahedron shape functions (identity for barycentric coords)."""  
    return lmbda  # (4,)  


@jax.jit  
def _interpolate_on_tri3(  
    p: jnp.ndarray,   
    tri_nodes: jnp.ndarray,   
    nodal_vals: jnp.ndarray  
) -> jnp.ndarray:  
    """Interpolate on linear triangle using barycentric coordinates."""  
    l = barycentric_coords_triangle(p, tri_nodes[0], tri_nodes[1], tri_nodes[2])  
    N = _shape_linear_triangle(l)  # (3,)  
    return (N[:, None] * nodal_vals).sum(axis=0)  


@jax.jit  
def _interpolate_on_tri6(  
    p: jnp.ndarray,   
    tri6_nodes: jnp.ndarray,   
    nodal_vals: jnp.ndarray  
) -> jnp.ndarray:  
    """Interpolate on quadratic triangle using 6-node shape functions."""  
    l = barycentric_coords_triangle(p, tri6_nodes[0], tri6_nodes[1], tri6_nodes[2])  
    Nq = tri6_shape_functions(l)  # (6,)  
    return (Nq[:, None] * nodal_vals).sum(axis=0)  


@jax.jit  
def _interpolate_on_tet4(  
    p: jnp.ndarray,   
    tet_nodes: jnp.ndarray,   
    nodal_vals: jnp.ndarray  
) -> jnp.ndarray:  
    """Interpolate on linear tetrahedron using barycentric coordinates."""  
    l = barycentric_coords_tetrahedron(  
        p, tet_nodes[0], tet_nodes[1], tet_nodes[2], tet_nodes[3]  
    )  
    N = _shape_linear_tetra(l)  # (4,)  
    return (N[:, None] * nodal_vals).sum(axis=0)  


@jax.jit  
def _interpolate_on_tet10(  
    p: jnp.ndarray,   
    tet10_nodes: jnp.ndarray,   
    nodal_vals: jnp.ndarray  
) -> jnp.ndarray:  
    """Interpolate on quadratic tetrahedron using 10-node shape functions."""  
    l = barycentric_coords_tetrahedron(  
        p, tet10_nodes[0], tet10_nodes[1], tet10_nodes[2], tet10_nodes[3]  
    )  
    Nq = tet10_shape_functions(l)  # (10,)  
    return (Nq[:, None] * nodal_vals).sum(axis=0)  


def _knn_indices(x: jnp.ndarray, nodes: jnp.ndarray, k: int) -> jnp.ndarray:  
    """Find k nearest neighbor indices using O(Pn) scan."""  
    # O(Pn) scan; for very large meshes, prefer a grid-hash neighbor search  
    d2 = jnp.sum((nodes - x[None, :]) ** 2, axis=1)  
    idx = jnp.argsort(d2)[:k]  
    return idx  


def _find_containing_element(x: jnp.ndarray, mesh: UnstructuredMesh) -> Optional[int]:  
    """  
    Find element containing point x using barycentric coordinate tests.  
    
    Returns element index or None if point is outside mesh.  
    """  
    if mesh.elements is None:  
        return None  
    
    nodes = mesh.nodes  
    elements = mesh.elements  
    
    for elem_idx in range(mesh.num_elements):  
        elem_nodes = nodes[elements[elem_idx]]  
        
        if mesh.etype == ElementType.triangle:  
            # Check if point is inside triangle  
            if elem_nodes.shape[0] >= 3:  
                bary = barycentric_coords_triangle(x, elem_nodes[0], elem_nodes[1], elem_nodes[2])  
                if jnp.all(bary >= -1e-6) and jnp.sum(bary) <= 1.0 + 1e-6:  # Small tolerance  
                    return elem_idx  
        else:  # tetra  
            # Check if point is inside tetrahedron  
            if elem_nodes.shape[0] >= 4:  
                bary = barycentric_coords_tetrahedron(x, elem_nodes[0], elem_nodes[1],   
                                                    elem_nodes[2], elem_nodes[3])  
                if jnp.all(bary >= -1e-6) and jnp.sum(bary) <= 1.0 + 1e-6:  # Small tolerance  
                    return elem_idx  
    
    return None  


@dataclass  
class UnstructuredField:  
    """  
    Spatial sampler on unstructured mesh with optional connectivity.  
    
    If connectivity is absent, uses kNN as pseudo-element nodes.  
    Supports both linear and quadratic elements with enhanced error handling.  
    
    Attributes  
    ----------  
    mesh : UnstructuredMesh  
        Mesh geometry and connectivity  
    knn_k : int, optional  
        Number of neighbors for kNN fallback (auto-determined if None)  
    strict_inside : bool  
        Whether to enforce strict inside-element checks  
    use_element_search : bool  
        Whether to search for containing elements (slower but more accurate)  
    """  
    mesh: UnstructuredMesh  
    knn_k: Optional[int] = None  
    strict_inside: bool = False  
    use_element_search: bool = False  

    def __post_init__(self):  
        """Initialize field with validated parameters."""  
        if self.knn_k is None:  
            self.knn_k = self._default_k()  
        
        # Validate k  
        max_k = min(self.mesh.num_nodes, 20)  # Reasonable upper limit  
        if self.knn_k > max_k:  
            self.knn_k = max_k  

    def _default_k(self) -> int:  
        """Determine default k for kNN based on element type and order."""  
        if self.mesh.order == 1:  
            return 3 if self.mesh.etype == ElementType.triangle else 4  
        else:  
            return 6 if self.mesh.etype == ElementType.triangle else 10  

    def _interpolate_single(self, x: jnp.ndarray, nodal_values: jnp.ndarray) -> jnp.ndarray:  
        """Interpolate at single point using mesh connectivity or kNN."""  
        nodes = self.mesh.nodes  
        elements = self.mesh.elements  
        etype = self.mesh.etype  
        order = self.mesh.order  
        k = self.knn_k or self._default_k()  

        # Try element-based interpolation if connectivity exists  
        if elements is not None and self.use_element_search:  
            elem_idx = _find_containing_element(x, self.mesh)  
            
            if elem_idx is not None:  
                # Use exact element  
                elem_nodes_idx = elements[elem_idx]  
                enodes = nodes[elem_nodes_idx]  
                evals = nodal_values[elem_nodes_idx]  
                
                # Perform element interpolation  
                return self._element_interpolate(x, enodes, evals, etype, order)  

        if elements is not None and not self.use_element_search:  
            # Use mesh connectivity: select element with closest centroid  
            elem_nodes = nodes[elements]        # (M, k, 3)  
            centroids = elem_nodes.mean(axis=1) # (M, 3)  
            d2 = jnp.sum((centroids - x[None, :]) ** 2, axis=1)  
            e_idx = jnp.argmin(d2)  
            enodes = elem_nodes[e_idx]              # (k, 3)  
            evals = nodal_values[elements[e_idx]]   # (k, C)  
        else:  
            # Fallback to kNN  
            idx = _knn_indices(x, nodes, k)  
            enodes = nodes[idx]  
            evals = nodal_values[idx]  

        # Perform interpolation  
        return self._element_interpolate(x, enodes, evals, etype, order)  
    
    def _element_interpolate(self, x: jnp.ndarray, enodes: jnp.ndarray,   
                           evals: jnp.ndarray, etype: ElementType, order: int) -> jnp.ndarray:  
        """Perform element-based interpolation."""  
        # Interpolate based on element type and order  
        if etype == ElementType.triangle:  
            if order == 1:  
                return _interpolate_on_tri3(x, enodes[:3], evals[:3])  
            else:  # order == 2  
                if enodes.shape[0] < 6:  
                    # Generate quadratic nodes from linear ones  
                    v = enodes[:3]  
                    mids = jnp.stack([  
                        (v[0] + v[1]) * 0.5,  
                        (v[1] + v[2]) * 0.5,  
                        (v[2] + v[0]) * 0.5  
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
        else:  # tetra  
            if order == 1:  
                return _interpolate_on_tet4(x, enodes[:4], evals[:4])  
            else:  # order == 2  
                if enodes.shape[0] < 10:  
                    # Generate quadratic nodes from linear ones  
                    v = enodes[:4]  
                    mids = jnp.stack([
                        0.5 * (v[0] + v[1]), 0.5 * (v[1] + v[2]), 0.5 * (v[2] + v[0]),  # tri edges
                        0.5 * (v[0] + v[3]), 0.5 * (v[1] + v[3]), 0.5 * (v[2] + v[3])   # tet edges
                    ], axis=0)
                    enodes_q = jnp.concatenate([v, mids], axis=0)
                    evals_q = jnp.concatenate([
                        evals[:4],
                        0.5 * (evals[0:1] + evals[1:2]),  # edge 0-1
                        0.5 * (evals[1:2] + evals[2:3]),  # edge 1-2
                        0.5 * (evals[2:3] + evals[0:1]),  # edge 2-0
                        0.5 * (evals[0:1] + evals[3:4]),  # edge 0-3
                        0.5 * (evals[1:2] + evals[3:4]),  # edge 1-3
                        0.5 * (evals[2:3] + evals[3:4]),  # edge 2-3
                    ], axis=0)
                else:
                    enodes_q = enodes[:10]
                    evals_q = evals[:10]
                return _interpolate_on_tet10(x, enodes_q, evals_q)

    def sample(self, positions: jnp.ndarray, nodal_values: jnp.ndarray) -> jnp.ndarray:
        """
        Sample field at multiple positions.

        Parameters
        ----------
        positions : jnp.ndarray
            Query positions, shape (N, 3)
        nodal_values : jnp.ndarray
            Nodal field values, shape (Pn, C) where C is number of components

        Returns
        -------
        jnp.ndarray
            Interpolated values, shape (N, C)
        """
        positions = _ensure_positions_shape(positions)
        nodal_values = _ensure_float32(nodal_values)
        
        if positions.shape[0] == 0:
            return np.zeros((0, nodal_values.shape[1]), dtype=np.float32)
        
        if nodal_values.shape[0] != self.mesh.num_nodes:
            raise ValueError(
                f"nodal_values has {nodal_values.shape[0]} rows, "
                f"but mesh has {self.mesh.num_nodes} nodes"
            )

        # Vectorized sampling
        interpolate_fn = lambda x: self._interpolate_single(x, nodal_values)
        
        if JAX_AVAILABLE:
            vectorized_fn = vmap(interpolate_fn)
            return vectorized_fn(positions)
        else:
            # NumPy fallback
            results = []
            for i in range(positions.shape[0]):
                result = interpolate_fn(positions[i])
                results.append(result)
            return np.array(results)

    def get_spatial_bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return spatial bounds of the mesh domain."""
        return self.mesh.get_spatial_bounds()

    def validate_field_data(self, nodal_values: jnp.ndarray) -> bool:
        """
        Validate that nodal values are consistent with mesh.
        
        Parameters
        ----------
        nodal_values : jnp.ndarray
            Nodal field values to validate
            
        Returns
        -------
        bool
            True if valid
            
        Raises
        ------
        ValueError
            If data is inconsistent
        """
        if nodal_values.shape[0] != self.mesh.num_nodes:
            raise ValueError(
                f"nodal_values has {nodal_values.shape[0]} rows, "
                f"but mesh has {self.mesh.num_nodes} nodes"
            )
        
        if not np.all(np.isfinite(nodal_values)):
            raise ValueError("nodal_values contains non-finite values")
        
        return True

    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        mesh_size = (self.mesh.nodes.size * 4 +  # float32 nodes
                    (self.mesh.elements.size * 4 if self.mesh.elements is not None else 0))  # int32 elements
        return mesh_size / 1024**2


# Factory functions for creating unstructured fields

def create_unstructured_field_2d(
    nodes: jnp.ndarray,
    triangles: Optional[jnp.ndarray] = None,
    order: int = 1,
    **kwargs
) -> UnstructuredField:
    """
    Create 2D unstructured field on triangular mesh.
    
    Parameters
    ----------
    nodes : jnp.ndarray
        Node coordinates, shape (Pn, 2) or (Pn, 3)
    triangles : jnp.ndarray, optional
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
    # Ensure 3D coordinates
    nodes = _ensure_positions_shape(nodes)
    
    # Infer element order from connectivity if provided
    if triangles is not None:
        if triangles.shape[1] == 6:
            order = 2
        elif triangles.shape[1] == 3:
            order = 1
        else:
            raise ValueError(f"Expected 3 or 6 nodes per triangle, got {triangles.shape[1]}")
    
    mesh = UnstructuredMesh(
        nodes=nodes,
        elements=triangles,
        etype=ElementType.triangle,
        order=order
    )
    
    return UnstructuredField(mesh=mesh, **kwargs)


def create_unstructured_field_3d(
    nodes: jnp.ndarray,
    tetrahedra: Optional[jnp.ndarray] = None,
    order: int = 1,
    **kwargs
) -> UnstructuredField:
    """
    Create 3D unstructured field on tetrahedral mesh.
    
    Parameters
    ----------
    nodes : jnp.ndarray
        Node coordinates, shape (Pn, 3)
    tetrahedra : jnp.ndarray, optional
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
    nodes = _ensure_positions_shape(nodes)
    
    # Infer element order from connectivity if provided
    if tetrahedra is not None:
        if tetrahedra.shape[1] == 10:
            order = 2
        elif tetrahedra.shape[1] == 4:
            order = 1
        else:
            raise ValueError(f"Expected 4 or 10 nodes per tetrahedron, got {tetrahedra.shape[1]}")
    
    mesh = UnstructuredMesh(
        nodes=nodes,
        elements=tetrahedra,
        etype=ElementType.tetra,
        order=order
    )
    
    return UnstructuredField(mesh=mesh, **kwargs)


def create_unstructured_from_vtk_data(
    vtk_data: dict,
    **kwargs
) -> UnstructuredField:
    """
    Create unstructured field from VTK data dictionary.
    
    Parameters
    ----------
    vtk_data : dict
        VTK data containing 'points' and optionally 'connectivity'
    **kwargs
        Additional field options
        
    Returns
    -------
    UnstructuredField
        Configured field sampler
    """
    if 'points' not in vtk_data:
        raise ValueError("VTK data must contain 'points'")
    
    nodes = _ensure_positions_shape(vtk_data['points'])
    
    # Extract connectivity if available
    elements = vtk_data.get('connectivity')
    cell_types = vtk_data.get('cell_types', [])
    
    # Determine element type from VTK cell types
    if len(cell_types) > 0:
        # VTK cell type constants
        VTK_TRIANGLE = 5
        VTK_QUAD = 9
        VTK_TETRA = 10
        VTK_HEXAHEDRON = 12
        VTK_QUADRATIC_TRIANGLE = 22
        VTK_QUADRATIC_TETRA = 24
        
        unique_types = np.unique(cell_types)
        
        if len(unique_types) > 1:
            raise ValueError(f"Mixed cell types not supported: {unique_types}")
        
        cell_type = unique_types[0]
        
        if cell_type in (VTK_TRIANGLE, VTK_QUADRATIC_TRIANGLE):
            etype = ElementType.triangle
            order = 1 if cell_type == VTK_TRIANGLE else 2
        elif cell_type in (VTK_TETRA, VTK_QUADRATIC_TETRA):
            etype = ElementType.tetra
            order = 1 if cell_type == VTK_TETRA else 2
        else:
            raise ValueError(f"Unsupported VTK cell type: {cell_type}")
    else:
        # Default to 3D tetrahedra if no type info
        etype = ElementType.tetra
        order = 1
    
    mesh = UnstructuredMesh(
        nodes=nodes,
        elements=elements,
        etype=etype,
        order=order
    )
    
    return UnstructuredField(mesh=mesh, **kwargs)


# Aliases for backwards compatibility
UnstructuredSampler = UnstructuredField  # Alias for backwards compatibility


# Performance utilities

def precompute_element_data(field: UnstructuredField) -> dict:
    """
    Precompute element data for faster interpolation.
    
    Parameters
    ----------
    field : UnstructuredField
        Field to precompute data for
        
    Returns
    -------
    dict
        Precomputed data for faster interpolation
    """
    if field.mesh.elements is None:
        return {}
    
    mesh = field.mesh
    nodes = mesh.nodes
    elements = mesh.elements
    
    # Precompute element centroids and bounds
    elem_nodes = nodes[elements]  # (M, k, 3)
    centroids = elem_nodes.mean(axis=1)  # (M, 3)
    
    # Compute element bounding boxes
    elem_mins = elem_nodes.min(axis=1)  # (M, 3)
    elem_maxs = elem_nodes.max(axis=1)  # (M, 3)
    
    return {
        'centroids': centroids,
        'element_bounds': np.stack([elem_mins, elem_maxs], axis=1),  # (M, 2, 3)
        'element_nodes': elem_nodes
    }


def optimized_element_search(
    x: jnp.ndarray, 
    precomputed_data: dict,
    mesh: UnstructuredMesh
) -> Optional[int]:
    """
    Optimized element search using precomputed data.
    
    Parameters
    ----------
    x : jnp.ndarray
        Query point, shape (3,)
    precomputed_data : dict
        Precomputed element data
    mesh : UnstructuredMesh
        Mesh geometry
        
    Returns
    -------
    Optional[int]
        Element index or None if not found
    """
    if not precomputed_data:
        return _find_containing_element(x, mesh)
    
    # Quick bounding box test first
    bounds = precomputed_data['element_bounds']  # (M, 2, 3)
    
    # Check which elements could contain the point
    inside_bounds = np.all(
        (x[None, :] >= bounds[:, 0, :]) & (x[None, :] <= bounds[:, 1, :]),
        axis=1
    )
    
    candidate_elements = np.where(inside_bounds)[0]
    
    # Test candidates with barycentric coordinates
    nodes = mesh.nodes
    elements = mesh.elements
    
    for elem_idx in candidate_elements:
        elem_nodes = nodes[elements[elem_idx]]
        
        if mesh.etype == ElementType.triangle:
            if elem_nodes.shape[0] >= 3:
                bary = barycentric_coords_triangle(x, elem_nodes[0], elem_nodes[1], elem_nodes[2])
                if jnp.all(bary >= -1e-6) and jnp.sum(bary) <= 1.0 + 1e-6:
                    return elem_idx
        else:  # tetra
            if elem_nodes.shape[0] >= 4:
                bary = barycentric_coords_tetrahedron(
                    x, elem_nodes[0], elem_nodes[1], elem_nodes[2], elem_nodes[3]
                )
                if jnp.all(bary >= -1e-6) and jnp.sum(bary) <= 1.0 + 1e-6:
                    return elem_idx
    
    return None