"""
KD-Tree Based Node Search for L2 Element Location

Simple and direct approach:
1. Build KD-tree from mesh node positions
2. For query position, find K nearest nodes
3. Test all elements connected to those nodes
4. First element containing position wins

Advantages:
- Very simple algorithm
- No complex octree/Morton structure
- Leverages existing KD-tree library (jaxkd)
- Should achieve ~100% retention for in-mesh particles

Performance:
- KD-tree query: O(log N) for nearest node
- Element tests: ~K nodes × ~10 elements/node = ~10K tests
- Expected K=1-3 sufficient for most cases
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple
from dataclasses import dataclass

import jaxtrace.config as config
from jaxtrace.gpu.search.point_in_tet_methods import point_in_tet_gpu

# Try to import jaxkd, fall back to error message if not available
try:
    import jaxkd as jk
    JAXKD_AVAILABLE = True
except ImportError:
    JAXKD_AVAILABLE = False
    jk = None


@dataclass
class NodeKDTreeStructure:
    """
    KD-tree structure for node-based element search.

    Attributes:
        node_positions: (n_nodes, 3) float64 - mesh node coordinates
        connectivity: (n_elements, 4) int32 - element connectivity
        node_to_elements_offsets: (n_nodes+1,) int32 - CSR offsets
        node_to_elements_data: (total_entries,) int32 - element IDs
        n_nodes: int - number of nodes
        n_elements: int - number of elements
        elements_per_node_mean: float - average elements per node
        elements_per_node_max: int - maximum elements per node
    """
    node_positions: np.ndarray
    connectivity: np.ndarray
    node_to_elements_offsets: np.ndarray
    node_to_elements_data: np.ndarray
    n_nodes: int
    n_elements: int
    elements_per_node_mean: float
    elements_per_node_max: int


@dataclass
class NodeKDTreeGPU:
    """
    GPU-resident KD-tree structure.

    Attributes:
        node_positions: (n_nodes, 3) float32 - on GPU (for point-in-tet)
        connectivity: (n_elements, 4) int32 - on GPU
        node_to_elements_offsets: (n_nodes+1,) int32 - on GPU
        node_to_elements_data: (total_entries,) int32 - on GPU
        kdtree: jaxkd tree structure - KD-tree on GPU (built from float64)
        n_nodes: int32
        n_elements: int32

    Note:
        The kdtree is built from float64 node positions (jaxkd requirement),
        but node_positions is stored as float32 for point-in-tet efficiency.
        Query positions are converted to float64 internally before querying.
    """
    node_positions: jax.Array
    connectivity: jax.Array
    node_to_elements_offsets: jax.Array
    node_to_elements_data: jax.Array
    kdtree: any  # jaxkd.KDTree
    n_nodes: jnp.int32
    n_elements: jnp.int32


def build_node_to_elements_mapping(
    connectivity: np.ndarray,
    n_nodes: int,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build inverted connectivity: node → [elements].

    Args:
        connectivity: (n_elements, 4) int32 - element-to-node connectivity
        n_nodes: int - number of nodes in mesh
        verbose: print progress

    Returns:
        node_to_elements_offsets: (n_nodes+1,) int32 - CSR offsets
        node_to_elements_data: (total_entries,) int32 - element IDs
    """
    from collections import defaultdict

    if verbose:
        print(f"\nBuilding node → elements mapping...")

    n_elements = connectivity.shape[0]

    # Build node → elements dictionary
    node_to_elements = defaultdict(list)

    for elem_id in range(n_elements):
        for node_id in connectivity[elem_id]:
            node_to_elements[node_id].append(elem_id)

    # Build CSR arrays
    node_to_elements_offsets = np.zeros(n_nodes + 1, dtype=np.int32)
    node_to_elements_lists = []

    for node_id in range(n_nodes):
        elem_list = node_to_elements.get(node_id, [])
        node_to_elements_offsets[node_id + 1] = node_to_elements_offsets[node_id] + len(elem_list)
        node_to_elements_lists.extend(elem_list)

    node_to_elements_data = np.array(node_to_elements_lists, dtype=np.int32)

    # Compute statistics
    elements_per_node = np.diff(node_to_elements_offsets)
    elements_per_node_mean = elements_per_node.mean()
    elements_per_node_max = elements_per_node.max()

    if verbose:
        print(f"  ✅ Node → elements mapping built!")
        print(f"    CSR entries: {len(node_to_elements_data):,}")
        print(f"    Elements per node: {elements_per_node_mean:.1f} (mean), {elements_per_node_max} (max)")

    return node_to_elements_offsets, node_to_elements_data


def build_kdtree_structure(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    verbose: bool = True
) -> NodeKDTreeStructure:
    """
    Build KD-tree structure from mesh nodes.

    Args:
        node_positions: (n_nodes, 3) float64 - node coordinates
        connectivity: (n_elements, 4) int32 - element connectivity
        verbose: print progress

    Returns:
        NodeKDTreeStructure ready for GPU upload
    """
    if not JAXKD_AVAILABLE:
        raise ImportError(
            "jaxkd not available. Install with: pip install jaxkd"
        )

    n_nodes = node_positions.shape[0]
    n_elements = connectivity.shape[0]

    if verbose:
        print(f"\n{'='*80}")
        print("Building KD-Tree Node Search Structure")
        print(f"{'='*80}")
        print(f"  Nodes: {n_nodes:,}")
        print(f"  Elements: {n_elements:,}")

    # Build node → elements mapping
    node_to_elements_offsets, node_to_elements_data = build_node_to_elements_mapping(
        connectivity, n_nodes, verbose=verbose
    )

    # Compute statistics
    elements_per_node = np.diff(node_to_elements_offsets)
    elements_per_node_mean = elements_per_node.mean()
    elements_per_node_max = int(elements_per_node.max())

    if verbose:
        print(f"\n{'='*80}")
        print("✅ KD-Tree Structure Complete!")
        print(f"{'='*80}")
        print(f"NodeKDTreeStructure(")
        print(f"  n_nodes={n_nodes:,},")
        print(f"  n_elements={n_elements:,},")
        print(f"  elements_per_node: mean={elements_per_node_mean:.1f}, max={elements_per_node_max}")
        print(f")")
        print(f"{'='*80}\n")

    return NodeKDTreeStructure(
        node_positions=node_positions,
        connectivity=connectivity,
        node_to_elements_offsets=node_to_elements_offsets,
        node_to_elements_data=node_to_elements_data,
        n_nodes=n_nodes,
        n_elements=n_elements,
        elements_per_node_mean=elements_per_node_mean,
        elements_per_node_max=elements_per_node_max,
    )


def upload_kdtree_to_gpu(
    structure: NodeKDTreeStructure,
    verbose: bool = True
) -> NodeKDTreeGPU:
    """
    Upload KD-tree structure to GPU.

    Args:
        structure: NodeKDTreeStructure to upload
        verbose: print progress

    Returns:
        NodeKDTreeGPU with data on GPU and KD-tree built
    """
    if not JAXKD_AVAILABLE:
        raise ImportError(
            "jaxkd not available. Install with: pip install jaxkd"
        )

    if verbose:
        print(f"\nUploading KD-Tree Structure to GPU...")

    # Upload arrays to GPU
    # NOTE: KD-tree needs float64 for internal calculations (jaxkd requirement)
    # Query positions will also need to be float64 when calling query_neighbors
    node_positions_gpu_f64 = jnp.array(structure.node_positions, dtype=jnp.float64)
    node_positions_gpu_f32 = jnp.array(structure.node_positions, dtype=jnp.float32)
    connectivity_gpu = jnp.array(structure.connectivity, dtype=jnp.int32)
    node_to_elements_offsets_gpu = jnp.array(structure.node_to_elements_offsets, dtype=jnp.int32)
    node_to_elements_data_gpu = jnp.array(structure.node_to_elements_data, dtype=jnp.int32)

    # Build KD-tree on GPU (requires float64)
    if verbose:
        print(f"  Building KD-tree from {structure.n_nodes:,} nodes...")

    kdtree = jk.build_tree(node_positions_gpu_f64)

    if verbose:
        print(f"  ✅ Upload complete!")
        print(f"    Nodes: {structure.n_nodes:,}")
        print(f"    Elements per node: {structure.elements_per_node_mean:.1f} (mean)")

    return NodeKDTreeGPU(
        node_positions=node_positions_gpu_f32,  # float32 for point-in-tet
        connectivity=connectivity_gpu,
        node_to_elements_offsets=node_to_elements_offsets_gpu,
        node_to_elements_data=node_to_elements_data_gpu,
        kdtree=kdtree,  # Built from float64 node_positions
        n_nodes=jnp.int32(structure.n_nodes),
        n_elements=jnp.int32(structure.n_elements),
    )


# ============================================================================
# Search Functions
# ============================================================================

def search_L2_kdtree_single(
    pos: jax.Array,
    kdtree_gpu: NodeKDTreeGPU,
    k_nearest: int = 3,
    max_tests: int = 256
) -> jnp.int32:
    """
    L2 search using KD-tree nearest nodes (single particle, not vmappable).

    WARNING: This function calls jk.query_neighbors which has Python control flow.
    It CANNOT be vmapped. Use search_L2_kdtree_batch for multiple particles.

    Algorithm:
    1. Find K nearest nodes to query position
    2. For each nearest node, get connected elements
    3. Test elements until one contains position

    Args:
        pos: (3,) float32 - query position
        kdtree_gpu: GPU-resident KD-tree structure
        k_nearest: number of nearest nodes to search (Python int, not traced)
        max_tests: maximum element tests (Python int, not traced)

    Returns:
        elem_id: int32 - found element ID, or -1 if not found
    """
    # Convert to float64 (jaxkd requirement)
    pos_f64 = pos.astype(jnp.float64)

    # Find K nearest nodes
    # NOTE: This has Python control flow and cannot be vmapped
    nearest_node_ids, distances = jk.query_neighbors(
        kdtree_gpu.kdtree, pos_f64.reshape(1, 3), k=k_nearest
    )
    nearest_node_ids = nearest_node_ids[0]  # (k,) - extract first query result

    elem_id = jnp.int32(-1)
    n_tests = jnp.int32(0)

    # Search elements connected to nearest nodes
    for k_idx in range(k_nearest):
        node_id = nearest_node_ids[k_idx]

        # Get elements connected to this node
        start = kdtree_gpu.node_to_elements_offsets[node_id]
        end = kdtree_gpu.node_to_elements_offsets[node_id + 1]

        # Test each element
        for elem_idx in range(start, end):
            # Stop if already found or exceeded max tests
            if (elem_id >= 0) or (n_tests >= max_tests):
                break

            test_elem_id = kdtree_gpu.node_to_elements_data[elem_idx]

            # Point-in-tet test using dispatcher (supports all methods including 'inverse')
            is_inside = point_in_tet_gpu(
                pos,
                test_elem_id,
                kdtree_gpu.connectivity,
                kdtree_gpu.node_positions,
                method=config.POINT_IN_TET_METHOD
            )

            n_tests += 1

            if is_inside:
                elem_id = test_elem_id
                break

    return elem_id


def _search_kdtree_with_prequeried_nodes(
    pos: jax.Array,
    nearest_node_ids: jax.Array,
    kdtree_gpu: NodeKDTreeGPU,
    k_nearest_py: int,
    max_tests_py: int
) -> jnp.int32:
    """
    JAX-traceable KD-tree search with pre-queried nearest node IDs.

    This function is designed to be called INSIDE a vmapped context where
    the KD-tree has already been queried externally.

    Args:
        pos: (3,) float32 - query position
        nearest_node_ids: (k_nearest,) int32 - pre-queried nearest node IDs
        kdtree_gpu: GPU-resident KD-tree structure
        k_nearest_py: number of nearest nodes (Python int, not traced)
        max_tests_py: maximum element tests (Python int, not traced)

    Returns:
        elem_id: int32 - found element ID, or -1 if not found
    """
    from jax import lax

    def search_one_node(k_idx, carry):
        """Search elements connected to one nearest node."""
        found_elem, n_tests_total = carry

        # Get node and its elements
        node_id = nearest_node_ids[k_idx]
        start = kdtree_gpu.node_to_elements_offsets[node_id]
        end = kdtree_gpu.node_to_elements_offsets[node_id + 1]
        n_elements = end - start

        def check_element(j, inner_carry):
            """Check one element (bounded loop body)."""
            inner_found, inner_tests = inner_carry

            # Active only if: (1) not yet found, (2) j < actual elements, (3) not exceeded max tests
            active = (inner_found == -1) & (j < n_elements) & (inner_tests < max_tests_py)

            # Get element ID
            elem_idx = start + j
            test_elem_id = jnp.where(active, kdtree_gpu.node_to_elements_data[elem_idx], jnp.int32(0))

            # Point-in-tet test (masked by active)
            is_inside = jnp.where(
                active,
                point_in_tet_gpu(
                    pos,
                    test_elem_id,
                    kdtree_gpu.connectivity,
                    kdtree_gpu.node_positions,
                    method=config.POINT_IN_TET_METHOD
                ),
                False
            )

            # Update found element if inside and active
            new_found = jnp.where(is_inside & active, test_elem_id, inner_found)
            new_tests = inner_tests + jnp.where(active, 1, 0)

            return (new_found, new_tests)

        # Bounded loop over elements in this node (max 256 per node to prevent explosion)
        n_to_test = jnp.minimum(n_elements, jnp.int32(256))
        found_elem, n_tests_total = lax.fori_loop(
            0, n_to_test, check_element, (found_elem, n_tests_total)
        )

        return (found_elem, n_tests_total)

    # Loop over K nearest nodes
    found_elem, _ = lax.fori_loop(
        0, k_nearest_py, search_one_node, (jnp.int32(-1), jnp.int32(0))
    )

    return found_elem


def search_L2_kdtree_batch(
    positions: jax.Array,
    kdtree_gpu: NodeKDTreeGPU,
    k_nearest: jnp.int32 = jnp.int32(3),
    max_tests: jnp.int32 = jnp.int32(256)
) -> jax.Array:
    """
    Batch KD-tree search.

    Strategy: Query all particles at once (jaxkd supports batch queries),
    then use vmap to process each particle's nearest nodes.

    Args:
        positions: (n_particles, 3) float32 - query positions
        kdtree_gpu: GPU-resident KD-tree structure
        k_nearest: number of nearest nodes to search (int or jnp.int32)
        max_tests: maximum element tests (int or jnp.int32)

    Returns:
        elem_ids: (n_particles,) int32 - found elements (-1 if not found)
    """
    # Convert JAX scalars to Python int (jaxkd expects Python int, not traced)
    k_nearest_py = int(k_nearest)
    max_tests_py = int(max_tests)

    # Convert positions to float64 (jaxkd requirement for dtype consistency)
    positions_f64 = positions.astype(jnp.float64)

    # Query ALL particles at once (jaxkd supports batch queries)
    # This avoids vmapping over the KD-tree query (which has Python control flow)
    nearest_node_ids_batch, distances_batch = jk.query_neighbors(
        kdtree_gpu.kdtree, positions_f64, k=k_nearest_py
    )
    # nearest_node_ids_batch: (n_particles, k_nearest)
    # distances_batch: (n_particles, k_nearest)

    # Vmap the traceable search function
    elem_ids = jax.vmap(
        lambda pos, node_ids: _search_kdtree_with_prequeried_nodes(
            pos, node_ids, kdtree_gpu, k_nearest_py, max_tests_py
        ),
        in_axes=(0, 0)
    )(positions, nearest_node_ids_batch)

    return elem_ids
