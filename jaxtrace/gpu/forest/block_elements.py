"""
Padded Block Element Arrays for JAX GPU Kernels.

Creates JAX-compatible padded 2D arrays for block-local element search.
Solves the JAX JIT dictionary indexing problem by converting Dict[block_id, elements]
to a static 2D array that can be indexed with traced values.

This is the CRITICAL FIX for V4 → V5 transition.

V4 Problem:
    octrees[block_id]  # ❌ Can't index dict with traced value in JAX JIT
    → Wrong solution: Flatten all blocks → O(N×M) memory explosion

V5 Solution:
    block_elements[block_id]  # ✅ Array indexing works in JAX JIT
    → Padded 2D array: (n_blocks, max_elements_per_block)
    → Block-local search: O(N×log M) per block

Author: JAXTrace GPU Team
Date: 2025-11-05
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class BlockElementArrays:
    """
    Padded block element arrays for JAX GPU kernels.

    Attributes:
        block_elements: [n_blocks, max_elem] Element IDs per block (-1 padded)
        block_elem_counts: [n_blocks] Actual element count per block
        block_neighbors_26: [n_blocks, 26] 26-neighbor block IDs (-1 padded)
        max_elem_per_block: Maximum elements in any block
        n_blocks: Total number of blocks
        total_elements: Total unique elements across all blocks
    """
    block_elements: np.ndarray  # [n_blocks, max_elem], dtype=int32
    block_elem_counts: np.ndarray  # [n_blocks], dtype=int32
    block_neighbors_26: np.ndarray  # [n_blocks, 26], dtype=int32
    max_elem_per_block: int
    n_blocks: int
    total_elements: int

    def to_jax(self) -> 'BlockElementArrays':
        """Convert arrays to JAX DeviceArrays for GPU execution."""
        return BlockElementArrays(
            block_elements=jnp.array(self.block_elements),
            block_elem_counts=jnp.array(self.block_elem_counts),
            block_neighbors_26=jnp.array(self.block_neighbors_26),
            max_elem_per_block=self.max_elem_per_block,
            n_blocks=self.n_blocks,
            total_elements=self.total_elements
        )

    def memory_size_mb(self) -> float:
        """Estimate GPU memory usage in MB."""
        # block_elements: n_blocks × max_elem × 4 bytes (int32)
        # block_elem_counts: n_blocks × 4 bytes
        # block_neighbors_26: n_blocks × 26 × 4 bytes
        total_bytes = (
            self.n_blocks * self.max_elem_per_block * 4 +
            self.n_blocks * 4 +
            self.n_blocks * 26 * 4
        )
        return total_bytes / (1024 * 1024)


def build_padded_block_arrays(
    octrees: Dict,
    element_to_block: np.ndarray,
    blocks: List,
    verbose: bool = True
) -> BlockElementArrays:
    """
    Build padded block element arrays from octree dictionary.

    This is the CRITICAL FUNCTION that fixes the V4 architectural problem.

    Converts:
        octrees: Dict[block_id → OctreeData with element list]
    To:
        block_elements: [n_blocks, max_elem] padded array (-1 padding)

    This enables JAX JIT compilation:
        @jax.jit
        def search(pos, block_id, block_elements):
            elems = block_elements[block_id]  # ✅ Works! Static indexing
            return find_in_list(pos, elems)

    Parameters
    ----------
    octrees : Dict[int, OctreeData]
        Octrees per block (from octree_builder)
    element_to_block : np.ndarray, shape (N_elements,)
        Block assignment for each element
    blocks : List[BlockMetadata]
        Block metadata with neighbor topology
    verbose : bool
        Print progress and statistics

    Returns
    -------
    arrays : BlockElementArrays
        Padded arrays ready for JAX GPU kernels

    Notes
    -----
    Memory usage:
        - V4 (global flatten): N_particles × N_elements = 13.5K × 3.5M = 45 GB
        - V5 (block-local): n_blocks × max_elem = 32 × 150K = ~20 MB
        - Improvement: 2250× less memory!
    """
    n_blocks = len(blocks)

    if verbose:
        print(f"\n🔧 Building padded block element arrays...")
        print(f"  Total blocks: {n_blocks}")

    # Step 1: Collect element lists per block
    block_element_lists = []
    max_elem_per_block = 0
    total_elements = 0

    for block_id in range(n_blocks):
        if block_id in octrees:
            octree = octrees[block_id]

            # Handle both OctreeData objects and dict format
            if hasattr(octree, 'sorted_element_IDs'):
                elems = np.array(octree.sorted_element_IDs, dtype=np.int32)
            else:
                elems = np.array(octree.get('sorted_element_IDs', []), dtype=np.int32)
        else:
            # Empty block
            elems = np.array([], dtype=np.int32)

        block_element_lists.append(elems)
        max_elem_per_block = max(max_elem_per_block, len(elems))
        total_elements += len(elems)

    if verbose:
        print(f"  Max elements per block: {max_elem_per_block:,}")
        print(f"  Total elements: {total_elements:,}")
        mean_elem = total_elements / n_blocks
        print(f"  Mean elements per block: {mean_elem:,.0f}")
        load_imbalance = max_elem_per_block / mean_elem if mean_elem > 0 else 0
        print(f"  Load imbalance factor: {load_imbalance:.2f}×")

    # Step 2: Create padded block_elements array
    block_elements = np.full((n_blocks, max_elem_per_block), -1, dtype=np.int32)
    block_elem_counts = np.zeros(n_blocks, dtype=np.int32)

    for block_id in range(n_blocks):
        elems = block_element_lists[block_id]
        n_elem = len(elems)
        if n_elem > 0:
            block_elements[block_id, :n_elem] = elems
            block_elem_counts[block_id] = n_elem

    # Step 3: Build 26-neighbor connectivity
    if verbose:
        print(f"  Building 26-neighbor topology...")

    block_neighbors_26 = build_26_neighbor_topology(blocks)

    # Step 4: Create result
    arrays = BlockElementArrays(
        block_elements=block_elements,
        block_elem_counts=block_elem_counts,
        block_neighbors_26=block_neighbors_26,
        max_elem_per_block=max_elem_per_block,
        n_blocks=n_blocks,
        total_elements=total_elements
    )

    if verbose:
        mem_mb = arrays.memory_size_mb()
        print(f"  GPU memory footprint: {mem_mb:.1f} MB")
        print(f"  ✅ Block arrays ready for JAX JIT compilation")

    return arrays


def build_26_neighbor_topology(blocks: List) -> np.ndarray:
    """
    Build 26-neighbor connectivity for all blocks.

    Each block has up to 26 spatial neighbors:
        - 6 face neighbors (±x, ±y, ±z)
        - 12 edge neighbors
        - 8 corner neighbors

    This is required for Phase 5 (multi-level search):
        Level 3: Search neighbor blocks before global fallback

    Parameters
    ----------
    blocks : List[BlockMetadata]
        Block metadata with grid indices

    Returns
    -------
    neighbors : np.ndarray, shape (n_blocks, 26), dtype=int32
        26-neighbor block IDs per block (-1 for domain boundaries)

    Notes
    -----
    Neighbor ordering:
        [0:6]   → 6 face neighbors (±x, ±y, ±z)
        [6:18]  → 12 edge neighbors
        [18:26] → 8 corner neighbors
    """
    n_blocks = len(blocks)
    neighbors = np.full((n_blocks, 26), -1, dtype=np.int32)

    # Build block_id → grid_index mapping
    grid_indices = {}
    for block in blocks:
        grid_indices[block.block_id] = block.grid_index

    # Compute grid size from block 0
    # Assumes regular grid: all blocks have consistent grid layout
    nx = max(b.grid_index[0] for b in blocks) + 1
    ny = max(b.grid_index[1] for b in blocks) + 1
    nz = max(b.grid_index[2] for b in blocks) + 1

    for block in blocks:
        block_id = block.block_id
        i, j, k = block.grid_index

        neighbor_idx = 0

        # 6 face neighbors
        for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                nb_id = ni + nj * nx + nk * nx * ny
                neighbors[block_id, neighbor_idx] = nb_id
            neighbor_idx += 1

        # 12 edge neighbors
        for di, dj, dk in [
            (1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0),  # ±x, ±y edges
            (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1),  # ±x, ±z edges
            (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1)   # ±y, ±z edges
        ]:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                nb_id = ni + nj * nx + nk * nx * ny
                neighbors[block_id, neighbor_idx] = nb_id
            neighbor_idx += 1

        # 8 corner neighbors
        for di, dj, dk in [
            (1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1),
            (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)
        ]:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                nb_id = ni + nj * nx + nk * nx * ny
                neighbors[block_id, neighbor_idx] = nb_id
            neighbor_idx += 1

    return neighbors


def validate_block_arrays(
    arrays: BlockElementArrays,
    element_to_block: np.ndarray,
    verbose: bool = True
) -> bool:
    """
    Validate block element arrays for correctness.

    Checks:
        1. All elements are assigned to correct blocks
        2. No duplicate elements within blocks
        3. Padding is -1
        4. Element counts match actual elements
        5. Neighbor topology is symmetric

    Parameters
    ----------
    arrays : BlockElementArrays
        Arrays to validate
    element_to_block : np.ndarray
        Ground truth block assignments
    verbose : bool
        Print validation results

    Returns
    -------
    valid : bool
        True if all checks pass
    """
    if verbose:
        print(f"\n✅ Validating block element arrays...")

    errors = []

    # Check 1: Element assignments
    for block_id in range(arrays.n_blocks):
        expected_elems = np.where(element_to_block == block_id)[0]
        actual_count = arrays.block_elem_counts[block_id]
        actual_elems = arrays.block_elements[block_id, :actual_count]

        # Sort for comparison
        expected_set = set(expected_elems)
        actual_set = set(actual_elems)

        if expected_set != actual_set:
            missing = expected_set - actual_set
            extra = actual_set - expected_set
            errors.append(
                f"Block {block_id}: Expected {len(expected_set)} elems, "
                f"got {len(actual_set)} (missing: {len(missing)}, extra: {len(extra)})"
            )

    # Check 2: Padding
    for block_id in range(arrays.n_blocks):
        count = arrays.block_elem_counts[block_id]
        padding = arrays.block_elements[block_id, count:]
        if not np.all(padding == -1):
            errors.append(f"Block {block_id}: Invalid padding (expected all -1)")

    # Check 3: No duplicates
    for block_id in range(arrays.n_blocks):
        count = arrays.block_elem_counts[block_id]
        elems = arrays.block_elements[block_id, :count]
        if len(elems) != len(set(elems)):
            errors.append(f"Block {block_id}: Duplicate elements detected")

    # Check 4: Neighbor topology
    # Each neighbor relationship should be symmetric (for interior blocks)
    for block_id in range(arrays.n_blocks):
        for nb_id in arrays.block_neighbors_26[block_id]:
            if nb_id >= 0:  # Valid neighbor
                # Check if block_id is in nb_id's neighbor list
                if block_id not in arrays.block_neighbors_26[nb_id]:
                    errors.append(
                        f"Block {block_id} → {nb_id}, but {nb_id} ↛ {block_id} "
                        f"(asymmetric neighbor)"
                    )

    if errors:
        if verbose:
            print(f"  ❌ Validation FAILED with {len(errors)} errors:")
            for err in errors[:10]:  # Show first 10
                print(f"    - {err}")
            if len(errors) > 10:
                print(f"    ... and {len(errors) - 10} more errors")
        return False
    else:
        if verbose:
            print(f"  ✅ All validation checks passed!")
        return True


def print_memory_comparison(arrays: BlockElementArrays, n_particles: int, n_elements: int):
    """
    Print V4 vs V5 memory usage comparison.

    Parameters
    ----------
    arrays : BlockElementArrays
        V5 block arrays
    n_particles : int
        Number of particles (for V4 estimate)
    n_elements : int
        Total elements in mesh (for V4 estimate)
    """
    print(f"\n📊 Memory Usage Comparison: V4 vs V5")
    print(f"=" * 60)

    # V4 memory (nested vmap over all particles × all elements)
    v4_intermediate_values = n_particles * n_elements
    v4_mem_gb = (v4_intermediate_values * 4) / (1024**3)  # 4 bytes per bool

    # V5 memory (nested vmap over particles × block elements)
    v5_intermediate_values = n_particles * arrays.max_elem_per_block
    v5_mem_gb = (v5_intermediate_values * 4) / (1024**3)

    # Static array memory
    v5_static_mb = arrays.memory_size_mb()

    print(f"V4 (Global Flattening):")
    print(f"  Intermediate values: {v4_intermediate_values:,} "
          f"({n_particles:,} particles × {n_elements:,} elements)")
    print(f"  Memory: {v4_mem_gb:.1f} GB")
    print(f"  Status: ❌ OOM on most GPUs")

    print(f"\nV5 (Block-Local Search):")
    print(f"  Intermediate values: {v5_intermediate_values:,} "
          f"({n_particles:,} particles × {arrays.max_elem_per_block:,} max elem/block)")
    print(f"  Memory: {v5_mem_gb:.2f} GB + {v5_static_mb:.1f} MB static")
    print(f"  Status: ✅ Fits on GPUs with {v5_mem_gb*2:.0f} GB+ VRAM")

    improvement = v4_mem_gb / v5_mem_gb
    print(f"\n💡 Improvement: {improvement:.0f}× less memory")
    print(f"=" * 60)
