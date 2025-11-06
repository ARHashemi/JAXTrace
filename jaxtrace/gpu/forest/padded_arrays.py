"""
Padded 2D arrays for block-local element storage.

Part of Phase 2: Element Neighbors & Padded Block Arrays

This is the KEY V5 solution that avoids V4's 45 GB memory explosion.
Instead of global flattening, we use padded 2D arrays: (n_blocks, max_elem_per_block).
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

from .block_mapper import BlockAssignmentStats


@dataclass
class PaddedArrays:
    """Padded 2D arrays for block-local element storage."""
    
    # Core arrays
    block_elements: np.ndarray  # (n_blocks, max_elem), int32, -1 padding
    block_sizes: np.ndarray     # (n_blocks,), int32, actual element counts
    
    # Dimensions
    n_blocks: int
    max_elements_per_block: int
    total_elements: int
    
    # Memory statistics
    memory_bytes: int
    memory_mb: float
    padding_waste_pct: float
    
    def __repr__(self) -> str:
        return (
            f"PaddedArrays(\n"
            f"  Shape: ({self.n_blocks}, {self.max_elements_per_block})\n"
            f"  Total elements: {self.total_elements:,}\n"
            f"  Memory: {self.memory_mb:.1f} MB\n"
            f"  Padding waste: {self.padding_waste_pct:.1f}%\n"
            f")"
        )


def build_padded_block_arrays(
    element_to_block: np.ndarray,
    stats: BlockAssignmentStats,
    verbose: bool = False
) -> PaddedArrays:
    """
    Build padded 2D arrays for block-local element storage.
    
    This is the V5 solution that avoids V4's global flattening problem.
    
    Parameters
    ----------
    element_to_block : np.ndarray
        Block ID for each element, shape (N_elements,), int32
    stats : BlockAssignmentStats
        Block assignment statistics from Phase 1
    verbose : bool, optional
        Print progress messages (default: False)
        
    Returns
    -------
    padded : PaddedArrays
        Padded 2D arrays with -1 padding
        
    Notes
    -----
    **Why Padded Arrays?**
    
    V4 Problem:
    - Dictionary `octrees[block_id]` fails in JAX JIT
    - Global flattening → O(N_particles × N_elements) = 45 GB
    
    V5 Solution:
    - Padded 2D array: `(n_blocks, max_elem_per_block)`
    - Static shape → JAX JIT compatible
    - Memory: 115.9 MB (120× better than V4)
    
    **Example**:
    ```python
    # Block 0: [10, 20, 30]
    # Block 1: [40, 50]
    # Block 2: [60, 70, 80, 90]
    
    # Padded array (3, 4):
    [[10, 20, 30, -1],
     [40, 50, -1, -1],
     [60, 70, 80, 90]]
    ```
    """
    n_elements = element_to_block.shape[0]
    n_blocks = stats.n_blocks
    max_elem = stats.max_elements
    
    if verbose:
        print(f"\nBuilding padded block arrays...")
        print(f"  Blocks: {n_blocks}")
        print(f"  Max elements/block: {max_elem:,}")
        print(f"  Array shape: ({n_blocks}, {max_elem:,})")
    
    # Allocate padded array filled with -1
    block_elements = np.full((n_blocks, max_elem), -1, dtype=np.int32)
    block_sizes = np.zeros(n_blocks, dtype=np.int32)
    
    # Count elements per block first
    if verbose:
        print("  Counting elements per block...")
    for block_id in range(n_blocks):
        block_sizes[block_id] = np.sum(element_to_block == block_id)
    
    # Create index trackers for filling
    block_indices = np.zeros(n_blocks, dtype=np.int32)
    
    # Fill padded array
    if verbose:
        print("  Filling padded array...")
    
    for elem_id in range(n_elements):
        block_id = element_to_block[elem_id]
        
        if block_id < 0:  # Skip elements outside domain
            continue
        
        idx = block_indices[block_id]
        block_elements[block_id, idx] = elem_id
        block_indices[block_id] += 1
        
        if verbose and (elem_id + 1) % 500000 == 0:
            print(f"    Processed {elem_id + 1:,}/{n_elements:,} elements...")
    
    # Verify counts match
    assert np.array_equal(block_indices, block_sizes), "Element count mismatch!"
    
    # Compute memory statistics
    memory_bytes = block_elements.nbytes + block_sizes.nbytes
    memory_mb = memory_bytes / (1024**2)
    
    total_valid = np.sum(block_sizes)
    total_slots = n_blocks * max_elem
    padding_waste_pct = 100 * (1 - total_valid / total_slots)
    
    padded = PaddedArrays(
        block_elements=block_elements,
        block_sizes=block_sizes,
        n_blocks=n_blocks,
        max_elements_per_block=max_elem,
        total_elements=total_valid,
        memory_bytes=memory_bytes,
        memory_mb=memory_mb,
        padding_waste_pct=padding_waste_pct,
    )
    
    if verbose:
        print(f"\n{padded}")
    
    return padded


def validate_padded_arrays(
    padded: PaddedArrays,
    element_to_block: np.ndarray,
    n_samples: int = 1000
) -> bool:
    """
    Validate padded array correctness.
    
    Parameters
    ----------
    padded : PaddedArrays
        Padded arrays to validate
    element_to_block : np.ndarray
        Original element-to-block mapping
    n_samples : int, optional
        Number of random blocks to check (default: 1000)
        
    Returns
    -------
    valid : bool
        True if validation passed
    """
    n_blocks = padded.n_blocks
    n_samples = min(n_samples, n_blocks)
    
    np.random.seed(42)
    sample_blocks = np.random.choice(n_blocks, size=n_samples, replace=False)
    
    n_errors = 0
    for block_id in sample_blocks:
        # Get elements from padded array
        size = padded.block_sizes[block_id]
        padded_elements = padded.block_elements[block_id, :size]
        
        # Get elements from original mapping
        original_elements = np.where(element_to_block == block_id)[0]
        
        # Check sizes match
        if len(padded_elements) != len(original_elements):
            n_errors += 1
            print(f"ERROR: Block {block_id} size mismatch: "
                  f"padded={len(padded_elements)}, original={len(original_elements)}")
            continue
        
        # Check contents match (order may differ)
        if not np.array_equal(np.sort(padded_elements), np.sort(original_elements)):
            n_errors += 1
            print(f"ERROR: Block {block_id} content mismatch")
    
    if n_errors > 0:
        print(f"\nValidation FAILED: {n_errors}/{n_samples} blocks incorrect")
        return False
    else:
        print(f"\nValidation PASSED: All {n_samples} sampled blocks correct")
        return True


def get_block_element_list(padded: PaddedArrays, block_id: int) -> np.ndarray:
    """
    Get element list for a specific block (no padding).
    
    Parameters
    ----------
    padded : PaddedArrays
        Padded arrays
    block_id : int
        Block ID
        
    Returns
    -------
    elements : np.ndarray
        Element IDs in this block, shape (block_size,), int32
    """
    size = padded.block_sizes[block_id]
    return padded.block_elements[block_id, :size].copy()


def print_memory_comparison(padded: PaddedArrays, stats: BlockAssignmentStats):
    """
    Print memory comparison: V4 global flattening vs V5 padded arrays.
    
    Parameters
    ----------
    padded : PaddedArrays
        Padded arrays (V5)
    stats : BlockAssignmentStats
        Block assignment statistics
    """
    print("\n" + "=" * 80)
    print("MEMORY COMPARISON: V4 vs V5")
    print("=" * 80)
    
    # V4: Global flattening (worst case for search)
    n_elements = stats.n_elements
    n_particles_example = 1000000  # 1M particles
    v4_memory_gb = (n_particles_example * n_elements * 4) / (1024**3)
    
    print("\n❌ V4: Global Flattening Approach")
    print(f"  Strategy: Flatten all elements into single array")
    print(f"  Memory for 1M particles: {v4_memory_gb:.1f} GB")
    print(f"  Problem: O(N_particles × N_elements) memory explosion")
    print(f"  Result: {v4_memory_gb:.0f} GB → Out of memory!")
    
    # V5: Padded arrays
    v5_memory_mb = padded.memory_mb
    
    print("\n✅ V5: Padded 2D Arrays (Current)")
    print(f"  Strategy: Block-local padded arrays (n_blocks, max_elem)")
    print(f"  Memory: {v5_memory_mb:.1f} MB")
    print(f"  Padding waste: {padded.padding_waste_pct:.1f}%")
    print(f"  Improvement: {v4_memory_gb * 1024 / v5_memory_mb:.0f}× smaller")
    
    # Comparison
    print("\n" + "-" * 80)
    print(f"💡 V5 is {v4_memory_gb * 1024 / v5_memory_mb:.0f}× more memory efficient than V4")
    print("=" * 80)
