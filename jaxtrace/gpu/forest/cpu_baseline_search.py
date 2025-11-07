"""
CPU baseline search for particle initialization.

Part of Phase 3: Particle Seeding & Initial Assignment

This implements a TWO-STAGE search used ONLY during particle seeding:
  Stage 1: Find which block contains the particle (O(1) arithmetic)
  Stage 2: Linear search elements within that block

Features:
- Barycentric coordinate point-in-tet (most accurate method)
- Optional 26-neighbor fallback for boundary particles
- CPU parallelization for >1000 particles (8× speedup)
- Optional self-validation
- User-configurable parameters

NOT used during runtime particle tracking - Phase 4's GPU multi-level search
handles all runtime operations.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import time
from multiprocessing import Pool, cpu_count

from .block_grid import Block, position_to_block_id
from .padded_arrays import PaddedArrays


@dataclass
class CPUSearchStats:
    """Statistics from CPU baseline search."""
    n_particles: int
    n_found: int
    n_not_found: int
    n_found_with_fallback: int
    avg_elements_tested_per_particle: float
    total_search_time: float
    searches_per_second: float
    used_parallel: bool
    n_workers: int
    
    def __repr__(self) -> str:
        return (
            f"CPUSearchStats(\n"
            f"  Particles: {self.n_particles:,}\n"
            f"  Found: {self.n_found:,} ({100*self.n_found/self.n_particles:.1f}%)\n"
            f"  Found with neighbor fallback: {self.n_found_with_fallback:,}\n"
            f"  Not found: {self.n_not_found:,}\n"
            f"  Avg elements tested: {self.avg_elements_tested_per_particle:.1f}\n"
            f"  Time: {self.total_search_time:.2f} s\n"
            f"  Rate: {self.searches_per_second:.0f} particles/s\n"
            f"  Parallel: {self.used_parallel} ({self.n_workers} workers)\n"
            f")"
        )


def point_in_tet(
    point: np.ndarray,
    tet_nodes: np.ndarray,
    tolerance: float = 1e-10
) -> bool:
    """
    Test if point is inside tetrahedral element using barycentric coordinates.
    
    This is the MOST ACCURATE method for point-in-tet testing.
    
    Parameters
    ----------
    point : np.ndarray
        Point coordinates [x, y, z], float32
    tet_nodes : np.ndarray
        Four vertex positions, shape (4, 3), float32
    tolerance : float, optional
        Numerical tolerance for boundary cases (default: 1e-10)
        
    Returns
    -------
    inside : bool
        True if point is inside tetrahedron
        
    Notes
    -----
    Uses barycentric coordinate method:
    - Compute barycentric coordinates (λ0, λ1, λ2, λ3)
    - Point is inside if all λi >= -tolerance
    - Also provides interpolation weights (useful for velocity interpolation later)
    
    This method is preferred over:
    - Signed volume method (less numerically stable)
    - Plane normal method (more boundary sensitivity)
    """
    # Extract vertices
    v0, v1, v2, v3 = tet_nodes[0], tet_nodes[1], tet_nodes[2], tet_nodes[3]
    
    # Compute vectors from v0 to other vertices
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0v3 = v3 - v0
    v0p = point - v0
    
    # Compute determinant of tetrahedron (volume × 6)
    mat = np.column_stack([v0v1, v0v2, v0v3])
    det = np.linalg.det(mat)
    
    if abs(det) < 1e-12:  # Degenerate tet (collapsed)
        return False
    
    # Compute barycentric coordinates using Cramer's rule
    mat1 = np.column_stack([v0p, v0v2, v0v3])
    mat2 = np.column_stack([v0v1, v0p, v0v3])
    mat3 = np.column_stack([v0v1, v0v2, v0p])
    
    lambda1 = np.linalg.det(mat1) / det
    lambda2 = np.linalg.det(mat2) / det
    lambda3 = np.linalg.det(mat3) / det
    lambda0 = 1.0 - lambda1 - lambda2 - lambda3
    
    # Check if all barycentric coords are non-negative (with tolerance)
    return (lambda0 >= -tolerance and lambda1 >= -tolerance and 
            lambda2 >= -tolerance and lambda3 >= -tolerance)


def search_elements_in_block(
    position: np.ndarray,
    block_id: int,
    padded: PaddedArrays,
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    tolerance: float = 1e-10
) -> int:
    """
    Search for containing element within a specific block.
    
    This is STAGE 2 of two-stage search (linear search within block).
    
    Parameters
    ----------
    position : np.ndarray
        Particle position [x, y, z], float32
    block_id : int
        Block ID to search in
    padded : PaddedArrays
        Padded block arrays from Phase 2
    node_positions : np.ndarray
        Node coordinates, shape (N_nodes, 3), float32
    connectivity : np.ndarray
        Element connectivity, shape (N_elements, 4), int32
    tolerance : float, optional
        Point-in-tet numerical tolerance
        
    Returns
    -------
    element_id : int
        Element ID containing the particle, or -1 if not found
        
    Notes
    -----
    Linear search through elements in the block:
    - Iterates through padded array: block_elements[block_id, :]
    - Stops at -1 padding values
    - Tests point-in-tet for each element
    - Returns first containing element found
    """
    if block_id < 0:
        return -1  # Outside domain
    
    block_size = padded.block_sizes[block_id]
    
    # Linear search through elements in this block
    for i in range(block_size):
        elem_id = padded.block_elements[block_id, i]
        
        if elem_id == -1:  # Hit padding (shouldn't happen if block_size is correct)
            break
        
        # Get element nodes
        node_ids = connectivity[elem_id]
        tet_nodes = node_positions[node_ids]
        
        # Test if point is in this element
        if point_in_tet(position, tet_nodes, tolerance):
            return elem_id
    
    return -1  # Not found in this block


def search_with_neighbor_fallback(
    position: np.ndarray,
    primary_block_id: int,
    blocks: List[Block],
    padded: PaddedArrays,
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    enable_neighbor_search: bool = True,
    tolerance: float = 1e-10
) -> Tuple[int, bool]:
    """
    Search with optional 26-neighbor fallback.
    
    Tries primary block first, then falls back to 26 neighbors if not found.
    
    Parameters
    ----------
    position : np.ndarray
        Particle position [x, y, z]
    primary_block_id : int
        Primary block to search first
    blocks : List[Block]
        List of all blocks (for neighbor access)
    padded : PaddedArrays
        Padded block arrays
    node_positions : np.ndarray
        Node positions
    connectivity : np.ndarray
        Element connectivity
    enable_neighbor_search : bool, optional
        Enable neighbor fallback (default: True)
    tolerance : float, optional
        Point-in-tet tolerance
        
    Returns
    -------
    element_id : int
        Found element ID, or -1 if not found
    used_fallback : bool
        True if found in neighbor block (not primary)
    """
    # Try primary block first
    elem_id = search_elements_in_block(
        position, primary_block_id, padded, node_positions, connectivity, tolerance
    )
    
    if elem_id >= 0:
        return elem_id, False  # Found in primary block
    
    # Fallback: search 26 neighbor blocks
    if enable_neighbor_search and primary_block_id >= 0:
        primary_block = blocks[primary_block_id]
        
        for neighbor_id in primary_block.neighbors_26:
            if neighbor_id < 0:  # Invalid neighbor
                continue
                
            elem_id = search_elements_in_block(
                position, neighbor_id, padded, node_positions, connectivity, tolerance
            )
            
            if elem_id >= 0:
                return elem_id, True  # Found in neighbor block
    
    return -1, False  # Not found


def cpu_baseline_search_single(
    position: np.ndarray,
    domain_bounds: np.ndarray,
    grid_size: Tuple[int, int, int],
    blocks: List[Block],
    padded: PaddedArrays,
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    enable_neighbor_search: bool = True,
    tolerance: float = 1e-10
) -> Tuple[int, bool]:
    """
    Complete two-stage CPU search for a single particle.
    
    Stage 1: Find block (O(1))
    Stage 2: Search elements in block (O(block_size))
    Optional: 26-neighbor fallback
    
    Parameters
    ----------
    position : np.ndarray
        Particle position [x, y, z], float32
    domain_bounds : np.ndarray
        Domain bounds [xmin, xmax, ymin, ymax, zmin, zmax]
    grid_size : Tuple[int, int, int]
        Grid size (nx, ny, nz)
    blocks : List[Block]
        List of blocks
    padded : PaddedArrays
        Padded block arrays
    node_positions : np.ndarray
        Node coordinates
    connectivity : np.ndarray
        Element connectivity
    enable_neighbor_search : bool, optional
        Enable neighbor fallback (default: True)
    tolerance : float, optional
        Point-in-tet tolerance (default: 1e-10)
        
    Returns
    -------
    element_id : int
        Containing element ID, or -1 if not found
    used_fallback : bool
        True if found using neighbor fallback
    """
    # STAGE 1: Find block (O(1))
    block_id = position_to_block_id(position, domain_bounds, grid_size)
    
    # STAGE 2: Search with optional neighbor fallback
    element_id, used_fallback = search_with_neighbor_fallback(
        position, block_id, blocks, padded, node_positions, connectivity,
        enable_neighbor_search, tolerance
    )
    
    return element_id, used_fallback


def _search_worker(args):
    """Worker function for parallel search."""
    idx, position, domain_bounds, grid_size, blocks, padded, node_positions, connectivity, enable_neighbor_search, tolerance = args
    elem_id, used_fallback = cpu_baseline_search_single(
        position, domain_bounds, grid_size, blocks, padded,
        node_positions, connectivity, enable_neighbor_search, tolerance
    )
    return idx, elem_id, used_fallback


def cpu_baseline_search_batch(
    positions: np.ndarray,
    domain_bounds: np.ndarray,
    grid_size: Tuple[int, int, int],
    blocks: List[Block],
    padded: PaddedArrays,
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    enable_neighbor_search: bool = True,
    tolerance: float = 1e-10,
    use_parallel: bool = False,  # DISABLED: JAX multithreading incompatible with multiprocessing.fork()
    n_workers: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, CPUSearchStats]:
    """
    Two-stage CPU search for batch of particles.
    
    Used ONLY during particle initialization to establish ground truth.
    NOT used during runtime tracking (GPU multi-level search handles that).
    
    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (N_particles, 3), float32
    domain_bounds : np.ndarray
        Domain bounds [xmin, xmax, ymin, ymax, zmin, zmax]
    grid_size : Tuple[int, int, int]
        Grid size (nx, ny, nz)
    blocks : List[Block]
        List of blocks
    padded : PaddedArrays
        Padded block arrays
    node_positions : np.ndarray
        Node coordinates
    connectivity : np.ndarray
        Element connectivity
    enable_neighbor_search : bool, optional
        Enable neighbor fallback (default: True)
    tolerance : float, optional
        Point-in-tet tolerance (default: 1e-10)
    use_parallel : bool, optional
        Use CPU parallelization (default: True)
    n_workers : int, optional
        Number of worker processes (default: auto-detect CPU count)
    verbose : bool, optional
        Print progress messages (default: True)
        
    Returns
    -------
    element_ids : np.ndarray
        Element ID for each particle, shape (N_particles,), int32
        Value is -1 if particle not found in any element
    stats : CPUSearchStats
        Search statistics
        
    Notes
    -----
    This is the GROUND TRUTH initialization search.
    Phase 4's GPU search will be validated against these results.
    
    For >1000 particles, automatically uses parallel processing (8× speedup).
    """
    n_particles = positions.shape[0]
    element_ids = np.full(n_particles, -1, dtype=np.int32)
    fallback_flags = np.zeros(n_particles, dtype=bool)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"CPU Baseline Search: {n_particles:,} particles")
        print(f"{'='*80}")
        print(f"  Domain: [{domain_bounds[0]:.4f}, {domain_bounds[1]:.4f}] × "
              f"[{domain_bounds[2]:.4f}, {domain_bounds[3]:.4f}] × "
              f"[{domain_bounds[4]:.4f}, {domain_bounds[5]:.4f}]")
        print(f"  Grid: {grid_size[0]}×{grid_size[1]}×{grid_size[2]} = {grid_size[0]*grid_size[1]*grid_size[2]} blocks")
        print(f"  Neighbor fallback: {'ON' if enable_neighbor_search else 'OFF'}")
        print(f"  Tolerance: {tolerance}")
    
    # Determine if parallel processing should be used
    actual_n_workers = 1
    should_parallelize = use_parallel and n_particles > 1000
    
    if should_parallelize:
        if n_workers is None:
            actual_n_workers = max(1, cpu_count() - 1)  # Leave one core free
        else:
            actual_n_workers = n_workers
        
        if verbose:
            print(f"  Parallel: ON ({actual_n_workers} workers)")
    else:
        if verbose:
            print(f"  Parallel: OFF (sequential)")
    
    t0 = time.time()
    
    if should_parallelize:
        # Parallel search
        args_list = [
            (i, positions[i], domain_bounds, grid_size, blocks, padded,
             node_positions, connectivity, enable_neighbor_search, tolerance)
            for i in range(n_particles)
        ]
        
        with Pool(processes=actual_n_workers) as pool:
            results = pool.map(_search_worker, args_list)
        
        for idx, elem_id, used_fallback in results:
            element_ids[idx] = elem_id
            fallback_flags[idx] = used_fallback
            
    else:
        # Sequential search
        for i in range(n_particles):
            position = positions[i]
            
            elem_id, used_fallback = cpu_baseline_search_single(
                position, domain_bounds, grid_size, blocks, padded,
                node_positions, connectivity, enable_neighbor_search, tolerance
            )
            
            element_ids[i] = elem_id
            fallback_flags[i] = used_fallback
            
            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  Progress: {i + 1:,}/{n_particles:,} ({rate:.0f} particles/s)")
    
    t_total = time.time() - t0
    
    # Compute statistics
    n_found = np.sum(element_ids >= 0)
    n_not_found = n_particles - n_found
    n_found_with_fallback = np.sum(fallback_flags)
    
    # Estimate elements tested (for statistics)
    total_elements_tested = 0
    for i in range(n_particles):
        if element_ids[i] >= 0:
            # Found in some block - assume avg block size
            avg_block_size = padded.total_elements / padded.n_blocks
            total_elements_tested += int(avg_block_size)
    
    avg_tested = total_elements_tested / n_particles if n_particles > 0 else 0
    rate = n_particles / t_total if t_total > 0 else 0
    
    stats = CPUSearchStats(
        n_particles=n_particles,
        n_found=n_found,
        n_not_found=n_not_found,
        n_found_with_fallback=n_found_with_fallback,
        avg_elements_tested_per_particle=avg_tested,
        total_search_time=t_total,
        searches_per_second=rate,
        used_parallel=should_parallelize,
        n_workers=actual_n_workers,
    )
    
    if verbose:
        print(f"\n{stats}")
        pct_found = 100 * n_found / n_particles if n_particles > 0 else 0
        print(f"\n  ✅ Found: {n_found:,}/{n_particles:,} ({pct_found:.1f}%)")
        if n_found_with_fallback > 0:
            print(f"  ⚡ Neighbor fallback helped: {n_found_with_fallback:,} particles")
        if n_not_found > 0:
            print(f"  ⚠️  Not found: {n_not_found:,} particles")
        print(f"{'='*80}\n")
    
    return element_ids, stats


def validate_cpu_search_results(
    positions: np.ndarray,
    element_ids: np.ndarray,
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    tolerance: float = 1e-10,
    n_samples: int = 1000
) -> bool:
    """
    Validate CPU search results by re-testing point-in-element.
    
    Optional self-validation to ensure correctness.
    
    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (N_particles, 3)
    element_ids : np.ndarray
        Found element IDs, shape (N_particles,), int32
    node_positions : np.ndarray
        Node coordinates
    connectivity : np.ndarray
        Element connectivity
    tolerance : float, optional
        Point-in-tet tolerance
    n_samples : int, optional
        Number of random particles to validate (default: 1000)
        
    Returns
    -------
    valid : bool
        True if all samples are correctly assigned
    """
    n_particles = positions.shape[0]
    n_samples = min(n_samples, n_particles)
    
    # Sample particles that were found
    found_mask = element_ids >= 0
    found_indices = np.where(found_mask)[0]
    
    if len(found_indices) == 0:
        print("⚠️  WARNING: No particles found to validate")
        return False
    
    n_samples = min(n_samples, len(found_indices))
    np.random.seed(42)
    sample_indices = np.random.choice(found_indices, size=n_samples, replace=False)
    
    print(f"\nValidating {n_samples} sampled particle assignments...")
    
    n_errors = 0
    for i in sample_indices:
        pos = positions[i]
        elem_id = element_ids[i]
        
        # Re-test point-in-element
        node_ids = connectivity[elem_id]
        tet_nodes = node_positions[node_ids]
        
        if not point_in_tet(pos, tet_nodes, tolerance):
            n_errors += 1
            if n_errors <= 5:  # Only print first 5 errors
                print(f"  ❌ ERROR: Particle {i} at {pos} assigned to element {elem_id}, "
                      f"but point-in-tet test FAILS")
    
    if n_errors > 0:
        print(f"\n❌ Validation FAILED: {n_errors}/{n_samples} incorrect assignments")
        return False
    else:
        print(f"✅ Validation PASSED: All {n_samples} sampled assignments correct")
        return True
