"""
Configuration switches for JAXTrace GPU tracking optimizations.

This module centralizes all user-configurable optimization flags for RK4 tracking.
Modify these flags to enable/disable specific optimizations and compare performance.

Usage:
    import jaxtrace.config as config
    config.POINT_IN_TET_METHOD = "skala"  # Switch to Skala method
    config.USE_AABB_FILTER = True         # Enable AABB pre-filter
"""

# ============================================================================
# Point-in-Tetrahedron Method Selection
# ============================================================================

POINT_IN_TET_METHOD = "skala"
"""
Point-in-tetrahedron containment test method.

Options:
    "current" - Reference implementation (barycentric/Cramer's rule)
                ~145 FLOPs, baseline performance
                Use for: Validation, debugging

    "skala" - GPU-optimized cross products (Skala 2014)
              ~48 FLOPs, ~3× speedup over current
              Use for: General meshes, production

    "axis_aligned" - Specialized for axis-aligned meshes
                     ~12 FLOPs for axis-aligned tets, ~3.3-12× speedup
                     Use for: ThreadedA mesh (100% axis-aligned)

Performance comparison (ThreadedA mesh, 100K particles):
┌─────────────────┬───────────────────┬─────────────────────┐
│ Method          │ Expected Throughput│ Expected Retention  │
├─────────────────┼───────────────────┼─────────────────────┤
│ current         │ 19,000 p/s        │ 93.57% (baseline)   │
│ skala           │ 55,000-65,000 p/s │ 93.57% (same)       │
│ axis_aligned    │ 180,000-230,000 p/s│ 93.57% (same)      │
└─────────────────┴───────────────────┴─────────────────────┘

Note: All methods produce identical results (bit-for-bit agreement expected).
"""

# ============================================================================
# AABB Pre-Filter (Optional, Phase 3)
# ============================================================================

USE_AABB_FILTER = False
"""
Enable axis-aligned bounding box (AABB) pre-filter before point-in-tet test.

When enabled:
    - Compute element AABB from 4 vertex positions (~24 FLOPs)
    - Test if point is in AABB before expensive point-in-tet test (~12 FLOPs)
    - Skip point-in-tet test if AABB test fails (early rejection)

Performance:
    - Speedup: 10-30% for spatially coherent queries
    - Overhead: 36 FLOPs per test (AABB computation + test)
    - Best for: Large elements, distant queries

Memory:
    - Precomputed AABBs: 3.5M elements × 6 floats × 4 bytes = 84 MB
    - Runtime computation: 0 MB (compute on-the-fly)

WARNING: Precomputed AABBs may cause OOM in vmap/scan contexts.
         Recommend runtime computation (compute AABB on-the-fly).

Default: False (disabled to avoid OOM risk)
"""

USE_PRECOMPUTED_AABB = False
"""
Use precomputed AABB arrays (requires 84 MB for ThreadedA mesh).

If False: Compute AABB on-the-fly from vertex positions
If True: Use precomputed (element_bbox_min, element_bbox_max) arrays

WARNING: Precomputed arrays can cause OOM when vmapped over particles.
         JAX broadcasts precomputed arrays to all particles during compilation,
         creating huge intermediate buffers (e.g., 225K particles × 3.5M elements).

Recommendation: Keep False unless you've verified no OOM issues.
"""

# ============================================================================
# L2 Global Search Method Selection
# ============================================================================

L2_SEARCH_METHOD = "morton"
"""
L2 global search method selection.

Options:
    "morton" - Morton curve-based search (original implementation)
               Uses space-filling curve for spatial indexing
               Morton codes from ELEMENT CENTROIDS
               Methods: radius, incremental, neighbors, hierarchical

    "mesh_aligned_octree" - Direct mesh-aligned octree search
                           Extracts intrinsic octree from Kuhn mesh
                           Single-cell lookup (center cell only)
                           ~5.9 elements per cell
                           74.6% retention (elements span multiple cells)
                           Requires Kuhn tetrahedral mesh structure

    "mesh_aligned_neighbors" - Mesh-aligned octree with pre-computed neighbor table (Option B)
                              Extracts intrinsic octree + builds CPU neighbor table
                              Searches primary cell + 26 spatial neighbors at 8 levels
                              89.74% retention, 13.9 tests/particle
                              1,504 particles/sec (8× slower than baseline)
                              Stable execution, no JAX memory issues
                              Memory: 134 MB (base 83 MB + neighbor table 51 MB)
                              Best for: Production tracking with high searchability
                              Requires Kuhn tetrahedral mesh structure

    "mesh_aligned_morton" - Hybrid: Morton radius search over mesh cells (NEW)
                           Combines intrinsic mesh structure + proven radius search
                           Morton codes from CELL CENTERS (not element centroids)
                           Radius search handles elements spanning cells
                           Expected ~98% retention
                           Requires Kuhn tetrahedral mesh structure

    "kdtree" - KD-tree node-based search (NEW)
               Find K nearest nodes, test connected elements
               ~95-100% retention with K=3
               ~64 tests per particle (K=3 × ~21 elem/node)
               ⚠️  LIMITATION: Only works for BATCH searches (initial assignment)
               Cannot be used in vmapped RK4 tracking (KD-tree query not traceable)
               Requires jaxkd library: pip install jaxkd

Performance comparison (FLA mesh, 225K particles):
┌──────────────────────────────┬──────────────────┬─────────────────────┬────────────────┐
│ Method                       │ Retention        │ Tests per particle  │ Throughput     │
├──────────────────────────────┼──────────────────┼─────────────────────┼────────────────┤
│ morton (radius=2)            │ ~93-98%          │ ~536 (in 5 leaves)  │ 12,106 p/s     │
│ morton (incremental)         │ ~93-98%          │ ~536 (adaptive)     │ 12,000 p/s     │
│ mesh_aligned_octree          │ ~74.6%           │ ~5.9 (in 1 cell)    │ 12,106 p/s     │
│ mesh_aligned_neighbors       │ ~89.74%          │ ~13.9 (27 cells)    │ 1,504 p/s      │
│ mesh_aligned_morton (2)      │ ~98% (expected)  │ ~30 (in 5 cells)    │ TBD            │
│ kdtree (K=3)                 │ ~95-100%         │ ~64 (batch only)    │ TBD            │
└──────────────────────────────┴──────────────────┴─────────────────────┴────────────────┘

Note: mesh_aligned_* methods only work with Kuhn meshes (axis-aligned tets).
      Will fall back to morton if mesh structure is incompatible.
      kdtree only works for batch searches, not vmapped RK4 tracking.

Default: "morton" (production-validated, works with any mesh)
"""

# ============================================================================
# Mesh-Aligned Octree Configuration
# ============================================================================

MESH_ALIGNED_MULTI_CELL_REGISTRATION = False
"""
Enable multi-cell vertex registration for mesh-aligned octree.

Problem:
    Current single-cell registration has 88.59% retention over 100 RK4 steps
    because 100% of Kuhn elements span cell boundaries (vertices at cube corners).
    When particle crosses to adjacent cell, element not found → particle lost.

Solution:
    Multi-cell vertex registration registers each element in ALL cells its
    vertices touch (~4 cells per element), improving retention to ~95%+.

Trade-offs:
    Single-Cell Registration (current):
        - Memory: 37.5 MB
        - Elements per cell: ~5.9
        - Cells per element: ~1.0
        - Retention: 88.59% over 100 steps
        - Tests per particle: ~35 (direct + neighbors)

    Multi-Cell Vertex Registration (this option):
        - Memory: 135 MB (+97.5 MB increase)
        - Elements per cell: ~23.6
        - Cells per element: ~4.0
        - Expected retention: ~95%+ over 100 steps
        - Tests per particle: ~141 (direct + neighbors)

When to enable:
    - Enable if retention is more important than memory/performance
    - Enable for long tracking runs (>100 steps)
    - Disable for memory-constrained scenarios or short runs

Default: False (use single-cell registration)
"""

# ============================================================================
# Initial Assignment Method Selection
# ============================================================================

INITIAL_ASSIGNMENT_METHOD = "cascade_radius"
"""
Method used for initial particle-to-element assignment.

Options:
    "cascade_radius" - Progressive radius expansion using global Morton octree.
                       Starts with a small radius and expands only for unassigned
                       particles. Works with any mesh type.
                       Radii: 500 → 1000 → 2000 → 5000 → 10000 → 100000

    "mesh_aligned_octree_multi_local" - Direct 3×3×3 local search using the
                       mesh-aligned multi-cell octree (same function as in RK4 L2).
                       Searches 27 cells × 8 levels per particle.
                       Requires Kuhn tetrahedral mesh + multi-cell octree built.
                       Uses batched vmap to avoid GPU OOM (batch_size configurable).
                       Faster for Kuhn meshes; may miss particles outside mesh.

Default: "cascade_radius" (works with any mesh, robust fallback)
"""

INITIAL_ASSIGNMENT_BATCH_SIZE = 50000
"""
Batch size for 'mesh_aligned_octree_multi_local' initial assignment.

The 3×3×3 search vmapped over all particles at once causes OOM for large particle
counts. This splits the work into batches that compile and run within GPU memory.

Reduce if you hit OOM during initial assignment.
Default: 50000 (safe for most GPUs with ~25 GB VRAM)
"""

# ============================================================================
# L1 Neighbor Search Optimization (Phase 4)
# ============================================================================

L1_SMART_NEIGHBOR_ORDERING = False
"""
Enable smart neighbor ordering based on particle velocity direction.

When enabled:
    - Compute dot product between particle velocity and neighbor direction
    - Sort neighbors by dot product (test most aligned neighbor first)
    - Expected to reduce L1 iterations by 20-40%

Performance:
    - Speedup: 5-15% overall (L1 is 34% of runtime)
    - Overhead: ~30 FLOPs per particle (velocity normalization + 4 dot products)

Default: False (Phase 4 optimization)
"""

L1_ADAPTIVE_SKIP = False
"""
Adaptively skip L1 neighbor search based on particle history.

When enabled:
    - Track per-particle L1 hit rate over last N steps
    - If hit rate < threshold, skip L1 for that particle
    - Reduces wasted L1 searches for particles with low spatial coherence

Performance:
    - Speedup: 10-20% for low-coherence flows
    - Memory: 100K particles × 4 bytes = 400 KB (per-particle hit counter)

Default: False (Phase 4 optimization)
"""

L1_MAX_HOPS = 3
"""
Maximum number of neighbor hops for L1 search before falling back to L2.

Current: Always test 4 face neighbors (1 hop)
With smart ordering: Test up to L1_MAX_HOPS neighbors in order of alignment

Range: 1-4 (tetrahedral elements have 4 face neighbors)
Default: 3 (test 3 most aligned neighbors before L2 fallback)
"""

# ============================================================================
# Debugging and Profiling
# ============================================================================

PROFILE_POINT_IN_TET = False
"""
Enable per-method profiling for point-in-tet tests.

When enabled:
    - Count number of calls per method
    - Track total time spent in each method
    - Print summary statistics after each timestep

Overhead: ~5% (counter increments + host synchronization)
Default: False (disable in production)
"""

VALIDATE_METHOD_AGREEMENT = False
"""
Validate agreement between current and selected method (debugging).

When enabled:
    - Run both methods on every query
    - Assert bit-for-bit agreement
    - Raise exception on mismatch

Overhead: 2× (doubles point-in-tet cost)
Default: False (use only during development)
"""

# ============================================================================
# Configuration Validation
# ============================================================================

def validate_config():
    """Validate configuration consistency and warn about common issues."""

    valid_methods = ["current", "skala", "axis_aligned"]
    if POINT_IN_TET_METHOD not in valid_methods:
        raise ValueError(f"Invalid POINT_IN_TET_METHOD: {POINT_IN_TET_METHOD}. "
                        f"Valid options: {valid_methods}")

    if USE_PRECOMPUTED_AABB and USE_AABB_FILTER:
        print("⚠️  WARNING: USE_PRECOMPUTED_AABB=True may cause OOM in vmap/scan!")
        print("   Consider setting USE_PRECOMPUTED_AABB=False to compute AABBs on-the-fly.")

    if L1_MAX_HOPS < 1 or L1_MAX_HOPS > 4:
        raise ValueError(f"L1_MAX_HOPS must be in range [1, 4], got {L1_MAX_HOPS}")

    if VALIDATE_METHOD_AGREEMENT and POINT_IN_TET_METHOD == "current":
        print("⚠️  WARNING: VALIDATE_METHOD_AGREEMENT=True with POINT_IN_TET_METHOD='current' "
              "will compare current method with itself (no-op).")

# Run validation on import
validate_config()
