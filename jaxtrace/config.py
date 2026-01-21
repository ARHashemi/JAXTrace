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
