"""
Configuration switches for JAXTrace GPU tracking optimizations.

This module centralizes all user-configurable optimization flags for RK4 tracking.
Modify these flags to enable/disable specific optimizations and compare performance.

Usage:
    import jaxtrace.config as config
    config.POINT_IN_TET_METHOD = "skala"  # Switch to Skala method
    config.USE_AABB_FILTER = True         # Enable AABB pre-filter
"""

import jax

# ============================================================================
# Floating-Point Precision
# ============================================================================

USE_FLOAT64 = False
"""
Use float64 (double precision) for all floating-point computations.

When True:
    - Enables JAX 64-bit mode (jax_enable_x64)
    - All mesh coordinates, velocities, particle positions use float64
    - GPU memory roughly doubles (~465 MB → ~930 MB for FLA mesh)
    - Better numerical accuracy for small domains and long tracking runs
    - ~1.7× slower due to doubled memory bandwidth

When False:
    - JAX default float32 precision
    - Lower GPU memory usage, ~1.7× faster
    - Sufficient with direct_inverse interpolation method

Default: False (float32, use --precision float64 for double precision)
"""

# The numpy/jnp dtype to use throughout JAXTrace
import numpy as np
import jax.numpy as jnp

# Initialize with current USE_FLOAT64 value
FLOAT_DTYPE_NP = np.float64 if USE_FLOAT64 else np.float32
FLOAT_DTYPE_JNP = jnp.float64 if USE_FLOAT64 else jnp.float32

# Numerical tolerances — tighter with float64 for better accuracy
POINT_IN_TET_TOLERANCE = 1e-10 if USE_FLOAT64 else 1e-6
"""
Tolerance for point-in-tetrahedron containment test.
Barycentric coordinates must be >= -tolerance to be considered inside.

float32: 1e-6 (safe for ~7 significant digits)
float64: 1e-10 (safe for ~15 significant digits, tighter containment)

Tighter tolerance reduces false-positive containment at element boundaries,
improving velocity interpolation accuracy for particles near element faces.
"""

DEGENERATE_ELEMENT_THRESHOLD = 1e-14 if USE_FLOAT64 else 1e-12
"""
Threshold for detecting degenerate tetrahedra (near-zero volume).
Elements with |det(M)| < threshold are treated as degenerate.

float32: 1e-12
float64: 1e-14 (tighter, detects fewer false degenerates)
"""

INTERPOLATION_DET_MIN = 1e-14 if USE_FLOAT64 else 1e-12
"""
Minimum determinant for Cramer's rule in barycentric velocity interpolation.
If |det| < threshold, det is clamped to avoid division by zero.

float32: 1e-12
float64: 1e-14
"""


def set_precision(use_float64: bool):
    """Set floating-point precision. Must be called before any JAX array creation.

    Updates USE_FLOAT64, dtype constants, tolerances, and enables jax_enable_x64.
    """
    global USE_FLOAT64, FLOAT_DTYPE_NP, FLOAT_DTYPE_JNP
    global POINT_IN_TET_TOLERANCE, DEGENERATE_ELEMENT_THRESHOLD, INTERPOLATION_DET_MIN

    USE_FLOAT64 = use_float64

    if USE_FLOAT64:
        jax.config.update("jax_enable_x64", True)

    FLOAT_DTYPE_NP = np.float64 if USE_FLOAT64 else np.float32
    FLOAT_DTYPE_JNP = jnp.float64 if USE_FLOAT64 else jnp.float32
    POINT_IN_TET_TOLERANCE = 1e-10 if USE_FLOAT64 else 1e-6
    DEGENERATE_ELEMENT_THRESHOLD = 1e-14 if USE_FLOAT64 else 1e-12
    INTERPOLATION_DET_MIN = 1e-14 if USE_FLOAT64 else 1e-12

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

OCTREE_REGISTRATION_METHOD = "parent_cube"
"""
Octree cell-to-element registration strategy for the mesh-aligned octree.

Options:
    "parent_cube" - Each element is registered in its ONE parent cube only.
                    The parent cube is the Kuhn hexahedral cell whose octree
                    subdivision produced this tetrahedron.
                    Elements per cell: 5-6 (Kuhn tets per cube), max ~8.
                    Non-Kuhn elements are assigned to their Kuhn neighbour's
                    cell, adding at most +1-2 per cell.
                    Combined with 3x3x3 neighbourhood search, this gives
                    100% found rate on Kuhn meshes.
                    Inner search loop is fully static (MAX_ELEMS_PER_CELL),
                    enabling XLA unrolling — critical for GPU performance.

    "vertex_multi" - Each element is registered in ALL cells its 4 vertices
                     touch (~4 cells per element).
                     Elements per cell: mean ~18, median 16, max ~129.
                     Inner search loop has dynamic bounds (CSR offsets),
                     preventing XLA unrolling — slower on AMD MI250X.
                     This was the original approach; retained for comparison.

Performance impact:
    parent_cube:  8 levels x 27 cells x 8 max_elems = 1,728 static iterations
    vertex_multi: 8 levels x 27 cells x dynamic(4-129) = up to 27,864, dynamic

Default: "parent_cube" (static inner loop, best GPU performance)
"""

MAX_ELEMS_PER_CELL = 32
"""
Maximum elements per cell for the static-bound inner search loop.

Only used when OCTREE_REGISTRATION_METHOD = "parent_cube".
The inner fori_loop in the 3x3x3 search uses this as a compile-time
constant upper bound, enabling XLA to unroll the loop.

Any parent-cube cell with more registered elements than this bound
gets its overflow SILENTLY TRUNCATED at search time, which manifests
as a small sigma-independent count of "search failed" queries. The
symptom is easy to miss because the failure count is a fixed hotspot
that doesn't grow with perturbation. See tests/paper_benchmarks
sec6_raw.log for the 8-vs-24 incident that motivated raising this
bound from 8 to 32.

Guidelines:
    - Pure Kuhn parent-cube registration (no non-Kuhn elements): the
      theoretical bound of Proposition 4.6 is 6, so 8 was originally
      considered sufficient.
    - Real meshes with any non-Kuhn elements handled by neighbour-
      borrowing (hybrid_non_kuhn=True): non-Kuhn elements pile into
      Kuhn neighbours, raising real max occupancy well above 6.
      Observed max on the cylA mesh (0.06% non-Kuhn): 24. Setting
      to 32 gives a safety margin.
    - Higher values increase the unrolled loop size proportionally.
      Total iterations per query per level = 27 cells x this value,
      so 32 -> 864 static iterations (vs 216 at 8). XLA still
      unrolls; empirical measurement on the cylA mesh at N_p=10000
      shows negligible per-step cost increase since the inner loop
      terminates early on empty cells.

The Alfeld-split extractor prints the actual observed max at build
time; if it exceeds this bound the extractor raises a hard error
rather than silently truncating. Adjust this constant upward if you
hit that error on a new mesh.

Default: 32 (safe margin above observed 24 max on cylA benchmark mesh)
"""

# ============================================================================
# RK4 Sub-Step Boundary Recovery
# ============================================================================

RK4_SUBSTEP_BBOX_CLAMP = False
"""
Clamp RK4 sub-step positions to the mesh bounding box before searching.

When enabled, intermediate RK4 positions (pos_k1, pos_k2, pos_k3) that overshoot
the mesh boundary are clamped back to the bounding box. This ensures the search
always has a chance of finding an element, preventing zero-velocity corruption.

Requires mesh_bbox_min and mesh_bbox_max to be passed to the RK4 constructor.

FEMUSS note: FEMUSS does NOT clamp substep positions. When a substage exits the
domain, the element search simply fails and k[i] = 0 (zero velocity for that
substage). This option is a JAXTrace extension for additional robustness.

Default: False
"""

RK4_FAILED_SUBSTAGE_POLICY = 'zero_vel'
"""
Policy when an RK4 substage element search fails (particle outside domain).

Options:
    'zero_vel'        - Use zero velocity for the failed substage (k[i] = 0).
                        The RK4 weighted sum naturally reduces the step size.
                        MATCHES FEMUSS RK4 BEHAVIOR: when element is not found,
                        FluidInteractionFunction is never called, so
                        ParticleInteractionQuantity stays 0 → k[i] = 0.

    'last_valid_vel'  - Reuse the previous substage's velocity.
                        k2 falls back to k1, k3→k2, k4→k3.
                        JAXTrace extension, NOT FEMUSS behavior.
                        Prevents velocity "holes" when particles temporarily
                        overshoot the mesh boundary during RK4 substages.

    'skip_step'       - If ANY substage search fails, discard the entire step.
                        Particle stays at its previous position (pos_final = pos).
                        Most conservative option, more aggressive than FEMUSS.

This policy applies AFTER RK4_SUBSTEP_BBOX_CLAMP (if enabled). Bbox clamping
runs before the search; this policy handles what happens when the search STILL
fails after clamping (or when clamping is disabled).

Default: 'zero_vel' (FEMUSS-equivalent)
"""

RK4_SUBSTEP_LAST_VALID_VEL = False
"""
DEPRECATED: Use RK4_FAILED_SUBSTAGE_POLICY = 'last_valid_vel' instead.

When True, overrides RK4_FAILED_SUBSTAGE_POLICY to 'last_valid_vel'.
Kept for backward compatibility.

Default: False
"""

# ============================================================================
# RK4 Boundary Projection (Final Position)
# ============================================================================

RK4_BOUNDARY_PROJECTION = False
"""
When the final-position search fails (particle exits mesh after RK4 integration),
clamp pos_final to the mesh bounding box (inset by RK4_BOUNDARY_PROJECTION_TOL)
and re-search. If the re-search succeeds, keep the particle alive at the
projected position.

This recovers particles that overshoot the mesh boundary by a small amount during
the full RK4 step. The clamped position lies just inside the mesh surface, which
is physically meaningful (the particle is projected back to the nearest boundary).

FEMUSS equivalent: kfl_keepParticlesInBoundingBox = .true.
FEMUSS clamps to MeshRange ± tol and marks ParticleHasLeftDomain = 1.

Requires mesh_bbox_min and mesh_bbox_max to be passed to the RK4 constructor.

Default: False
"""

RK4_BOUNDARY_PROJECTION_TOL = 1e-6
"""
Inward tolerance for boundary projection clamping.

When clamping to the bounding box, the position is pushed inward by this
tolerance to avoid landing exactly on the boundary face where point-in-tet
tests may be ambiguous.

FEMUSS uses a similar tolerance in its boundary clamping:
    Position(idime) = MeshRange(2,idime) - tol

Default: 1e-6 (matches FEMUSS)
"""

RK4_BOUNDARY_WALLS = None
"""
Per-wall control for boundary clamping and projection.

Affects both RK4_SUBSTEP_BBOX_CLAMP (substep clamping) and
RK4_BOUNDARY_PROJECTION (final position projection). Each wall can be
independently configured for different boundary treatments.

When None (default): all 6 walls use 'clamp' behaviour.

When set to a dict: keys from {'x_min','x_max','y_min','y_max','z_min','z_max'},
each mapping to a wall treatment:

    'clamp'  — Apply bbox clamping on this wall (substep and/or projection).
               Particles that overshoot are pushed back to the wall ± tolerance.
               This is the default when RK4_BOUNDARY_WALLS is None.

    'outlet' — No boundary treatment on this wall. Particles that exit through
               this wall get elem_id = -1 and are permanently lost.
               Physically correct for open boundaries / outlets.

Walls not listed in the dict default to 'clamp'.

Example — clamp all walls except x_max (outlet):
    RK4_BOUNDARY_WALLS = {
        'x_min': 'clamp', 'x_max': 'outlet',
        'y_min': 'clamp', 'y_max': 'clamp',
        'z_min': 'clamp', 'z_max': 'clamp',
    }

Example — all walls clamped (equivalent to None):
    RK4_BOUNDARY_WALLS = None

Default: None (all walls clamped)
"""

# ============================================================================
# Level-Set Velocity Masking
# ============================================================================

RK4_LEVELSET_MASK = False
"""
Enable level-set-based velocity masking for particles inside the tool.

When True, a nodal level-set array must be uploaded to GPU and passed to the
RK4 builder. At each RK4 substage, the level-set is interpolated at the
particle position using the same barycentric coordinates as velocity.
The behavior depends on RK4_LEVELSET_MODE.

FEMUSS reference (Mod_ParticleTracer.f90, line 1646):
    if (levelSetValue < 0.0_rp) AddFluidInteraction = .false.
In FEMUSS RK4 mode, this means k[i] = 0 for substages inside the tool,
which corresponds to RK4_LEVELSET_MODE = 'zero_vel'.

Requires: levelset_gpu array passed to the RK4 builder.
Default: False
"""

RK4_LEVELSET_MODE = 'zero_vel'
"""
Level-set masking mode (only applies when RK4_LEVELSET_MASK = True).

Options:
    'zero_vel'  - Zero velocity for substages inside tool (level-set < 0).
                  Each affected substage contributes k[i] = 0 to the RK4 sum.
                  Unaffected substages contribute normally.
                  MATCHES FEMUSS RK4 BEHAVIOR: ParticleInteractionQuantity
                  stays 0 when FluidInteractionFunction is not called, so
                  k[i] = 0.

    'skip_step' - If ANY RK4 substage is inside the tool (level-set < 0),
                  discard the ENTIRE step. Particle stays at its previous
                  position (pos_final = pos_start, elem_final = elem_start).
                  MORE AGGRESSIVE than FEMUSS RK4 (which only zeros the
                  affected substages, not the whole step).
                  Use case: prevent any tool-contaminated displacement from
                  accumulating in particles near the tool boundary.

Default: 'zero_vel' (FEMUSS-equivalent)
"""

RK4_L0_SKIP_BOUNDARY_ELEMENTS = True
"""
Skip L0 cached-element check for elements at the tool boundary (mixed level-set).

When True, elements whose nodes have BOTH positive and negative level-set values
are flagged as "boundary elements". L0 caching is bypassed for these elements,
forcing a fresh L1/L2 search every substage — matching FEMUSS behavior, which
always does a fresh octree search (no caching at all).

This fixes trajectory divergence near the tool: adjacent elements at the tool
boundary can have opposite level-set sign at the centroid, so L0 caching keeps
a particle in an element where velocity is applied (LS >= 0) while FEMUSS's
fresh search would find the adjacent element where velocity is zeroed (LS < 0).

Requires: RK4_LEVELSET_MASK = True and levelset_gpu array.
Only affects elements with mixed level-set sign (~11% of mesh typically).
Interior elements (all-positive or all-negative LS) still benefit from L0 caching.

Default: True (FEMUSS-equivalent)
"""

LEVELSET_FIELD_NAME = 'LEVEL'
"""
Name of the level-set field in PVTU files.
Default: 'LEVEL'
"""

# ============================================================================
# RK4 Stats Collection
# ============================================================================

RK4_COLLECT_STATS = True
"""
Collect per-step L0/L1/L2/miss search statistics during RK4 integration.

When True, uses search_l0_l1_l2_with_level (returns hit_level alongside elem_id)
and aggregates counts across all particles × 5 RK4 sub-step searches per step.

When False, uses search_l0_l1_l2_single (returns only elem_id). The level-tracking
code is completely eliminated from the compiled kernel at JIT trace time.

The overhead is small (5 int8 values per particle + 4 sum reductions per step),
but disabling may help in production runs where every microsecond counts.

Default: True
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
        print("WARNING: USE_PRECOMPUTED_AABB=True may cause OOM in vmap/scan!")
        print("   Consider setting USE_PRECOMPUTED_AABB=False to compute AABBs on-the-fly.")

    if L1_MAX_HOPS < 1 or L1_MAX_HOPS > 4:
        raise ValueError(f"L1_MAX_HOPS must be in range [1, 4], got {L1_MAX_HOPS}")

    if VALIDATE_METHOD_AGREEMENT and POINT_IN_TET_METHOD == "current":
        print("WARNING: VALIDATE_METHOD_AGREEMENT=True with POINT_IN_TET_METHOD='current' "
              "will compare current method with itself (no-op).")

    valid_levelset_modes = ('zero_vel', 'skip_step')
    if RK4_LEVELSET_MODE not in valid_levelset_modes:
        raise ValueError(f"Invalid RK4_LEVELSET_MODE: '{RK4_LEVELSET_MODE}'. "
                        f"Valid options: {valid_levelset_modes}")

    valid_policies = ('zero_vel', 'last_valid_vel', 'skip_step')
    if RK4_FAILED_SUBSTAGE_POLICY not in valid_policies:
        raise ValueError(f"Invalid RK4_FAILED_SUBSTAGE_POLICY: '{RK4_FAILED_SUBSTAGE_POLICY}'. "
                        f"Valid options: {valid_policies}")

    if RK4_SUBSTEP_LAST_VALID_VEL and RK4_FAILED_SUBSTAGE_POLICY != 'last_valid_vel':
        print(f"WARNING: RK4_SUBSTEP_LAST_VALID_VEL=True overrides "
              f"RK4_FAILED_SUBSTAGE_POLICY='{RK4_FAILED_SUBSTAGE_POLICY}' "
              f"to 'last_valid_vel'")

    if RK4_BOUNDARY_WALLS is not None and not isinstance(RK4_BOUNDARY_WALLS, dict):
        print(f"WARNING: RK4_BOUNDARY_WALLS should be None or a dict, "
              f"got '{RK4_BOUNDARY_WALLS}'. Will be treated as None (all clamp).")

# Run validation on import
validate_config()
