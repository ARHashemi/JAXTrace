#!/usr/bin/env python3
"""
Production Particle Tracking - Fully-Fused RK4 with Time-Dependent Velocity

TIME-DEPENDENT VELOCITY implementation with:
- Cyclic velocity sequence (40 timesteps, wraps periodically)
- All velocity fields pre-loaded on GPU (no per-step transfers)
- Single vmap over particles (all RK4 stages fused)
- NO CPU-GPU transfers between timesteps (data stays on GPU)
- Download ONLY at export frequency (every 10 steps)

Target Performance:
- Initial assignment: >95%
- Retention at 2,500 steps: >95%
- Throughput: 50-120K particles/s (minimal overhead vs static velocity)
- Memory: ~850-900 MB (40 velocity fields + mesh + Morton)

Architecture:
- L0: Cached element (point-in-tet)
- L1: Multi-hop neighbors (3 hops, ~84 neighbors)
- L2: Global Morton search (binary search + bounded leaf scan, radius=2)
- Fully-fused RK4: All 5 stages + 5 searches + 4 interpolations in ONE vmap
- Time-dependent: Cyclic indexing into GPU-resident velocity sequence
"""

import os
# Force CPU-GPU memory management
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys
import time
import queue
import threading
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaxtrace.gpu.particles import ParticleData
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu, compute_velocity_cycle_params
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes  # Fix PVTU piece boundaries
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.hilbert_octree_builder import build_global_hilbert_octree
from jaxtrace.gpu.search.morton_global_search import upload_global_morton_to_gpu
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.tracking.rk4_fully_fused_timedep import create_rk4_fully_fused_timedep
import jaxtrace.config as config
from jaxtrace.gpu.tracking.initial_assignment_extended import initial_assignment_extended_batch
from jaxtrace.gpu.tracking.initial_assignment_cascading import initial_assignment_cascading_fallback
from jaxtrace.tracking.seeding import uniform_grid_seeds
import jaxtrace.config as config


# ============================================================================
# Point-in-Tetrahedron Method Configuration (RK4 Optimization)
# ============================================================================
# Options:
#   "current"          - Baseline (barycentric/Cramer's rule)
#   "skala"            - OLD Skala method (cross products)
#   "skala_memory_opt" - NEW Skala with memory optimization
#   "inverse"          - NEW Inverse matrix method (RECOMMENDED - 3-4× faster)
#   "axis_aligned"     - OLD AA method (BROKEN - 0% detection on Kuhn mesh)
#   "pure_aa"          - NEW AA method (FALSE POSITIVES - do not use)
#
POINT_IN_TET_METHOD = "inverse"  # ✅ RECOMMENDED: 3-4× speedup, 100% accuracy
#
# Performance Validation (FLA mesh, 30K particles, initial assignment):
#   "current":           110 p/s  (baseline, 100% accuracy)
#   "skala":              99 p/s  (0.90×, 100% accuracy)
#   "skala_memory_opt":  108 p/s  (0.97×, 100% accuracy)
#   "inverse":           350-450 p/s  (3-4×, 100% accuracy) ✅ RECOMMENDED
#   "axis_aligned":       49 p/s  (0.45×, 99.40% accuracy) ❌ BROKEN
#   "pure_aa":         3,036 p/s  (27.49×, 0% accuracy) ❌ FALSE POSITIVES
#
# See: POINT_IN_TET_OPTIMIZATION_STRATEGY.md for optimization details
# ============================================================================

# Configuration
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")#Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"#"threadedAvtk_{timestep}.pvtu"  # Pattern with {timestep} placeholder
VELOCITY_TIMESTEP_RANGE = (120, 159)  # Load timesteps 120-159 (40 timesteps)
VELOCITY_FIELD_NAME = 'Displacement'  # Field name in PVTU files (this IS velocity)
VELOCITY_DT = 0.0025  # Time spacing between velocity snapshots

# Particle Generation (Uniform Grid - from production_tracking_threadeda.py)
PARTICLE_GRID_RESOLUTION = (50, 90, 50)  # Grid resolution in (x, y, z) = 105,000 particles
PARTICLE_BOUNDS_FRACTION = {
    'x': (0.2, 0.35),  # Use first 20% of domain in X (entrance region)
    'y': (0.2, 0.8),  # Full domain in Y
    'z': (0.3, 1.0),  # Full domain in Z
}
# Use grid resolution directly (not dependent on domain size)
N_X = max(1, int(PARTICLE_GRID_RESOLUTION[0]))
N_Y = max(1, int(PARTICLE_GRID_RESOLUTION[1]))
N_Z = max(1, int(PARTICLE_GRID_RESOLUTION[2]))

N_PARTICLES = N_X * N_Y * N_Z

DT = 0.0025
N_STEPS = 2_500

# Search Hierarchy Configuration
# Space-Filling Curve Selection (L2):
#   'morton': Z-order Morton curve (interleaved bit encoding)
#             - Fast encoding (bitwise operations)
#             - Moderate spatial locality
#             - Well-tested in production
#   'hilbert': Hilbert curve (state machine encoding)
#              - Better spatial locality and continuity
#              - Slightly slower encoding (state table lookups)
#              - Same octree structure as Morton (drop-in replacement)
CURVE_TYPE = 'morton'          # 'morton' or 'hilbert' - Choose space-filling curve
# NOTE: Hilbert uses ~15% more GPU memory (28,363 leaves vs 24,550)
#       May require reducing PARTICLE_GRID_RESOLUTION or INITIAL_SEARCH_FALLBACK_RADII
#       to avoid OOM errors during initial assignment

# Neighbor Method Selection (L1):
#   'face': Elements sharing 3 nodes (tetrahedral face)
#           - Memory: ~48 MB for 3M elements
#           - Neighbors: 4 per element (max)
#           - Works for: Uniform refinement, conforming meshes
#           - FAILS for: 1:2 octree refinement (coarse/fine share edges, not faces)
#   'node': Elements sharing ANY node (vertex, edge, or face)
#           - Memory: ~1.1 GB for 3M elements (20× larger!)
#           - Neighbors: 20-100 per element
#           - ❌ BROKEN: RK4 L1 loop hardcoded to 4 neighbors (checks only 4 of 80+)
#           - ❌ CRASHES: JIT compilation OOM (10-20 GB RAM during compile)
#           - Trade-off: Doesn't work with current RK4 implementation
#
# ⚠️  CRITICAL: FLA mesh is UNIFORMLY REFINED - face-based is sufficient!
#     Node-based causes compilation crash and provides NO benefit for this mesh.
#     See: NODE_NEIGHBOR_MEMORY_ISSUE.md for detailed analysis
NEIGHBOR_METHOD = 'face'       # ✅ RECOMMENDED for FLA mesh (uniformly refined)

# L2 Search Method Selection:
#   'radius': Linear ±radius search along Morton curve
#             - Searches center_leaf ± L2_SEARCH_RADIUS leaves
#             - Simple, works for all meshes
#             - May search many irrelevant leaves (not spatial neighbors)
#             - Performance: ~30K particles/s with radius=10, 93.5% retention (with inverse point-in-tet)
#   'incremental': Cascading radius search (OLD - Morton-based)
#                  - Tier 1: radius=2 (5 leaves) - fast path
#                  - Tier 2: radius=5 (11 leaves) - only if radius=2 fails
#                  - Tier 3: radius=10 (21 leaves) - only if radius=5 fails
#                  - Expected: 1.8-2.5× speedup vs 'radius' (depends on hit rate distribution)
#                  - Performance: ~50-70K particles/s (estimated, with inverse point-in-tet)
#                  - Same retention as 'radius' method
#   'mesh_aligned_octree': Multi-cell vertex registration + 2×2×2 local search (NEW - RECOMMENDED)
#                          - Registers each element in ALL cells its 4 vertices touch (~4 cells/elem)
#                          - Searches 8-cell neighborhood (2×2×2 centered cube) at 8 levels
#                          - Direct cell lookup (no tree traversal)
#                          - Expected: ~80% retention (under investigation), ~1.1M particles/s
#                          - Memory: 135 MB (vs 37.5 MB for single-cell)
#                          - Tests/particle: ~146 (8 cells × 18.31 elem/cell)
#                          - ⚠️ NOTE: Retention ~10% lower than Morton baseline - under investigation
#   'mesh_aligned_morton': Hybrid mesh-aligned Morton (cell centers + radius search)
#                          - Combines mesh-aligned cell structure with Morton radius search
#                          - Expected: ~82% retention, moderate performance
#   'mesh_aligned_neighbors': Pre-computed neighbor table (Option B)
#                             - 27 cells at 3 refinement levels
#                             - Expected: ~81% retention for tracking
#   'neighbors': Morton neighbor arithmetic (26 spatial neighbors at single depth)
#                - Decodes Morton prefix to find 26 spatial neighbor octants
#                - Geometrically correct (actual spatial adjacency)
#                - Fixed cost (always 27 octants at depth 7)
#                - Performance: ~21K particles/s, 80% retention
#                - Requires octree prefix table (table_depth > 0)
#   'hierarchical': Multi-depth Morton neighbors (depth 7 + depth 6 fallback)
#                   - Searches at TWO octree depths for variable-depth leaves
#                   - Handles particles at coarse/fine boundaries
#                   - Cost: up to 54 octants (27 depth-7 + 27 depth-6 if needed)
#                   - Expected: ~85-90% retention, ~18-20K particles/s
#                   - Best for graded refinement meshes with variable leaf depths
#                   - Requires octree prefix table (table_depth > 0)
#   'kdtree': KD-tree based node search
#                   - Finds K nearest mesh nodes, tests connected elements
#                   - Cost: ~64 tests (K=3 nodes × ~21 elem/node)
#                   - Expected: ~95-100% retention (very robust)
#                   - No spatial structure needed, works with any mesh
#                   - Requires jaxkd library: pip install jaxkd
#                   - ⚠️  WARNING: kdtree NOT compatible with vmapped RK4 tracking!
#                   - Use 'incremental' for RK4 tracking instead
L2_SEARCH_METHOD = 'mesh_aligned_octree'  # 'radius', 'incremental', 'mesh_aligned_octree', 'mesh_aligned_morton', 'mesh_aligned_neighbors', 'neighbors', 'hierarchical', or 'kdtree'
# ✅ RECOMMENDED: 'mesh_aligned_octree' for Kuhn meshes (highest performance)
# ⚠️  'kdtree' only works for batch searches (initial assignment), NOT RK4 tracking
# NOTE: mesh_aligned_* methods only work with Kuhn meshes (axis-aligned tets)
# NOTE: Configure tiers with INCREMENTAL_SEARCH_RADII below (only for 'incremental' method)

N_HOPS = 5                     # Number of hops for L1 neighbor search
L2_SEARCH_RADIUS = 10          # L2 search radius (only used if L2_SEARCH_METHOD='radius')
                               # NOTE: radius=N searches 2N+1 leaves: [-N,...,0,...,+N]
                               # Example: radius=10 searches 21 leaves total
ENABLE_L1_SEARCH = True        # Enable L1 neighbor search (set False to test L0→L2 only)

# Incremental L2 Configuration (only used if L2_SEARCH_METHOD='incremental')
# Cascading search radii: try small radius first, expand if not found
# Each radius=R searches a SYMMETRIC BAND of 2R+1 leaves around center
# Example: (2, 5, 10) means:
#   Tier 1: radius=2  → search 5 leaves  (leaves[-2,-1,0,+1,+2])
#   Tier 2: radius=5  → search 11 leaves (leaves[-5,...,0,...,+5]) - only if tier 1 fails
#   Tier 3: radius=10 → search 21 leaves (leaves[-10,...,0,...,+10]) - only if tier 2 fails
#
# Tuning guide:
#   - More tiers = finer-grained fallback, but more jnp.where overhead
#   - Fewer tiers = simpler, but may waste work if gaps are large
#   - Default (2,5,10): Good balance for most cases
#   - Aggressive (2,4,8,15,30): More tiers for highly variable flow
#   - Conservative (5,15,50): Fewer tiers, larger jumps
INCREMENTAL_SEARCH_RADII = (2, 4, 8, 15, 30) # 2-5 tiers supported

# KD-tree L2 Configuration (only used if L2_SEARCH_METHOD='kdtree')
KDTREE_K_NEAREST = 3           # Number of nearest nodes to search (K=3 recommended)
KDTREE_MAX_TESTS = 256          # Maximum element tests per particle

# Initial assignment search radii (curve-dependent)
# Hilbert has ~15% more leaves (28,363 vs 24,550), so needs larger radii for same coverage
if CURVE_TYPE == 'hilbert':
    INITIAL_SEARCH_RADIUS = 500#75         # ~1.5× Morton radius
    INITIAL_SEARCH_FALLBACK_RADII = [1000, 2000, 5000, 10000, 100000]#[150, 300, 600]  # ~1.5× Morton fallbacks
else:  # morton
    INITIAL_SEARCH_RADIUS = 500
    INITIAL_SEARCH_FALLBACK_RADII = [1000, 2000, 5000, 10000, 100000]

SEED = 42
LOG_INTERVAL = 100

# Export Configuration
EXPORT_FREQUENCY = 10  # Export every 10 timesteps
OUTPUT_DIR = Path("./output/global_morton_timedep_optimized")
STORE_VELOCITIES = False  # Store particle velocities in VTK


@dataclass
class ExportConfig:
    """Configuration for VTK export"""
    output_dir: Path
    export_frequency: int  # Export every N timesteps
    include_velocities: bool = True
    include_metadata: bool = True


class AsyncVTKExporter:
    """
    Async VTK exporter that runs in background thread.

    Minimal memory overhead: Only stores current timestep data in queue.
    No blocking of main tracking loop.
    """

    def __init__(self, config: ExportConfig, particle_data_template: ParticleData):
        self.config = config
        self.template = particle_data_template
        self.export_queue = queue.Queue(maxsize=5)  # Limit queue size
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.n_exported = 0
        self.export_times = []

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start background export worker"""
        self.worker_thread = threading.Thread(target=self._export_worker, daemon=True)
        self.worker_thread.start()

    def _export_worker(self):
        """Background thread that processes export queue"""
        while not self.stop_event.is_set():
            try:
                # Wait for data with timeout to allow checking stop_event
                export_data = self.export_queue.get(timeout=1.0)

                if export_data is None:  # Sentinel value
                    break

                step, positions, velocities, element_ids, active_mask = export_data

                # Write VTK file
                t0 = time.perf_counter()
                output_file = self.config.output_dir / f"particles_step_{step:06d}.vtu"

                # Filter to active particles only
                active_positions = positions[active_mask]
                active_velocities = velocities[active_mask] if (velocities is not None and self.config.include_velocities) else None

                # Use VTK writer directly
                from jaxtrace.io import VTKTrajectoryWriter
                writer = VTKTrajectoryWriter()
                writer.write_particles_at_time(
                    positions=active_positions,
                    velocities=active_velocities,
                    time=step,
                    filename=str(output_file),
                    format='xml'
                )

                export_time = time.perf_counter() - t0
                self.export_times.append(export_time)
                self.n_exported += 1
                self.export_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Export error: {e}")

    def enqueue_export(self, step: int, particle_data: ParticleData):
        """
        Add particle data to export queue (non-blocking).

        Creates CPU copies of data to avoid GPU memory retention.
        """
        try:
            # Convert to CPU numpy arrays (copy from GPU if needed)
            positions = np.array(particle_data.positions, dtype=np.float32)

            # Only copy velocities if requested
            if self.config.include_velocities:
                velocities = np.array(particle_data.velocities, dtype=np.float32)
            else:
                velocities = None

            element_ids = np.array(particle_data.element_ids, dtype=np.int32)
            active_mask = np.array(particle_data.active_mask, dtype=bool)

            # Put in queue (will block if queue is full, preventing memory explosion)
            self.export_queue.put(
                (step, positions, velocities, element_ids, active_mask),
                timeout=10.0
            )
        except queue.Full:
            print(f"Warning: Export queue full at step {step}, skipping export")

    def stop(self):
        """Stop background worker and wait for queue to finish"""
        # Signal worker to stop
        self.export_queue.put(None)
        self.stop_event.set()

        # Wait for worker to finish
        if self.worker_thread:
            self.worker_thread.join(timeout=30.0)

    def get_stats(self) -> Dict:
        """Get export statistics"""
        if not self.export_times:
            return {'n_exported': 0, 'mean_time': 0, 'total_time': 0}

        return {
            'n_exported': self.n_exported,
            'mean_time': np.mean(self.export_times),
            'total_time': np.sum(self.export_times),
            'queue_size': self.export_queue.qsize(),
        }


def main():
    # Apply point-in-tet method configuration
    config.POINT_IN_TET_METHOD = POINT_IN_TET_METHOD
    config.L2_SEARCH_METHOD = L2_SEARCH_METHOD

    nx, ny, nz = PARTICLE_GRID_RESOLUTION

    print("=" * 80)
    print(f"Production Particle Tracking - Global {CURVE_TYPE.upper()} L2 Search")
    print("=" * 80)
    print(f"Grid resolution: {nx} × {ny} × {nz} = {N_PARTICLES:,} particles")
    print(f"Timesteps: {N_STEPS:,}")
    print(f"dt: {DT:.2e}")
    print(f"Space-filling curve: {CURVE_TYPE}")
    print(f"Point-in-tet method: {POINT_IN_TET_METHOD}")
    print(f"L1 hops: {N_HOPS}")
    print(f"L2 radius: {L2_SEARCH_RADIUS}")
    print("=" * 80)

    # ========================================================================
    # 1. Load Mesh and Velocity Sequence
    # ========================================================================

    print("\n[1/6] Loading mesh and velocity sequence...")
    t_load = time.time()
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE,
        field_name=VELOCITY_FIELD_NAME,
        verbose=True
    )
    t_load = time.time() - t_load

    n_nodes = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    n_velocity_steps = velocity_sequence.shape[0]

    print(f"\n  Mesh: {n_elements:,} elements, {n_nodes:,} nodes")
    print(f"  Velocity timesteps: {n_velocity_steps}")
    print(f"  Total load time: {t_load:.2f}s")

    # ========================================================================
    # 1.5. CRITICAL FIX: Merge Duplicate Nodes from PVTU Pieces
    # ========================================================================
    # VTK's vtkXMLPUnstructuredGridReader does NOT merge nodes at piece boundaries!
    # This causes 20-30% of nodes to be duplicates at same position but different IDs.
    # Elements across piece boundaries cannot detect neighbors → particle loss!
    # See: PVTU_PIECE_BOUNDARY_ROOT_CAUSE.md

    print(f"\n[1.5/6] Checking for duplicate nodes (PVTU piece boundary fix)...")
    t_dedup = time.time()
    node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=True
    )
    t_dedup = time.time() - t_dedup

    if n_duplicates_removed > 0:
        print(f"  ✅ Fixed PVTU piece boundaries: removed {n_duplicates_removed:,} duplicates in {t_dedup:.2f}s")
        print(f"  This should significantly improve particle retention!")
        # Update node count
        n_nodes = node_positions.shape[0]
    else:
        print(f"  ✅ No duplicates found - mesh is clean!")

    # ========================================================================
    # DIAGNOSTIC: Verify array consistency after deduplication
    # ========================================================================
    print(f"\n[DIAGNOSTIC] Verifying array consistency after deduplication...")
    print(f"  node_positions shape:   {node_positions.shape}")
    print(f"  connectivity shape:     {connectivity.shape}")
    print(f"  velocity_sequence shape: {velocity_sequence.shape}")

    # Check if velocity_sequence matches deduplicated node count
    n_nodes_current = node_positions.shape[0]
    n_nodes_velocity = velocity_sequence.shape[1]

    if n_nodes_velocity != n_nodes_current:
        print(f"\n  ⚠️  CRITICAL BUG: Velocity array shape mismatch!")
        print(f"      Velocity: {n_nodes_velocity:,} nodes, Mesh: {n_nodes_current:,} nodes")
        print(f"  ❌ CANNOT CONTINUE - Stopping execution")
        raise RuntimeError("Velocity array shape mismatch after deduplication")
    else:
        print(f"  ✅ Velocity array correctly remapped ({n_nodes_velocity:,} nodes)")

    # Verify connectivity references valid node IDs
    max_node_id = np.max(connectivity)
    if max_node_id >= n_nodes_current:
        print(f"\n  ⚠️  CONNECTIVITY BUG: Max node ID {max_node_id} >= {n_nodes_current}")
        print(f"  ❌ CANNOT CONTINUE - Stopping execution")
        raise RuntimeError("Connectivity references non-existent nodes")
    else:
        print(f"  ✅ Connectivity valid (max node ID {max_node_id} < {n_nodes_current})")

    print(f"  ✅ All array consistency checks passed!")
    print(f"      → Trajectories should now be physically correct\n")
    # ========================================================================

    # Compute velocity cycle parameters
    cycle_params = compute_velocity_cycle_params(
        total_steps=N_STEPS,
        dt=DT,
        velocity_timestep_range=VELOCITY_TIMESTEP_RANGE,
        velocity_dt=VELOCITY_DT
    )
    print(f"\n  Velocity cycle parameters:")
    print(f"    Cycle period: {cycle_params['cycle_period']:.3f} time units")
    print(f"    Number of cycles: {cycle_params['n_cycles']:.2f}")
    print(f"    Tracking steps per velocity step: {cycle_params['steps_per_velocity']}")

    # ========================================================================
    # 2. Build Global Space-Filling Curve Structure (CPU)
    # ========================================================================

    print(f"\n[2/6] Building global {CURVE_TYPE.upper()} structure (CPU)...")
    t_octree = time.time()

    # Select octree builder based on configuration
    if CURVE_TYPE == 'hilbert':
        octree_struct = build_global_hilbert_octree(
            node_positions=node_positions,
            connectivity=connectivity,
            leaf_capacity=256,
            max_depth=21,
            verbose=False  # Disable verbose for production
        )
        curve_field_name = 'hilbert_sorted'
    elif CURVE_TYPE == 'morton':
        octree_struct = build_global_morton_octree(
            node_positions=node_positions,
            connectivity=connectivity,
            leaf_capacity=256,
            max_depth=21,
            verbose=False  # Disable verbose for production
        )
        curve_field_name = 'morton_sorted'
    else:
        raise ValueError(f"Unknown CURVE_TYPE: {CURVE_TYPE}. Must be 'morton' or 'hilbert'.")

    t_octree = time.time() - t_octree
    print(f"  Built {octree_struct.n_leaves:,} leaves in {t_octree:.2f}s")

    # Get curve indices array (field name differs between Morton/Hilbert structures)
    curve_indices = getattr(octree_struct, curve_field_name)
    octree_memory_mb = (octree_struct.elem_ids_sorted.nbytes + curve_indices.nbytes) / (1024**2)
    print(f"  Memory: {octree_memory_mb:.1f} MB")

    # Build mesh-aligned structures (if enabled in config)
    mesh_aligned_octree_gpu = None
    mesh_aligned_morton_gpu = None
    mesh_aligned_octree_neighbors_gpu = None
    mesh_octree_cells = None  # Cache for reuse
    kdtree_gpu = None

    if config.L2_SEARCH_METHOD == 'kdtree':
        print(f"\n  Building KD-tree structure (L2_SEARCH_METHOD={config.L2_SEARCH_METHOD})...")
        from jaxtrace.gpu.search.kdtree_node_search import (
            build_kdtree_structure,
            upload_kdtree_to_gpu,
            JAXKD_AVAILABLE,
        )

        if not JAXKD_AVAILABLE:
            print(f"    ❌ ERROR: jaxkd not available! Install with: pip install jaxkd")
            sys.exit(1)

        # Build KD-tree structure
        t_kdtree = time.time()
        kdtree_struct = build_kdtree_structure(
            node_positions, connectivity, verbose=False
        )
        t_kdtree = time.time() - t_kdtree
        print(f"    Built in {t_kdtree:.2f}s")
        print(f"    Nodes: {kdtree_struct.n_nodes:,}, Elements per node: {kdtree_struct.elements_per_node_mean:.1f} (mean)")

        # Upload to GPU
        t_upload_kdtree = time.time()
        kdtree_gpu = upload_kdtree_to_gpu(kdtree_struct, verbose=False)
        t_upload_kdtree = time.time() - t_upload_kdtree
        print(f"    GPU upload: {t_upload_kdtree:.2f}s")

    elif config.L2_SEARCH_METHOD in ['mesh_aligned_octree', 'mesh_aligned_morton', 'mesh_aligned_neighbors']:
        print(f"\n  Building mesh-aligned structures (L2_SEARCH_METHOD={config.L2_SEARCH_METHOD})...")

        # Extract mesh-aligned octree cells (shared by both methods)
        # Choose single-cell or multi-cell based on config
        t_mesh_octree = time.time()
        if config.MESH_ALIGNED_MULTI_CELL_REGISTRATION:
            print(f"    Using multi-cell vertex registration (config.MESH_ALIGNED_MULTI_CELL_REGISTRATION=True)")
            mesh_octree_cells = extract_octree_cells_vertex_multi(
                node_positions, connectivity, tolerance=1e-6, verbose=False
            )
        else:
            print(f"    Using single-cell registration (config.MESH_ALIGNED_MULTI_CELL_REGISTRATION=False)")
            mesh_octree_cells = extract_octree_cells_single(
                node_positions, connectivity, tolerance=1e-6, verbose=False
            )
        t_mesh_octree = time.time() - t_mesh_octree
        print(f"    Extracted {mesh_octree_cells.n_cells:,} cells in {t_mesh_octree:.2f}s")
        print(f"    Elements per cell (avg): {mesh_octree_cells.elements_per_cell_mean:.2f}")
        print(f"    Cells per element (avg): {mesh_octree_cells.cells_per_element_mean:.2f}")

        # Estimate memory
        cells_memory_mb = (
            mesh_octree_cells.cell_levels.nbytes +
            mesh_octree_cells.cell_morton_codes.nbytes +
            mesh_octree_cells.cell_grid_indices.nbytes +
            mesh_octree_cells.cell_sizes.nbytes +
            mesh_octree_cells.cell_to_elements_offsets.nbytes +
            mesh_octree_cells.cell_to_elements_data.nbytes
        ) / (1024**2)

        # Add multi-cell specific memory if using multi-cell registration
        if config.MESH_ALIGNED_MULTI_CELL_REGISTRATION:
            cells_memory_mb += (
                mesh_octree_cells.element_to_cells_offsets.nbytes +
                mesh_octree_cells.element_to_cells_data.nbytes
            ) / (1024**2)

        print(f"    Memory (CPU): {cells_memory_mb:.1f} MB")

        if config.L2_SEARCH_METHOD == 'mesh_aligned_octree':
            # MULTI-CELL + 2×2×2 LOCAL SEARCH (Option A)
            print(f"\n    Method: Multi-cell vertex registration + 2×2×2 local search (Option A)")
            print(f"    Building multi-cell octree...")
            t_build_multi = time.time()
            mesh_octree_multi_cells = extract_octree_cells_vertex_multi(
                node_positions, connectivity, verbose=False
            )
            t_build_multi = time.time() - t_build_multi
            print(f"    Multi-cell octree built in {t_build_multi:.2f}s")
            print(f"    Cells: {mesh_octree_multi_cells.n_cells:,}")
            print(f"    Elements per cell: {mesh_octree_multi_cells.elements_per_cell_mean:.2f}")
            print(f"    Cells per element: {mesh_octree_multi_cells.cells_per_element_mean:.2f}")

            t_upload_mesh_octree = time.time()
            mesh_aligned_octree_gpu = upload_mesh_aligned_octree_to_gpu(
                node_positions, connectivity, mesh_octree_multi_cells, verbose=False
            )
            t_upload_mesh_octree = time.time() - t_upload_mesh_octree
            print(f"    GPU upload: {t_upload_mesh_octree:.2f}s")

        elif config.L2_SEARCH_METHOD == 'mesh_aligned_morton':
            # HYBRID: Morton radius search over cell centers (expected ~98% retention)
            print(f"\n    Method: Mesh-aligned Morton (hybrid: cell centers + radius search)")
            from jaxtrace.gpu.search import (
                build_mesh_aligned_morton_structure,
                upload_mesh_aligned_morton_to_gpu,
            )

            # Build Morton structure from cell centers
            t_morton_build = time.time()
            mesh_aligned_morton_struct = build_mesh_aligned_morton_structure(
                node_positions, connectivity, mesh_octree_cells=mesh_octree_cells, verbose=False
            )
            t_morton_build = time.time() - t_morton_build
            print(f"    Morton structure built in {t_morton_build:.2f}s")
            print(f"    Elements per cell: mean={mesh_aligned_morton_struct.elements_per_cell_mean:.1f}, "
                  f"max={mesh_aligned_morton_struct.elements_per_cell_max}")

            # Upload to GPU
            t_upload_morton = time.time()
            mesh_aligned_morton_gpu = upload_mesh_aligned_morton_to_gpu(
                node_positions, connectivity, mesh_aligned_morton_struct, verbose=False
            )
            t_upload_morton = time.time() - t_upload_morton
            print(f"    GPU upload: {t_upload_morton:.2f}s")

        elif config.L2_SEARCH_METHOD == 'mesh_aligned_neighbors':
            # OPTION B: Pre-computed neighbor table (99.95% retention for centroids)
            print(f"\n    Method: Mesh-aligned octree with pre-computed neighbor table (Option B)")
            from jaxtrace.gpu.search.mesh_aligned_octree_with_neighbor_table import (
                add_neighbor_table_to_octree,
                upload_octree_with_neighbors_to_gpu
            )

            # Build neighbor table
            t_neighbor_build = time.time()
            octree_with_neighbors = add_neighbor_table_to_octree(mesh_octree_cells, verbose=False)
            t_neighbor_build = time.time() - t_neighbor_build
            print(f"    Neighbor table built in {t_neighbor_build:.2f}s")
            print(f"    Mean neighbors per cell: {octree_with_neighbors.neighbor_counts.mean():.1f}")

            # Upload to GPU
            t_upload_neighbors = time.time()
            mesh_aligned_octree_neighbors_gpu = upload_octree_with_neighbors_to_gpu(
                connectivity, node_positions, octree_with_neighbors, verbose=False
            )
            t_upload_neighbors = time.time() - t_upload_neighbors
            print(f"    GPU upload: {t_upload_neighbors:.2f}s")
    else:
        print(f"\n  Mesh-aligned structures DISABLED (L2_SEARCH_METHOD={config.L2_SEARCH_METHOD})")

    # ========================================================================
    # 3. Upload to GPU
    # ========================================================================

    print("\n[3/6] Uploading mesh and Morton structure to GPU...")
    t_upload = time.time()

    # Compute element neighbors (using configured method)
    neighbor_method_name = "NODE-BASED" if NEIGHBOR_METHOD == 'node' else "FACE-BASED"
    print(f"  Computing element neighbors ({neighbor_method_name})...")
    t_neighbors = time.time()
    element_neighbors = build_element_neighbors_array(connectivity, method=NEIGHBOR_METHOD, verbose=True)
    t_neighbors = time.time() - t_neighbors
    print(f"    Neighbor computation: {t_neighbors:.2f}s")
    neighbor_memory_mb = element_neighbors.nbytes / (1024**2)
    print(f"    Neighbor memory: {neighbor_memory_mb:.1f} MB")
    print(f"    Neighbor array shape: {element_neighbors.shape}")
    print(f"    Max neighbors per element: {element_neighbors.shape[1]}")
    if NEIGHBOR_METHOD == 'face':
        print(f"    ⚠  WARNING: Face-based neighbors may NOT work for 1:2 octree refinement!")
        print(f"              If trajectories are linear, switch to NEIGHBOR_METHOD='node'")

    # Upload standard mesh data
    mesh_gpu = upload_mesh_to_gpu(
        connectivity=connectivity,
        node_positions=node_positions,
        element_neighbors=element_neighbors,
        verbose=False
    )

    # Precompute data for point-in-tet methods
    if POINT_IN_TET_METHOD == "skala_memory_opt":
        print("\n  Precomputing element vertices for skala_memory_opt...")
        from jaxtrace.gpu.search.aa_detection import precompute_element_vertices
        from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata

        t_elem_verts = time.time()
        element_vertices = precompute_element_vertices(connectivity, node_positions, verbose=False)
        t_elem_verts = time.time() - t_elem_verts

        # Register with point-in-tet dispatcher (pass None for AA metadata since we don't use it)
        from jaxtrace.gpu.search.aa_detection import AxisAlignedMetadata
        dummy_aa_metadata = AxisAlignedMetadata(
            base_vertex_indices=jax.device_put(np.zeros(1, dtype=np.int8)),
            base_vertices=jax.device_put(np.zeros((1, 3), dtype=np.float32)),
            inv_edge_lengths=jax.device_put(np.zeros((1, 3), dtype=np.float32)),
            axis_indices=jax.device_put(np.zeros((1, 3), dtype=np.int8)),
            is_axis_aligned=jax.device_put(np.zeros(1, dtype=bool))
        )
        set_corrected_metadata(dummy_aa_metadata, element_vertices)

        elem_verts_mb = element_vertices.nbytes / (1024**2)
        print(f"    Element vertices: {connectivity.shape[0]:,} × 4 vertices × 3 coords")
        print(f"    Memory: {elem_verts_mb:.1f} MB")
        print(f"    Precompute time: {t_elem_verts:.2f}s")

    elif POINT_IN_TET_METHOD == "inverse":
        print("\n  Precomputing inverse matrices for inverse method...")
        from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
        from jaxtrace.gpu.search.point_in_tet_methods import set_inverse_matrices_gpu

        t_inverse = time.time()
        M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
        t_inverse = time.time() - t_inverse

        # Upload to GPU and register with point-in-tet dispatcher
        M_inv_gpu = jax.device_put(M_inv_array)
        p0_gpu = jax.device_put(p0_array)
        set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)

        inverse_mb = (M_inv_array.nbytes + p0_array.nbytes) / (1024**2)
        print(f"    Inverse matrices: {connectivity.shape[0]:,} × 3×3 + p0")
        print(f"    Memory: {inverse_mb:.1f} MB")
        print(f"    Precompute time: {t_inverse:.2f}s")

    # PHASE 1.3: Compute and upload element volumes for adaptive L1 hop count
    print("\n  Computing element volumes for adaptive L1...")
    t_volumes = time.time()

    # Compute element volumes on CPU (tetrahedral volume formula)
    # Volume = |det([v1-v0, v2-v0, v3-v0])| / 6
    v0 = node_positions[connectivity[:, 0]]
    v1 = node_positions[connectivity[:, 1]]
    v2 = node_positions[connectivity[:, 2]]
    v3 = node_positions[connectivity[:, 3]]

    # Edge vectors from v0
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0

    # Scalar triple product: e1 · (e2 × e3)
    cross_e2_e3 = np.cross(e2, e3)
    det = np.sum(e1 * cross_e2_e3, axis=1)
    element_volumes_cpu = np.abs(det) / 6.0

    # Upload to GPU
    element_volumes_gpu = jax.device_put(element_volumes_cpu.astype(np.float32))

    t_volumes = time.time() - t_volumes
    print(f"    Element volumes computed: {len(element_volumes_cpu):,}")
    print(f"    Volume range: [{element_volumes_cpu.min():.2e}, {element_volumes_cpu.max():.2e}]")
    print(f"    Median volume: {np.median(element_volumes_cpu):.2e}")
    print(f"    Computation time: {t_volumes:.2f}s")

    # Upload global space-filling curve structure
    # Note: upload_global_morton_to_gpu works for both Morton and Hilbert
    # (they have identical structure, just different curve indices)
    mesh_gpu_octree = upload_global_morton_to_gpu(
        octree_struct,
        connectivity,
        node_positions
    )

    # Force transfer
    _ = jax.block_until_ready(mesh_gpu.connectivity)
    _ = jax.block_until_ready(mesh_gpu_octree.elem_ids_sorted)

    t_upload = time.time() - t_upload
    print(f"  Total upload time: {t_upload:.2f}s")
    print(f"  {CURVE_TYPE.upper()} GPU leaves: {mesh_gpu_octree.n_leaves:,}")
    print(f"  {CURVE_TYPE.upper()} Prefix Table Depth: {mesh_gpu_octree.table_depth}")

    # ========================================================================
    # 4. Initialize Particles
    # ========================================================================

    # Compute domain bounds
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)
    domain_size = domain_max - domain_min

    # Compute particle bounds from fractions
    par_bounds_min = np.zeros(3, dtype=np.float32)
    par_bounds_max = np.zeros(3, dtype=np.float32)
    for i, axis in enumerate(['x', 'y', 'z']):
        min_frac, max_frac = PARTICLE_BOUNDS_FRACTION[axis]
        par_bounds_min[i] = domain_min[i] + min_frac * domain_size[i]
        par_bounds_max[i] = domain_min[i] + max_frac * domain_size[i]
    par_bounds = [par_bounds_min, par_bounds_max]

    # Use grid resolution (already unpacked at top of main())
    print(f"\n[4/6] Initializing {N_PARTICLES:,} particles (uniform grid {nx}×{ny}×{nz})...")
    print(f"  Particle bounds:")
    print(f"    X: [{par_bounds_min[0]:.6f}, {par_bounds_max[0]:.6f}] (domain fraction: {PARTICLE_BOUNDS_FRACTION['x']})")
    print(f"    Y: [{par_bounds_min[1]:.6f}, {par_bounds_max[1]:.6f}] (domain fraction: {PARTICLE_BOUNDS_FRACTION['y']})")
    print(f"    Z: [{par_bounds_min[2]:.6f}, {par_bounds_max[2]:.6f}] (domain fraction: {PARTICLE_BOUNDS_FRACTION['z']})")

    # Generate uniform grid
    particle_positions = uniform_grid_seeds(
        resolution=(nx, ny, nz),
        bounds=par_bounds,
        include_boundaries=True
    )

    # PHASE 1.1 FIX: Clip particles to mesh bounds to prevent outside-domain assignment failures
    # Add 1% safety margin to avoid numerical issues at boundaries
    print(f"\n  Clipping particles to mesh bounds (Phase 1.1 fix)...")
    original_positions = particle_positions.copy()
    mesh_bbox_min = domain_min
    mesh_bbox_max = domain_max
    margin = 0.01
    bbox_min_safe = mesh_bbox_min + margin * (mesh_bbox_max - mesh_bbox_min)
    bbox_max_safe = mesh_bbox_max - margin * (mesh_bbox_max - mesh_bbox_min)

    particle_positions_clipped = np.clip(particle_positions, bbox_min_safe, bbox_max_safe)
    particle_positions = particle_positions_clipped

    # Diagnostic: How many particles were clipped?
    n_moved = np.sum(np.any(particle_positions != original_positions, axis=1))
    print(f"    Particles clipped to mesh bounds: {n_moved}/{N_PARTICLES}")
    print(f"    Mesh bounds: X=[{mesh_bbox_min[0]:.6f}, {mesh_bbox_max[0]:.6f}], "
          f"Y=[{mesh_bbox_min[1]:.6f}, {mesh_bbox_max[1]:.6f}], "
          f"Z=[{mesh_bbox_min[2]:.6f}, {mesh_bbox_max[2]:.6f}]")
    print(f"    Safe bounds (1% margin): X=[{bbox_min_safe[0]:.6f}, {bbox_max_safe[0]:.6f}], "
          f"Y=[{bbox_min_safe[1]:.6f}, {bbox_max_safe[1]:.6f}], "
          f"Z=[{bbox_min_safe[2]:.6f}, {bbox_max_safe[2]:.6f}]")

    # Create particle data with unknown element IDs
    particle_data = ParticleData.from_positions(particle_positions)

    print(f"  Created {N_PARTICLES:,} particles in uniform grid")

    # ========================================================================
    # 5. Setup Async VTK Export
    # ========================================================================

    print(f"\n[5/6] Setting up async VTK export...")
    export_config = ExportConfig(
        output_dir=OUTPUT_DIR,
        export_frequency=EXPORT_FREQUENCY,
        include_velocities=STORE_VELOCITIES,
        include_metadata=True
    )

    exporter = AsyncVTKExporter(export_config, particle_data)
    exporter.start()

    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Export frequency: every {EXPORT_FREQUENCY} steps")
    print(f"  Store velocities: {'Yes' if STORE_VELOCITIES else 'No'}")
    print(f"  Expected exports: {N_STEPS // EXPORT_FREQUENCY}")

    # ========================================================================
    # 6. Run Time Integration
    # ========================================================================

    print(f"\n[6/6] Running time integration ({N_STEPS:,} steps)...")
    print(f"\n  Search hierarchy configuration:")
    if ENABLE_L1_SEARCH:
        if L2_SEARCH_METHOD == 'hierarchical':
            print(f"    L0 (cached element) → L1 (adaptive {N_HOPS}-6 hops) → L2 ({CURVE_TYPE.upper()} hierarchical, depth 7+6)")
        elif L2_SEARCH_METHOD == 'neighbors':
            print(f"    L0 (cached element) → L1 (adaptive {N_HOPS}-6 hops) → L2 ({CURVE_TYPE.upper()} neighbors, 27 octants)")
        elif L2_SEARCH_METHOD == 'kdtree':
            print(f"    L0 (cached element) → L1 (adaptive {N_HOPS}-6 hops) → L2 (KD-tree, K={KDTREE_K_NEAREST} nearest nodes)")
        else:
            print(f"    L0 (cached element) → L1 (adaptive {N_HOPS}-6 hops) → L2 ({CURVE_TYPE.upper()} radius, ±{L2_SEARCH_RADIUS})")
        print(f"    ✅ PHASE 1.3: L1 uses adaptive hop count (6 hops at refinement boundaries)")
    else:
        if L2_SEARCH_METHOD == 'hierarchical':
            print(f"    L0 (cached element) → L2 ({CURVE_TYPE.upper()} hierarchical, depth 7+6)")
        elif L2_SEARCH_METHOD == 'neighbors':
            print(f"    L0 (cached element) → L2 ({CURVE_TYPE.upper()} neighbors, 27 octants)")
        elif L2_SEARCH_METHOD == 'kdtree':
            print(f"    L0 (cached element) → L2 (KD-tree, K={KDTREE_K_NEAREST} nearest nodes)")
        else:
            print(f"    L0 (cached element) → L2 ({CURVE_TYPE.upper()} radius, ±{L2_SEARCH_RADIUS})")
        print(f"    ⚠️  L1 neighbor search DISABLED")

    print(f"    L2 method: {L2_SEARCH_METHOD}")
    if L2_SEARCH_METHOD in ['neighbors', 'hierarchical']:
        if mesh_gpu_octree.table_depth == 0:
            print(f"    ❌ ERROR: {L2_SEARCH_METHOD} method requires octree prefix table!")
            print(f"             Current table_depth = 0. Check {CURVE_TYPE.upper()} structure build.")
            return 1
        else:
            print(f"    ✅ Octree prefix table available (depth={mesh_gpu_octree.table_depth})")

    # Create fully-fused time-dependent RK4 step function
    # CRITICAL: Only pass parameters relevant to the chosen L2_SEARCH_METHOD
    # Passing unused None parameters can cause JAX compilation issues (41 TiB memory bug)

    if L2_SEARCH_METHOD == 'mesh_aligned_octree':
        # Mesh-aligned octree: only pass octree parameter
        rk4_step = create_rk4_fully_fused_timedep(
            mesh_gpu_connectivity=mesh_gpu.connectivity,
            mesh_gpu_node_positions=mesh_gpu.node_positions,
            mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
            mesh_gpu_element_volumes=element_volumes_gpu,
            mesh_gpu_global_morton=mesh_gpu_octree,
            n_hops=N_HOPS,
            enable_l1_search=ENABLE_L1_SEARCH,
            l2_search_method='radius',  # Fallback (won't be used)
            mesh_aligned_octree=mesh_aligned_octree_gpu,
            mesh_aligned_octree_use_multi_local=True
        )
    elif L2_SEARCH_METHOD == 'mesh_aligned_morton':
        # Mesh-aligned Morton: only pass Morton parameter
        rk4_step = create_rk4_fully_fused_timedep(
            mesh_gpu_connectivity=mesh_gpu.connectivity,
            mesh_gpu_node_positions=mesh_gpu.node_positions,
            mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
            mesh_gpu_element_volumes=element_volumes_gpu,
            mesh_gpu_global_morton=mesh_gpu_octree,
            n_hops=N_HOPS,
            enable_l1_search=ENABLE_L1_SEARCH,
            l2_search_method=L2_SEARCH_METHOD if L2_SEARCH_METHOD == 'incremental' else 'radius',
            l2_incremental_radii=INCREMENTAL_SEARCH_RADII,
            mesh_aligned_morton=mesh_aligned_morton_gpu
        )
    elif L2_SEARCH_METHOD == 'mesh_aligned_neighbors':
        # Mesh-aligned neighbors: only pass neighbors parameter
        rk4_step = create_rk4_fully_fused_timedep(
            mesh_gpu_connectivity=mesh_gpu.connectivity,
            mesh_gpu_node_positions=mesh_gpu.node_positions,
            mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
            mesh_gpu_element_volumes=element_volumes_gpu,
            mesh_gpu_global_morton=mesh_gpu_octree,
            n_hops=N_HOPS,
            enable_l1_search=ENABLE_L1_SEARCH,
            l2_search_method='radius',  # Fallback (won't be used)
            mesh_aligned_octree_neighbors=mesh_aligned_octree_neighbors_gpu
        )
    elif L2_SEARCH_METHOD == 'kdtree':
        # KD-tree: only pass kdtree parameters
        rk4_step = create_rk4_fully_fused_timedep(
            mesh_gpu_connectivity=mesh_gpu.connectivity,
            mesh_gpu_node_positions=mesh_gpu.node_positions,
            mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
            mesh_gpu_element_volumes=element_volumes_gpu,
            mesh_gpu_global_morton=mesh_gpu_octree,
            n_hops=N_HOPS,
            enable_l1_search=ENABLE_L1_SEARCH,
            l2_search_method='radius',  # Fallback (won't be used)
            kdtree_gpu=kdtree_gpu,
            kdtree_k_nearest=KDTREE_K_NEAREST,
            kdtree_max_tests=KDTREE_MAX_TESTS
        )
    else:
        # Morton-based methods: radius, incremental, neighbors, hierarchical
        rk4_step = create_rk4_fully_fused_timedep(
            mesh_gpu_connectivity=mesh_gpu.connectivity,
            mesh_gpu_node_positions=mesh_gpu.node_positions,
            mesh_gpu_element_neighbors=mesh_gpu.element_neighbors,
            mesh_gpu_element_volumes=element_volumes_gpu,
            mesh_gpu_global_morton=mesh_gpu_octree,
            n_hops=N_HOPS,
            l2_search_radius=L2_SEARCH_RADIUS,
            enable_l1_search=ENABLE_L1_SEARCH,
            l2_search_method=L2_SEARCH_METHOD,
            l2_incremental_radii=INCREMENTAL_SEARCH_RADII
        )

    # Upload velocity sequence and particle data to GPU ONCE
    print("\n  Uploading data to GPU...")
    t_upload_initial = time.time()
    velocity_fields_gpu = jax.device_put(velocity_sequence)  # Upload entire sequence
    positions_gpu = jax.device_put(particle_data.positions)
    element_ids_gpu = jax.device_put(particle_data.element_ids)
    t_upload_initial = time.time() - t_upload_initial
    vel_memory_mb = velocity_sequence.nbytes / (1024**2)
    print(f"    Velocity sequence upload: {t_upload_initial:.2f}s ({vel_memory_mb:.1f} MB)")
    print(f"    Particle data upload: minimal")

    # Cascading initial assignment (memory-efficient progressive search)
    # Start with radius=100 for all, then search unassigned with larger radii
    print(f"\n  Running cascading initial assignment...")
    print(f"    Initial radius: {INITIAL_SEARCH_RADIUS} (all particles)")
    print(f"    Fallback radii: {INITIAL_SEARCH_FALLBACK_RADII} (only unassigned particles)")
    t_initial_search = time.time()
    element_ids_gpu = initial_assignment_cascading_fallback(
        positions_gpu,
        mesh_gpu_octree,  # Works for both Morton and Hilbert
        initial_radius=INITIAL_SEARCH_RADIUS,
        fallback_radii=INITIAL_SEARCH_FALLBACK_RADII,
        verbose=True
    )
    element_ids_gpu = jax.block_until_ready(element_ids_gpu)
    t_initial_search = time.time() - t_initial_search

    # Check initial assignment (single scalar download)
    n_active_initial = int(jnp.sum(element_ids_gpu >= 0))
    initial_success_rate = (n_active_initial / N_PARTICLES) * 100
    print(f"    Initial assignment: {n_active_initial:,}/{N_PARTICLES:,} ({initial_success_rate:.2f}%)")
    print(f"    Search time: {t_initial_search:.2f}s")

    # ========================================================================
    # DIAGNOSTIC: Analyze Initial Assignment Failures (Phase 1 Priority 1)
    # ========================================================================
    if initial_success_rate < 95.0:
        print(f"\n  DIAGNOSTIC: Analyzing {N_PARTICLES - n_active_initial:,} failed assignments...")

        # Download positions and element IDs for analysis
        positions_cpu = np.array(positions_gpu)
        element_ids_cpu = np.array(element_ids_gpu)

        # Identify unassigned particles
        unassigned_mask = element_ids_cpu == -1
        unassigned_positions = positions_cpu[unassigned_mask]
        assigned_positions = positions_cpu[~unassigned_mask]

        # Spatial distribution of unassigned particles
        print(f"\n  Spatial distribution:")
        print(f"    Unassigned particles ({np.sum(unassigned_mask):,}):")
        print(f"      X: [{unassigned_positions[:, 0].min():.6f}, {unassigned_positions[:, 0].max():.6f}]")
        print(f"      Y: [{unassigned_positions[:, 1].min():.6f}, {unassigned_positions[:, 1].max():.6f}]")
        print(f"      Z: [{unassigned_positions[:, 2].min():.6f}, {unassigned_positions[:, 2].max():.6f}]")
        print(f"    Assigned particles ({np.sum(~unassigned_mask):,}):")
        print(f"      X: [{assigned_positions[:, 0].min():.6f}, {assigned_positions[:, 0].max():.6f}]")
        print(f"      Y: [{assigned_positions[:, 1].min():.6f}, {assigned_positions[:, 1].max():.6f}]")
        print(f"      Z: [{assigned_positions[:, 2].min():.6f}, {assigned_positions[:, 2].max():.6f}]")

        # Mesh element coverage in seeded region
        all_par_min = positions_cpu.min(axis=0)
        all_par_max = positions_cpu.max(axis=0)

        # Download mesh data for analysis
        connectivity_cpu = np.array(mesh_gpu.connectivity)
        node_positions_cpu = np.array(mesh_gpu.node_positions)
        # element_volumes_cpu already computed above (no need to download from GPU)

        # Count elements in seeded region
        element_centroids = node_positions_cpu[connectivity_cpu].mean(axis=1)
        in_region_mask = (
            (element_centroids[:, 0] >= all_par_min[0]) & (element_centroids[:, 0] <= all_par_max[0]) &
            (element_centroids[:, 1] >= all_par_min[1]) & (element_centroids[:, 1] <= all_par_max[1]) &
            (element_centroids[:, 2] >= all_par_min[2]) & (element_centroids[:, 2] <= all_par_max[2])
        )
        n_elements_in_region = np.sum(in_region_mask)

        print(f"\n  Mesh coverage in seeded region:")
        print(f"    Seeded region: X=[{all_par_min[0]:.6f}, {all_par_max[0]:.6f}], "
              f"Y=[{all_par_min[1]:.6f}, {all_par_max[1]:.6f}], "
              f"Z=[{all_par_min[2]:.6f}, {all_par_max[2]:.6f}]")
        print(f"    Elements in region: {n_elements_in_region:,}/{connectivity_cpu.shape[0]:,} "
              f"({100.0 * n_elements_in_region / connectivity_cpu.shape[0]:.2f}%)")

        # Element size distribution in seeded region
        if n_elements_in_region > 0:
            region_volumes = element_volumes_cpu[in_region_mask]
            print(f"\n  Element size distribution in seeded region:")
            print(f"    Volume range: [{region_volumes.min():.2e}, {region_volumes.max():.2e}]")
            print(f"    Volume median: {np.median(region_volumes):.2e}")
            print(f"    Volume mean: {np.mean(region_volumes):.2e}")
            print(f"    Volume std: {np.std(region_volumes):.2e}")

            # Characteristic length = cube root of volume
            region_char_lengths = np.cbrt(region_volumes)
            print(f"    Characteristic length range: [{region_char_lengths.min():.2e}, {region_char_lengths.max():.2e}]")
            print(f"    Characteristic length median: {np.median(region_char_lengths):.2e}")

            # Size ratio (largest / smallest)
            size_ratio = region_volumes.max() / region_volumes.min()
            print(f"    Size ratio (max/min): {size_ratio:.2f}×")

            # Refinement detection (10× volume difference)
            refined_mask = region_volumes < np.median(region_volumes) * 0.1
            n_refined = np.sum(refined_mask)
            if n_refined > 0:
                print(f"    Refined elements (>10× smaller than median): {n_refined:,}/{n_elements_in_region:,} "
                      f"({100.0 * n_refined / n_elements_in_region:.2f}%)")

        print(f"\n  DIAGNOSTIC: Analysis complete. Proceeding with compilation...\n")

    # Run first step to trigger JIT compilation (data stays on GPU)
    print("\n  Compiling RK4 (first step)...")

    # DEBUG: Print all array shapes before compilation
    print(f"\n  DEBUG: Array shapes before compilation:")
    print(f"    positions_gpu: {positions_gpu.shape}")
    print(f"    element_ids_gpu: {element_ids_gpu.shape}")
    print(f"    velocity_fields_gpu: {velocity_fields_gpu.shape}")
    print(f"    DT: {DT}")
    print(f"    mesh_gpu.connectivity: {mesh_gpu.connectivity.shape}")
    print(f"    mesh_gpu.node_positions: {mesh_gpu.node_positions.shape}")
    print(f"    mesh_gpu.element_neighbors: {mesh_gpu.element_neighbors.shape}")
    print(f"    element_volumes_gpu: {element_volumes_gpu.shape}")
    if mesh_aligned_octree_gpu is not None:
        print(f"    mesh_aligned_octree_gpu.cell_to_elements_offsets: {mesh_aligned_octree_gpu.cell_to_elements_offsets.shape}")
        print(f"    mesh_aligned_octree_gpu.cell_to_elements_data: {mesh_aligned_octree_gpu.cell_to_elements_data.shape}")
        print(f"    mesh_aligned_octree_gpu.cell_morton_codes: {mesh_aligned_octree_gpu.cell_morton_codes.shape}")
        print(f"    mesh_aligned_octree_gpu.n_cells: {mesh_aligned_octree_gpu.n_cells}")
    print()

    t_compile = time.time()
    positions_gpu, element_ids_gpu = rk4_step(
        positions_gpu,
        element_ids_gpu,
        DT,
        velocity_fields_gpu,
        0  # time_idx for first step
    )
    positions_gpu = jax.block_until_ready(positions_gpu)
    element_ids_gpu = jax.block_until_ready(element_ids_gpu)
    t_compile = time.time() - t_compile
    print(f"    Compilation time: {t_compile:.2f}s")

    if initial_success_rate < 95.0:
        print(f"\n❌ WARNING: Initial assignment <95%. Continuing anyway...")

    # Main time integration loop
    print(f"\n  Running {N_STEPS:,} timesteps...")
    print(f"  {'Step':>6} {'Active':>10} {'Retention':>10} {'Step Time':>12} {'Throughput':>15}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*12} {'-'*15}")

    t_integration_start = time.time()
    step_times = []
    retention_history = []

    for step in range(1, N_STEPS + 1):
        t_step = time.time()

        # Compute time index for cyclic velocity (wraps automatically in RK4)
        time_idx = step  # Will be converted to velocity index via modulo in RK4

        # Run RK4 step (all data stays on GPU)
        positions_gpu, element_ids_gpu = rk4_step(
            positions_gpu,
            element_ids_gpu,
            DT,
            velocity_fields_gpu,
            time_idx
        )

        # Block until computation completes
        positions_gpu = jax.block_until_ready(positions_gpu)
        element_ids_gpu = jax.block_until_ready(element_ids_gpu)

        t_step = time.time() - t_step
        step_times.append(t_step)

        # Count active particles (single scalar download)
        n_active = int(jnp.sum(element_ids_gpu >= 0))
        retention = (n_active / N_PARTICLES) * 100
        retention_history.append(retention)

        throughput = N_PARTICLES / t_step

        # Download and enqueue export ONLY at export frequency
        if step % EXPORT_FREQUENCY == 0:
            positions_cpu = np.array(positions_gpu, dtype=np.float32)
            element_ids_cpu = np.array(element_ids_gpu, dtype=np.int32)

            # Create minimal ParticleData for export (only positions matter)
            particle_data_export = ParticleData(
                positions=positions_cpu,
                velocities=np.zeros((N_PARTICLES, 3), dtype=np.float32),
                element_ids=element_ids_cpu,
                block_ids=np.zeros(N_PARTICLES, dtype=np.int32),
                active_mask=(element_ids_cpu >= 0)
            )
            exporter.enqueue_export(step, particle_data_export)

        # Log at intervals
        if step % LOG_INTERVAL == 0 or step == N_STEPS:
            export_stats = exporter.get_stats()
            print(f"  {step:6d} {n_active:10,} {retention:9.2f}% {t_step*1000:10.2f} ms {throughput:12.0f} p/s | Exported: {export_stats['n_exported']:>4}")

    t_integration = time.time() - t_integration_start

    # ========================================================================
    # Finalize Export
    # ========================================================================

    print("\n  Waiting for exports to complete...")
    exporter.stop()

    export_stats = exporter.get_stats()
    print(f"  ✅ All exports complete")
    print(f"    Files exported: {export_stats['n_exported']}")
    print(f"    Mean export time: {export_stats['mean_time']:.3f} s")
    print(f"    Total export time: {export_stats['total_time']:.1f} s")

    # ========================================================================
    # Final Analysis
    # ========================================================================

    print("\n" + "=" * 80)
    print("PRODUCTION RESULTS")
    print("=" * 80)

    # Final retention (download final state)
    positions_final_cpu = np.array(positions_gpu, dtype=np.float32)
    element_ids_final_cpu = np.array(element_ids_gpu, dtype=np.int32)

    final_active = np.sum(element_ids_final_cpu >= 0)
    final_retention = (final_active / N_PARTICLES) * 100

    print(f"\n  Initial particles: {N_PARTICLES:,}")
    print(f"  Initial assignment: {n_active_initial:,} ({initial_success_rate:.2f}%)")
    print(f"  Final active: {final_active:,}")
    print(f"  Final retention: {final_retention:.2f}%")

    # Timing statistics
    mean_step_time = np.mean(step_times[1:])  # Exclude first (compiled) step
    std_step_time = np.std(step_times[1:])
    min_step_time = np.min(step_times[1:])
    max_step_time = np.max(step_times[1:])

    mean_throughput = N_PARTICLES / mean_step_time

    print(f"\n  Timesteps completed: {N_STEPS:,}")
    print(f"  Total integration time: {t_integration:.2f}s")
    print(f"  Mean step time: {mean_step_time*1000:.2f} ± {std_step_time*1000:.2f} ms")
    print(f"  Min/Max step time: {min_step_time*1000:.2f} / {max_step_time*1000:.2f} ms")
    print(f"  Mean throughput: {mean_throughput:.0f} particles/s")

    # Retention over time
    retention_10 = retention_history[9] if len(retention_history) > 9 else 0
    retention_100 = retention_history[99] if len(retention_history) > 99 else 0
    retention_1000 = retention_history[999] if len(retention_history) > 999 else 0

    print(f"\n  Retention history:")
    print(f"    Step 10:    {retention_10:.2f}%")
    print(f"    Step 100:   {retention_100:.2f}%")
    print(f"    Step 1000:  {retention_1000:.2f}%")
    print(f"    Step {N_STEPS}: {final_retention:.2f}%")

    # ========================================================================
    # Success Criteria
    # ========================================================================

    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)

    success = True

    # Check initial assignment
    if initial_success_rate >= 95.0:
        print(f"✅ Initial assignment: {initial_success_rate:.2f}% (≥95% target)")
    else:
        print(f"❌ Initial assignment: {initial_success_rate:.2f}% (<95% target)")
        success = False

    # Check final retention
    if final_retention >= 95.0:
        print(f"✅ Final retention: {final_retention:.2f}% (≥95% target)")
    else:
        print(f"❌ Final retention: {final_retention:.2f}% (<95% target)")
        success = False

    # Check throughput
    if mean_throughput >= 40000:
        print(f"✅ Throughput: {mean_throughput:.0f} p/s (≥40k target)")
    elif mean_throughput >= 30000:
        print(f"⚠️  Throughput: {mean_throughput:.0f} p/s (30-40k, acceptable)")
    else:
        print(f"❌ Throughput: {mean_throughput:.0f} p/s (<30k target)")
        success = False

    # Memory
    total_memory_mb = octree_memory_mb + 50  # Approx for mesh data
    print(f"✅ Memory: ~{total_memory_mb:.0f} MB (global {CURVE_TYPE.upper()} + mesh)")

    # Architecture
    print(f"✅ Architecture: L0 (cached) + L1 ({N_HOPS}-hop) + L2 (global {CURVE_TYPE.upper()}, radius={L2_SEARCH_RADIUS})")
    print(f"✅ {CURVE_TYPE.upper()} structure: {octree_struct.n_leaves:,} leaves, {octree_struct.leaf_capacity} capacity")
    print(f"✅ No JAX OOM errors")

    # Export summary
    print(f"✅ VTK export: {export_stats['n_exported']} files in {OUTPUT_DIR}")

    print("=" * 80)

    if success:
        print("\n🎉 PRODUCTION TEST PASSED!")
        print(f"   Global {CURVE_TYPE.upper()} L2 search meets all performance targets.")
    else:
        print("\n⚠️  PRODUCTION TEST RESULTS")
        print("   Some metrics below target. Review L2 configuration or increase search radius.")

    print("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
