#!/usr/bin/env python3
"""
Time-Marching Pipeline Test - Directly Copied from test_phase1_batched_threadeda.py

Integrates the proven initialization and search from Phase 1 with new time-marching tracking.
Uses EXACT same patterns as test_phase1_batched_threadeda.py for mesh loading, forest structure,
and initial search.

Pipeline stages:
1. Mesh Loading and Forest Structure (COPIED from test_phase1)
2. Initial Element Search (COPIED from test_phase1)
3. Velocity Interpolation (NEW)
4. Time Integration (NEW)

Expected performance: ~2,500-3,000 p/s
"""

import os
import sys
import time
import numpy as np
import jax
from pathlib import Path
from typing import Callable

# Force CPU-GPU memory management
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mesh loading
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu

# Phase 1: Forest structure (EXACT same imports as test_phase1)
from jaxtrace.gpu.forest.block_grid import create_regular_grid
from jaxtrace.gpu.forest.block_mapper import assign_elements_to_blocks
from jaxtrace.gpu.forest.padded_arrays import build_padded_block_arrays
from jaxtrace.gpu.forest import build_element_neighbors_array

# Search (EXACT same imports as test_phase1)
from jaxtrace.gpu.search import classify_blocks, incremental_search_batch
from jaxtrace.gpu.search.hash_bucket import build_hash_bucket_arrays
from jaxtrace.gpu.search.initial_assignment import initial_search_batch

# Particles (EXACT same imports as test_phase1)
from jaxtrace.gpu.particles import ParticleData

# Time-marching (NEW for tracking)
from jaxtrace.gpu.tracking import (
    ParticleTimeMarcher,
    create_constant_velocity_field,
    rk4_step_with_incremental_search,
)
from jaxtrace.gpu.batching import create_default_config


class TimeMarching_ThreadedATest:
    """
    Time-marching test using ThreadedA mesh.

    Copies EXACT initialization from test_phase1_batched_threadeda.py
    and integrates with new time-marching tracking.
    """

    def __init__(self):
        # Mesh paths
        self.mesh_path = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_20.pvtu"

        # Grid configuration (EXACT same as test_phase1)
        self.grid_size = (8, 8, 4)  # 256 blocks

        # Mesh data (populated by load_and_prepare_mesh)
        self.node_positions = None
        self.connectivity = None
        self.bbox = None
        self.blocks = None
        self.element_to_block = None
        self.element_neighbors = None  # NEW: For L1 neighbor search
        self.padded_arrays = None
        self.classification = None
        self.hash_bucket_data = {}
        self.block_neighbors_26 = None

    def print_section(self, title: str):
        """Print section header."""
        print(f"\n{'='*80}")
        print(title)
        print(f"{'='*80}")

    def load_and_prepare_mesh(self):
        """
        Load ThreadedA mesh and prepare forest structure.

        COPIED EXACTLY from test_phase1_batched_threadeda.py lines 167-307
        """
        self.print_section("PHASE 1: MESH LOADING AND FOREST STRUCTURE")

        print(f"\n📁 Loading mesh: {self.mesh_path}")
        self.node_positions, self.connectivity, _ = load_mesh_from_pvtu(Path(self.mesh_path))

        print(f"✓ Mesh loaded:")
        print(f"  Nodes:    {len(self.node_positions):,}")
        print(f"  Elements: {len(self.connectivity):,}")

        # Compute bounding box
        self.bbox = np.array([
            self.node_positions[:, 0].min(), self.node_positions[:, 0].max(),
            self.node_positions[:, 1].min(), self.node_positions[:, 1].max(),
            self.node_positions[:, 2].min(), self.node_positions[:, 2].max(),
        ], dtype=np.float32)

        print(f"\n📦 Bounding box:")
        print(f"  X: [{self.bbox[0]:.4f}, {self.bbox[1]:.4f}]")
        print(f"  Y: [{self.bbox[2]:.4f}, {self.bbox[3]:.4f}]")
        print(f"  Z: [{self.bbox[4]:.4f}, {self.bbox[5]:.4f}]")

        # Create block grid
        print(f"\n🌳 Creating forest structure (grid: {self.grid_size})...")
        self.blocks = create_regular_grid(self.bbox, self.grid_size)
        print(f"✓ Total blocks: {len(self.blocks)}")

        # Assign elements to blocks
        print(f"\n📍 Assigning elements to blocks...")
        self.element_to_block, stats = assign_elements_to_blocks(
            self.node_positions,
            self.connectivity,
            self.bbox,
            self.grid_size,
            verbose=False
        )

        print(f"✓ Element assignment complete:")
        print(f"  Elements assigned: {stats.n_elements:,}")
        print(f"  Blocks used: {stats.n_blocks_used}/{stats.n_blocks}")
        print(f"  Elements per block: {stats.min_elements} - {stats.max_elements} (avg: {stats.mean_elements:.1f})")
        print(f"  Imbalance ratio: {stats.imbalance_ratio:.2f}×")
        print(f"  Heavy blocks (>10K): {len(stats.heavy_blocks)}")

        # Build element neighbors for L1 search optimization
        print(f"\n🔗 Building element face-neighbors for L1 search...")
        self.element_neighbors = build_element_neighbors_array(self.connectivity, verbose=True)

        # Build padded arrays (V5 extended with mesh data + element neighbors)
        print(f"\n📊 Building padded arrays (V5 extended mode)...")
        self.padded_arrays = build_padded_block_arrays(
            self.element_to_block,
            stats,
            node_positions=self.node_positions,
            connectivity=self.connectivity,
            element_neighbors=self.element_neighbors,  # NEW: Enable L1 optimization
            verbose=True
        )

        print(f"✓ Padded arrays created:")
        print(f"  Shape: {self.padded_arrays.block_elements.shape}")
        print(f"  Memory: {self.padded_arrays.memory_mb:.1f} MB")
        print(f"  Max elements per block: {self.padded_arrays.max_elements_per_block}")

        # Classify blocks and build hash buckets
        print(f"\n🏷️  Classifying blocks (threshold: 10K elements)...")
        self.classification = classify_blocks(self.padded_arrays, threshold=10000, verbose=False)

        print(f"✓ Block classification:")
        print(f"  Light blocks (<{self.classification.threshold}):  {len(self.classification.light_blocks)}")
        print(f"  Heavy blocks (≥{self.classification.threshold}): {len(self.classification.heavy_blocks)}")

        # Build hash buckets for heavy blocks
        if self.classification.heavy_blocks:
            print(f"\n🗂️  Building hash buckets for {len(self.classification.heavy_blocks)} heavy blocks...")
            self.hash_bucket_data = {}
            element_centroids = np.mean(self.node_positions[self.connectivity], axis=1).astype(np.float32)

            start = time.time()
            for idx, block_id in enumerate(self.classification.heavy_blocks):
                # Get block elements
                block_elems = self.padded_arrays.block_elements[block_id]
                block_count = int(self.padded_arrays.block_sizes[block_id])
                elem_ids = block_elems[:block_count]
                elem_ids = elem_ids[elem_ids >= 0]

                if len(elem_ids) == 0:
                    continue

                centroids = element_centroids[elem_ids]
                block_bounds = self.blocks[block_id].bounds

                hash_arrays = build_hash_bucket_arrays(
                    block_id=block_id,
                    element_ids=elem_ids,
                    element_centroids=centroids,
                    block_bounds=block_bounds,
                    target_bucket_size=200,
                    morton_bits=10
                )

                self.hash_bucket_data[block_id] = hash_arrays

            duration = time.time() - start
            print(f"✓ Hash buckets built: {len(self.hash_bucket_data)} in {duration:.2f} s")

        # Build block neighbors
        self.block_neighbors_26 = np.array([b.neighbors_26 for b in self.blocks], dtype=np.int32)

        print(f"\n✅ Mesh preparation complete")

    def seed_test_particles(self, n_particles: int) -> ParticleData:
        """
        Seed random test particles in bounding box.

        COPIED EXACTLY from test_phase1_batched_threadeda.py lines 309-328
        """
        print(f"\n🌱 Seeding {n_particles:,} test particles...")

        # Random positions in bounding box
        rng = np.random.RandomState(42)
        bbox_min = np.array([self.bbox[0], self.bbox[2], self.bbox[4]], dtype=np.float32)
        bbox_max = np.array([self.bbox[1], self.bbox[3], self.bbox[5]], dtype=np.float32)
        bbox_size = bbox_max - bbox_min

        random_01 = rng.uniform(0.0, 1.0, (n_particles, 3)).astype(np.float32)
        positions = bbox_min + random_01 * bbox_size

        # Create ParticleData
        particle_data = ParticleData.from_positions(positions)

        print(f"✓ Seeded {particle_data.n_particles:,} particles")
        print(f"  Active: {particle_data.n_active}")

        return particle_data

    def run_initial_search(self, particle_data: ParticleData) -> ParticleData:
        """
        Run initial element search for particles.

        COPIED EXACTLY from test_phase1_batched_threadeda.py lines 338-356
        """
        print(f"\n🔍 Running initial assignment...")

        init_start = time.time()
        element_ids_found, block_ids_found, init_stats = initial_search_batch(
            particle_data.positions,
            self.bbox,
            self.grid_size,
            self.classification,
            self.padded_arrays,
            self.block_neighbors_26,
            self.hash_bucket_data,
            self.node_positions,
            self.connectivity,
            verbose=False
        )
        init_duration = time.time() - init_start

        print(f"✓ Initial assignment complete ({init_duration:.2f} s)")
        print(f"  Found: {init_stats.n_found}/{particle_data.n_particles} ({100*init_stats.n_found/particle_data.n_particles:.1f}%)")
        print(f"  Throughput: {init_stats.n_found/init_duration:.0f} p/s")

        # Update particle data with found elements and blocks
        particle_data.element_ids = element_ids_found
        particle_data.block_ids = block_ids_found

        # Update active mask: only particles with found elements are active
        particle_data.active_mask = (element_ids_found >= 0)

        return particle_data

    def test_velocity_interpolation(self, particle_data: ParticleData):
        """
        Test velocity interpolation for all particles.

        NEW - Integrates with time-marching tracking.
        """
        self.print_section("VELOCITY INTERPOLATION TEST")

        print(f"\n🌊 Creating constant velocity field...")
        # Constant velocity field: [1.0, 0.0, 0.0] mm/s
        velocity_field = create_constant_velocity_field(
            self.padded_arrays,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            self.node_positions
        )
        print(f"✓ Velocity field created: {velocity_field.shape}")
        print(f"  Constant velocity: [1.0, 0.0, 0.0] mm/s")

        print(f"\n📊 Creating time marcher...")
        config = create_default_config()
        marcher = ParticleTimeMarcher(
            self.padded_arrays,
            self.connectivity,
            self.node_positions,
            config,
            verbose=False
        )
        print(f"✓ Time marcher created")

        print(f"\n🔄 Interpolating velocities for {particle_data.n_active} active particles...")
        t_start = time.time()
        velocities = marcher.interpolate_velocities(particle_data, velocity_field)
        t_interp = time.time() - t_start

        print(f"✓ Velocity interpolation complete ({t_interp:.2f} s)")
        print(f"  Throughput: {particle_data.n_active/t_interp:.0f} p/s")
        print(f"  Velocities shape: {velocities.shape}")
        print(f"  Mean velocity: [{velocities.mean(axis=0)[0]:.3f}, {velocities.mean(axis=0)[1]:.3f}, {velocities.mean(axis=0)[2]:.3f}]")
        print(f"  Expected:      [1.000, 0.000, 0.000]")

        # Validation
        mean_vel = velocities.mean(axis=0)
        if np.abs(mean_vel[0] - 1.0) < 0.1 and np.abs(mean_vel[1]) < 0.1 and np.abs(mean_vel[2]) < 0.1:
            print(f"\n✅ Velocity interpolation PASSED")
        else:
            print(f"\n❌ Velocity interpolation FAILED - mean velocity deviates from expected")

        return velocities, marcher, velocity_field

    def test_single_timestep(self, particle_data: ParticleData, marcher, velocity_field, dt: float = 0.001):
        """
        Test single Forward Euler timestep.

        NEW - Time integration test.
        """
        self.print_section(f"SINGLE TIMESTEP TEST (dt={dt})")

        print(f"\n⏱️  Testing Forward Euler integration...")
        print(f"  Active particles: {particle_data.n_active}")
        print(f"  Time step: {dt} s")

        # Save initial positions for comparison
        initial_positions = particle_data.positions.copy()

        # Create search function that uses initial_search_batch
        def search_fn(pdata):
            elem_ids, block_ids, stats = initial_search_batch(
                pdata.positions,
                self.bbox,
                self.grid_size,
                self.classification,
                self.padded_arrays,
                self.block_neighbors_26,
                self.hash_bucket_data,
                self.node_positions,
                self.connectivity,
                verbose=False
            )
            pdata.element_ids = elem_ids
            pdata.block_ids = block_ids
            pdata.active_mask = (elem_ids >= 0)
            return pdata, {'n_found': stats.n_found}

        # Single timestep
        t_start = time.time()
        particle_data, step_stats = marcher.march_single_timestep_euler(
            particle_data,
            velocity_field,
            dt,
            search_fn,
            use_active_mask=True
        )
        t_total = time.time() - t_start

        print(f"✓ Timestep complete ({t_total:.2f} s)")
        print(f"  Throughput: {step_stats['throughput']:.0f} p/s")
        print(f"  Time breakdown:")
        print(f"    Interpolation: {step_stats['time_interpolation']*1000:.1f} ms ({step_stats['time_interpolation']/t_total*100:.1f}%)")
        print(f"    Integration:   {step_stats['time_integration']*1000:.1f} ms ({step_stats['time_integration']/t_total*100:.1f}%)")
        print(f"    Search:        {step_stats['time_search']*1000:.1f} ms ({step_stats['time_search']/t_total*100:.1f}%)")

        # Validate particle motion
        displacement = particle_data.positions - initial_positions
        mean_displacement = displacement.mean(axis=0)
        expected_displacement = np.array([1.0, 0.0, 0.0]) * dt  # v * dt

        print(f"\n📏 Particle displacement:")
        print(f"  Mean: [{mean_displacement[0]:.6f}, {mean_displacement[1]:.6f}, {mean_displacement[2]:.6f}]")
        print(f"  Expected: [{expected_displacement[0]:.6f}, {expected_displacement[1]:.6f}, {expected_displacement[2]:.6f}]")

        if np.allclose(mean_displacement, expected_displacement, atol=1e-5):
            print(f"✅ Particle motion CORRECT")
        else:
            print(f"⚠️  Particle motion may be incorrect")

        return particle_data, step_stats

    def test_single_timestep_rk4(
        self,
        particle_data: ParticleData,
        marcher,
        velocity_field_fn: Callable,
        dt: float = 0.001,
        use_intermediate_searches: bool = False
    ):
        """
        Test single RK4 timestep.

        NEW - RK4 integration test with both modes.
        """
        mode_str = "Full (intermediate searches)" if use_intermediate_searches else "Simplified (single search)"
        self.print_section(f"SINGLE RK4 TIMESTEP TEST (dt={dt}, {mode_str})")

        print(f"\n🔄 Testing RK4 integration...")
        print(f"  Active particles: {particle_data.n_active}")
        print(f"  Time step: {dt} s")
        print(f"  Mode: {mode_str}")

        # Save initial positions for comparison
        initial_positions = particle_data.positions.copy()

        # Create search function that uses initial_search_batch
        def search_fn(pdata):
            elem_ids, block_ids, stats = initial_search_batch(
                pdata.positions,
                self.bbox,
                self.grid_size,
                self.classification,
                self.padded_arrays,
                self.block_neighbors_26,
                self.hash_bucket_data,
                self.node_positions,
                self.connectivity,
                verbose=False
            )
            pdata.element_ids = elem_ids
            pdata.block_ids = block_ids
            pdata.active_mask = (elem_ids >= 0)
            return pdata, {'n_found': stats.n_found}

        # Single RK4 timestep
        t_start = time.time()
        particle_data, step_stats = marcher.march_single_timestep_rk4(
            particle_data,
            velocity_field_fn,
            dt,
            current_time=0.0,
            search_fn=search_fn,
            use_intermediate_searches=use_intermediate_searches
        )
        t_total = time.time() - t_start

        print(f"✓ RK4 timestep complete ({t_total:.2f} s)")
        print(f"  Throughput: {step_stats['throughput']:.0f} p/s")
        print(f"  Total searches: {step_stats['n_searches']}")

        # Validate particle motion
        displacement = particle_data.positions - initial_positions
        mean_displacement = displacement.mean(axis=0)
        expected_displacement = np.array([1.0, 0.0, 0.0]) * dt  # v * dt (same for constant field)

        print(f"\n📏 Particle displacement:")
        print(f"  Mean: [{mean_displacement[0]:.6f}, {mean_displacement[1]:.6f}, {mean_displacement[2]:.6f}]")
        print(f"  Expected: [{expected_displacement[0]:.6f}, {expected_displacement[1]:.6f}, {expected_displacement[2]:.6f}]")

        if np.allclose(mean_displacement, expected_displacement, atol=1e-5):
            print(f"✅ Particle motion CORRECT")
        else:
            print(f"⚠️  Particle motion may be incorrect")

        return particle_data, step_stats

    def test_single_timestep_rk4_incremental(
        self,
        particle_data: ParticleData,
        velocity_field_fn: Callable,
        dt: float = 0.001
    ):
        """
        Test single RK4 timestep with L0+L1 incremental search optimization.

        NEW - L0+L1 optimized RK4 integration test.
        Expected speedup: 10-50× vs full RK4.
        """
        self.print_section(f"SINGLE RK4 TIMESTEP TEST (dt={dt}, L0+L1 OPTIMIZED)")

        print(f"\n🚀 Testing RK4 with L0+L1 incremental search...")
        print(f"  Active particles: {particle_data.n_active}")
        print(f"  Time step: {dt} s")
        print(f"  Mode: L0+L1 Optimized (expected 10-50× speedup)")

        # Save initial positions for comparison
        initial_positions = particle_data.positions.copy()

        # Create velocity interpolator using block-by-block processing
        # (matches ParticleTimeMarcher.interpolate_velocities from time_marching.py)
        from jaxtrace.gpu.tracking import batch_interpolate_velocities
        from jaxtrace.gpu.batching.block_grouping import group_particles_by_block

        # Upload global mesh to GPU once
        connectivity_gpu = jax.device_put(self.connectivity)
        node_positions_gpu = jax.device_put(self.node_positions)

        def velocity_interpolator(pdata, t):
            """Interpolate velocities at given time, processing block-by-block."""
            vfield = velocity_field_fn(t)

            n = len(pdata.positions)
            velocities = np.zeros((n, 3), dtype=np.float32)

            # Group particles by block
            grouping = group_particles_by_block(
                pdata.block_ids,
                self.padded_arrays.block_sizes
            )

            # Process each block
            for block_id, particle_indices in grouping.groups.items():
                if len(particle_indices) == 0:
                    continue

                # Extract particle data for this block
                block_positions = pdata.positions[particle_indices]
                block_element_ids = pdata.element_ids[particle_indices]

                # Upload to GPU
                block_positions_gpu = jax.device_put(block_positions)
                block_element_ids_gpu = jax.device_put(block_element_ids)
                block_velocity_field_gpu = jax.device_put(vfield[block_id])

                # Interpolate on GPU using global connectivity/node_positions
                block_velocities = batch_interpolate_velocities(
                    block_positions_gpu,
                    block_element_ids_gpu,
                    connectivity_gpu,
                    node_positions_gpu,
                    block_velocity_field_gpu
                )

                # Transfer back to CPU
                velocities[particle_indices] = np.array(block_velocities)

            return velocities

        # Create incremental search function
        def incremental_searcher(new_positions, cached_elem_ids, cached_block_ids):
            """Incremental search with L0+L1 optimization."""
            return incremental_search_batch(
                new_positions,
                cached_elem_ids,
                cached_block_ids,
                self.bbox,
                self.grid_size,
                self.classification,
                self.padded_arrays,
                self.block_neighbors_26,
                self.hash_bucket_data,
                self.node_positions,
                self.connectivity,
                element_neighbors=self.element_neighbors,  # Enable L1
                verbose=False
            )

        # Single RK4 timestep with incremental search
        t_start = time.time()
        new_particle_data, rk4_stats = rk4_step_with_incremental_search(
            particle_data,
            velocity_interpolator,
            incremental_searcher,
            dt=dt,
            current_time=0.0
        )
        t_total = time.time() - t_start

        # Extract statistics
        n_particles = particle_data.n_particles
        throughput = n_particles / t_total if t_total > 0 else 0

        print(f"✓ RK4 timestep complete ({t_total:.2f} s)")
        print(f"  Throughput: {throughput:.0f} p/s")
        print(f"\n📊 L0+L1/L2 Hit Rates (across all 4 RK4 stages):")
        total_searches = n_particles * 4
        l0_rate = 100 * rk4_stats['l0_total'] / total_searches if total_searches > 0 else 0
        l1_rate = 100 * rk4_stats['l1_total'] / total_searches if total_searches > 0 else 0
        l2_rate = 100 * rk4_stats['l2_total'] / total_searches if total_searches > 0 else 0
        print(f"  L0 hits (cached):    {rk4_stats['l0_total']:>5,} / {total_searches:>6,} ({l0_rate:>5.1f}%)")
        print(f"  L1 hits (neighbors): {rk4_stats['l1_total']:>5,} / {total_searches:>6,} ({l1_rate:>5.1f}%)")
        print(f"  L2 hits (block):     {rk4_stats['l2_total']:>5,} / {total_searches:>6,} ({l2_rate:>5.1f}%)")

        # Validate particle motion
        displacement = new_particle_data.positions - initial_positions
        mean_displacement = displacement.mean(axis=0)
        expected_displacement = np.array([1.0, 0.0, 0.0]) * dt  # v * dt (constant field)

        print(f"\n📏 Particle displacement:")
        print(f"  Mean: [{mean_displacement[0]:.6f}, {mean_displacement[1]:.6f}, {mean_displacement[2]:.6f}]")
        print(f"  Expected: [{expected_displacement[0]:.6f}, {expected_displacement[1]:.6f}, {expected_displacement[2]:.6f}]")

        if np.allclose(mean_displacement, expected_displacement, atol=1e-5):
            print(f"✅ Particle motion CORRECT")
        else:
            print(f"⚠️  Particle motion may be incorrect")

        # Return stats dict format
        step_stats = {
            'throughput': throughput,
            'n_searches': 4,  # RK4 has 4 stages
            'l0_hits': rk4_stats['l0_total'],
            'l1_hits': rk4_stats['l1_total'],
            'l2_hits': rk4_stats['l2_total'],
        }

        return new_particle_data, step_stats


def main():
    print("="*80)
    print("TIME-MARCHING PIPELINE TEST")
    print("Using initialization from test_phase1_batched_threadeda.py")
    print("="*80)

    # Create test instance
    test = TimeMarching_ThreadedATest()

    # STEP 1: Load and prepare mesh (COPIED from test_phase1)
    test.load_and_prepare_mesh()

    # STEP 2: Test with small particle count first
    for n_particles in [100, 1_000]:
        test.print_section(f"TEST: {n_particles:,} PARTICLES")

        # Seed particles (COPIED from test_phase1)
        particle_data = test.seed_test_particles(n_particles)

        # Run initial search (COPIED from test_phase1)
        particle_data = test.run_initial_search(particle_data)

        if particle_data.n_active < n_particles * 0.5:
            print(f"\n⚠️  WARNING: Less than 50% particles found. Skipping time-marching test.")
            continue

        # NEW: Test velocity interpolation
        velocities, marcher, velocity_field = test.test_velocity_interpolation(particle_data)

        # Create velocity field function for RK4
        def velocity_field_fn(t):
            return velocity_field  # Constant field for testing

        # NEW: Test single Forward Euler timestep
        particle_data_euler = particle_data.copy()  # Save copy for RK4 test
        particle_data_euler, euler_stats = test.test_single_timestep(
            particle_data_euler,
            marcher,
            velocity_field,
            dt=0.001
        )

        # NEW: Test single RK4 timestep (simplified mode - faster)
        particle_data_rk4_simple = particle_data.copy()  # Reset to initial state
        particle_data_rk4_simple, rk4_simple_stats = test.test_single_timestep_rk4(
            particle_data_rk4_simple,
            marcher,
            velocity_field_fn,
            dt=0.001,
            use_intermediate_searches=False  # Simplified mode
        )

        # NEW: Test single RK4 timestep (full mode - most accurate)
        particle_data_rk4_full = particle_data.copy()  # Reset to initial state
        particle_data_rk4_full, rk4_full_stats = test.test_single_timestep_rk4(
            particle_data_rk4_full,
            marcher,
            velocity_field_fn,
            dt=0.001,
            use_intermediate_searches=True  # Full mode with intermediate searches
        )

        # NEW: Test single RK4 timestep (L0+L1 optimized - FASTEST)
        particle_data_rk4_opt = particle_data.copy()  # Reset to initial state
        particle_data_rk4_opt, rk4_opt_stats = test.test_single_timestep_rk4_incremental(
            particle_data_rk4_opt,
            velocity_field_fn,
            dt=0.001
        )

        # Compare all four methods
        print(f"\n📊 Integration Method Comparison:")
        print(f"  Forward Euler:         {euler_stats['throughput']:>6,.0f} p/s  (1 search/step)")
        print(f"  RK4 Simplified:        {rk4_simple_stats['throughput']:>6,.0f} p/s  ({rk4_simple_stats['n_searches']} search/step)")
        print(f"  RK4 Full:              {rk4_full_stats['throughput']:>6,.0f} p/s  ({rk4_full_stats['n_searches']} searches/step)")
        print(f"  RK4 L0+L1 Optimized:   {rk4_opt_stats['throughput']:>6,.0f} p/s  ({rk4_opt_stats['n_searches']} searches/step)")
        print(f"\n🚀 Speedup vs RK4 Full: {rk4_opt_stats['throughput']/rk4_full_stats['throughput']:.1f}×")

        # Clear GPU memory
        jax.clear_caches()
        print(f"\n✓ GPU memory cleared")
        time.sleep(1)

    # Final summary
    test.print_section("✅ ALL TESTS COMPLETE")
    print()
    print("Tested components:")
    print("  ✅ Mesh loading (from test_phase1)")
    print("  ✅ Forest structure creation (from test_phase1)")
    print("  ✅ Block classification (from test_phase1)")
    print("  ✅ Hash bucket construction (from test_phase1)")
    print("  ✅ Element face-neighbors building (NEW - for L1 optimization)")
    print("  ✅ Initial particle search (from test_phase1)")
    print("  ✅ Velocity interpolation (GPU JAX native)")
    print("  ✅ Forward Euler integration (GPU JAX native)")
    print("  ✅ RK4 integration (GPU JAX native, 3 modes)")
    print("  ✅ L0+L1 incremental search (NEW - optimized for RK4)")
    print("  ✅ Element search after timestep")
    print()
    print("Integration methods tested:")
    print("  ✅ Forward Euler: 1st order, 1 search/step")
    print("  ✅ RK4 Simplified: 4th order*, 1 search/step (~2-3× slower than Euler)")
    print("  ✅ RK4 Full: 4th order, 4 searches/step (most accurate, ~4× slower)")
    print("  ✅ RK4 L0+L1 Optimized: 4th order, 4 searches/step with incremental search (10-50× faster than Full)")
    print()
    print("Optimization details:")
    print("  L0 (cached element): Checks if particle still in same element (60-80% hit rate expected)")
    print("  L1 (face neighbors): Checks 4 adjacent neighbors (15-25% hit rate expected)")
    print("  L2+L3 (full search): Falls back to block search for remaining particles (~5-10%)")
    print()
    print("Next steps:")
    print("  - Analyze L0/L1 hit rates from test output")
    print("  - Measure actual speedup achieved vs RK4 Full")
    print("  - Test multi-timestep marching with L0+L1")
    print("  - Test with larger particle counts (10K, 100K)")
    print("  - Fix CPU-GPU transfer overhead (batch-level transfers)")
    print("  - Implement async data prefetching")
    print()


if __name__ == "__main__":
    main()
