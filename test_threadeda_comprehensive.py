#!/usr/bin/env python3
"""
Comprehensive Integration Test: All Phases on ThreadedA Mesh

Tests all implemented phases (1-4 + GPU initial assignment) on the real
ThreadedA mesh with detailed performance monitoring.

Phases Tested:
    Phase 1: Forest structure and block partitioning
    Phase 2: Element neighbors and padded arrays
    Phase 3: Particle seeding
    Phase 4: Multi-level search
    Phase 4+: GPU initial assignment

Monitoring:
    - Step-by-step timing
    - Memory usage (RAM)
    - CPU utilization
    - GPU utilization (if available)
    - Detailed performance metrics

Expected Results:
    - ThreadedA mesh: ~3.5M elements
    - Block grid: 8×8×4 = 512 blocks
    - Heavy blocks: ~32-64 (>10K elements each)
    - Memory usage: <500 MB (target <8 GB)
    - GPU initial assignment: >1,000 particles/s
"""

import sys
import time
import psutil
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import json

sys.path.insert(0, '/home/arhashemi/Workspace/welding/JAXTrace')

# Phase 1: Forest structure
from jaxtrace.gpu.forest.block_grid import create_regular_grid
from jaxtrace.gpu.forest.block_mapper import assign_elements_to_blocks

# Phase 2: Padded arrays and neighbors
from jaxtrace.gpu.forest.padded_arrays import build_padded_block_arrays
from jaxtrace.gpu.forest.element_neighbors import build_element_adjacency

# Phase 3: Particle seeding
from jaxtrace.gpu.particle_seeding import (
    create_seeding_config_from_mesh,
    seed_particles_random_uniform,
    filter_particles_inside_mesh,
)

# Phase 4: Multi-level search
from jaxtrace.gpu.search import (
    classify_blocks,
    build_hash_bucket_arrays,
    multi_level_search_batch,
    print_performance_report,
)

# Phase 4: V2 Multi-level search (JAX vmap)
from jaxtrace.gpu.search.multi_level_search_v2 import multi_level_search_batch as multi_level_search_batch_v2

# GPU initial assignment
from jaxtrace.gpu.search import (
    initial_search_batch,
    InitialSearchStats,
)

# Mesh loading
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu


@dataclass
class SystemResources:
    """System resource snapshot."""
    timestamp: float
    cpu_percent: float
    memory_used_gb: float
    memory_percent: float
    gpu_available: bool
    gpu_util_percent: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None


@dataclass
class PhaseMetrics:
    """Metrics for a single phase."""
    phase_name: str
    duration_sec: float
    memory_before_gb: float
    memory_after_gb: float
    memory_delta_gb: float
    cpu_percent_avg: float
    gpu_util_avg: Optional[float] = None
    additional_metrics: Optional[Dict[str, Any]] = None


class ResourceMonitor:
    """Monitor system resources during test execution."""

    def __init__(self):
        self.process = psutil.Process()
        self.snapshots = []
        self.gpu_available = self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except:
            return False

    def snapshot(self) -> SystemResources:
        """Take a snapshot of current system resources."""
        mem_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent(interval=0.1)

        snapshot = SystemResources(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_used_gb=mem_info.rss / 1024**3,
            memory_percent=self.process.memory_percent(),
            gpu_available=self.gpu_available,
        )

        if self.gpu_available:
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

                snapshot.gpu_util_percent = gpu_util.gpu
                snapshot.gpu_memory_used_gb = gpu_mem.used / 1024**3
                snapshot.gpu_memory_total_gb = gpu_mem.total / 1024**3
            except:
                pass

        self.snapshots.append(snapshot)
        return snapshot

    def get_avg_metrics(self, start_idx: int, end_idx: int) -> Dict[str, float]:
        """Get average metrics between two snapshot indices."""
        if end_idx <= start_idx:
            return {}

        snapshots = self.snapshots[start_idx:end_idx]

        metrics = {
            'cpu_percent_avg': np.mean([s.cpu_percent for s in snapshots]),
            'memory_gb_avg': np.mean([s.memory_used_gb for s in snapshots]),
        }

        if self.gpu_available and snapshots[0].gpu_util_percent is not None:
            metrics['gpu_util_avg'] = np.mean([s.gpu_util_percent for s in snapshots if s.gpu_util_percent is not None])
            metrics['gpu_memory_gb_avg'] = np.mean([s.gpu_memory_used_gb for s in snapshots if s.gpu_memory_used_gb is not None])

        return metrics


class ComprehensiveTest:
    """Comprehensive integration test for all phases."""

    def __init__(self, mesh_path: str, output_dir: Path):
        self.mesh_path = mesh_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = ResourceMonitor()
        self.phase_metrics = []

        # Test configuration
        self.grid_size = (8, 8, 4)  # 512 blocks
        self.n_test_particles = 10000  # 10K particles for performance test

    def print_header(self, title: str):
        """Print section header."""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    def print_resources(self, snapshot: SystemResources, prefix: str = ""):
        """Print resource snapshot."""
        print(f"{prefix}CPU: {snapshot.cpu_percent:.1f}% | "
              f"RAM: {snapshot.memory_used_gb:.2f} GB ({snapshot.memory_percent:.1f}%)")

        if snapshot.gpu_available and snapshot.gpu_util_percent is not None:
            print(f"{prefix}GPU: {snapshot.gpu_util_percent:.1f}% | "
                  f"VRAM: {snapshot.gpu_memory_used_gb:.2f}/{snapshot.gpu_memory_total_gb:.2f} GB")

    def run_phase(self, phase_name: str, func, *args, **kwargs):
        """Run a phase with monitoring."""
        self.print_header(f"PHASE: {phase_name}")

        # Take snapshot before
        snap_before = self.monitor.snapshot()
        snapshot_start_idx = len(self.monitor.snapshots) - 1

        print(f"\n[BEFORE]")
        self.print_resources(snap_before, "  ")

        # Run phase
        print(f"\n[EXECUTING] {phase_name}...")
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        # Take snapshot after
        snap_after = self.monitor.snapshot()
        snapshot_end_idx = len(self.monitor.snapshots)

        # Compute metrics
        avg_metrics = self.monitor.get_avg_metrics(snapshot_start_idx, snapshot_end_idx)

        metrics = PhaseMetrics(
            phase_name=phase_name,
            duration_sec=duration,
            memory_before_gb=snap_before.memory_used_gb,
            memory_after_gb=snap_after.memory_used_gb,
            memory_delta_gb=snap_after.memory_used_gb - snap_before.memory_used_gb,
            cpu_percent_avg=avg_metrics.get('cpu_percent_avg', 0),
            gpu_util_avg=avg_metrics.get('gpu_util_avg'),
        )

        self.phase_metrics.append(metrics)

        print(f"\n[AFTER]")
        self.print_resources(snap_after, "  ")

        print(f"\n[METRICS]")
        print(f"  Duration: {duration:.2f} s")
        print(f"  Memory Δ: {metrics.memory_delta_gb:+.3f} GB")
        print(f"  CPU avg: {metrics.cpu_percent_avg:.1f}%")
        if metrics.gpu_util_avg is not None:
            print(f"  GPU avg: {metrics.gpu_util_avg:.1f}%")

        print(f"\n✅ {phase_name} Complete")

        return result, metrics

    def test_phase1_forest_structure(self):
        """Test Phase 1: Forest structure and block partitioning."""

        def run():
            # Load mesh
            print(f"Loading mesh: {self.mesh_path}")
            node_positions, connectivity, _ = load_mesh_from_pvtu(Path(self.mesh_path))

            print(f"  Nodes: {len(node_positions):,}")
            print(f"  Elements: {len(connectivity):,}")

            # Compute bounding box
            bbox = np.array([
                node_positions[:, 0].min(), node_positions[:, 0].max(),
                node_positions[:, 1].min(), node_positions[:, 1].max(),
                node_positions[:, 2].min(), node_positions[:, 2].max(),
            ], dtype=np.float32)

            print(f"\nBounding box:")
            print(f"  X: [{bbox[0]:.3f}, {bbox[1]:.3f}]")
            print(f"  Y: [{bbox[2]:.3f}, {bbox[3]:.3f}]")
            print(f"  Z: [{bbox[4]:.3f}, {bbox[5]:.3f}]")

            # Create block grid
            print(f"\nCreating block grid {self.grid_size}...")
            blocks = create_regular_grid(bbox, self.grid_size)
            print(f"  Total blocks: {len(blocks)}")

            # Assign elements to blocks
            print(f"\nAssigning elements to blocks...")
            element_to_block, stats = assign_elements_to_blocks(
                node_positions,
                connectivity,
                bbox,
                self.grid_size,
                verbose=True
            )

            print(f"\n[PHASE 1 STATISTICS]")
            print(f"  Elements assigned: {stats.n_elements:,}")
            print(f"  Blocks used: {stats.n_blocks_used}/{stats.n_blocks}")
            print(f"  Elements per block: {stats.min_elements} - {stats.max_elements} (avg: {stats.mean_elements:.1f})")
            print(f"  Imbalance ratio: {stats.imbalance_ratio:.2f}×")
            print(f"  Heavy blocks (>10K): {len(stats.heavy_blocks)}")

            return {
                'node_positions': node_positions,
                'connectivity': connectivity,
                'bbox': bbox,
                'blocks': blocks,
                'element_to_block': element_to_block,
                'stats': stats,
            }

        return self.run_phase("Phase 1: Forest Structure", run)

    def test_phase2_padded_arrays(self, phase1_data):
        """Test Phase 2: Element neighbors and padded arrays."""

        def run():
            element_to_block = phase1_data['element_to_block']
            stats = phase1_data['stats']
            connectivity = phase1_data['connectivity']

            # Build padded arrays
            print("Building padded block arrays...")
            padded = build_padded_block_arrays(element_to_block, stats, verbose=True)

            print(f"\n[PADDED ARRAYS]")
            print(f"  Shape: {padded.block_elements.shape}")
            print(f"  Memory: {padded.memory_mb:.1f} MB")
            print(f"  Max elements per block: {padded.max_elements_per_block}")

            # Build element adjacency
            print("\nBuilding element adjacency...")
            start = time.time()
            element_neighbors = build_element_adjacency(connectivity)
            duration = time.time() - start

            print(f"  Time: {duration:.2f} s")
            print(f"  Neighbor array shape: {element_neighbors.shape}")

            # Count statistics
            n_neighbors = np.sum(element_neighbors >= 0, axis=1)
            print(f"  Elements with neighbors: {np.sum(n_neighbors > 0):,}/{len(connectivity):,}")
            print(f"  Avg neighbors per element: {n_neighbors.mean():.2f}")
            print(f"  Max neighbors: {n_neighbors.max()}")

            return {
                'padded': padded,
                'element_neighbors': element_neighbors,
            }

        return self.run_phase("Phase 2: Padded Arrays & Neighbors", run)

    def test_phase3_particle_seeding(self, phase1_data):
        """Test Phase 3: Particle seeding."""

        def run():
            node_positions = phase1_data['node_positions']
            connectivity = phase1_data['connectivity']
            bbox = phase1_data['bbox']

            # Create seeding config
            print(f"Creating seeding config for {self.n_test_particles:,} particles...")

            # Convert bbox format: [xmin, xmax, ymin, ymax, zmin, zmax] -> bbox_min, bbox_max
            bbox_min = np.array([bbox[0], bbox[2], bbox[4]], dtype=np.float32)
            bbox_max = np.array([bbox[1], bbox[3], bbox[5]], dtype=np.float32)

            # Calculate density_per_axis to approximate n_test_particles
            # n_particles ≈ nx * ny * nz, so density ≈ n_particles^(1/3)
            density = int(np.ceil(self.n_test_particles ** (1/3)))

            config = create_seeding_config_from_mesh(
                bbox_min,
                bbox_max,
                margin=0.0,
                density_per_axis=(density, density, density)
            )

            # Seed particles
            print(f"Seeding particles randomly in bounding box...")
            particle_positions = seed_particles_random_uniform(config)

            # Filter to only particles inside mesh (using GPU initial assignment later)
            print(f"\nGenerated {len(particle_positions):,} candidate particles")
            print(f"  Position range X: [{particle_positions[:, 0].min():.3f}, {particle_positions[:, 0].max():.3f}]")
            print(f"  Position range Y: [{particle_positions[:, 1].min():.3f}, {particle_positions[:, 1].max():.3f}]")
            print(f"  Position range Z: [{particle_positions[:, 2].min():.3f}, {particle_positions[:, 2].max():.3f}]")

            print(f"\n[PARTICLE SEEDING]")
            print(f"  Particles generated: {len(particle_positions):,}")
            print(f"  Position shape: {particle_positions.shape}")
            print(f"  Note: Element assignment will be done via GPU initial search")

            return {
                'particle_positions': particle_positions,
            }

        return self.run_phase("Phase 3: Particle Seeding", run)

    def test_phase4_multi_level_search(self, phase1_data, phase2_data, phase3_data):
        """Test Phase 4: Multi-level search."""

        def run():
            node_positions = phase1_data['node_positions']
            connectivity = phase1_data['connectivity']
            blocks = phase1_data['blocks']
            padded = phase2_data['padded']
            element_neighbors = phase2_data['element_neighbors']
            particle_positions = phase3_data['particle_positions']

            # Classify blocks
            print("Classifying blocks (threshold: 10,000 elements)...")
            classification = classify_blocks(padded, threshold=10000, verbose=True)

            # Build hash buckets for heavy blocks
            print(f"\nBuilding hash buckets for {len(classification.heavy_blocks)} heavy blocks...")
            hash_bucket_data = {}
            element_centroids = np.mean(node_positions[connectivity], axis=1).astype(np.float32)

            start = time.time()
            for idx, block_id in enumerate(classification.heavy_blocks):
                if (idx + 1) % 10 == 0:
                    print(f"  Progress: {idx+1}/{len(classification.heavy_blocks)}", end='\r')

                # Get block elements
                block_elems = padded.block_elements[block_id]
                block_count = int(padded.block_sizes[block_id])
                elem_ids = block_elems[:block_count]
                elem_ids = elem_ids[elem_ids >= 0]

                if len(elem_ids) == 0:
                    continue

                centroids = element_centroids[elem_ids]
                block_bounds = blocks[block_id].bounds

                hash_arrays = build_hash_bucket_arrays(
                    block_id=block_id,
                    element_ids=elem_ids,
                    element_centroids=centroids,
                    block_bounds=block_bounds,
                    target_bucket_size=200,
                    morton_bits=10
                )

                hash_bucket_data[block_id] = hash_arrays

            duration = time.time() - start
            print(f"\n  Hash buckets built: {len(hash_bucket_data)} in {duration:.2f} s")

            # Build block neighbors
            block_neighbors_26 = np.array([b.neighbors_26 for b in blocks], dtype=np.int32)

            # For multi-level search test, we need cached element IDs
            # Use GPU initial assignment first to get initial assignments
            print(f"\nGetting initial assignments for test particles...")
            n_test = min(1000, len(particle_positions))
            test_positions = particle_positions[:n_test]

            # Use GPU initial assignment to get starting point
            test_elem_ids, test_block_ids, init_stats = initial_search_batch(
                test_positions,
                phase1_data['bbox'],
                self.grid_size,
                classification,
                padded,
                block_neighbors_26,
                hash_bucket_data,
                node_positions,
                connectivity,
                verbose=False
            )

            print(f"  Initial assignments: {init_stats.n_found:,}/{n_test:,} found")
            print (f"  Found in primary block (L2): {init_stats.n_found_in_primary_block:,}")
            print (f"  Found in neighbor blocks (L3): {init_stats.n_found_in_neighbor_blocks:,}")
            print (f"  Not found: {init_stats.n_not_found:,}")
            print (f"  Total search Time: {init_stats.total_search_time:.2f} s")
            print (f"  Throughput: {init_stats.particles_per_second:.0f} particles/s")

            # Now test multi-level search with cached elements
            print(f"\nTesting multi-level search V1 (Python loop) on {n_test:,} particles...")
            # Perturb positions slightly to test search
            test_positions_perturbed = test_positions + np.random.normal(0, 0.0001, test_positions.shape)

            element_ids_v1, block_ids_v1, search_stats_v1 = multi_level_search_batch(
                test_positions_perturbed,
                test_elem_ids,  # Use initial assignments as cache
                test_block_ids,  # Cached block IDs
                classification,
                padded.block_elements,
                padded.block_sizes,
                element_neighbors,
                block_neighbors_26,
                hash_bucket_data,
                node_positions,
                connectivity,
                verbose=True
            )

            # Compute n_found from individual hit counters
            n_found_v1 = search_stats_v1.l0_hits + search_stats_v1.l1_hits + search_stats_v1.l2_hits + search_stats_v1.l3_hits
            throughput_v1 = search_stats_v1.n_particles / search_stats_v1.total_time if search_stats_v1.total_time > 0 else 0

            print(f"\n[V1 RESULTS]")
            print(f"  Particles tested: {search_stats_v1.n_particles:,}")
            print(f"  Found: {n_found_v1:,} ({100*n_found_v1/search_stats_v1.n_particles:.1f}%)")
            print(f"  L0 (cached) hits: {search_stats_v1.l0_hits:,} ({100*search_stats_v1.l0_hits/search_stats_v1.n_particles:.1f}%)")
            print(f"  L1 (neighbors) hits: {search_stats_v1.l1_hits:,} ({100*search_stats_v1.l1_hits/search_stats_v1.n_particles:.1f}%)")
            print(f"  L2 (block) hits: {search_stats_v1.l2_hits:,} ({100*search_stats_v1.l2_hits/search_stats_v1.n_particles:.1f}%)")
            print(f"  L3 (neighbor blocks) hits: {search_stats_v1.l3_hits:,} ({100*search_stats_v1.l3_hits/search_stats_v1.n_particles:.1f}%)")
            print(f"  Throughput: {throughput_v1:.0f} particles/s")
            print(f"  Time: {search_stats_v1.total_time:.2f} s")

            # Now test V2 (JAX vmap vectorized version)
            print(f"\nTesting multi-level search V2 (JAX vmap) on {n_test:,} particles...")
            print("  Warming up JIT...")
            # Warmup run with small batch
            _, _, _ = multi_level_search_batch_v2(
                test_positions_perturbed[:10],
                test_elem_ids[:10],
                test_block_ids[:10],
                classification,
                padded.block_elements,
                padded.block_sizes,
                element_neighbors,
                block_neighbors_26,
                hash_bucket_data,
                node_positions,
                connectivity,
                verbose=False
            )

            print("  Running full test...")
            element_ids_v2, block_ids_v2, search_stats_v2 = multi_level_search_batch_v2(
                test_positions_perturbed,
                test_elem_ids,
                test_block_ids,
                classification,
                padded.block_elements,
                padded.block_sizes,
                element_neighbors,
                block_neighbors_26,
                hash_bucket_data,
                node_positions,
                connectivity,
                verbose=False
            )

            # Compute stats for V2
            n_found_v2 = search_stats_v2.l0_hits + search_stats_v2.l1_hits + search_stats_v2.l2_hits + search_stats_v2.l3_hits
            throughput_v2 = search_stats_v2.n_particles / search_stats_v2.total_time if search_stats_v2.total_time > 0 else 0

            print(f"\n[V2 RESULTS]")
            print(f"  Particles tested: {search_stats_v2.n_particles:,}")
            print(f"  Found: {n_found_v2:,} ({100*n_found_v2/search_stats_v2.n_particles:.1f}%)")
            print(f"  L0 (cached) hits: {search_stats_v2.l0_hits:,} ({100*search_stats_v2.l0_hits/search_stats_v2.n_particles:.1f}%)")
            print(f"  L1 (neighbors) hits: {search_stats_v2.l1_hits:,} ({100*search_stats_v2.l1_hits/search_stats_v2.n_particles:.1f}%)")
            print(f"  L2 (block) hits: {search_stats_v2.l2_hits:,} ({100*search_stats_v2.l2_hits/search_stats_v2.n_particles:.1f}%)")
            print(f"  L3 (neighbor blocks) hits: {search_stats_v2.l3_hits:,} ({100*search_stats_v2.l3_hits/search_stats_v2.n_particles:.1f}%)")
            print(f"  Throughput: {throughput_v2:.0f} particles/s")
            print(f"  Time: {search_stats_v2.total_time:.2f} s")

            # Compare V1 vs V2
            speedup = throughput_v2 / throughput_v1 if throughput_v1 > 0 else 0
            matching = np.sum(element_ids_v1 == element_ids_v2)
            match_rate = 100 * matching / n_test

            print(f"\n[V1 vs V2 COMPARISON]")
            print(f"  V1 throughput:  {throughput_v1:>8.0f} p/s  ({search_stats_v1.total_time:.2f} s)")
            print(f"  V2 throughput:  {throughput_v2:>8.0f} p/s  ({search_stats_v2.total_time:.2f} s)")
            print(f"  Speedup:        {speedup:>8.1f}×")
            print(f"  Results match:  {matching:>8,}/{n_test:,} ({match_rate:.1f}%)")

            if speedup > 10:
                print(f"  ✅ V2 IS SIGNIFICANTLY FASTER: {speedup:.1f}× speedup achieved!")
            elif speedup > 1.5:
                print(f"  ✅ V2 IS FASTER: {speedup:.1f}× speedup achieved!")
            elif speedup > 1.0:
                print(f"  ⚠️  V2 IS SLIGHTLY FASTER: {speedup:.1f}× speedup")
            else:
                print(f"  ❌ V2 IS SLOWER: {speedup:.1f}×")

            if match_rate < 95:
                print(f"  ⚠️  WARNING: Low match rate ({match_rate:.1f}%)")
            else:
                print(f"  ✅ Results match well ({match_rate:.1f}%)")

            # Use V1 stats for backward compatibility
            search_stats = search_stats_v1
            element_ids = element_ids_v1
            block_ids_updated = block_ids_v1

            return {
                'classification': classification,
                'hash_bucket_data': hash_bucket_data,
                'block_neighbors_26': block_neighbors_26,
                'search_stats': search_stats,
            }

        return self.run_phase("Phase 4: Multi-Level Search", run)

    def test_gpu_initial_assignment(self, phase1_data, phase2_data, phase4_data):
        """Test GPU initial assignment."""

        def run():
            node_positions = phase1_data['node_positions']
            connectivity = phase1_data['connectivity']
            padded = phase2_data['padded']
            classification = phase4_data['classification']
            hash_bucket_data = phase4_data['hash_bucket_data']
            block_neighbors_26 = phase4_data['block_neighbors_26']

            # Generate random test particles in domain
            bbox = phase1_data['bbox']
            n_test = 5000

            print(f"Testing GPU initial assignment on {n_test:,} random particles...")

            # Random positions in bounding box
            test_positions = np.random.uniform(
                [bbox[0], bbox[2], bbox[4]],
                [bbox[1], bbox[3], bbox[5]],
                (n_test, 3)
            ).astype(np.float32)

            element_ids, block_ids, stats = initial_search_batch(
                test_positions,
                bbox,
                self.grid_size,
                classification,
                padded,
                block_neighbors_26,
                hash_bucket_data,
                node_positions,
                connectivity,
                verbose=True
            )

            print(f"\n[GPU INITIAL ASSIGNMENT RESULTS]")
            print(f"  Particles: {stats.n_particles:,}")
            print(f"  Found: {stats.n_found:,} ({100*stats.n_found/stats.n_particles:.1f}%)")
            print(f"  - Primary block: {stats.n_found_in_primary_block:,} ({100*stats.n_found_in_primary_block/stats.n_particles:.1f}%)")
            print(f"  - Neighbor blocks: {stats.n_found_in_neighbor_blocks:,} ({100*stats.n_found_in_neighbor_blocks/stats.n_particles:.1f}%)")
            print(f"  Not found: {stats.n_not_found:,} ({100*stats.n_not_found/stats.n_particles:.1f}%)")
            print(f"  Throughput: {stats.particles_per_second:.0f} particles/s")
            print(f"  Time: {stats.total_search_time:.2f} s")

            return {
                'initial_stats': stats,
            }

        return self.run_phase("GPU Initial Assignment", run)

    def generate_report(self):
        """Generate comprehensive test report."""
        self.print_header("COMPREHENSIVE TEST REPORT")

        print("\n" + "-" * 80)
        print("PHASE SUMMARY")
        print("-" * 80)

        total_time = sum(m.duration_sec for m in self.phase_metrics)
        total_memory = self.phase_metrics[-1].memory_after_gb - self.phase_metrics[0].memory_before_gb

        for i, metrics in enumerate(self.phase_metrics, 1):
            print(f"\n{i}. {metrics.phase_name}")
            print(f"   Duration: {metrics.duration_sec:.2f} s ({100*metrics.duration_sec/total_time:.1f}% of total)")
            print(f"   Memory Δ: {metrics.memory_delta_gb:+.3f} GB")
            print(f"   CPU avg: {metrics.cpu_percent_avg:.1f}%")
            if metrics.gpu_util_avg is not None:
                print(f"   GPU avg: {metrics.gpu_util_avg:.1f}%")

        print("\n" + "-" * 80)
        print("TOTALS")
        print("-" * 80)
        print(f"Total time: {total_time:.2f} s ({total_time/60:.1f} min)")
        print(f"Total memory increase: {total_memory:+.3f} GB")
        print(f"Final memory usage: {self.phase_metrics[-1].memory_after_gb:.2f} GB")

        # Save to JSON
        report_path = self.output_dir / "test_report.json"
        report_data = {
            'test_config': {
                'mesh_path': self.mesh_path,
                'grid_size': self.grid_size,
                'n_test_particles': self.n_test_particles,
            },
            'phase_metrics': [asdict(m) for m in self.phase_metrics],
            'totals': {
                'total_time_sec': total_time,
                'total_memory_delta_gb': total_memory,
                'final_memory_gb': self.phase_metrics[-1].memory_after_gb,
            },
        }

        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\n✅ Report saved to: {report_path}")

    def run_all(self):
        """Run all phases sequentially."""
        self.print_header("COMPREHENSIVE INTEGRATION TEST - THREADEDA MESH")
        print(f"Mesh: {self.mesh_path}")
        print(f"Grid size: {self.grid_size}")
        print(f"Test particles: {self.n_test_particles:,}")
        print(f"Output directory: {self.output_dir}")

        # Phase 1
        phase1_data, _ = self.test_phase1_forest_structure()

        # Phase 2
        phase2_data, _ = self.test_phase2_padded_arrays(phase1_data)

        # Phase 3
        phase3_data, _ = self.test_phase3_particle_seeding(phase1_data)

        # Phase 4
        phase4_data, _ = self.test_phase4_multi_level_search(phase1_data, phase2_data, phase3_data)

        # GPU Initial Assignment
        gpu_data, _ = self.test_gpu_initial_assignment(phase1_data, phase2_data, phase4_data)

        # Generate report
        self.generate_report()

        print("\n" + "=" * 80)
        print("✅ COMPREHENSIVE TEST COMPLETE")
        print("=" * 80)


def main():
    """Main entry point."""

    # Configuration
    # Use timestep 20+ as mesh is refined during first 20 timesteps
    mesh_path = "/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_20.pvtu"
    output_dir = Path("logs/threadeda_comprehensive_test")

    print(f"\nNote: Using timestep 20 as mesh is refined during first 20 timesteps")
    print(f"      Expected ~3.5M elements at this timestep\n")

    # Run test
    test = ComprehensiveTest(mesh_path, output_dir)
    test.run_all()


if __name__ == "__main__":
    main()
