#!/usr/bin/env python3
"""
Comprehensive V5 Performance and Accuracy Test.

This test validates the V5 block-local search implementation with:
1. Detailed resource monitoring (GPU/CPU memory, utilization)
2. Per-stage timing and performance metrics
3. Accuracy validation against CPU ground truth
4. Comparison with V4 implementation
5. Comprehensive logging and reporting

Test Stages:
    1. Mesh Loading
    2. Block Infrastructure Building
    3. Element Neighbor Computation
    4. Padded Block Array Creation
    5. Particle Seeding
    6. CPU Search (Ground Truth)
    7. V5 GPU Search (Block-Local)
    8. V4 GPU Search (Global - if memory allows)
    9. Accuracy Validation
    10. Performance Analysis

Output:
    - Detailed console output with progress tracking
    - JSON log: logs/v5_performance_test.json
    - Summary report: logs/v5_performance_report.txt

Author: JAXTrace Team
Date: 2025-11-05
"""

import sys
import os
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from utils.resource_monitor import ResourceMonitor
from jaxtrace.io import VTKUnstructuredTimeSeriesReader


def load_mesh_and_fields(mesh_dir: Path, monitor: ResourceMonitor) -> Dict:
    """Stage 1: Load mesh and field data."""
    with monitor.stage("1. Mesh Loading"):
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        from glob import glob

        # Find .pvtu files
        pattern = str(mesh_dir / "*.pvtu")
        files = sorted(glob(pattern))

        if not files:
            raise FileNotFoundError(f"No .pvtu files found: {pattern}")

        print(f"  Found {len(files)} .pvtu files")
        print(f"  Loading: {files[0]}")

        # Load first file
        reader = vtk.vtkXMLPUnstructuredGridReader()
        reader.SetFileName(files[0])
        reader.Update()
        mesh = reader.GetOutput()

        # Extract mesh data
        nodes = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float32)
        n_points = nodes.shape[0]

        # Extract connectivity (tetrahedral)
        connectivity_list = []
        for i in range(mesh.GetNumberOfCells()):
            cell = mesh.GetCell(i)
            if cell.GetCellType() == vtk.VTK_TETRA:
                ids = [cell.GetPointId(j) for j in range(4)]
                connectivity_list.append(ids)

        connectivity = np.array(connectivity_list, dtype=np.int32)

        print(f"  Mesh: {n_points:,} nodes, {len(connectivity):,} elements")

        return {
            'nodes': nodes,
            'connectivity': connectivity
        }


def build_block_infrastructure(mesh_data: Dict, grid_size: tuple, monitor: ResourceMonitor) -> Dict:
    """Stage 2: Build block infrastructure."""
    with monitor.stage("2. Block Infrastructure"):
        from jaxtrace.gpu.forest.block_builder import create_regular_forest_grid
        from jaxtrace.gpu.forest.block_mapper import assign_elements_to_blocks

        # Compute domain bounds
        nodes = mesh_data['nodes']
        bbox = np.array([
            nodes[:, 0].min(), nodes[:, 0].max(),
            nodes[:, 1].min(), nodes[:, 1].max(),
            nodes[:, 2].min(), nodes[:, 2].max()
        ], dtype=np.float32)

        print(f"  Domain bounds: [{bbox[0]:.4f}, {bbox[1]:.4f}] × "
              f"[{bbox[2]:.4f}, {bbox[3]:.4f}] × [{bbox[4]:.4f}, {bbox[5]:.4f}]")
        print(f"  Grid size: {grid_size}")

        # Create blocks
        blocks = create_regular_forest_grid(bbox, grid_size)
        print(f"  Created {len(blocks)} blocks")

        # Assign elements to blocks
        element_to_block, stats = assign_elements_to_blocks(
            mesh_data['nodes'],
            mesh_data['connectivity'],
            bbox,
            grid_size,
            verbose=False
        )

        return {
            'blocks': blocks,
            'element_to_block': element_to_block,
            'bbox': bbox,
            'grid_size': grid_size
        }


def build_octrees(mesh_data: Dict, block_data: Dict, monitor: ResourceMonitor) -> Dict:
    """Stage 3: Build simple octrees per block."""
    with monitor.stage("3. Octree Building"):
        from dataclasses import dataclass

        @dataclass
        class SimpleOctree:
            block_id: int
            sorted_element_IDs: np.ndarray

        octrees = {}
        n_blocks = len(block_data['blocks'])

        for block in block_data['blocks']:
            elem_ids = np.where(block_data['element_to_block'] == block.block_id)[0]
            if len(elem_ids) > 0:
                octrees[block.block_id] = SimpleOctree(
                    block_id=block.block_id,
                    sorted_element_IDs=elem_ids.astype(np.int32)
                )

        print(f"  Built {len(octrees)} octrees")

        return octrees


def build_element_neighbors(connectivity: np.ndarray, monitor: ResourceMonitor) -> np.ndarray:
    """Stage 4: Build element neighbor adjacency."""
    with monitor.stage("4. Element Neighbor Computation"):
        from jaxtrace.gpu.forest.element_neighbors import build_element_adjacency

        element_neighbors = build_element_adjacency(
            connectivity,
            max_neighbors=32
        )

        print(f"  Element neighbors: {element_neighbors.shape}")

        return element_neighbors


def build_v5_block_arrays(octrees: Dict, block_data: Dict, monitor: ResourceMonitor) -> object:
    """Stage 5: Build V5 padded block arrays."""
    with monitor.stage("5. V5 Padded Block Array Creation"):
        from jaxtrace.gpu.forest.block_elements import (
            build_padded_block_arrays,
            validate_block_arrays
        )

        block_arrays = build_padded_block_arrays(
            octrees,
            block_data['element_to_block'],
            block_data['blocks'],
            verbose=True
        )

        print(f"\n  Validating block arrays...")
        is_valid = validate_block_arrays(
            block_arrays,
            block_data['element_to_block'],
            verbose=True
        )

        if not is_valid:
            print("  ⚠️  Validation warnings detected")

        return block_arrays


def seed_particles(bbox: np.ndarray, n_particles: int, monitor: ResourceMonitor) -> np.ndarray:
    """Stage 6: Seed particles."""
    with monitor.stage("6. Particle Seeding"):
        np.random.seed(42)
        particle_positions = np.random.uniform(
            low=[bbox[0], bbox[2], bbox[4]],
            high=[bbox[1], bbox[3], bbox[5]],
            size=(n_particles, 3)
        )

        print(f"  Seeded {n_particles:,} particles")
        print(f"  Position range: [{particle_positions.min():.4f}, {particle_positions.max():.4f}]")

        return particle_positions


def run_cpu_search(
    particle_positions: np.ndarray,
    mesh_data: Dict,
    block_data: Dict,
    octrees: Dict,
    monitor: ResourceMonitor
) -> tuple:
    """Stage 7: CPU search (ground truth)."""
    with monitor.stage("7. CPU Search (Ground Truth)"):
        from jaxtrace.gpu.initial_search_jax import find_initial_elements_batch, GPUConfig

        config = GPUConfig(force_cpu=True)

        partition_data = {
            'bbox_global': block_data['bbox'],
            'bbox_min': block_data['bbox'][[0, 2, 4]],
            'bbox_max': block_data['bbox'][[1, 3, 5]],
            'grid_size': block_data['grid_size'],
            'block_size': (block_data['bbox'][1] - block_data['bbox'][0]) / block_data['grid_size'][0]
        }

        element_IDs_cpu, stats_cpu = find_initial_elements_batch(
            particle_positions,
            mesh_data,
            partition_data,
            octrees,
            config=config,
            verbose=False  # Suppress internal logging
        )

        print(f"  Found: {stats_cpu['n_found']:,}/{stats_cpu['n_particles']:,}")
        print(f"  Time: {stats_cpu['time_elapsed']:.2f}s")
        print(f"  Time/particle: {stats_cpu['time_per_particle_ms']:.3f} ms")

        return element_IDs_cpu, stats_cpu


def run_v5_gpu_search(
    particle_positions: np.ndarray,
    mesh_data: Dict,
    block_data: Dict,
    octrees: Dict,
    element_neighbors: np.ndarray,
    monitor: ResourceMonitor
) -> tuple:
    """Stage 8: V5 GPU search (block-local)."""
    with monitor.stage("8. V5 GPU Search (Block-Local)"):
        from jaxtrace.gpu.initial_search_jax import find_initial_elements_batch, GPUConfig

        config = GPUConfig(
            use_gpu_initial_search=True,
            use_block_local_search=True,
            use_gpu_multi_level=True,
            validate_block_arrays=False  # Already validated
        )

        partition_data = {
            'bbox_global': block_data['bbox'],
            'bbox_min': block_data['bbox'][[0, 2, 4]],
            'bbox_max': block_data['bbox'][[1, 3, 5]],
            'grid_size': block_data['grid_size'],
            'block_size': (block_data['bbox'][1] - block_data['bbox'][0]) / block_data['grid_size'][0]
        }

        element_IDs_v5, stats_v5 = find_initial_elements_batch(
            particle_positions,
            mesh_data,
            partition_data,
            octrees,
            blocks=block_data['blocks'],
            element_to_block=block_data['element_to_block'],
            element_neighbors=element_neighbors,
            config=config,
            verbose=True  # Show V5 pipeline details
        )

        return element_IDs_v5, stats_v5


def run_v4_gpu_search(
    particle_positions: np.ndarray,
    mesh_data: Dict,
    block_data: Dict,
    octrees: Dict,
    monitor: ResourceMonitor
) -> Optional[tuple]:
    """Stage 9: V4 GPU search (global - if memory allows)."""
    try:
        with monitor.stage("9. V4 GPU Search (Global - Legacy)"):
            from jaxtrace.gpu.initial_search_jax import find_initial_elements_batch, GPUConfig

            config = GPUConfig(
                use_gpu_initial_search=True,
                use_block_local_search=False,  # Disable V5, use V4
                use_gpu_multi_level=False
            )

            partition_data = {
                'bbox_global': block_data['bbox'],
                'grid_size': block_data['grid_size']
            }

            element_IDs_v4, stats_v4 = find_initial_elements_batch(
                particle_positions,
                mesh_data,
                partition_data,
                octrees,
                config=config,
                verbose=False
            )

            print(f"  Found: {stats_v4['n_found']:,}")
            print(f"  Time: {stats_v4['time_elapsed']:.2f}s")

            return element_IDs_v4, stats_v4

    except Exception as e:
        print(f"  ❌ V4 search failed (expected for large meshes): {e}")
        return None


def validate_accuracy(
    element_IDs_cpu: np.ndarray,
    element_IDs_v5: np.ndarray,
    element_IDs_v4: Optional[np.ndarray],
    monitor: ResourceMonitor
) -> Dict:
    """Stage 10: Accuracy validation."""
    with monitor.stage("10. Accuracy Validation"):
        n_particles = len(element_IDs_cpu)

        # V5 vs CPU
        v5_matches = np.sum(element_IDs_v5 == element_IDs_cpu)
        v5_accuracy = 100 * v5_matches / n_particles

        print(f"\n  V5 vs CPU:")
        print(f"    Matches: {v5_matches:,}/{n_particles:,} ({v5_accuracy:.2f}%)")

        if v5_accuracy < 100.0:
            mismatches = np.where(element_IDs_v5 != element_IDs_cpu)[0]
            print(f"    Mismatches: {len(mismatches)}")
            for idx in mismatches[:10]:
                print(f"      Particle {idx}: V5={element_IDs_v5[idx]}, CPU={element_IDs_cpu[idx]}")

        # V4 vs CPU (if available)
        v4_accuracy = None
        if element_IDs_v4 is not None:
            v4_matches = np.sum(element_IDs_v4 == element_IDs_cpu)
            v4_accuracy = 100 * v4_matches / n_particles
            print(f"\n  V4 vs CPU:")
            print(f"    Matches: {v4_matches:,}/{n_particles:,} ({v4_accuracy:.2f}%)")

        return {
            'v5_accuracy': v5_accuracy,
            'v4_accuracy': v4_accuracy,
            'v5_matches': v5_matches,
            'n_particles': n_particles
        }


def analyze_performance(
    stats_cpu: Dict,
    stats_v5: Dict,
    stats_v4: Optional[Dict],
    monitor: ResourceMonitor
) -> Dict:
    """Stage 11: Performance analysis."""
    with monitor.stage("11. Performance Analysis"):
        # Speedups
        v5_speedup = stats_cpu['time_elapsed'] / stats_v5['time_elapsed']

        print(f"\n  CPU Performance:")
        print(f"    Time: {stats_cpu['time_elapsed']:.2f}s")
        print(f"    Time/particle: {stats_cpu['time_per_particle_ms']:.3f} ms")

        print(f"\n  V5 GPU Performance:")
        print(f"    Time: {stats_v5['time_elapsed']:.2f}s")
        print(f"    Time/particle: {stats_v5['time_per_particle_ms']:.3f} ms")
        print(f"    Speedup vs CPU: {v5_speedup:.1f}×")

        if stats_v5.get('used_v5', False):
            print(f"    ✅ V5 block-local search used")
        else:
            print(f"    ⚠️  V5 not enabled (fell back to V4/CPU)")

        # V4 comparison
        v4_speedup = None
        if stats_v4 is not None:
            v4_speedup = stats_cpu['time_elapsed'] / stats_v4['time_elapsed']
            v5_vs_v4 = stats_v4['time_elapsed'] / stats_v5['time_elapsed']

            print(f"\n  V4 GPU Performance:")
            print(f"    Time: {stats_v4['time_elapsed']:.2f}s")
            print(f"    Time/particle: {stats_v4['time_per_particle_ms']:.3f} ms")
            print(f"    Speedup vs CPU: {v4_speedup:.1f}×")
            print(f"    V5 vs V4: {v5_vs_v4:.1f}× {'faster' if v5_vs_v4 > 1 else 'slower'}")

        return {
            'v5_speedup_vs_cpu': v5_speedup,
            'v4_speedup_vs_cpu': v4_speedup
        }


def generate_report(
    monitor: ResourceMonitor,
    accuracy_results: Dict,
    performance_results: Dict,
    output_file: str
):
    """Generate comprehensive performance report."""
    print("\n" + "="*80)
    print("📊 GENERATING COMPREHENSIVE REPORT")
    print("="*80)

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("V5 BLOCK-LOCAL SEARCH - COMPREHENSIVE PERFORMANCE REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Accuracy Section
        f.write("ACCURACY RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Particles: {accuracy_results['n_particles']:,}\n")
        f.write(f"V5 Accuracy: {accuracy_results['v5_accuracy']:.2f}%\n")
        f.write(f"V5 Matches: {accuracy_results['v5_matches']:,}/{accuracy_results['n_particles']:,}\n")
        if accuracy_results['v4_accuracy'] is not None:
            f.write(f"V4 Accuracy: {accuracy_results['v4_accuracy']:.2f}%\n")
        f.write("\n")

        # Performance Section
        f.write("PERFORMANCE RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"V5 Speedup vs CPU: {performance_results['v5_speedup_vs_cpu']:.1f}×\n")
        if performance_results['v4_speedup_vs_cpu'] is not None:
            f.write(f"V4 Speedup vs CPU: {performance_results['v4_speedup_vs_cpu']:.1f}×\n")
            improvement = performance_results['v5_speedup_vs_cpu'] / performance_results['v4_speedup_vs_cpu']
            f.write(f"V5 Improvement over V4: {improvement:.1f}×\n")
        f.write("\n")

        # Resource Usage
        f.write("RESOURCE USAGE BY STAGE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Stage':<45} {'Time(s)':<10} {'GPU(MB)':<10} {'CPU(MB)':<10}\n")
        f.write("-"*80 + "\n")

        for stage in monitor.stages:
            f.write(f"{stage.name:<45} {stage.duration_s:<10.2f} "
                   f"{stage.gpu_memory_peak_mb:<10.1f} {stage.cpu_memory_peak_mb:<10.1f}\n")

        f.write("-"*80 + "\n")
        f.write(f"{'TOTAL':<45} {sum(s.duration_s for s in monitor.stages):<10.2f} "
               f"{max(s.gpu_memory_peak_mb for s in monitor.stages):<10.1f} "
               f"{max(s.cpu_memory_peak_mb for s in monitor.stages):<10.1f}\n")
        f.write("\n")

        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-"*80 + "\n")

        if accuracy_results['v5_accuracy'] == 100.0:
            f.write("✅ ACCURACY: Perfect match with CPU ground truth\n")
        else:
            f.write(f"⚠️  ACCURACY: {100 - accuracy_results['v5_accuracy']:.2f}% mismatch\n")
            f.write("   → Investigate mismatched particles\n")

        if performance_results['v5_speedup_vs_cpu'] > 10:
            f.write("✅ PERFORMANCE: Excellent speedup vs CPU\n")
        elif performance_results['v5_speedup_vs_cpu'] > 5:
            f.write("⚡ PERFORMANCE: Good speedup, can be optimized\n")
        else:
            f.write("⚠️  PERFORMANCE: Low speedup, investigate bottlenecks\n")

        peak_gpu = max(s.gpu_memory_peak_mb for s in monitor.stages)
        if peak_gpu < 500:
            f.write(f"✅ MEMORY: Excellent GPU usage ({peak_gpu:.0f} MB)\n")
        elif peak_gpu < 2000:
            f.write(f"⚡ MEMORY: Moderate GPU usage ({peak_gpu:.0f} MB)\n")
        else:
            f.write(f"⚠️  MEMORY: High GPU usage ({peak_gpu:.0f} MB)\n")

        f.write("\n" + "="*80 + "\n")

    print(f"📄 Report saved to: {output_file}")


def main():
    """Main test function."""
    print("="*80)
    print("V5 COMPREHENSIVE PERFORMANCE TEST")
    print("="*80)

    # Configuration
    mesh_dir = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule")
    n_particles = 1000  # Start with 1K particles
    grid_size = (4, 4, 2)  # 32 blocks

    # Check mesh exists
    if not mesh_dir.exists():
        print(f"❌ Mesh directory not found: {mesh_dir}")
        print("   Please update mesh_dir in the script")
        return 1

    # Initialize resource monitor
    monitor = ResourceMonitor(enable_gpu=True, enable_cpu=True)

    try:
        # Stage 1: Load mesh
        mesh_data = load_mesh_and_fields(mesh_dir, monitor)

        # Stage 2: Build blocks
        block_data = build_block_infrastructure(mesh_data, grid_size, monitor)

        # Stage 3: Build octrees
        octrees = build_octrees(mesh_data, block_data, monitor)

        # Stage 4: Build element neighbors
        element_neighbors = build_element_neighbors(mesh_data['connectivity'], monitor)

        # Stage 5: Build V5 block arrays
        block_arrays = build_v5_block_arrays(octrees, block_data, monitor)

        # Stage 6: Seed particles
        particle_positions = seed_particles(block_data['bbox'], n_particles, monitor)

        # Prepare mesh_data dict for search functions
        search_mesh_data = {
            'positions': mesh_data['nodes'],
            'connectivity': mesh_data['connectivity']
        }

        # Stage 7: CPU search (ground truth)
        element_IDs_cpu, stats_cpu = run_cpu_search(
            particle_positions, search_mesh_data, block_data, octrees, monitor
        )

        # Stage 8: V5 GPU search
        element_IDs_v5, stats_v5 = run_v5_gpu_search(
            particle_positions, search_mesh_data, block_data, octrees,
            element_neighbors, monitor
        )

        # Stage 9: V4 GPU search (optional, may OOM)
        result_v4 = run_v4_gpu_search(
            particle_positions, search_mesh_data, block_data, octrees, monitor
        )
        element_IDs_v4, stats_v4 = result_v4 if result_v4 else (None, None)

        # Stage 10: Validate accuracy
        accuracy_results = validate_accuracy(
            element_IDs_cpu, element_IDs_v5, element_IDs_v4, monitor
        )

        # Stage 11: Analyze performance
        performance_results = analyze_performance(
            stats_cpu, stats_v5, stats_v4, monitor
        )

        # Print summary
        monitor.print_summary()

        # Save logs
        os.makedirs("logs", exist_ok=True)
        monitor.save_log("logs/v5_performance_test.json")

        # Generate report
        generate_report(
            monitor, accuracy_results, performance_results,
            "logs/v5_performance_report.txt"
        )

        # Final verdict
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)

        success = (
            accuracy_results['v5_accuracy'] >= 99.5 and
            stats_v5.get('used_v5', False) and
            performance_results['v5_speedup_vs_cpu'] > 1.0
        )

        if success:
            print("✅ ALL TESTS PASSED")
            print("   - V5 block-local search working correctly")
            print("   - Accuracy ≥ 99.5%")
            print("   - Performance improvement vs CPU")
            print("   - Ready for production use")
        else:
            print("⚠️  TESTS INCOMPLETE")
            if accuracy_results['v5_accuracy'] < 99.5:
                print(f"   - Accuracy: {accuracy_results['v5_accuracy']:.1f}% (target: ≥99.5%)")
            if not stats_v5.get('used_v5', False):
                print("   - V5 not enabled (check dependencies)")
            if performance_results['v5_speedup_vs_cpu'] <= 1.0:
                print("   - No speedup vs CPU (check GPU availability)")

        print("="*80)

        return 0 if success else 1

    except Exception as e:
        import traceback
        print(f"\n❌ Test failed with exception:")
        print(traceback.format_exc())
        monitor.print_summary()
        return 1


if __name__ == "__main__":
    sys.exit(main())
