#!/usr/bin/env python3
"""
Test to Compare Octree vs Blockwise Initialization for Particle Assignment

This test:
1. Loads real mesh data
2. Initializes both octree and blockwise structures
3. Creates particles at element centroids with tiny random perturbations
4. Runs both octree search and blockwise search
5. Compares results against known ground truth (true element IDs)
6. Reports accuracy and performance metrics for both methods

Expected Outcomes:
- Both methods should find 100% of particles (since they're inside elements)
- Performance comparison will show which is faster for initial assignment
- Accuracy comparison will verify both methods return correct element IDs
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Tuple, Dict

# Force CPU-GPU memory management
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# JAXTrace imports
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.forest.block_grid import create_regular_grid
from jaxtrace.gpu.forest.block_mapper import assign_elements_to_blocks
from jaxtrace.gpu.forest.padded_arrays import build_padded_block_arrays
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search import classify_blocks
from jaxtrace.gpu.search.hash_bucket import build_hash_bucket_arrays
from jaxtrace.gpu.search.initial_assignment import initial_search_batch
from jaxtrace.gpu.search.octree_builder import build_octree_for_level, flatten_octree_to_arrays
from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu
from jaxtrace.gpu.search.octree_search_gpu import search_level2_octree_scan


def load_mesh_and_initialize_structures(
    mesh_path: Path,
    grid_size: Tuple[int, int, int] = (8, 8, 4),  # 256 blocks like production
    octree_max_depth: int = 15,
    octree_max_leaf_size: int = 50,
    octree_level_threshold: float = 1.1
) -> Dict:
    """
    Load mesh and initialize both octree and blockwise structures.

    Returns dict with:
    - node_positions, connectivity, velocity_field
    - element_neighbors, bbox
    - octree_metadata_gpu, octree_elements_gpu
    - padded_arrays, classification, block_neighbors_26, hash_bucket_data
    - mesh_gpu
    - element_centroids (for particle generation)
    """
    print("=" * 80)
    print("LOADING MESH AND INITIALIZING STRUCTURES")
    print("=" * 80)
    print()

    # ========================================================================
    # Load Mesh
    # ========================================================================
    print(f"Loading mesh from: {mesh_path}")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    t_load = time.time()
    node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
        mesh_path,
        field_name='Displacement'
    )

    print(f"✓ Loaded mesh: {len(node_positions):,} nodes, {len(connectivity):,} elements")
    print(f"  Time: {time.time() - t_load:.2f} s")
    print()

    # Ensure velocity is 3D and float32
    if velocity_field.shape[1] == 2:
        velocity_field = np.column_stack([velocity_field, np.zeros(velocity_field.shape[0])])
    velocity_field = velocity_field.astype(np.float32)

    # Build element neighbors
    t_neighbors = time.time()
    print("Building element neighbors...")
    element_neighbors = build_element_neighbors_array(connectivity, verbose=False)
    print(f"✓ Element neighbors built ({time.time() - t_neighbors:.2f} s)")
    print()

    # Compute bbox
    bbox = [
        node_positions[:, 0].min(), node_positions[:, 0].max(),
        node_positions[:, 1].min(), node_positions[:, 1].max(),
        node_positions[:, 2].min(), node_positions[:, 2].max()
    ]

    # ========================================================================
    # Initialize OCTREE Structure
    # ========================================================================
    print("-" * 80)
    print("INITIALIZING OCTREE")
    print("-" * 80)
    print()

    # Load LEVEL field for octree building
    print("Loading LEVEL field from mesh...")
    import vtk
    from vtk.util import numpy_support

    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(mesh_path))
    reader.Update()
    vtk_mesh = reader.GetOutput()

    cell_data = vtk_mesh.GetCellData()
    point_data = vtk_mesh.GetPointData()

    level_field = None

    if cell_data.HasArray('LEVEL'):
        level_field = numpy_support.vtk_to_numpy(cell_data.GetArray('LEVEL')).astype(np.float32)
        print(f"✓ Found LEVEL in cell data: {len(level_field):,} elements")
    elif point_data.HasArray('LEVEL'):
        print(f"✓ Found LEVEL in point data")
        node_level = numpy_support.vtk_to_numpy(point_data.GetArray('LEVEL')).astype(np.float32)
        level_field = np.array([
            node_level[connectivity[i]].max()
            for i in range(len(connectivity))
        ], dtype=np.float32)
        print(f"✓ Computed element levelset: {len(level_field):,} elements")
    print()

    # Compute element centroids
    print("Computing element centroids...")
    element_centroids = np.array([
        node_positions[connectivity[i]].mean(axis=0)
        for i in range(len(connectivity))
    ], dtype=np.float32)
    element_ids_octree = np.arange(len(connectivity), dtype=np.int32)
    print(f"✓ Computed {len(element_centroids):,} centroids")
    print()

    # Build octree
    print(f"Building octree (max_depth={octree_max_depth}, max_leaf_size={octree_max_leaf_size})...")
    t_octree = time.time()

    nodes, metadata = build_octree_for_level(
        element_centroids,
        element_ids_octree,
        level_field=level_field,
        level_threshold=octree_level_threshold,
        max_depth=octree_max_depth,
        max_leaf_size=octree_max_leaf_size,
        use_levelset=True
    )

    print(f"✓ Built octree ({time.time() - t_octree:.2f} s)")
    print(f"  Filtered elements: {metadata['n_elements']:,}/{len(connectivity):,}")
    print(f"  Total nodes: {metadata['n_nodes']:,}")
    print(f"  Max depth: {metadata['max_depth']}")
    print()

    # Flatten to GPU-compatible arrays
    node_metadata_np, node_elements_np = flatten_octree_to_arrays(nodes, max_leaf_size=octree_max_leaf_size)

    # Upload octree to GPU
    print("Uploading octree to GPU...")
    octree_metadata_gpu = jax.device_put(node_metadata_np)
    octree_elements_gpu = jax.device_put(node_elements_np)
    print(f"✓ Octree uploaded to GPU")
    print()

    # ========================================================================
    # Initialize BLOCKWISE Structure
    # ========================================================================
    print("-" * 80)
    print("INITIALIZING BLOCKWISE STRUCTURE")
    print("-" * 80)
    print()

    # Create block grid
    print(f"Creating regular grid {grid_size}...")
    t_grid = time.time()
    blocks = create_regular_grid(bbox, grid_size)
    print(f"✓ Created {grid_size[0]}×{grid_size[1]}×{grid_size[2]} grid ({time.time() - t_grid:.2f} s)")
    print(f"  Total blocks: {len(blocks)}")
    print()

    # Assign elements to blocks
    print("Assigning elements to blocks...")
    t_assign = time.time()
    element_to_block, stats = assign_elements_to_blocks(
        node_positions,
        connectivity,
        bbox,
        grid_size,
        verbose=False
    )
    print(f"✓ Element assignment ({time.time() - t_assign:.2f} s)")
    print(f"  Blocks used: {stats.n_blocks_used}/{stats.n_blocks}")
    print(f"  Elements per block: {stats.min_elements} - {stats.max_elements}")
    print()

    # Build padded arrays
    print("Building padded block arrays...")
    t_padded = time.time()
    padded_arrays = build_padded_block_arrays(
        element_to_block,
        stats,
        node_positions=node_positions,
        connectivity=connectivity,
        element_neighbors=element_neighbors,
        verbose=False
    )
    print(f"✓ Built padded arrays ({time.time() - t_padded:.2f} s)")
    print(f"  Shape: {padded_arrays.block_elements.shape}")
    print(f"  Memory: {padded_arrays.memory_mb:.2f} MB")
    print()

    # Classify blocks
    print("Classifying blocks...")
    t_classify = time.time()
    classification = classify_blocks(padded_arrays, threshold=10000, verbose=False)
    print(f"✓ Classified blocks ({time.time() - t_classify:.2f} s)")
    print(f"  Light blocks: {len(classification.light_blocks)}")
    print(f"  Heavy blocks: {len(classification.heavy_blocks)}")
    print()

    # Build 26-connectivity for blocks
    print("Building block 26-connectivity...")
    t_neighbors = time.time()
    block_neighbors_26 = np.array([b.neighbors_26 for b in blocks], dtype=np.int32)
    print(f"✓ Built 26-connectivity ({time.time() - t_neighbors:.2f} s)")
    print()

    # Build hash buckets for heavy blocks
    print("Building hash buckets...")
    t_hash = time.time()
    hash_bucket_data = {}
    if classification.heavy_blocks:
        element_centroids = np.mean(node_positions[connectivity], axis=1).astype(np.float32)

        for block_id in classification.heavy_blocks:
            block_elems = padded_arrays.block_elements[block_id]
            block_count = int(padded_arrays.block_sizes[block_id])
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

    print(f"✓ Built hash buckets ({time.time() - t_hash:.2f} s)")
    print(f"  Heavy blocks processed: {len(hash_bucket_data)}")
    print()

    # ========================================================================
    # Upload Mesh to GPU
    # ========================================================================
    print("-" * 80)
    print("UPLOADING MESH TO GPU")
    print("-" * 80)
    print()

    t_upload = time.time()
    mesh_gpu = upload_mesh_to_gpu(
        connectivity,
        node_positions,
        element_neighbors,
        verbose=True
    )
    print(f"✓ Uploaded mesh to GPU ({time.time() - t_upload:.2f} s)")
    print()

    return {
        'node_positions': node_positions,
        'connectivity': connectivity,
        'velocity_field': velocity_field,
        'element_neighbors': element_neighbors,
        'bbox': bbox,
        'grid_size': grid_size,
        'element_centroids': element_centroids,
        # Octree structures
        'octree_metadata_gpu': octree_metadata_gpu,
        'octree_elements_gpu': octree_elements_gpu,
        'octree_max_depth': octree_max_depth,
        # Blockwise structures
        'blocks': blocks,
        'padded_arrays': padded_arrays,
        'classification': classification,
        'block_neighbors_26': block_neighbors_26,
        'hash_bucket_data': hash_bucket_data,
        # GPU mesh
        'mesh_gpu': mesh_gpu
    }


def generate_particles_at_centroids(
    element_centroids: np.ndarray,
    connectivity: np.ndarray,
    node_positions: np.ndarray,
    n_particles: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate particles at element centroids with small random perturbations.

    Returns:
    - particle_positions: (n_particles, 3) array
    - true_element_ids: (n_particles,) array (ground truth)
    """
    print("=" * 80)
    print(f"GENERATING {n_particles:,} TEST PARTICLES")
    print("=" * 80)
    print()

    n_elements = len(element_centroids)

    # Randomly select elements
    np.random.seed(42)  # For reproducibility
    selected_elements = np.random.choice(n_elements, size=n_particles, replace=True)

    # Compute minimum element size (for perturbation scaling)
    print("Computing minimum element size...")
    element_sizes = []
    for i in range(min(1000, n_elements)):  # Sample 1000 elements for speed
        tet_nodes = node_positions[connectivity[i]]
        # Compute characteristic size as minimum edge length
        edges = [
            np.linalg.norm(tet_nodes[1] - tet_nodes[0]),
            np.linalg.norm(tet_nodes[2] - tet_nodes[0]),
            np.linalg.norm(tet_nodes[3] - tet_nodes[0]),
            np.linalg.norm(tet_nodes[2] - tet_nodes[1]),
            np.linalg.norm(tet_nodes[3] - tet_nodes[1]),
            np.linalg.norm(tet_nodes[3] - tet_nodes[2])
        ]
        element_sizes.append(min(edges))

    min_element_size = np.min(element_sizes)
    perturbation_scale = 0.01 * min_element_size  # 1% of minimum element size

    print(f"✓ Minimum element size: {min_element_size:.6e}")
    print(f"✓ Perturbation scale: {perturbation_scale:.6e} (1% of min size)")
    print()

    # Generate particles at centroids with perturbations
    print("Placing particles at element centroids with perturbations...")
    particle_positions = []
    true_element_ids = []

    for elem_id in selected_elements:
        centroid = element_centroids[elem_id]

        # Add small random perturbation (uniform in [-scale, +scale])
        perturbation = np.random.uniform(-perturbation_scale, perturbation_scale, size=3)
        position = centroid + perturbation

        particle_positions.append(position)
        true_element_ids.append(elem_id)

    particle_positions = np.array(particle_positions, dtype=np.float32)
    true_element_ids = np.array(true_element_ids, dtype=np.int32)

    print(f"✓ Generated {len(particle_positions):,} particles")
    print(f"  Position shape: {particle_positions.shape}")
    print(f"  True element IDs shape: {true_element_ids.shape}")
    print()

    return particle_positions, true_element_ids


def test_octree_search(
    particle_positions: np.ndarray,
    mesh_gpu,
    octree_metadata_gpu,
    octree_elements_gpu,
    octree_max_depth: int,
    true_element_ids: np.ndarray
) -> Dict:
    """
    Test octree search and measure performance.

    Returns dict with:
    - found_element_ids: array of found element IDs
    - n_found: number of particles found
    - accuracy: fraction matching true element IDs
    - time_search: search time in seconds
    - throughput: particles/second
    """
    print("=" * 80)
    print("TESTING OCTREE SEARCH")
    print("=" * 80)
    print()

    n_particles = len(particle_positions)

    # Upload particle positions to GPU
    print("Uploading particle positions to GPU...")
    positions_gpu = jax.device_put(particle_positions)

    # Initialize cached element IDs to -1 (all particles need search)
    cached_element_ids = jnp.full(n_particles, -1, dtype=jnp.int32)

    print(f"✓ Uploaded {n_particles:,} particle positions")
    print()

    # Warm-up JIT compilation
    print("Warming up JIT compilation...")
    _ = search_level2_octree_scan(
        positions_gpu[:10],
        cached_element_ids[:10],
        octree_metadata_gpu,
        octree_elements_gpu,
        mesh_gpu.node_positions,
        mesh_gpu.connectivity,
        max_depth=octree_max_depth
    )
    jax.block_until_ready(_)
    print("✓ JIT compilation complete")
    print()

    # Run octree search (timed)
    print("Running octree search...")
    t_start = time.perf_counter()

    found_element_ids_gpu = search_level2_octree_scan(
        positions_gpu,
        cached_element_ids,
        octree_metadata_gpu,
        octree_elements_gpu,
        mesh_gpu.node_positions,
        mesh_gpu.connectivity,
        max_depth=octree_max_depth
    )

    # Wait for GPU to complete
    jax.block_until_ready(found_element_ids_gpu)
    t_search = time.perf_counter() - t_start

    # Download results
    found_element_ids = np.array(found_element_ids_gpu)

    print(f"✓ Octree search complete")
    print(f"  Time: {t_search:.4f} s")
    print(f"  Throughput: {n_particles/t_search:,.1f} p/s")
    print()

    # Analyze results
    found_mask = found_element_ids >= 0
    n_found = found_mask.sum()

    # Compute accuracy (fraction matching true element IDs)
    matches = (found_element_ids[found_mask] == true_element_ids[found_mask])
    n_correct = matches.sum()
    accuracy = n_correct / n_particles if n_particles > 0 else 0.0

    print(f"OCTREE RESULTS:")
    print(f"  Found: {n_found}/{n_particles} ({100*n_found/n_particles:.2f}%)")
    print(f"  Correct: {n_correct}/{n_particles} ({100*accuracy:.2f}%)")
    print(f"  Mismatches: {n_found - n_correct}")
    print()

    return {
        'found_element_ids': found_element_ids,
        'n_found': n_found,
        'n_correct': n_correct,
        'accuracy': accuracy,
        'time_search': t_search,
        'throughput': n_particles / t_search
    }


def test_blockwise_search(
    particle_positions: np.ndarray,
    bbox: list,
    grid_size: Tuple[int, int, int],
    classification,
    padded_arrays,
    block_neighbors_26,
    hash_bucket_data,
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    true_element_ids: np.ndarray
) -> Dict:
    """
    Test blockwise search and measure performance.

    Returns dict with:
    - found_element_ids: array of found element IDs
    - n_found: number of particles found
    - accuracy: fraction matching true element IDs
    - time_search: search time in seconds
    - throughput: particles/second
    - search_stats: detailed search statistics
    """
    print("=" * 80)
    print("TESTING BLOCKWISE SEARCH")
    print("=" * 80)
    print()

    n_particles = len(particle_positions)

    # Run blockwise search (timed)
    print("Running blockwise search...")
    t_start = time.perf_counter()

    found_element_ids, block_ids, search_stats = initial_search_batch(
        particle_positions,
        bbox,
        grid_size,
        classification,
        padded_arrays,
        block_neighbors_26,
        hash_bucket_data,
        node_positions,
        connectivity,
        verbose=False
    )

    t_search = time.perf_counter() - t_start

    print(f"✓ Blockwise search complete")
    print(f"  Time: {t_search:.4f} s")
    print(f"  Throughput: {n_particles/t_search:,.1f} p/s")
    print()

    # Analyze results
    found_mask = found_element_ids >= 0
    n_found = found_mask.sum()

    # Compute accuracy (fraction matching true element IDs)
    matches = (found_element_ids[found_mask] == true_element_ids[found_mask])
    n_correct = matches.sum()
    accuracy = n_correct / n_particles if n_particles > 0 else 0.0

    print(f"BLOCKWISE RESULTS:")
    print(f"  Found: {n_found}/{n_particles} ({100*n_found/n_particles:.2f}%)")
    print(f"  Correct: {n_correct}/{n_particles} ({100*accuracy:.2f}%)")
    print(f"  Mismatches: {n_found - n_correct}")
    print()

    print(f"SEARCH STATISTICS:")
    for key, value in search_stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
    print()

    return {
        'found_element_ids': found_element_ids,
        'block_ids': block_ids,
        'n_found': n_found,
        'n_correct': n_correct,
        'accuracy': accuracy,
        'time_search': t_search,
        'throughput': n_particles / t_search,
        'search_stats': search_stats
    }


def main():
    """Main test execution."""

    # Configuration
    MESH_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule/threadedAvtk_120.pvtu")
    GRID_SIZE = (8, 8, 4)  # 256 blocks (same as production) - 40×40×40 causes OOM!
    OCTREE_MAX_DEPTH = 15
    OCTREE_MAX_LEAF_SIZE = 50
    OCTREE_LEVEL_THRESHOLD = 1.1
    N_PARTICLES = 50000  # Test with 50k particles

    print("\n" + "=" * 80)
    print("OCTREE vs BLOCKWISE INITIALIZATION COMPARISON TEST")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Mesh: {MESH_PATH.name}")
    print(f"  Grid size: {GRID_SIZE}")
    print(f"  Octree max depth: {OCTREE_MAX_DEPTH}")
    print(f"  Octree max leaf size: {OCTREE_MAX_LEAF_SIZE}")
    print(f"  Octree level threshold: {OCTREE_LEVEL_THRESHOLD}")
    print(f"  Number of particles: {N_PARTICLES:,}")
    print()

    # Load mesh and initialize structures
    data = load_mesh_and_initialize_structures(
        MESH_PATH,
        grid_size=GRID_SIZE,
        octree_max_depth=OCTREE_MAX_DEPTH,
        octree_max_leaf_size=OCTREE_MAX_LEAF_SIZE,
        octree_level_threshold=OCTREE_LEVEL_THRESHOLD
    )

    # Generate test particles
    particle_positions, true_element_ids = generate_particles_at_centroids(
        data['element_centroids'],
        data['connectivity'],
        data['node_positions'],
        n_particles=N_PARTICLES
    )

    # Test octree search
    octree_results = test_octree_search(
        particle_positions,
        data['mesh_gpu'],
        data['octree_metadata_gpu'],
        data['octree_elements_gpu'],
        data['octree_max_depth'],
        true_element_ids
    )

    # Test blockwise search
    blockwise_results = test_blockwise_search(
        particle_positions,
        data['bbox'],
        data['grid_size'],
        data['classification'],
        data['padded_arrays'],
        data['block_neighbors_26'],
        data['hash_bucket_data'],
        data['node_positions'],
        data['connectivity'],
        true_element_ids
    )

    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL COMPARISON: OCTREE vs BLOCKWISE")
    print("=" * 80)
    print()

    print(f"{'Method':<20} {'Found':<15} {'Accuracy':<15} {'Time (s)':<12} {'Throughput (p/s)':<20}")
    print("-" * 82)

    octree_found_pct = 100 * octree_results['n_found'] / N_PARTICLES
    octree_accuracy_pct = 100 * octree_results['accuracy']
    blockwise_found_pct = 100 * blockwise_results['n_found'] / N_PARTICLES
    blockwise_accuracy_pct = 100 * blockwise_results['accuracy']

    print(f"{'Octree':<20} {f'{octree_found_pct:.2f}%':<15} {f'{octree_accuracy_pct:.2f}%':<15} "
          f"{octree_results['time_search']:<12.4f} {octree_results['throughput']:<20,.1f}")

    print(f"{'Blockwise':<20} {f'{blockwise_found_pct:.2f}%':<15} {f'{blockwise_accuracy_pct:.2f}%':<15} "
          f"{blockwise_results['time_search']:<12.4f} {blockwise_results['throughput']:<20,.1f}")

    print()

    # Speedup comparison
    speedup = blockwise_results['time_search'] / octree_results['time_search']
    if speedup > 1.0:
        print(f"✓ Octree is {speedup:.2f}× FASTER than blockwise")
    else:
        print(f"✓ Blockwise is {1/speedup:.2f}× FASTER than octree")
    print()

    # Accuracy comparison
    if octree_results['accuracy'] > blockwise_results['accuracy']:
        acc_diff = octree_results['accuracy'] - blockwise_results['accuracy']
        print(f"✓ Octree is more accurate by {100*acc_diff:.2f}%")
    elif blockwise_results['accuracy'] > octree_results['accuracy']:
        acc_diff = blockwise_results['accuracy'] - octree_results['accuracy']
        print(f"✓ Blockwise is more accurate by {100*acc_diff:.2f}%")
    else:
        print(f"✓ Both methods have identical accuracy")
    print()

    # Recommendations
    print("RECOMMENDATIONS:")
    if octree_results['accuracy'] >= 0.999 and blockwise_results['accuracy'] >= 0.999:
        print("✓ Both methods achieve >99.9% accuracy - suitable for production")
        if speedup > 1.2:
            print(f"✓ Recommend OCTREE for initial assignment (faster by {speedup:.2f}×)")
        elif speedup < 0.8:
            print(f"✓ Recommend BLOCKWISE for initial assignment (faster by {1/speedup:.2f}×)")
        else:
            print("✓ Both methods have similar performance - either is suitable")
    else:
        if octree_results['accuracy'] > blockwise_results['accuracy']:
            print(f"⚠ Octree has better accuracy ({100*octree_results['accuracy']:.2f}% vs {100*blockwise_results['accuracy']:.2f}%)")
        else:
            print(f"⚠ Blockwise has better accuracy ({100*blockwise_results['accuracy']:.2f}% vs {100*octree_results['accuracy']:.2f}%)")
    print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
