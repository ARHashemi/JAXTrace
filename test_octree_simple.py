#!/usr/bin/env python3
"""
Simple octree correctness test - copies initialization from production script
"""
import numpy as np
import jax
import jax.numpy as jnp
import time
from pathlib import Path

# Copy exact imports from production script lines 58-72
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
from jaxtrace.gpu.forest.block_grid import create_regular_grid
from jaxtrace.gpu.forest.block_mapper import assign_elements_to_blocks
from jaxtrace.gpu.forest.padded_arrays import build_padded_block_arrays
from jaxtrace.gpu.forest import build_element_neighbors_array
from jaxtrace.gpu.search import classify_blocks
from jaxtrace.gpu.search.initial_assignment import initial_search_batch
from jaxtrace.gpu.search.hash_bucket import build_hash_bucket_arrays
from jaxtrace.tracking.seeding import uniform_grid_seeds
from jaxtrace.gpu.search.octree_search_gpu import search_level2_octree_scan
from jaxtrace.gpu.tracking.mesh_data_gpu import MeshDataGPU

# Configuration - copy from production
MESH_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule")
GRID_SIZE = (16, 16, 16)
N_PARTICLES = 10_000

print("="*80)
print("SIMPLE OCTREE CORRECTNESS TEST")
print("="*80)
print()

# Step 1: Load mesh (copy from production lines 269-280)
print("Loading mesh...")
t0 = time.perf_counter()
node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
    MESH_PATH,
    field_name='Displacement'
)
t_load = time.perf_counter() - t0

if velocity_field.shape[1] == 2:
    velocity_field = np.column_stack([velocity_field, np.zeros(velocity_field.shape[0])])
velocity_field = velocity_field.astype(np.float32)

print(f"✓ Mesh loaded ({t_load:.2f} s)")
print(f"  Elements: {len(connectivity):,}")
print(f"  Nodes: {len(node_positions):,}")
print()

# Step 2: Compute bbox (copy from production lines 392-397)
bbox = np.array([
    node_positions[:, 0].min(), node_positions[:, 0].max(),
    node_positions[:, 1].min(), node_positions[:, 1].max(),
    node_positions[:, 2].min(), node_positions[:, 2].max(),
], dtype=np.float32)

print(f"Bounding box:")
print(f"  X: [{bbox[0]:.4f}, {bbox[1]:.4f}]")
print(f"  Y: [{bbox[2]:.4f}, {bbox[3]:.4f}]")
print(f"  Z: [{bbox[4]:.4f}, {bbox[5]:.4f}]")
print()

# Step 3: Create blocks (copy from production lines 405-409)
print("Creating block grid...")
t0 = time.perf_counter()
blocks = create_regular_grid(bbox, GRID_SIZE)
t_grid = time.perf_counter() - t0
print(f"✓ Block grid created ({t_grid:.2f} s): {len(blocks)} blocks")

# Step 4: Assign elements to blocks (copy from production lines 411-423)
print("Assigning elements to blocks...")
t0 = time.perf_counter()
element_to_block, stats = assign_elements_to_blocks(
    node_positions,
    connectivity,
    bbox,
    GRID_SIZE,
    verbose=False
)
t_assign = time.perf_counter() - t0
print(f"✓ Element assignment ({t_assign:.2f} s)")
print(f"  Blocks used: {stats.n_blocks_used}/{stats.n_blocks}")
print(f"  Elements per block: {stats.min_elements} - {stats.max_elements}")

# Step 5: Build element neighbors (copy from production lines 425-429)
print("Building element neighbors...")
t0 = time.perf_counter()
element_neighbors = build_element_neighbors_array(connectivity, verbose=False)
t_neighbors = time.perf_counter() - t0
print(f"✓ Element neighbors built ({t_neighbors:.2f} s)")
print()

# Step 6: Build padded arrays (copy from production lines 534-549)
print("Building padded arrays...")
t0 = time.perf_counter()
padded_arrays = build_padded_block_arrays(
    element_to_block,
    stats,
    node_positions=node_positions,
    connectivity=connectivity,
    element_neighbors=element_neighbors,
    verbose=False
)
t_padded = time.perf_counter() - t0
print(f"✓ Padded arrays ({t_padded:.2f} s)")
print(f"  Shape: {padded_arrays.block_elements.shape}")
print(f"  Memory: {padded_arrays.memory_mb:.1f} MB")
print()

# Step 7: Classify blocks (copy from production lines 552-556)
classification = classify_blocks(padded_arrays, threshold=10000, verbose=False)
print(f"✓ Block classification:")
print(f"  Light blocks: {len(classification.light_blocks)}")
print(f"  Heavy blocks: {len(classification.heavy_blocks)}")
print()

# Step 8: Build hash buckets (copy from production lines 558-585)
hash_bucket_data = {}
if classification.heavy_blocks:
    print(f"Building hash buckets for {len(classification.heavy_blocks)} heavy blocks...")
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

# Step 9: Block neighbors (copy from production line 587)
block_neighbors_26 = np.array([b.neighbors_26 for b in blocks], dtype=np.int32)

# Step 10: Generate particles (copy from production lines 666-690)
print()
print("Generating particles...")
domain_min = np.array([bbox[0], bbox[2], bbox[4]], dtype=np.float32)
domain_max = np.array([bbox[1], bbox[3], bbox[5]], dtype=np.float32)
domain_size = domain_max - domain_min

par_bounds_min = np.zeros(3, dtype=np.float32)
par_bounds_max = np.zeros(3, dtype=np.float32)

PARTICLE_BOUNDS_FRACTION = {'x': (0.1, 0.9), 'y': (0.1, 0.9), 'z': (0.1, 0.9)}
for i, axis in enumerate(['x', 'y', 'z']):
    min_frac, max_frac = PARTICLE_BOUNDS_FRACTION[axis]
    par_bounds_min[i] = domain_min[i] + min_frac * domain_size[i]
    par_bounds_max[i] = domain_min[i] + max_frac * domain_size[i]

par_bounds = [par_bounds_min, par_bounds_max]

particle_positions = uniform_grid_seeds(
    bounds=par_bounds,
    count=N_PARTICLES
)

print(f"✓ Generated {len(particle_positions):,} particles")
print()

# Step 11: Initial assignment (copy from production lines 711-739)
print("Finding containing elements...")
t0 = time.perf_counter()

element_ids, block_ids, search_stats = initial_search_batch(
    particle_positions,
    bbox,
    GRID_SIZE,
    classification,
    padded_arrays,
    block_neighbors_26,
    hash_bucket_data,
    node_positions,
    connectivity,
    verbose=False
)

t_search = time.perf_counter() - t0

found_mask = element_ids >= 0
n_found = found_mask.sum()

print(f"✓ Initial assignment ({t_search:.2f} s):")
print(f"  Found: {n_found}/{N_PARTICLES} ({100*n_found/N_PARTICLES:.1f}%)")
print()

# ============================================================================
# Step 12: Build Octree (copy from production lines 474-524)
# ============================================================================
print("="*80)
print("OCTREE BUILD")
print("="*80)
print()

# Load LEVEL field for octree filtering
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu
import vtk
from vtk.util import numpy_support
from pathlib import Path

pvtu_files = list(MESH_PATH.glob("*.pvtu"))
if not pvtu_files:
    print("ERROR: No .pvtu file found")
    exit(1)

reader = vtk.vtkXMLPUnstructuredGridReader()
reader.SetFileName(str(pvtu_files[0]))
reader.Update()
mesh_vtk = reader.GetOutput()
cell_data = mesh_vtk.GetCellData()
point_data = mesh_vtk.GetPointData()

# Extract LEVEL field (copy from production)
level_field = None
if cell_data.HasArray('LEVEL'):
    level_field = numpy_support.vtk_to_numpy(cell_data.GetArray('LEVEL')).astype(np.float32)
    print(f"✓ LEVEL field found (cell data)")
elif point_data.HasArray('LEVEL'):
    node_level = numpy_support.vtk_to_numpy(point_data.GetArray('LEVEL')).astype(np.float32)
    level_field = np.array([
        node_level[connectivity[i]].max()
        for i in range(len(connectivity))
    ], dtype=np.float32)
    print(f"✓ LEVEL field found (point data, converted to cell data)")

if level_field is not None:
    print(f"  LEVEL range: [{level_field.min():.6f}, {level_field.max():.6f}]")
    print()

    # Compute element centroids
    element_centroids = np.array([
        node_positions[connectivity[i]].mean(axis=0)
        for i in range(len(connectivity))
    ], dtype=np.float32)

    element_ids_all = np.arange(len(connectivity), dtype=np.int32)

    # Build octree
    from jaxtrace.gpu.search.octree_builder import build_octree_for_level, flatten_octree_to_arrays

    OCTREE_LEVELSET_THRESHOLD = 0.012
    OCTREE_MAX_DEPTH = 10
    OCTREE_MAX_LEAF_SIZE = 200

    print(f"Building octree (levelset < {OCTREE_LEVELSET_THRESHOLD})...")
    t0 = time.perf_counter()

    nodes, metadata = build_octree_for_level(
        element_centroids,
        element_ids_all,
        level_field=level_field,
        level_threshold=OCTREE_LEVELSET_THRESHOLD,
        max_depth=OCTREE_MAX_DEPTH,
        max_leaf_size=OCTREE_MAX_LEAF_SIZE,
        use_levelset=True
    )

    t_octree_build = time.perf_counter() - t0

    print(f"✓ Octree built ({t_octree_build:.2f} s)")
    print(f"  Filtered elements: {metadata['n_elements']:,}/{len(connectivity):,} ({metadata['n_elements']/len(connectivity)*100:.1f}%)")
    print(f"  Total nodes: {metadata['n_nodes']:,}")
    print(f"  Leaf nodes: {metadata['n_leaves']:,}")
    print(f"  Max depth: {metadata['max_depth']}")
    print()

    # Flatten to GPU arrays
    print("Flattening octree to fixed-size arrays...")
    node_metadata_np, node_elements_np = flatten_octree_to_arrays(nodes, max_leaf_size=OCTREE_MAX_LEAF_SIZE)

    print(f"  Metadata array: {node_metadata_np.shape} ({node_metadata_np.nbytes / (1024**2):.2f} MB)")
    print(f"  Elements array: {node_elements_np.shape} ({node_elements_np.nbytes / (1024**2):.2f} MB)")
    print()

    # Upload to GPU
    print("Uploading octree to GPU...")
    octree_metadata_gpu = jax.device_put(node_metadata_np)
    octree_elements_gpu = jax.device_put(node_elements_np)

    print(f"✓ Octree uploaded to GPU")
    print(f"  Total octree memory: {(node_metadata_np.nbytes + node_elements_np.nbytes) / (1024**2):.2f} MB")
    print()

    # ============================================================================
    # Step 13: Perturb Particle Positions (0.001 perturbation)
    # ============================================================================
    print("="*80)
    print("PARTICLE PERTURBATION")
    print("="*80)
    print()

    # Only perturb found particles
    positions_found = particle_positions[found_mask]
    element_ids_found = element_ids[found_mask]

    print(f"Applying 0.001 perturbation to {len(positions_found):,} found particles...")
    perturbation = np.random.uniform(-0.001, 0.001, positions_found.shape).astype(np.float32)
    positions_perturbed = positions_found + perturbation

    print(f"✓ Perturbation applied")
    print(f"  Mean displacement: {np.linalg.norm(perturbation, axis=1).mean():.6f}")
    print(f"  Max displacement: {np.linalg.norm(perturbation, axis=1).max():.6f}")
    print()

    # ============================================================================
    # Step 14: Octree Search - Force ALL particles through L2
    # ============================================================================
    print("="*80)
    print("OCTREE SEARCH (FORCED L2)")
    print("="*80)
    print()

    # Upload to GPU
    positions_gpu = jax.device_put(positions_perturbed)
    cached_ids_gpu = jnp.full(len(positions_perturbed), -1, dtype=jnp.int32)  # Force all through octree

    # Upload mesh to GPU
    mesh_gpu = MeshDataGPU(
        connectivity=jax.device_put(connectivity),
        node_positions=jax.device_put(node_positions)
    )

    print(f"Running octree search on {len(positions_perturbed):,} particles (forced L2)...")
    print(f"  Cached IDs: ALL -1 (forces octree for all particles)")
    print()

    # Record GPU memory before
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True)
    gpu_mem_before = int(result.stdout.strip().split('\n')[0])

    # Run octree search with timing
    t0 = time.perf_counter()

    element_ids_octree = search_level2_octree_scan(
        positions_gpu,
        cached_ids_gpu,
        octree_metadata_gpu,
        octree_elements_gpu,
        mesh_gpu.node_positions,
        mesh_gpu.connectivity,
        max_depth=OCTREE_MAX_DEPTH
    )

    element_ids_octree.block_until_ready()  # Ensure GPU completes
    t_octree_search = time.perf_counter() - t0

    # Record GPU memory after
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True)
    gpu_mem_after = int(result.stdout.strip().split('\n')[0])

    # Download results
    element_ids_octree_np = np.array(element_ids_octree)

    # Statistics
    found_octree_mask = element_ids_octree_np >= 0
    n_found_octree = found_octree_mask.sum()

    print(f"✓ Octree search complete ({t_octree_search:.4f} s)")
    print(f"  Throughput: {len(positions_perturbed) / t_octree_search:,.0f} particles/s")
    print(f"  Time per particle: {t_octree_search / len(positions_perturbed) * 1e6:.2f} μs")
    print(f"  Found: {n_found_octree}/{len(positions_perturbed)} ({100*n_found_octree/len(positions_perturbed):.1f}%)")
    print(f"  GPU memory: {gpu_mem_before} MB → {gpu_mem_after} MB (delta: {gpu_mem_after - gpu_mem_before} MB)")
    print()

    # ============================================================================
    # Step 15: Validation - Random Sampling with Point-in-Tet
    # ============================================================================
    print("="*80)
    print("VALIDATION (RANDOM SAMPLING)")
    print("="*80)
    print()

    from jaxtrace.gpu.search.level0_cached import point_in_tet_jax

    N_VALIDATION_SAMPLES = 100

    # Sample only particles that octree found
    valid_indices = np.where(found_octree_mask)[0]
    if len(valid_indices) > N_VALIDATION_SAMPLES:
        sample_indices = np.random.choice(valid_indices, N_VALIDATION_SAMPLES, replace=False)
    else:
        sample_indices = valid_indices

    print(f"Validating {len(sample_indices)} random particles with point-in-tet...")

    n_correct = 0
    n_incorrect = 0
    errors = []

    for i in sample_indices:
        pos = positions_perturbed[i]
        predicted_elem = element_ids_octree_np[i]

        if predicted_elem < 0:
            continue

        # Get tetrahedron nodes
        tet_nodes = node_positions[connectivity[predicted_elem]]

        # Test if point is inside
        pos_gpu = jax.device_put(pos)
        tet_gpu = jax.device_put(tet_nodes)
        is_inside = point_in_tet_jax(pos_gpu, tet_gpu)
        is_inside_cpu = bool(np.array(is_inside))

        if is_inside_cpu:
            n_correct += 1
        else:
            n_incorrect += 1
            errors.append({
                'particle_idx': i,
                'position': pos,
                'predicted_elem': predicted_elem,
                'distance_to_centroid': np.linalg.norm(pos - element_centroids[predicted_elem])
            })

    print(f"✓ Validation complete")
    print(f"  Correct: {n_correct}/{len(sample_indices)} ({100*n_correct/len(sample_indices):.1f}%)")
    print(f"  Incorrect: {n_incorrect}/{len(sample_indices)} ({100*n_incorrect/len(sample_indices):.1f}%)")

    if n_incorrect > 0:
        print(f"\n  Top 5 errors:")
        for err in errors[:5]:
            print(f"    Particle {err['particle_idx']}: elem {err['predicted_elem']}, "
                  f"dist to centroid: {err['distance_to_centroid']:.6f}")
    print()

    # ============================================================================
    # Step 16: Performance Summary
    # ============================================================================
    print("="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print()

    print(f"Initialization:")
    print(f"  Mesh loading: {t_load:.2f} s")
    print(f"  Block grid: {t_grid:.2f} s")
    print(f"  Element assignment: {t_assign:.2f} s")
    print(f"  Element neighbors: {t_neighbors:.2f} s")
    print(f"  Padded arrays: {t_padded:.2f} s")
    print(f"  Initial search: {t_search:.2f} s")
    print(f"  Octree build: {t_octree_build:.2f} s")
    print()

    print(f"Octree Search (Pure L2):")
    print(f"  Time: {t_octree_search:.4f} s")
    print(f"  Throughput: {len(positions_perturbed) / t_octree_search:,.0f} particles/s")
    print(f"  Success rate: {100*n_found_octree/len(positions_perturbed):.1f}%")
    print(f"  Validation accuracy: {100*n_correct/len(sample_indices):.1f}%")
    print()

    print(f"Memory:")
    print(f"  Padded arrays: {padded_arrays.memory_mb:.1f} MB")
    print(f"  Octree: {(node_metadata_np.nbytes + node_elements_np.nbytes) / (1024**2):.2f} MB")
    print(f"  GPU delta: {gpu_mem_after - gpu_mem_before} MB")
    print()

else:
    print("ERROR: LEVEL field not found, cannot build octree")
    print()

print("="*80)
print("TEST COMPLETE")
print("="*80)
