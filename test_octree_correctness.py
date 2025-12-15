"""
Octree Search Correctness and Performance Test

This script tests the octree L2 search implementation by:
1. Loading real mesh (ThreadedA)
2. Building filtered octree
3. Initializing 10,000 particles
4. Validating initialization (random sampling)
5. Perturbing particle positions (0.001 perturbation)
6. Running octree search on ALL particles
7. Validating octree search results
8. Recording GPU/CPU performance, memory, and timing
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("OCTREE CORRECTNESS AND PERFORMANCE TEST")
print("=" * 80)
print()

# ============================================================================
# Configuration
# ============================================================================
N_PARTICLES = 10_000
PERTURBATION = 0.001  # Small perturbation in mesh units
N_VALIDATION_SAMPLES = 100  # Random particles to validate

# Mesh file - use PVTU to get LEVEL field
MESH_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule")

# Octree configuration
OCTREE_LEVELSET_THRESHOLD = 0.012
OCTREE_MAX_DEPTH = 10

print(f"Configuration:")
print(f"  Mesh: {MESH_PATH}")
print(f"  Particles: {N_PARTICLES:,}")
print(f"  Perturbation: {PERTURBATION}")
print(f"  Validation samples: {N_VALIDATION_SAMPLES}")
print(f"  Octree threshold: {OCTREE_LEVELSET_THRESHOLD}")
print(f"  Octree max depth: {OCTREE_MAX_DEPTH}")
print()

# ============================================================================
# Step 1: Load Mesh
# ============================================================================
print("=" * 80)
print("STEP 1: LOAD MESH")
print("=" * 80)
print()

from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu

t0 = time.perf_counter()
node_positions, connectivity, velocity_field = load_mesh_from_pvtu(
    MESH_PATH,  # Directory containing .pvtu
    field_name='Displacement'
)
t_load = time.perf_counter() - t0

# Ensure velocity is 3D and float32
if velocity_field.shape[1] == 2:
    velocity_field = np.column_stack([velocity_field, np.zeros(velocity_field.shape[0])])
velocity_field = velocity_field.astype(np.float32)

print(f"✓ Mesh loaded ({t_load:.2f} s)")
print(f"  Elements: {len(connectivity):,}")
print(f"  Nodes: {len(node_positions):,}")
print(f"  Memory: {(connectivity.nbytes + node_positions.nbytes + velocity_field.nbytes) / 1024**2:.1f} MB")
print()

# ============================================================================
# Step 2: Build Element Adjacency
# ============================================================================
print("=" * 80)
print("STEP 2: BUILD ELEMENT ADJACENCY")
print("=" * 80)
print()

from jaxtrace.gpu.forest.element_adjacency import build_element_neighbors_array

t0 = time.perf_counter()
element_neighbors = build_element_neighbors_array(connectivity)
t_neighbors = time.perf_counter() - t0

print(f"✓ Element neighbors built ({t_neighbors:.2f} s)")
print(f"  Memory: {element_neighbors.nbytes / 1024**2:.1f} MB")
print()

# ============================================================================
# Step 3: Compute Element Centroids and Level Field
# ============================================================================
print("=" * 80)
print("STEP 3: COMPUTE ELEMENT CENTROIDS AND LEVEL FIELD")
print("=" * 80)
print()

# Compute centroids
t0 = time.perf_counter()
element_centroids = np.zeros((len(connectivity), 3), dtype=np.float32)
for i, elem in enumerate(connectivity):
    element_centroids[i] = node_positions[elem].mean(axis=0)
t_centroids = time.perf_counter() - t0

print(f"✓ Element centroids computed ({t_centroids:.2f} s)")
print(f"  Memory: {element_centroids.nbytes / 1024**2:.1f} MB")
print()

# Extract level field from PVTU
print("Extracting LEVEL field from mesh...")

import vtk
from vtk.util import numpy_support

# Find PVTU file
pvtu_files = list(MESH_PATH.glob("*.pvtu"))
if not pvtu_files:
    raise ValueError(f"No PVTU file found in {MESH_PATH}")

pvtu_file = pvtu_files[0]
print(f"  Loading from: {pvtu_file.name}")

reader = vtk.vtkXMLPUnstructuredGridReader()
reader.SetFileName(str(pvtu_file))
reader.Update()
vtk_mesh = reader.GetOutput()

# Check both cell data and point data for LEVEL
cell_data = vtk_mesh.GetCellData()
point_data = vtk_mesh.GetPointData()

level_field = None

if cell_data.HasArray('LEVEL'):
    # LEVEL stored per element (ideal case)
    level_field = numpy_support.vtk_to_numpy(cell_data.GetArray('LEVEL')).astype(np.float32)
    print(f"✓ Found LEVEL in cell data: {len(level_field):,} elements")
elif point_data.HasArray('LEVEL'):
    # LEVEL stored per node - compute per-element by taking max of element's nodes
    print(f"✓ Found LEVEL in point data: {vtk_mesh.GetNumberOfPoints():,} nodes")
    node_level = numpy_support.vtk_to_numpy(point_data.GetArray('LEVEL')).astype(np.float32)
    print(f"  Node levelset range: [{node_level.min():.6f}, {node_level.max():.6f}]")

    print(f"  Computing per-element levelset (max of element's nodes)...")
    level_field = np.array([
        node_level[connectivity[i]].max()
        for i in range(len(connectivity))
    ], dtype=np.float32)
    print(f"✓ Computed element levelset")
else:
    raise ValueError("LEVEL field not found in mesh!")

print(f"  Level field range: [{level_field.min():.6f}, {level_field.max():.6f}]")
print(f"  Mean: {level_field.mean():.6f}, Std: {level_field.std():.6f}")
print()

# ============================================================================
# Step 4: Build Octree
# ============================================================================
print("=" * 80)
print("STEP 4: BUILD OCTREE")
print("=" * 80)
print()

from jaxtrace.gpu.search.octree_builder import build_octree_for_level, flatten_octree_to_arrays

t0 = time.perf_counter()

# Create element IDs
element_ids = np.arange(len(connectivity), dtype=np.int32)

# Build octree with levelset filtering
nodes, metadata = build_octree_for_level(
    element_centroids,
    element_ids,
    level_field=level_field,
    level_threshold=OCTREE_LEVELSET_THRESHOLD,
    max_depth=OCTREE_MAX_DEPTH,
    max_leaf_size=50,
    use_levelset=True
)

t_octree_build = time.perf_counter() - t0

print(f"✓ Octree built ({t_octree_build:.2f} s)")
print(f"  Filtered elements: {metadata['n_elements']:,}/{len(connectivity):,} ({metadata['n_elements']/len(connectivity)*100:.1f}%)")
print(f"  Total nodes: {metadata['n_nodes']:,}")
print(f"  Leaf nodes: {metadata['n_leaves']:,}")
print(f"  Max depth: {metadata['max_depth']}")

# Flatten to GPU-compatible arrays
print()
print("Flattening octree to fixed-size arrays...")
octree_metadata, octree_elements = flatten_octree_to_arrays(nodes, max_leaf_size=50)

print(f"  Metadata array: {octree_metadata.shape} ({octree_metadata.nbytes / 1024:.1f} KB)")
print(f"  Elements array: {octree_elements.shape} ({octree_elements.nbytes / 1024:.1f} KB)")
print(f"  Total memory: {(octree_metadata.nbytes + octree_elements.nbytes) / 1024**2:.1f} MB")
print()

# ============================================================================
# Step 5: Upload to GPU
# ============================================================================
print("=" * 80)
print("STEP 5: UPLOAD TO GPU")
print("=" * 80)
print()

from jaxtrace.gpu.tracking.mesh_data_gpu import upload_mesh_to_gpu

# Get initial GPU memory
gpu_mem_before = jax.devices()[0].memory_stats()['bytes_in_use'] / 1024**2

t0 = time.perf_counter()

# Upload mesh
mesh_gpu = upload_mesh_to_gpu(
    connectivity,
    node_positions,
    element_neighbors,
    verbose=True
)

# Upload octree
octree_metadata_gpu = jax.device_put(octree_metadata.astype(np.float32))
octree_elements_gpu = jax.device_put(octree_elements.astype(np.int32))

t_upload = time.perf_counter() - t0

gpu_mem_after = jax.devices()[0].memory_stats()['bytes_in_use'] / 1024**2
gpu_mem_used = gpu_mem_after - gpu_mem_before

print(f"✓ Data uploaded to GPU ({t_upload:.2f} s)")
print(f"  GPU memory used: {gpu_mem_used:.1f} MB")
print(f"  Total GPU memory: {gpu_mem_after:.1f} MB")
print()

# ============================================================================
# Step 6: Generate Particles
# ============================================================================
print("=" * 80)
print("STEP 6: GENERATE PARTICLES")
print("=" * 80)
print()

from jaxtrace.tracking.seeding import uniform_grid_seeds

# Compute bounding box from node positions
bbox = np.array([
    node_positions[:, 0].min(), node_positions[:, 0].max(),
    node_positions[:, 1].min(), node_positions[:, 1].max(),
    node_positions[:, 2].min(), node_positions[:, 2].max()
], dtype=np.float32)

# Compute domain bounds
domain_min = np.array([bbox[0], bbox[2], bbox[4]], dtype=np.float32)
domain_max = np.array([bbox[1], bbox[3], bbox[5]], dtype=np.float32)
domain_size = domain_max - domain_min

print(f"Domain bounds:")
print(f"  X: [{domain_min[0]:.4f}, {domain_max[0]:.4f}]")
print(f"  Y: [{domain_min[1]:.4f}, {domain_max[1]:.4f}]")
print(f"  Z: [{domain_min[2]:.4f}, {domain_max[2]:.4f}]")
print()

# Generate particles in central region (avoid boundaries)
PARTICLE_BOUNDS_FRACTION = {
    'x': (0.1, 0.9),
    'y': (0.1, 0.9),
    'z': (0.1, 0.9)
}

par_bounds_min = np.zeros(3, dtype=np.float32)
par_bounds_max = np.zeros(3, dtype=np.float32)

for i, axis in enumerate(['x', 'y', 'z']):
    min_frac, max_frac = PARTICLE_BOUNDS_FRACTION[axis]
    par_bounds_min[i] = domain_min[i] + min_frac * domain_size[i]
    par_bounds_max[i] = domain_min[i] + max_frac * domain_size[i]

par_bounds = [par_bounds_min, par_bounds_max]

# Compute grid resolution for N_PARTICLES
particles_per_dim = int(np.ceil(N_PARTICLES ** (1/3)))
grid_resolution = (particles_per_dim, particles_per_dim, particles_per_dim)

print(f"Generating {N_PARTICLES:,} particles...")
print(f"  Grid resolution: {grid_resolution}")

particle_positions = uniform_grid_seeds(
    resolution=grid_resolution,
    bounds=par_bounds,
    include_boundaries=True
)

# Trim to exactly N_PARTICLES
particle_positions = particle_positions[:N_PARTICLES]

print(f"✓ Generated {len(particle_positions):,} particles")
print()

# ============================================================================
# Step 7: Initial Particle Assignment
# ============================================================================
print("=" * 80)
print("STEP 7: INITIAL PARTICLE ASSIGNMENT")
print("=" * 80)
print()

from jaxtrace.gpu.search.initial_assignment import initial_search_batch
from jaxtrace.gpu.search import classify_blocks

print("Using block-based search for initialization (same as production)...")

# First classify blocks
print("Classifying blocks...")
t_classify = time.perf_counter()
classification = classify_blocks(
    node_positions,
    connectivity,
    bbox,
    (16, 16, 16),  # Same grid size as production
    verbose=False
)
t_classify = time.perf_counter() - t_classify
print(f"✓ Blocks classified ({t_classify:.2f} s)")
print()

t0 = time.perf_counter()

# Use production-quality block-based search
element_ids, block_ids, search_stats = initial_search_batch(
    particle_positions,
    bbox,
    (16, 16, 16),  # Grid size
    classification,
    batch_size=1024,
    verbose=False
)
element_ids = element_ids.astype(np.int32)

# Upload to GPU for octree test
positions_gpu = jax.device_put(particle_positions.astype(np.float32))
element_ids_gpu = jax.device_put(element_ids)

t_init = time.perf_counter() - t0

n_found = np.sum(element_ids >= 0)
n_lost = N_PARTICLES - n_found

print(f"✓ Initial assignment complete ({t_init:.2f} s)")
print(f"  Found: {n_found:,} ({n_found/N_PARTICLES*100:.1f}%)")
print(f"  Lost: {n_lost:,} ({n_lost/N_PARTICLES*100:.1f}%)")
print()

if n_found < N_PARTICLES * 0.95:
    print(f"⚠️  WARNING: Low initialization rate ({n_found/N_PARTICLES*100:.1f}%)")
    print(f"   This may indicate particles outside mesh domain")
    print()

# ============================================================================
# Step 8: Validate Initialization (Random Sampling)
# ============================================================================
print("=" * 80)
print("STEP 8: VALIDATE INITIALIZATION")
print("=" * 80)
print()

print(f"Validating {N_VALIDATION_SAMPLES} random particles...")

# Select random found particles
found_indices = np.where(element_ids >= 0)[0]
sample_indices = np.random.choice(found_indices, min(N_VALIDATION_SAMPLES, len(found_indices)), replace=False)

validation_errors = 0
for idx in sample_indices:
    pos = particle_positions[idx]
    elem_id = element_ids[idx]

    # Get element nodes
    node_ids = connectivity[elem_id]
    tet_nodes = node_positions[node_ids]

    # Check if point is actually inside
    from jaxtrace.gpu.search.level0_cached import point_in_tet
    inside = point_in_tet(pos, tet_nodes)

    if not inside:
        validation_errors += 1
        if validation_errors <= 5:  # Print first few errors
            print(f"  ❌ Particle {idx}: pos={pos}, elem={elem_id} - NOT INSIDE!")

if validation_errors == 0:
    print(f"✓ All {len(sample_indices)} sampled particles validated correctly")
else:
    print(f"❌ Validation errors: {validation_errors}/{len(sample_indices)} ({validation_errors/len(sample_indices)*100:.1f}%)")
print()

# ============================================================================
# Step 9: Perturb Particle Positions
# ============================================================================
print("=" * 80)
print("STEP 9: PERTURB PARTICLE POSITIONS")
print("=" * 80)
print()

print(f"Applying perturbation: {PERTURBATION:.6f}")

# Generate random perturbations
np.random.seed(42)
perturbations = np.random.randn(N_PARTICLES, 3).astype(np.float32) * PERTURBATION

# Apply perturbations
particle_positions_perturbed = particle_positions + perturbations

# Compute perturbation statistics
perturbation_norms = np.linalg.norm(perturbations, axis=1)

print(f"✓ Particles perturbed")
print(f"  Perturbation magnitude:")
print(f"    Mean: {perturbation_norms.mean():.6f}")
print(f"    Min: {perturbation_norms.min():.6f}")
print(f"    Max: {perturbation_norms.max():.6f}")
print()

# ============================================================================
# Step 10: Octree Search on ALL Particles
# ============================================================================
print("=" * 80)
print("STEP 10: OCTREE SEARCH ON ALL PARTICLES")
print("=" * 80)
print()

from jaxtrace.gpu.search.octree_search_gpu import search_level2_octree_scan

# Upload perturbed positions to GPU
positions_perturbed_gpu = jax.device_put(particle_positions_perturbed.astype(np.float32))

# Create dummy cached IDs (all -1 to force octree search for ALL particles)
cached_ids_gpu = jnp.full(N_PARTICLES, -1, dtype=jnp.int32)

print(f"Running octree search on ALL {N_PARTICLES:,} particles...")
print(f"  Note: Setting all cached_ids = -1 to force octree search")
print()

# Warm-up JIT compilation
print("JIT compilation warm-up...")
t0 = time.perf_counter()

_ = search_level2_octree_scan(
    positions_perturbed_gpu[:100],  # Small batch for warm-up
    cached_ids_gpu[:100],
    octree_metadata_gpu,
    octree_elements_gpu,
    mesh_gpu.node_positions,
    mesh_gpu.connectivity,
    max_depth=OCTREE_MAX_DEPTH
)
_.block_until_ready()

t_warmup = time.perf_counter() - t0
print(f"✓ Warm-up complete ({t_warmup:.2f} s)")
print()

# Actual octree search
print("Running full octree search...")

# Record GPU memory before search
gpu_mem_before_search = jax.devices()[0].memory_stats()['bytes_in_use'] / 1024**2

t0 = time.perf_counter()

element_ids_octree_gpu = search_level2_octree_scan(
    positions_perturbed_gpu,
    cached_ids_gpu,
    octree_metadata_gpu,
    octree_elements_gpu,
    mesh_gpu.node_positions,
    mesh_gpu.connectivity,
    max_depth=OCTREE_MAX_DEPTH
)
element_ids_octree_gpu.block_until_ready()

t_search = time.perf_counter() - t0

# Record GPU memory after search
gpu_mem_after_search = jax.devices()[0].memory_stats()['bytes_in_use'] / 1024**2

# Download results
element_ids_octree = np.array(element_ids_octree_gpu, dtype=np.int32)

n_found_octree = np.sum(element_ids_octree >= 0)
n_lost_octree = N_PARTICLES - n_found_octree

print(f"✓ Octree search complete ({t_search:.4f} s)")
print(f"  Throughput: {N_PARTICLES / t_search:.1f} particles/s")
print(f"  Time per particle: {t_search / N_PARTICLES * 1e6:.2f} μs")
print(f"  Found: {n_found_octree:,} ({n_found_octree/N_PARTICLES*100:.1f}%)")
print(f"  Lost: {n_lost_octree:,} ({n_lost_octree/N_PARTICLES*100:.1f}%)")
print(f"  GPU memory used: {gpu_mem_after_search - gpu_mem_before_search:.1f} MB")
print()

# ============================================================================
# Step 11: Validate Octree Search Results
# ============================================================================
print("=" * 80)
print("STEP 11: VALIDATE OCTREE SEARCH RESULTS")
print("=" * 80)
print()

print(f"Validating {N_VALIDATION_SAMPLES} random octree results...")

# Select random found particles
found_octree_indices = np.where(element_ids_octree >= 0)[0]
sample_octree_indices = np.random.choice(
    found_octree_indices,
    min(N_VALIDATION_SAMPLES, len(found_octree_indices)),
    replace=False
)

octree_validation_errors = 0
for idx in sample_octree_indices:
    pos = particle_positions_perturbed[idx]
    elem_id = element_ids_octree[idx]

    # Get element nodes
    node_ids = connectivity[elem_id]
    tet_nodes = node_positions[node_ids]

    # Check if point is actually inside
    from jaxtrace.gpu.search.level0_cached import point_in_tet
    inside = point_in_tet(pos, tet_nodes)

    if not inside:
        octree_validation_errors += 1
        if octree_validation_errors <= 5:  # Print first few errors
            print(f"  ❌ Particle {idx}: pos={pos}, elem={elem_id} - NOT INSIDE!")

if octree_validation_errors == 0:
    print(f"✓ All {len(sample_octree_indices)} sampled particles validated correctly")
else:
    print(f"❌ Validation errors: {octree_validation_errors}/{len(sample_octree_indices)} ({octree_validation_errors/len(sample_octree_indices)*100:.1f}%)")
print()

# ============================================================================
# Step 12: Compare with Ground Truth (Brute-Force Search)
# ============================================================================
print("=" * 80)
print("STEP 12: GROUND TRUTH COMPARISON")
print("=" * 80)
print()

print("Running brute-force search on perturbed positions for ground truth...")

t0 = time.perf_counter()

element_ids_ground_truth_gpu = initial_search_brute_force(
    positions_perturbed_gpu,
    mesh_gpu.connectivity,
    mesh_gpu.node_positions
)
element_ids_ground_truth_gpu.block_until_ready()

element_ids_ground_truth = np.array(element_ids_ground_truth_gpu, dtype=np.int32)

t_ground_truth = time.perf_counter() - t0

print(f"✓ Ground truth search complete ({t_ground_truth:.2f} s)")
print(f"  Speedup: {t_ground_truth / t_search:.1f}× (octree vs brute-force)")
print()

# Compare results
n_matches = np.sum(element_ids_octree == element_ids_ground_truth)
n_mismatches = N_PARTICLES - n_matches

# For particles found by both, check if they're in same element
both_found = (element_ids_octree >= 0) & (element_ids_ground_truth >= 0)
n_both_found = np.sum(both_found)
n_match_among_found = np.sum(element_ids_octree[both_found] == element_ids_ground_truth[both_found])

print(f"Comparison with ground truth:")
print(f"  Total matches: {n_matches:,} / {N_PARTICLES:,} ({n_matches/N_PARTICLES*100:.1f}%)")
print(f"  Mismatches: {n_mismatches:,} ({n_mismatches/N_PARTICLES*100:.1f}%)")
print(f"  Both found: {n_both_found:,}")
print(f"  Agree among found: {n_match_among_found:,} / {n_both_found:,} ({n_match_among_found/n_both_found*100:.1f}%)")
print()

# Analyze mismatches
if n_mismatches > 0:
    print(f"Analyzing mismatches (showing first 10):")

    mismatch_indices = np.where(element_ids_octree != element_ids_ground_truth)[0][:10]

    for idx in mismatch_indices:
        octree_elem = element_ids_octree[idx]
        gt_elem = element_ids_ground_truth[idx]

        print(f"  Particle {idx}:")
        print(f"    Octree:       {octree_elem}")
        print(f"    Ground truth: {gt_elem}")

        # Check if octree result is actually correct
        if octree_elem >= 0:
            pos = particle_positions_perturbed[idx]
            node_ids = connectivity[octree_elem]
            tet_nodes = node_positions[node_ids]
            inside = point_in_tet(pos, tet_nodes)
            print(f"    Octree result valid: {inside}")
    print()

# ============================================================================
# Step 13: Performance Summary
# ============================================================================
print("=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print()

print(f"Timing Breakdown:")
print(f"  Mesh loading:          {t_load:.3f} s")
print(f"  Element neighbors:     {t_neighbors:.3f} s")
print(f"  Octree building:       {t_octree:.3f} s")
print(f"  GPU upload:            {t_upload:.3f} s")
print(f"  Initial assignment:    {t_init:.3f} s")
print(f"  Octree JIT warm-up:    {t_warmup:.3f} s")
print(f"  Octree search:         {t_search:.4f} s ⭐")
print(f"  Ground truth search:   {t_ground_truth:.3f} s")
print()

print(f"Octree Performance:")
print(f"  Throughput:        {N_PARTICLES / t_search:,.1f} particles/s")
print(f"  Time per particle: {t_search / N_PARTICLES * 1e6:.2f} μs")
print(f"  Speedup:           {t_ground_truth / t_search:.1f}× vs brute-force")
print()

print(f"Memory Usage:")
print(f"  GPU memory:        {gpu_mem_after:.1f} MB")
print(f"  Octree:            {(octree_metadata.nbytes + octree_elements.nbytes) / 1024**2:.1f} MB")
print(f"  Mesh:              {(connectivity.nbytes + node_positions.nbytes) / 1024**2:.1f} MB")
print()

print(f"Accuracy:")
print(f"  Octree hit rate:   {n_found_octree/N_PARTICLES*100:.1f}%")
print(f"  Ground truth rate: {np.sum(element_ids_ground_truth >= 0)/N_PARTICLES*100:.1f}%")
print(f"  Agreement:         {n_matches/N_PARTICLES*100:.1f}%")
print(f"  Validation errors: {octree_validation_errors}/{len(sample_octree_indices)}")
print()

# ============================================================================
# Final Status
# ============================================================================
print("=" * 80)

# Determine overall status
overall_success = True
issues = []

if n_found_octree < N_PARTICLES * 0.95:
    overall_success = False
    issues.append(f"Low octree hit rate ({n_found_octree/N_PARTICLES*100:.1f}%)")

if octree_validation_errors > 0:
    overall_success = False
    issues.append(f"Validation errors ({octree_validation_errors} found)")

if n_match_among_found < n_both_found * 0.95:
    overall_success = False
    issues.append(f"Low agreement with ground truth ({n_match_among_found/n_both_found*100:.1f}%)")

if overall_success:
    print("✅ TEST PASSED - Octree search is correct and performant")
else:
    print("❌ TEST FAILED - Issues detected:")
    for issue in issues:
        print(f"   - {issue}")

print("=" * 80)
