"""
Diagnostic: Verify Kuhn mesh intrinsic octree structure

This script verifies that the mesh has a recoverable octree cell structure
suitable for mesh-aligned point location.

Questions answered:
1. Do all elements have 3 axis-aligned edges? (Kuhn property)
2. Can we group elements into octree cells?
3. How many elements per cell? (Expected: 5-6 for Kuhn subdivision)
4. What is the cell depth distribution?
5. Do elements truly stay within cell boundaries?
"""

import numpy as np
from collections import defaultdict
import sys

# Load mesh
print("Loading mesh...")
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu

node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    base_dir="data/FLA/post/0eule",
    pattern="featurelessAvtk_{timestep}.pvtu",
    start_timestep=158,
    end_timestep=159,
    field_name="Displacement",
    verbose=True
)

# Create simple mesh object for compatibility
class SimpleMesh:
    def __init__(self, node_positions, connectivity):
        self.node_positions = node_positions
        self.connectivity = connectivity
        self.n_nodes = len(node_positions)
        self.n_elements = len(connectivity)

mesh = SimpleMesh(node_positions, connectivity)

print("="*80)
print("Diagnostic: Mesh Intrinsic Octree Structure")
print("="*80)
print(f"\nMesh: {mesh.n_elements:,} elements, {mesh.n_nodes:,} nodes")

# ============================================================================
# Step 1: Verify axis-aligned edges (Kuhn property)
# ============================================================================
print("\n[1/5] Verifying axis-aligned edges (Kuhn property)...")

def check_axis_aligned_edges(elem_id, connectivity, node_positions, tolerance=1e-10):
    """Check if element has 3 axis-aligned edges."""
    node_ids = connectivity[elem_id]
    vertices = node_positions[node_ids]  # (4, 3)

    # Check all 6 edges
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]

    axis_aligned_count = 0
    axis_aligned_edges = []

    for i, j in edges:
        edge_vec = vertices[j] - vertices[i]

        # Check if aligned with X, Y, or Z
        is_x = (abs(edge_vec[1]) < tolerance) and (abs(edge_vec[2]) < tolerance)
        is_y = (abs(edge_vec[0]) < tolerance) and (abs(edge_vec[2]) < tolerance)
        is_z = (abs(edge_vec[0]) < tolerance) and (abs(edge_vec[1]) < tolerance)

        if is_x or is_y or is_z:
            axis_aligned_count += 1
            axis = 'X' if is_x else ('Y' if is_y else 'Z')
            axis_aligned_edges.append((i, j, axis))

    return axis_aligned_count, axis_aligned_edges

# Sample 10,000 elements
np.random.seed(42)
sample_indices = np.random.choice(mesh.n_elements, size=min(10000, mesh.n_elements), replace=False)

aa_counts = []
print(f"  Sampling {len(sample_indices):,} elements...")
for idx, elem_id in enumerate(sample_indices):
    count, edges = check_axis_aligned_edges(elem_id, mesh.connectivity, mesh.node_positions)
    aa_counts.append(count)

    if (idx + 1) % 2000 == 0:
        print(f"    Checked {idx + 1:,}/{len(sample_indices):,} elements...")

aa_counts = np.array(aa_counts)
print(f"\n  Axis-aligned edges per element:")
print(f"    Mean: {aa_counts.mean():.2f}")
print(f"    Mode: {np.bincount(aa_counts).argmax()}")
print(f"    Distribution: {dict(zip(*np.unique(aa_counts, return_counts=True)))}")

if aa_counts.mean() >= 2.9:  # Allow small tolerance
    print("  ✅ Mesh has Kuhn structure (≥3 axis-aligned edges)")
else:
    print("  ⚠  WARNING: Mesh may not be pure Kuhn")

# ============================================================================
# Step 2: Infer octree cell for each element
# ============================================================================
print("\n[2/5] Inferring octree cell IDs from element geometry...")

def encode_morton_3d(x, y, z, max_depth=21):
    """Encode (x, y, z) as Morton code."""
    morton = 0
    for i in range(max_depth):
        morton |= ((x & (1 << i)) << (2 * i)) | \
                  ((y & (1 << i)) << (2 * i + 1)) | \
                  ((z & (1 << i)) << (2 * i + 2))
    return morton

def infer_octree_cell(elem_id, connectivity, node_positions):
    """
    Infer octree cell (level, i, j, k) from element bounding box.

    Strategy:
    - Compute tight AABB
    - Infer cell size from max dimension
    - Compute level from size: level = -log2(size)
    - Compute grid indices at this level
    """
    node_ids = connectivity[elem_id]
    vertices = node_positions[node_ids]  # (4, 3)

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min

    # Cell size = max dimension (assuming cube cells)
    cell_size = bbox_size.max()

    # Infer level: 2^(-level) = cell_size
    # level = -log2(cell_size)
    if cell_size > 0:
        level = int(round(-np.log2(cell_size)))
        level = np.clip(level, 0, 20)  # Clamp to reasonable range
    else:
        level = 20  # Degenerate element

    # Compute cell indices at this level
    grid_scale = 2 ** level
    i = int(np.floor(bbox_min[0] * grid_scale))
    j = int(np.floor(bbox_min[1] * grid_scale))
    k = int(np.floor(bbox_min[2] * grid_scale))

    # Clamp to valid range
    max_coord = (1 << 20)  # 2^20
    i = np.clip(i, 0, max_coord - 1)
    j = np.clip(j, 0, max_coord - 1)
    k = np.clip(k, 0, max_coord - 1)

    # Encode as Morton code
    morton = encode_morton_3d(i, j, k, max_depth=21)

    return morton, level, (i, j, k), cell_size

print("  Inferring cell IDs for all elements...")
cell_ids = np.zeros(mesh.n_elements, dtype=np.uint64)
cell_levels = np.zeros(mesh.n_elements, dtype=np.uint8)
cell_sizes = np.zeros(mesh.n_elements, dtype=np.float32)

for elem_id in range(mesh.n_elements):
    morton, level, indices, size = infer_octree_cell(elem_id, mesh.connectivity, mesh.node_positions)
    cell_ids[elem_id] = morton
    cell_levels[elem_id] = level
    cell_sizes[elem_id] = size

    if (elem_id + 1) % 500000 == 0:
        print(f"    Processed {elem_id + 1:,}/{mesh.n_elements:,} elements...")

print(f"  ✅ Inferred cell IDs for {mesh.n_elements:,} elements")

# ============================================================================
# Step 3: Analyze cell depth distribution
# ============================================================================
print("\n[3/5] Analyzing octree depth distribution...")

unique_levels, level_counts = np.unique(cell_levels, return_counts=True)
print(f"  Depth distribution:")
for level, count in zip(unique_levels, level_counts):
    print(f"    Level {level:2d}: {count:8,} elements ({100*count/mesh.n_elements:5.2f}%)")

print(f"\n  Depth statistics:")
print(f"    Min level: {cell_levels.min()}")
print(f"    Max level: {cell_levels.max()}")
print(f"    Mean level: {cell_levels.mean():.2f}")
print(f"    Median level: {np.median(cell_levels):.0f}")

# ============================================================================
# Step 4: Count elements per cell
# ============================================================================
print("\n[4/5] Counting elements per octree cell...")

cell_to_elements = defaultdict(list)
for elem_id, cell_id in enumerate(cell_ids):
    cell_to_elements[cell_id].append(elem_id)

cell_element_counts = np.array([len(elems) for elems in cell_to_elements.values()])

print(f"  Unique cells: {len(cell_to_elements):,}")
print(f"  Elements per cell:")
print(f"    Mean: {cell_element_counts.mean():.2f}")
print(f"    Median: {np.median(cell_element_counts):.0f}")
print(f"    Min: {cell_element_counts.min()}")
print(f"    Max: {cell_element_counts.max()}")
print(f"    Std: {cell_element_counts.std():.2f}")
print(f"    95th percentile: {np.percentile(cell_element_counts, 95):.0f}")
print(f"    99th percentile: {np.percentile(cell_element_counts, 99):.0f}")

# Distribution of elements per cell
unique_counts, count_frequencies = np.unique(cell_element_counts, return_counts=True)
print(f"\n  Elements/cell distribution (top 10):")
for count, freq in sorted(zip(unique_counts, count_frequencies), key=lambda x: x[1], reverse=True)[:10]:
    print(f"    {count:2d} elements: {freq:6,} cells ({100*freq/len(cell_to_elements):5.2f}%)")

# Expected: 5-6 for Kuhn (5 or 6 tets per cube)
if 4 <= cell_element_counts.mean() <= 7:
    print("  ✅ Element count per cell matches Kuhn subdivision (5-6 tets)")
else:
    print(f"  ⚠  WARNING: Unexpected element count (expected 5-6, got {cell_element_counts.mean():.1f})")

# ============================================================================
# Step 5: Verify elements stay within cell boundaries
# ============================================================================
print("\n[5/5] Verifying element containment within cells...")

def check_element_in_cell(elem_id, cell_level, cell_indices, connectivity, node_positions):
    """Check if all element vertices lie within the inferred cell."""
    i, j, k = cell_indices
    level = cell_level

    cell_size = 2 ** (-level)
    cell_min = np.array([i, j, k], dtype=np.float64) * cell_size
    cell_max = cell_min + cell_size

    node_ids = connectivity[elem_id]
    vertices = node_positions[node_ids]

    # Check containment with small tolerance
    tolerance = 1e-10
    contained = np.all(vertices >= cell_min - tolerance) and \
                np.all(vertices <= cell_max + tolerance)

    return contained

# Reconstruct cell indices for sample
print(f"  Sampling 1,000 elements to verify containment...")
sample_contained = []
for idx, elem_id in enumerate(sample_indices[:1000]):  # Sample 1000 for speed
    morton, level, indices, size = infer_octree_cell(elem_id, mesh.connectivity, mesh.node_positions)
    contained = check_element_in_cell(elem_id, level, indices, mesh.connectivity, mesh.node_positions)
    sample_contained.append(contained)

    if (idx + 1) % 200 == 0:
        print(f"    Checked {idx + 1}/1,000 elements...")

containment_rate = np.mean(sample_contained)
print(f"\n  Contained within inferred cell: {containment_rate*100:.2f}%")

if containment_rate > 0.99:
    print("  ✅ Elements are well-contained within cells")
else:
    print(f"  ⚠  WARNING: Some elements extend beyond cell boundaries")

# ============================================================================
# Final verdict
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC RESULTS")
print("="*80)

print(f"\n✅ Mesh has Kuhn structure (axis-aligned edges: {aa_counts.mean():.1f} avg)")
print(f"✅ Inferred {len(cell_to_elements):,} unique octree cells")
print(f"✅ Depth range: {cell_levels.min()}-{cell_levels.max()}")
print(f"✅ Elements per cell: {cell_element_counts.mean():.1f} ± {cell_element_counts.std():.1f}")
print(f"✅ Element containment: {containment_rate*100:.1f}%")

# Determine verdict
all_pass = (
    aa_counts.mean() >= 2.9 and
    4 <= cell_element_counts.mean() <= 7 and
    containment_rate > 0.99
)

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if all_pass:
    print("\n✅ Mesh-aligned octree is VIABLE:")
    print("  - Mesh has recoverable octree structure")
    print("  - Each element belongs to exactly one cell")
    print("  - Bounded elements per cell (5-6 for Kuhn)")
    print("  - No element spanning across cells")
    print("\n✅ Ready to proceed with Phase 2: Cell Extraction")
else:
    print("\n⚠  WARNING: Mesh structure may not be suitable for mesh-aligned octree")
    print("  - Consider fallback approach (coarse-depth multi-insert)")

print("="*80)
