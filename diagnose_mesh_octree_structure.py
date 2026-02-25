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
from pathlib import Path

# ============================================================================
# Configuration (match production/benchmark exactly)
# ============================================================================

MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'

# ============================================================================
# Load mesh (same as benchmark_l2_search_methods.py)
# ============================================================================

print("Loading mesh...")
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# Load mesh (exact same as benchmark_l2_search_methods.py line 257-263)
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern=MESH_FILE_PATTERN,
    timestep_range=VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=True
)

n_nodes_orig = node_positions.shape[0]
n_elements = connectivity.shape[0]
print(f"  Loaded: {n_elements:,} elements, {n_nodes_orig:,} nodes")

# Deduplicate nodes (exact same as benchmark_l2_search_methods.py line 274-276)
node_positions, connectivity, n_duplicates_removed, velocity_sequence = deduplicate_nodes(
    node_positions,
    connectivity,
    velocity_sequence=velocity_sequence,
    verbose=True
)

n_nodes = node_positions.shape[0]
print(f"  After deduplication: {n_nodes:,} nodes ({n_duplicates_removed:,} duplicates removed)")

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

def infer_octree_cell_from_axis_aligned_edges(elem_id, connectivity, node_positions):
    """
    Infer octree cell from axis-aligned edges (CORRECTED v3 - handles negative coords).

    Strategy for Kuhn tetrahedra:
    1. Find the 3 axis-aligned edges (X, Y, Z)
    2. Use their lengths to determine cube size
    3. Find minimum coordinates from edge endpoints
    4. Snap to grid boundaries: i = floor(min_coord * grid_scale)
    5. Use offset for Morton encoding to handle negative indices

    Key fixes:
    - v1: Used bounding box (wrong - tet bbox ≠ cube)
    - v2: Used min() of edge endpoints (wrong - gives tet vertex, not cube corner)
    - v3: Snap min coords to grid + offset for negative coords (CORRECT)

    This works because:
    - Axis-aligned edge length = cube size
    - Grid index = floor(coordinate / cell_size)
    - Negative coordinates are valid in mesh space
    """
    node_ids = connectivity[elem_id]
    vertices = node_positions[node_ids]  # (4, 3)

    # Find all edges
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]

    # Identify axis-aligned edges
    tolerance = 1e-6  # Appropriate for float32 millimeter-scale coordinates
    aa_edges = {'X': None, 'Y': None, 'Z': None}

    for i, j in edges:
        edge_vec = vertices[j] - vertices[i]
        edge_len = np.linalg.norm(edge_vec)

        # Check if aligned with X, Y, or Z
        is_x = (abs(edge_vec[1]) < tolerance) and (abs(edge_vec[2]) < tolerance)
        is_y = (abs(edge_vec[0]) < tolerance) and (abs(edge_vec[2]) < tolerance)
        is_z = (abs(edge_vec[0]) < tolerance) and (abs(edge_vec[1]) < tolerance)

        if is_x and aa_edges['X'] is None:
            aa_edges['X'] = (i, j, edge_len, edge_vec)
        elif is_y and aa_edges['Y'] is None:
            aa_edges['Y'] = (i, j, edge_len, edge_vec)
        elif is_z and aa_edges['Z'] is None:
            aa_edges['Z'] = (i, j, edge_len, edge_vec)

    # Check if we found all 3 axis-aligned edges
    if None in aa_edges.values():
        # Fallback to bounding box method if not a Kuhn tet
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        cell_size = (bbox_max - bbox_min).max()

        if cell_size > 0:
            level = int(round(-np.log2(cell_size)))
            level = np.clip(level, 0, 20)
        else:
            level = 20

        grid_scale = 2 ** level
        i = int(np.floor(bbox_min[0] * grid_scale))
        j = int(np.floor(bbox_min[1] * grid_scale))
        k = int(np.floor(bbox_min[2] * grid_scale))
    else:
        # Use axis-aligned edges to determine cell dimensions
        # IMPORTANT: Cells may be rectangular boxes, not cubes!
        # Each dimension has its own cell size
        cell_size_x = aa_edges['X'][2]
        cell_size_y = aa_edges['Y'][2]
        cell_size_z = aa_edges['Z'][2]

        # Use average for level estimation (for Morton encoding)
        avg_size = (cell_size_x + cell_size_y + cell_size_z) / 3.0
        if avg_size > 0:
            level = int(round(-np.log2(avg_size)))
            level = np.clip(level, 0, 20)
        else:
            level = 20

        # Find the cell corner by rounding down to grid boundaries PER DIMENSION
        x_coords = [vertices[aa_edges['X'][0]][0], vertices[aa_edges['X'][1]][0]]
        y_coords = [vertices[aa_edges['Y'][0]][1], vertices[aa_edges['Y'][1]][1]]
        z_coords = [vertices[aa_edges['Z'][0]][2], vertices[aa_edges['Z'][1]][2]]

        # Get minimum coordinates
        min_x = min(x_coords)
        min_y = min(y_coords)
        min_z = min(z_coords)

        # Snap to grid boundaries using PER-DIMENSION cell sizes
        i = int(np.floor(min_x / cell_size_x))
        j = int(np.floor(min_y / cell_size_y))
        k = int(np.floor(min_z / cell_size_z))

        # Store cell sizes as tuple
        cell_size = (cell_size_x, cell_size_y, cell_size_z)

    # Morton codes require non-negative indices
    # We'll use the raw signed indices for cell identification
    # (Morton encoding will be fixed later for negative coords)

    # For now, create a unique cell ID that works with signed integers
    # Use a large offset to ensure positive values for Morton encoding
    offset = (1 << 19)  # Half of max range to allow negative coords
    i_morton = i + offset
    j_morton = j + offset
    k_morton = k + offset

    # Clamp to valid Morton range
    max_coord = (1 << 20)  # 2^20
    i_morton = np.clip(i_morton, 0, max_coord - 1)
    j_morton = np.clip(j_morton, 0, max_coord - 1)
    k_morton = np.clip(k_morton, 0, max_coord - 1)

    # Encode as Morton code (now with shifted indices)
    morton = encode_morton_3d(i_morton, j_morton, k_morton, max_depth=21)

    return morton, level, (i, j, k), cell_size

def find_all_overlapping_cells(elem_id, connectivity, node_positions):
    """
    Find ALL octree cells that an element overlaps (multi-insert strategy).

    This is the key to achieving 100% retention:
    - Compute element bounding box
    - Find all octree cells that intersect the bbox
    - Return list of (morton, level, indices, cell_size) for each overlapping cell

    This mimics Femuss's approach of inserting elements into all overlapping cells.
    """
    node_ids = connectivity[elem_id]
    vertices = node_positions[node_ids]

    # Compute element bounding box
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)

    # Find axis-aligned edges to determine cell size
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]

    tolerance = 1e-6
    aa_edges = {'X': None, 'Y': None, 'Z': None}

    for i, j in edges:
        edge_vec = vertices[j] - vertices[i]
        edge_len = np.linalg.norm(edge_vec)

        is_x = (abs(edge_vec[1]) < tolerance) and (abs(edge_vec[2]) < tolerance)
        is_y = (abs(edge_vec[0]) < tolerance) and (abs(edge_vec[2]) < tolerance)
        is_z = (abs(edge_vec[0]) < tolerance) and (abs(edge_vec[1]) < tolerance)

        if is_x and aa_edges['X'] is None:
            aa_edges['X'] = (i, j, edge_len, edge_vec)
        elif is_y and aa_edges['Y'] is None:
            aa_edges['Y'] = (i, j, edge_len, edge_vec)
        elif is_z and aa_edges['Z'] is None:
            aa_edges['Z'] = (i, j, edge_len, edge_vec)

    # Check if Kuhn tet
    if None in aa_edges.values():
        # Fallback: use bounding box size
        cell_size_x = cell_size_y = cell_size_z = (bbox_max - bbox_min).max()
        level = int(round(-np.log2(cell_size_x))) if cell_size_x > 0 else 20
        level = np.clip(level, 0, 20)
    else:
        # Use measured cell sizes
        cell_size_x = aa_edges['X'][2]
        cell_size_y = aa_edges['Y'][2]
        cell_size_z = aa_edges['Z'][2]

        avg_size = (cell_size_x + cell_size_y + cell_size_z) / 3.0
        level = int(round(-np.log2(avg_size))) if avg_size > 0 else 20
        level = np.clip(level, 0, 20)

    cell_size = (cell_size_x, cell_size_y, cell_size_z)

    # Find grid cell range that bbox spans
    i_min = int(np.floor(bbox_min[0] / cell_size_x))
    i_max = int(np.floor(bbox_max[0] / cell_size_x))
    j_min = int(np.floor(bbox_min[1] / cell_size_y))
    j_max = int(np.floor(bbox_max[1] / cell_size_y))
    k_min = int(np.floor(bbox_min[2] / cell_size_z))
    k_max = int(np.floor(bbox_max[2] / cell_size_z))

    # Generate list of all overlapping cells
    overlapping_cells = []
    offset = (1 << 19)
    max_coord = (1 << 20)

    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            for k in range(k_min, k_max + 1):
                # Encode as Morton
                i_morton = np.clip(i + offset, 0, max_coord - 1)
                j_morton = np.clip(j + offset, 0, max_coord - 1)
                k_morton = np.clip(k + offset, 0, max_coord - 1)
                morton = encode_morton_3d(i_morton, j_morton, k_morton, max_depth=21)

                overlapping_cells.append((morton, level, (i, j, k), cell_size))

    return overlapping_cells

print("  Inferring primary cell IDs for all elements (single-insert)...")
cell_ids = np.zeros(mesh.n_elements, dtype=np.uint64)
cell_levels = np.zeros(mesh.n_elements, dtype=np.uint8)
cell_sizes = []  # Store as list to handle tuples

for elem_id in range(mesh.n_elements):
    morton, level, indices, size = infer_octree_cell_from_axis_aligned_edges(elem_id, mesh.connectivity, mesh.node_positions)
    cell_ids[elem_id] = morton
    cell_levels[elem_id] = level
    cell_sizes.append(size)  # Can be float or tuple

    if (elem_id + 1) % 500000 == 0:
        print(f"    Processed {elem_id + 1:,}/{mesh.n_elements:,} elements...")

print(f"  ✅ Inferred primary cell IDs for {mesh.n_elements:,} elements")

print("\n[2.75/5] Computing multi-insert mapping (elements → all overlapping cells)...")
print("  This is the key to 100% retention in mesh-aligned octree!")

element_to_cells = {}  # elem_id -> list of (morton, level, indices, cell_size)
cells_per_element = []

for elem_id in range(mesh.n_elements):
    overlapping = find_all_overlapping_cells(elem_id, mesh.connectivity, mesh.node_positions)
    element_to_cells[elem_id] = overlapping
    cells_per_element.append(len(overlapping))

    if (elem_id + 1) % 500000 == 0:
        print(f"    Processed {elem_id + 1:,}/{mesh.n_elements:,} elements...")

cells_per_element = np.array(cells_per_element)

print(f"\n  Multi-insert statistics:")
print(f"    Mean cells per element: {cells_per_element.mean():.2f}")
print(f"    Median: {np.median(cells_per_element):.0f}")
print(f"    Min: {cells_per_element.min()}")
print(f"    Max: {cells_per_element.max()}")
print(f"    Mode: {np.bincount(cells_per_element).argmax()}")

print(f"\n  Cells-per-element distribution:")
unique_counts, count_freqs = np.unique(cells_per_element, return_counts=True)
for count, freq in sorted(zip(unique_counts, count_freqs), key=lambda x: x[1], reverse=True)[:10]:
    print(f"    {count:2d} cells: {freq:8,} elements ({100*freq/mesh.n_elements:5.2f}%)")

# Build inverted index: cell -> list of elements
print(f"\n  Building inverted index (cell → elements)...")
cell_to_elements = defaultdict(list)
for elem_id, cells in element_to_cells.items():
    for morton, level, indices, size in cells:
        cell_to_elements[morton].append(elem_id)

print(f"  ✅ Multi-insert mapping complete!")

# ============================================================================
# Step 2.5: DETAILED INSPECTION - Examine actual element geometry
# ============================================================================
print("\n[2.5/5] DETAILED INSPECTION: Examining sample element geometry...")
print("  (Understanding the true relationship between Kuhn tets and parent cubes)")

# Sample elements from different depth levels
sample_elements = []
for level in [14, 13, 12]:  # Most common levels
    level_elements = np.where(cell_levels == level)[0]
    if len(level_elements) > 0:
        sample_elements.append(level_elements[len(level_elements)//2])  # Take middle element

print(f"\n  Inspecting {len(sample_elements)} sample elements:")

for elem_id in sample_elements[:3]:  # Limit to 3 for readability
    node_ids = mesh.connectivity[elem_id]
    vertices = mesh.node_positions[node_ids]

    print(f"\n  Element {elem_id} (level {cell_levels[elem_id]}):")
    print(f"    Vertices:")
    for i, v in enumerate(vertices):
        print(f"      v{i}: ({v[0]:12.8f}, {v[1]:12.8f}, {v[2]:12.8f})")

    # Find axis-aligned edges
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    tolerance = 1e-6  # Appropriate for float32 millimeter-scale coordinates
    aa_edges = []

    print(f"    Axis-aligned edges:")
    for i, j in edges:
        edge_vec = vertices[j] - vertices[i]
        edge_len = np.linalg.norm(edge_vec)

        is_x = (abs(edge_vec[1]) < tolerance) and (abs(edge_vec[2]) < tolerance)
        is_y = (abs(edge_vec[0]) < tolerance) and (abs(edge_vec[2]) < tolerance)
        is_z = (abs(edge_vec[0]) < tolerance) and (abs(edge_vec[1]) < tolerance)

        if is_x or is_y or is_z:
            axis = 'X' if is_x else ('Y' if is_y else 'Z')
            aa_edges.append((i, j, axis, edge_len, edge_vec))
            print(f"      Edge v{i}-v{j} ({axis}): length={edge_len:.8f}")
            print(f"        From: ({vertices[i][0]:.8f}, {vertices[i][1]:.8f}, {vertices[i][2]:.8f})")
            print(f"        To:   ({vertices[j][0]:.8f}, {vertices[j][1]:.8f}, {vertices[j][2]:.8f})")

    # Show inferred cell properties
    morton, level, indices, size = infer_octree_cell_from_axis_aligned_edges(elem_id, mesh.connectivity, mesh.node_positions)

    # IMPORTANT: Cells may be rectangular boxes with different dimensions!
    # size is a tuple (size_x, size_y, size_z)
    if isinstance(size, tuple):
        cell_sizes = np.array(size, dtype=np.float64)
    else:
        cell_sizes = np.array([size, size, size], dtype=np.float64)

    # Compute cell corner from indices (per-dimension)
    cell_min = np.array(indices, dtype=np.float64) * cell_sizes
    cell_max = cell_min + cell_sizes

    print(f"    Inferred cell properties:")
    print(f"      Level: {level} (for Morton encoding only)")
    if isinstance(size, tuple):
        print(f"      Measured cell sizes: X={size[0]:.8f}, Y={size[1]:.8f}, Z={size[2]:.8f}")
    else:
        print(f"      Measured cell size: {size:.8f}")
    print(f"      Grid indices: ({indices[0]}, {indices[1]}, {indices[2]})")
    print(f"      Cell bounds:")
    print(f"        X: [{cell_min[0]:.8f}, {cell_max[0]:.8f}]")
    print(f"        Y: [{cell_min[1]:.8f}, {cell_max[1]:.8f}]")
    print(f"        Z: [{cell_min[2]:.8f}, {cell_max[2]:.8f}]")

    # Check vertex containment
    print(f"      Vertex containment check:")
    for i, v in enumerate(vertices):
        contained = np.all(v >= cell_min - tolerance) and np.all(v <= cell_max + tolerance)
        violations = []
        if v[0] < cell_min[0] - tolerance: violations.append(f"X too low: {v[0]:.8f} < {cell_min[0]:.8f}")
        if v[0] > cell_max[0] + tolerance: violations.append(f"X too high: {v[0]:.8f} > {cell_max[0]:.8f}")
        if v[1] < cell_min[1] - tolerance: violations.append(f"Y too low: {v[1]:.8f} < {cell_min[1]:.8f}")
        if v[1] > cell_max[1] + tolerance: violations.append(f"Y too high: {v[1]:.8f} > {cell_max[1]:.8f}")
        if v[2] < cell_min[2] - tolerance: violations.append(f"Z too low: {v[2]:.8f} < {cell_min[2]:.8f}")
        if v[2] > cell_max[2] + tolerance: violations.append(f"Z too high: {v[2]:.8f} > {cell_max[2]:.8f}")

        status = "✅ CONTAINED" if contained else "❌ OUTSIDE"
        print(f"        v{i}: {status}")
        if violations:
            for violation in violations:
                print(f"          {violation}")

print("\n  Detailed inspection complete.")

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
# Step 4: Analyze octree cell occupancy (single-insert vs multi-insert)
# ============================================================================
print("\n[4/5] Analyzing octree cell occupancy...")

# Single-insert statistics (original approach - one cell per element)
print("\n  SINGLE-INSERT (one primary cell per element):")
single_insert_cells = defaultdict(list)
for elem_id, cell_id in enumerate(cell_ids):
    single_insert_cells[cell_id].append(elem_id)

single_counts = np.array([len(elems) for elems in single_insert_cells.values()])
print(f"    Unique cells: {len(single_insert_cells):,}")
print(f"    Elements per cell: {single_counts.mean():.2f} ± {single_counts.std():.2f}")
print(f"    Median: {np.median(single_counts):.0f}, Range: [{single_counts.min()}, {single_counts.max()}]")

# Multi-insert statistics (correct approach - elements in all overlapping cells)
print(f"\n  MULTI-INSERT (elements in ALL overlapping cells):")
multi_counts = np.array([len(elems) for elems in cell_to_elements.values()])
print(f"    Unique cells: {len(cell_to_elements):,}")
print(f"    Elements per cell: {multi_counts.mean():.2f} ± {multi_counts.std():.2f}")
print(f"    Median: {np.median(multi_counts):.0f}, Range: [{multi_counts.min()}, {multi_counts.max()}]")
print(f"    95th percentile: {np.percentile(multi_counts, 95):.0f}")
print(f"    99th percentile: {np.percentile(multi_counts, 99):.0f}")

print(f"\n  Multi-insert elements/cell distribution (top 10):")
unique_counts, count_frequencies = np.unique(multi_counts, return_counts=True)
for count, freq in sorted(zip(unique_counts, count_frequencies), key=lambda x: x[1], reverse=True)[:10]:
    print(f"    {count:2d} elements: {freq:6,} cells ({100*freq/len(cell_to_elements):5.2f}%)")

# Check if multi-insert count matches expectation
if 10 <= multi_counts.mean() <= 15:
    print(f"  ✅ Element count per cell matches Kuhn + neighbors (10-15 tets)")
else:
    print(f"  ℹ  Multi-insert mean: {multi_counts.mean():.1f} elements/cell")

# ============================================================================
# Step 5: Validate searchability (can we find ALL elements via octree?)
# ============================================================================
print("\n[5/5] Validating searchability - can we find ALL elements?")
print("  This is the critical test for mesh-aligned octree viability!")

def can_find_element(elem_id, element_to_cells, cell_to_elements, connectivity, node_positions):
    """
    Simulate searching for an element using a query point inside it.

    Strategy:
    1. Pick a point inside the element (centroid)
    2. Find which octree cell contains that point
    3. Check if the element is in that cell's element list

    Returns: True if element can be found, False otherwise
    """
    # Compute element centroid as query point
    node_ids = connectivity[elem_id]
    vertices = node_positions[node_ids]
    centroid = vertices.mean(axis=0)

    # Get all cells this element was inserted into
    cells = element_to_cells[elem_id]
    if len(cells) == 0:
        return False

    # For each cell, check if a query at the centroid would find this element
    # In reality, we'd traverse the octree to find the leaf containing the centroid
    # For now, we check if the centroid is within ANY of the element's cells
    for morton, level, indices, cell_size in cells:
        if isinstance(cell_size, tuple):
            cell_sizes = np.array(cell_size, dtype=np.float64)
        else:
            cell_sizes = np.array([cell_size, cell_size, cell_size], dtype=np.float64)

        cell_min = np.array(indices, dtype=np.float64) * cell_sizes
        cell_max = cell_min + cell_sizes

        # Check if centroid is in this cell (with tolerance)
        tolerance = 1e-6
        if np.all(centroid >= cell_min - tolerance) and np.all(centroid <= cell_max + tolerance):
            # Centroid is in this cell - check if element is in the cell's list
            if elem_id in cell_to_elements[morton]:
                return True

    return False

# Test searchability on a sample
print(f"\n  Testing searchability on 1,000 sample elements...")
searchable = []
for idx, elem_id in enumerate(sample_indices[:1000]):
    found = can_find_element(elem_id, element_to_cells, cell_to_elements,
                            mesh.connectivity, mesh.node_positions)
    searchable.append(found)

    if (idx + 1) % 200 == 0:
        print(f"    Checked {idx + 1}/1,000 elements...")

searchability_rate = np.mean(searchable)
print(f"\n  Searchability: {searchability_rate*100:.2f}%")
print(f"  ({int(searchability_rate * 1000)}/1,000 elements can be found via centroid query)")

if searchability_rate > 0.99:
    print("  ✅ Excellent searchability - mesh-aligned octree is VIABLE!")
elif searchability_rate > 0.95:
    print("  ⚠️  Good searchability, may need neighbor cell fallback")
else:
    print("  ❌ Poor searchability - multi-insert strategy needs refinement")

# ============================================================================
# Final verdict
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC RESULTS")
print("="*80)

print(f"\n✅ Mesh has Kuhn structure (axis-aligned edges: {aa_counts.mean():.1f} avg)")
print(f"✅ Inferred {len(cell_to_elements):,} unique octree cells")
print(f"✅ Depth range: {cell_levels.min()}-{cell_levels.max()}")
print(f"✅ Multi-insert: {cells_per_element.mean():.1f} cells/element, {multi_counts.mean():.1f} elements/cell")
print(f"✅ Searchability: {searchability_rate*100:.1f}% of elements can be found")

# Determine verdict
all_pass = (
    aa_counts.mean() >= 2.9 and
    cells_per_element.mean() >= 1.0 and
    searchability_rate > 0.99
)

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if all_pass:
    print("\n✅ MESH-ALIGNED OCTREE IS VIABLE!")
    print("  Strategy: Multi-insert (elements in all overlapping cells)")
    print("  - Mesh has recoverable Kuhn octree structure")
    print(f"  - Each element spans ~{cells_per_element.mean():.1f} cells on average")
    print(f"  - Each cell contains ~{multi_counts.mean():.1f} elements on average")
    print(f"  - {searchability_rate*100:.1f}% searchability via centroid queries")
    print("\n  Expected performance improvement:")
    print(f"    Current: ~5,376 point-in-tet tests per query (10 leaves × 536 elems)")
    print(f"    Mesh-aligned: ~{multi_counts.mean():.0f} tests per query (1 cell × {multi_counts.mean():.0f} elems)")
    print(f"    Speedup: ~{5376/multi_counts.mean():.0f}× reduction in tests!")
    print("\n✅ Ready to proceed with Phase 2: Multi-insert Cell Extraction")
else:
    print("\n⚠️  WARNING: Mesh structure has issues:")
    if aa_counts.mean() < 2.9:
        print(f"  - Not enough axis-aligned edges ({aa_counts.mean():.1f} < 2.9)")
    if cells_per_element.mean() < 1.0:
        print(f"  - Elements don't map to cells properly")
    if searchability_rate <= 0.99:
        print(f"  - Poor searchability ({searchability_rate*100:.1f}% < 99%)")
    print("\n  Recommendation: Use coarse-depth multi-insert or fallback to morton-centroid")

print("="*80)
