"""
Diagnose element assignment accuracy by checking if assigned elements
actually contain the particles.

This will help determine if wrong trajectories are due to:
1. Wrong initial element assignment
2. Wrong RK4 search during tracking
3. Interpolation issues
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.search.morton_octree_builder import build_global_morton_octree
from jaxtrace.gpu.search.morton_global_search import (
    upload_global_morton_to_gpu,
    position_to_leaf_id_octree,
    point_in_tet_gpu
)

print("=" * 80)
print("ELEMENT ASSIGNMENT ACCURACY DIAGNOSTIC")
print("=" * 80)

# Load mesh (copy pattern from production script)
print("\n1. Loading mesh...")
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 120)  # Load just one timestep
VELOCITY_FIELD_NAME = 'Displacement'

node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern=MESH_FILE_PATTERN,
    timestep_range=VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=False
)
print(f"   Loaded: {connectivity.shape[0]:,} elements, {node_positions.shape[0]:,} nodes")

# Deduplicate nodes
print("\n1.5. Checking for duplicate nodes...")
node_positions, connectivity, n_duplicates = deduplicate_nodes(
    node_positions, connectivity, verbose=False
)
if n_duplicates > 0:
    print(f"   Removed {n_duplicates:,} duplicate nodes")
else:
    print(f"   No duplicates found")

# Build Morton octree (copy exact pattern from production script)
print("\n2. Building Morton octree...")
octree_struct = build_global_morton_octree(
    node_positions=node_positions,
    connectivity=connectivity,
    leaf_capacity=256,
    max_depth=21,
    verbose=False
)
print(f"   Built: {octree_struct.n_leaves:,} leaves")

# Upload to GPU
mesh_gpu_octree = upload_global_morton_to_gpu(
    octree_struct,
    connectivity,
    node_positions
)
print(f"   Uploaded to GPU: n_leaves={mesh_gpu_octree.n_leaves:,}, table_depth={mesh_gpu_octree.table_depth}")

# Load particle positions from a saved state
print("\n3. Loading particle positions...")
try:
    # Try to load from latest output
    import glob
    output_files = sorted(glob.glob("outputs/particles_*.npy"))
    if output_files:
        latest = output_files[-1]
        print(f"   Loading from: {latest}")
        positions = np.load(latest)
        print(f"   Loaded: {positions.shape[0]} particles")
    else:
        raise FileNotFoundError
except:
    # Fallback: Generate test particles in refined region
    print("   No saved particles found. Generating test particles in refined region...")
    # Refined region bounds (approximate, from previous analysis)
    x_min, x_max = -0.018000, 0.009000
    y_min, y_max = -0.013800, 0.013800
    z_min, z_max = -0.007000, 0.000000

    n_test = 10000
    np.random.seed(42)
    positions = np.random.uniform(
        low=[x_min, y_min, z_min],
        high=[x_max, y_max, z_max],
        size=(n_test, 3)
    ).astype(np.float32)
    print(f"   Generated: {n_test} test particles")

# Transfer to GPU
positions_gpu = jax.device_put(positions, jax.devices()[0])

print("\n4. Testing element assignment accuracy with cascading radius search...")
print("   Method: cascading_fallback (same as production) -> verify with point_in_tet")

# Import cascading assignment (same as production)
from jaxtrace.gpu.tracking.initial_assignment_cascading import initial_assignment_cascading_fallback

# Use same radii as production (from production_tracking_fully_fused_timedep.py lines 143-144)
INITIAL_SEARCH_RADIUS = 500
INITIAL_SEARCH_FALLBACK_RADII = [1000, 2000, 5000, 10000, 100000]

print(f"   Initial radius: {INITIAL_SEARCH_RADIUS}")
print(f"   Fallback radii: {INITIAL_SEARCH_FALLBACK_RADII}")
print("   Running cascading initial assignment...")

# Run cascading assignment (exactly like production)
elem_ids = initial_assignment_cascading_fallback(
    positions_gpu,
    mesh_gpu_octree,
    initial_radius=INITIAL_SEARCH_RADIUS,
    fallback_radii=INITIAL_SEARCH_FALLBACK_RADII,
    verbose=True
)

# Verify if assigned elements actually contain the particles
print("\n   Verifying spatial accuracy with point_in_tet...")

def verify_assignment(pos, elem_id, mesh_gpu):
    """Verify if particle is actually inside assigned element (barycentric method)."""
    is_inside = jnp.where(
        elem_id >= 0,
        point_in_tet_gpu(pos, elem_id, mesh_gpu.connectivity, mesh_gpu.node_positions),
        jnp.bool_(False)
    )
    return is_inside

def verify_assignment_volume(pos, elem_id, mesh_gpu):
    """
    Alternative verification using SIGNED VOLUME method.

    Mathematical principle:
    A point P is inside tetrahedron ABCD if and only if:
    - sign(volume(PABC)) == sign(volume(DABC)) AND
    - sign(volume(DPBC)) == sign(volume(DABC)) AND
    - sign(volume(DAPC)) == sign(volume(DABC)) AND
    - sign(volume(DABP)) == sign(volume(DABC))

    This is DIFFERENT from the barycentric method used in point_in_tet_gpu.
    """
    # Only test if element is assigned
    def compute_signed_volume(p0, p1, p2, p3):
        """Compute signed volume of tetrahedron using scalar triple product."""
        v1 = p1 - p0
        v2 = p2 - p0
        v3 = p3 - p0
        # Volume = (1/6) * |v1 · (v2 × v3)|
        # We only need sign, so skip the 1/6 factor
        cross_v2_v3 = jnp.array([
            v2[1] * v3[2] - v2[2] * v3[1],
            v2[2] * v3[0] - v2[0] * v3[2],
            v2[0] * v3[1] - v2[1] * v3[0]
        ])
        signed_vol = jnp.dot(v1, cross_v2_v3)
        return signed_vol

    def check_inside():
        # Get node indices and positions
        nodes = mesh_gpu.connectivity[elem_id]
        p0 = mesh_gpu.node_positions[nodes[0]]
        p1 = mesh_gpu.node_positions[nodes[1]]
        p2 = mesh_gpu.node_positions[nodes[2]]
        p3 = mesh_gpu.node_positions[nodes[3]]

        # Compute reference volume (original tetrahedron)
        vol_ref = compute_signed_volume(p0, p1, p2, p3)

        # Compute volumes of sub-tetrahedra with query point
        vol_0 = compute_signed_volume(pos, p1, p2, p3)  # Replace p0 with pos
        vol_1 = compute_signed_volume(p0, pos, p2, p3)  # Replace p1 with pos
        vol_2 = compute_signed_volume(p0, p1, pos, p3)  # Replace p2 with pos
        vol_3 = compute_signed_volume(p0, p1, p2, pos)  # Replace p3 with pos

        # Point is inside if all sub-volumes have same sign as reference
        # (allowing small numerical tolerance)
        tol = 1e-10
        same_sign_0 = (vol_0 * vol_ref) >= -tol
        same_sign_1 = (vol_1 * vol_ref) >= -tol
        same_sign_2 = (vol_2 * vol_ref) >= -tol
        same_sign_3 = (vol_3 * vol_ref) >= -tol

        return same_sign_0 & same_sign_1 & same_sign_2 & same_sign_3

    is_inside = jnp.where(elem_id >= 0, check_inside(), jnp.bool_(False))
    return is_inside

# Create vmapped versions
verify_batch = jax.vmap(verify_assignment, in_axes=(0, 0, None))
verify_batch_volume = jax.vmap(verify_assignment_volume, in_axes=(0, 0, None))

# Verify all assignments with BOTH methods
print("   Method 1: Barycentric coordinates (used in search code)...")
is_inside_flags_barycentric = verify_batch(positions_gpu, elem_ids, mesh_gpu_octree)

print("   Method 2: Signed volume (independent verification)...")
is_inside_flags_volume = verify_batch_volume(positions_gpu, elem_ids, mesh_gpu_octree)

# Use barycentric as primary result (matches search code)
is_inside_flags = is_inside_flags_barycentric

# Get leaf IDs for analysis (using single leaf lookup)
def get_leaf_id(pos, mesh_gpu):
    return position_to_leaf_id_octree(pos, mesh_gpu)

get_leaf_batch = jax.vmap(get_leaf_id, in_axes=(0, None))
leaf_ids = get_leaf_batch(positions_gpu, mesh_gpu_octree)

# Convert to numpy for analysis
elem_ids_np = np.array(elem_ids)
is_inside_barycentric_np = np.array(is_inside_flags_barycentric)
is_inside_volume_np = np.array(is_inside_flags_volume)
leaf_ids_np = np.array(leaf_ids)

# Analyze results
n_total = len(positions)
n_assigned = np.sum(elem_ids_np >= 0)

# Results from barycentric method (used in search code)
n_correct_barycentric = np.sum(is_inside_barycentric_np)
n_wrong_barycentric = n_assigned - n_correct_barycentric

# Results from volume method (independent verification)
n_correct_volume = np.sum(is_inside_volume_np)
n_wrong_volume = n_assigned - n_correct_volume

# Compare methods
n_disagree = np.sum(is_inside_barycentric_np != is_inside_volume_np)

n_unassigned = n_total - n_assigned

print(f"\n5. RESULTS:")
print(f"   Total particles:        {n_total}")
print(f"   Assigned (elem >= 0):   {n_assigned} ({100*n_assigned/n_total:.2f}%)")
print(f"   Unassigned (elem = -1): {n_unassigned} ({100*n_unassigned/n_total:.2f}%)")
print(f"\n   Method 1 - Barycentric (used in search code):")
print(f"     Correctly inside:     {n_correct_barycentric} ({100*n_correct_barycentric/n_total:.2f}%)")
print(f"     WRONG (outside elem): {n_wrong_barycentric} ({100*n_wrong_barycentric/n_total:.2f}%)")
print(f"\n   Method 2 - Signed Volume (independent):")
print(f"     Correctly inside:     {n_correct_volume} ({100*n_correct_volume/n_total:.2f}%)")
print(f"     WRONG (outside elem): {n_wrong_volume} ({100*n_wrong_volume/n_total:.2f}%)")
print(f"\n   Method Comparison:")
print(f"     Disagreements:        {n_disagree} ({100*n_disagree/n_total:.2f}%)")

# Use barycentric as primary (matches search code)
n_correct = n_correct_barycentric
n_wrong = n_wrong_barycentric
is_inside_np = is_inside_barycentric_np

if n_wrong > 0:
    print(f"\n6. CRITICAL BUG DETECTED:")
    print(f"   {n_wrong} particles are assigned to elements that don't contain them!")
    print(f"   This explains wrong trajectories.")

    # Analyze wrong assignments
    wrong_mask = (elem_ids_np >= 0) & (~is_inside_np)
    wrong_indices = np.where(wrong_mask)[0]

    print(f"\n   Sample wrong assignments:")
    for idx in wrong_indices[:5]:
        pos = positions[idx]
        elem_id = elem_ids_np[idx]
        leaf_id = leaf_ids_np[idx]
        print(f"   Particle {idx}: pos={pos}, assigned elem={elem_id}, leaf={leaf_id}")
        print(f"                    But point_in_tet says: OUTSIDE")

    print(f"\n7. RECOMMENDED FIX:")
    print(f"   The position_to_leaf_id_octree() function is returning wrong leaves.")
    print(f"   The prefix table + Morton range check is spatially inaccurate.")
    print(f"   Solution: Implement proper binary search on leaf Morton ranges.")
    print(f"   OR: Always use search radius > 0 to search neighboring leaves.")
else:
    print(f"\n6. ACCURACY CHECK PASSED:")
    print(f"   All assigned particles are correctly inside their elements.")
    print(f"   Wrong trajectories must be due to other issues (interpolation, RK4).")

# Additional diagnostic: Check if search radius would help
if n_wrong > 0:
    print(f"\n8. Testing if search radius would fix wrong assignments...")

    def test_with_radius(pos, mesh_gpu, radius):
        """Test if searching with radius finds correct element."""
        from jaxtrace.gpu.search.morton_global_search import search_L2_global_morton_single
        elem_id = search_L2_global_morton_single(pos, mesh_gpu, search_radius=radius)
        is_inside = jnp.where(
            elem_id >= 0,
            point_in_tet_gpu(pos, elem_id, mesh_gpu.connectivity, mesh_gpu.node_positions),
            jnp.bool_(False)
        )
        return elem_id, is_inside

    # Create vmapped version
    test_with_radius_batch = jax.vmap(test_with_radius, in_axes=(0, None, None))

    # Test wrong particles with radius=10
    wrong_positions = positions_gpu[wrong_indices[:min(100, len(wrong_indices))]]
    elem_ids_r10, is_inside_r10 = test_with_radius_batch(wrong_positions, mesh_gpu_octree, jnp.int32(10))

    n_fixed = np.sum(np.array(is_inside_r10))
    print(f"   With radius=10: {n_fixed}/{len(wrong_positions)} fixed")

    if n_fixed == len(wrong_positions):
        print(f"   -> Radius search fixes all wrong assignments!")
        print(f"   -> Current issue: L2_SEARCH_RADIUS in production might be too small")
    else:
        print(f"   -> Radius search helps but doesn't fix all issues")
        print(f"   -> Deeper bug in position_to_leaf or Morton encoding")

print("\n" + "=" * 80)
