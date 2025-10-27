#!/usr/bin/env python3
"""
Test octree element assignment fix.

This tests the fix for elements spanning multiple octree cells.
"""

import numpy as np
import sys

print("=" * 70)
print("OCTREE ELEMENT ASSIGNMENT FIX TEST")
print("=" * 70)

# Import the fixed octree builder
from jaxtrace.fields.octree_fem_interpolator_optimized import (
    build_octree_mesh_optimized,
    create_octree_fem_interpolator_optimized
)

print("\n📋 Test Case: Element Spanning All 8 Octants")
print("=" * 70)

# Create a tetrahedron that spans all 8 octants around origin
points = np.array([
    [-1.0, -1.0, -1.0],  # Node 0: Octant 0 (---)
    [ 1.0,  1.0,  1.0],  # Node 1: Octant 7 (+++)
    [ 1.0, -1.0,  1.0],  # Node 2: Octant 5 (+-+)
    [-1.0,  1.0, -1.0],  # Node 3: Octant 2 (-+-)
], dtype=np.float32)

connectivity = np.array([[0, 1, 2, 3]], dtype=np.int32)

print(f"\nElement nodes:")
for i, p in enumerate(points):
    octant = 0
    if p[0] >= 0: octant += 1
    if p[1] >= 0: octant += 2
    if p[2] >= 0: octant += 4
    print(f"  Node {i}: {p} → Octant {octant}")

print(f"\nElement centroid: {points.mean(axis=0)}")
print(f"Element bounds: {points.min(axis=0)} to {points.max(axis=0)}")

# Build octree with very low max_elements_per_leaf to force subdivision
print(f"\n🌲 Building octree (forcing subdivision)...")
mesh = build_octree_mesh_optimized(
    points,
    connectivity,
    max_elements_per_leaf=0,  # Force subdivision even with 1 element
    max_depth=2  # Allow 2 levels
)

print(f"\n🔍 Testing Query Points in All 8 Octants")
print("=" * 70)

# Test points in all 8 octants (all should be inside or near the tetrahedron)
test_points = np.array([
    [-0.3, -0.3, -0.3],  # Octant 0 (---)
    [ 0.3, -0.3, -0.3],  # Octant 1 (+--)
    [-0.3,  0.3, -0.3],  # Octant 2 (-+-)
    [ 0.3,  0.3, -0.3],  # Octant 3 (++-)
    [-0.3, -0.3,  0.3],  # Octant 4 (--+)
    [ 0.3, -0.3,  0.3],  # Octant 5 (+-+)
    [-0.3,  0.3,  0.3],  # Octant 6 (-++)
    [ 0.3,  0.3,  0.3],  # Octant 7 (+++)
], dtype=np.float32)

# Use node positions as field values (simple test)
field_values = points.copy()

# Create interpolator
print(f"\n🔧 Creating interpolator...")
interpolator = create_octree_fem_interpolator_optimized(mesh)

# Test interpolation
print(f"\n📊 Interpolation Results:")
print("-" * 70)

results = interpolator(test_points, field_values)

all_good = True
for i, (test_point, result) in enumerate(zip(test_points, results)):
    octant = 0
    if test_point[0] >= 0: octant += 1
    if test_point[1] >= 0: octant += 2
    if test_point[2] >= 0: octant += 4

    # Check if result is reasonable (should be close to test point since field = positions)
    distance = np.linalg.norm(result - test_point)

    # If distance is large, likely fallback was used (wrong!)
    is_good = distance < 1.0  # Should be close since we're interpolating positions

    status = "✅ GOOD" if is_good else "❌ BAD (likely fallback)"
    print(f"Octant {octant}: query={test_point} → result={result}")
    print(f"          Distance from query: {distance:.3f} {status}")

    if not is_good:
        all_good = False

print("\n" + "=" * 70)
if all_good:
    print("✅ TEST PASSED: All query points interpolated correctly!")
    print("   Fix is working - elements found in all relevant octants")
    sys.exit(0)
else:
    print("❌ TEST FAILED: Some query points used fallback")
    print("   Fix may not be working correctly")
    sys.exit(1)
