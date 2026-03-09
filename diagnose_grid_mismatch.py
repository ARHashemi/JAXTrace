#!/usr/bin/env python3
"""
Diagnose grid index computation mismatch.

Goal: Understand why floor(centroid / cell_size) gives different results
during extraction vs during query.
"""

import numpy as np

# From verify_search_correctness.log, Particle 0
centroid_y = -0.00307466
cell_size_extraction = np.float64(0.00015972)  # Stored in octree
cell_size_query = np.float32(0.00015972)       # Loaded to GPU

print(f"{'='*80}")
print("Grid Index Computation Mismatch Diagnosis")
print(f"{'='*80}\n")

print(f"Centroid Y: {centroid_y}")
print(f"Cell size (extraction, float64): {cell_size_extraction}")
print(f"Cell size (query, float32): {cell_size_query}")
print()

# Compute grid index (extraction - float64)
j_extraction_f64 = np.floor(centroid_y / cell_size_extraction)
print(f"Extraction (float64):")
print(f"  centroid_y / cell_size = {centroid_y / cell_size_extraction}")
print(f"  floor(...) = {j_extraction_f64}")
print(f"  int(...) = {int(j_extraction_f64)}")
print()

# Compute grid index (query - float32)
j_query_f32 = np.floor(np.float32(centroid_y) / cell_size_query)
print(f"Query (float32):")
print(f"  centroid_y / cell_size = {np.float32(centroid_y) / cell_size_query}")
print(f"  floor(...) = {j_query_f32}")
print(f"  int(...) = {int(j_query_f32)}")
print()

# Check if they differ
if int(j_extraction_f64) != int(j_query_f32):
    print(f"❌ MISMATCH: {int(j_extraction_f64)} != {int(j_query_f32)}")
    print(f"   Difference: {int(j_extraction_f64) - int(j_query_f32)}")
else:
    print(f"✅ MATCH: {int(j_extraction_f64)} == {int(j_query_f32)}")

print()
print(f"{'='*80}")
print("ROOT CAUSE HYPOTHESIS")
print(f"{'='*80}\n")
print("Possible causes:")
print("1. float64 → float32 precision loss in cell_size")
print("2. float64 → float32 precision loss in centroid position")
print("3. Different rounding modes between extraction vs query")
print("4. Compiler optimization differences")
print()
print(f"Testing hypothesis 1: Same dtype throughout")
print()

# Test with consistent float64
centroid_y_f64 = np.float64(-0.00307466)
cell_size_f64 = np.float64(0.00015972)
j_f64 = int(np.floor(centroid_y_f64 / cell_size_f64))
print(f"  float64 throughout: j = {j_f64}")

# Test with consistent float32
centroid_y_f32 = np.float32(-0.00307466)
cell_size_f32 = np.float32(0.00015972)
j_f32 = int(np.floor(centroid_y_f32 / cell_size_f32))
print(f"  float32 throughout: j = {j_f32}")

if j_f64 != j_f32:
    print()
    print(f"✅ CONFIRMED: float64 → float32 conversion causes mismatch!")
    print(f"   Extraction uses float64: j = {j_f64}")
    print(f"   Query uses float32: j = {j_f32}")
    print()
    print("SOLUTION: Use float32 throughout, or ensure consistent precision")
else:
    print()
    print("Hypothesis 1 rejected - precision is not the issue")

print()
