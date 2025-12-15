# Octree vs Blockwise Initialization Comparison Test

## Purpose

This test compares the accuracy and performance of two particle initialization methods:
1. **Octree search** - Hierarchical spatial structure (used in Scenario #2)
2. **Blockwise search** - Regular grid with hash buckets (used in production_tracking_3hop_l2_octree.py)

## Test Design

### Particle Generation Strategy

Particles are placed at **element centroids with small random perturbations**:

1. Compute centroids for all mesh elements
2. Randomly select N elements
3. Place particle at each centroid + small random offset
4. Random offset is **1% of minimum element size** (guarantees particle stays inside element)
5. Store true element ID as ground truth

This ensures:
- ✓ All particles are definitely inside an element (100% should be found)
- ✓ Ground truth is known for accuracy validation
- ✓ Realistic particle distribution (not just at exact centroids)

### Test Procedure

1. **Load mesh data** - Real production mesh (threadedAvtk_120.pvtu)
2. **Initialize octree structure**:
   - Build octree with level-field filtering (threshold=1.1)
   - Max depth: 15, max leaf size: 50
   - Upload to GPU
3. **Initialize blockwise structure**:
   - Create 40×40×40 regular grid
   - Build padded block arrays
   - Classify blocks (INTERIOR/BOUNDARY/MIXED/OUTSIDE)
   - Build 26-connectivity for blocks
   - Create hash buckets
4. **Generate test particles** (default: 50,000):
   - Place at element centroids with 1% perturbation
   - Record true element IDs
5. **Test octree search**:
   - Upload particles to GPU
   - JIT-compile search function (warm-up)
   - Measure search time
   - Download results
   - Compare found elements vs ground truth
6. **Test blockwise search**:
   - Use same particle positions
   - Measure search time
   - Compare found elements vs ground truth

### Metrics

For each method:
- **Found rate**: % of particles assigned to an element
- **Accuracy**: % of found elements matching ground truth
- **Search time**: Total time in seconds
- **Throughput**: Particles processed per second
- **Mismatches**: Number of particles assigned to wrong element

## Expected Results

### Accuracy
Both methods should achieve **>99.9% accuracy** because:
- Particles are inside elements (1% perturbation is very small)
- Both methods use point-in-tet checks for validation

### Performance
Expected throughput:
- **Octree**: 100,000-500,000 p/s (GPU-parallelized, scan-based)
- **Blockwise**: 50,000-150,000 p/s (CPU-based with hash lookups)

**Prediction**: Octree should be **2-5× faster** due to:
- Full GPU parallelization (vmap over all particles)
- Efficient spatial pruning (hierarchical traversal)
- No CPU-side hash lookups

### Memory
- **Octree**: ~2-5 MB (filtered structure with level field)
- **Blockwise**: ~50-100 MB (padded arrays + hash buckets)

## Running the Test

```bash
source .venv/bin/activate
python test_octree_vs_blockwise_initialization.py
```

Or with custom particle count:
```bash
# Edit N_PARTICLES in main() function, then run
python test_octree_vs_blockwise_initialization.py
```

## Configuration

Parameters in `main()`:
- `MESH_PATH`: Path to mesh file (default: threadedAvtk_120.pvtu)
- `GRID_SIZE`: Blockwise grid resolution (default: 40×40×40)
- `OCTREE_MAX_DEPTH`: Maximum octree depth (default: 15)
- `OCTREE_MAX_LEAF_SIZE`: Max elements per leaf (default: 50)
- `OCTREE_LEVEL_THRESHOLD`: Level field threshold (default: 1.1)
- `N_PARTICLES`: Number of test particles (default: 50,000)

## Output

The test produces:

1. **Initialization timing**:
   - Mesh loading time
   - Octree build time
   - Blockwise structure build time
   - GPU upload time

2. **Per-method results**:
   - Found rate (%)
   - Accuracy (%)
   - Search time (s)
   - Throughput (p/s)
   - Number of mismatches

3. **Comparison table**:
   ```
   Method       Found           Accuracy        Time (s)     Throughput (p/s)
   ----------------------------------------------------------------------------------
   Octree       100.00%         99.95%          0.1234       405,184.3
   Blockwise    99.98%          99.92%          0.5678       88,028.7
   ```

4. **Recommendations**:
   - Which method is faster (and by how much)
   - Which method is more accurate
   - Which method to use for production

## Why This Test Matters

### Problem Context
User reported that Scenario #2 with octree was extremely slow (300× slower than expected). This test helps determine:

1. **Is octree fundamentally slow?**
   - If octree is fast here, problem is in time-stepping loop (not search)
   - If octree is slow here, octree implementation has issues

2. **Should we use blockwise instead?**
   - If blockwise is faster AND accurate, switch to it
   - If octree is faster, keep it but debug time-stepping

3. **Is accuracy the issue?**
   - If octree assigns wrong elements, particles will be lost
   - Ground truth comparison reveals if search is correct

### Impact on Scenario #2
If octree is slow or inaccurate:
- **Option 1**: Switch to blockwise search for L2 fallback
- **Option 2**: Fix octree implementation (check depth, leaf size, pruning)
- **Option 3**: Hybrid approach (blockwise for init, octree for tracking)

If octree is fast and accurate:
- Confirms bottleneck is in RK4 loop (GPU sync, export, etc.)
- Focus debugging on time-stepping, not search

## Files

- **Test script**: `test_octree_vs_blockwise_initialization.py`
- **Documentation**: `OCTREE_VS_BLOCKWISE_TEST.md` (this file)

## Dependencies

Required imports:
- `jaxtrace.gpu.mesh_loader.load_mesh_from_pvtu`
- `jaxtrace.gpu.forest.*` (blockwise structures)
- `jaxtrace.gpu.search.*` (octree and blockwise search)
- `jaxtrace.gpu.tracking.mesh_data_gpu.upload_mesh_to_gpu`
- `jax`, `numpy`, `vtk`

All imports match patterns from `test_rk4_scenario2.py` and `production_tracking_3hop_l2_octree.py`.

## Notes

1. **Reproducibility**: Uses `np.random.seed(42)` for consistent particle generation
2. **JIT warm-up**: Octree search includes warm-up run (not counted in timing)
3. **GPU synchronization**: Uses `jax.block_until_ready()` for accurate timing
4. **Mesh loading**: Uses same pattern as existing tests (no errors expected)
5. **Perturbation scale**: 1% of minimum element size ensures particles stay inside

## Troubleshooting

If test fails:
- **Mesh not found**: Check MESH_PATH points to correct location
- **Import errors**: Ensure all jaxtrace modules are in PYTHONPATH
- **CUDA errors**: Check GPU memory (mesh + octree + blockwise ~200 MB)
- **Shape errors**: Verify mesh has tetrahedral elements (4 nodes per element)
- **Zero found**: Check perturbation scale (may be too large if mesh has tiny elements)

## Next Steps After Test

Based on results:

### If Octree is Fast (>200k p/s) and Accurate (>99.9%)
→ Octree is not the problem. Debug Scenario #2 time-stepping loop:
  - Check for forced GPU syncs (like `jnp.sum()` assignment)
  - Profile RK4 stages (k1, k2, k3, k4)
  - Verify temporal batching is working
  - Check export overhead

### If Blockwise is Faster
→ Consider switching L2 fallback from octree to blockwise:
  - Modify Scenario #2 to use blockwise search for L2
  - Keep L0 (cached) + L1 (3-hop) as is
  - Measure end-to-end performance improvement

### If Both are Slow (<50k p/s)
→ General GPU/search issue:
  - Check GPU utilization during search
  - Profile with JAX profiler
  - Verify JIT compilation is working
  - Check for CPU-GPU transfer overhead

### If Accuracy is Low (<99%)
→ Search correctness issue:
  - Check point-in-tet implementation
  - Verify coordinate systems match
  - Inspect mismatched particles (near boundaries?)
  - Adjust tolerance in point-in-tet checks
