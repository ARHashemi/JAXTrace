# GPU-Native Particle Tracking Examples

This directory contains examples demonstrating the GPU-native particle tracking implementation (Phases 0-2).

## Files

### `phase_0_1_2_demo.ipynb`

Comprehensive Jupyter notebook demonstrating all implemented phases:

**What it shows:**
1. **Phase 0** - Forest grid creation and visualization
2. **Phase 1** - CPU-based element search with statistics
3. **Phase 2** - GPU-accelerated search with JAX
4. Performance comparisons (CPU vs GPU)
5. Cache hit rate validation
6. Visual izations of results

**To run:**
```bash
cd /home/arhashemi/Workspace/welding/JAXTrace
source .venv/bin/activate
jupyter notebook examples/gpu/phase_0_1_2_demo.ipynb
```

**Requirements:**
- ThreadedA mesh data at `/home/arhashemi/Workspace/welding/Edgar/ThreadedA/post/0eule`
- JAX with GPU support (optional, will fall back to CPU)
- matplotlib for visualizations

**Expected runtime:**
- ~2-3 minutes total
- Mesh loading: ~10s
- Element-to-block assignment: ~2-3s
- Element neighbor extraction: ~30-45s
- CPU search (1000 particles): ~0.5-2s
- GPU search (with JIT): ~2-3s first call, ~0.1-0.5s subsequent

**Expected results:**
- ✅ 32 blocks created (4×4×2 grid)
- ✅ ~110K elements per block (good load balance)
- ✅ 85-95% Level 0 cache hit rate (validates implementation!)
- ✅ Exact match between CPU and GPU results

### `gpu_forest.py`

Simple Python script showing Phase 0 functionality (forest grid creation and visualization).

**To run:**
```bash
source .venv/bin/activate
python examples/gpu/gpu_forest.py
```

## Troubleshooting

### Import Errors

If you get import errors:
```python
from jaxtrace.io import open_dataset  # ✅ Correct
# NOT: from jaxtrace.io import read_pvtu  # ❌ Doesn't exist
```

### Mesh Path

Update the mesh path in the notebook if your ThreadedA data is elsewhere:
```python
mesh_path = Path("/path/to/your/ThreadedA/post/0eule")
```

### GPU Not Available

The code will automatically fall back to CPU if GPU is not available. You'll see a warning but everything will still work (just slower for large particle counts).

## What to Look For

### Forest Grid Visualization
- Should show 32 blocks arranged in 4×4×2 grid
- Blocks should cover the domain bounds evenly
- Neighbor relationships should be correct (6-face connectivity)

### Block Occupancy
- Histogram should be relatively narrow (good load balance)
- 3D scatter should show relatively uniform distribution
- Imbalance factor should be < 2× (much better than VTK's 2.65×)

### Search Statistics
- **First search (no cache)**: High Level 2 hits (block search)
- **Second search (with cache)**: 85-95% Level 0 hits (cached element)
- This validates the caching strategy!

### CPU vs GPU
- For 1000 particles: GPU may be slower due to transfer overhead
- Break-even point: ~5K-10K particles
- For 10K+ particles: GPU should be significantly faster

## Next Steps

After running this notebook successfully:

1. **Increase particle count** - Try 10K or 100K particles to see GPU benefits
2. **Different seed locations** - Test with particles in different regions
3. **Phase 3** - Add ghost regions for seamless block transitions
4. **Phase 4** - Add time integration and field interpolation

## Support

If you encounter issues:
1. Check that all imports work (run first cell)
2. Verify ThreadedA mesh path is correct
3. Check that JAX is installed (`pip install jax jaxlib`)
4. Look at Phase 0-2 documentation in `docs/gpu/`
