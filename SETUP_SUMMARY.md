# JAXTrace Setup Summary - Temporal Batching Ready

## ✅ Configuration Complete

The `example_workflow.py` file has been configured to use **temporal batching by default** for optimal performance with AMR data.

## Current Configuration

### Temporal Batching Settings (ENABLED)

```python
'use_temporal_batching': True,  # Using temporal batching (grid hash)
'temporal_window_size': 30,      # Process 30 velocity timesteps per window
'grid_resolution': 32,           # 32^3 grid cells for spatial indexing
```

### Data Settings

```python
'data_pattern': "/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule/featurelessAvtk_*.pvtu"
'max_timesteps_to_load': 40     # Load 40 velocity timesteps
'skip_initial_timesteps': 20    # Skip first 20 timesteps (AMR warmup)
'use_stable_mesh_only': True    # Auto-filter inconsistent meshes
```

### Particle Settings

```python
'particle_concentrations': {'x': 40, 'y': 70, 'z': 30}  # ~84,000 particles
'particle_distribution': 'uniform'
'particle_bounds_fraction': {'x': (0.1, 0.25), 'y': (0.0, 1.0), 'z': (0.0, 1.0)}
```

### Tracking Settings

```python
'n_timesteps': 2500              # Tracking timesteps
'dt': 0.0025                     # Time step size
'integrator': 'rk4'              # RK4 integration
'record_velocities': False       # Save 50% memory!
```

### Performance Settings

```python
'device': 'gpu'                  # Use GPU
'memory_limit_gb': 3.0           # GPU memory limit
```

## How to Run

### 1. Activate Virtual Environment

```bash
source .venv/bin/activate
```

### 2. Run Workflow

```bash
python example_workflow.py
```

### 3. Expected Output

```
================================================================================
CONFIGURATION SUMMARY
================================================================================
📁 Data pattern: /path/to/data/*.pvtu
⏱  Timesteps to load: 40
🔍 Auto-detect stable mesh: enabled
⏭️  Skip initial timesteps: 20
🌲 Octree: max_elements=32, max_depth=12
🎯 Particles: {'x': 40, 'y': 70, 'z': 30}, distribution=uniform
🏃 Tracking: 2500 steps, dt=0.0025, integrator=rk4
🚪 Boundary: x-axis, inlet=none, outlet=absorbing
💻 Device: gpu, memory=3.0 GB
================================================================================

================================================================================
3. VELOCITY FIELD
================================================================================
🔧 Using TEMPORAL BATCHING with Grid Hash (for AMR data)
   Window size: 30 velocity timesteps
   Grid resolution: 32^3 cells
✅ Temporal batching field created: 40 VTK files found

================================================================================
4. PARTICLE TRACKING
================================================================================
📏 Loading first timestep to determine field bounds...
   Field bounds: [...] to [...]
🎯 Generating particles...
   Domain size: X=..., Y=..., Z=...
   Grid resolution: 40 x 70 x 30 = 84000 particles
✅ Generated 84000 particles with uniform grid distribution

🚪 Boundary conditions:
   All boundaries: Reflective
   ⚠️  Note: Temporal batching currently uses reflective boundaries
   Advanced boundary conditions (continuous inlet) will be added in future release

🚀 Setting up temporal batching tracker...
   💾 Recording positions only (velocities not stored)

🏃 Running temporal batching particle tracking...
   Tracking 84000 particles for 2500 timesteps
   Time span: (0.0, 6.25), dt=0.0025
   Data dt: 0.001, Temporal window size: 30 velocity timesteps
   Progress: |████████████████████████████| 100.0%
✅ Tracking completed in XXX seconds
```

## Switching Back to Octree FEM

If you need to use octree FEM (spatial batching) instead:

```python
user_config.update({
    'use_temporal_batching': False,      # Disable temporal batching
    # Octree settings will be used instead
})
```

## Performance Expectations

### Temporal Batching (Current Setup)

- **Spatial Index Build**: ~0.1s per timestep × 30 = ~3 seconds total
- **GPU Utilization**: 80-95%
- **Memory**: ~50-100 MB for grid hash + trajectory memory
- **Best For**: AMR data, variable mesh, large datasets

### Octree FEM (Alternative)

- **Spatial Index Build**: ~10s (single octree)
- **GPU Utilization**: 15-30%
- **Memory**: ~400-800 MB for octree + trajectory memory
- **Best For**: Fixed mesh, continuous inlet, maximum accuracy

## Key Files

- ✅ `example_workflow.py` - Main workflow (temporal batching enabled)
- ✅ `config_example.py` - Configuration examples
- ✅ `test_temporal_batching.py` - Test script
- ✅ `TEMPORAL_BATCHING.md` - Full documentation
- ✅ `jaxtrace/fields/grid_hash_field.py` - Grid hash implementation
- ✅ `jaxtrace/fields/temporal_field.py` - Temporal field loader
- ✅ `jaxtrace/tracking/temporal_tracker.py` - Temporal batching tracker

## Memory Estimates

For current configuration:
- **Particles**: 84,000
- **Timesteps**: 2,500
- **Position Memory**: ~2.5 GB (84k × 2500 × 3 × 4 bytes)
- **Velocity Memory**: 0 GB (disabled)
- **Grid Hash**: ~100 MB
- **Total Estimated**: ~2.6 GB

## Troubleshooting

### If you get import errors:

```bash
# Reinstall JAXTrace in development mode
source .venv/bin/activate
pip install -e .
```

### If you get memory errors:

```python
# Reduce particles or timesteps
user_config.update({
    'particle_concentrations': {'x': 30, 'y': 50, 'z': 20},  # Fewer particles
    'n_timesteps': 1000,                                     # Fewer timesteps
})
```

### If you get GPU errors:

```python
# Switch to CPU
user_config.update({
    'device': 'cpu',
})
```

## Next Steps

1. **Test Run**: `python test_temporal_batching.py`
2. **Full Run**: `python example_workflow.py`
3. **Customize**: Edit `user_config` in `example_workflow.py`
4. **Read Docs**: See `TEMPORAL_BATCHING.md` for details

---

**Status**: ✅ Ready to run with temporal batching!

**Date**: 2025-10-08
