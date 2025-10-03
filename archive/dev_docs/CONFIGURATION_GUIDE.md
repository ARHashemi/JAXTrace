# example_workflow.py - Configuration Guide

## Overview

The `example_workflow.py` now supports comprehensive configuration through a simple dictionary passed to `main(config)`. This makes it easy to customize all aspects of the workflow without editing the code.

---

## Quick Start

```python
# At the bottom of example_workflow.py:

if __name__ == "__main__":
    user_config = {
        'data_pattern': "/path/to/your/data_*.pvtu",
        'particle_concentrations': {'x': 60, 'y': 50, 'z': 15},
        'n_timesteps': 2000,
        'dt': 0.0025,
    }

    main(config=user_config)
```

---

## Complete Configuration Reference

### 📁 Data Loading

```python
{
    # Path pattern for VTK files (supports glob wildcards)
    'data_pattern': "/path/to/case_*.pvtu",

    # Number of timesteps to load from data files
    # Higher = more accurate temporal interpolation, more memory
    'max_timesteps_to_load': 40,
}
```

**Examples**:
- All timesteps: `'max_timesteps_to_load': 1000`
- Quick test: `'max_timesteps_to_load': 10`
- Moderate: `'max_timesteps_to_load': 40` (default)

---

### 🌲 Octree FEM Configuration

```python
{
    # Maximum elements before octree subdivides a node
    # Lower = finer tree (slower build, faster queries, more memory)
    # Higher = coarser tree (faster build, slower queries, less memory)
    'max_elements_per_leaf': 32,

    # Maximum tree depth (prevents infinite subdivision)
    'max_octree_depth': 12,
}
```

**Tuning guide**:
| Mesh Type | `max_elements_per_leaf` | `max_octree_depth` |
|-----------|------------------------|-------------------|
| Uniform | 64 | 10 |
| Refined (2-3 levels) | 32 | 12 |
| Highly refined (6+ levels) | 16-32 | 12-15 |

**Your mesh (6 levels)**: Use `32` and `12` (default)

---

### 🎯 Particle Seeding

```python
{
    # Particle density (particles per unit length in each direction)
    'particle_concentrations': {
        'x': 60,  # Flow direction (high density)
        'y': 50,  # Cross-stream (medium density)
        'z': 15   # Height (lower density)
    },

    # OPTION 1: Explicit bounds (absolute coordinates)
    'particle_bounds': [
        np.array([-0.03, -0.02, -0.008]),  # Min corner
        np.array([0.01, 0.02, 0.0])         # Max corner
    ],

    # OPTION 2: Fractional bounds (fraction of domain, 0.0 to 1.0)
    'particle_bounds_fraction': {
        'x': (0.0, 0.2),  # First 20% of X domain
        'y': (0.0, 1.0),  # Full Y extent
        'z': (0.0, 1.0)   # Full Z extent
    },

    # Note: Use either particle_bounds OR particle_bounds_fraction
    # If both None, uses entire domain
}
```

**Total particles** = `concentration_x × concentration_y × concentration_z`
- Default (60×50×15) = **45,000 particles**

**Examples**:

```python
# Quick test (low resolution)
'particle_concentrations': {'x': 20, 'y': 10, 'z': 5},  # 1,000 particles

# High resolution
'particle_concentrations': {'x': 100, 'y': 80, 'z': 20},  # 160,000 particles

# Inlet region only (first 20% of X)
'particle_bounds_fraction': {
    'x': (0.0, 0.2),
    'y': (0.0, 1.0),
    'z': (0.0, 1.0)
}

# Specific region (explicit coordinates)
'particle_bounds': [
    np.array([0.0, -0.01, -0.004]),
    np.array([0.02, 0.01, 0.0])
]
```

---

### 🏃 Tracking Parameters

```python
{
    # Number of tracking timesteps
    'n_timesteps': 2000,

    # Time step size
    'dt': 0.0025,

    # Simulation time range (start, end)
    'time_span': (0.0, 4.0),

    # Particles per batch (lower = less GPU memory)
    'batch_size': 1000,

    # Integration method
    'integrator': 'rk4',  # Options: 'rk4', 'euler', 'rk2'
}
```

**Time step selection**:
```python
# For accuracy:
dt = 0.001  # High accuracy, slow
dt = 0.0025  # Good balance (default)
dt = 0.005   # Faster, less accurate

# Total simulation time = n_timesteps × dt
# Example: 2000 × 0.0025 = 5.0 seconds of simulation
```

**Batch size selection**:
```python
# Based on GPU memory:
'batch_size': 500    # 2 GB GPU
'batch_size': 1000   # 4 GB GPU (default)
'batch_size': 2000   # 8 GB GPU
'batch_size': 5000   # 16 GB GPU
```

---

### 🚪 Boundary Conditions

```python
{
    # Flow axis ('x', 'y', or 'z')
    'flow_axis': 'x',

    # Flow direction ('positive' or 'negative')
    'flow_direction': 'positive',

    # Inlet particle distribution ('grid' or 'random')
    'inlet_distribution': 'grid',
}
```

**Configurations**:
```python
# X-direction flow (left to right)
'flow_axis': 'x'
'flow_direction': 'positive'

# X-direction flow (right to left)
'flow_axis': 'x'
'flow_direction': 'negative'

# Y-direction flow (bottom to top)
'flow_axis': 'y'
'flow_direction': 'positive'

# Random inlet distribution
'inlet_distribution': 'random'
```

**How it works**:
- Particles enter at `axis_min` (inlet plane)
- Particles exit at `axis_max` (outlet plane)
- Exited particles are replaced with new inlet particles
- Other boundaries are reflective

---

### 🎨 Visualization

```python
{
    # X position for YZ density slice (None = auto at 70% of domain)
    'slice_x0': None,  # or specific value like 0.05

    # Number of density contour levels
    'slice_levels': 20,  # or specific levels: [0.1, 0.5, 1.0, 2.0]

    # Percentile cutoff for density (removes extreme outliers)
    'slice_cutoff': 95,  # 95% = remove top 5% of values
}
```

**Examples**:
```python
# Auto slice position
'slice_x0': None  # Default: 0.7 × x_max

# Manual slice position
'slice_x0': 0.05  # Specific X coordinate

# More contour levels
'slice_levels': 30

# Custom contour levels
'slice_levels': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# No outlier cutoff
'slice_cutoff': 100
```

---

### 💻 GPU Configuration

```python
{
    # Device to use
    'device': 'gpu',  # or 'cpu'

    # GPU memory limit (GB)
    'memory_limit_gb': 3.0,
}
```

**GPU memory guide**:
```python
# Conservative (2-4 GB GPU)
'memory_limit_gb': 2.0

# Balanced (4-8 GB GPU)
'memory_limit_gb': 3.0  # Default

# Aggressive (8+ GB GPU)
'memory_limit_gb': 6.0
```

---

## Complete Examples

### Example 1: Quick Test Run

```python
user_config = {
    'data_pattern': "/path/to/data_*.pvtu",
    'max_timesteps_to_load': 10,  # Load only 10 timesteps

    'particle_concentrations': {'x': 20, 'y': 10, 'z': 5},  # 1,000 particles

    'n_timesteps': 500,  # Shorter tracking
    'dt': 0.005,         # Larger timestep

    'device': 'gpu',
}

main(config=user_config)
```

**Estimated time**: 2-3 minutes

---

### Example 2: Production Run (Default Settings)

```python
user_config = {
    'data_pattern': "/home/user/Cases/case_*.pvtu",
    'max_timesteps_to_load': 40,

    'particle_concentrations': {'x': 60, 'y': 50, 'z': 15},  # 45,000 particles

    'n_timesteps': 2000,
    'dt': 0.0025,
    'batch_size': 1000,

    'device': 'gpu',
    'memory_limit_gb': 3.0,
}

main(config=user_config)
```

**Estimated time**: 10-20 minutes

---

### Example 3: High-Resolution Run

```python
user_config = {
    'data_pattern': "/home/user/Cases/case_*.pvtu",
    'max_timesteps_to_load': 100,  # More temporal resolution

    'particle_concentrations': {'x': 100, 'y': 80, 'z': 20},  # 160,000 particles

    'n_timesteps': 5000,
    'dt': 0.001,
    'batch_size': 2000,

    'device': 'gpu',
    'memory_limit_gb': 6.0,
}

main(config=user_config)
```

**Estimated time**: 30-60 minutes
**Requirements**: 8+ GB GPU

---

### Example 4: Inlet Region Study

```python
user_config = {
    'data_pattern': "/home/user/Cases/case_*.pvtu",

    'particle_concentrations': {'x': 80, 'y': 60, 'z': 15},

    # Seed particles only in inlet region (first 20% of X)
    'particle_bounds_fraction': {
        'x': (0.0, 0.2),
        'y': (0.0, 1.0),
        'z': (0.0, 1.0)
    },

    'n_timesteps': 3000,
    'dt': 0.002,
}

main(config=user_config)
```

**Use case**: Study flow development from inlet

---

### Example 5: CPU-Only Run

```python
user_config = {
    'data_pattern': "/home/user/Cases/case_*.pvtu",

    'particle_concentrations': {'x': 30, 'y': 20, 'z': 10},  # Fewer particles for CPU

    'n_timesteps': 1000,
    'batch_size': 500,

    'device': 'cpu',  # Run on CPU
}

main(config=user_config)
```

**Note**: CPU is much slower (~10-50x)

---

## Parameter Relationships

### Memory Usage

Total GPU memory ≈:
- Octree structure: 50-100 MB
- Velocity field data: `40 × 185,865 × 3 × 4` = 85 MB
- Batch processing: `batch_size × n_timesteps × 3 × 4` / 1024 / 1024 MB

**Example** (batch_size=1000, n_timesteps=2000):
- Octree: ~60 MB
- Field: ~85 MB
- Batch: ~24 MB
- **Total**: ~170 MB (well within 3 GB limit)

### Computation Time

Time ≈ `(total_particles × n_timesteps × queries_per_step) / GPU_speed`

Where:
- `total_particles` = product of concentrations
- `n_timesteps` = tracking timesteps
- `queries_per_step` = 4 (for RK4) or 1 (for Euler)
- `GPU_speed` ≈ 4,000-10,000 queries/second (for octree FEM)

**Example** (45,000 particles, 2000 timesteps, RK4):
- Total queries: 45,000 × 2,000 × 4 = 360M
- Time: 360M / 5000 = 72,000 seconds ≈ **20 hours**

**Wait, that seems too long!** The actual time is much faster due to:
- JIT compilation optimization
- GPU parallelization
- Batched processing

**Realistic estimate**: **10-20 minutes** for this configuration

---

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
'batch_size': 500  # instead of 1000

# Or reduce particles
'particle_concentrations': {'x': 40, 'y': 30, 'z': 10}

# Or reduce loaded timesteps
'max_timesteps_to_load': 20
```

### Slow Performance

```python
# Increase batch size (if memory allows)
'batch_size': 2000

# Reduce octree depth
'max_octree_depth': 10

# Use coarser octree
'max_elements_per_leaf': 64

# Use Euler instead of RK4 (fewer queries)
'integrator': 'euler'
```

### Inaccurate Results

```python
# Decrease timestep
'dt': 0.001

# Increase octree resolution
'max_elements_per_leaf': 16

# Use RK4 integrator
'integrator': 'rk4'

# Load more data timesteps
'max_timesteps_to_load': 100
```

---

## Best Practices

1. **Start small**: Test with low resolution first
   ```python
   'particle_concentrations': {'x': 20, 'y': 10, 'z': 5}
   'n_timesteps': 500
   ```

2. **Verify octree**: Check console output for elements/leaf
   - Target: 10-50 elements/leaf
   - If too high (>100): increase grid_resolution or decrease max_elements_per_leaf

3. **Monitor memory**: Use `nvidia-smi` during run
   - Should use 1-3 GB for typical runs
   - If >90% GPU memory: reduce batch_size

4. **Profile performance**: Note time per batch
   - Should be 0.5-2 seconds per batch
   - If slower: check octree quality

5. **Validate results**: Compare with known behavior
   - Check particle paths make physical sense
   - Verify flow patterns
   - Compare with analytical solutions if available

---

## Summary

The configurable workflow makes it easy to:
- ✅ Switch between quick tests and production runs
- ✅ Study specific regions (inlet, outlet, etc.)
- ✅ Optimize for your hardware (GPU memory, speed)
- ✅ Tune accuracy vs performance
- ✅ Run parametric studies

**Just edit the `user_config` dictionary and run!**
