# JAXTrace Usage Guide

## Running JAXTrace

JAXTrace provides several ways to run particle tracking workflows:

### 1. Quick Test (Recommended for First Run)

Test that JAXTrace is working correctly with a minimal synthetic field:

```bash
# Using run.py
python run.py --test

# Using module execution
python -m jaxtrace --test

# Direct execution
python -m tests.test_quick
```

**Output:**
- Creates `output_test/` directory
- Generates `test_trajectories.png`
- Runs in ~2-5 seconds
- Uses 25 particles, 100 timesteps

---

### 2. Main Workflow with Default Configuration

Run the full particle tracking workflow with default settings:

```bash
# Using run.py (recommended)
python run.py

# Using module execution
python -m jaxtrace

# Direct execution
python example_workflow.py
```

**Default Configuration:**
- Data: Looks for VTK files at configured path
- Particles: 60×50×15 uniform grid
- Tracking: 2000 timesteps, dt=0.0025
- Boundary: Continuous inlet, absorbing outlet
- Output: `output/` directory

---

### 3. Custom Configuration

Create a configuration file and run with custom parameters:

**Step 1: Create `myconfig.py`**
```python
# myconfig.py
config = {
    'data_pattern': "/path/to/data_*.pvtu",
    'particle_concentrations': {'x': 40, 'y': 30, 'z': 10},
    'particle_distribution': 'gaussian',
    'n_timesteps': 1000,
    'dt': 0.005,
    'boundary_inlet': 'continuous',
    'boundary_outlet': 'absorbing',
}
```

**Step 2: Run with config**
```bash
python run.py --config myconfig.py
```

---

### 4. Version and Help

```bash
# Check version
python -m jaxtrace --version

# Get help
python run.py --help
```

---

## Configuration Options

### Quick Configuration Examples

#### Test Run (Fast)
```python
config = {
    'particle_concentrations': {'x': 20, 'y': 10, 'z': 5},
    'n_timesteps': 500,
    'max_timesteps_to_load': 10,
}
```

#### High-Resolution Run
```python
config = {
    'particle_concentrations': {'x': 100, 'y': 80, 'z': 20},
    'n_timesteps': 5000,
    'dt': 0.001,
}
```

#### Gaussian Distribution
```python
config = {
    'particle_distribution': 'gaussian',
    'gaussian_std': {'x': 0.15, 'y': 0.15, 'z': 0.2},
}
```

#### No Inlet, Decay Mode
```python
config = {
    'boundary_inlet': 'none',
    'boundary_outlet': 'absorbing',
}
```

#### Adaptive Mesh Refinement (AMR) Data
```python
config = {
    'use_stable_mesh_only': True,   # Auto-detect stable mesh size
    'skip_initial_timesteps': 0,    # Or manually skip first N timesteps
}
```

#### Use Time Interval from VTK Files
```python
config = {
    'use_data_dt': True,  # Override dt with timestamps from VTK files
    'dt': 0.0025,         # Fallback value if data dt cannot be determined
}
```

#### Custom Density Estimation
```python
# KDE only with custom bandwidth
config = {
    'density_methods': ['kde'],
    'kde_bandwidth': 0.15,
    'kde_bandwidth_rule': 'silverman'
}

# Adaptive SPH
config = {
    'density_methods': ['sph'],
    'sph_adaptive': True,
    'sph_n_neighbors': 50,
    'sph_kernel_type': 'wendland'
}

# Skip density analysis for faster runs
config = {
    'perform_density_analysis': False
}
```

### All Available Parameters

See [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) for complete parameter reference.

---

## File Structure

### Input Files
- **VTK/PVTU data**: Time series velocity field data
- **Configuration files**: Python files with `config` dictionary

### Output Files
```
output/
├── particles_final.png          # Final particle positions
├── trajectories_2d.png          # 2D trajectory plot
├── density_analysis.png         # KDE/SPH density
├── density_yz_slice_x_*.png    # YZ density contour
├── trajectory.vtp               # VTK export (single file)
├── trajectory_series_*.vtp      # VTK time series
├── summary_report.md            # Analysis summary
└── enhanced_report.md           # Detailed report
```

---

## Troubleshooting

### Common Issues

**1. Import Error: No module named 'jaxtrace'**
```bash
# Install in development mode
pip install -e .
```

**2. CUDA/GPU errors**
```bash
# Use CPU device
config = {'device': 'cpu'}
```

**3. Out of memory**
```bash
# Reduce batch size or particle count
config = {
    'batch_size': 500,
    'particle_concentrations': {'x': 30, 'y': 20, 'z': 10},
}
```

**4. No VTK files found**
```bash
# Check data pattern
config = {'data_pattern': "/absolute/path/to/data_*.pvtu"}
```

**5. Test imports before full run**
```bash
python test_quick.py
```

---

## API Usage (Programmatic)

For custom workflows, use JAXTrace as a library:

```python
import jaxtrace as jt
from jaxtrace.tracking import create_tracker
from jaxtrace.tracking.boundary import reflective_boundary

# Configure
jt.configure(dtype="float32", device="gpu", memory_limit_gb=4.0)

# Load field
field = jt.TimeSeriesField(data=..., times=..., positions=...)

# Create tracker
tracker = create_tracker(
    integrator_name="rk4",
    field=field,
    boundary_condition=reflective_boundary(bounds),
    batch_size=1000,
)

# Track particles
trajectory = tracker.track_particles(
    initial_positions=seeds,
    time_span=(0.0, 5.0),
    n_timesteps=1000,
)

# Analyze
from jaxtrace.density import KDEEstimator
kde = KDEEstimator(positions=trajectory.positions[-1])
density = kde.evaluate(query_points)
```

See [example_workflow.py](example_workflow.py) for complete examples.

---

## Performance Tips

1. **Start with quick test**: `python run.py --test`
2. **Use GPU if available**: `config = {'device': 'gpu'}`
3. **Optimize batch size**: Balance between memory and speed
4. **Reduce data loading**: `max_timesteps_to_load=20` for testing
5. **Choose appropriate integrator**: RK4 (accurate) vs Euler (fast)

---

## Next Steps

- See [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) for detailed parameter tuning
- See [QUICK_CONFIG_REFERENCE.md](QUICK_CONFIG_REFERENCE.md) for copy-paste configs
- See [example_workflow.py](example_workflow.py) for implementation examples
- See [README_OCTREE_FEM.md](README_OCTREE_FEM.md) for octree FEM interpolation details
