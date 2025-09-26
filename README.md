# JAXTrace

🚀 **Memory-optimized Lagrangian particle tracking with JAX acceleration**

JAXTrace is a high-performance Python package for particle tracking in fluid flows, featuring advanced memory management strategies, JAX acceleration, and comprehensive analysis tools. Designed for computational fluid dynamics applications with large-scale particle simulations.

## ✨ Key Features & Strengths

### 🔥 **JAX-Powered Performance**
- **Optional JAX acceleration** with automatic NumPy fallbacks
- **Vectorized computations** for maximum throughput
- **JIT compilation** for optimized particle integration - functions compiled once, executed at C speed
- **GPU support** for large-scale simulations with seamless CPU/GPU computation switching
- **Automatic differentiation** - built-in gradient computation capabilities

### 🧠 **Advanced Memory Management**
- **Chunked processing** for datasets larger than available RAM
- **Adaptive batch sizing** with out-of-memory recovery - automatically adjusts based on available memory
- **Memory monitoring** with automatic optimization - real-time tracking prevents crashes
- **Configurable memory limits** (default: 6GB)
- **Garbage collection** - automatic cleanup of intermediate results

### 🌊 **Comprehensive Flow Physics**
- **Time-dependent velocity fields** with temporal interpolation
- **Multiple integrator schemes** (Euler, RK2, RK4)
- **Advanced boundary conditions** (periodic, reflective, inlet/outlet)
- **Grid-preserving particle replacement** - maintains spatial structure at inlet boundaries
- **Continuous flow analysis** - seamless particle replacement for steady-state studies

### 📊 **Rich Density Analysis**
- **Kernel Density Estimation (KDE)** with Scott/Silverman bandwidth rules
- **Smoothed Particle Hydrodynamics (SPH)** with multiple kernel types
- **2D/3D density visualization** with automatic slicing
- **Efficient neighbor search** using hash grids
- **Multi-scale density analysis** - from local SPH to global KDE analysis

### 📈 **Professional Visualization & I/O**
- **Static plots** with matplotlib integration
- **Interactive visualization** via Plotly and PyVista
- **VTK export** for ParaView/VisIt compatibility - industry-standard format support
- **Trajectory animation** with MP4/AVI export
- **Statistical validation** - built-in trajectory quality metrics

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM recommended (configurable)
- Optional: CUDA-capable GPU for acceleration

### Core Dependencies
```
numpy>=1.20.0
matplotlib>=3.3.0
vtk>=9.0.0
scipy>=1.7.0
jax>=0.4.0
jaxlib>=0.4.0
```

## 🔧 Installation

### 1. Install Dependencies
```bash
# Install from requirements file
pip install -r requirements.txt

# For GPU acceleration (NVIDIA CUDA)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 2. Install JAXTrace
```bash
# Development installation (recommended)
pip install -e .

# Or standard installation
pip install .
```

### 3. Verify Installation
```python
import jaxtrace as jt
print(f"JAXTrace {jt.__version__}")
print(f"JAX available: {jt.JAX_AVAILABLE}")
jt.get_system_info()  # Display system capabilities
```

## 🚀 Quick Start

### Basic Particle Tracking
```python
import numpy as np
import jaxtrace as jt

# Load velocity field data
field = jt.io.open_dataset("path/to/velocity_*.vtk").load_time_series()
ts_field = jt.TimeSeriesField(
    data=field["velocity_data"],
    times=field["times"],
    positions=field["positions"]
)

# Configure particle tracking
tracker = jt.create_tracker(
    integrator_name="rk4",
    field=ts_field,
    boundary_condition=jt.periodic_boundary(domain_bounds),
    max_memory_gb=6.0,
    use_jax_jit=True
)

# Generate initial particle positions
initial_positions = jt.random_seeds(
    n_particles=10000,
    domain_bounds=((0,0,0), (1,1,1)),
    rng_seed=42
)

# Track particles
trajectory = tracker.track_particles(
    initial_positions=initial_positions,
    time_span=(0.0, 10.0),
    dt=0.01
)

print(f"Tracked {trajectory.N} particles for {trajectory.T} timesteps")
```

### Advanced Inlet/Outlet Boundaries
```python
# Create continuous inlet boundary with grid preservation
inlet_boundary = jt.continuous_inlet_boundary_factory(
    inlet_position=0.0,
    outlet_position=1.0,
    flow_axis="x",
    domain_bounds=domain_bounds,
    concentrations={"x": 50, "y": 20, "z": 20}  # Grid resolution
)

tracker = jt.create_tracker(
    integrator_name="rk4",
    field=ts_field,
    boundary_condition=inlet_boundary
)
```

### Density Analysis
```python
# Kernel Density Estimation
kde = jt.KDEEstimator(
    positions=trajectory.positions[-1],  # Final positions
    bandwidth_rule="scott",
    normalize=True
)
density_grid = kde.evaluate_on_grid(grid_bounds, resolution=100)

# SPH Density Estimation
sph = jt.SPHDensityEstimator(
    positions=trajectory.positions[-1],
    smoothing_length=0.1,
    kernel_type="cubic_spline"
)
sph_density = sph.evaluate(query_points)
```

## 📖 Documentation

### Core Modules
- **`jaxtrace.io`**: VTK/HDF5 data loading with memory optimization
- **`jaxtrace.fields`**: Temporal field interpolation and periodic wrapping
- **`jaxtrace.tracking`**: Particle integration with boundary conditions
- **`jaxtrace.density`**: KDE and SPH density estimation
- **`jaxtrace.visualization`**: Static and interactive plotting tools

### Examples
- **`example_workflow_clean.py`**: Complete workflow with inlet/outlet boundaries
- **`example_workflow_minimal.py`**: Basic tracking example
- **`example_workflow2.py`**: Advanced features demonstration

## 🤝 Contributing

JAXTrace is developed for high-performance particle tracking in fluid dynamics. Contributions welcome for:
- New integrator schemes
- Advanced boundary conditions
- GPU optimization
- Visualization enhancements

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🔬 Citation

If you use JAXTrace in your research, please cite:
```
JAXTrace: Memory-optimized Lagrangian particle tracking with JAX acceleration
https://github.com/ARHashemi/JAXTrace
```

---

*Built with ❤️ for academic research*