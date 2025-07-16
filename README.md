# JAXTrace: Memory-Optimized Particle Tracking with JAX

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![JAX](https://img.shields.io/badge/JAX-enabled-green.svg)](https://jax.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

JAXTrace is a high-performance, memory-optimized particle tracking library for computational fluid dynamics (CFD) applications. Built with JAX for GPU acceleration and designed for handling large VTK datasets efficiently.

## ✨ Key Features

- **🚀 High Performance**: GPU-accelerated with JAX for maximum speed
- **💾 Memory Optimized**: Intelligent caching and streaming for large datasets
- **🔧 Flexible Integration**: Multiple integration methods (Euler, RK2, RK4)
- **📐 Advanced Interpolation**: Finite element and nearest neighbor methods
- **📊 Rich Visualization**: 3D plots, cross-sections, and density estimation
- **⚙️ Auto-Optimization**: Automatic parameter tuning for system constraints
- **📁 VTK Support**: Native support for VTK/ParaView file formats

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install jaxtrace

# With GPU support
pip install jaxtrace[gpu]

# With all optional dependencies
pip install jaxtrace[all]
```

### Basic Usage

```python
import jaxtrace
from jaxtrace import VTKReader, ParticleTracker, ParticleVisualizer

# Load VTK data
reader = VTKReader("simulation_*.pvtu", max_time_steps=50)

# Initialize tracker
tracker = ParticleTracker(
    vtk_reader=reader,
    interpolation_method='finite_element',
    integration_method='rk4'
)

# Create initial particles
initial_positions = tracker.create_particle_grid(
    resolution=(30, 30, 30),
    bounds_padding=0.1
)

# Track particles
final_positions = tracker.track_particles(
    initial_positions=initial_positions,
    dt=0.01,
    n_steps=1000
)

# Visualize results
visualizer = ParticleVisualizer(
    final_positions=final_positions,
    initial_positions=initial_positions
)
visualizer.plot_3d_positions()
visualizer.plot_density(method='jax_kde')
```

## 📖 Documentation

### Core Components

#### 1. VTKReader
Memory-optimized VTK file reader with intelligent caching:

```python
reader = VTKReader(
    file_pattern="simulation_*.pvtu",
    max_time_steps=100,           # Limit memory usage
    cache_size_limit=5,           # Keep 5 timesteps in memory
    velocity_field_name="velocity" # Auto-detected if None
)

# Get grid information
grid_info = reader.get_grid_info()
print(f"Grid points: {grid_info['n_points']}")
print(f"Time steps: {grid_info['n_timesteps']}")
```

#### 2. ParticleTracker
High-performance particle tracking with multiple integration methods:

```python
tracker = ParticleTracker(
    vtk_reader=reader,
    max_gpu_memory_gb=8.0,
    k_neighbors=8,                    # For finite element interpolation
    shape_function='linear',          # 'linear', 'quadratic', 'cubic', 'gaussian'
    interpolation_method='finite_element'  # or 'nearest_neighbor'
)

# Integration methods: 'euler', 'rk2', 'rk4'
final_positions = tracker.track_particles(
    initial_positions=initial_positions,
    dt=0.01,
    n_steps=1000,
    integration_method='rk4',
    save_trajectories=True  # Store full particle paths
)
```

#### 3. ParticleVisualizer
Comprehensive visualization with advanced density estimation:

```python
visualizer = ParticleVisualizer(
    final_positions=final_positions,
    initial_positions=initial_positions,
    trajectories=trajectories  # Optional
)

# 3D visualization
visualizer.plot_3d_positions(show_trajectories=True)

# Cross-section analysis
visualizer.plot_cross_sections(plane='xy', position=0.0)

# Advanced density estimation
visualizer.plot_density(
    method='jax_kde',        # 'jax_kde', 'sph', 'seaborn'
    plane='xy',
    grid_resolution=200,
    bandwidth=0.3
)

# Interactive plots (requires Plotly)
visualizer.plot_interactive_3d()
```

### Configuration Presets

JAXTrace provides optimized configurations for different scenarios:

```python
from jaxtrace.utils import get_memory_config

# Available presets
configs = {
    'low_memory': get_memory_config('low_memory'),      # 4GB RAM, basic methods
    'medium_memory': get_memory_config('medium_memory'), # 8GB RAM, balanced
    'high_memory': get_memory_config('high_memory'),    # 16GB RAM, advanced methods
    'high_accuracy': get_memory_config('high_accuracy') # 12GB RAM, maximum accuracy
}

# Use preset configuration
config = get_memory_config('medium_memory')
final_positions = tracker.track_particles(**config)
```

### Memory Optimization

For large datasets, JAXTrace provides several optimization strategies:

```python
# Spatial and temporal subsampling
final_positions = tracker.track_particles_with_subsampling(
    initial_positions=large_particle_set,
    dt=0.01,
    n_steps=2000,
    spatial_subsample_factor=2,  # Use every 2nd particle
    temporal_subsample_factor=2  # Use every 2nd timestep
)

# Automatic parameter optimization
from jaxtrace.utils import optimize_parameters_for_system

optimized_config = optimize_parameters_for_system(
    base_config=config,
    target_memory_gb=4.0,      # Target memory usage
    target_runtime_hours=2.0   # Target runtime
)
```

### Custom Particle Distributions

Create specialized initial particle distributions:

```python
from jaxtrace.utils import create_custom_particle_distribution

# Different distribution types
box_bounds = ((0, 10), (0, 5), (0, 8))  # (xmin,xmax), (ymin,ymax), (zmin,zmax)

distributions = {
    'uniform': create_custom_particle_distribution(
        box_bounds, 'uniform', n_particles=10000
    ),
    'gaussian': create_custom_particle_distribution(
        box_bounds, 'gaussian', n_particles=10000
    ),
    'clustered': create_custom_particle_distribution(
        box_bounds, 'clustered', n_particles=10000
    )
}
```

## 🔧 Advanced Features

### Density Estimation Methods

JAXTrace supports multiple density estimation methods:

1. **JAX KDE**: GPU-accelerated Gaussian Kernel Density Estimation
2. **SPH**: Smoothed Particle Hydrodynamics density estimation
3. **Seaborn**: Traditional scipy-based estimation (fallback)

```python
# JAX KDE with custom parameters
visualizer.plot_density(
    method='jax_kde',
    bandwidth=0.2,
    bandwidth_method='scott',  # or 'silverman'
    grid_resolution=200
)

# SPH density estimation
visualizer.plot_density(
    method='sph',
    kernel_type='cubic_spline',  # 'gaussian', 'wendland'
    adaptive=True,
    n_neighbors=32
)
```

### Performance Monitoring

```python
from jaxtrace.utils import monitor_memory_usage, create_progress_callback

# Monitor memory usage
@monitor_memory_usage
def my_tracking_function():
    return tracker.track_particles(...)

# Progress callbacks
progress_callback = create_progress_callback(
    update_frequency=50,
    show_memory=True,
    show_time=True
)

final_positions = tracker.track_particles(
    ...,
    progress_callback=progress_callback
)
```

### Benchmarking

Compare performance of different methods:

```python
from jaxtrace.utils import benchmark_interpolation_methods

# Benchmark interpolation methods
results = benchmark_interpolation_methods(
    n_particles=10000,
    n_evaluations=1000,
    methods=['nearest_neighbor', 'finite_element']
)

print(f"Finite Element: {results['finite_element']['time_per_evaluation_ms']:.2f} ms/eval")
print(f"Nearest Neighbor: {results['nearest_neighbor']['time_per_evaluation_ms']:.2f} ms/eval")
```

## 📊 Examples

### Complete Workflow Example

```python
import jaxtrace
from jaxtrace import VTKReader, ParticleTracker, ParticleVisualizer
from jaxtrace.utils import get_memory_config, create_progress_callback

# 1. Load VTK data
reader = VTKReader("cfd_simulation_*.pvtu", max_time_steps=100)

# 2. Initialize tracker with optimized settings
config = get_memory_config('medium_memory')
tracker = ParticleTracker(reader, **config)

# 3. Create initial particles
initial_positions = tracker.create_particle_grid(
    resolution=config['particle_resolution'],
    bounds_padding=0.1
)

# 4. Track particles with progress monitoring
progress_callback = create_progress_callback(show_memory=True)
final_positions, trajectories = tracker.track_particles(
    initial_positions=initial_positions,
    dt=0.01,
    n_steps=500,
    integration_method=config['integration_method'],
    save_trajectories=True,
    progress_callback=progress_callback
)

# 5. Comprehensive visualization
visualizer = ParticleVisualizer(
    final_positions=final_positions,
    initial_positions=initial_positions,
    trajectories=trajectories
)

# Multiple visualization types
visualizer.plot_3d_positions(show_trajectories=True)
visualizer.plot_displacement_analysis()
visualizer.plot_combined_analysis(method='jax_kde')
visualizer.plot_interactive_3d()  # If Plotly available
```

### Custom Integration Example

```python
# Custom tracking with specific parameters
tracker = ParticleTracker(
    vtk_reader=reader,
    max_gpu_memory_gb=12.0,
    k_neighbors=12,
    shape_function='cubic',
    interpolation_method='finite_element'
)

# High-accuracy tracking
final_positions = tracker.track_particles(
    initial_positions=initial_positions,
    dt=0.005,  # Smaller timestep
    n_steps=2000,
    integration_method='rk4',
    time_step_stride=1  # Use every timestep
)
```

## 📋 Requirements

### Core Dependencies
- Python >= 3.8
- NumPy >= 1.20.0
- JAX >= 0.4.0
- VTK >= 9.0.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0
- psutil >= 5.8.0

### Optional Dependencies
- **GPU Support**: `jax[cuda]`, `pynvml`
- **Interactive Visualization**: `plotly >= 5.0.0`
- **Development**: `pytest`, `black`, `flake8`, `sphinx`

## 🔧 Installation Options

```bash
# Basic CPU-only installation
pip install jaxtrace

# GPU support (CUDA)
pip install jaxtrace[gpu]

# Interactive visualization
pip install jaxtrace[interactive]

# Development tools
pip install jaxtrace[dev]

# Everything
pip install jaxtrace[all]
```

### From Source

```bash
git clone https://github.com/jaxtrace/jaxtrace.git
cd jaxtrace
pip install -e .
```

## 🚀 Performance Tips

1. **GPU Usage**: Install JAX with CUDA support for best performance
2. **Memory Management**: Use appropriate configuration presets
3. **Batch Size**: JAXTrace automatically optimizes batch sizes
4. **Subsampling**: Use spatial/temporal subsampling for large datasets
5. **Integration Method**: RK4 > RK2 > Euler (accuracy vs speed tradeoff)
6. **Interpolation**: Finite element > Nearest neighbor (accuracy vs speed)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/jaxtrace/jaxtrace.git
cd jaxtrace
pip install -e .[dev]

# Run tests
pytest tests/

# Format code
black jaxtrace/
flake8 jaxtrace/
```

## 📄 License

JAXTrace is released under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **JAX Team**: For the amazing JAX framework
- **VTK/ParaView**: For VTK file format support  
- **Scientific Python Community**: For the excellent ecosystem

## 📚 Citation

If you use JAXTrace in your research, please cite:

```bibtex
@software{jaxtrace2024,
  title={JAXTrace: Memory-Optimized Particle Tracking with JAX},
  author={JAXTrace Development Team},
  year={2024},
  url={https://github.com/jaxtrace/jaxtrace}
}
```

## 🔗 Links

- **Documentation**: https://jaxtrace.readthedocs.io/
- **GitHub**: https://github.com/jaxtrace/jaxtrace
- **PyPI**: https://pypi.org/project/jaxtrace/
- **Issues**: https://github.com/jaxtrace/jaxtrace/issues

## 📈 Roadmap

- [ ] **Multi-GPU Support**: Distributed particle tracking
- [ ] **HDF5 Integration**: Support for HDF5 datasets
- [ ] **Adaptive Time Stepping**: Automatic dt adjustment
- [ ] **Machine Learning**: Neural network-based interpolation
- [ ] **Cloud Integration**: AWS/GCP deployment support
- [ ] **Real-time Visualization**: Live tracking visualization

---

**JAXTrace** - Bringing high-performance particle tracking to everyone! 🚀