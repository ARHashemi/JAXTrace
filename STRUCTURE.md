# JAXTrace Repository Structure

This document describes the repository structure and organization.

## 📁 Directory Structure

```
JAXTrace/
├── jaxtrace/              # Core package
│   ├── __init__.py       # Package initialization and API exports
│   ├── __main__.py       # Module execution entrypoint (python -m jaxtrace)
│   ├── fields/           # Velocity field interpolation
│   ├── tracking/         # Particle tracking and integration
│   ├── density/          # KDE and SPH density estimation
│   ├── visualization/    # Plotting and visualization
│   ├── io/              # Data loading and export
│   └── utils/           # Utilities and helpers
│
├── tests/               # Test suite
│   ├── __init__.py
│   ├── smoke_test.py    # Quick functionality test
│   ├── structure_test.py # Package structure validation
│   └── test_quick.py    # Fast debugging test (2-5 seconds)
│
├── run.py               # Main CLI entrypoint
├── config_example.py    # Configuration template
│
├── example_workflow.py          # Full workflow example
├── example_workflow_minimal.py  # Minimal example
│
├── README.md            # Main documentation
├── USAGE.md             # Usage guide
├── STRUCTURE.md         # This file
├── LICENSE              # License file
│
├── requirements.txt     # Dependencies
├── pyproject.toml       # Package metadata
│
└── .gitignore           # Git ignore rules
```

## 📦 Core Package (`jaxtrace/`)

### Main Components

- **`__init__.py`** - Package initialization
  - API exports
  - Configuration utilities
  - Version information

- **`__main__.py`** - Module execution
  - Enables `python -m jaxtrace`
  - Version and test commands

- **`fields/`** - Velocity field handling
  - Time series interpolation
  - Octree FEM for adaptive meshes
  - Structured and unstructured grids
  - Memory-efficient data structures

- **`tracking/`** - Particle tracking
  - Integration schemes (Euler, RK2, RK4)
  - Boundary conditions (reflective, periodic, continuous inlet)
  - Batch processing with memory management
  - Particle seeding strategies

- **`density/`** - Density estimation
  - Kernel Density Estimation (KDE)
  - Smoothed Particle Hydrodynamics (SPH)
  - Efficient neighbor search

- **`visualization/`** - Visualization tools
  - Static plots (matplotlib)
  - Interactive plots (Plotly)
  - Animation and video export
  - 2D/3D rendering

- **`io/`** - Input/Output
  - VTK data loading (PVTU/VTU)
  - HDF5 support
  - Trajectory export
  - Data registry

- **`utils/`** - Utilities
  - JAX configuration and device management
  - Memory monitoring and optimization
  - System diagnostics
  - Reporting tools

## 🚀 Entry Points

### 1. Main Script (`run.py`)
Command-line interface with argument parsing:
```bash
python run.py --help              # Show help
python run.py --test              # Quick test (~5 seconds)
python run.py                     # Default workflow
python run.py --config FILE       # Custom configuration
```

**Features:**
- Argument parsing
- Config file loading
- Quick test mode
- Clear error messages

### 2. Module Execution (`jaxtrace/__main__.py`)
Run as Python module:
```bash
python -m jaxtrace --version      # Show version
python -m jaxtrace --test         # Quick test
python -m jaxtrace                # Main workflow
```

**Features:**
- Works with installed package
- Version checking
- Test mode support

### 3. Direct Examples
```bash
python example_workflow.py
python example_workflow_minimal.py
python -m tests.test_quick
```

## 📝 Examples

### `example_workflow.py`
Complete workflow demonstrating all features:
- VTK data loading with octree FEM interpolation
- Comprehensive configuration system (27+ parameters)
- Multiple particle distributions (uniform, gaussian, random)
- Flexible boundary conditions (inlet/outlet separation)
- Full density analysis (KDE and SPH)
- Multi-plot visualization suite
- YZ density slice with dual cutoffs
- VTK trajectory export

**Key features:**
- Optimized octree FEM for adaptive meshes
- Grid-preserving continuous inlet
- Configurable via dictionary or external file
- Comprehensive output (plots, VTK, reports)

### `example_workflow_minimal.py`
Minimal working example (~150 lines):
- Synthetic time-dependent vortex field
- Basic particle tracking setup
- Simple visualization
- Good starting point for learning

### `config_example.py`
Template configuration file with:
- All available parameters documented
- Preset configurations (test, high-res, etc.)
- Performance estimation function
- Copy-paste ready examples

## 🧪 Tests

### `tests/smoke_test.py`
Quick functionality test verifying:
- Package imports work correctly
- Basic field creation
- Particle tracking pipeline
- Trajectory analysis functions

**Usage:**
```bash
python tests/smoke_test.py
```

### `tests/structure_test.py`
Package structure validation ensuring:
- All modules are importable
- API exports are correct
- No missing dependencies
- Proper module organization

**Usage:**
```bash
python tests/structure_test.py
```

### `tests/test_quick.py`
Fast debugging test for quick verification:
- Synthetic 2D vortex field
- 25 particles, 100 timesteps
- Runs in ~2-5 seconds
- Minimal output (`output_test/`)

**Usage:**
```bash
python run.py --test
python -m jaxtrace --test
python -m tests.test_quick
```

**Purpose:**
- Verify installation
- Quick debugging
- Test workflow changes
- CI/CD smoke testing

## 📚 Documentation

### `README.md`
Main project documentation:
- Project overview and features
- Installation instructions
- Quick start guide
- API examples
- Citation information

### `USAGE.md`
Detailed usage guide:
- All running methods
- Configuration options
- Troubleshooting guide
- Performance tips
- File structure explanation

### `STRUCTURE.md` (This file)
Repository structure reference:
- Directory organization
- Module descriptions
- Entry point documentation
- Test suite overview

### `LICENSE`
European Union Public License (EUPL) v1.2

## 🔧 Configuration Files

### `requirements.txt`
Python dependencies:
```
numpy>=1.20.0
matplotlib>=3.3.0
vtk>=9.0.0
scipy>=1.7.0
jax>=0.4.0
jaxlib>=0.4.0
psutil>=5.8.0
pynvml>=11.0.0  # Optional, for GPU monitoring
```

### `pyproject.toml`
Package metadata for setuptools:
- Package name and version
- Author and license information
- Dependencies (core and optional)
- Entry points configuration
- Build system requirements

### `.gitignore`
Excludes from version control:
- Output directories (`output*/`)
- Python cache (`__pycache__/`, `*.pyc`)
- Build artifacts (`*.egg-info/`, `dist/`, `build/`)
- Development files (`.venv/`, `.mypy_cache/`, etc.)
- Archive folder (`archive/`)
- IDE files (`.vscode/`, `.idea/`)

## 🎯 What's Included

**Essential Files:**
- ✅ Core package (`jaxtrace/`)
- ✅ Test suite (`tests/`)
- ✅ Entry points (`run.py`, `jaxtrace/__main__.py`)
- ✅ Examples (3 files)
- ✅ Documentation (`README.md`, `USAGE.md`, `STRUCTURE.md`)
- ✅ Configuration (`requirements.txt`, `pyproject.toml`)
- ✅ License (`LICENSE`)

**Total:** Clean, minimal structure with ~15 root files

## ✅ Verification

### Check Package Structure
```bash
python tests/structure_test.py
```

### Run Smoke Test
```bash
python tests/smoke_test.py
```

### Quick Functionality Test
```bash
python run.py --test
```

### Verify Installation
```bash
python -c "import jaxtrace; print(jaxtrace.__version__)"
python -m jaxtrace --version
```

### Test Examples
```bash
# Minimal example
python example_workflow_minimal.py

# Full workflow (requires VTK data or uses synthetic fallback)
python example_workflow.py
```

## 🚀 Quick Start

```bash
# 1. Install
pip install -r requirements.txt
pip install -e .

# 2. Test
python run.py --test

# 3. Run minimal example
python example_workflow_minimal.py

# 4. Run with custom config
cp config_example.py myconfig.py
# Edit myconfig.py
python run.py --config myconfig.py
```

## 📊 Module Organization

### Import Hierarchy

```python
# Top-level imports
import jaxtrace as jt

# Core modules
from jaxtrace.fields import TimeSeriesField
from jaxtrace.tracking import create_tracker
from jaxtrace.density import KDEEstimator

# Utilities
from jaxtrace.utils import check_system_requirements
from jaxtrace.visualization import plot_trajectory_2d
```

### Common Patterns

```python
# Configuration
jt.configure(dtype="float32", device="gpu")

# Field creation
field = jt.TimeSeriesField(data=..., times=..., positions=...)

# Tracking
tracker = create_tracker(integrator_name="rk4", field=field)
trajectory = tracker.track_particles(initial_positions=seeds)

# Analysis
kde = jt.KDEEstimator(positions=trajectory.positions[-1])
```

## 🎉 Production Ready

This structure is optimized for:
- ✅ Clean git history
- ✅ Easy navigation
- ✅ Clear documentation
- ✅ Modular organization
- ✅ Test coverage
- ✅ User-friendly API
- ✅ PyPI distribution
- ✅ Professional development
