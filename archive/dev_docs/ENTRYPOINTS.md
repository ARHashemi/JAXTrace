# JAXTrace Entrypoints

This document describes all available ways to run JAXTrace.

## Available Entrypoints

### 1. `run.py` - Main Runner Script ✨ RECOMMENDED

The primary entrypoint for running JAXTrace workflows.

```bash
# Show help
python run.py --help

# Quick test (synthetic field, 25 particles)
python run.py --test

# Run with default configuration
python run.py

# Run with custom configuration
python run.py --config myconfig.py
```

**Features:**
- Argument parsing with helpful error messages
- Automatic config file loading
- Quick test mode for debugging
- Clear error reporting

---

### 2. `python -m jaxtrace` - Module Execution

Run JAXTrace as a Python module.

```bash
# Show version
python -m jaxtrace --version

# Quick test
python -m jaxtrace --test

# Main workflow (if in repo)
python -m jaxtrace
```

**Features:**
- Works with installed package
- Version checking
- Module-based execution

---

### 3. Direct Script Execution

Run workflow scripts directly.

```bash
# Main workflow
python example_workflow.py

# Quick test
python test_quick.py

# Other examples
python example_workflow_minimal.py
```

**Features:**
- Direct control
- No wrapper overhead
- Easy to modify

---

## Test Scripts

### Quick Test (`test_quick.py`)

**Purpose:** Fast debugging and verification

**Characteristics:**
- Synthetic 2D vortex field
- 25 particles
- 100 timesteps
- ~2-5 seconds runtime
- Minimal output

**Run with:**
```bash
python run.py --test
python -m jaxtrace --test
python test_quick.py
```

**Output:**
```
output_test/
└── test_trajectories.png
```

---

### Full Test (`example_workflow.py`)

**Purpose:** Complete workflow with all features

**Characteristics:**
- Real VTK data (or synthetic fallback)
- Configurable particles (default: 45,000)
- Configurable timesteps (default: 2000)
- Octree FEM interpolation
- Full density analysis
- Complete visualization suite

**Run with:**
```bash
python run.py
python example_workflow.py
```

**Output:**
```
output/
├── particles_final.png
├── trajectories_2d.png
├── density_analysis.png
├── density_yz_slice_x_*.png
├── trajectory.vtp
├── trajectory_series_*.vtp
├── summary_report.md
└── enhanced_report.md
```

---

## Configuration Files

### Creating a Config File

Create a Python file with a `config` dictionary:

**myconfig.py:**
```python
config = {
    # Data
    'data_pattern': "/path/to/data_*.pvtu",
    'max_timesteps_to_load': 40,

    # Particles
    'particle_concentrations': {'x': 60, 'y': 50, 'z': 15},
    'particle_distribution': 'uniform',  # or 'gaussian', 'random'

    # Tracking
    'n_timesteps': 2000,
    'dt': 0.0025,
    'integrator': 'rk4',

    # Boundary
    'flow_axis': 'x',
    'boundary_inlet': 'continuous',
    'boundary_outlet': 'absorbing',

    # GPU
    'device': 'gpu',
    'memory_limit_gb': 3.0,
}
```

**Run with:**
```bash
python run.py --config myconfig.py
```

---

## Decision Tree: Which Entrypoint to Use?

```
Start
  ├─ First time using JAXTrace?
  │   └─> python run.py --test
  │
  ├─ Want quick debugging?
  │   └─> python run.py --test
  │
  ├─ Running with default config?
  │   └─> python run.py
  │
  ├─ Have custom configuration?
  │   └─> python run.py --config myconfig.py
  │
  ├─ Want to check version?
  │   └─> python -m jaxtrace --version
  │
  ├─ Modifying the workflow code?
  │   └─> python example_workflow.py
  │
  └─ Installed as package?
      └─> python -m jaxtrace
```

---

## Requirements

### For `run.py` and `python -m jaxtrace`:
```bash
# Install JAXTrace
pip install -e .

# Or ensure it's in PYTHONPATH
export PYTHONPATH=/path/to/JAXTrace:$PYTHONPATH
```

### For direct script execution:
```bash
# Just need dependencies
pip install -r requirements.txt

# And be in the repo directory
cd /path/to/JAXTrace
python example_workflow.py
```

---

## Files Overview

| File | Purpose | Entry Point | Config Support |
|------|---------|-------------|----------------|
| `run.py` | Main runner with arg parsing | ✅ Primary | ✅ Yes |
| `jaxtrace/__main__.py` | Module execution | ✅ Secondary | ❌ No |
| `test_quick.py` | Quick test script | ✅ Testing | ❌ No |
| `example_workflow.py` | Full workflow | ✅ Direct | ⚠️ Edit file |

---

## Examples

### Example 1: First Run (Testing)
```bash
# Test that everything works
python run.py --test

# Expected output:
# ✅ Test passed! JAXTrace is working correctly.
```

### Example 2: Default Workflow
```bash
# Run with defaults
python run.py

# Check output
ls output/
```

### Example 3: Custom High-Resolution Run
```bash
# Create config
cat > highres.py << 'EOF'
config = {
    'particle_concentrations': {'x': 100, 'y': 80, 'z': 20},
    'n_timesteps': 5000,
    'dt': 0.001,
    'device': 'gpu',
}
EOF

# Run
python run.py --config highres.py
```

### Example 4: Gaussian Distribution
```bash
cat > gaussian.py << 'EOF'
config = {
    'particle_distribution': 'gaussian',
    'gaussian_std': {'x': 0.1, 'y': 0.1, 'z': 0.15},
    'boundary_inlet': 'reflective',
    'boundary_outlet': 'reflective',
}
EOF

python run.py --config gaussian.py
```

---

## Troubleshooting

### Error: "No module named 'jaxtrace'"
```bash
# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH
```

### Error: "No module named 'example_workflow'"
```bash
# Make sure you're in the repo root
cd /path/to/JAXTrace
python run.py
```

### Error: Config file not found
```bash
# Use absolute path
python run.py --config /absolute/path/to/config.py

# Or relative path from current directory
python run.py --config ./configs/myconfig.py
```

---

## Summary

**Recommended workflow:**
1. First run: `python run.py --test` (verify installation)
2. Test with data: `python run.py` (default config)
3. Production: `python run.py --config myconfig.py` (custom config)

**Quick reference:**
- Test: `python run.py --test`
- Default: `python run.py`
- Custom: `python run.py --config FILE`
- Version: `python -m jaxtrace --version`
