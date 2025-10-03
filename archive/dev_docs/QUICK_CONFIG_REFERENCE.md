# Quick Configuration Reference Card

## Minimal Configuration

```python
if __name__ == "__main__":
    user_config = {
        'data_pattern': "/path/to/your/data_*.pvtu",
    }
    main(config=user_config)
```

All other parameters use sensible defaults.

---

## Most Common Customizations

### Change Particle Count

```python
# Low (1,000 particles)
'particle_concentrations': {'x': 20, 'y': 10, 'z': 5}

# Medium (45,000 particles) - DEFAULT
'particle_concentrations': {'x': 60, 'y': 50, 'z': 15}

# High (160,000 particles)
'particle_concentrations': {'x': 100, 'y': 80, 'z': 20}
```

### Change Tracking Duration

```python
# Quick test (500 steps ≈ 1.25s simulation)
'n_timesteps': 500
'dt': 0.0025

# Default (2000 steps ≈ 5s simulation)
'n_timesteps': 2000
'dt': 0.0025

# Long run (5000 steps ≈ 5s simulation, higher resolution)
'n_timesteps': 5000
'dt': 0.001
```

### Seed Particles in Specific Region

```python
# Inlet only (first 20% of X)
'particle_bounds_fraction': {
    'x': (0.0, 0.2),
    'y': (0.0, 1.0),
    'z': (0.0, 1.0)
}

# Outlet only (last 20% of X)
'particle_bounds_fraction': {
    'x': (0.8, 1.0),
    'y': (0.0, 1.0),
    'z': (0.0, 1.0)
}

# Bottom half (Z)
'particle_bounds_fraction': {
    'x': (0.0, 1.0),
    'y': (0.0, 1.0),
    'z': (0.0, 0.5)
}
```

---

## Pre-configured Scenarios

### Test Run (Fast)
```python
user_config = {
    'data_pattern': "/path/to/data_*.pvtu",
    'max_timesteps_to_load': 10,
    'particle_concentrations': {'x': 20, 'y': 10, 'z': 5},
    'n_timesteps': 500,
    'dt': 0.005,
}
# Time: ~2-3 minutes
```

### Standard Run (Default)
```python
user_config = {
    'data_pattern': "/path/to/data_*.pvtu",
    # Everything else uses defaults
}
# Time: ~10-20 minutes
```

### High-Quality Run
```python
user_config = {
    'data_pattern': "/path/to/data_*.pvtu",
    'max_timesteps_to_load': 100,
    'particle_concentrations': {'x': 100, 'y': 80, 'z': 20},
    'n_timesteps': 5000,
    'dt': 0.001,
    'batch_size': 2000,
    'memory_limit_gb': 6.0,
}
# Time: ~30-60 minutes, requires 8+ GB GPU
```

---

## Parameter Quick Reference

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `max_timesteps_to_load` | 40 | 10-200 | Data resolution |
| `max_elements_per_leaf` | 32 | 16-64 | Octree quality |
| `particle_concentrations['x']` | 60 | 10-200 | X particle density |
| `particle_concentrations['y']` | 50 | 10-200 | Y particle density |
| `particle_concentrations['z']` | 15 | 5-50 | Z particle density |
| `n_timesteps` | 2000 | 100-10000 | Tracking duration |
| `dt` | 0.0025 | 0.0001-0.01 | Time step |
| `batch_size` | 1000 | 100-5000 | GPU memory vs speed |
| `memory_limit_gb` | 3.0 | 1.0-8.0 | GPU memory limit |

---

## Performance Tuning

### Faster (Lower Quality)
```python
{
    'max_timesteps_to_load': 10,
    'particle_concentrations': {'x': 30, 'y': 20, 'z': 10},
    'n_timesteps': 1000,
    'dt': 0.005,
    'max_elements_per_leaf': 64,
    'integrator': 'euler',
}
```

### Slower (Higher Quality)
```python
{
    'max_timesteps_to_load': 100,
    'particle_concentrations': {'x': 100, 'y': 80, 'z': 20},
    'n_timesteps': 5000,
    'dt': 0.001,
    'max_elements_per_leaf': 16,
    'integrator': 'rk4',
}
```

---

## Memory Tuning

### Low Memory (2 GB GPU)
```python
{
    'batch_size': 500,
    'particle_concentrations': {'x': 40, 'y': 30, 'z': 10},
    'memory_limit_gb': 1.5,
}
```

### High Memory (8 GB GPU)
```python
{
    'batch_size': 2000,
    'particle_concentrations': {'x': 100, 'y': 80, 'z': 20},
    'memory_limit_gb': 6.0,
}
```

---

## Troubleshooting Cheat Sheet

| Problem | Solution |
|---------|----------|
| Out of memory | ↓ `batch_size`, ↓ particles, ↓ `max_timesteps_to_load` |
| Too slow | ↑ `batch_size`, ↑ `max_elements_per_leaf`, use `'integrator': 'euler'` |
| Inaccurate | ↓ `dt`, ↓ `max_elements_per_leaf`, ↑ `max_timesteps_to_load` |
| Particles escape | Check boundary conditions, ↓ `dt` |
| Octree too coarse | ↓ `max_elements_per_leaf`, ↑ `max_octree_depth` |

---

## Copy-Paste Templates

### Template 1: Minimal
```python
if __name__ == "__main__":
    user_config = {
        'data_pattern': "/path/to/data_*.pvtu",
    }
    main(config=user_config)
```

### Template 2: Custom Particles
```python
if __name__ == "__main__":
    user_config = {
        'data_pattern': "/path/to/data_*.pvtu",
        'particle_concentrations': {'x': 80, 'y': 60, 'z': 20},
        'n_timesteps': 3000,
    }
    main(config=user_config)
```

### Template 3: Inlet Study
```python
if __name__ == "__main__":
    user_config = {
        'data_pattern': "/path/to/data_*.pvtu",
        'particle_bounds_fraction': {
            'x': (0.0, 0.2),
            'y': (0.0, 1.0),
            'z': (0.0, 1.0)
        },
        'n_timesteps': 3000,
    }
    main(config=user_config)
```

### Template 4: Full Custom
```python
if __name__ == "__main__":
    user_config = {
        # Data
        'data_pattern': "/path/to/data_*.pvtu",
        'max_timesteps_to_load': 40,

        # Octree
        'max_elements_per_leaf': 32,

        # Particles
        'particle_concentrations': {'x': 60, 'y': 50, 'z': 15},

        # Tracking
        'n_timesteps': 2000,
        'dt': 0.0025,
        'batch_size': 1000,

        # GPU
        'device': 'gpu',
        'memory_limit_gb': 3.0,
    }
    main(config=user_config)
```

---

## One-Line Modifications

```python
# Quick test
user_config.update({'n_timesteps': 500, 'particle_concentrations': {'x': 20, 'y': 10, 'z': 5}})

# High resolution
user_config.update({'particle_concentrations': {'x': 100, 'y': 80, 'z': 20}})

# Inlet region
user_config.update({'particle_bounds_fraction': {'x': (0.0, 0.2), 'y': (0.0, 1.0), 'z': (0.0, 1.0)}})

# CPU mode
user_config.update({'device': 'cpu', 'batch_size': 500})
```

---

**For detailed explanations, see `CONFIGURATION_GUIDE.md`**
