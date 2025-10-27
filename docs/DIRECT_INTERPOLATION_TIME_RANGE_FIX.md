# Direct Interpolation Time Range Configuration

## The Error

When running the refactored code with direct interpolation mode, you may encounter:

```
ValueError: ❌ MESH TOPOLOGY MISMATCH - Direct interpolation requires consistent mesh topology!

Timestep 0 (t=0.0) has 2301 nodes,
but reference mesh (revolution cycle) has 780922 nodes.
```

## Root Cause

The direct interpolation mode requires **all timesteps to have identical mesh topology** (same number of nodes and connectivity). This is because:

1. The reference mesh (positions and connectivity) is loaded once from the revolution cycle (timestep 120)
2. Direct interpolators are built using this reference mesh structure
3. When tracking starts at t=0.0, it tries to load velocities from the refinement phase (timestep 0)
4. Refinement phase has only 2,301 nodes vs revolution cycle's 780,922 nodes → **MISMATCH**

## Solution Options

### Option 1: Adjust Time Range (Recommended for Memory Savings)

**Use direct interpolation mode but track only within the revolution cycle:**

```python
config = {
    # ... other config ...
    'use_shared_coarse_octree': True,
    'use_direct_interpolation': True,  # Default, can omit
    'time_span': (120.0, 159.0),  # ← MATCH REVOLUTION CYCLE!
    'revolution_timesteps': 40,
}
```

**Pros**:
- ✅ 99% memory savings (1 MB vs 5-8 GB)
- ✅ Fast and efficient
- ✅ Works perfectly for revolution cycle tracking

**Cons**:
- ❌ Cannot track through refinement phase (timesteps 0-119)

### Option 2: Use Legacy Mode (For Full Time Range)

**Use legacy interpolation mode to support varying topology:**

```python
config = {
    # ... other config ...
    'use_shared_coarse_octree': True,
    'use_direct_interpolation': False,  # ← ENABLE LEGACY MODE
    'time_span': (0.0, 159.0),  # Can use full range
}
```

**Pros**:
- ✅ Supports full time range including refinement phase
- ✅ Handles varying mesh topology
- ✅ Proven implementation

**Cons**:
- ❌ Uses 5-8 GB memory (vs 1 MB for direct mode)
- ❌ Slower initialization (builds third octree)

## Recommended Configuration

For most use cases with AMR data where you care about the steady-state revolution cycle:

**Use Option 1** (direct mode with adjusted time range):

```python
# In example_workflow.py or your config:
user_config = {
    # Data
    'vtk_pattern': '/path/to/data/featurelessAvtk_*.pvtu',
    'use_shared_coarse_octree': True,
    'revolution_timesteps': 40,

    # Direct interpolation (memory-efficient)
    'use_direct_interpolation': True,  # Default, can omit

    # Tracking time range - MATCH REVOLUTION CYCLE!
    'time_span': (120.0, 159.0),  # Timesteps 120-159 (40 timesteps)
    'n_timesteps': 2000,
    'dt': 0.0025,

    # ... rest of config ...
}
```

This gives you:
- **1 MB memory** instead of 5-8 GB
- **Full tracking accuracy** within revolution cycle
- **Fast performance**

## Why This Limitation Exists

The direct interpolation mode is designed for the common case where:
1. AMR refinement happens initially (varying topology)
2. Simulation reaches steady state (fixed topology - revolution cycle)
3. **Tracking focuses on the steady-state behavior** (revolution cycle)

This matches the typical workflow:
- Build octree from revolution cycle mesh (timesteps 120-159)
- Track particles through revolution cycle (t=120.0 to t=159.0)
- Analyze steady-state particle behavior

## Future Work

Potential enhancements to support varying topology in direct mode:
1. **Per-timestep reference meshes**: Store positions/connectivity for each unique topology
2. **Topology detection**: Automatically switch reference mesh when topology changes
3. **Hybrid approach**: Use direct mode for revolution cycle, fall back to legacy for refinement

These would add complexity but could support the full time range while maintaining memory efficiency.

## Summary

**For your current use case (revolution cycle tracking):**

✅ Change `time_span` from `(0.0, 6.25)` to `(120.0, 159.0)`

This enables the memory-efficient direct interpolation mode while avoiding the refinement phase with varying topology.
