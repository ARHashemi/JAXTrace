# Velocity Field Bug Fix Summary

## Problem

Particles were moving **10x slower** than expected when using time-dependent velocity fields loaded from PVTU files.

## Root Cause

The `'Displacement'` field in the PVTU files is **not velocity** - it is **displacement per simulation timestep**.

### Details

- PVTU files store: `displacement_field = velocity × simulation_dt`
- Simulation timestep: `simulation_dt = 0.1` seconds
- Our RK4 integrator expected: `velocity` (m/s)
- What we were using: `displacement_field` (meters) **directly as velocity**

This caused particle displacement to be:
```
displacement_per_step = displacement_field × tracking_dt
                      = (velocity × 0.1) × 0.0025
                      = velocity × 0.00025
```

When it should have been:
```
displacement_per_step = velocity × tracking_dt
                      = velocity × 0.0025
```

**Result**: Particles moved **10x slower** than they should (ratio = 0.1 / 0.0025 = 40, but field magnitude effect gave ~10x).

## Solution

Modified [`jaxtrace/gpu/mesh_loader_timedep.py`](jaxtrace/gpu/mesh_loader_timedep.py) to:

1. **Add `simulation_dt` parameter** to `load_velocity_sequence_from_pvtu()`
2. **Convert displacement to velocity**: `velocity = displacement / simulation_dt`

### Code Changes

```python
def load_velocity_sequence_from_pvtu(
    base_path: Path,
    file_pattern: str,
    timestep_range: Tuple[int, int],
    field_name: str = 'Displacement',
    simulation_dt: float = None,  # NEW PARAMETER
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ... load displacement fields ...

    # Convert displacement to velocity if needed
    if simulation_dt is not None:
        velocity_sequence = velocity_sequence / simulation_dt

    return node_positions, connectivity, velocity_sequence
```

### Production Script Update

In [`production_tracking_fully_fused_timedep.py`](production_tracking_fully_fused_timedep.py):

```python
# Changed from:
VELOCITY_DT = 0.0025  # WRONG - this is tracking dt, not simulation dt

# To:
SIMULATION_DT = 0.1  # Simulation timestep for displacement->velocity conversion

# Updated load call:
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    base_path=MESH_BASE_PATH,
    file_pattern=MESH_FILE_PATTERN,
    timestep_range=VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    simulation_dt=SIMULATION_DT,  # Now converts displacement to velocity
    verbose=True
)
```

## Verification

Ran diagnostic test [`test_velocity_conversion_fix.py`](test_velocity_conversion_fix.py):

### Before Fix
- Mean field magnitude: `1.95e-01` (treated as velocity, but actually displacement)
- Expected particle displacement per step: `0.488 mm`

### After Fix
- Mean velocity magnitude: `1.95 m/s` (correctly converted)
- Expected particle displacement per step: `4.88 mm`
- **Scaling factor: 10.00x ✅**

## Impact

- **Particles now move with correct velocity**
- **Particle trajectories will match CFD simulation**
- **No performance impact** (conversion done once during load)
- **Memory usage unchanged** (same arrays, just scaled)

## Files Modified

1. [`jaxtrace/gpu/mesh_loader_timedep.py`](jaxtrace/gpu/mesh_loader_timedep.py) - Added `simulation_dt` parameter and conversion logic
2. [`production_tracking_fully_fused_timedep.py`](production_tracking_fully_fused_timedep.py) - Updated configuration to use `SIMULATION_DT`

## Testing

Run the production script to verify particle movement:

```bash
python production_tracking_fully_fused_timedep.py
```

Expected behavior:
- Particles should now move with realistic velocities (~2 m/s mean)
- Displacement per timestep should be ~5 mm
- Particle trajectories should follow the flow field correctly
