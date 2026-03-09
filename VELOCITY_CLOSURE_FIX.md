# Fix for 40-Timestep Velocity Closure Issue

**IF** the 2-timestep test confirms that 40 timesteps causes the error, this document provides the fix.

## Problem

The current RK4 implementation creates a closure over `velocity_fields_gpu` with shape `(40, 571173, 3)`. During JAX compilation, this causes a 22.46 TiB intermediate array to be allocated, crashing the compilation.

## Solution: Pass Velocity as Parameter

Instead of closing over the full velocity sequence, we'll:
1. Extract the single timestep **before** calling the vmapped function
2. Pass just the `(n_nodes, 3)` velocity field as a parameter
3. This eliminates the large closure

## Code Changes

### File: `jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py`

#### Change 1: Modify Function Signature (Line ~455)

**Current**:
```python
def rk4_fully_fused_step_timedep(
    positions_gpu: jax.Array,
    element_ids_gpu: jax.Array,
    dt: float,
    velocity_fields_gpu: jax.Array,
    time_idx: int
) -> Tuple[jax.Array, jax.Array]:
    """
    Single RK4 timestep with time-dependent velocity (fully fused).
    """
    n_timesteps = velocity_fields_gpu.shape[0]

    # Cyclic indexing for velocity
    vel_idx = time_idx % n_timesteps
    velocity_field = velocity_fields_gpu[vel_idx]

    # Single-particle RK4 with all stages fused
    def rk4_single_particle(pos: jax.Array, elem_id: jax.Array):
        ...
```

**Fixed**:
```python
def rk4_fully_fused_step_timedep(
    positions_gpu: jax.Array,
    element_ids_gpu: jax.Array,
    dt: float,
    velocity_fields_gpu: jax.Array,
    time_idx: int
) -> Tuple[jax.Array, jax.Array]:
    """
    Single RK4 timestep with time-dependent velocity (fully fused).
    """
    n_timesteps = velocity_fields_gpu.shape[0]

    # Cyclic indexing for velocity - EXTRACT BEFORE VMAP
    vel_idx = time_idx % n_timesteps
    velocity_field = velocity_fields_gpu[vel_idx]  # (n_nodes, 3) - single timestep

    # Single-particle RK4 with velocity as parameter
    def rk4_single_particle(pos: jax.Array, elem_id: jax.Array, vel_field: jax.Array):
        """RK4 for single particle with velocity field passed explicitly."""
        ...
```

#### Change 2: Update Interpolation Calls (Lines ~490-506)

**Current**:
```python
# Stage 1: k1 = f(t, y)
elem_k1 = search_l0_l1_l2_single(pos, elem_id)
vel_k1 = interpolate_velocity_single(pos, elem_k1, velocity_field)
pos_k1 = pos + 0.5 * dt * vel_k1

# Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
elem_k2 = search_l0_l1_l2_single(pos_k1, elem_k1)
vel_k2 = interpolate_velocity_single(pos_k1, elem_k2, velocity_field)
...
```

**No change needed** - `velocity_field` is already passed as parameter to `interpolate_velocity_single`.

The key is that `velocity_field` is now a **parameter** to `rk4_single_particle` instead of being captured from the closure.

#### Change 3: Update Vmap Call (Lines ~516-520)

**Current**:
```python
# SINGLE vmap over all particles (fully fused)
positions_final, element_ids_final = jax.vmap(rk4_single_particle)(
    positions_gpu, element_ids_gpu
)
```

**Fixed**:
```python
# SINGLE vmap over all particles with velocity field broadcasted
# velocity_field is (n_nodes, 3), broadcast to all particles (they all use same field)
positions_final, element_ids_final = jax.vmap(
    rk4_single_particle,
    in_axes=(0, 0, None)  # vmap over positions and element_ids, broadcast velocity_field
)(positions_gpu, element_ids_gpu, velocity_field)
```

**Key**: `in_axes=(0, 0, None)` means:
- First arg (`pos`): vmap over axis 0 (one per particle)
- Second arg (`elem_id`): vmap over axis 0 (one per particle)
- Third arg (`vel_field`): broadcast (None) - same `(n_nodes, 3)` array shared by all particles

## Complete Modified Function

<details>
<summary>Click to expand full fixed function</summary>

```python
def rk4_fully_fused_step_timedep(
    positions_gpu: jax.Array,
    element_ids_gpu: jax.Array,
    dt: float,
    velocity_fields_gpu: jax.Array,
    time_idx: int
) -> Tuple[jax.Array, jax.Array]:
    """
    Single RK4 timestep with time-dependent velocity (fully fused).

    All operations fused into single vmap over particles:
    - All 5 RK4 stages (k1, k2, k3, k4, final)
    - All 5 L0+L1+L2 searches
    - All 4 velocity interpolations

    Args:
        positions_gpu: (N, 3) particle positions
        element_ids_gpu: (N,) cached element IDs
        dt: timestep size
        velocity_fields_gpu: (n_timesteps, n_nodes, 3) velocity sequence
        time_idx: index into velocity sequence (cyclic with modulo)

    Returns:
        positions_final: (N, 3) updated positions
        element_ids_final: (N,) updated element IDs
    """
    n_timesteps = velocity_fields_gpu.shape[0]

    # Cyclic indexing - EXTRACT SINGLE TIMESTEP BEFORE VMAP
    # This avoids closing over the full (n_timesteps, n_nodes, 3) array
    vel_idx = time_idx % n_timesteps
    velocity_field = velocity_fields_gpu[vel_idx]  # (n_nodes, 3) - single timestep

    # Single-particle RK4 with velocity field as explicit parameter
    def rk4_single_particle(pos: jax.Array, elem_id: jax.Array, vel_field: jax.Array):
        """
        RK4 for single particle with all stages inline.

        Args:
            pos: (3,) particle position
            elem_id: scalar element ID
            vel_field: (n_nodes, 3) velocity field for this timestep (BROADCASTED to all particles)
        """

        # Stage 1: k1 = f(t, y)
        elem_k1 = search_l0_l1_l2_single(pos, elem_id)
        vel_k1 = interpolate_velocity_single(pos, elem_k1, vel_field)
        pos_k1 = pos + 0.5 * dt * vel_k1

        # Stage 2: k2 = f(t + dt/2, y + dt/2 * k1)
        elem_k2 = search_l0_l1_l2_single(pos_k1, elem_k1)
        vel_k2 = interpolate_velocity_single(pos_k1, elem_k2, vel_field)
        pos_k2 = pos + 0.5 * dt * vel_k2

        # Stage 3: k3 = f(t + dt/2, y + dt/2 * k2)
        elem_k3 = search_l0_l1_l2_single(pos_k2, elem_k2)
        vel_k3 = interpolate_velocity_single(pos_k2, elem_k3, vel_field)
        pos_k3 = pos + dt * vel_k3

        # Stage 4: k4 = f(t + dt, y + dt * k3)
        elem_k4 = search_l0_l1_l2_single(pos_k3, elem_k3)
        vel_k4 = interpolate_velocity_single(pos_k3, elem_k4, vel_field)

        # Final position: y_n+1 = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        pos_final = pos + (dt / 6.0) * (vel_k1 + 2.0*vel_k2 + 2.0*vel_k3 + vel_k4)

        # Final element search
        elem_final = search_l0_l1_l2_single(pos_final, elem_k4)

        return pos_final, elem_final

    # SINGLE vmap over all particles with velocity field broadcasted
    # velocity_field is (n_nodes, 3), broadcast to all particles
    positions_final, element_ids_final = jax.vmap(
        rk4_single_particle,
        in_axes=(0, 0, None)  # vmap positions and element_ids, broadcast velocity_field
    )(positions_gpu, element_ids_gpu, velocity_field)

    return positions_final, element_ids_final
```

</details>

## Why This Fixes It

**Before**:
- Closure captures `velocity_fields_gpu` with shape `(40, 571173, 3)` = 274 MB
- JAX compilation tries to inline/expand this through all vmapped operations
- Creates 22.46 TiB intermediate (unknown expansion factor)

**After**:
- Extract single `velocity_field` with shape `(571173, 3)` = 6.85 MB (40× smaller!)
- Only this small array is passed to vmapped function
- JAX broadcasts the `(n_nodes, 3)` array to all particles (efficient)
- No huge closure, no huge intermediate

## Testing the Fix

After applying changes:

```bash
# Revert to 40 timesteps
sed -i 's/VELOCITY_TIMESTEP_RANGE = (158, 159)/VELOCITY_TIMESTEP_RANGE = (120, 159)/' production_tracking_fully_fused_timedep.py

# Run production script
python3 production_tracking_fully_fused_timedep.py 2>&1 | tee logs/production_with_velocity_fix.log
```

**Expected**: Compilation succeeds, runs 40 timesteps without error.

## Alternative: Even More Explicit

If the above still has issues, we can be even more explicit by passing velocity_field from the top level:

```python
# In create_rk4_fully_fused_timedep(), change the returned function:
def rk4_fully_fused_step_timedep(
    positions_gpu: jax.Array,
    element_ids_gpu: jax.Array,
    dt: float,
    velocity_field: jax.Array,  # Changed: now expects SINGLE timestep
    time_idx: int  # Optional: kept for compatibility
) -> Tuple[jax.Array, jax.Array]:
    """Now expects single velocity field, not sequence."""

    # No extraction needed - velocity_field is already (n_nodes, 3)

    def rk4_single_particle(pos, elem_id, vel_field):
        ...

    return jax.vmap(rk4_single_particle, in_axes=(0, 0, None))(
        positions_gpu, element_ids_gpu, velocity_field
    )
```

Then in production script:
```python
# Extract velocity field BEFORE calling rk4_step
vel_idx = step % n_velocity_steps
velocity_field_single = velocity_fields_gpu[vel_idx]

positions_gpu, element_ids_gpu = rk4_step(
    positions_gpu,
    element_ids_gpu,
    DT,
    velocity_field_single,  # Single timestep
    step
)
```

This moves the extraction completely outside the RK4 function, giving JAX no chance to mess it up.

## Summary

The fix changes 3 things:
1. Extract single velocity timestep **before** vmap
2. Pass velocity as explicit parameter to `rk4_single_particle`
3. Use `in_axes=(0, 0, None)` to broadcast velocity to all particles

This eliminates the 274 MB closure over 40 timesteps and replaces it with a 6.85 MB parameter that's efficiently broadcast.
