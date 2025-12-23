#!/usr/bin/env python3
"""
Time-Dependent Mesh Loader

Utilities for loading sequences of velocity fields from PVTU files
for transient/periodic particle tracking simulations.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List
from jaxtrace.gpu.mesh_loader import load_mesh_from_pvtu


def load_velocity_sequence_from_pvtu(
    base_path: Path,
    file_pattern: str,
    timestep_range: Tuple[int, int],
    field_name: str = 'Displacement',
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a sequence of velocity fields from PVTU files.

    Loads mesh topology from first file, then loads velocity field from each
    timestep in the range. All files must have identical mesh topology.

    Parameters
    ----------
    base_path : Path
        Directory containing PVTU files
    file_pattern : str
        File naming pattern with {timestep} placeholder
        Example: "threadedAvtk_{timestep}.pvtu"
    timestep_range : Tuple[int, int]
        (start, end) timestep range (inclusive)
        Example: (120, 159) loads 40 timesteps
    field_name : str, default='Displacement'
        Name of velocity field in PVTU files
    verbose : bool, default=True
        Print loading progress

    Returns
    -------
    node_positions : np.ndarray
        (n_nodes, 3) float32 - node coordinates (from first file)
    connectivity : np.ndarray
        (n_elements, 4) int32 - element connectivity (from first file)
    velocity_sequence : np.ndarray
        (n_timesteps, n_nodes, 3) float32 - velocity field sequence

    Examples
    --------
    >>> node_pos, conn, vel_seq = load_velocity_sequence_from_pvtu(
    ...     Path("/data/mesh"),
    ...     "threadedAvtk_{timestep}.pvtu",
    ...     (120, 159),
    ...     field_name='Displacement'
    ... )
    >>> vel_seq.shape
    (40, 900658, 3)
    """
    start, end = timestep_range
    n_timesteps = end - start + 1

    if verbose:
        print(f"\nLoading velocity sequence:")
        print(f"  Pattern: {file_pattern}")
        print(f"  Range: {start}-{end} ({n_timesteps} timesteps)")
        print(f"  Field: '{field_name}'")

    # Load first file to get mesh topology
    first_file = base_path / file_pattern.format(timestep=start)
    if verbose:
        print(f"\n  Loading mesh topology from: {first_file.name}")

    node_positions, connectivity, first_velocity = load_mesh_from_pvtu(
        first_file,
        field_name=field_name
    )

    n_nodes = node_positions.shape[0]
    n_elements = connectivity.shape[0]

    if verbose:
        print(f"    Nodes: {n_nodes:,}")
        print(f"    Elements: {n_elements:,}")

    # Allocate velocity sequence array
    velocity_sequence = np.zeros((n_timesteps, n_nodes, 3), dtype=np.float32)
    velocity_sequence[0] = first_velocity

    # Load remaining velocity fields
    if verbose:
        print(f"\n  Loading velocity fields:")

    for i, timestep in enumerate(range(start + 1, end + 1)):
        file_path = base_path / file_pattern.format(timestep=timestep)

        # Load only velocity field (mesh topology assumed identical)
        _, _, velocity = load_mesh_from_pvtu(file_path, field_name=field_name)

        # Validate shape
        if velocity.shape != (n_nodes, 3):
            raise ValueError(
                f"Velocity field shape mismatch at timestep {timestep}: "
                f"expected {(n_nodes, 3)}, got {velocity.shape}"
            )

        velocity_sequence[i + 1] = velocity

        if verbose and (i + 1) % 10 == 0:
            print(f"    Loaded {i + 2}/{n_timesteps} timesteps...")

    if verbose:
        print(f"    Loaded {n_timesteps}/{n_timesteps} timesteps")
        memory_mb = velocity_sequence.nbytes / (1024**2)
        print(f"    Memory: {memory_mb:.1f} MB")

    return node_positions, connectivity, velocity_sequence


def compute_velocity_cycle_params(
    total_steps: int,
    dt: float,
    velocity_timestep_range: Tuple[int, int],
    velocity_dt: float
) -> dict:
    """
    Compute parameters for cyclic velocity indexing.

    Parameters
    ----------
    total_steps : int
        Total number of particle tracking steps
    dt : float
        Particle tracking timestep size
    velocity_timestep_range : Tuple[int, int]
        (start, end) range of velocity timesteps loaded
    velocity_dt : float
        Time spacing between velocity snapshots

    Returns
    -------
    params : dict
        Dictionary with cycle parameters:
        - n_velocity_steps: number of velocity timesteps
        - cycle_period: physical time period of one velocity cycle
        - steps_per_velocity: particle steps per velocity timestep
        - n_cycles: number of complete cycles in simulation

    Examples
    --------
    >>> params = compute_velocity_cycle_params(
    ...     total_steps=2500,
    ...     dt=0.0025,
    ...     velocity_timestep_range=(120, 159),
    ...     velocity_dt=0.1
    ... )
    >>> params['n_cycles']
    1.5625  # 40 velocity steps cycled over 2500 tracking steps
    """
    start, end = velocity_timestep_range
    n_velocity_steps = end - start + 1
    cycle_period = n_velocity_steps * velocity_dt
    total_time = total_steps * dt
    n_cycles = total_time / cycle_period

    steps_per_velocity = int(velocity_dt / dt)

    return {
        'n_velocity_steps': n_velocity_steps,
        'cycle_period': cycle_period,
        'total_time': total_time,
        'n_cycles': n_cycles,
        'steps_per_velocity': steps_per_velocity
    }
