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
import jaxtrace.config as config


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

    If a file is missing the requested field, the loader skips forward to find
    the first file that has it (for mesh topology). All subsequent files must
    contain the field or a RuntimeError is raised.

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
        (n_nodes, 3) float32 - node coordinates (from first file with field)
    connectivity : np.ndarray
        (n_elements, 4) int32 - element connectivity (from first file with field)
    velocity_sequence : np.ndarray
        (n_valid, n_nodes, 3) float32 - velocity field sequence (only valid timesteps)

    Raises
    ------
    RuntimeError
        If no files in the range contain the requested field, or if a non-first
        file is missing the field (only the leading files may be skipped).
    """
    start, end = timestep_range
    all_timesteps = list(range(start, end + 1))

    if verbose:
        print(f"\nLoading velocity sequence:")
        print(f"  Pattern: {file_pattern}")
        print(f"  Range: {start}-{end} ({len(all_timesteps)} timesteps)")
        print(f"  Field: '{field_name}'")

    # --- Find first timestep that has the field (for mesh topology) ---
    node_positions = None
    connectivity = None
    first_valid_idx = None

    for idx, timestep in enumerate(all_timesteps):
        file_path = base_path / file_pattern.format(timestep=timestep)
        node_positions, connectivity, velocity = load_mesh_from_pvtu(
            file_path, field_name=field_name
        )
        if velocity is not None:
            first_valid_idx = idx
            first_velocity = velocity
            break
        elif verbose:
            print(f"  ⚠ Skipping timestep {timestep}: field '{field_name}' not found")

    if first_valid_idx is None:
        raise RuntimeError(
            f"None of the {len(all_timesteps)} files in range {start}-{end} "
            f"contain field '{field_name}'"
        )

    valid_timesteps = all_timesteps[first_valid_idx:]
    n_skipped = first_valid_idx
    n_valid = len(valid_timesteps)

    n_nodes = node_positions.shape[0]
    n_elements = connectivity.shape[0]

    if verbose:
        if n_skipped > 0:
            print(f"  Skipped {n_skipped} timestep(s) without '{field_name}', "
                  f"starting from timestep {valid_timesteps[0]}")
        print(f"    Nodes: {n_nodes:,}")
        print(f"    Elements: {n_elements:,}")

    # Allocate velocity sequence array (only valid timesteps)
    velocity_sequence = np.zeros((n_valid, n_nodes, 3), dtype=config.FLOAT_DTYPE_NP)
    velocity_sequence[0] = first_velocity

    # Load remaining velocity fields
    if verbose:
        print(f"\n  Loading velocity fields:")

    for i, timestep in enumerate(valid_timesteps[1:], start=1):
        file_path = base_path / file_pattern.format(timestep=timestep)

        # Load only velocity field (mesh topology assumed identical)
        _, _, velocity = load_mesh_from_pvtu(file_path, field_name=field_name)

        if velocity is None:
            raise RuntimeError(
                f"Field '{field_name}' missing in timestep {timestep} "
                f"(file: {file_path}). Only leading timesteps may be skipped; "
                f"gaps within the sequence are not supported."
            )

        # Validate shape
        if velocity.shape != (n_nodes, 3):
            raise ValueError(
                f"Velocity field shape mismatch at timestep {timestep}: "
                f"expected {(n_nodes, 3)}, got {velocity.shape}"
            )

        velocity_sequence[i] = velocity

        if verbose and i % 10 == 0:
            print(f"    Loaded {i + 1}/{n_valid} timesteps...")

    if verbose:
        print(f"    Loaded {n_valid}/{n_valid} timesteps")
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
