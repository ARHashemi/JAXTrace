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


def load_field_sequences_from_pvtu(
    base_path: Path,
    file_pattern: str,
    timestep_range: Tuple[int, int],
    field_names: List[str],
    verbose: bool = True,
):
    """
    Load multiple per-node fields from a sequence of PVTU files in a
    single traversal.

    Each PVTU file is opened, parsed, and decompressed **once** —
    every requested field is extracted from the same VTK reader output
    before the file is closed. This is materially faster than calling
    :func:`load_velocity_sequence_from_pvtu` once per field (PVTU I/O
    + base64 decode + decompression dominate the per-file cost; the
    additional point-data array reads are essentially free).

    Field components are auto-detected: 1-component fields are stacked
    into ``(n_timesteps, n_nodes)`` arrays; multi-component fields
    into ``(n_timesteps, n_nodes, n_components)``.

    Parameters
    ----------
    base_path : Path
        Directory containing the PVTU files.
    file_pattern : str
        File naming pattern with a ``{timestep}`` placeholder, e.g.
        ``"cylA_{timestep}.pvtu"``.
    timestep_range : Tuple[int, int]
        ``(start, end)`` inclusive timestep range.
    field_names : list[str]
        Names of point-data fields to load. Missing fields raise a
        ``RuntimeError`` (use :func:`load_velocity_sequence_from_pvtu`
        if you need leading-missing-frame tolerance).
    verbose : bool
        If True, print per-file progress.

    Returns
    -------
    node_positions : np.ndarray
        ``(n_nodes, 3)`` float32 — node coordinates from the first file.
    connectivity : np.ndarray
        ``(n_elements, 4)`` int32 — element connectivity from the first file.
    sequences : dict[str, np.ndarray]
        ``{field_name: stack}``. ``stack`` is ``(n_timesteps, n_nodes)``
        for scalar fields and ``(n_timesteps, n_nodes, n_components)``
        for vector fields.
    """
    if not field_names:
        raise ValueError("field_names must contain at least one name")

    start, end = timestep_range
    timesteps = list(range(start, end + 1))
    n_steps = len(timesteps)

    if verbose:
        print(f"\nLoading {len(field_names)} field(s) over {n_steps} timesteps:")
        print(f"  Pattern: {file_pattern}")
        print(f"  Fields:  {field_names}")

    node_positions = None
    connectivity = None
    sequences: dict = {}
    n_nodes = None

    for i, ts in enumerate(timesteps):
        file_path = base_path / file_pattern.format(timestep=ts)
        pos, conn, fields_dict = load_mesh_from_pvtu(
            file_path, field_names=field_names, verbose=False,
        )

        if i == 0:
            node_positions = pos.astype(config.FLOAT_DTYPE_NP)
            connectivity = conn
            n_nodes = pos.shape[0]
            # Allocate the output stacks now that we know the per-field
            # component count from the first timestep.
            for name in field_names:
                arr = fields_dict.get(name)
                if arr is None:
                    raise RuntimeError(
                        f"Field '{name}' not found in {file_path}"
                    )
                if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
                    sequences[name] = np.zeros(
                        (n_steps, n_nodes), dtype=config.FLOAT_DTYPE_NP,
                    )
                elif arr.ndim == 2:
                    sequences[name] = np.zeros(
                        (n_steps, n_nodes, arr.shape[1]),
                        dtype=config.FLOAT_DTYPE_NP,
                    )
                else:
                    raise ValueError(
                        f"Field '{name}' has unsupported shape {arr.shape}"
                    )

        # Validate consistency and store per-step values.
        for name in field_names:
            arr = fields_dict.get(name)
            if arr is None:
                raise RuntimeError(
                    f"Field '{name}' missing at timestep {ts} "
                    f"(file: {file_path}). All requested fields must be "
                    f"present in every timestep."
                )
            arr = np.asarray(arr).reshape(-1) if sequences[name].ndim == 2 \
                else np.asarray(arr).reshape(n_nodes, -1)
            sequences[name][i] = arr.astype(config.FLOAT_DTYPE_NP)

        if verbose and (i % 10 == 0 or i == n_steps - 1):
            print(f"  Loaded timestep {ts} ({i + 1}/{n_steps})")

    if verbose:
        total_mb = sum(s.nbytes for s in sequences.values()) / (1024 ** 2)
        print(f"  Total field memory: {total_mb:.1f} MB across "
              f"{len(sequences)} field(s)")

    return node_positions, connectivity, sequences


def load_scalar_sequence_from_pvtu(
    base_path: Path,
    file_pattern: str,
    timestep_range: Tuple[int, int],
    field_name: str,
    verbose: bool = True,
) -> np.ndarray:
    """
    Load a sequence of per-node scalar fields from PVTU files.

    Companion to ``load_velocity_sequence_from_pvtu`` for fields with one
    component per node (Temperature, Pressure, LEVEL, ...). Mesh topology
    is assumed identical across timesteps and is not returned; call the
    velocity loader first if you need positions/connectivity.

    Parameters
    ----------
    base_path, file_pattern, timestep_range, field_name, verbose
        Same meaning as ``load_velocity_sequence_from_pvtu``.

    Returns
    -------
    np.ndarray
        ``(n_timesteps, n_nodes)`` float array. Same precision policy as the
        velocity loader (``config.FLOAT_DTYPE_NP``).
    """
    start, end = timestep_range
    all_timesteps = list(range(start, end + 1))

    if verbose:
        print(f"\nLoading scalar sequence '{field_name}':")
        print(f"  Range: {start}-{end} ({len(all_timesteps)} timesteps)")

    # Find first timestep that has the field (handle leading missing entries).
    first_idx = None
    first_arr = None
    n_nodes = None
    for idx, ts in enumerate(all_timesteps):
        file_path = base_path / file_pattern.format(timestep=ts)
        _, _, arr = load_mesh_from_pvtu(file_path, field_name=field_name)
        if arr is not None:
            first_idx = idx
            # Coerce to 1D in case VTK returned (n_nodes, 1).
            first_arr = np.asarray(arr).reshape(-1)
            n_nodes = first_arr.shape[0]
            break

    if first_idx is None:
        raise RuntimeError(
            f"None of the {len(all_timesteps)} files in range {start}-{end} "
            f"contain scalar field '{field_name}'"
        )

    valid_ts = all_timesteps[first_idx:]
    n_valid = len(valid_ts)
    sequence = np.zeros((n_valid, n_nodes), dtype=config.FLOAT_DTYPE_NP)
    sequence[0] = first_arr.astype(config.FLOAT_DTYPE_NP)

    for i, ts in enumerate(valid_ts[1:], start=1):
        file_path = base_path / file_pattern.format(timestep=ts)
        _, _, arr = load_mesh_from_pvtu(file_path, field_name=field_name)
        if arr is None:
            raise RuntimeError(
                f"Scalar field '{field_name}' missing in timestep {ts} "
                f"(file: {file_path}). Only leading timesteps may be skipped."
            )
        arr = np.asarray(arr).reshape(-1)
        if arr.shape[0] != n_nodes:
            raise ValueError(
                f"Scalar field shape mismatch at timestep {ts}: "
                f"expected ({n_nodes},), got {arr.shape}"
            )
        sequence[i] = arr.astype(config.FLOAT_DTYPE_NP)

    if verbose:
        memory_mb = sequence.nbytes / (1024 ** 2)
        print(f"    Loaded {n_valid}/{n_valid} timesteps  ({memory_mb:.1f} MB)")

    return sequence


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
