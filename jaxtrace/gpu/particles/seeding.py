"""
Particle seeding with flexible control over distribution and density.

Part of Phase 3: Particle Seeding & Initial Assignment

Provides comprehensive control over:
- Seeding bounding box (can be subset of domain)
- Per-axis particle density (particles per unit length)
- Distribution type (uniform grid, random, stratified)
- Total particle count or density-based specification
"""

import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import jaxtrace.config as config


@dataclass
class ParticleState:
    """
    Complete particle state for tracking simulation.
    
    Attributes
    ----------
    positions : np.ndarray
        Particle positions, shape (N, 3), float32
    element_ids : np.ndarray
        Containing element IDs, shape (N,), int32
        Value is -1 if particle not in mesh
    block_ids : np.ndarray
        Containing block IDs, shape (N,), int32
        Value is -1 if particle outside domain
    velocities : np.ndarray, optional
        Particle velocities, shape (N, 3), float32
        None if not yet interpolated
    active : np.ndarray, optional
        Active flags, shape (N,), bool
        True if particle is active (in mesh), False if inactive
    """
    positions: np.ndarray
    element_ids: np.ndarray
    block_ids: np.ndarray
    velocities: Optional[np.ndarray] = None
    active: Optional[np.ndarray] = None
    
    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return self.positions.shape[0]
    
    @property
    def n_active(self) -> int:
        """Number of active particles."""
        if self.active is None:
            return np.sum(self.element_ids >= 0)
        return np.sum(self.active)
    
    def __repr__(self) -> str:
        n_found = np.sum(self.element_ids >= 0)
        pct_found = 100 * n_found / self.n_particles if self.n_particles > 0 else 0
        return (
            f"ParticleState(\n"
            f"  Total particles: {self.n_particles:,}\n"
            f"  Found in mesh: {n_found:,} ({pct_found:.1f}%)\n"
            f"  Velocities: {'computed' if self.velocities is not None else 'not computed'}\n"
            f")"
        )


@dataclass
class SeedingConfig:
    """
    Configuration for particle seeding.
    
    Provides two modes:
    1. Density-based: Specify particles per unit length on each axis
    2. Count-based: Specify total particle count
    
    Attributes
    ----------
    bbox : np.ndarray
        Seeding bounding box [xmin, xmax, ymin, ymax, zmin, zmax], float32
        Can be subset of domain for targeted seeding
    density_x : float, optional
        Particles per unit length in x-direction (particles/meter)
        If None, uses n_particles instead
    density_y : float, optional
        Particles per unit length in y-direction (particles/meter)
    density_z : float, optional
        Particles per unit length in z-direction (particles/meter)
    n_particles : int, optional
        Total particle count (overrides density if specified)
    distribution : str
        Distribution type: 'uniform', 'random', 'stratified'
    jitter : float
        Random perturbation factor for uniform grids (0.0 = no jitter, 0.5 = max)
    seed : int, optional
        Random seed for reproducibility
    """
    bbox: np.ndarray
    density_x: Optional[float] = None
    density_y: Optional[float] = None
    density_z: Optional[float] = None
    n_particles: Optional[int] = None
    distribution: str = 'uniform'
    jitter: float = 0.0
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.n_particles is None and (self.density_x is None or 
                                          self.density_y is None or 
                                          self.density_z is None):
            raise ValueError("Must specify either n_particles or all three densities")
        
        if self.distribution not in ['uniform', 'random', 'stratified']:
            raise ValueError(f"Invalid distribution: {self.distribution}")
        
        if not (0.0 <= self.jitter <= 0.5):
            raise ValueError("Jitter must be in range [0.0, 0.5]")
    
    def compute_particle_count(self) -> int:
        """
        Compute total particle count based on density or explicit count.
        
        Returns
        -------
        n_particles : int
            Total number of particles to seed
        """
        if self.n_particles is not None:
            return self.n_particles
        
        # Compute from density
        dx = self.bbox[1] - self.bbox[0]
        dy = self.bbox[3] - self.bbox[2]
        dz = self.bbox[5] - self.bbox[4]
        
        nx = int(np.ceil(dx * self.density_x))
        ny = int(np.ceil(dy * self.density_y))
        nz = int(np.ceil(dz * self.density_z))
        
        return nx * ny * nz
    
    def compute_grid_spacing(self) -> Tuple[float, float, float]:
        """
        Compute grid spacing for uniform distribution.
        
        Returns
        -------
        hx, hy, hz : float
            Grid spacing in each direction
        """
        dx = self.bbox[1] - self.bbox[0]
        dy = self.bbox[3] - self.bbox[2]
        dz = self.bbox[5] - self.bbox[4]
        
        if self.n_particles is not None:
            # Distribute particles proportionally based on domain aspect ratio
            volume = dx * dy * dz
            density_total = self.n_particles / volume
            
            # Equal spacing per unit length
            spacing = (1.0 / density_total) ** (1/3)
            return spacing, spacing, spacing
        else:
            # Use specified densities
            hx = 1.0 / self.density_x
            hy = 1.0 / self.density_y
            hz = 1.0 / self.density_z
            return hx, hy, hz


def seed_particles_uniform(
    bbox: np.ndarray,
    density_x: Optional[float] = None,
    density_y: Optional[float] = None,
    density_z: Optional[float] = None,
    n_particles: Optional[int] = None,
    jitter: float = 0.0,
    seed: Optional[int] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Seed particles on a uniform grid with optional jitter.
    
    Parameters
    ----------
    bbox : np.ndarray
        Seeding bounding box [xmin, xmax, ymin, ymax, zmin, zmax]
    density_x : float, optional
        Particles per unit length in x (particles/meter)
    density_y : float, optional
        Particles per unit length in y (particles/meter)
    density_z : float, optional
        Particles per unit length in z (particles/meter)
    n_particles : int, optional
        Total particle count (overrides density)
    jitter : float, optional
        Random perturbation factor [0.0, 0.5] (default: 0.0 = no jitter)
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, optional
        Print seeding information
        
    Returns
    -------
    positions : np.ndarray
        Particle positions, shape (N, 3), float32
        
    Examples
    --------
    # Density-based seeding
    >>> positions = seed_particles_uniform(
    ...     bbox=np.array([0, 0.01, 0, 0.01, 0, 0.01]),
    ...     density_x=1000,  # 1000 particles/meter = 1 particle/mm
    ...     density_y=1000,
    ...     density_z=500,   # 0.5 particles/mm in z
    ... )
    
    # Count-based seeding
    >>> positions = seed_particles_uniform(
    ...     bbox=domain_bounds,
    ...     n_particles=10000
    ... )
    
    # With jitter for better coverage
    >>> positions = seed_particles_uniform(
    ...     bbox=domain_bounds,
    ...     density_x=1000, density_y=1000, density_z=1000,
    ...     jitter=0.2  # 20% random perturbation
    ... )
    """
    config = SeedingConfig(
        bbox=bbox,
        density_x=density_x,
        density_y=density_y,
        density_z=density_z,
        n_particles=n_particles,
        distribution='uniform',
        jitter=jitter,
        seed=seed,
    )
    
    if seed is not None:
        np.random.seed(seed)
    
    # Compute grid parameters
    dx = bbox[1] - bbox[0]
    dy = bbox[3] - bbox[2]
    dz = bbox[5] - bbox[4]
    
    if n_particles is not None:
        # Distribute based on total count
        volume = dx * dy * dz
        density_total = n_particles / volume
        spacing = (1.0 / density_total) ** (1/3)
        
        nx = max(1, int(np.ceil(dx / spacing)))
        ny = max(1, int(np.ceil(dy / spacing)))
        nz = max(1, int(np.ceil(dz / spacing)))
        
        hx = dx / nx
        hy = dy / ny
        hz = dz / nz
    else:
        # Use specified densities
        nx = max(1, int(np.ceil(dx * density_x)))
        ny = max(1, int(np.ceil(dy * density_y)))
        nz = max(1, int(np.ceil(dz * density_z)))
        
        hx = dx / nx
        hy = dy / ny
        hz = dz / nz
    
    if verbose:
        print(f"\nUniform particle seeding:")
        print(f"  Bounding box: [{bbox[0]:.4f}, {bbox[1]:.4f}] × "
              f"[{bbox[2]:.4f}, {bbox[3]:.4f}] × [{bbox[4]:.4f}, {bbox[5]:.4f}]")
        print(f"  Grid: {nx} × {ny} × {nz} = {nx*ny*nz:,} particles")
        print(f"  Spacing: hx={hx:.6f}, hy={hy:.6f}, hz={hz:.6f}")
        if jitter > 0:
            print(f"  Jitter: {jitter*100:.1f}%")
    
    # Create grid
    x = np.linspace(bbox[0] + hx/2, bbox[1] - hx/2, nx)
    y = np.linspace(bbox[2] + hy/2, bbox[3] - hy/2, ny)
    z = np.linspace(bbox[4] + hz/2, bbox[5] - hz/2, nz)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(config.FLOAT_DTYPE_NP)
    
    # Apply jitter if requested
    if jitter > 0:
        perturbation = np.random.uniform(
            -jitter, jitter, size=positions.shape
        ).astype(config.FLOAT_DTYPE_NP)
        
        # Scale perturbation by grid spacing
        perturbation[:, 0] *= hx
        perturbation[:, 1] *= hy
        perturbation[:, 2] *= hz
        
        positions += perturbation
        
        # Clamp to bounding box
        positions[:, 0] = np.clip(positions[:, 0], bbox[0], bbox[1])
        positions[:, 1] = np.clip(positions[:, 1], bbox[2], bbox[3])
        positions[:, 2] = np.clip(positions[:, 2], bbox[4], bbox[5])
    
    return positions


def seed_particles_random(
    bbox: np.ndarray,
    n_particles: int,
    seed: Optional[int] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Seed particles randomly within bounding box.
    
    Parameters
    ----------
    bbox : np.ndarray
        Seeding bounding box [xmin, xmax, ymin, ymax, zmin, zmax]
    n_particles : int
        Number of particles to seed
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, optional
        Print seeding information
        
    Returns
    -------
    positions : np.ndarray
        Particle positions, shape (N, 3), float32
    """
    if seed is not None:
        np.random.seed(seed)
    
    if verbose:
        print(f"\nRandom particle seeding:")
        print(f"  Bounding box: [{bbox[0]:.4f}, {bbox[1]:.4f}] × "
              f"[{bbox[2]:.4f}, {bbox[3]:.4f}] × [{bbox[4]:.4f}, {bbox[5]:.4f}]")
        print(f"  Particles: {n_particles:,}")
    
    positions = np.random.uniform(
        low=[bbox[0], bbox[2], bbox[4]],
        high=[bbox[1], bbox[3], bbox[5]],
        size=(n_particles, 3)
    ).astype(config.FLOAT_DTYPE_NP)
    
    return positions


def seed_particles_stratified(
    bbox: np.ndarray,
    density_x: float,
    density_y: float,
    density_z: float,
    seed: Optional[int] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Seed particles using stratified sampling.
    
    Combines uniform grid structure with random perturbation within each cell.
    Better coverage than pure random, less regular than uniform grid.
    
    Parameters
    ----------
    bbox : np.ndarray
        Seeding bounding box [xmin, xmax, ymin, ymax, zmin, zmax]
    density_x : float
        Particles per unit length in x
    density_y : float
        Particles per unit length in y
    density_z : float
        Particles per unit length in z
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, optional
        Print seeding information
        
    Returns
    -------
    positions : np.ndarray
        Particle positions, shape (N, 3), float32
        
    Notes
    -----
    Stratified sampling divides the domain into cells and places one particle
    randomly within each cell. This provides better statistical coverage than
    pure random sampling while avoiding regularity artifacts of uniform grids.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Start with uniform grid centers
    positions = seed_particles_uniform(
        bbox, density_x, density_y, density_z, 
        jitter=0.0, seed=None, verbose=False
    )
    
    # Compute cell sizes
    dx = bbox[1] - bbox[0]
    dy = bbox[3] - bbox[2]
    dz = bbox[5] - bbox[4]
    
    nx = int(np.ceil(dx * density_x))
    ny = int(np.ceil(dy * density_y))
    nz = int(np.ceil(dz * density_z))
    
    hx = dx / nx
    hy = dy / ny
    hz = dz / nz
    
    # Random perturbation within each cell
    perturbation = np.random.uniform(
        [-hx/2, -hy/2, -hz/2],
        [hx/2, hy/2, hz/2],
        size=positions.shape
    ).astype(config.FLOAT_DTYPE_NP)
    
    positions += perturbation
    
    # Clamp to bounding box
    positions[:, 0] = np.clip(positions[:, 0], bbox[0], bbox[1])
    positions[:, 1] = np.clip(positions[:, 1], bbox[2], bbox[3])
    positions[:, 2] = np.clip(positions[:, 2], bbox[4], bbox[5])
    
    if verbose:
        print(f"\nStratified particle seeding:")
        print(f"  Bounding box: [{bbox[0]:.4f}, {bbox[1]:.4f}] × "
              f"[{bbox[2]:.4f}, {bbox[3]:.4f}] × [{bbox[4]:.4f}, {bbox[5]:.4f}]")
        print(f"  Grid: {nx} × {ny} × {nz} = {positions.shape[0]:,} particles")
        print(f"  Cell size: hx={hx:.6f}, hy={hy:.6f}, hz={hz:.6f}")
        print(f"  Each cell contains one randomly placed particle")
    
    return positions


def seed_particles(
    config: SeedingConfig,
    verbose: bool = False
) -> np.ndarray:
    """
    Unified particle seeding with configuration object.
    
    Parameters
    ----------
    config : SeedingConfig
        Seeding configuration
    verbose : bool, optional
        Print seeding information
        
    Returns
    -------
    positions : np.ndarray
        Particle positions, shape (N, 3), float32
    """
    if config.distribution == 'uniform':
        return seed_particles_uniform(
            config.bbox,
            config.density_x,
            config.density_y,
            config.density_z,
            config.n_particles,
            config.jitter,
            config.seed,
            verbose
        )
    elif config.distribution == 'random':
        n = config.compute_particle_count()
        return seed_particles_random(
            config.bbox, n, config.seed, verbose
        )
    elif config.distribution == 'stratified':
        return seed_particles_stratified(
            config.bbox,
            config.density_x,
            config.density_y,
            config.density_z,
            config.seed,
            verbose
        )
    else:
        raise ValueError(f"Unknown distribution: {config.distribution}")


def compute_particle_density(
    positions: np.ndarray,
    bbox: Optional[np.ndarray] = None
) -> Tuple[float, float, float, float]:
    """
    Compute particle density statistics.
    
    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (N, 3)
    bbox : np.ndarray, optional
        Bounding box for density calculation
        If None, uses positions bounding box
        
    Returns
    -------
    density_total : float
        Particles per unit volume (particles/m³)
    density_x : float
        Particles per unit length in x (particles/m)
    density_y : float
        Particles per unit length in y (particles/m)
    density_z : float
        Particles per unit length in z (particles/m)
    """
    if bbox is None:
        bbox = np.array([
            positions[:, 0].min(), positions[:, 0].max(),
            positions[:, 1].min(), positions[:, 1].max(),
            positions[:, 2].min(), positions[:, 2].max(),
        ])
    
    dx = bbox[1] - bbox[0]
    dy = bbox[3] - bbox[2]
    dz = bbox[5] - bbox[4]
    volume = dx * dy * dz
    
    n = positions.shape[0]
    
    density_total = n / volume if volume > 0 else 0
    density_x = n / dx if dx > 0 else 0
    density_y = n / dy if dy > 0 else 0
    density_z = n / dz if dz > 0 else 0
    
    return density_total, density_x, density_y, density_z
