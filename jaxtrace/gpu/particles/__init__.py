"""
Particle management for GPU-native particle tracking.

Part of Phase 3: Particle Seeding & Initial Assignment
"""

from .seeding import (
    ParticleState,
    SeedingConfig,
    seed_particles,
    seed_particles_uniform,
    seed_particles_random,
    seed_particles_stratified,
    compute_particle_density,
)

__all__ = [
    "ParticleState",
    "SeedingConfig",
    "seed_particles",
    "seed_particles_uniform",
    "seed_particles_random",
    "seed_particles_stratified",
    "compute_particle_density",
]
