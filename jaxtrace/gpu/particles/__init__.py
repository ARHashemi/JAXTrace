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

# Import ParticleData from parent's standalone particles.py using importlib
import importlib.util
import os

_particles_py_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'particles.py')
_spec = importlib.util.spec_from_file_location("_particles_standalone", _particles_py_path)
_particles_standalone = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_particles_standalone)
ParticleData = _particles_standalone.ParticleData

__all__ = [
    "ParticleState",
    "ParticleData",  # Add ParticleData to exports
    "SeedingConfig",
    "seed_particles",
    "seed_particles_uniform",
    "seed_particles_random",
    "seed_particles_stratified",
    "compute_particle_density",
]
