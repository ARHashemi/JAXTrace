"""
GPU-Native Particle Tracking using Forest-of-Octrees Architecture.

This package implements high-performance particle tracking on GPU using JAX
with a forest-of-octrees spatial decomposition strategy.

V3 Implementation - Phase 0 (Infrastructure)

Key Features (planned):
- Flat array data structures optimized for JAX
- Morton code spatial partitioning
- Multi-level element search (cached → neighbors → block)
- Minimal scan carry (positions, element_IDs, active only)
- Memory-efficient design for 8GB GPU

Current Status: Phase 0 - Infrastructure and analysis tools
"""

__version__ = "0.3.0-alpha-phase0"
__author__ = "JAXTrace Development Team"

# Phase 0: Infrastructure and analysis
# Note: Old GPU implementation (Phase 2) archived to archive/gpu_v1_old/
# New V3 implementation will be built phase-by-phase starting from Phase 1

# Phase 0: Configuration
from .config import GPUForestConfig

# Phase 3: Particles (new clean implementation)
# from .particles import (
#     ParticleState,
#     SeedingConfig,
#     seed_particles,
#     seed_particles_uniform,
# )

# Phase 0: Analysis tools (new V3)
# mesh_analysis.py - Mesh statistics and GPU config recommendations
# test_meshes.py - Synthetic test mesh generators

__all__ = [
    "GPUForestConfig",
    # "ParticleState",
    # "SeedingConfig",
]
