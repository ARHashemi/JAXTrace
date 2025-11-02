"""
GPU-Native Particle Tracking using Forest-of-Octrees Architecture.

This package implements high-performance particle tracking on GPU using JAX
with a forest-of-octrees spatial decomposition strategy.

Key Features:
- Forest-of-octrees domain decomposition
- Block-level spatial batching
- Element ID caching with three-tier search
- Pure JAX GPU kernels with nested vmap
- Memory-efficient design for 4GB VRAM

Main Components:
- config: Configuration dataclasses
- forest: Forest block management and spatial partitioning
- kernels: GPU particle update kernels
- time_marching: Time integration with lax.scan

Usage:
    from jaxtrace.gpu import GPUForestTracker

    tracker = GPUForestTracker(
        mesh_path="path/to/mesh.pvtu",
        block_grid=(4, 4, 2),
        field_name="Displacement"
    )

    trajectories = tracker.track(seeds, timesteps, dt)
"""

__version__ = "0.1.0-alpha"
__author__ = "JAXTrace Development Team"

# Phase 0: Configuration only
from .config import GPUForestConfig

__all__ = ["GPUForestConfig"]
