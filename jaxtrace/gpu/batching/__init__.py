"""
Batched block-wise particle tracking for JAX GPU.

This module implements a two-level batching architecture:
1. Particle batching: Process particles in batches to prevent OOM
2. Block-wise processing: Within each batch, group particles by block

Key components:
- validation: Mesh validation and heavy block detection
- memory_utils: VRAM monitoring and batch size calculation
- block_grouping: Particle grouping by block
- batch_config: Configuration with auto-tuning
- batch_processor: Main batching loop

Architecture:
docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md
"""

from .validation import (
    validate_mesh_for_gpu,
    detect_block_imbalance,
    MeshValidationResult,
)

__all__ = [
    'validate_mesh_for_gpu',
    'detect_block_imbalance',
    'MeshValidationResult',
]
