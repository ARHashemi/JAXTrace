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

from .memory_utils import (
    get_gpu_memory_info,
    calculate_safe_batch_size,
    GPUMemoryInfo,
)

from .block_grouping import (
    group_particles_by_block,
    batch_light_blocks,
    ParticleGrouping,
)

from .batch_config import (
    BatchConfig,
    create_default_config,
    validate_config,
    print_config_summary,
    suggest_config_improvements,
)

from .batch_processor import (
    process_batch,
    track_particles_batched,
    print_batch_statistics,
    BatchStatistics,
    ProcessorStatistics,
)

__all__ = [
    # Validation
    'validate_mesh_for_gpu',
    'detect_block_imbalance',
    'MeshValidationResult',
    # Memory
    'get_gpu_memory_info',
    'calculate_safe_batch_size',
    'GPUMemoryInfo',
    # Grouping
    'group_particles_by_block',
    'batch_light_blocks',
    'ParticleGrouping',
    # Config
    'BatchConfig',
    'create_default_config',
    'validate_config',
    'print_config_summary',
    'suggest_config_improvements',
    # Processor
    'process_batch',
    'track_particles_batched',
    'print_batch_statistics',
    'BatchStatistics',
    'ProcessorStatistics',
]
