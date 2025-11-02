"""
Configuration for GPU Forest-of-Octrees Particle Tracking.

This module provides configuration dataclasses for GPU-native particle tracking
with forest-of-octrees spatial decomposition.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import yaml


@dataclass
class GPUForestConfig:
    """
    Configuration for GPU forest-of-octrees particle tracking.

    This configuration controls all aspects of the GPU-native particle tracker,
    including forest partitioning, memory allocation, field selection, and
    performance tuning.

    Attributes:
        block_grid: Forest grid size (nx, ny, nz). Default (4,4,2) = 32 blocks.
                   Recommended for 3.5M cell mesh. Increase to (8,8,4) for
                   higher GPU utilization if needed.

        max_octree_depth: Maximum octree refinement level. Default 12 provides
                         good balance between memory and search performance.

        field_name: Name of velocity field in VTK files. Default "Displacement"
                   is the primary field for ThreadedA mesh.

        auto_detect_field: If True, automatically detect vector fields if
                          field_name not found.

        revolution_cycle: Timestep range (start, end) for revolution cycle.
                         None means auto-detect from mesh. For ThreadedA,
                         this is typically (120, 159).

        build_forest_from_timestep: Which timestep to use for forest construction.
                                   -1 means auto-detect most refined timestep.

        max_particles_per_block: Maximum particles per block (for padding).
                                Default 10000 is safe for 4GB VRAM.

        ghost_layer_thickness: Number of element layers for ghost regions.
                              1 layer is sufficient for most cases.

        skip_empty_blocks: If True, skip blocks with no elements during
                          GPU kernel execution.

        enable_load_balancing: If True, enable dynamic block splitting for
                              load balancing. (Phase 8 feature)

        save_trajectory: If True, save full trajectory to memory/disk.

        trajectory_stride: Save trajectory every N timesteps (1 = every step).

    Examples:
        # Default configuration (32 blocks, suitable for ThreadedA)
        >>> config = GPUForestConfig()

        # High GPU utilization (256 blocks)
        >>> config = GPUForestConfig(block_grid=(8, 8, 4))

        # Memory-efficient (8 blocks)
        >>> config = GPUForestConfig(block_grid=(2, 2, 2))

        # Load from YAML
        >>> config = GPUForestConfig.from_yaml("config.yaml")
    """

    # Block configuration (user-tunable)
    block_grid: Tuple[int, int, int] = (4, 4, 2)
    """Forest grid size. (4,4,2) = 32 blocks for production mesh."""

    max_octree_depth: int = 12
    """Maximum octree refinement level."""

    # Field configuration
    field_name: str = "Displacement"
    """Velocity field name in VTK (ThreadedA uses 'Displacement')."""

    auto_detect_field: bool = True
    """Auto-detect vector fields if field_name not found."""

    # Timestep configuration
    revolution_cycle: Optional[Tuple[int, int]] = None
    """Timestep range (start, end). None = auto-detect."""

    build_forest_from_timestep: int = -1
    """-1 = auto-detect most refined timestep."""

    # Memory configuration
    max_particles_per_block: int = 10000
    """Maximum particles per block (for padding)."""

    ghost_layer_thickness: int = 1
    """Ghost region thickness (1 layer sufficient)."""

    # Performance tuning (Phase 8 features)
    skip_empty_blocks: bool = True
    """Skip empty blocks during GPU execution."""

    enable_load_balancing: bool = False
    """Enable dynamic block splitting (Phase 8)."""

    # Output configuration
    save_trajectory: bool = True
    """Save full trajectory to memory."""

    trajectory_stride: int = 1
    """Save every N timesteps (1 = all)."""

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate block grid
        if len(self.block_grid) != 3:
            raise ValueError(f"block_grid must be (nx, ny, nz), got {self.block_grid}")

        if any(n <= 0 for n in self.block_grid):
            raise ValueError(f"block_grid dimensions must be > 0, got {self.block_grid}")

        # Validate max_octree_depth
        if self.max_octree_depth < 1 or self.max_octree_depth > 20:
            raise ValueError(f"max_octree_depth must be in [1, 20], got {self.max_octree_depth}")

        # Validate max_particles_per_block
        if self.max_particles_per_block <= 0:
            raise ValueError(f"max_particles_per_block must be > 0, got {self.max_particles_per_block}")

        # Validate ghost_layer_thickness
        if self.ghost_layer_thickness < 0:
            raise ValueError(f"ghost_layer_thickness must be >= 0, got {self.ghost_layer_thickness}")

        # Validate trajectory_stride
        if self.trajectory_stride < 1:
            raise ValueError(f"trajectory_stride must be >= 1, got {self.trajectory_stride}")

    @property
    def n_blocks(self) -> int:
        """Total number of forest blocks."""
        return self.block_grid[0] * self.block_grid[1] * self.block_grid[2]

    @classmethod
    def from_yaml(cls, filepath: str) -> "GPUForestConfig":
        """
        Load configuration from YAML file.

        Args:
            filepath: Path to YAML configuration file

        Returns:
            GPUForestConfig instance

        Example YAML:
            forest:
              block_grid: [4, 4, 2]
              max_octree_depth: 12

            field:
              name: "Displacement"
              auto_detect: true

            memory:
              max_particles_per_block: 10000
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        # Flatten nested structure
        config_dict = {}

        if 'forest' in data:
            config_dict['block_grid'] = tuple(data['forest'].get('block_grid', (4, 4, 2)))
            config_dict['max_octree_depth'] = data['forest'].get('max_octree_depth', 12)
            config_dict['ghost_layer_thickness'] = data['forest'].get('ghost_layer_thickness', 1)

        if 'field' in data:
            config_dict['field_name'] = data['field'].get('name', 'Displacement')
            config_dict['auto_detect_field'] = data['field'].get('auto_detect', True)

        if 'mesh' in data:
            if 'revolution_cycle' in data['mesh'] and data['mesh']['revolution_cycle']:
                config_dict['revolution_cycle'] = tuple(data['mesh']['revolution_cycle'])
            config_dict['build_forest_from_timestep'] = data['mesh'].get('build_from_timestep', -1)

        if 'memory' in data:
            config_dict['max_particles_per_block'] = data['memory'].get('max_particles_per_block', 10000)

        if 'performance' in data:
            config_dict['skip_empty_blocks'] = data['performance'].get('skip_empty_blocks', True)
            config_dict['enable_load_balancing'] = data['performance'].get('enable_load_balancing', False)

        if 'output' in data:
            config_dict['save_trajectory'] = data['output'].get('save_trajectory', True)
            config_dict['trajectory_stride'] = data['output'].get('trajectory_stride', 1)

        return cls(**config_dict)

    def to_yaml(self, filepath: str):
        """
        Save configuration to YAML file.

        Args:
            filepath: Path to output YAML file
        """
        data = {
            'forest': {
                'block_grid': list(self.block_grid),
                'max_octree_depth': self.max_octree_depth,
                'ghost_layer_thickness': self.ghost_layer_thickness,
            },
            'field': {
                'name': self.field_name,
                'auto_detect': self.auto_detect_field,
            },
            'mesh': {
                'revolution_cycle': list(self.revolution_cycle) if self.revolution_cycle else None,
                'build_from_timestep': self.build_forest_from_timestep,
            },
            'memory': {
                'max_particles_per_block': self.max_particles_per_block,
            },
            'performance': {
                'skip_empty_blocks': self.skip_empty_blocks,
                'enable_load_balancing': self.enable_load_balancing,
            },
            'output': {
                'save_trajectory': self.save_trajectory,
                'trajectory_stride': self.trajectory_stride,
            }
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)

    def __str__(self) -> str:
        """Pretty print configuration."""
        return f"""GPUForestConfig:
  Forest:
    Block grid: {self.block_grid[0]}×{self.block_grid[1]}×{self.block_grid[2]} = {self.n_blocks} blocks
    Max octree depth: {self.max_octree_depth}
    Ghost layer: {self.ghost_layer_thickness}

  Field:
    Name: '{self.field_name}'
    Auto-detect: {self.auto_detect_field}

  Mesh:
    Revolution cycle: {self.revolution_cycle}
    Build from timestep: {self.build_forest_from_timestep}

  Memory:
    Max particles/block: {self.max_particles_per_block:,}

  Performance:
    Skip empty blocks: {self.skip_empty_blocks}
    Load balancing: {self.enable_load_balancing}

  Output:
    Save trajectory: {self.save_trajectory}
    Stride: {self.trajectory_stride}
"""
