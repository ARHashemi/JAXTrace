"""
Forest Block Visualization.

Visualize forest-of-octrees block decomposition with optional particle overlay.
"""

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .block_builder import BlockMetadata


def visualize_forest_blocks(
    blocks: List[BlockMetadata],
    particle_positions: Optional[np.ndarray] = None,
    save_path: str = 'forest_blocks.png',
    title: str = 'Forest-of-Octrees Block Decomposition',
    show_block_ids: bool = True,
    figsize: Tuple[int, int] = (18, 14)
):
    """
    Visualize forest block decomposition with 3D view and 2D projections.

    Creates a 4-panel figure:
    - Top-left: 3D wireframe view
    - Top-right: XY projection
    - Bottom-left: XZ projection
    - Bottom-right: YZ projection

    Args:
        blocks: List of BlockMetadata
        particle_positions: Optional particle positions [N, 3] to overlay
        save_path: Output file path
        title: Figure title
        show_block_ids: If True, label blocks with IDs
        figsize: Figure size (width, height)

    Example:
        >>> from jaxtrace.gpu.forest import create_regular_forest_grid
        >>> blocks = create_regular_forest_grid(bounds, (4, 4, 2))
        >>> visualize_forest_blocks(blocks, save_path='forest.png')
    """
    fig = plt.figure(figsize=figsize)

    # 3D view
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    _plot_blocks_3d(ax3d, blocks, particle_positions, show_block_ids)
    ax3d.set_title(f'{title} (3D View)')

    # XY projection
    ax_xy = fig.add_subplot(2, 2, 2)
    _plot_blocks_2d(ax_xy, blocks, particle_positions, proj_axes=(0, 1), labels=('X (m)', 'Y (m)'))
    ax_xy.set_title('XY Projection')

    # XZ projection
    ax_xz = fig.add_subplot(2, 2, 3)
    _plot_blocks_2d(ax_xz, blocks, particle_positions, proj_axes=(0, 2), labels=('X (m)', 'Z (m)'))
    ax_xz.set_title('XZ Projection')

    # YZ projection
    ax_yz = fig.add_subplot(2, 2, 4)
    _plot_blocks_2d(ax_yz, blocks, particle_positions, proj_axes=(1, 2), labels=('Y (m)', 'Z (m)'))
    ax_yz.set_title('YZ Projection')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Forest visualization saved to: {save_path}")

    return fig


def _plot_blocks_3d(
    ax: Axes3D,
    blocks: List[BlockMetadata],
    particles: Optional[np.ndarray],
    show_ids: bool
):
    """Plot blocks in 3D with wireframe boxes."""
    for block in blocks:
        b = block.bounds

        # Create box vertices
        vertices = np.array([
            [b[0], b[2], b[4]], [b[1], b[2], b[4]],  # bottom face
            [b[1], b[3], b[4]], [b[0], b[3], b[4]],
            [b[0], b[2], b[5]], [b[1], b[2], b[5]],  # top face
            [b[1], b[3], b[5]], [b[0], b[3], b[5]]
        ])

        # Draw box edges
        edges = [
            [0,1],[1,2],[2,3],[3,0],  # bottom
            [4,5],[5,6],[6,7],[7,4],  # top
            [0,4],[1,5],[2,6],[3,7]   # vertical
        ]

        for edge in edges:
            points = vertices[edge]
            ax.plot3D(*points.T, 'b-', alpha=0.4, linewidth=1.0)

        # Label block ID at center
        if show_ids:
            c = block.center
            ax.text(c[0], c[1], c[2], str(block.block_id),
                   fontsize=6, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Overlay particles if provided
    if particles is not None:
        ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2],
                  c='red', s=5, alpha=0.6, label='Particles')
        ax.legend()

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.grid(True, alpha=0.3)


def _plot_blocks_2d(
    ax: plt.Axes,
    blocks: List[BlockMetadata],
    particles: Optional[np.ndarray],
    proj_axes: Tuple[int, int],
    labels: Tuple[str, str]
):
    """Plot blocks in 2D projection."""
    # Track bounds for axis limits
    all_xmin, all_xmax = float('inf'), float('-inf')
    all_ymin, all_ymax = float('inf'), float('-inf')

    for block in blocks:
        b = block.bounds
        ax0, ax1 = proj_axes

        # Create rectangle
        xmin = b[ax0 * 2]
        xmax = b[ax0 * 2 + 1]
        ymin = b[ax1 * 2]
        ymax = b[ax1 * 2 + 1]

        rect = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, edgecolor='blue', linewidth=1.5, alpha=0.8
        )
        ax.add_patch(rect)

        # Add block ID label at center
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        ax.text(center_x, center_y, str(block.block_id),
               fontsize=6, ha='center', va='center',
               color='blue', alpha=0.7)

        # Update bounds
        all_xmin = min(all_xmin, xmin)
        all_xmax = max(all_xmax, xmax)
        all_ymin = min(all_ymin, ymin)
        all_ymax = max(all_ymax, ymax)

    # Set axis limits with small margin
    margin_x = (all_xmax - all_xmin) * 0.05
    margin_y = (all_ymax - all_ymin) * 0.05
    ax.set_xlim(all_xmin - margin_x, all_xmax + margin_x)
    ax.set_ylim(all_ymin - margin_y, all_ymax + margin_y)

    # Overlay particles if provided
    if particles is not None:
        ax.scatter(particles[:, proj_axes[0]], particles[:, proj_axes[1]],
                  c='red', s=2, alpha=0.5, label='Particles')
        ax.legend()

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def visualize_forest_with_mesh_pieces(
    blocks: List[BlockMetadata],
    mesh_piece_bounds: List[np.ndarray],
    save_path: str = 'forest_with_mesh.png',
    figsize: Tuple[int, int] = (18, 14)
):
    """
    Visualize forest blocks overlaid on VTK mesh piece decomposition.

    Useful for comparing forest partitioning with existing mesh decomposition.

    Args:
        blocks: Forest block metadata
        mesh_piece_bounds: List of [xmin, xmax, ymin, ymax, zmin, zmax] for mesh pieces
        save_path: Output file path
        figsize: Figure size

    Example:
        >>> # Load mesh piece bounds from VTK
        >>> mesh_bounds = [piece.GetBounds() for piece in mesh_pieces]
        >>> visualize_forest_with_mesh_pieces(blocks, mesh_bounds)
    """
    fig = plt.figure(figsize=figsize)

    ax3d = fig.add_subplot(1, 1, 1, projection='3d')

    # Plot forest blocks (blue)
    for block in blocks:
        b = block.bounds
        vertices = np.array([
            [b[0], b[2], b[4]], [b[1], b[2], b[4]],
            [b[1], b[3], b[4]], [b[0], b[3], b[4]],
            [b[0], b[2], b[5]], [b[1], b[2], b[5]],
            [b[1], b[3], b[5]], [b[0], b[3], b[5]]
        ])

        edges = [
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]
        ]

        for edge in edges:
            points = vertices[edge]
            ax3d.plot3D(*points.T, 'b-', alpha=0.6, linewidth=2.0, label='Forest' if block.block_id == 0 else '')

    # Plot mesh pieces (orange)
    for i, b in enumerate(mesh_piece_bounds):
        vertices = np.array([
            [b[0], b[2], b[4]], [b[1], b[2], b[4]],
            [b[1], b[3], b[4]], [b[0], b[3], b[4]],
            [b[0], b[2], b[5]], [b[1], b[2], b[5]],
            [b[1], b[3], b[5]], [b[0], b[3], b[5]]
        ])

        edges = [
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]
        ]

        for edge in edges:
            points = vertices[edge]
            ax3d.plot3D(*points.T, 'orange', alpha=0.3, linewidth=0.5, label='Mesh Pieces' if i == 0 else '')

    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title('Forest Blocks (blue) vs Mesh Pieces (orange)')
    ax3d.legend()
    ax3d.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Forest + mesh visualization saved to: {save_path}")

    return fig
