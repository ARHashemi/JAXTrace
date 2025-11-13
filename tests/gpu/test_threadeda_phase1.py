"""
Phase 1 integration test with ThreadedA mesh.

Tests block grid generation and element-to-block assignment on real mesh.
"""

import pytest
import numpy as np
from pathlib import Path

from jaxtrace.io import VTKUnstructuredTimeSeriesReader
from jaxtrace.gpu.forest.block_grid import create_regular_grid
from jaxtrace.gpu.forest.block_mapper import assign_elements_to_block_list, validate_assignment


# ThreadedA mesh location
THREADEDA_DIR = Path("../Edgar/ThreadedA/post/0eule").expanduser().resolve()
THREADEDA_PATTERN = str(THREADEDA_DIR / "threadedAvtk_*.pvtu")


@pytest.mark.skipif(not THREADEDA_DIR.exists(), reason="ThreadedA mesh not available")
class TestThreadedAPhase1:
    """Integration tests with ThreadedA mesh."""

    def test_load_threadeda_mesh(self):
        """Test that ThreadedA mesh can be loaded."""
        reader = VTKUnstructuredTimeSeriesReader(THREADEDA_PATTERN)
        timesteps = reader.get_timesteps()

        assert len(timesteps) > 0, "No timesteps found"

        t0 = timesteps[0]
        nodes = reader.read_nodes(t0)
        connectivity = reader.read_connectivity(t0)

        print(f"\nThreadedA mesh loaded:")
        print(f"  Nodes: {nodes.shape[0]:,}")
        print(f"  Elements: {connectivity.shape[0]:,}")
        print(f"  Element type: {connectivity.shape[1]} nodes per element")

        assert nodes.shape[0] > 0
        assert connectivity.shape[0] > 0
        assert connectivity.shape[1] == 4  # Tetrahedral elements

    def test_create_block_grid_threadeda(self):
        """Test block grid creation for ThreadedA domain."""
        # From Phase 0 analysis
        domain_bounds = np.array([-0.03, 0.03, -0.023, 0.023, -0.01, 0.0], dtype=np.float32)
        grid_size = (4, 4, 2)  # 32 blocks

        blocks = create_regular_grid(domain_bounds, grid_size)

        assert len(blocks) == 32
        print(f"\nCreated {len(blocks)} blocks for ThreadedA")
        print(f"Block 0 bounds: {blocks[0].bounds}")
        print(f"Block 0 volume: {blocks[0].volume:.6e}")

    def test_assign_elements_threadeda(self):
        """Test element-to-block assignment for ThreadedA."""
        # Load mesh
        reader = VTKUnstructuredTimeSeriesReader(THREADEDA_PATTERN)
        t0 = reader.get_timesteps()[0]
        nodes = reader.read_nodes(t0)
        connectivity = reader.read_connectivity(t0)

        print(f"\nAssigning {connectivity.shape[0]:,} elements to blocks...")

        # Create blocks
        domain_bounds = np.array([-0.03, 0.03, -0.023, 0.023, -0.01, 0.0], dtype=np.float32)
        grid_size = (4, 4, 2)
        blocks = create_regular_grid(domain_bounds, grid_size)

        # Assign elements
        element_to_block, stats = assign_elements_to_block_list(
            nodes, connectivity, blocks,
            heavy_threshold=10000,
            verbose=True
        )

        # Verify results
        assert element_to_block.shape[0] == connectivity.shape[0]
        assert stats.n_elements == connectivity.shape[0]
        assert stats.n_blocks == 32

        print(f"\n{stats}")

        # Check Phase 0 predictions
        print(f"\nPhase 0 predicted ~110K elements/block")
        print(f"Actual mean: {stats.mean_elements:,.0f}")
        print(f"Imbalance ratio: {stats.imbalance_ratio:.2f}x")

        # Verify no elements outside domain (or very few due to numerical precision)
        n_outside = np.sum(element_to_block == -1)
        pct_outside = 100 * n_outside / connectivity.shape[0]
        print(f"Elements outside domain: {n_outside} ({pct_outside:.4f}%)")
        assert pct_outside < 0.1, f"Too many elements outside domain: {pct_outside:.2f}%"

        # Check for heavy blocks
        if stats.heavy_blocks:
            print(f"\nHeavy blocks detected: {stats.heavy_blocks}")
            print("These will require hash bucket subdivision in Phase 4")

        return element_to_block, stats

    def test_validate_assignment_threadeda(self):
        """Test validation of element-to-block assignment."""
        # Load mesh
        reader = VTKUnstructuredTimeSeriesReader(THREADEDA_PATTERN)
        t0 = reader.get_timesteps()[0]
        nodes = reader.read_nodes(t0)
        connectivity = reader.read_connectivity(t0)

        # Create blocks
        domain_bounds = np.array([-0.03, 0.03, -0.023, 0.023, -0.01, 0.0], dtype=np.float32)
        grid_size = (4, 4, 2)
        blocks = create_regular_grid(domain_bounds, grid_size)

        # Assign elements
        element_to_block, stats = assign_elements_to_block_list(
            nodes, connectivity, blocks, verbose=False
        )

        # Validate assignment
        print(f"\nValidating element-to-block assignment (sampling 1000 elements)...")
        valid = validate_assignment(element_to_block, nodes, connectivity, blocks, n_samples=1000)

        assert valid, "Validation failed"

    def test_block_occupancy_distribution(self):
        """Test and visualize block occupancy distribution."""
        # Load mesh
        reader = VTKUnstructuredTimeSeriesReader(THREADEDA_PATTERN)
        t0 = reader.get_timesteps()[0]
        nodes = reader.read_nodes(t0)
        connectivity = reader.read_connectivity(t0)

        # Create blocks and assign
        domain_bounds = np.array([-0.03, 0.03, -0.023, 0.023, -0.01, 0.0], dtype=np.float32)
        grid_size = (4, 4, 2)
        blocks = create_regular_grid(domain_bounds, grid_size)

        element_to_block, stats = assign_elements_to_block_list(
            nodes, connectivity, blocks, verbose=False
        )

        # Analyze distribution
        occupancies = [stats.elements_per_block[i] for i in range(stats.n_blocks)]
        occupancies_sorted = sorted(occupancies, reverse=True)

        print(f"\nBlock occupancy distribution:")
        print(f"  Top 5 blocks: {occupancies_sorted[:5]}")
        print(f"  Bottom 5 blocks: {occupancies_sorted[-5:]}")

        # Check if distribution is reasonable
        # Expect most blocks to be non-empty for ThreadedA
        assert stats.n_blocks_used >= 20, f"Too many empty blocks: {stats.n_blocks_empty}/32"

        # Check imbalance is within acceptable range
        # Phase 0 predicted 8.6x imbalance, allow up to 15x
        assert stats.imbalance_ratio < 15.0, f"Imbalance too high: {stats.imbalance_ratio:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
