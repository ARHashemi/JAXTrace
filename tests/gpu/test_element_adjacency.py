"""
Tests for element face-adjacency extraction.

Part of Phase 2: Element Neighbors & Padded Block Arrays
"""

import pytest
import numpy as np

from jaxtrace.gpu.forest.element_adjacency import (
    get_tet_faces,
    build_face_to_element_map,
    extract_element_neighbors,
    validate_neighbor_symmetry,
    AdjacencyStats
)


class TestTetFaces:
    """Test tetrahedral face extraction."""
    
    def test_single_tet_faces(self):
        """Test face extraction for single tet."""
        nodes = np.array([0, 1, 2, 3], dtype=np.int32)
        faces = get_tet_faces(0, nodes)
        
        assert faces.shape == (4, 3)
        
        # Each face should be sorted
        for face in faces:
            assert np.all(face[:-1] <= face[1:])
        
        # Check expected faces (sorted)
        expected_faces = np.array([
            [1, 2, 3],  # Opposite node 0
            [0, 2, 3],  # Opposite node 1
            [0, 1, 3],  # Opposite node 2
            [0, 1, 2],  # Opposite node 3
        ], dtype=np.int32)
        
        assert np.array_equal(np.sort(faces, axis=0), np.sort(expected_faces, axis=0))
    
    def test_face_sorting(self):
        """Test that faces are consistently sorted."""
        nodes1 = np.array([10, 20, 30, 40], dtype=np.int32)
        nodes2 = np.array([40, 30, 20, 10], dtype=np.int32)
        
        faces1 = get_tet_faces(0, nodes1)
        faces2 = get_tet_faces(1, nodes2)
        
        # Faces should be sorted for consistent hashing
        for face in faces1:
            assert face[0] <= face[1] <= face[2]
        for face in faces2:
            assert face[0] <= face[1] <= face[2]


class TestFaceToElementMap:
    """Test face-to-element mapping."""
    
    def test_two_adjacent_tets(self):
        """Test two tets sharing a face."""
        connectivity = np.array([
            [0, 1, 2, 3],  # Tet 0
            [1, 2, 3, 4],  # Tet 1 (shares face [1,2,3] with Tet 0)
        ], dtype=np.int32)
        
        face_map = build_face_to_element_map(connectivity, verbose=False)
        
        # Face [1, 2, 3] should map to both elements
        shared_face = tuple(sorted([1, 2, 3]))
        assert shared_face in face_map
        assert len(face_map[shared_face]) == 2
        assert face_map[shared_face] == {0, 1}
    
    def test_isolated_tet(self):
        """Test single isolated tet."""
        connectivity = np.array([
            [0, 1, 2, 3],
        ], dtype=np.int32)
        
        face_map = build_face_to_element_map(connectivity, verbose=False)
        
        # Should have 4 faces, all boundary (single element)
        assert len(face_map) == 4
        for face_tuple, elements in face_map.items():
            assert len(elements) == 1
            assert 0 in elements


class TestElementNeighbors:
    """Test element neighbor extraction."""
    
    def test_two_adjacent_tets_neighbors(self):
        """Test neighbor extraction for two adjacent tets."""
        connectivity = np.array([
            [0, 1, 2, 3],  # Tet 0
            [1, 2, 3, 4],  # Tet 1 (shares face with Tet 0)
        ], dtype=np.int32)
        
        neighbors, stats = extract_element_neighbors(connectivity, verbose=False)
        
        # Both elements should be neighbors
        assert 1 in neighbors[0]
        assert 0 in neighbors[1]
        
        # Stats
        assert stats.n_elements == 2
        assert stats.min_neighbors_per_element == 1
        assert stats.max_neighbors_per_element == 1
        assert stats.avg_neighbors_per_element == 1.0
    
    def test_isolated_tet_no_neighbors(self):
        """Test isolated tet has no neighbors."""
        connectivity = np.array([
            [0, 1, 2, 3],
        ], dtype=np.int32)
        
        neighbors, stats = extract_element_neighbors(connectivity, verbose=False)
        
        # No neighbors
        assert len(neighbors[0]) == 0
        assert stats.min_neighbors_per_element == 0
        assert stats.max_neighbors_per_element == 0
        assert stats.n_boundary_faces == 4
        assert stats.n_internal_faces == 0
    
    def test_three_tets_sharing_edge(self):
        """Test three tets forming a simple mesh."""
        connectivity = np.array([
            [0, 1, 2, 3],  # Tet 0
            [1, 2, 3, 4],  # Tet 1 (shares face with 0)
            [1, 2, 4, 5],  # Tet 2 (shares face with 1)
        ], dtype=np.int32)
        
        neighbors, stats = extract_element_neighbors(connectivity, verbose=False)
        
        # Tet 0 neighbors: [1]
        assert 1 in neighbors[0]
        assert len(neighbors[0]) == 1
        
        # Tet 1 neighbors: [0, 2]
        assert 0 in neighbors[1]
        assert 2 in neighbors[1]
        assert len(neighbors[1]) == 2
        
        # Tet 2 neighbors: [1]
        assert 1 in neighbors[2]
        assert len(neighbors[2]) == 1
        
        # Stats
        assert stats.n_elements == 3
        assert stats.max_neighbors_per_element == 2


class TestNeighborSymmetry:
    """Test neighbor symmetry validation."""
    
    def test_symmetry_validation_passes(self):
        """Test that valid neighbor relationships pass validation."""
        connectivity = np.array([
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 4, 5],
        ], dtype=np.int32)
        
        neighbors, _ = extract_element_neighbors(connectivity, verbose=False)
        
        valid = validate_neighbor_symmetry(neighbors, n_samples=3)
        assert valid
    
    def test_symmetry_validation_detects_error(self):
        """Test that asymmetric relationships are detected."""
        # Create deliberately asymmetric neighbors
        neighbors = {
            0: np.array([1], dtype=np.int32),
            1: np.array([], dtype=np.int32),  # Missing 0!
        }
        
        valid = validate_neighbor_symmetry(neighbors, n_samples=2)
        assert not valid


class TestAdjacencyStats:
    """Test adjacency statistics."""
    
    def test_stats_computation(self):
        """Test that statistics are correctly computed."""
        connectivity = np.array([
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
        ], dtype=np.int32)
        
        neighbors, stats = extract_element_neighbors(connectivity, verbose=False)
        
        assert stats.n_elements == 3
        assert stats.avg_neighbors_per_element == pytest.approx((1 + 2 + 1) / 3)
        assert stats.n_boundary_faces + stats.n_internal_faces == stats.n_faces


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
