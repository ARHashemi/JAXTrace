<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Extracting and Verifying Intrinsic Octree from Kuhn Tetrahedral Mesh

## Understanding Kuhn Decomposition and Octree Relationship

Your mesh uses **Kuhn decomposition**, which subdivides each cube into **exactly 6 tetrahedra** sharing a common diagonal edge. This creates a natural octree hierarchy since:[^1]

- Each cube in the octree → 6 tetrahedral elements (fixed ratio)[^1]
- 1:2 refinement splits each cube into 8 child cubes
- Each child cube → 6 more tetrahedra

**Expected elements per octree cell: Always 6** (if properly aligned)[^2][^1]

Your observation of **6, 12, and 24 elements** indicates **octree-mesh misalignment** - cells spanning multiple mesh cubes.

***

## Best Methods to Extract Intrinsic Octree

### Method 1: Hanging Node Reconstruction (Recommended)

**Principle:** Adaptively refined meshes with 2:1 refinement create **hanging nodes** at refinement boundaries. These nodes encode the octree structure directly.[^3][^4][^2]

**Algorithm:**

```python
def extract_octree_from_hanging_nodes(mesh):
    octree = {}
    
    # Step 1: Identify all hanging nodes
    hanging_nodes = find_hanging_nodes(mesh)
    
    # Step 2: For each hanging node, identify parent edge/face
    for hnode in hanging_nodes:
        parent_face = find_parent_face(hnode)
        
        # Hanging node indicates refinement level transition
        coarse_cell = get_cell_containing_face(parent_face)
        fine_cells = get_adjacent_small_cells(parent_face)
        
        # Record parent-child relationship
        octree[coarse_cell] = fine_cells
    
    # Step 3: Build complete hierarchy bottom-up
    leaf_cells = get_leaf_cells(mesh)
    build_hierarchy_from_leaves(octree, leaf_cells)
    
    return octree

def find_hanging_nodes(mesh):
    """Node is hanging if it lies on edge/face of larger element"""
    hanging = []
    for node in mesh.nodes:
        adjacent_elems = get_adjacent_elements(node)
        if has_size_mismatch(adjacent_elems):
            hanging.append(node)
    return hanging
```

**Key insight:** Hanging nodes appear **only at octree cell boundaries** where refinement level changes by exactly 1 (2:1 constraint).[^5][^6]

**Advantages:**

- Directly encodes refinement hierarchy[^2][^3]
- No geometric tolerance issues
- Preserves 2:1 balance by construction

**Reference:** Extensively used in octree-based AMR[^7][^8][^4]

### Method 2: Kuhn Decomposition Inverse Mapping

**Principle:** Reverse the Kuhn decomposition to identify parent cubes.[^1]

**Algorithm:**

```python
def reconstruct_from_kuhn_pattern(tetrahedra):
    cubes = {}
    
    for tet in tetrahedra:
        # Kuhn tetrahedra share common diagonal edge
        # Identify the diagonal edge (longest edge)
        diag_edge = get_diagonal_edge(tet)  # endpoints x0, x1
        
        # All 6 tets in same cube share this diagonal
        cube_key = frozenset(diag_edge)
        
        if cube_key not in cubes:
            cubes[cube_key] = {
                'diagonal': diag_edge,
                'tetrahedra': [],
                'level': compute_octree_level(diag_edge)
            }
        cubes[cube_key]['tetrahedra'].append(tet)
    
    # Verify each cube has exactly 6 tetrahedra
    for cube_id, data in cubes.items():
        assert len(data['tetrahedra']) == 6, \
            f"Cube {cube_id} has {len(data['tetrahedra'])} tets, expected 6"
    
    # Build octree from identified cubes
    return build_octree_from_cubes(cubes)

def get_diagonal_edge(tet):
    """Find internal diagonal of Kuhn tetrahedron"""
    # The diagonal is the longest edge
    edges = get_all_edges(tet)
    return max(edges, key=lambda e: edge_length(e))

def compute_octree_level(diagonal_endpoints):
    """Octree level based on cube size"""
    x0, x1 = diagonal_endpoints
    cube_size = np.linalg.norm(x1 - x0) / np.sqrt(3)  # diagonal = √3 * side
    level = int(np.round(np.log2(1.0 / cube_size)))
    return level
```

**Key properties of Kuhn decomposition:**[^1]

1. Each cube subdivided into **exactly 6 tetrahedra**
2. All 6 share common **body diagonal** as edge
3. Diagonal connects opposite cube vertices
4. For unit cube with corner at origin: diagonal connects (0,0,0) to (1,1,1)

**Advantages:**

- Exploits known decomposition structure
- Deterministic (no tolerance parameters)
- Directly yields 6 elements per cell


### Method 3: Morton/Z-Order Key Assignment (Complementary)

**Principle:** Assign Morton keys to each tetrahedral element based on its centroid.[^9][^10]

**Algorithm:**

```python
def assign_morton_keys(mesh, max_level=8):
    """Assign Morton keys to identify octree cells"""
    
    for tet in mesh.tetrahedra:
        centroid = compute_centroid(tet)
        
        # Convert to Morton key at appropriate level
        level = detect_element_level(tet)
        morton_key = point_to_morton(centroid, level)
        
        tet.morton_key = morton_key
        tet.octree_level = level
    
    # Group tetrahedra by Morton key
    octree_cells = group_by_morton_key(mesh.tetrahedra)
    
    # Verify grouping: each cell should have 6 tets at same level
    for morton_key, tets in octree_cells.items():
        verify_kuhn_group(tets)
    
    return octree_cells

def point_to_morton(point, level):
    """Convert point to Morton key"""
    scale = 2**level
    ix = int(point.x * scale)
    iy = int(point.y * scale)
    iz = int(point.z * scale)
    
    # Interleave bits (Z-order curve)
    return interleave_bits(ix, iy, iz)

def detect_element_level(tet):
    """Infer octree level from element size"""
    bbox = compute_bounding_box(tet)
    cube_size = max(bbox.width, bbox.height, bbox.depth)
    return int(np.round(-np.log2(cube_size)))
```

**Advantages:**

- Provides fast spatial indexing[^9]
- Enables parent-child relationship via bit operations
- Complementary to geometric methods

**Use case:** After extracting octree via Method 1 or 2, assign Morton keys for efficient point location queries.

***

## Comprehensive Verification Tests

### Test 1: Element Count Consistency

**Invariant:** Each octree leaf cell contains **exactly 6 tetrahedra**[^1]

```python
def test_element_count_per_cell(octree):
    """Verify 6 tetrahedra per cube"""
    for cell_id, cell_data in octree.leaf_cells.items():
        n_elements = len(cell_data['tetrahedra'])
        assert n_elements == 6, \
            f"Cell {cell_id} has {n_elements} elements, expected 6"
        print(f"✓ Cell {cell_id}: {n_elements} elements")
```

**If this fails:** Octree boundaries don't align with mesh structure.

### Test 2: 2:1 Balance Verification

**Invariant:** Neighboring octree cells differ by at most 1 refinement level[^6][^5]

```python
def test_21_balance(octree):
    """Verify 2:1 balance constraint"""
    violations = []
    
    for cell in octree.leaf_cells.values():
        neighbors = get_face_neighbors(cell, octree)
        
        for neighbor in neighbors:
            level_diff = abs(cell.level - neighbor.level)
            
            if level_diff > 1:
                violations.append({
                    'cell': cell.id,
                    'neighbor': neighbor.id,
                    'level_diff': level_diff
                })
    
    assert len(violations) == 0, \
        f"Found {len(violations)} balance violations"
    print(f"✓ 2:1 balance: {len(octree.leaf_cells)} cells checked")
```

**Expected:** Zero violations for properly balanced mesh.[^7][^6]

### Test 3: Hierarchical Completeness

**Invariant:** Every internal node has **exactly 8 children**; every leaf is at refinement level ≤ max_level

```python
def test_hierarchy_completeness(octree):
    """Verify complete octree structure"""
    def check_node(node):
        if node.is_leaf:
            # Leaves at deepest level must contain 6 tets
            if node.level == octree.max_level:
                assert len(node.tetrahedra) == 6
        else:
            # Internal nodes must have exactly 8 children
            assert len(node.children) == 8, \
                f"Node {node.id} has {len(node.children)} children"
            for child in node.children:
                # Child level = parent level + 1
                assert child.level == node.level + 1
                check_node(child)
    
    check_node(octree.root)
    print(f"✓ Hierarchy: {octree.total_nodes} nodes verified")
```


### Test 4: Parent-Child Spatial Containment

**Invariant:** Each child cell must be **spatially contained** within its parent[^9]

```python
def test_spatial_containment(octree):
    """Verify geometric parent-child relationships"""
    def check_containment(node):
        if not node.is_leaf:
            parent_bbox = node.bounding_box
            
            for child in node.children:
                child_bbox = child.bounding_box
                
                # Child must be contained in parent
                assert contains(parent_bbox, child_bbox), \
                    f"Child {child.id} not contained in parent {node.id}"
                
                # Child volume should be 1/8 of parent
                volume_ratio = child_bbox.volume / parent_bbox.volume
                assert abs(volume_ratio - 0.125) < 1e-10
                
                check_containment(child)
    
    check_containment(octree.root)
    print("✓ Spatial containment verified")
```


### Test 5: Hanging Node Consistency

**Invariant:** Hanging nodes appear **only** at 2:1 refinement boundaries[^3][^2]

```python
def test_hanging_nodes(mesh, octree):
    """Verify hanging nodes mark refinement boundaries"""
    hanging_nodes = find_hanging_nodes(mesh)
    
    for hnode in hanging_nodes:
        # Find adjacent cells
        coarse_cell, fine_cells = find_boundary_cells(hnode, octree)
        
        # Verify level difference is exactly 1
        for fine_cell in fine_cells:
            level_diff = fine_cell.level - coarse_cell.level
            assert level_diff == 1, \
                f"Hanging node {hnode.id}: level diff = {level_diff}"
        
        # Verify geometric position
        assert is_on_cell_boundary(hnode, coarse_cell)
        assert is_midpoint_of_edge_or_face(hnode, coarse_cell)
    
    print(f"✓ Hanging nodes: {len(hanging_nodes)} verified")
```


### Test 6: Morton Key Ordering

**Invariant:** Morton keys increase along space-filling curve; siblings have consecutive keys[^9]

```python
def test_morton_ordering(octree):
    """Verify Z-order curve properties"""
    leaves = octree.get_leaves_in_morton_order()
    
    # Check monotonic ordering
    for i in range(len(leaves)-1):
        assert leaves[i].morton_key < leaves[i+1].morton_key
    
    # Check sibling relationships
    for node in octree.internal_nodes:
        children_keys = [child.morton_key for child in node.children]
        
        # Siblings should have keys with same prefix
        parent_bits = extract_parent_bits(children_keys[^0])
        for key in children_keys[1:]:
            assert extract_parent_bits(key) == parent_bits
    
    print(f"✓ Morton ordering: {len(leaves)} leaves verified")
```


### Test 7: Mesh Coverage

**Invariant:** Union of all octree leaf cells **exactly covers** the mesh domain

```python
def test_mesh_coverage(mesh, octree):
    """Verify octree covers entire mesh without gaps/overlaps"""
    
    # Every tetrahedral element must belong to exactly one leaf cell
    tet_assignments = {}
    for leaf in octree.leaf_cells.values():
        for tet in leaf.tetrahedra:
            assert tet.id not in tet_assignments, \
                f"Tet {tet.id} assigned to multiple cells"
            tet_assignments[tet.id] = leaf.id
    
    # All mesh elements must be assigned
    assert len(tet_assignments) == len(mesh.tetrahedra), \
        f"Coverage: {len(tet_assignments)}/{len(mesh.tetrahedra)} tets"
    
    # Volume check
    mesh_volume = sum(tet.volume for tet in mesh.tetrahedra)
    octree_volume = sum(cell.volume for cell in octree.leaf_cells.values())
    assert abs(mesh_volume - octree_volume) / mesh_volume < 1e-6
    
    print(f"✓ Coverage: {len(mesh.tetrahedra)} elements accounted for")
```


### Test 8: Neighbor Connectivity

**Invariant:** Face-adjacent cells in octree correspond to face-adjacent elements in mesh[^9]

```python
def test_neighbor_connectivity(octree, mesh):
    """Verify octree topology matches mesh connectivity"""
    
    for cell in octree.leaf_cells.values():
        octree_neighbors = get_face_neighbors(cell, octree)
        
        # Get mesh-level neighbors through shared faces
        mesh_neighbors_set = set()
        for tet in cell.tetrahedra:
            for face in tet.faces:
                neighbor_tets = mesh.get_elements_sharing_face(face)
                for neighbor_tet in neighbor_tets:
                    if neighbor_tet not in cell.tetrahedra:
                        neighbor_cell = get_cell_containing(neighbor_tet, octree)
                        mesh_neighbors_set.add(neighbor_cell.id)
        
        octree_neighbor_ids = {n.id for n in octree_neighbors}
        
        # Must match (accounting for multiple faces at boundaries)
        assert mesh_neighbors_set == octree_neighbor_ids, \
            f"Neighbor mismatch for cell {cell.id}"
    
    print("✓ Neighbor connectivity verified")
```


***

## Recommended Implementation Workflow

```python
class OctreeExtractor:
    def __init__(self, mesh, max_level=8):
        self.mesh = mesh
        self.max_level = max_level
        self.octree = None
    
    def extract(self):
        """Complete extraction pipeline"""
        
        # Step 1: Extract using hanging nodes (most reliable)
        print("Step 1: Extracting from hanging nodes...")
        self.octree = self.extract_from_hanging_nodes()
        
        # Step 2: Cross-validate with Kuhn decomposition
        print("Step 2: Validating with Kuhn decomposition...")
        self.validate_kuhn_structure()
        
        # Step 3: Assign Morton keys for fast queries
        print("Step 3: Assigning Morton keys...")
        self.assign_morton_keys()
        
        # Step 4: Build parent-child relationships
        print("Step 4: Building hierarchy...")
        self.build_complete_hierarchy()
        
        return self.octree
    
    def verify(self):
        """Run all verification tests"""
        print("\n=== Verification Tests ===")
        
        tests = [
            ("Element count", test_element_count_per_cell),
            ("2:1 balance", test_21_balance),
            ("Hierarchy", test_hierarchy_completeness),
            ("Containment", test_spatial_containment),
            ("Hanging nodes", test_hanging_nodes),
            ("Morton order", test_morton_ordering),
            ("Coverage", test_mesh_coverage),
            ("Connectivity", test_neighbor_connectivity),
        ]
        
        passed = 0
        for name, test_func in tests:
            try:
                test_func(self.octree, self.mesh)
                passed += 1
            except AssertionError as e:
                print(f"✗ {name}: {e}")
        
        print(f"\nPassed {passed}/{len(tests)} tests")
        return passed == len(tests)
```


***

## Additional Recommendations

### Performance Optimization

- **Use linear octree representation**: Store only leaves in Morton order[^10][^9]
- **Compute parent-child via bit operations**: Avoid pointer-based tree traversal[^9]
- **Cache neighbor relationships**: Expensive to recompute


### Debugging Tools

```python
def visualize_octree_slice(octree, z_plane=0.5):
    """2D slice visualization for debugging"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    for cell in octree.leaf_cells.values():
        if abs(cell.center.z - z_plane) < cell.size/2:
            rect = plt.Rectangle(
                (cell.min.x, cell.min.y),
                cell.size, cell.size,
                fill=False,
                edgecolor='blue',
                linewidth=2-cell.level/octree.max_level
            )
            ax.add_patch(rect)
            
            # Mark element count
            ax.text(cell.center.x, cell.center.y, 
                   str(len(cell.tetrahedra)),
                   ha='center', va='center')
    
    plt.axis('equal')
    plt.title(f'Octree slice at z={z_plane}')
    plt.show()
```

The hanging node method combined with Kuhn decomposition validation provides the most robust extraction. Your consistent 6-element-per-cell result will confirm correct octree-mesh alignment.[^1]
<span style="display:none">[^11][^12][^13][^14][^15][^16][^17][^18][^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57]</span>

<div align="center">⁂</div>

[^1]: https://users.cs.utah.edu/~tch/notes/PSSAT/IR/SAT/Kuhn1.pdf

[^2]: An_octree-based_adaptive_semi-Lagrangian_VOF_appro.pdf

[^3]: 1-s2.0-S0167844222003901-main.pdf

[^4]: https://repositories.lib.utexas.edu/bitstreams/a6afdf34-6629-47bf-a9d5-e68bc46a1f6b/download

[^5]: https://stackoverflow.com/questions/25309784/whats-the-algorithm-for-21-balancing-a-linear-octree

[^6]: https://p4est.github.io/papers/IsaacBursteddeGhattas12.pdf

[^7]: https://arxiv.org/abs/1406.0089

[^8]: https://arxiv.org/pdf/1701.04329.pdf

[^9]: 169627.169640.pdf

[^10]: https://arxiv.org/pdf/1712.00408.pdf

[^11]: 1-s2.0-S004578252400793X-main.pdf

[^12]: (105) 2023-FEAD.pdf

[^13]: https://arxiv.org/abs/2511.05145

[^14]: https://onlinelibrary.wiley.com/doi/10.1111/cgf.13958

[^15]: https://link.springer.com/10.1007/s10921-024-01102-8

[^16]: http://link.springer.com/10.1007/978-3-642-01273-0_92

[^17]: http://ieeexplore.ieee.org/document/6747587/

[^18]: https://link.springer.com/10.1007/978-3-030-31726-3_38

[^19]: http://oa.upm.es/91913/

[^20]: http://link.springer.com/10.1007/978-3-319-29451-3_22

[^21]: http://link.springer.com/10.1007/978-3-642-15907-7_34

[^22]: https://dl.acm.org/doi/10.1145/2024156.2024182

[^23]: https://www.mdpi.com/1099-4300/23/9/1156/pdf

[^24]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8472141/

[^25]: https://arxiv.org/abs/1408.3891

[^26]: http://arxiv.org/pdf/2401.05984.pdf

[^27]: http://arxiv.org/pdf/1809.07917.pdf

[^28]: https://www.aimsciences.org/article/doi/10.3934/acse.2025012

[^29]: https://www.sciencedirect.com/science/article/abs/pii/S0301932209000305

[^30]: http://www.cad.zju.edu.cn/home/jqfeng/papers/Multi-scale surface reconstruction based on an adaptive signed distance field(draft).pdf

[^31]: https://www.scipedia.com/wd/images/e/e5/Draft_Sanchez_Pinedo_223769139_9505_M12_ExaQUte_deliverable_2.2_First_Release_Octree-MeshGenerator\&ParallelMeshAdaptationKernel.pdf

[^32]: https://arxiv.org/html/2401.05984v2

[^33]: https://oat-3d.github.io

[^34]: https://www.sciencedirect.com/science/article/pii/S0965997814000817

[^35]: https://www.sciencedirect.com/science/article/abs/pii/S0021999125007612

[^36]: http://link.springer.com/10.1007/s00366-016-0444-3

[^37]: https://onlinelibrary.wiley.com/doi/10.1002/nme.1620310302

[^38]: http://ieeexplore.ieee.org/document/5952690/

[^39]: https://www.semanticscholar.org/paper/573087176038f49d5a20e84cc8182c4bba845786

[^40]: http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=904062

[^41]: https://link.springer.com/10.1007/s10596-021-10042-5

[^42]: https://www.semanticscholar.org/paper/63f69427d722e7b2637f41fb56f1362836dccb75

[^43]: http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=755872

[^44]: http://ieeexplore.ieee.org/document/7406379/

[^45]: https://arxiv.org/pdf/2307.06345.pdf

[^46]: http://arxiv.org/pdf/2305.15615.pdf

[^47]: https://arxiv.org/pdf/1105.1611.pdf

[^48]: http://arxiv.org/pdf/2411.02658.pdf

[^49]: http://arxiv.org/pdf/2103.10830.pdf

[^50]: https://arxiv.org/pdf/2504.02790.pdf

[^51]: https://web.me.iastate.edu/jmchsu/files/Varduhn_et_al-2015-IJNME.pdf

[^52]: https://kennyweiss.com/papers/Weiss10.eg_star.pdf

[^53]: https://digitalcommons.odu.edu/context/msve_etds/article/1010/viewcontent/Rashid_multi_material.pdf

[^54]: https://people.eecs.berkeley.edu/~jrs/meshpapers/FreitagGooch.pdf

[^55]: https://www.gidsimulation.com/downloads/octree-tetrahedra-mesher-module/

[^56]: https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.5207

[^57]: https://kennyweiss.com/papers/Weiss11.gis.pdf

