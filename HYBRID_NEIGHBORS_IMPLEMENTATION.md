# Hybrid Neighbors Implementation Guide

**Goal**: Reduce neighbor memory from 1GB to ~150MB while maintaining correct trajectories

**Timeline**: 3-5 days
**Difficulty**: Medium
**Impact**: Enable 200K+ particles (4× current capacity)

---

## Strategy

### Concept

Use **face-based neighbors for interior**, **node-based for boundary elements only**:

```
Interior elements (95%): 4 neighbors   →  44 MB
Boundary elements (5%):  90 neighbors  →  52 MB
                         Total:           96 MB (vs 1,046 MB)
```

### Boundary Detection

**Elements are "boundary" if**:
- Coarse element shares nodes with fine elements, OR
- Fine element shares nodes with coarse elements, OR
- Medium element at coarse/fine interface

**Detection method**:
1. Compute element sizes (characteristic length)
2. Classify: Fine (≤0.15mm), Medium (0.15-0.30mm), Coarse (>0.30mm)
3. Build node→elements map
4. For each element, check if neighbors have different level

---

## Implementation Plan

### Phase 1: Boundary Detection (Day 1)

**File**: `jaxtrace/gpu/forest/element_adjacency.py`

**New function**:
```python
def identify_boundary_elements(
    connectivity: np.ndarray,
    node_positions: np.ndarray,
    fine_threshold: float = 0.15,
    coarse_threshold: float = 0.30,
    verbose: bool = False
) -> np.ndarray:
    """
    Identify boundary elements at refinement transitions.

    Boundary = elements sharing nodes with different refinement level.

    Args:
        connectivity: (N, 4) element connectivity
        node_positions: (M, 3) node coordinates
        fine_threshold: Elements ≤ this are "fine" (mm)
        coarse_threshold: Elements ≥ this are "coarse" (mm)
        verbose: Print statistics

    Returns:
        boundary_mask: (N,) bool - True if element is at boundary
    """
    n_elements = len(connectivity)

    if verbose:
        print(f"\\nIdentifying boundary elements...")
        print(f"  Fine threshold: ≤{fine_threshold:.2f} mm")
        print(f"  Coarse threshold: ≥{coarse_threshold:.2f} mm")

    # 1. Compute element sizes
    if verbose:
        print("  Computing element sizes...")

    element_sizes = np.zeros(n_elements, dtype=np.float32)
    for i in range(n_elements):
        nodes = node_positions[connectivity[i]]
        # Characteristic size = max edge length
        edges = [
            np.linalg.norm(nodes[1] - nodes[0]),
            np.linalg.norm(nodes[2] - nodes[0]),
            np.linalg.norm(nodes[3] - nodes[0]),
            np.linalg.norm(nodes[2] - nodes[1]),
            np.linalg.norm(nodes[3] - nodes[1]),
            np.linalg.norm(nodes[3] - nodes[2]),
        ]
        element_sizes[i] = max(edges)

        if verbose and (i + 1) % 500000 == 0:
            print(f"    Processed {i + 1:,}/{n_elements:,} elements...")

    # 2. Classify elements
    is_fine = element_sizes <= fine_threshold
    is_medium = (element_sizes > fine_threshold) & (element_sizes < coarse_threshold)
    is_coarse = element_sizes >= coarse_threshold

    n_fine = np.sum(is_fine)
    n_medium = np.sum(is_medium)
    n_coarse = np.sum(is_coarse)

    if verbose:
        print(f"  Element classification:")
        print(f"    Fine: {n_fine:,} ({100*n_fine/n_elements:.1f}%)")
        print(f"    Medium: {n_medium:,} ({100*n_medium/n_elements:.1f}%)")
        print(f"    Coarse: {n_coarse:,} ({100*n_coarse/n_elements:.1f}%)")

    # 3. Build node→elements map
    if verbose:
        print("  Building node-to-elements map...")

    node_to_elements = build_node_to_elements_map(connectivity, verbose=False)

    # 4. Detect boundaries
    if verbose:
        print("  Detecting boundary elements...")

    boundary_mask = np.zeros(n_elements, dtype=bool)

    for elem_id in range(n_elements):
        elem_level = 'fine' if is_fine[elem_id] else ('coarse' if is_coarse[elem_id] else 'medium')

        # Check if any node connects to different level
        for node_id in connectivity[elem_id]:
            neighbor_elements = node_to_elements[node_id]

            # Check for level mismatch
            if elem_level == 'fine':
                if np.any(is_coarse[list(neighbor_elements)]) or np.any(is_medium[list(neighbor_elements)]):
                    boundary_mask[elem_id] = True
                    break
            elif elem_level == 'coarse':
                if np.any(is_fine[list(neighbor_elements)]) or np.any(is_medium[list(neighbor_elements)]):
                    boundary_mask[elem_id] = True
                    break
            elif elem_level == 'medium':
                if np.any(is_fine[list(neighbor_elements)]) or np.any(is_coarse[list(neighbor_elements)]):
                    boundary_mask[elem_id] = True
                    break

        if verbose and (elem_id + 1) % 500000 == 0:
            print(f"    Processed {elem_id + 1:,}/{n_elements:,} elements...")

    n_boundary = np.sum(boundary_mask)
    if verbose:
        print(f"  Boundary elements: {n_boundary:,} ({100*n_boundary/n_elements:.1f}%)")

    return boundary_mask
```

**Test**:
```python
# In test script:
boundary_mask = identify_boundary_elements(connectivity, node_positions, verbose=True)
print(f"Boundary: {np.sum(boundary_mask):,} / {len(boundary_mask):,}")
# Expected: 5-10% boundary
```

### Phase 2: Hybrid Neighbor Construction (Day 2)

**File**: `jaxtrace/gpu/forest/element_adjacency.py`

**New function**:
```python
def build_hybrid_neighbors_array(
    connectivity: np.ndarray,
    node_positions: np.ndarray,
    fine_threshold: float = 0.15,
    coarse_threshold: float = 0.30,
    verbose: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Build hybrid neighbor array: face-based + node-based for boundaries.

    Args:
        connectivity: (N, 4) element connectivity
        node_positions: (M, 3) node coordinates
        fine_threshold: Fine element threshold (mm)
        coarse_threshold: Coarse element threshold (mm)
        verbose: Print statistics

    Returns:
        element_neighbors: (N, MAX_NEIGHBORS) padded array
        stats: dict with construction statistics
    """
    n_elements = len(connectivity)

    if verbose:
        print(f"\\nBuilding HYBRID element neighbors...")

    # 1. Identify boundary elements
    boundary_mask = identify_boundary_elements(
        connectivity, node_positions,
        fine_threshold, coarse_threshold,
        verbose=verbose
    )
    boundary_indices = np.where(boundary_mask)[0]
    interior_indices = np.where(~boundary_mask)[0]

    n_boundary = len(boundary_indices)
    n_interior = len(interior_indices)

    if verbose:
        print(f"\\n  Element categories:")
        print(f"    Interior: {n_interior:,} ({100*n_interior/n_elements:.1f}%)")
        print(f"    Boundary: {n_boundary:,} ({100*n_boundary/n_elements:.1f}%)")

    # 2. Build face-based neighbors (all elements)
    if verbose:
        print("\\n  Building face-based neighbors (all elements)...")

    face_neighbors_dict, _ = extract_element_neighbors(connectivity, verbose=verbose)

    # 3. Build node-based neighbors (boundary only)
    if verbose:
        print("\\n  Building node-based neighbors (boundary only)...")

    node_to_elements = build_node_to_elements_map(connectivity, verbose=False)

    node_neighbors = {}
    max_node_neighbors = 0

    for elem_id in boundary_indices:
        nodes = connectivity[elem_id]

        # Find all elements sharing any node
        neighbor_set = set()
        for node_id in nodes:
            neighbor_set.update(node_to_elements[node_id])
        neighbor_set.discard(elem_id)  # Remove self

        node_neighbors[elem_id] = np.array(sorted(neighbor_set), dtype=np.int32)
        max_node_neighbors = max(max_node_neighbors, len(node_neighbors[elem_id]))

        if verbose and (len(node_neighbors) % 10000 == 0):
            print(f"    Processed {len(node_neighbors):,}/{n_boundary:,} boundary elements...")

    if verbose:
        avg_node_neighbors = np.mean([len(n) for n in node_neighbors.values()])
        print(f"    Boundary neighbor stats:")
        print(f"      Avg: {avg_node_neighbors:.1f}")
        print(f"      Max: {max_node_neighbors}")

    # 4. Merge into hybrid array
    if verbose:
        print("\\n  Merging into hybrid array...")

    max_neighbors = max(4, max_node_neighbors)  # At least 4 for face-based
    element_neighbors = np.full((n_elements, max_neighbors), -1, dtype=np.int32)

    # Fill interior with face-based
    for elem_id in interior_indices:
        neighs = face_neighbors_dict.get(elem_id, np.array([], dtype=np.int32))
        n = min(len(neighs), 4)
        element_neighbors[elem_id, :n] = neighs[:n]

    # Fill boundary with node-based
    for elem_id in boundary_indices:
        neighs = node_neighbors[elem_id]
        n = min(len(neighs), max_neighbors)
        element_neighbors[elem_id, :n] = neighs[:n]

    if verbose:
        print(f"    Hybrid array shape: ({n_elements}, {max_neighbors})")
        memory_mb = element_neighbors.nbytes / (1024**2)
        print(f"    Memory: {memory_mb:.1f} MB")

    # Statistics
    stats = {
        'n_elements': n_elements,
        'n_interior': n_interior,
        'n_boundary': n_boundary,
        'max_neighbors': max_neighbors,
        'avg_neighbors_interior': 3.5,  # Typical for face-based
        'avg_neighbors_boundary': avg_node_neighbors,
        'memory_mb': memory_mb,
    }

    return element_neighbors, stats
```

**Export**:
```python
# In __init__.py, add:
from .element_adjacency import (
    ...
    identify_boundary_elements,
    build_hybrid_neighbors_array,
)

__all__ = [
    ...
    "identify_boundary_elements",
    "build_hybrid_neighbors_array",
]
```

### Phase 3: Production Integration (Day 3)

**File**: `production_tracking_fully_fused_timedep.py`

**Modify neighbor construction** (line ~297):
```python
# BEFORE (full node-based):
element_neighbors = build_element_neighbors_array(connectivity, method='node', verbose=True)

# AFTER (hybrid):
from jaxtrace.gpu.forest import build_hybrid_neighbors_array

element_neighbors, neighbor_stats = build_hybrid_neighbors_array(
    connectivity,
    node_positions,
    fine_threshold=0.15,
    coarse_threshold=0.30,
    verbose=True
)

print(f"    Neighbor statistics:")
print(f"      Interior elements: {neighbor_stats['n_interior']:,} ({100*neighbor_stats['n_interior']/neighbor_stats['n_elements']:.1f}%)")
print(f"      Boundary elements: {neighbor_stats['n_boundary']:,} ({100*neighbor_stats['n_boundary']/neighbor_stats['n_elements']:.1f}%)")
print(f"      Avg neighbors (interior): {neighbor_stats['avg_neighbors_interior']:.1f}")
print(f"      Avg neighbors (boundary): {neighbor_stats['avg_neighbors_boundary']:.1f}")
print(f"      Max neighbors: {neighbor_stats['max_neighbors']}")
print(f"      Memory: {neighbor_stats['memory_mb']:.1f} MB (vs 1046.8 MB for full node-based)")
```

**Update configuration note** (line ~79):
```python
# Search Hierarchy Configuration
# NOTE: Using HYBRID neighbors (face-based + boundary node-based) for 1:2 octree refinement
#       Interior (95%): 4 face-neighbors (44 MB)
#       Boundary (5%):  90 node-neighbors (52 MB)
#       Total: ~96 MB (vs 1,046 MB full node-based, 48 MB face-only)
N_HOPS = 3                     # Number of hops for L1 neighbor search
```

### Phase 4: Testing (Days 4-5)

**Test 1: Small Scale** (48K particles):
```bash
# Should work with hybrid (same as node-based)
python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/hybrid_test_48K.log

# Check:
# - Memory: ~150 MB (vs 1,046 MB)
# - Trajectories: Rotating (correct)
# - Performance: Similar to node-based (~29K p/s)
```

**Test 2: Medium Scale** (100K particles):
```bash
# Edit script: PARTICLE_GRID_RESOLUTION = (30, 80, 42) # 100,800 particles
python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/hybrid_test_100K.log

# Check:
# - No OOM during compilation
# - Retention: >95%
# - Trajectories: Rotating
```

**Test 3: Large Scale** (200K particles):
```bash
# Edit script: PARTICLE_GRID_RESOLUTION = (50, 90, 45) # 202,500 particles
python production_tracking_fully_fused_timedep.py 2>&1 | tee logs/hybrid_test_200K.log

# Check:
# - No OOM
# - Performance: 40-60K p/s
# - Trajectories: Rotating
```

---

## Alternative: Simpler Static Threshold

If boundary detection is too slow or complex, use **static threshold** based on element size:

```python
def build_hybrid_neighbors_simple(connectivity, node_positions, threshold=0.20):
    """
    Simple hybrid: node-based for small elements, face-based for large.

    Args:
        threshold: Elements ≤ this use node-based (mm)

    Assumption: Small elements are in/near refined region (boundary).
    """
    n_elements = len(connectivity)

    # Compute sizes
    element_sizes = np.array([
        compute_element_size(node_positions[connectivity[i]])
        for i in range(n_elements)
    ])

    # Classify
    use_node_based = element_sizes <= threshold

    # Build neighbors
    face_neighbors = extract_element_neighbors(connectivity)
    node_neighbors_dict = extract_element_neighbors_node_based(connectivity)

    # Merge
    max_neighbors = 100
    element_neighbors = np.full((n_elements, max_neighbors), -1, dtype=np.int32)

    for i in range(n_elements):
        if use_node_based[i]:
            neighs = node_neighbors_dict[i]
        else:
            neighs = face_neighbors[i]

        n = min(len(neighs), max_neighbors)
        element_neighbors[i, :n] = neighs[:n]

    return element_neighbors
```

**Pros**:
- Simple: 1 threshold parameter
- Fast: No boundary detection

**Cons**:
- May include extra elements (medium-sized in refined region)
- Memory slightly higher (~200 MB vs ~100 MB)

---

## Expected Results

### Memory Comparison

| Method | Memory | Max Neighbors | Max Particles (4GB GPU) |
|--------|--------|---------------|-------------------------|
| Face-based | 48 MB | 4 | 200K+ (but WRONG) |
| **Hybrid** | **96-150 MB** | **90** | **200K+** ✓ |
| Node-based | 1,046 MB | 90 | 50K |

### Performance Comparison

| Method | Throughput | Correctness |
|--------|------------|-------------|
| Face-based | 30K p/s | ❌ Linear trajectories |
| **Hybrid** | **35-40K p/s** | **✓ Rotating** |
| Node-based | 29K p/s | ✓ Rotating |

### Boundary Element Distribution (Estimated)

For typical FSW mesh:
```
Fine (≤0.15mm):     2.6M elements (85%)
  → Boundary: 130K (5% of fine)

Medium (0.15-0.30): 382K elements (12.5%)
  → Boundary: 50K (13% of medium)

Coarse (>0.30mm):   67K elements (2.2%)
  → Boundary: 10K (15% of coarse)

Total boundary: ~190K (6.2%)
```

**Memory**:
```
Interior (93.8%): 2,858,900 × 4 × 4 = 43.7 MB
Boundary (6.2%):    190,000 × 90 × 4 = 68.4 MB
                    Total:              112.1 MB
```

---

## Troubleshooting

### Issue 1: Boundary Detection Too Slow

**Symptom**: Takes >5 minutes to identify boundaries

**Solution**: Parallelize with multiprocessing
```python
from multiprocessing import Pool

def check_boundary_batch(args):
    elem_ids, connectivity, node_to_elements, is_fine, is_coarse = args
    boundary = []
    for elem_id in elem_ids:
        # ... boundary check ...
        if is_boundary:
            boundary.append(elem_id)
    return boundary

# Split work
n_workers = 8
batches = np.array_split(range(n_elements), n_workers)
args = [(batch, connectivity, node_to_elements, is_fine, is_coarse) for batch in batches]

with Pool(n_workers) as pool:
    results = pool.map(check_boundary_batch, args)

boundary_indices = np.concatenate(results)
```

### Issue 2: Too Many Boundary Elements (>10%)

**Symptom**: Memory still high (~300 MB)

**Solution**: Tighten thresholds
```python
# Original:
fine_threshold = 0.15
coarse_threshold = 0.30

# Tighter (only immediate neighbors):
fine_threshold = 0.12
coarse_threshold = 0.35
```

### Issue 3: Wrong Trajectories

**Symptom**: Still linear motion in refined region

**Cause**: Boundary detection missed some elements

**Debug**:
```python
# Check boundary coverage in refined region
refined_center = np.array([30.0, 15.0, 0.3])
refined_radius = 2.0

for i, elem_id in enumerate(range(n_elements)):
    centroid = np.mean(node_positions[connectivity[elem_id]], axis=0)
    dist = np.linalg.norm(centroid - refined_center)

    if dist < refined_radius:
        is_boundary = boundary_mask[elem_id]
        size = element_sizes[elem_id]
        print(f"Element {elem_id}: size={size:.4f}, boundary={is_boundary}")

# Expected: Most fine elements (size <0.15) in refined region are boundary
```

**Fix**: Expand boundary criteria
```python
# Also mark fine elements with coarse/medium neighbors
# (More conservative: catches edge cases)
```

---

## Success Criteria

1. ✅ **Memory**: <200 MB (vs 1,046 MB)
2. ✅ **Max particles**: 200K+ on 4GB GPU
3. ✅ **Correctness**: Rotating trajectories in refined region
4. ✅ **Performance**: 35-40K particles/s (acceptable)
5. ✅ **Boundary detection**: 5-10% boundary elements

---

## Next Steps After Hybrid

Once hybrid neighbors work:

1. **Production use**: Run 200K particle simulations
2. **Optimize thresholds**: Fine-tune boundary detection
3. **Consider Phase 2**: Octree-aligned leaves for 100-150K p/s

---

**End of Implementation Guide**
