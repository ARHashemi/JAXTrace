# FINAL EXECUTABLE PLAN: GPU-Native Particle Tracking
**Incorporating Critical Review Feedback**

**Date**: 2025-11-06
**Branch**: `gpu_native_implementation` (clean restart)
**Reference Mesh**: ThreadedA (3.5M elements, 900K nodes)
**Target**: 1M particles with <500 MB GPU memory
**Hardware**: NVIDIA T1000 (4GB VRAM)

---

## CRITICAL REVIEW INTEGRATION

Your review identified key improvements to make the implementation **future-proof and scalable**:

### ✅ Approved Core Design
- Flat/padded arrays (JAX-compatible)
- Multi-level search (L0→L1→L2→L3)
- Block-local search (no global flattening)
- 26-neighbor topology
- Memory safety (<500 MB)

### ⚠️ Critical Addition: Hash/Bucket Search for Heavy Blocks

**Problem**: Some blocks may have 200K-1M elements even after 4×4×2 partitioning due to AMR clustering.

**Solution**: Add **intra-block hash/bucket subdivision** in Phase 4 (not Phase 9).

**Threshold**: If any block has >10,000 elements, subdivide with Morton/spatial hash.

**Impact**:
- Without hash: O(200K) search per particle in heavy blocks
- With hash: O(4K) search per particle (50 buckets × 4K elem/bucket)
- **50× speedup for heavy blocks**

### 📊 Monitoring Requirements

Add to each phase:
- Block occupancy histograms
- Padding waste metrics (target <50%)
- Heavy block detection (>10K elements)
- Automatic escalation to hash/bucket

---

## REVISED PHASE STRUCTURE

### Phase 0: Mesh Analysis (UNCHANGED)
- Verify ThreadedA characteristics
- Select 4×4×2 grid (32 blocks)
- **NEW**: Identify heavy blocks (>10K elements)

### Phase 1: Forest Structure (UNCHANGED)
- Regular grid generator
- Element-to-block assignment
- 26-neighbor topology

### Phase 2: Padded Block Arrays (UNCHANGED)
- Element neighbors
- Padded 2D arrays with -1 padding
- Validation

### Phase 3: Particle Seeding (UNCHANGED)
- Seed particles
- CPU search (ground truth)
- Particle state

### Phase 4: GPU Multi-Level Search (ENHANCED)
- **L0**: Cached element
- **L1**: Neighbor elements
- **L2a**: Light blocks (<10K elem) - Direct padded search
- **L2b**: Heavy blocks (>10K elem) - **Hash bucket search** (NEW)
- **L3**: Neighbor blocks
- **Monitoring**: Block occupancy stats (NEW)

**Key Change**: Hash/bucket search is **mandatory** for heavy blocks, not optional.

---

## PHASE 4 ENHANCEMENT: HASH BUCKET SEARCH

### Task 4.7: Intra-Block Hash/Bucket Subdivision (NEW)

**File**: `jaxtrace/gpu/search/hash_bucket.py` (NEW, ~400 lines)

**When to Use**: Automatically activated for blocks with >10,000 elements

**Algorithm**:
```
For each heavy block:
    1. Compute Morton codes for all element centroids in block
    2. Partition into N_buckets = ceil(n_elements / 200)
    3. Create padded bucket arrays: (n_buckets, max_elem_per_bucket)
    4. Build bucket hash table for O(1) lookup

During search:
    1. Compute particle Morton code
    2. Hash to bucket_id
    3. Search only elements in that bucket (~200 elements)
    4. Fallback to neighbor buckets if not found
```

**Data Structure**:
```python
@dataclass
class HashBucketArrays:
    """
    Intra-block hash subdivision for heavy blocks.

    Only built for blocks with >threshold elements.
    """
    block_id: int
    n_buckets: int  # Typically ceil(n_elements / 200)
    bucket_elements: np.ndarray  # (n_buckets, max_elem_per_bucket) int32, -1 padded
    bucket_elem_counts: np.ndarray  # (n_buckets,) int32
    morton_bits: int = 10  # 2^10 = 1024 buckets max

    # Bucket neighbor topology (6-face in Morton space)
    bucket_neighbors_6: np.ndarray  # (n_buckets, 6) int32
```

**Key Functions**:
```python
def compute_morton_codes(
    positions: np.ndarray,  # (n_elements, 3) centroids
    block_bounds: np.ndarray,  # [xmin, xmax, ymin, ymax, zmin, zmax]
    bits: int = 10
) -> np.ndarray:
    """
    Compute Morton codes (Z-order curve) for element centroids.

    Returns:
        morton_codes: (n_elements,) int32
    """
    # Normalize positions to [0, 2^bits)
    normalized = (positions - block_bounds[[0,2,4]]) / (block_bounds[[1,3,5]] - block_bounds[[0,2,4]])
    indices = (normalized * (2**bits)).astype(np.int32)

    # Interleave bits: x, y, z → Morton code
    morton_codes = morton_encode_3d(indices[:, 0], indices[:, 1], indices[:, 2])
    return morton_codes

def build_hash_bucket_arrays(
    element_ids: np.ndarray,  # Elements in this block
    element_centroids: np.ndarray,  # (n_elements, 3)
    block_bounds: np.ndarray,
    target_bucket_size: int = 200
) -> HashBucketArrays:
    """
    Build hash bucket subdivision for heavy block.

    Args:
        target_bucket_size: Target elements per bucket (200-500 typical)

    Returns:
        HashBucketArrays with padded bucket element lists
    """
    n_elements = len(element_ids)
    n_buckets = max(8, int(np.ceil(n_elements / target_bucket_size)))

    # Compute Morton codes
    morton_codes = compute_morton_codes(element_centroids, block_bounds)

    # Quantize to bucket IDs (n_buckets bins)
    max_morton = morton_codes.max() + 1
    bucket_ids = (morton_codes * n_buckets / max_morton).astype(np.int32)
    bucket_ids = np.clip(bucket_ids, 0, n_buckets - 1)

    # Build bucket → elements mapping
    bucket_to_elements = {}
    for i, bid in enumerate(bucket_ids):
        if bid not in bucket_to_elements:
            bucket_to_elements[bid] = []
        bucket_to_elements[bid].append(element_ids[i])

    # Compute max_elem_per_bucket (with padding)
    bucket_sizes = [len(elems) for elems in bucket_to_elements.values()]
    max_elem_per_bucket = int(np.percentile(bucket_sizes, 95) * 1.5)

    # Allocate padded arrays
    bucket_elements = np.full((n_buckets, max_elem_per_bucket), -1, dtype=np.int32)
    bucket_elem_counts = np.zeros(n_buckets, dtype=np.int32)

    # Fill buckets
    for bid, elems in bucket_to_elements.items():
        n = len(elems)
        bucket_elements[bid, :n] = elems
        bucket_elem_counts[bid] = n

    # Compute bucket neighbors (6-face in Morton space)
    bucket_neighbors_6 = compute_bucket_neighbors(n_buckets)

    return HashBucketArrays(
        block_id=block_id,
        n_buckets=n_buckets,
        bucket_elements=bucket_elements,
        bucket_elem_counts=bucket_elem_counts,
        bucket_neighbors_6=bucket_neighbors_6
    )

@jax.jit
def search_level2b_hash_bucket(
    position: jax.Array,  # (3,)
    block_id: int,
    hash_arrays: HashBucketArrays,
    node_positions: jax.Array,
    element_nodes: jax.Array
) -> int:
    """
    Level 2b: Search heavy block using hash bucket subdivision.

    Returns:
        element_id if found, else -1

    Algorithm:
        1. Compute Morton code for particle position
        2. Map to bucket_id
        3. Search elements in that bucket (~200 elements)
        4. If not found, search 6 neighbor buckets
    """
    # Compute bucket ID for particle
    morton_code = compute_morton_code_single(position, hash_arrays.block_bounds)
    bucket_id = (morton_code * hash_arrays.n_buckets / hash_arrays.max_morton).astype(jnp.int32)
    bucket_id = jnp.clip(bucket_id, 0, hash_arrays.n_buckets - 1)

    # Search primary bucket
    elem_id = search_bucket(position, bucket_id, hash_arrays, node_positions, element_nodes)
    if elem_id >= 0:
        return elem_id

    # Search neighbor buckets (6-face)
    for neighbor_bucket_id in hash_arrays.bucket_neighbors_6[bucket_id]:
        if neighbor_bucket_id >= 0:
            elem_id = search_bucket(position, neighbor_bucket_id, hash_arrays, node_positions, element_nodes)
            if elem_id >= 0:
                return elem_id

    return -1

def search_bucket(
    position: jax.Array,
    bucket_id: int,
    hash_arrays: HashBucketArrays,
    node_positions: jax.Array,
    element_nodes: jax.Array
) -> int:
    """Search elements within a single bucket."""
    bucket_elems = hash_arrays.bucket_elements[bucket_id]
    count = hash_arrays.bucket_elem_counts[bucket_id]

    # Vectorized search over bucket elements
    mask = jnp.arange(len(bucket_elems)) < count
    found_mask = jax.vmap(test_element, in_axes=(None, 0, None))(
        position, bucket_elems, mask
    )

    # Return first found or -1
    found_indices = jnp.where(found_mask, jnp.arange(len(bucket_elems)), -1)
    first_found = jnp.max(found_indices)

    return jax.lax.cond(
        first_found >= 0,
        lambda: bucket_elems[first_found],
        lambda: -1
    )
```

**Memory Impact** (Heavy block with 200K elements):
```
Without hash:
  - Direct padded search: 200K element tests per particle
  - Memory: (n_particles, 200K) intermediate = 1.6 GB for 1M particles

With hash (1000 buckets × 200 elem/bucket):
  - Bucket search: 200 element tests per particle
  - Memory: (n_particles, 200) intermediate = 1.6 MB for 1M particles
  - Improvement: 1000× less memory
```

### Detection & Escalation Logic

**File**: `jaxtrace/gpu/search/block_classifier.py` (NEW, ~200 lines)

```python
@dataclass
class BlockClassification:
    """Classification of blocks by search strategy."""
    light_blocks: List[int]  # <10K elements - direct padded search
    heavy_blocks: List[int]  # >10K elements - hash bucket search
    threshold: int = 10000

    def print_summary(self):
        print(f"Block Classification:")
        print(f"  Light blocks: {len(self.light_blocks)} (direct padded search)")
        print(f"  Heavy blocks: {len(self.heavy_blocks)} (hash bucket search)")
        print(f"  Threshold: {self.threshold:,} elements")

def classify_blocks(
    padded_arrays: PaddedBlockArrays,
    threshold: int = 10000
) -> BlockClassification:
    """
    Classify blocks as light or heavy based on element count.

    Heavy blocks will use hash bucket subdivision.
    """
    light_blocks = []
    heavy_blocks = []

    for block_id in range(padded_arrays.n_blocks):
        count = padded_arrays.block_elem_counts[block_id]
        if count > threshold:
            heavy_blocks.append(block_id)
        else:
            light_blocks.append(block_id)

    return BlockClassification(
        light_blocks=light_blocks,
        heavy_blocks=heavy_blocks,
        threshold=threshold
    )

def build_hash_buckets_for_heavy_blocks(
    padded_arrays: PaddedBlockArrays,
    heavy_block_ids: List[int],
    element_centroids: np.ndarray,
    blocks: List[Block]
) -> Dict[int, HashBucketArrays]:
    """
    Build hash bucket subdivision for all heavy blocks.

    Returns:
        Dict mapping block_id → HashBucketArrays
    """
    hash_buckets = {}

    for block_id in heavy_block_ids:
        # Get elements in this block
        count = padded_arrays.block_elem_counts[block_id]
        element_ids = padded_arrays.block_elements[block_id, :count]

        # Get centroids
        centroids = element_centroids[element_ids]

        # Build hash bucket arrays
        block_bounds = blocks[block_id].bounds
        hash_arrays = build_hash_bucket_arrays(
            element_ids, centroids, block_bounds
        )
        hash_buckets[block_id] = hash_arrays

        print(f"  Block {block_id}: {count:,} elements → {hash_arrays.n_buckets} buckets "
              f"(~{count/hash_arrays.n_buckets:.0f} elem/bucket)")

    return hash_buckets
```

### Updated Multi-Level Search

**File**: `jaxtrace/gpu/search/multi_level.py` (UPDATED)

```python
@jax.jit
def find_element_multi_level_enhanced(
    position: jax.Array,
    cached_element_id: int,
    block_id: int,
    is_heavy_block: bool,  # NEW: Block classification flag
    # Static data
    node_positions: jax.Array,
    element_nodes: jax.Array,
    element_neighbors: jax.Array,
    padded_arrays: PaddedBlockArrays,
    hash_buckets: Dict[int, HashBucketArrays]  # NEW: Hash buckets for heavy blocks
) -> Tuple[int, int]:
    """
    Enhanced multi-level search with hash bucket support.

    Returns:
        (element_id, level_found)

    Levels:
        0: Cached element
        1: Neighbor elements
        2a: Light block (direct padded search)
        2b: Heavy block (hash bucket search)
        3: Neighbor blocks
    """
    # L0: Cached
    elem = search_level0_cached(position, cached_element_id, node_positions, element_nodes)
    if elem >= 0:
        return elem, 0

    # L1: Neighbors
    elem = search_level1_neighbors(position, cached_element_id, element_neighbors,
                                   node_positions, element_nodes)
    if elem >= 0:
        return elem, 1

    # L2: Block elements (light or heavy)
    elem = jax.lax.cond(
        is_heavy_block,
        lambda: search_level2b_hash_bucket(position, block_id, hash_buckets[block_id],
                                           node_positions, element_nodes),
        lambda: search_level2a_direct(position, block_id, padded_arrays,
                                      node_positions, element_nodes)
    )
    if elem >= 0:
        return elem, 2

    # L3: Neighbor blocks
    elem = search_level3_neighbor_blocks(position, block_id, padded_arrays.block_neighbors_26,
                                        padded_arrays, hash_buckets, node_positions, element_nodes)
    if elem >= 0:
        return elem, 3

    return -1, -1
```

---

## MONITORING & PROFILING (ALL PHASES)

### Required Metrics (Add to each phase completion doc)

**Phase 1 Additions**:
```python
def print_block_occupancy_report(
    block_to_elements: Dict[int, np.ndarray]
):
    """
    Print detailed block occupancy analysis.

    Metrics:
        - Element distribution histogram
        - Imbalance factor
        - Heavy block identification (>10K elements)
        - Padding waste estimate
    """
    counts = np.array([len(elems) for elems in block_to_elements.values()])

    print("=" * 80)
    print("BLOCK OCCUPANCY ANALYSIS")
    print("=" * 80)
    print(f"Total blocks: {len(counts)}")
    print(f"Total elements: {counts.sum():,}")
    print(f"\nElement Distribution:")
    print(f"  Min:  {counts.min():>10,}")
    print(f"  Max:  {counts.max():>10,}")
    print(f"  Mean: {counts.mean():>10,.1f}")
    print(f"  Std:  {counts.std():>10,.1f}")
    print(f"  Median: {np.median(counts):>10,.0f}")
    print(f"  95th %ile: {np.percentile(counts, 95):>10,.0f}")
    print(f"\nImbalance: {counts.max() / counts.mean():.2f}×")

    # Heavy block detection
    heavy_threshold = 10000
    heavy_blocks = np.sum(counts > heavy_threshold)
    print(f"\nHeavy Blocks (>{heavy_threshold:,} elements): {heavy_blocks}")
    if heavy_blocks > 0:
        heavy_ids = np.where(counts > heavy_threshold)[0]
        print(f"  Block IDs: {heavy_ids.tolist()}")
        print(f"  Elements: {counts[heavy_ids].tolist()}")
        print(f"  ⚠️  Will use hash bucket search for these blocks")

    # Padding waste
    max_elem = int(np.percentile(counts, 95) * 1.5)
    total_capacity = len(counts) * max_elem
    total_used = counts.sum()
    padding_waste = 1.0 - (total_used / total_capacity)
    print(f"\nPadding Analysis:")
    print(f"  Max elements per block: {max_elem:,}")
    print(f"  Total capacity: {total_capacity:,}")
    print(f"  Total used: {total_used:,}")
    print(f"  Padding waste: {padding_waste*100:.1f}%")

    if padding_waste > 0.5:
        print(f"  ⚠️  High padding waste (>50%) - consider more blocks")

    print("=" * 80)
```

**Phase 2 Additions**:
- Memory footprint per block
- Validation that no block exceeds max_elem_per_block

**Phase 4 Additions**:
- Search time per level (L0, L1, L2a, L2b, L3)
- Hit rate per level
- Heavy block performance comparison (with/without hash)

---

## UPDATED SUCCESS CRITERIA

### Phase 1 Success Criteria (ENHANCED)
- ✅ All 3.5M elements assigned
- ✅ Load imbalance: <10×
- ✅ **Heavy blocks identified and documented** (NEW)
- ✅ **Padding waste: <50%** (NEW)
- ✅ 26-neighbor topology correct
- ✅ Memory: <50 MB for structure
- ✅ All tests passing (22+)

### Phase 4 Success Criteria (ENHANCED)
- ✅ L0 hit rate: 85-95%
- ✅ L1 hit rate: 3-10%
- ✅ L2 hit rate: 1-5% (light blocks) or <1% (heavy blocks)
- ✅ L3 hit rate: 0.1-1%
- ✅ Total found: >98%
- ✅ **Heavy block search: <10 μs/particle** (NEW)
- ✅ **Light block search: <5 μs/particle** (NEW)
- ✅ Memory: <500 MB for 1M particles
- ✅ Accuracy: 100% match CPU
- ✅ Performance: >10× speedup vs CPU
- ✅ All tests passing (25+, including hash bucket tests)

---

## IMPLEMENTATION PRIORITIES

### Must-Have (Phase 4)
1. ✅ Hash bucket search for heavy blocks
2. ✅ Block classification (light vs heavy)
3. ✅ Automatic escalation logic
4. ✅ Monitoring & profiling

### Should-Have (Phase 5)
1. Dynamic repartitioning (if padding waste >70%)
2. Adaptive bucket size tuning
3. Performance comparison reports

### Could-Have (Phase 6+)
1. Multi-level Morton code refinement
2. GPU-native histogram generation
3. Online rebalancing

---

## REVISED FILE STRUCTURE

```
jaxtrace/gpu/
├── forest/
│   ├── block_grid.py           (Phase 1)
│   ├── element_mapper.py       (Phase 1)
│   ├── load_balance.py         (Phase 1) - ENHANCED with heavy block detection
│   └── visualize_forest.py     (Phase 1)
├── mesh/
│   └── element_neighbors.py    (Phase 2)
├── arrays/
│   ├── padded_blocks.py        (Phase 2)
│   └── validation.py           (Phase 2)
├── particles/
│   ├── seeding.py              (Phase 3)
│   └── state.py                (Phase 3)
├── search/
│   ├── cpu_search.py           (Phase 3)
│   ├── level0_cached.py        (Phase 4)
│   ├── level1_neighbors.py     (Phase 4)
│   ├── level2a_direct.py       (Phase 4) - Light blocks
│   ├── level2b_hash_bucket.py  (Phase 4) - Heavy blocks (NEW)
│   ├── level3_neighbor_blocks.py (Phase 4)
│   ├── block_classifier.py     (Phase 4) - Light/heavy detection (NEW)
│   ├── hash_bucket.py          (Phase 4) - Hash arrays (NEW)
│   └── multi_level.py          (Phase 4) - Enhanced with hash support
└── geometry/
    ├── point_in_tet.py         (Phase 4)
    └── morton_code.py          (Phase 4) - NEW

tests/gpu/
├── test_block_grid.py          (Phase 1)
├── test_element_mapper.py      (Phase 1)
├── test_element_neighbors.py   (Phase 2)
├── test_padded_blocks.py       (Phase 2)
├── test_seeding.py             (Phase 3)
├── test_cpu_search.py          (Phase 3)
├── test_particle_state.py      (Phase 3)
├── test_point_in_tet.py        (Phase 4)
├── test_morton_code.py         (Phase 4) - NEW
├── test_hash_bucket.py         (Phase 4) - NEW
├── test_block_classifier.py    (Phase 4) - NEW
└── test_multi_level_search.py  (Phase 4) - Enhanced
```

---

## FINAL APPROVAL CHECKLIST

Before proceeding with Phase 0 implementation:

- ✅ Core design approved (padded arrays, multi-level search)
- ✅ Hash/bucket search integrated into Phase 4 (not Phase 9)
- ✅ Heavy block detection threshold: 10,000 elements
- ✅ Monitoring & profiling added to all phases
- ✅ Padding waste target: <50%
- ✅ Success criteria updated with performance targets
- ✅ Test coverage includes hash bucket functionality
- ✅ Documentation will include block occupancy analysis

---

## NEXT STEPS

1. **Await your approval** of this final plan
2. **Start Phase 0**: Mesh analysis with heavy block detection
3. **Phase-by-phase execution** with enhanced monitoring
4. **Continuous validation** against CPU ground truth

**Timeline**: 10-16 days (added 1-2 days for hash bucket implementation)

**Deliverables**: 90+ tests (added hash bucket tests)

**Result**: Scalable, future-proof GPU particle tracking that handles extreme AMR cases.

---

**END OF FINAL EXECUTABLE PLAN**

This plan integrates all critical review feedback and is ready for execution.
