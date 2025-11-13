# Phase 4: GPU Multi-Level Search - Task Breakdown

**Status**: 🚧 IN PROGRESS
**Priority**: ⚠️ CRITICAL - Required for acceptable performance on heavy blocks
**Estimated Duration**: 2-3 days

---

## Overview

Phase 4 implements the complete GPU multi-level search pipeline with hash bucket subdivision for heavy blocks. This is the **critical phase** that enables efficient particle tracking on ThreadedA's heavy blocks containing 91% of elements.

### Why Phase 4 is Critical

**Heavy Block Performance Challenge**:
- 4 heavy blocks (21, 22, 25, 26) contain 828K-949K elements each
- Without hash buckets: O(900K) search = 450,000 μs/particle = **UNACCEPTABLE**
- With hash buckets: O(200) search = 100 μs/particle = **4,500× speedup**

**Key Innovation**: Hash bucket subdivision using Morton codes to partition heavy blocks into spatial buckets of ~200 elements each.

---

## Task Breakdown

### Task 4.1: Block Classification
**Priority**: HIGH (foundation for all other tasks)
**Estimated Time**: 3-4 hours
**File**: `jaxtrace/gpu/search/block_classifier.py` (~200 lines)

**Deliverables**:
```python
@dataclass
class BlockClassification:
    light_blocks: List[int]  # <10K elements
    heavy_blocks: List[int]  # >10K elements
    threshold: int = 10000

def classify_blocks(
    padded_arrays: PaddedArrays,
    threshold: int = 10000
) -> BlockClassification:
    """Classify blocks as light or heavy."""
```

**Tests**: `tests/gpu/test_block_classifier.py`
- Test light/heavy block detection
- Test threshold behavior
- Test with ThreadedA (4 heavy blocks expected)

**Success Criteria**:
- Correctly identify 4 heavy blocks in ThreadedA
- 28 light blocks detected
- Memory overhead < 1 MB

---

### Task 4.2: Morton Code & Hash Bucket Module
**Priority**: HIGH (required for L2b)
**Estimated Time**: 4-5 hours
**File**: `jaxtrace/gpu/search/hash_bucket.py` (~400 lines)

**Deliverables**:
```python
@dataclass
class HashBucketArrays:
    block_id: int
    n_buckets: int
    bucket_elements: np.ndarray  # (n_buckets, max_elem_per_bucket) int32
    bucket_elem_counts: np.ndarray  # (n_buckets,) int32
    bucket_neighbors_6: np.ndarray  # (n_buckets, 6) int32
    morton_bits: int = 10

def compute_morton_codes(
    positions: np.ndarray,
    block_bounds: np.ndarray,
    bits: int = 10
) -> np.ndarray:
    """Compute Morton Z-order codes for spatial hashing."""

def build_hash_bucket_arrays(
    element_ids: np.ndarray,
    element_centroids: np.ndarray,
    block_bounds: np.ndarray,
    target_bucket_size: int = 200
) -> HashBucketArrays:
    """Build hash bucket subdivision for heavy block."""

def compute_bucket_neighbors(n_buckets: int) -> np.ndarray:
    """Compute 6-face neighbors in Morton space."""
```

**Tests**: `tests/gpu/test_hash_bucket.py`
- Test Morton code computation
- Test bucket assignment uniformity
- Test bucket neighbor topology
- Test with heavy block (949K elements → ~4,746 buckets)

**Success Criteria**:
- ~200 elements/bucket average
- Bucket neighbors correctly computed
- Memory < 50 MB per heavy block
- Build time < 5 seconds per heavy block

---

### Task 4.3: Level 0 - Cached Element Search
**Priority**: MEDIUM
**Estimated Time**: 2-3 hours
**File**: `jaxtrace/gpu/search/level0_cached.py` (~150 lines)

**Deliverables**:
```python
@jax.jit
def search_level0_cached(
    position: jax.Array,  # (3,)
    cached_element_id: int,
    node_positions: jax.Array,
    element_nodes: jax.Array
) -> int:
    """
    L0: Check if particle still in cached element.

    Returns:
        cached_element_id if still inside, else -1
    """
```

**Tests**: `tests/gpu/test_level0.py`
- Test cache hit (particle in same element)
- Test cache miss (particle moved)
- Test performance (< 1 μs)

**Success Criteria**:
- Expected hit rate: 85-95% for small time steps
- Performance: < 1 μs per particle

---

### Task 4.4: Level 1 - Neighbor Element Search
**Priority**: MEDIUM
**Estimated Time**: 2-3 hours
**File**: `jaxtrace/gpu/search/level1_neighbors.py` (~200 lines)

**Deliverables**:
```python
@jax.jit
def search_level1_neighbors(
    position: jax.Array,
    cached_element_id: int,
    element_neighbors: jax.Array,  # (max_neighbors,) from Phase 2
    node_positions: jax.Array,
    element_nodes: jax.Array
) -> int:
    """
    L1: Check 3-4 neighbor elements.

    Returns:
        element_id if found in neighbors, else -1
    """
```

**Tests**: `tests/gpu/test_level1.py`
- Test neighbor hit (particle in adjacent element)
- Test with Phase 2 neighbor arrays
- Test performance (< 5 μs)

**Success Criteria**:
- Expected hit rate: 3-10% (particles crossing element boundaries)
- Performance: < 5 μs per particle
- Uses Phase 2 neighbor arrays

---

### Task 4.5: Level 2a - Light Block Direct Search
**Priority**: HIGH
**Estimated Time**: 3-4 hours
**File**: `jaxtrace/gpu/search/level2a_light.py` (~250 lines)

**Deliverables**:
```python
@jax.jit
def search_level2a_light_block(
    position: jax.Array,
    block_id: int,
    padded_arrays: PaddedArrays,
    node_positions: jax.Array,
    element_nodes: jax.Array
) -> int:
    """
    L2a: Direct search in light block (<10K elements).

    Uses Phase 2 padded arrays for vectorized search.

    Returns:
        element_id if found, else -1
    """
```

**Tests**: `tests/gpu/test_level2a.py`
- Test with light blocks (< 10K elements)
- Test vectorized performance
- Test with ThreadedA light blocks

**Success Criteria**:
- Performance: < 10 μs per particle for blocks with 1K-10K elements
- Memory: Uses Phase 2 padded arrays (no additional memory)
- Hit rate: 1-5%

---

### Task 4.6: Level 2b - Heavy Block Hash Bucket Search
**Priority**: ⚠️ CRITICAL
**Estimated Time**: 5-6 hours
**File**: `jaxtrace/gpu/search/level2b_heavy.py` (~350 lines)

**Deliverables**:
```python
@jax.jit
def search_level2b_hash_bucket(
    position: jax.Array,
    block_id: int,
    hash_arrays: HashBucketArrays,
    node_positions: jax.Array,
    element_nodes: jax.Array
) -> int:
    """
    L2b: Hash bucket search in heavy block (>10K elements).

    Algorithm:
        1. Compute Morton code for particle position
        2. Map to bucket_id
        3. Search elements in that bucket (~200 elements)
        4. If not found, search 6 neighbor buckets

    Returns:
        element_id if found, else -1
    """

@jax.jit
def search_bucket(
    position: jax.Array,
    bucket_id: int,
    hash_arrays: HashBucketArrays,
    node_positions: jax.Array,
    element_nodes: jax.Array
) -> int:
    """Search elements within a single bucket."""
```

**Tests**: `tests/gpu/test_level2b.py`
- Test hash bucket lookup
- Test neighbor bucket fallback
- Test with ThreadedA heavy blocks (828K-949K elements)
- Performance benchmarking

**Success Criteria**:
- Performance: < 100 μs per particle for 900K element blocks
- Speedup: 4,000-5,000× vs direct search
- Memory: < 50 MB per heavy block
- Hit rate: 1-5%

---

### Task 4.7: Level 3 - Neighbor Block Search
**Priority**: MEDIUM
**Estimated Time**: 3-4 hours
**File**: `jaxtrace/gpu/search/level3_neighbor_blocks.py` (~300 lines)

**Deliverables**:
```python
@jax.jit
def search_level3_neighbor_blocks(
    position: jax.Array,
    primary_block_id: int,
    block_classification: BlockClassification,
    padded_arrays: PaddedArrays,
    hash_arrays_dict: Dict[int, HashBucketArrays],
    node_positions: jax.Array,
    element_nodes: jax.Array
) -> int:
    """
    L3: Search 26 neighbor blocks.

    Automatically dispatches to L2a (light) or L2b (heavy)
    based on block classification.

    Returns:
        element_id if found, else -1
    """
```

**Tests**: `tests/gpu/test_level3.py`
- Test 26-neighbor topology
- Test mixed light/heavy neighbors
- Test boundary particles

**Success Criteria**:
- Performance: < 1000 μs per particle (worst case, 26 blocks)
- Hit rate: 0.1-1% (boundary cases)

---

### Task 4.8: Multi-Level Search Orchestrator
**Priority**: ⚠️ CRITICAL (integrates all levels)
**Estimated Time**: 4-5 hours
**File**: `jaxtrace/gpu/search/multi_level_search.py` (~400 lines)

**Deliverables**:
```python
@jax.jit
def multi_level_search(
    position: jax.Array,
    cached_element_id: int,
    cached_block_id: int,
    block_classification: BlockClassification,
    padded_arrays: PaddedArrays,
    hash_arrays_dict: Dict[int, HashBucketArrays],
    element_neighbors: jax.Array,
    node_positions: jax.Array,
    element_nodes: jax.Array
) -> Tuple[int, int, int]:
    """
    Complete multi-level search pipeline.

    Search order:
        L0: Cached element
        L1: Neighbor elements
        L2: Current block (L2a light or L2b heavy)
        L3: Neighbor blocks

    Returns:
        (element_id, block_id, level_found)
    """

@dataclass
class SearchStats:
    """Per-level search statistics."""
    n_particles: int
    l0_hits: int
    l1_hits: int
    l2_hits: int
    l3_hits: int
    not_found: int
    l0_time: float
    l1_time: float
    l2_time: float
    l3_time: float
```

**Tests**: `tests/gpu/test_multi_level_search.py`
- Test complete L0→L1→L2→L3 pipeline
- Test early termination at each level
- Test with ThreadedA mesh
- Performance benchmarking

**Success Criteria**:
- L0 hit rate: 85-95%
- L1 hit rate: 3-10%
- L2 hit rate: 1-5%
- L3 hit rate: 0.1-1%
- Not found: < 0.01%
- Total throughput: > 10,000 particles/second

---

### Task 4.9: Monitoring & Profiling
**Priority**: MEDIUM
**Estimated Time**: 2-3 hours
**File**: `jaxtrace/gpu/search/monitoring.py` (~200 lines)

**Deliverables**:
```python
@dataclass
class PerformanceMetrics:
    """Detailed performance metrics."""
    per_level_times: Dict[str, float]
    per_level_hit_rates: Dict[str, float]
    heavy_block_stats: Dict[int, Dict]
    memory_usage: Dict[str, float]

def print_performance_report(metrics: PerformanceMetrics):
    """Print human-readable performance report."""

def log_search_statistics(
    search_stats: SearchStats,
    output_file: str
):
    """Log statistics to file for analysis."""
```

**Tests**: `tests/gpu/test_monitoring.py`
- Test metrics collection
- Test report generation
- Test with ThreadedA

**Success Criteria**:
- Real-time statistics collection
- < 1% performance overhead
- Clear reporting of bottlenecks

---

### Task 4.10: Integration Test & Validation
**Priority**: ⚠️ CRITICAL
**Estimated Time**: 3-4 hours
**File**: `test_phase4_multi_level_search.py`

**Test Coverage**:
1. **Small mesh test** (synthetic, 1K elements)
2. **ThreadedA test** (real mesh, 3.5M elements)
3. **Performance benchmarking** (1M particles)
4. **Accuracy validation** (vs Phase 3 CPU baseline)
5. **Memory profiling** (< 500 MB target)

**Success Criteria**:
- 100% accuracy vs CPU baseline (Phase 3)
- >10,000 particles/second throughput
- < 500 MB total memory
- L0-L3 hit rates match expectations
- Heavy block search < 100 μs/particle

---

## File Structure

```
jaxtrace/gpu/search/
├── __init__.py
├── block_classifier.py        # Task 4.1
├── hash_bucket.py              # Task 4.2
├── level0_cached.py            # Task 4.3
├── level1_neighbors.py         # Task 4.4
├── level2a_light.py            # Task 4.5
├── level2b_heavy.py            # Task 4.6
├── level3_neighbor_blocks.py   # Task 4.7
├── multi_level_search.py       # Task 4.8
└── monitoring.py               # Task 4.9

tests/gpu/
├── test_block_classifier.py
├── test_hash_bucket.py
├── test_level0.py
├── test_level1.py
├── test_level2a.py
├── test_level2b.py
├── test_level3.py
├── test_multi_level_search.py
└── test_monitoring.py

docs/phase4/
├── PHASE4_TASK_BREAKDOWN.md    # This file
└── PHASE4_COMPLETE.md          # To be created upon completion
```

---

## Dependencies

### From Phase 1:
- Block grid and element-to-block mapping
- Block neighbor topology

### From Phase 2:
- Padded 2D arrays: (32, 949,632)
- Element neighbor arrays
- Memory budget tracking

### From Phase 3:
- ParticleState data structure
- CPU baseline search for validation

---

## Estimated Timeline

| Task | Duration | Dependencies |
|------|----------|--------------|
| 4.1: Block Classifier | 3-4 hrs | Phase 1, 2 |
| 4.2: Hash Bucket | 4-5 hrs | Phase 1, 2 |
| 4.3: L0 Cached | 2-3 hrs | None |
| 4.4: L1 Neighbors | 2-3 hrs | Phase 2 |
| 4.5: L2a Light | 3-4 hrs | Phase 2, 4.1 |
| 4.6: L2b Heavy | 5-6 hrs | 4.1, 4.2 |
| 4.7: L3 Neighbor | 3-4 hrs | 4.1, 4.5, 4.6 |
| 4.8: Orchestrator | 4-5 hrs | 4.3-4.7 |
| 4.9: Monitoring | 2-3 hrs | 4.8 |
| 4.10: Integration | 3-4 hrs | All above |
| **TOTAL** | **31-43 hours** | **~2-3 days** |

---

## Success Criteria Summary

| Metric | Target | Critical |
|--------|--------|----------|
| Heavy block search | < 100 μs/particle | ✅ YES |
| Light block search | < 10 μs/particle | ✅ YES |
| L0 hit rate | 85-95% | ✅ YES |
| Total throughput | > 10,000 particles/s | ✅ YES |
| Memory | < 500 MB | ✅ YES |
| Accuracy | 100% vs CPU baseline | ✅ YES |
| Heavy blocks detected | 4 (ThreadedA) | ✅ YES |
| Speedup vs CPU | > 1,000× | ✅ YES |

---

## Next Steps After Phase 4

Once Phase 4 is complete, we can:
1. **Replace Phase 3 CPU search** with GPU L2 search for initial element assignment (1,000× speedup)
2. **Proceed to Phase 5**: Dynamic repartitioning for evolving simulations
3. **Optimize further**: Adaptive bucket sizes, GPU memory pooling

---

**Status**: Ready to begin implementation
**Start Date**: 2025-11-07
**Target Completion**: 2025-11-09
